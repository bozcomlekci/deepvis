"""Jaxpr engine shared by every JAX flavor.

1. **Trace** ``f(param_arrays, input_leaves)`` to a jaxpr. Framework adapters mark module
   boundaries with ``jax.named_scope("dv:<path>")``; pure functions may use their own scopes.
2. **Group** equations into graph nodes:
   * atomic modules (Dense, Conv, attention ...) swallow all of their equations;
   * container-level primitives are clustered into readable ops (``Softmax``, ``ReLU``,
     residual ``add`` ...) — single-entry runs of primitives fold together;
   * pure functions without scopes get *parameter groups* (``x @ W + b`` → one Linear node).
3. **Evaluate** the jaxpr equation by equation, recording node outputs, applying
   interventions and (optionally) adding zero "delta" perturbations whose gradients are
   the activation gradients. Everything is jit- and grad-compatible.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend import core as jcore

from ...core.graph import ModelGraph, Node, ParamInfo
from ...core.kinds import Role, canonical_kind, function_kind, infer_roles

MARK = "dv:"
TRANSPARENT = {"convert_element_type", "copy", "copy_p", "stop_gradient", "broadcast_in_dim", "squeeze",
               "expand_dims", "pvary", "sharding_constraint", "mesh_cast", "reshard", "optimization_barrier"}
PARAM_TRANSPARENT = TRANSPARENT | {"reshape", "transpose"}
CALL_PRIMS = {"jit", "pjit", "custom_jvp_call", "custom_vjp_call", "custom_vjp_call_jaxpr", "closed_call",
              "core_call", "remat", "checkpoint"}


@dataclass
class ModuleMeta:
    path: str
    type_name: str
    attrs: dict = field(default_factory=dict)
    atomic: bool = False  # treat as a leaf even if it has sub-modules (attention ...)

    @property
    def kind(self) -> str:
        return canonical_kind(self.type_name, self.attrs)


@dataclass
class Group:
    id: str
    kind: str
    type_name: str
    op: str  # module | function | param
    parent: str
    eqns: list[int] = field(default_factory=list)
    params: dict[str, ParamInfo] = field(default_factory=dict)
    prims: list[str] = field(default_factory=list)
    out_vars: list = field(default_factory=list)
    in_vars: list = field(default_factory=list)
    closed: bool = False
    inferred: bool = False  # pure-mode parameter group
    attrs: dict = field(default_factory=dict)
    vars: set = field(default_factory=set)
    entries: set = field(default_factory=set)
    module: str | None = None

    @property
    def primary(self):
        return self.out_vars[-1] if self.out_vars else None


@dataclass
class Program:
    closed: Any  # ClosedJaxpr
    n_params: int
    in_tree: Any
    out_tree: Any
    groups: dict[str, Group]
    graph: ModelGraph
    hooks_at: dict[int, list[str]]  # eqn index -> node ids whose primary output it produces
    input_ids: list[str]
    node_primary: dict[str, Any]
    node_input: dict[str, Any]


def _is_literal(v) -> bool:
    return isinstance(v, jcore.Literal)


def _is_drop(v) -> bool:
    return type(v).__name__ == "DropVar"


def _call_name(eqn) -> str | None:
    p = eqn.params
    if "name" in p and isinstance(p["name"], str):
        return p["name"]
    for key in ("call_jaxpr", "fun_jaxpr", "jaxpr"):
        sub = p.get(key)
        if sub is None:
            continue
        inner = getattr(sub, "jaxpr", sub)
        eqns = getattr(inner, "eqns", [])
        for e in eqns:
            n = _call_name(e) if e.primitive.name in CALL_PRIMS else None
            if n:
                return n
    return None


def _scope_paths(eqn, marker_mode: bool) -> list[str]:
    stack = str(eqn.source_info.name_stack)
    comps = [c for c in stack.split("/") if c] if stack else []
    if marker_mode:
        return [c[len(MARK):] for c in comps if c.startswith(MARK)]
    paths, cur = [], []
    for c in comps:
        if "(" in c:  # transformation frames such as jvp(...)
            continue
        cur.append(c)
        paths.append(".".join(cur))
    return paths


STRUCTURAL_PRIMS = {"reshape", "transpose", "concatenate", "gather", "slice", "dynamic_slice", "pad"}


def _prim_kind(prims: list[str], eqns) -> str:
    s = set(prims)
    if any(p.startswith("reduce_window_max") for p in s):
        return "MaxPool2d"
    if any(p.startswith("reduce_window") for p in s):
        return "AvgPool2d"
    if prims and prims[0] in STRUCTURAL_PRIMS:
        return {"reshape": "Reshape", "transpose": "Transpose", "concatenate": "Concat"}.get(prims[0], "getitem")
    if "scan" in s or "while" in s:
        return "Scan"
    if "reduce_sum" in s and s <= {"reduce_sum", "div", "mul", "convert_element_type", "broadcast_in_dim", "squeeze"}:
        return "mean"
    if "dot_general" in s:
        return "matmul"
    if any(p.startswith("conv_general") for p in s):
        return "Conv2d"
    if {"exp", "reduce_sum", "div"} <= s:
        return "Softmax"
    if "logistic" in s:
        return "SiLU" if "mul" in s else "Sigmoid"
    if "erf" in s:
        return "GELU"
    if "tanh" in s:
        return "GELU" if ("integer_pow" in s or "pow" in s) else "Tanh"
    if "rsqrt" in s and ("reduce_sum" in s or "reduce_mean" in s):
        return "LayerNorm"
    if prims == ["max"] and any(_is_literal(v) and np.all(np.asarray(v.val) == 0) for e in eqns for v in e.invars):
        return "ReLU"
    if len(s) == 1:
        return function_kind(prims[0])
    return function_kind(prims[-1])


def build_program(fn: Callable, param_arrays: list, args: tuple, kwargs: dict, *, name: str, framework: str,
                  param_paths: list[str], param_trainable: list[bool], modules: dict[str, ModuleMeta] | None,
                  input_names: list[str] | None = None) -> Program:
    """Trace ``fn(param_arrays, *args, **kwargs)`` and build the grouped graph."""
    in_leaves, in_tree = jax.tree_util.tree_flatten((args, kwargs))
    arr_pos = [i for i, v in enumerate(in_leaves) if hasattr(v, "shape")]
    static = {i: v for i, v in enumerate(in_leaves) if i not in set(arr_pos)}

    def flat_fn(params, arrs):
        leaves = list(in_leaves)
        for i, a in zip(arr_pos, arrs):
            leaves[i] = a
        for i, v in static.items():
            leaves[i] = v
        a, k = jax.tree_util.tree_unflatten(in_tree, leaves)
        return fn(params, *a, **k)

    arrs = [jnp.asarray(in_leaves[i]) for i in arr_pos]
    closed, out_shape = jax.make_jaxpr(flat_fn, return_shape=True)(list(param_arrays), arrs)
    out_tree = jax.tree_util.tree_structure(out_shape)
    jaxpr = closed.jaxpr
    n_p = len(param_arrays)
    param_vars = jaxpr.invars[:n_p]
    input_vars = jaxpr.invars[n_p:]

    marker_mode = modules is not None
    infos = []
    for path, arr, tr in zip(param_paths, param_arrays, param_trainable):
        mod, _, local = path.rpartition(".")
        infos.append(ParamInfo(path, local, mod, tuple(arr.shape), str(arr.dtype), tr))
    param_of = {v: infos[i] for i, v in enumerate(param_vars)}  # var -> ParamInfo (param-derived vars)

    # ---------------------------------------------------------- module metadata
    if marker_mode:
        mods = dict(modules)
    else:  # pure mode: named scopes become modules
        mods = {}
        for e in jaxpr.eqns:
            for p in _scope_paths(e, False):
                mods.setdefault(p, ModuleMeta(p, "Scope"))
    children: dict[str, list[str]] = {p: [] for p in mods}
    for p in mods:
        if p:
            parent = p.rpartition(".")[0]
            while parent and parent not in mods:
                parent = parent.rpartition(".")[0]
            children.setdefault(parent, []).append(p)

    def is_atomic(p: str) -> bool:
        m = mods.get(p)
        if m is None:
            return False
        return m.atomic or not children.get(p) or m.kind == "MultiheadAttention"

    def leaf_of(paths: list[str]) -> str | None:
        for p in paths:
            if p in mods and is_atomic(p):
                return p
        return None

    def container_of(paths: list[str]) -> str:
        for p in reversed(paths):
            if p in mods and not is_atomic(p):
                return p
        return ""

    groups: dict[str, Group] = {}
    producer: dict[Any, str] = {}
    alias: dict[Any, Any] = {}
    input_ids = []
    for i, v in enumerate(input_vars):
        nm = (input_names[i] if input_names and i < len(input_names) else ("x" if len(input_vars) == 1 else f"x{i}"))
        nid = f"@in:{nm}"
        input_ids.append(nid)
        producer[v] = nid

    def resolve(v):
        while v in alias:
            v = alias[v]
        return v

    counters: dict[str, int] = {}
    last_gid: str | None = None
    param_nodes: dict[str, str] = {}

    def new_group(gid_base: str, **kw) -> Group:
        k = counters.get(gid_base, 0)
        counters[gid_base] = k + 1
        gid = gid_base if k == 0 else f"{gid_base}@{k + 1}"
        g = Group(gid, **kw)
        if k > 0:
            g.module = gid_base
            g.attrs["shared"] = True
        groups[gid] = g
        return g

    current_instance: dict[str, str] = {}

    for idx, eqn in enumerate(jaxpr.eqns):
        prim = eqn.primitive.name
        paths = _scope_paths(eqn, marker_mode)
        leaf = leaf_of(paths)
        in_vars = [resolve(v) for v in eqn.invars if not _is_literal(v)]
        p_in = [param_of[v] for v in in_vars if v in param_of]
        d_in = [v for v in in_vars if v not in param_of and (v in producer or v in set(input_vars))]

        # param-derived bookkeeping (reshape/broadcast of a weight is still "the weight")
        if leaf is None and p_in and not d_in and prim in PARAM_TRANSPARENT and len(eqn.outvars) == 1:
            param_of[eqn.outvars[0]] = p_in[0]
            continue

        target: Group | None = None
        if leaf is not None:
            gid = current_instance.get(leaf)
            if gid is not None and last_gid == gid:  # still inside the same call
                target = groups[gid]
            else:  # first call, or a re-entry (shared weights) -> new call instance
                m = mods[leaf]
                target = new_group(leaf, kind=m.kind, type_name=m.type_name, op="module",
                                   parent=container_of(paths[: paths.index(leaf)]), attrs=dict(m.attrs))
                current_instance[leaf] = target.id
        elif not d_in and not p_in:
            continue  # constant sub-computation (iota, masks ...): no node
        else:
            parent = container_of(paths)
            if p_in and not marker_mode:
                # pure mode: parameter groups; a bias add joins the matmul that feeds it
                for v in d_in:
                    pg = groups.get(producer.get(v, ""))
                    if pg is not None and pg.inferred and pg.parent == parent and pg.id == last_gid and not any(
                            p.path in {q.path for q in pg.params.values()} for p in p_in):
                        target = pg
                        break
                if target is None:
                    info = p_in[0]
                    base = info.module or info.path
                    target = new_group(base, kind="Function", type_name="Params", op="module", parent=parent)
                    target.inferred = True
            elif p_in and marker_mode:
                # a container reads its own parameter (e.g. learned positional embedding)
                for info in p_in:
                    if info.path not in param_nodes:
                        pid = f"@param:{info.path}"
                        pg = Group(pid, "Parameter", "Parameter", "param", container_of(paths))
                        pg.params[info.name] = info
                        groups[pid] = pg
                        param_nodes[info.path] = pid
                    pv = [v for v in in_vars if v in param_of and param_of[v].path == info.path][0]
                    producer[pv] = param_nodes[info.path]
                    groups[param_nodes[info.path]].out_vars.append(pv)
                d_in = in_vars
            if target is None:
                if prim in TRANSPARENT and len(d_in) == 1 and len(eqn.outvars) == 1:
                    alias[eqn.outvars[0]] = d_in[0]
                    continue
                if prim in STRUCTURAL_PRIMS:  # reshapes/transposes are their own (closed) nodes
                    target = new_group("@" + prim, kind=prim, type_name=prim, op="function", parent=parent)
                    target.closed = True
                elif prim in CALL_PRIMS:
                    nm = _call_name(eqn) or prim
                    kind = function_kind(nm)
                    target = new_group(f"@{kind.lower() if kind == nm else kind}", kind=kind, type_name=nm,
                                       op="function", parent=parent)
                    target.closed = True
                else:
                    cands = {producer.get(v) for v in d_in}
                    for cid in cands:
                        c = groups.get(cid or "")
                        if c is None or c.closed or c.op != "function" or c.parent != parent:
                            continue
                        if all(v in c.vars or v in c.entries for v in d_in) and any(v in c.vars for v in d_in):
                            target = c
                            break
                    if target is None:
                        target = new_group("@" + prim, kind=prim, type_name=prim, op="function", parent=parent)
                        target.entries = set(d_in)
        target.eqns.append(idx)
        target.prims.append(prim)
        for info in p_in:
            target.params.setdefault(info.name if target.op != "module" or not target.inferred else info.path, info)
        for v in eqn.outvars:
            if not _is_drop(v):
                producer[v] = target.id
                target.vars.add(v)
        last_gid = target.id

    # ------------------------------------------------ outputs / inputs / edges
    consumers: dict[Any, set[str]] = {}
    for gid, g in groups.items():
        for i in g.eqns:
            for v in jaxpr.eqns[i].invars:
                if _is_literal(v):
                    continue
                rv = resolve(v)
                src = producer.get(rv)
                if src is not None and src != gid:
                    consumers.setdefault(rv, set()).add(gid)
                    if rv not in g.in_vars:
                        g.in_vars.append(rv)
    out_vars = [resolve(v) for v in jaxpr.outvars if not _is_literal(v)]
    for gid, g in groups.items():
        if g.op == "param":
            continue
        order = []
        for i in g.eqns:
            for v in jaxpr.eqns[i].outvars:
                if not _is_drop(v) and (v in consumers or v in out_vars):
                    order.append(v)
        g.out_vars = order

    # rename op clusters by inferred kind, give readable unique ids
    used: dict[str, int] = {}
    rename: dict[str, str] = {}
    for gid, g in list(groups.items()):
        if g.op == "function" and (not g.closed or g.prims[0] in STRUCTURAL_PRIMS):
            g.kind = _prim_kind(g.prims, [jaxpr.eqns[i] for i in g.eqns])
            g.type_name = "+".join(dict.fromkeys(g.prims))
        if g.op == "function":
            base = "@" + (g.kind if g.kind[0].isupper() else g.kind)
            k = used.get(base, 0)
            used[base] = k + 1
            rename[gid] = base if k == 0 else f"{base}_{k}"
        elif g.inferred:
            g.kind, roles = _infer_param_group(g, jaxpr)
            g.attrs["roles"] = roles
    if rename:
        groups = {rename.get(k, k): v for k, v in groups.items()}
        for k, v in groups.items():
            v.id = k
        producer = {v: rename.get(p, p) for v, p in producer.items()}

    # ------------------------------------------------------------ graph
    g = ModelGraph(name, framework)
    root_meta = mods.get("")
    root_is_leaf = root_meta is not None and is_atomic("")
    if not root_is_leaf:
        g.add_node(Node("", root_meta.type_name if root_meta else name, root_meta.type_name if root_meta else name,
                        "container", parent=None, label=name, attrs=dict(root_meta.attrs) if root_meta else {}))
    for p in sorted(mods, key=lambda s: (s.count("."), s)):
        if p == "" or is_atomic(p):
            continue
        if any(a in mods and is_atomic(a) for a in _ancestors(p)):
            continue
        parent = p.rpartition(".")[0]
        while parent and parent not in mods:
            parent = parent.rpartition(".")[0]
        m = mods[p]
        g.add_node(Node(p, m.type_name, m.type_name, "container", parent=parent, attrs=dict(m.attrs),
                        label=p.split(".")[-1]))
    for nid in input_ids:
        g.add_node(Node(nid, "Input", "Input", "input", parent="" if not root_is_leaf else None,
                        label=nid.split(":", 1)[1]))
    node_primary, node_input = {}, {}
    for gid, grp in groups.items():
        if grp.op != "param" and not grp.out_vars:
            continue
        params = dict(grp.params)
        if grp.op == "module" and marker_mode:
            owner = grp.module or gid
            params = {}
            for info in infos:
                if info.module == owner or info.module.startswith(owner + ".") or (owner == "" and root_is_leaf):
                    params[info.path[len(owner) + 1:] if owner else info.path] = info
        if grp.inferred:
            roles = grp.attrs.pop("roles", {})
            params = {i.name if i.module == (grp.module or gid) else i.path: i for i in grp.params.values()}
            roles = {r: Role(next((k for k, v in params.items() if v.path == rr.name), rr.name), rr.out_axis,
                             rr.in_axis) for r, rr in roles.items()}
        else:
            roles = infer_roles(grp.kind, {k: v.shape for k, v in params.items()}, framework)
        parent = grp.parent if grp.parent in g.nodes else ("" if "" in g.nodes else None)
        pv = grp.primary if grp.op != "param" else (grp.out_vars[0] if grp.out_vars else None)
        out_shapes = [tuple(pv.aval.shape)] if pv is not None else []
        in_shapes = [tuple(grp.in_vars[0].aval.shape)] if grp.in_vars else []
        attrs = dict(grp.attrs)
        if grp.op == "function":
            attrs["prims"] = list(dict.fromkeys(grp.prims))
        if grp.op == "module" and gid:
            label = gid[len(parent) + 1:] if parent and gid.startswith(parent + ".") else gid
        else:
            label = gid.lstrip("@") or name
        g.add_node(Node(gid, grp.kind, grp.type_name, grp.op, parent=parent, params=params, roles=roles,
                        attrs=attrs, in_shapes=in_shapes, out_shapes=out_shapes, module=grp.module, label=label))
        if pv is not None:
            node_primary[gid] = pv
        if grp.in_vars:
            node_input[gid] = grp.in_vars[0]
    g.add_node(Node("@out", "Output", "Output", "output", parent="" if "" in g.nodes else None, label="output",
                    in_shapes=[tuple(v.aval.shape) for v in out_vars[:1]]))
    g.link_children()
    for nid, v in zip(input_ids, input_vars):
        g.nodes[nid].out_shapes = [tuple(v.aval.shape)]
    for gid, grp in groups.items():
        if gid not in g.nodes:
            continue
        for v in grp.in_vars:
            src = producer.get(v)
            if src in g.nodes and src != gid:
                g.add_edge(src, gid, tuple(v.aval.shape))
    for v in out_vars:
        src = producer.get(v)
        if src in g.nodes:
            g.add_edge(src, "@out", tuple(v.aval.shape))

    hooks_at: dict[int, list[str]] = {}
    var_eqn = {}
    for i, e in enumerate(jaxpr.eqns):
        for v in e.outvars:
            var_eqn[v] = i
    for nid, v in node_primary.items():
        if v in var_eqn:
            hooks_at.setdefault(var_eqn[v], []).append(nid)
    return Program(closed, n_p, in_tree, out_tree, groups, g, hooks_at, input_ids, node_primary, node_input)


def _ancestors(p: str) -> list[str]:
    parts = p.split(".")
    return [".".join(parts[:i]) for i in range(len(parts) - 1, -1, -1)]


def _infer_param_group(g: Group, jaxpr) -> tuple[str, dict[str, Role]]:
    prims = set(g.prims)
    roles: dict[str, Role] = {}
    eqns = [jaxpr.eqns[i] for i in g.eqns]
    kind = "Function"
    for e in eqns:
        if e.primitive.name == "dot_general":
            kind = "Linear"
            (lc, rc), (lb, rb) = e.params["dimension_numbers"]
            for side, contract, batch in ((1, rc, rb), (0, lc, lb)):
                v = e.invars[side]
                info = next((p for p in g.params.values() if not _is_literal(v) and tuple(v.aval.shape) == p.shape),
                            None)
                if info is not None:
                    nd = len(info.shape)
                    rest = [a for a in range(nd) if a not in contract and a not in batch]
                    roles["weight"] = Role(info.path, out_axis=rest[0] if rest else None,
                                           in_axis=contract[0] if contract else None)
                    break
        elif e.primitive.name.startswith("conv_general"):
            kind = "Conv2d"
        elif e.primitive.name in ("gather", "dynamic_slice", "take"):
            kind = "Embedding"
        elif e.primitive.name in ("scan", "while"):
            kind = "Scan"
            break
    if kind == "Linear":
        w = roles.get("weight")
        for p in g.params.values():
            if w is None or p.path != w.name:
                if len(p.shape) == 1:
                    roles["bias"] = Role(p.path, out_axis=0)
    elif kind == "Function" and prims <= {"mul", "add"}:
        kind = "Affine"
    return kind, roles


# ------------------------------------------------------------------ evaluation
def eval_eqn(eqn, invals):
    bp = eqn.primitive.get_bind_params(eqn.params)
    if isinstance(bp, tuple) and len(bp) == 2 and isinstance(bp[1], dict):  # jax < 0.10: (subfuns, params)
        subfuns, bind_params = bp
    else:
        subfuns, bind_params = [], bp
    mgr = getattr(getattr(eqn, "ctx", None), "manager", None)
    with mgr if mgr is not None else contextlib.nullcontext():
        ans = eqn.primitive.bind(*subfuns, *invals, **bind_params)
    return ans if eqn.primitive.multiple_results else [ans]


def evaluate(prog: Program, params: list, input_arrays: list, record: frozenset = frozenset(),
             interventions: dict | None = None, deltas: dict | None = None, record_inputs: bool = False,
             ctx_factory: Callable | None = None):
    """Interpret the program. Returns (flat_outputs, recorded, recorded_inputs)."""
    jaxpr = prog.closed.jaxpr
    env: dict = {}

    def read(v):
        return v.val if _is_literal(v) else env[v]

    for v, c in zip(jaxpr.constvars, prog.closed.consts):
        env[v] = c
    for v, a in zip(jaxpr.invars, list(params) + list(input_arrays)):
        env[v] = a
    rec, rec_in = {}, {}
    interventions = interventions or {}
    for nid, v in zip(prog.input_ids, jaxpr.invars[prog.n_params:]):
        if nid in interventions or (deltas and nid in deltas):
            val = env[v]
            for fn in interventions.get(nid, []):
                val = fn(val, ctx_factory(nid))
            if deltas and nid in deltas:
                val = val + deltas[nid]
            env[v] = val
        if nid in record:
            rec[nid] = env[v]
    for i, eqn in enumerate(jaxpr.eqns):
        outs = eval_eqn(eqn, [read(v) for v in eqn.invars])
        for v, o in zip(eqn.outvars, outs):
            if not _is_drop(v):
                env[v] = o
        for nid in prog.hooks_at.get(i, ()):
            v = prog.node_primary[nid]
            val = env[v]
            for fn in interventions.get(nid, []):
                val = fn(val, ctx_factory(nid))
            if deltas and nid in deltas:
                val = val + deltas[nid]
            env[v] = val
            if nid in record:
                rec[nid] = val
                if record_inputs and nid in prog.node_input:
                    rec_in[nid] = env.get(prog.node_input[nid])
    outs = [read(v) for v in jaxpr.outvars]
    if "@out" in record and outs:
        rec["@out"] = outs[0]
    return outs, rec, rec_in
