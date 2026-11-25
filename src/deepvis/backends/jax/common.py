"""Shared machinery for JAX backends (flax.linen, flax.nnx, equinox, pure functions)."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from ...core.graph import ModelGraph, ParamInfo
from ...core.tensor import first_array
from ..base import ArrayOps, Backend, InterventionContext, RunResult
from .interp import ModuleMeta, Program, build_program, evaluate


class JaxOps(ArrayOps):
    framework = "jax"

    def set_index(self, x, idx, value):
        return x.at[idx].set(value)

    def asarray(self, value, like):
        return jnp.asarray(np.asarray(value), dtype=like.dtype)

    def clip(self, x, lo, hi):
        return jnp.clip(x, lo, hi)

    def zeros_like(self, x):
        return jnp.zeros_like(x)

    def to_numpy(self, x):
        return np.asarray(x)

    def feature_index(self, x, axis, neurons):
        nd = len(x.shape)
        ax = axis % nd
        sel = jnp.asarray(np.asarray(neurons)) if isinstance(neurons, (list, tuple, np.ndarray)) else neurons
        return (slice(None),) * ax + (sel,)


def keypath_str(kp) -> str:
    parts = []
    for k in kp:
        name = type(k).__name__
        if name == "GetAttrKey":
            parts.append(str(k.name))
        elif name == "SequenceKey":
            parts.append(str(k.idx))
        elif name == "DictKey":
            parts.append(str(k.key))
        elif name == "FlattenedIndexKey":
            parts.append(str(k.key))
        else:
            parts.append(str(k).strip(".[]'\""))
    return ".".join(parts)


def is_param_leaf(x) -> bool:
    if not hasattr(x, "shape") or not hasattr(x, "dtype"):
        return False
    try:
        return jnp.issubdtype(x.dtype, jnp.floating) or jnp.issubdtype(x.dtype, jnp.complexfloating)
    except TypeError:  # typed PRNG keys etc.
        return False


def _sig(args, kwargs) -> tuple:
    leaves, tree = jax.tree_util.tree_flatten((args, kwargs))
    return (str(tree),) + tuple(
        (tuple(v.shape), str(v.dtype)) if hasattr(v, "shape") else ("static", repr(v)) for v in leaves)


class JaxBackend(Backend):
    """Subclasses fill ``self._leaves`` / ``self._treedef`` / ``self._paths`` and implement ``_call``."""

    framework = "jax"
    marker_mode = True
    channels_last = True

    def __init__(self, name: str, sample_input=None):
        super().__init__(name)
        self.ops = JaxOps()
        self._sample = sample_input
        self._leaves: list = []
        self._treedef = None
        self._param_idx: list[int] = []  # positions of float-array leaves in _leaves
        self._paths: list[str] = []  # param path per entry of _param_idx
        self._trainable: list[bool] = []
        self._path_pos: dict[str, int] = {}
        self._programs: dict[tuple, Program] = {}
        self._jit_cache: dict = {}
        self._program: Program | None = None

    # ------------------------------------------------------------- state
    def _set_state(self, leaves: list, treedef, path_of_leaf: list[str], trainable_of_leaf: list[bool] | None = None):
        self._leaves = list(leaves)
        self._treedef = treedef
        self._param_idx, self._paths, self._trainable = [], [], []
        for i, (leaf, p) in enumerate(zip(leaves, path_of_leaf)):
            if is_param_leaf(leaf):
                self._param_idx.append(i)
                self._paths.append(p)
                self._trainable.append(True if trainable_of_leaf is None else trainable_of_leaf[i])
        seen: dict[str, int] = {}
        for k, p in enumerate(self._paths):  # disambiguate duplicate paths
            if p in seen:
                self._paths[k] = f"{p}#{k}"
            seen[p] = k
        self._path_pos = {p: k for k, p in enumerate(self._paths)}

    def param_arrays(self) -> list:
        return [self._leaves[i] for i in self._param_idx]

    def leaves_with(self, params: list) -> list:
        leaves = list(self._leaves)
        for i, a in zip(self._param_idx, params):
            leaves[i] = a
        return leaves

    def param_paths(self) -> list[str]:
        return list(self._paths)

    def param_info(self, path: str) -> ParamInfo:
        k = self._path_pos[path]
        a = self._leaves[self._param_idx[k]]
        mod, _, local = path.rpartition(".")
        return ParamInfo(path, local, mod, tuple(a.shape), str(a.dtype), self._trainable[k])

    def get_param(self, path: str) -> np.ndarray:
        return np.array(self._leaves[self._param_idx[self._path_pos[path]]], copy=True)

    def get_native_param(self, path: str):
        return self._leaves[self._param_idx[self._path_pos[path]]]

    # --------------------------------------------------------- loading (training loops)
    def _native_tree(self, native):
        """``native`` (as returned by ``export()``) -> a pytree with this backend's treedef."""
        return native

    def _grad_path(self, kp) -> str:
        return keypath_str(kp)

    def load(self, native) -> None:
        leaves, treedef = jax.tree_util.tree_flatten(self._native_tree(native))
        if len(leaves) != len(self._leaves):
            raise ValueError(f"expected a tree with {len(self._leaves)} leaves like export(), got {len(leaves)}")
        for i in self._param_idx:
            if tuple(np.shape(leaves[i])) != tuple(np.shape(self._leaves[i])):
                raise ValueError(f"shape mismatch at leaf {i}: {np.shape(leaves[i])} vs {np.shape(self._leaves[i])}")
        self._leaves = list(leaves)
        self._on_params_changed()

    def grads(self, native=None) -> dict[str, np.ndarray]:
        if native is None:
            return {}
        out = {}
        for kp, g in jax.tree_util.tree_flatten_with_path(native)[0]:
            p = self._grad_path(kp)
            if p in self._path_pos and hasattr(g, "shape"):
                out[p] = np.asarray(g)
        return out

    def set_param(self, path: str, value) -> None:
        k = self._path_pos[path]
        old = self._leaves[self._param_idx[k]]
        v = np.asarray(value)
        if tuple(v.shape) != tuple(old.shape):
            raise ValueError(f"shape mismatch for {path}: {v.shape} vs {tuple(old.shape)}")
        self._leaves[self._param_idx[k]] = jnp.asarray(v, dtype=old.dtype)
        self._on_params_changed()

    def _on_params_changed(self) -> None:  # nnx syncs the live module
        pass

    # --------------------------------------------------- subclass contract
    def _call(self, params: list, args: tuple, kwargs: dict, scoped: bool):  # pragma: no cover
        raise NotImplementedError

    def _module_meta(self) -> dict[str, ModuleMeta] | None:
        return None

    def _input_names(self, args) -> list[str] | None:
        return None

    # ------------------------------------------------------------- graph
    def _sample_args(self):
        if self._sample is None:
            return None
        return self._sample if isinstance(self._sample, tuple) else (self._sample,)

    def program(self, args: tuple, kwargs: dict | None = None) -> Program:
        kwargs = kwargs or {}
        key = _sig(args, kwargs)
        if key not in self._programs:
            self._programs[key] = build_program(
                lambda p, *a, **k: self._call(p, a, k, scoped=True), self.param_arrays(), args, kwargs,
                name=self.name, framework=self.framework, param_paths=self._paths, param_trainable=self._trainable,
                modules=self._module_meta() if self.marker_mode else None, input_names=self._input_names(args))
        return self._programs[key]

    def build_graph(self, *inputs, **kwargs) -> ModelGraph:
        args = inputs if inputs else self._sample_args()
        if args is None:
            raise ValueError(f"{self.framework} models need a sample input to build the graph: "
                             "dv.wrap(model, ..., sample_input=x) or net.trace(x)")
        prog = self.program(tuple(args), kwargs)
        self._program = prog
        self._graph = prog.graph
        if self.taps or self.interventions:
            self._check_registrations()
        return prog.graph

    def _check_registrations(self) -> None:
        g = self._graph
        for nid, _ in list(self.taps.values()) + list(self.interventions.values()):
            if nid not in g.nodes:
                raise KeyError(f"unknown node {nid!r}")

    def feature_axis(self, node) -> int:
        return -1

    # --------------------------------------------------------- execution
    def _ctx_factory(self, prog: Program):
        g = prog.graph
        return lambda nid: InterventionContext(g.nodes[nid], self.ops, -1)

    def _input_arrays(self, args, kwargs):
        leaves, _ = jax.tree_util.tree_flatten((args, kwargs))
        return [jnp.asarray(v) for v in leaves if hasattr(v, "shape")]

    def forward(self, *args, **kwargs):
        taps, ints = self.active_taps(), self.active_interventions()
        if not taps and not ints:
            leaves, tree = jax.tree_util.tree_flatten((args, kwargs))
            dyn = [i for i, v in enumerate(leaves) if hasattr(v, "shape")]
            key = ("fast",) + _sig(args, kwargs)
            if key not in self._jit_cache:
                # arrays (positional or keyword) are inputs of the compiled call; everything else is part of the
                # key (by value), so it may be baked in — closing over a keyword array would reuse its first value
                static = [None if i in dyn else v for i, v in enumerate(leaves)]

                def call(p, arrays, _static=static, _dyn=tuple(dyn), _tree=tree):
                    full = list(_static)
                    for i, a in zip(_dyn, arrays):
                        full[i] = a
                    a_, k_ = jax.tree_util.tree_unflatten(_tree, full)
                    return self._call(p, a_, k_, scoped=False)

                self._jit_cache[key] = jax.jit(call)
            return self._jit_cache[key](self.param_arrays(), [leaves[i] for i in dyn])
        res = self.run(args, kwargs, record=list(taps), record_inputs=bool(taps))  # run() adds persistent ones
        for nid, fns in taps.items():
            for fn in fns:
                if nid in res.acts:
                    fn(nid, res.acts[nid], res.inputs.get(nid))
        return res.output

    def run(self, args, kwargs=None, record="all", record_inputs=False, interventions=None, grads=False,
            loss_fn=None, capture_attention=False) -> RunResult:
        kwargs = kwargs or {}
        args = tuple(args)
        prog = self.program(args, kwargs)
        if self._graph is None:
            self._graph = prog.graph
        g = prog.graph
        if record in ("all", True):
            wanted = frozenset(n.id for n in g.nodes.values() if n.is_compute)
        elif not record:
            wanted = frozenset()
        else:
            wanted = frozenset([record] if isinstance(record, str) else record)
        ints: dict = {}
        for src in (self.active_interventions(), interventions or {}):  # persistent first, then per-call
            for nid, fns in src.items():
                ints.setdefault(nid, []).extend(fns)
        for nid in ints:
            if nid not in g.nodes:
                raise KeyError(f"unknown node {nid!r}")
        ctx = self._ctx_factory(prog)
        arrs = self._input_arrays(args, kwargs)
        params = self.param_arrays()

        if not grads:
            key = ("run", id(prog), wanted, record_inputs, tuple((k, tuple(id(f) for f in v)) for k, v in
                                                                  sorted(ints.items())))
            fn = self._jit_cache.get(key)
            if fn is None:
                def _run(p, a):
                    return evaluate(prog, p, a, wanted, ints, None, record_inputs, ctx)
                fn = jax.jit(_run)
                if len(self._jit_cache) > 64:
                    self._jit_cache.clear()
                self._jit_cache[key] = fn
            outs, rec, rec_in = fn(params, arrs)
            out = jax.tree_util.tree_unflatten(prog.out_tree, outs)
            res = RunResult(out, dict(rec), dict(rec_in))
        else:
            dnodes = [n for n in wanted if n in prog.node_primary or n in prog.input_ids]
            shapes = {}
            for n in dnodes:
                v = prog.node_primary.get(n)
                if v is None:
                    v = prog.closed.jaxpr.invars[prog.n_params + prog.input_ids.index(n)]
                if jnp.issubdtype(v.aval.dtype, jnp.floating):
                    shapes[n] = jnp.zeros(v.aval.shape, v.aval.dtype)

            def lossf(p, d):
                outs, rec, rec_in = evaluate(prog, p, arrs, wanted, ints, d, record_inputs, ctx)
                out = jax.tree_util.tree_unflatten(prog.out_tree, outs)
                loss = loss_fn(out) if loss_fn is not None else jnp.sum(first_array(out))
                return loss, (out, rec, rec_in)

            (loss, (out, rec, rec_in)), (gp, gd) = jax.value_and_grad(lossf, argnums=(0, 1), has_aux=True)(
                params, shapes)
            res = RunResult(out, dict(rec), dict(rec_in))
            res.loss = float(loss)
            res.grads = {k: np.asarray(v) for k, v in gd.items()}
            res.param_grads = {p: np.asarray(v) for p, v in zip(self._paths, gp)}
        if capture_attention:
            for nid, n in g.nodes.items():
                if n.kind == "MultiheadAttention" and nid in wanted:
                    att = self._attention_weights(prog, nid, params, arrs, ints, ctx)
                    if att is not None:
                        res.attention[nid] = att
                        parts = _attention_parts_from(res.acts, nid, att, res.inputs.get(nid), self._param_or_none)
                        if parts is not None:
                            res.attention_parts[nid] = parts
        return res

    def _attention_weights(self, prog, nid, params, arrs, ints, ctx):
        """Softmax output inside an attention module = attention pattern."""
        grp = prog.groups.get(nid)
        if grp is None:
            return None
        jaxpr = prog.closed.jaxpr
        target = None
        for i in grp.eqns:
            e = jaxpr.eqns[i]
            if e.primitive.name == "div" and any(jaxpr.eqns[j].primitive.name == "exp" for j in grp.eqns if j < i):
                target = e.outvars[0]
            if e.primitive.name in ("custom_jvp_call", "jit", "pjit") and "softmax" in str(e.params.get("name", "")):
                target = e.outvars[0]
        if target is None:
            return None
        env_val = _eval_until(prog, params, arrs, target, ints, ctx)
        return None if env_val is None else np.asarray(env_val)

    def _param_or_none(self, path: str):
        try:
            return np.asarray(self.get_param(path)) if path in self._paths else None
        except (KeyError, ValueError):
            return None

    # --------------------------------------------------------------- misc
    def to_native(self, x):
        return jnp.asarray(x)


def _attention_parts_from(acts: dict, nid: str, probs, x=None, get_param=None) -> dict | None:
    """Inside a flax / nnx attention module, from its query / key / value projections — recorded, or recomputed
    from its weights and its input (self-attention) — and its softmax: q, k, v, scores = qkᵀ/√d (NaN where the mask
    blocked), probs, out = probs · v, as (batch, heads, tokens, …) arrays."""
    names = [(f"{nid}.{a}", f"{nid}.{b}", f"{nid}.{c}") for a, b, c in
             (("query", "key", "value"), ("q_proj", "k_proj", "v_proj"), ("query_proj", "key_proj", "value_proj"))]
    ids = next((t for t in names if all(i in acts for i in t)), None)
    if ids is not None:
        q, k, v = (np.asarray(acts[i], dtype=np.float32) for i in ids)
    elif x is not None and get_param is not None:
        x = np.asarray(x, dtype=np.float32)
        qkv = []
        for proj in ("query", "key", "value"):
            W = get_param(f"{nid}.{proj}.kernel")  # (features, heads, dim) — flax DenseGeneral
            if W is None or W.ndim != 3:
                return None
            b = get_param(f"{nid}.{proj}.bias")
            y = np.einsum("...f,fhd->...hd", x, np.asarray(W, dtype=np.float32))
            qkv.append(y + (np.asarray(b, dtype=np.float32) if b is not None else 0))
        q, k, v = qkv
    else:
        return None
    if q.ndim == 3:  # unbatched (tokens, heads, dim)
        q, k, v = q[None], k[None], v[None]
    if q.ndim != 4:
        return None
    q, k, v = (np.swapaxes(t, 1, 2) for t in (q, k, v))  # (batch, tokens, heads, dim) → (batch, heads, tokens, dim)
    p = np.asarray(probs, dtype=np.float32)
    if p.ndim == 3:
        p = p[None]
    s = q @ np.swapaxes(k, -1, -2) / np.sqrt(q.shape[-1])
    if p.shape != s.shape:
        return None
    return {"q": q, "k": k, "v": v, "scores": np.where(p > 0, s, np.nan), "probs": p, "out": p @ v}


def _eval_until(prog: Program, params, arrs, target, ints, ctx):
    from .interp import _is_drop, _is_literal, eval_eqn

    jaxpr = prog.closed.jaxpr
    env = {}
    for v, c in zip(jaxpr.constvars, prog.closed.consts):
        env[v] = c
    for v, a in zip(jaxpr.invars, list(params) + list(arrs)):
        env[v] = a
    for i, eqn in enumerate(jaxpr.eqns):
        outs = eval_eqn(eqn, [v.val if _is_literal(v) else env[v] for v in eqn.invars])
        for v, o in zip(eqn.outvars, outs):
            if not _is_drop(v):
                env[v] = o
        for nid in prog.hooks_at.get(i, ()):
            v = prog.node_primary[nid]
            for fn in ints.get(nid, []):
                env[v] = fn(env[v], ctx(nid))
        if target in env:
            return env[target]
    return None


def wrap_callable_meta(fn: Callable) -> str:
    return getattr(fn, "__name__", None) or getattr(getattr(fn, "fun", None), "__name__", None) or type(fn).__name__

