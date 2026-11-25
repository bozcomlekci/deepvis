"""ONNX: any network exported to ONNX — from PyTorch, TensorFlow / Keras (tf2onnx), scikit-learn, PaddlePaddle … —
run in onnxruntime.

* layers from the exporter's module scopes: PyTorch's ``/layer1/layer1.0/conv1/Conv`` names (TorchScript exporter),
  the ``pkg.torch.onnx.name_scopes`` metadata (dynamo exporter), ``scope/op`` names (tf2onnx), and dotted
  initializer names (``layer1.0.conv1.weight``);
* a node that owns weights is a layer (Conv, Gemm, MatMul + bias Add → Linear, BatchNormalization, LayerNormalization,
  Gather on a table → Embedding); the rest are operations;
* every intermediate tensor is recorded by exposing it as an output of the session; weight edits rebuild it.

onnxruntime has no hooks, so ONNX models can be traced, inspected and edited but not intervened on or
differentiated: those need the original PyTorch / JAX model.
"""

from __future__ import annotations

import ast
import copy
import os
import re
from typing import Any

import numpy as np

from ..core.graph import ModelGraph, Node, ParamInfo
from ..core.kinds import ELEMENTWISE_KINDS, FUNCTION_KINDS, NORM_KINDS, Role, canonical_kind, function_kind
from .base import ArrayOps, Backend, RunResult

TRANSPARENT = {"Identity", "Dropout"}  # inference: output = input
META = {"Shape", "Size", "Constant", "ConstantOfShape", "Range", "NonZero"}  # sizes and constants, never drawn

OP_KINDS = {
    "Gemm": "Linear", "MatMul": "matmul", "Einsum": "matmul", "Relu": "ReLU", "LeakyRelu": "LeakyReLU",
    "Sigmoid": "Sigmoid", "Tanh": "Tanh", "Softmax": "Softmax", "LogSoftmax": "LogSoftmax", "Gelu": "GELU", "Elu": "ELU",
    "Selu": "SELU", "Celu": "CELU", "Softplus": "Softplus", "Softsign": "Softsign", "HardSigmoid": "Hardsigmoid",
    "HardSwish": "Hardswish", "Mish": "Mish", "PRelu": "PReLU", "Erf": "erf", "Exp": "exp", "Log": "log", "Sqrt": "sqrt",
    "Pow": "pow", "Abs": "abs", "Neg": "neg", "Reciprocal": "reciprocal", "Sin": "sin", "Cos": "cos", "Add": "add",
    "Sub": "sub", "Mul": "mul", "Div": "div", "Where": "where", "GlobalAveragePool": "AdaptiveAvgPool2d",
    "GlobalMaxPool": "AdaptiveMaxPool2d", "Flatten": "Flatten", "Reshape": "Reshape", "Squeeze": "Reshape",
    "Unsqueeze": "Reshape", "Transpose": "Transpose", "Concat": "Concat", "Split": "split", "Slice": "getitem",
    "Gather": "getitem", "GatherElements": "getitem", "Expand": "expand", "Tile": "repeat",
    "BatchNormalization": "BatchNorm", "LayerNormalization": "LayerNorm", "InstanceNormalization": "InstanceNorm2d",
    "GroupNormalization": "GroupNorm", "Resize": "Upsample", "Upsample": "Upsample", "ReduceMean": "mean",
    "ReduceSum": "sum", "ReduceMax": "max", "Clip": "Hardtanh", "Cast": "Identity",
}
_POOL = {"MaxPool": "MaxPool", "AveragePool": "AvgPool"}
_ATTR_NAMES = {"kernel_shape": "kernel_size", "strides": "stride", "pads": "padding", "dilations": "dilation",
               "group": "groups", "epsilon": "eps", "axis": "dim", "alpha": "negative_slope"}
# where a layer op keeps its weight: Conv's kernel, Gemm's matrix, Gather's table …
_WEIGHT_SLOT = {"Conv": 1, "ConvTranspose": 1, "Gemm": 1, "MatMul": 1, "BatchNormalization": 1,
                "LayerNormalization": 1, "InstanceNormalization": 1, "GroupNormalization": 1, "LSTM": 1, "GRU": 1,
                "RNN": 1, "PRelu": 1, "Gather": 0}
_KNOWN = ELEMENTWISE_KINDS | NORM_KINDS | set(FUNCTION_KINDS.values())
_GENERATED = re.compile(r"^(onnx::|val_\d|_val_\d|_?tensor_constant|const(ant)?_?\d|arg\d|_onnx)")


def _generated(name: str) -> bool:
    """A tensor name the exporter made up (``onnx::Conv_193``, ``val_78``), not the model's own parameter name."""
    return "::" in name or bool(_GENERATED.match(name))


class OnnxOps(ArrayOps):
    framework = "onnx"

    def set_index(self, x, idx, value):
        y = np.array(x, copy=True)
        y[idx] = value
        return y

    def asarray(self, value, like):
        return np.asarray(value, dtype=np.asarray(like).dtype)

    def clip(self, x, lo, hi):
        return np.clip(x, lo, hi)

    def zeros_like(self, x):
        return np.zeros_like(x)

    def to_numpy(self, x):
        return np.asarray(x)


def _scope_levels(parts: list[str]) -> list[str]:
    """``["layer1", "layer1.0", "conv1"]`` → ``["layer1", "layer1.0", "layer1.0.conv1"]``. A part is a local name, or
    a dotted path relative to one of the levels above (TorchScript names a Sequential's children from the nearest
    module that isn't one: ``features.1/block/block.0/block.0.0``)."""
    levels: list[str] = []
    comps: list[str] = []
    for c in parts:
        c = c.strip()
        if not c:
            continue
        for k in range(len(comps)):  # longest match first: a full path, then shorter tails
            if c.startswith(".".join(comps[k:]) + "."):
                comps = comps[:k] + c.split(".")
                break
        else:
            comps = comps + c.split(".")
        levels.append(".".join(comps))
    return levels


def _rebase(init: str, scope: str) -> str | None:
    """Where a weight called ``init`` sits, seen from the op's ``scope``: under it (an attention's ``out_proj``) or
    beside it (a convolution with its BatchNorm folded in, scoped as the BatchNorm). Twin blocks that start out equal
    share one stored tensor (``layers.0.…`` used in ``layers.1``): indices are re-based onto the op's own block."""
    ip, sp = init.split("."), scope.split(".") if scope else []

    def same(a, b):
        return a == b or (a.isdigit() and b.isdigit())

    if len(ip) > len(sp) and all(same(a, b) for a, b in zip(ip, sp)):
        return ".".join(sp + ip[len(sp):])
    if sp and len(ip) == len(sp) and all(same(a, b) for a, b in zip(ip[:-1], sp[:-1])):
        return ".".join(sp[:-1] + ip[-1:])
    return None


def _scope(parts: list[str]) -> str:
    levels = _scope_levels(parts)
    return levels[-1] if levels else ""


def _common(paths: list[str]) -> str:
    """The deepest module containing all of ``paths``."""
    split = [p.split(".") if p else [] for p in paths]
    out = []
    for xs in zip(*split):
        if any(x != xs[0] for x in xs):
            break
        out.append(xs[0])
    return ".".join(out)


class OnnxBackend(Backend):
    framework = "onnx"
    function_taps = False

    def __init__(self, model, name: str | None = None, sample_input=None):
        import onnx
        from onnx import numpy_helper

        mp = onnx.load(os.fspath(model)) if isinstance(model, (str, os.PathLike)) else copy.deepcopy(model)
        try:
            from onnx import inliner

            mp = inliner.inline_local_functions(mp)  # exporters may wrap modules as ONNX functions
        except Exception:  # noqa: BLE001 - older onnx: nothing to inline
            pass
        try:
            mp = onnx.shape_inference.infer_shapes(mp)
        except Exception:  # noqa: BLE001 - shapes are decoration
            pass
        stem = os.path.splitext(os.path.basename(os.fspath(model)))[0] if isinstance(model, (str, os.PathLike)) else ""
        super().__init__(name or stem or mp.graph.name or "onnx")
        self.model = mp
        self.ops = OnnxOps()
        self.trace_mode = "onnx"
        self._sample = sample_input
        g = mp.graph
        self._arrays: dict[str, np.ndarray] = {}   # initializer name → value (all, incl. integer constants)
        for init in g.initializer:
            self._arrays[init.name] = numpy_helper.to_array(init).copy()
        init_names = set(self._arrays)
        self.input_names = [i.name for i in g.input if i.name not in init_names]
        self.output_names = [o.name for o in g.output]
        self._types = {v.name: v.type for v in list(g.value_info) + list(g.input) + list(g.output)}
        self._paths: dict[str, str] = {}    # param path → initializer name
        self._infos: dict[str, ParamInfo] = {}
        self._sessions: dict[str, Any] = {}
        self._layers: list[dict] = []       # one per drawn node: id, onnx node indices, primary output tensor …
        self._analyze()

    # ------------------------------------------------------------------ structure
    def _float_init(self, name: str) -> bool:
        """A weight: a float initializer with at least two numbers (one-number ones are constants — scales, eps)."""
        a = self._arrays.get(name)
        return a is not None and a.dtype.kind == "f" and a.size > 1

    def _share_directly(self) -> None:
        """The TorchScript exporter stores identical tensors once and hands them out through ``Identity`` nodes (all
        the zero biases a fresh BatchNorm folds into convolutions): point every user at the stored tensor."""
        g = self.model.graph
        to: dict[str, str] = {}
        for n in g.node:
            if n.op_type == "Identity" and len(n.input) == 1 and (n.input[0] in self._arrays or n.input[0] in to):
                to[n.output[0]] = to.get(n.input[0], n.input[0])
        if not to:
            return
        outputs = {o.name for o in g.output}
        for n in g.node:
            for j, t in enumerate(n.input):
                if t in to:
                    n.input[j] = to[t]
        keep = [n for n in g.node if not (n.op_type == "Identity" and n.output[0] in to and n.output[0] not in outputs)]
        del g.node[:]
        g.node.extend(keep)

    def _is_bias(self, name: str, weight: str) -> bool:
        """A bias for ``x @ weight``: one number per output, as ``(n,)`` or ``(1, n)`` (scikit-learn's)."""
        b, w = self._arrays.get(name), self._arrays.get(weight)  # one output: a one-number bias still counts
        if b is None or w is None or b.dtype.kind != "f" or not self._float_init(weight):
            return False
        return b.size == w.shape[-1] and b.shape[-1] == w.shape[-1] and all(d == 1 for d in b.shape[:-1])

    def _analyze(self) -> None:
        self._share_directly()
        g = self.model.graph
        nodes = list(g.node)
        metas = [self._meta(n) for n in nodes]
        made_by = {o: i for i, n in enumerate(nodes) for o in n.output if o}
        # data-dependent nodes: fed (transitively) by an input; the rest compute constants and stay hidden
        data = set(self.input_names)
        live = [False] * len(nodes)
        for i, n in enumerate(nodes):
            if n.op_type not in META and any(t in data for t in n.input):
                live[i] = True
                data.update(o for o in n.output if o)
        self._data, self._live = data, live
        consumers: dict[str, list[int]] = {}
        for i, n in enumerate(nodes):
            for t in n.input:
                consumers.setdefault(t, []).append(i)
        alias: dict[str, str] = {}
        singles = []
        for i, n in enumerate(nodes):
            if not live[i]:
                continue
            if n.op_type in TRANSPARENT or (n.op_type == "Cast" and len(n.input) == 1):
                alias[n.output[0]] = n.input[0]
            else:
                singles.append(i)
        self._alias = alias
        # one PyTorch op is one node: the dynamo exporter tags each ONNX node with the op it came from (the last
        # name scope), so a decomposed attention or GELU folds back into one
        groups: list[list[int]] = []
        by_op: dict[tuple, int] = {}
        for i in singles:
            key = metas[i][0] if metas[i] else None
            if key is not None and key in by_op:
                groups[by_op[key]].append(i)
                continue
            if key is not None:
                by_op[key] = len(groups)
            groups.append([i])
        groups = sorted((part for m in groups for part in ([m] if len(m) == 1 or self._convex(m, nodes, consumers)
                                                           else [[i] for i in m])), key=lambda m: m[0])
        group_of = {i: gi for gi, m in enumerate(groups) for i in m}
        merged: set[int] = set()
        for m in groups:  # x @ W + b (TorchScript exporter): one Linear
            n = nodes[m[0]]
            if len(m) != 1 or n.op_type != "MatMul" or len(n.input) != 2 or not self._float_init(n.input[1]):
                continue
            users = consumers.get(n.output[0], [])
            if len(users) == 1 and users[0] in group_of and len(groups[group_of[users[0]]]) == 1:
                add = nodes[users[0]]
                other = [t for t in add.input if t != n.output[0]]
                if add.op_type == "Add" and len(other) == 1 and self._is_bias(other[0], n.input[1]):
                    m.append(users[0])
                    merged.add(group_of[users[0]])
        groups = [m for gi, m in enumerate(groups) if gi not in merged]
        # which module each group belongs to; ops the exporter's optimizer made carry no scope: give them the
        # deepest module their neighbours share (GPT-2's fused q/k/v Split sits in its attention)
        self._scope_cls: dict[str, str] = {}
        self._scope_seen: set[str] = set()
        anchors = [self._anchor(m, nodes) for m in groups]
        placed = [self._module_path(nodes[m[0] if a is None else a], metas[m[0] if a is None else a])
                  for m, a in zip(groups, anchors)]
        paths = [p for p, _ in placed]
        known = [k for _, k in placed]
        if any(known) and not all(known):
            where = {i: gi for gi, m in enumerate(groups) for i in m}

            def neighbours(i):
                for t in nodes[i].input:
                    j = made_by.get(self._resolve(t))
                    if j in where:
                        yield where[j]
                todo = [o for o in nodes[i].output if o]
                while todo:
                    for c in consumers.get(todo.pop(), []):
                        if c in where:
                            yield where[c]
                        elif live[c] and nodes[c].output and nodes[c].output[0] in alias:
                            todo.extend(nodes[c].output)  # through an Identity / Dropout

            for _ in range(4):
                changed = False
                for gi, m in enumerate(groups):
                    if known[gi]:
                        continue
                    near = [paths[gj] for i in m for gj in neighbours(i) if known[gj] and gj != gi]
                    if near:
                        paths[gi], known[gi], changed = _common(near), True, True
                if not changed:
                    break
        kinds_of = [self._group_op(m, nodes, metas) for m in groups]
        has_params = [bool(self._params(m, nodes, a)[0]) for m, a in zip(groups, anchors)]
        # a module that owns weights is one layer: the other ops in its own scope (GPT-2 Conv1D's reshapes around
        # its matmul) fold into it, like a PyTorch leaf module
        owner: dict[str, int] = {}
        for gi, path in enumerate(paths):
            if path and has_params[gi]:
                owner.setdefault(path, gi)
        by_scope: dict[str, list[int]] = {}
        for gi, path in enumerate(paths):
            by_scope.setdefault(path, []).append(gi)
        for path, gis in by_scope.items():  # a weightless leaf module made of several ops (GELU's mul/tanh/pow)
            if (path and path not in owner and len(gis) > 1 and not any(q.startswith(path + ".") for q in by_scope)
                    and len({kinds_of[gi] for gi in gis}) > 1):
                owner[path] = gis[0]
        folded: dict[int, list[int]] = {}
        keep = []
        for gi, path in enumerate(paths):
            o = owner.get(path)
            if o is not None and o != gi and not has_params[gi]:
                folded.setdefault(o, []).extend(groups[gi])
            else:
                keep.append(gi)
        for o, extra in list(folded.items()):  # merge only a contiguous block: nothing may leave it and come back
            if not self._convex(sorted(groups[o] + extra), nodes, consumers):
                keep.extend(gi for gi, path in enumerate(paths) if path == paths[o] and gi != o and gi not in keep
                            and not has_params[gi])
                del folded[o]
        keep.sort()
        ops_in: dict[str, set] = {}
        for gi in keep:
            ops_in.setdefault(paths[gi], set()).add(kinds_of[gi])
        nested = {p for p in ops_in if any(q.startswith(p + ".") for q in ops_in)}
        share = {}
        for gi in keep:
            share[paths[gi]] = share.get(paths[gi], 0) + 1
        used: set[str] = set()
        self._layers = []
        for gi in keep:
            members = sorted(groups[gi] + folded.get(gi, []))
            path = paths[gi]
            # a layer: it owns weights, it is a folded leaf module, or its scope is a leaf module doing one kind of
            # op (once, or per call)
            is_layer = has_params[gi] or (bool(path) and (gi in folded or (path not in nested
                                                                            and len(ops_in[path]) == 1)))
            # a module whose own ops don't all fold into its weight op (nn.MultiheadAttention: in_proj, the attention,
            # out_proj) stays a container; the weight op inside is named after its weight
            crowded = is_layer and has_params[gi] and gi not in folded and (share[path] > 1 or path in nested)
            cls = self._scope_cls.get(path) if is_layer and not crowded else None
            kind, roles, params, attrs, type_name = self._describe(members, nodes, metas, anchors[gi], cls)
            if is_layer and not path:  # weights but no names to place them (scikit-learn): linear, linear_1 …
                base = kind.lower() if kind[0].isupper() else kind
                nid, k = base, 1
                while nid in used:
                    nid, k = f"{base}_{k}", k + 1
            elif is_layer:
                anchor = nodes[anchors[gi]] if anchors[gi] is not None else None
                base = f"{path}.{self._weight_name(params, kind, anchor, path)}" if crowded else path
                nid, k = base, 2
                while nid in used:
                    nid, k = f"{base}@{k}", k + 1
            else:
                fx = metas[members[0]][0][-1] if metas[members[0]] else ""
                base = "@" + (fx if fx and fx.isidentifier() else kind.lower())
                nid, k = base, 1
                while nid in used:
                    nid, k = f"{base}_{k}", k + 1
            used.add(nid)
            inside = {o for m in members for o in nodes[m].output}
            self._layers.append({"id": nid, "members": members, "kind": kind, "roles": roles, "params": params,
                                 "out": nodes[members[-1]].output[0],  # nodes are in topological order: last = output
                                 "ins": [t for m in members for t in nodes[m].input if t and t not in inside],
                                 "scope": (path if crowded else path.rpartition(".")[0]) if is_layer else path,
                                 "op": "module" if is_layer else "function", "type": type_name, "attrs": attrs})
        layer_ids = {L["id"] for L in self._layers}
        for L in self._layers:  # an op whose scope is a layer sits beside it, in the layer's own container
            while L["scope"] in layer_ids:
                L["scope"] = L["scope"].rpartition(".")[0]

    @staticmethod
    def _weight_name(params: dict, kind: str, node, path: str) -> str:
        """``in_proj_weight`` → ``in_proj``; ``cls_token`` → ``cls_token``. A folded copy (``val_27``, the transposed
        matrix) is named from the parameter the dynamo exporter's op read (``%p_layers_0_self_attn_in_proj_weight``),
        else after the kind."""
        init = next(iter(params.values()), "")
        local = init.rpartition(".")[2] if not _generated(init) else ""
        if not local and node is not None:
            fx = next((p.value for p in node.metadata_props if p.key == "pkg.torch.onnx.fx_node"), "")
            prefix = "p_" + path.replace(".", "_") + "_"
            local = next((m[len(prefix):] for m in re.findall(r"%(p_\w+)", fx) if m.startswith(prefix)), "")
        for suffix in ("_weight", "_bias", ".weight"):
            if local.endswith(suffix) and len(local) > len(suffix):
                return local[: -len(suffix)]
        return local if local and local not in ("weight", "bias", "kernel") else kind.lower()

    def _meta(self, n) -> tuple[tuple, tuple] | None:
        """(name scopes, class hierarchy) from the PyTorch dynamo exporter: ``('', 'layer1', 'layer1.0', …, 'relu_3')``
        down to the op itself, and the classes along it (``('ResNet', …, 'aten.relu.default')``)."""
        props = {p.key: p.value for p in n.metadata_props}
        raw = props.get("pkg.torch.onnx.name_scopes")
        if not raw:
            return None
        try:
            scopes = tuple(str(s) for s in ast.literal_eval(raw))
            classes = tuple(str(c) for c in ast.literal_eval(props.get("pkg.torch.onnx.class_hierarchy") or "[]"))
        except (ValueError, SyntaxError):
            return None
        return (scopes, classes) if scopes else None

    _FAMILY = {"Conv": "Conv", "ConvTranspose": "Conv", "Gemm": ("Linear", "Conv1D"), "MatMul": ("Linear", "Conv1D"),
               "Gather": "Embedding", "BatchNormalization": "BatchNorm", "LayerNormalization": "Norm"}

    def _module_path(self, n, meta) -> tuple[str, bool]:
        """(the module this op belongs to, whether the file says so) — from the exporter's scope metadata, its
        scoped node name or its weight's name."""
        init = next((t.rpartition(".")[0] for t in n.input if self._float_init(t) and "." in t and not _generated(t)),
                    None)
        if meta:
            scopes, classes = meta
            for k, sc in enumerate(scopes[:-1]):
                if sc:
                    self._scope_seen.add(sc)
                    if k < len(classes) - 1:
                        self._scope_cls.setdefault(sc, classes[k].rsplit(".", 1)[-1])
            path = _scope(list(scopes[:-1]))
            fam = self._FAMILY.get(n.op_type)
            owner = classes[-2].rsplit(".", 1)[-1] if len(classes) >= 2 else ""
            if init is not None and fam and not any(f in owner for f in ((fam,) if isinstance(fam, str) else fam)):
                return _rebase(init, path) or path, True  # an attention's out_proj; a conv scoped as its folded BN
            return path, True
        if "/" in n.name:  # TorchScript exporter: /layer1/layer1.0/conv1/Conv
            levels = _scope_levels(n.name.split("/")[:-1])
            self._scope_seen.update(levels)
            if levels:
                under = _rebase(init, levels[-1]) if init is not None else None
                return (under if under and under.startswith(levels[-1] + ".") else levels[-1]), True
        if init is not None:  # a weight called layer1.0.conv1.weight lives in layer1.0.conv1
            return init, True
        return "", False

    def _convex(self, members: list[int], nodes, consumers) -> bool:
        """No drawn path leaves ``members`` and re-enters them (merging them keeps the graph acyclic). Size
        computations (``Shape`` → … → an LSTM's zero initial state) are hidden, so they don't count."""
        inside = set(members)
        outs = {o for m in members for o in nodes[m].output}
        frontier = [c for o in outs for c in consumers.get(o, []) if c not in inside and self._live[c]]
        seen = set()
        while frontier:
            i = frontier.pop()
            if i in seen:
                continue
            seen.add(i)
            for o in nodes[i].output:
                for c in consumers.get(o, []):
                    if c in inside:
                        return False
                    if self._live[c]:
                        frontier.append(c)
        return True

    @staticmethod
    def _aten(meta) -> str | None:
        """The PyTorch op an ONNX node came from (``aten.scaled_dot_product_attention.default`` → its name)."""
        if not meta or not meta[1]:
            return None
        parts = meta[1][-1].split(".")
        return parts[1] if len(parts) >= 3 and parts[0] in ("aten", "prims") else None

    def _group_op(self, members, nodes, metas) -> str:
        aten = self._aten(metas[members[0]])
        return aten if aten and len(members) > 1 else nodes[members[0]].op_type

    @staticmethod
    def _leaf_kind(ops) -> str:
        """What a weightless module made of several ONNX ops computes, from the ops it uses."""
        s = {n.op_type for n in ops}
        if "Erf" in s or ("Tanh" in s and "Pow" in s):
            return "GELU"
        if "Sigmoid" in s and "Mul" in s:
            return "SiLU"
        if {"ReduceMean", "Sqrt"} <= s or {"ReduceMean", "Reciprocal"} <= s:
            return "RMSNorm" if "Sub" not in s else "LayerNorm"
        if {"Exp", "ReduceSum", "Div"} <= s:
            return "Softmax"
        return "Function"

    def _anchor(self, members, nodes) -> int | None:
        """The member that holds the layer's weight in a known slot (a Conv's kernel, a Gemm's matrix …), if any."""
        for m in members:
            n = nodes[m]
            slot = _WEIGHT_SLOT.get(n.op_type)
            if slot is None or len(n.input) <= slot or not self._float_init(n.input[slot]):
                continue
            nd = self._arrays[n.input[slot]].ndim
            if n.op_type in ("MatMul", "Gather") and nd != 2:
                continue  # a batched constant, a lookup in a table that isn't an embedding matrix
            return m
        return None

    def _params(self, members, nodes, anchor) -> tuple[dict, dict]:
        """({local name: initializer}, roles) for one drawn node: the weights in an anchor's slots, plus any named
        weight an elementwise op uses (a learned positional embedding, an RMSNorm gain). Exporter-made constants
        (``val_78``: a causal mask) are not weights."""
        params: dict[str, str] = {}
        roles: dict[str, Role] = {}
        if anchor is not None:
            n = nodes[anchor]
            op = n.op_type
            ins = list(n.input)

            def put(local, idx, role=None):
                if len(ins) > idx and ins[idx] and self._float_init(ins[idx]):
                    params[local] = ins[idx]
                    roles[local] = role or Role(local)

            if op in ("Conv", "ConvTranspose"):
                out_ax, in_ax = (0, 1) if op == "Conv" else (1, 0)
                put("weight", 1, Role("weight", out_axis=out_ax, in_axis=in_ax))
                put("bias", 2, Role("bias", out_axis=0))
            elif op == "Gemm":
                trans = next((a.i for a in n.attribute if a.name == "transB"), 0)
                put("weight", 1, Role("weight", out_axis=0 if trans else 1, in_axis=1 if trans else 0))
                put("bias", 2, Role("bias", out_axis=0))
            elif op == "MatMul":
                put("weight", 1, Role("weight", out_axis=1, in_axis=0))
                for m in members:  # its bias: an Add right after it
                    add = nodes[m]
                    if add.op_type == "Add" and n.output[0] in add.input:
                        b = next((t for t in add.input if t != n.output[0] and self._is_bias(t, ins[1])), None)
                        if b is not None:
                            params["bias"], roles["bias"] = b, Role("bias", out_axis=self._arrays[b].ndim - 1)
                            break
            elif op == "BatchNormalization":
                for k, local in enumerate(("weight", "bias", "running_mean", "running_var"), start=1):
                    put(local, k, Role(local, out_axis=0))
            elif op in ("LayerNormalization", "InstanceNormalization", "GroupNormalization"):
                put("weight", 1, Role("weight", out_axis=0))
                put("bias", 2, Role("bias", out_axis=0))
            elif op == "Gather":
                put("weight", 0, Role("weight", out_axis=1, in_axis=0))
            elif op in ("LSTM", "GRU", "RNN"):
                put("weight_ih", 1)
                put("weight_hh", 2)
                put("bias", 3)
            elif op == "PRelu":
                put("weight", 1, Role("weight", out_axis=0))
        for m in members:
            for t in nodes[m].input:
                if t in params.values() or not self._float_init(t) or _generated(t):
                    continue
                local = t.rpartition(".")[2] or "weight"
                while local in params:
                    local += "_"
                params[local], roles[local] = t, Role(local)
        return params, roles

    def _describe(self, members, nodes, metas, anchor, cls) -> tuple[str, dict, dict, dict, str]:
        """(canonical kind, roles, params, attributes, type name) for one drawn node."""
        params, roles = self._params(members, nodes, anchor)
        ops = [nodes[m] for m in members]
        if anchor is not None:
            n = nodes[anchor]
            op = n.op_type
            w = self._arrays[params["weight"]] if "weight" in params else None
            if op in ("Conv", "ConvTranspose"):
                kind = ("ConvTranspose" if op == "ConvTranspose" else "Conv") + f"{max(w.ndim - 2, 1)}d"
            elif op in ("Gemm", "MatMul"):
                kind = "Linear"
            elif op == "Gather":
                kind = "Embedding"
            elif op == "InstanceNormalization":
                kind = "InstanceNorm2d"
            else:
                kind = OP_KINDS.get(op, op)
            return kind, roles, params, self._attrs(n, params), cls or op
        aten = {self._aten(metas[m]) for m in members} - {None}
        ckind = canonical_kind(cls) if cls else None
        if ckind and (ckind != cls or ckind in _KNOWN):  # the module's class says what it computes
            kind = ckind
        elif len(members) > 1 and len(aten) == 1:
            kind = function_kind(next(iter(aten)))
        elif len(members) > 1:
            kind = self._leaf_kind(ops)
            if kind == "Function" and params:  # a module adding its own weight (ViT's position embeddings): that op
                holder = next(o for o in ops if any(t in params.values() for t in o.input))
                kind = OP_KINDS.get(holder.op_type, holder.op_type)
        elif ops[0].op_type in _POOL:
            ks = next((list(a.ints) for a in ops[0].attribute if a.name == "kernel_shape"), [])
            kind = f"{_POOL[ops[0].op_type]}{max(len(ks), 1)}d"
        else:
            kind = OP_KINDS.get(ops[0].op_type, ops[0].op_type)
        if kind in NORM_KINDS:  # a gain / shift per feature
            roles = {k: Role(k, out_axis=0) if self._arrays[params[k]].ndim == 1 else r for k, r in roles.items()}
        type_name = cls or (next(iter(aten)) if len(members) > 1 and len(aten) == 1 else ops[0].op_type)
        return kind, roles, params, self._attrs(ops[0], params), type_name

    def _attrs(self, n, params) -> dict:
        from onnx import helper

        out = {}
        for a in n.attribute:
            try:
                v = helper.get_attribute_value(a)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(v, bytes):
                v = v.decode("utf-8", "replace")
            if isinstance(v, (int, float, str)) or (isinstance(v, list) and all(isinstance(x, (int, float)) for x in v)):
                out[_ATTR_NAMES.get(a.name, a.name)] = v
        shape = self._arrays[params["weight"]].shape if "weight" in params else None
        if shape is not None and n.op_type in ("Gemm", "MatMul") and len(shape) == 2:
            trans = out.get("transB", 0) if n.op_type == "Gemm" else 0
            out["out_features"], out["in_features"] = (shape[0], shape[1]) if trans else (shape[1], shape[0])
        if shape is not None and n.op_type in ("Conv", "ConvTranspose"):
            g = out.get("groups", 1)
            out["out_channels"], out["in_channels"] = ((shape[0], shape[1] * g) if n.op_type == "Conv"
                                                       else (shape[1] * g, shape[0]))
        if shape is not None and n.op_type == "Gather":
            out["num_embeddings"], out["embedding_dim"] = shape
        if n.op_type in ("LSTM", "GRU", "RNN") and "weight_ih" in params:
            out["input_size"] = self._arrays[params["weight_ih"]].shape[-1]
            out["bidirectional"] = out.get("direction") == "bidirectional"
            out.setdefault("num_layers", 1)
        return out

    # ------------------------------------------------------------------ params
    def _register_params(self) -> None:
        """Parameters are named after the layer that uses them (``layer1.0.conv1.weight``), whatever the exporter
        called the tensor (``onnx::Conv_193`` once BatchNorm is folded in, a tied ``lm_head.weight`` used by the
        embedding)."""
        self._paths, self._infos = {}, {}
        for L in self._layers:
            owner = L["id"]
            for local, init in L["params"].items():
                path = f"{owner}.{local}" if owner and not owner.startswith("@") else (
                    init if not _generated(init) else f"{owner.lstrip('@')}.{local}")
                k = 2
                while path in self._paths:
                    path, k = f"{path.split('#')[0]}#{k}", k + 1
                self._paths[path] = init
                L.setdefault("param_paths", {})[local] = path
                a = self._arrays[init]
                self._infos[path] = ParamInfo(path, local, owner, tuple(a.shape), str(a.dtype), True)
        claimed = set(self._paths.values())
        for name, a in self._arrays.items():  # a named weight no drawn node uses directly: still editable
            if name not in claimed and self._float_init(name) and not _generated(name):
                self._paths[name] = name
                self._infos[name] = ParamInfo(name, name.rpartition(".")[2], name.rpartition(".")[0], tuple(a.shape),
                                              str(a.dtype), True)

    def param_paths(self) -> list[str]:
        if not self._paths:
            self._register_params()
        return list(self._paths)

    def param_info(self, path: str) -> ParamInfo:
        self.param_paths()
        return self._infos[path]

    def get_param(self, path: str) -> np.ndarray:
        self.param_paths()
        return np.array(self._arrays[self._paths[path]], copy=True)

    def set_param(self, path: str, value) -> None:
        self.param_paths()
        init = self._paths[path]
        v = np.asarray(value)
        if v.shape != self._arrays[init].shape:
            raise ValueError(f"shape mismatch for {path}: {v.shape} vs {self._arrays[init].shape}")
        self._unshare(init)
        init = self._paths[path]
        self._arrays[init] = v.astype(self._arrays[init].dtype, copy=True)
        self._sessions = {}  # rebuilt with the new weights on the next run

    def _unshare(self, init: str) -> None:
        """Exporters store identical tensors once (every freshly initialised LayerNorm's ones): before an edit, give
        each layer that only shares the tensor by coincidence its own copy. A module called twice keeps sharing."""
        users = [p for p, t in self._paths.items() if t == init]
        if len(users) < 2:
            return
        natural = init.rpartition(".")[0] if not _generated(init) else None
        for path in users:
            L = next((L for L in self._layers if path in L.get("param_paths", {}).values()), None)
            if L is None or (natural is not None and L["id"].split("@")[0] == natural):
                continue
            new, k = path, 2
            while new in self._arrays:
                new, k = f"{path}~{k}", k + 1
            self._arrays[new] = self._arrays[init].copy()
            for m in L["members"]:
                node = self.model.graph.node[m]
                for j, t in enumerate(node.input):
                    if t == init:
                        node.input[j] = new
            L["params"] = {loc: (new if t == init else t) for loc, t in L["params"].items()}
            L["ins"] = [new if t == init else t for t in L["ins"]]
            self._paths[path] = new

    # ------------------------------------------------------------------ graph
    def build_graph(self, *inputs, **kwargs) -> ModelGraph:
        self.param_paths()
        g = ModelGraph(self.name, "onnx")
        g.add_node(Node("", "Model", self.name, "container", label=self.name))
        containers: dict[str, str] = {}
        layer_ids = {L["id"] for L in self._layers}
        for L in self._layers:
            parts = L["scope"].split(".") if L["scope"] else []
            for k in range(1, len(parts) + 1):
                p = ".".join(parts[:k])
                if p not in layer_ids:  # a layer is never also a container
                    containers.setdefault(p, self._container_class(p, parts[k] if k < len(parts) else ""))
        for p in sorted(containers, key=lambda s: (s.count("."), s)):
            parent = p.rpartition(".")[0]
            while parent and parent not in containers:
                parent = parent.rpartition(".")[0]
            g.add_node(Node(p, containers[p], containers[p], "container", parent=parent, label=p.split(".")[-1]))
        producer: dict[str, str] = {}
        for name in self.input_names:
            nid = f"@in:{name}"
            g.add_node(Node(nid, "Input", "Input", "input", parent="", label=name))
            producer[name] = nid
        for L in self._layers:
            infos = {local: self._infos[path] for local, path in L.get("param_paths", {}).items()}
            parent = L["scope"]
            while parent and parent not in containers:
                parent = parent.rpartition(".")[0]
            label = L["id"].rpartition(".")[2] if L["op"] == "module" else L["kind"]
            g.add_node(Node(L["id"], L["kind"], L["type"], L["op"], parent=parent, params=infos, roles=L["roles"],
                            attrs=L["attrs"], label=label))
            for o in self.model.graph.node[L["members"][-1]].output:
                producer.setdefault(o, L["id"])
            producer[L["out"]] = L["id"]
            for m in L["members"]:
                for o in self.model.graph.node[m].output:
                    producer.setdefault(o, L["id"])
        for L in self._layers:
            for t in L["ins"]:
                src = producer.get(self._resolve(t))
                if src is not None and src != L["id"]:
                    g.add_edge(src, L["id"])
        g.add_node(Node("@out", "Output", "Output", "output", parent="", label="output"))
        for o in self.output_names:
            src = producer.get(self._resolve(o))
            if src is not None:
                g.add_edge(src, "@out")
        g.link_children()
        self._graph = g
        args = inputs if inputs else self._sample_args()
        if args is not None:
            try:
                res = self.run(args, record="all")
                for nid, v in res.acts.items():
                    if nid in g.nodes:
                        g.nodes[nid].out_shapes = [tuple(np.shape(v))]
                for nid, v in res.inputs.items():
                    if nid in g.nodes and v is not None:
                        g.nodes[nid].in_shapes = [tuple(np.shape(v))]
                for nid, a in zip(self.input_names, args):
                    g.nodes[f"@in:{nid}"].out_shapes = [tuple(np.shape(a))]
                g.nodes["@out"].in_shapes = [tuple(np.shape(res.output if not isinstance(res.output, tuple)
                                                             else res.output[0]))]
                for e in g.edges:
                    s = g.nodes[e.src]
                    if s.out_shapes:
                        e.shape = tuple(s.out_shapes[0])
            except Exception as exc:  # noqa: BLE001 - shapes are decoration
                self._shape_error = exc
        return g

    def _container_class(self, path: str, child: str) -> str:
        if path in self._scope_cls:
            return self._scope_cls[path]
        numbered = child.isdigit()
        if path in self._scope_seen:  # a module that ran (TorchScript names carry no classes)
            return "Sequential" if numbered else "Block"
        return "ModuleList" if numbered else "Block"  # never called itself: a list of modules

    def _resolve(self, t: str) -> str:
        while t in self._alias:
            t = self._alias[t]
        return t

    def _sample_args(self):
        if self._sample is None:
            return None
        return self._sample if isinstance(self._sample, tuple) else (self._sample,)

    def example_inputs(self) -> tuple:
        """Inputs shaped like the model's declared ones (dynamic sizes → 1; float → N(0, 1), integer → small ids)."""
        from onnx import helper

        rng = np.random.default_rng(0)
        out = []
        for name in self.input_names:
            t = self._types[name].tensor_type
            dims = [d.dim_value if d.dim_value > 0 else 1 for d in t.shape.dim]
            dtype = helper.tensor_dtype_to_np_dtype(t.elem_type)
            if np.dtype(dtype).kind == "f":
                out.append(rng.standard_normal(dims).astype(dtype))
            else:
                out.append(rng.integers(0, 10, dims).astype(dtype))
        return tuple(out)

    # ------------------------------------------------------------------ execution
    def _session(self, record: bool):
        key = "record" if record else "plain"
        if key in self._sessions:
            return self._sessions[key]
        import onnx
        import onnxruntime as ort
        from onnx import numpy_helper

        mp = copy.deepcopy(self.model)
        del mp.graph.initializer[:]
        mp.graph.initializer.extend(numpy_helper.from_array(a, name) for name, a in self._arrays.items())
        names = []
        if record:
            have = {o.name for o in mp.graph.output}
            for L in self._layers:
                for t in [L["out"]] + [self._resolve(x) for x in L["ins"]]:
                    if t in self._data and t not in have and t in self._types and t not in self.input_names:
                        mp.graph.output.append(onnx.ValueInfoProto(name=t, type=self._types[t]))
                        have.add(t)
            names = [o.name for o in mp.graph.output]
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        sess = ort.InferenceSession(mp.SerializeToString(), opts, providers=["CPUExecutionProvider"])
        self._sessions[key] = (sess, names or [o.name for o in mp.graph.output])
        return self._sessions[key]

    def _feeds(self, args) -> dict:
        from onnx import helper

        from ..core.tensor import to_numpy

        if len(args) != len(self.input_names):
            raise ValueError(f"this ONNX model takes {len(self.input_names)} input(s) ({', '.join(self.input_names)}), "
                             f"got {len(args)}")
        feeds = {}
        for name, a in zip(self.input_names, args):
            dtype = helper.tensor_dtype_to_np_dtype(self._types[name].tensor_type.elem_type)
            feeds[name] = np.asarray(to_numpy(a)).astype(dtype, copy=False)
        return feeds

    def forward(self, *args, **kwargs):
        sess, names = self._session(record=False)
        outs = sess.run(None, self._feeds(args))
        return outs[0] if len(outs) == 1 else tuple(outs)

    def run(self, args, kwargs=None, record="all", record_inputs=False, interventions=None, grads=False,
            loss_fn=None, capture_attention=False) -> RunResult:
        from ..errors import OptionError

        if interventions or self.active_interventions():
            raise OptionError("ONNX models run in onnxruntime, which can't change a tensor in the middle of a run.",
                              "trace and edit weights here; for interventions wrap the original PyTorch / JAX model.")
        if grads:
            raise OptionError("ONNX models carry no gradients.", "use the original PyTorch / JAX model for grads=True.")
        sess, names = self._session(record=True)
        feeds = self._feeds(tuple(args))
        vals = dict(zip(names, sess.run(names, feeds)))
        vals.update(feeds)
        outs = [vals[o] for o in self.output_names]
        res = RunResult(outs[0] if len(outs) == 1 else tuple(outs))
        g = self._graph
        wanted = None if record in ("all", True) else set([record] if isinstance(record, str) else (record or []))
        for L in self._layers:
            if wanted is not None and L["id"] not in wanted:
                continue
            v = vals.get(L["out"])
            if v is not None:
                res.acts[L["id"]] = v
                first = next((t for t in map(self._resolve, L["ins"]) if t in self._data and t in vals), None)
                res.inputs[L["id"]] = vals.get(first) if first else None
        for name in self.input_names:
            if wanted is None or f"@in:{name}" in wanted:
                res.acts[f"@in:{name}"] = feeds[name]
        if g is not None and (wanted is None or "@out" in wanted):
            res.acts["@out"] = outs[0]
        for nid, fns in self.active_taps().items():
            for fn in fns:
                if nid in res.acts:
                    fn(nid, res.acts[nid], res.inputs.get(nid))
        return res

    def feature_axis(self, node: Node) -> int:
        conv_like = node.kind.startswith(("Conv", "BatchNorm", "MaxPool", "AvgPool", "Adaptive", "InstanceNorm",
                                          "GroupNorm"))
        return 1 if conv_like or (node.out_shapes and len(node.out_shapes[0]) == 4) else -1

    def export(self):
        """The ONNX model with the current (edited) weights."""
        from onnx import numpy_helper

        mp = copy.deepcopy(self.model)
        del mp.graph.initializer[:]
        mp.graph.initializer.extend(numpy_helper.from_array(a, name) for name, a in self._arrays.items())
        return mp

    def save(self, path: str) -> str:
        import onnx

        big = sum(a.nbytes for a in self._arrays.values()) > 1 << 30  # protobuf stops at 2 GB: weights beside it
        onnx.save(self.export(), path, save_as_external_data=big, location=os.path.basename(path) + ".data")
        return path

    def to_native(self, x):
        return np.asarray(x)

    def clone(self) -> "OnnxBackend":
        return OnnxBackend(self.export(), self.name, self._sample)


def is_onnx(model: Any) -> bool:
    return type(model).__name__ == "ModelProto" and type(model).__module__.startswith("onnx")
