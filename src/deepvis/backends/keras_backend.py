"""Keras 3: any Keras model — Sequential, functional or subclassed — on the JAX, PyTorch or TensorFlow backend.

* layers come from the model's own layer tree: attribute names for subclassed models (``self.blocks[0]`` →
  ``blocks.0``), layer names for Sequential and functional ones; nested models are containers;
* every layer call is recorded as it runs (the dataflow follows the tensors from call to call), so a layer called
  twice is drawn twice (``dense``, ``dense@2``); the ops of a functional graph (``keras.ops.relu`` on a symbolic
  tensor) are drawn too;
* weights are the Keras variables (``dense.kernel``), edited in place with ``assign``;
* interventions replace a layer's output during its call. Gradients need the backend's own autodiff and are not
  wired up yet.
"""

from __future__ import annotations

import contextlib
import re
from typing import Any

import numpy as np

from ..core.graph import ModelGraph, Node, ParamInfo
from ..core.kinds import Role, canonical_kind, function_kind, infer_roles
from .base import ArrayOps, Backend, InterventionContext, RunResult

# recurrent cells run inside the backend's scan and attention mixes its projections with raw ops: one node each,
# like torch's nn.LSTM / nn.MultiheadAttention
ATOMIC = {"LSTM", "GRU", "RNN", "MultiheadAttention"}
_AUTO_NAME = re.compile(r"^(keras_tensor|input_layer|input)(_\d+)?$")
_CONFIG = {"filters": "out_channels", "units": "out_features", "strides": "stride", "pool_size": "kernel_size",
           "epsilon": "eps", "rate": "p", "input_dim": "num_embeddings", "output_dim": "embedding_dim",
           "dilation_rate": "dilation", "negative_slope": "negative_slope"}


class KerasOps(ArrayOps):
    """Interventions see NumPy arrays (whatever the Keras backend); the result goes back into the layer's output."""

    framework = "keras"

    def to_numpy(self, x):
        return _np(x)


def _np(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach") and hasattr(x, "cpu"):  # the PyTorch backend: skip Keras' copying path
        return x.detach().cpu().numpy()
    import keras

    return np.asarray(keras.ops.convert_to_numpy(x))


def _is_tensor(x) -> bool:
    return hasattr(x, "shape") and hasattr(x, "dtype") and not isinstance(x, (type, np.dtype))


def _flat(obj) -> list:
    if _is_tensor(obj):
        return [obj]
    if isinstance(obj, dict):
        return [t for v in obj.values() for t in _flat(v)]
    if isinstance(obj, (list, tuple)):
        return [t for v in obj for t in _flat(v)]
    return []


def _replace_first(obj, new):
    """``obj`` with its first tensor swapped for ``new`` (same nesting)."""
    done = [False]

    def go(o):
        if done[0]:
            return o
        if _is_tensor(o):
            done[0] = True
            return new
        if isinstance(o, tuple):
            return tuple(go(v) for v in o)
        if isinstance(o, list):
            return [go(v) for v in o]
        if isinstance(o, dict):
            return {k: go(v) for k, v in o.items()}
        return o

    return go(obj)


def _is_graph_model(layer) -> bool:
    return any(c.__name__ in ("Functional", "Sequential") and c.__module__.startswith("keras.")
               for c in type(layer).__mro__)


def _children(layer) -> list[tuple[str, Any]]:
    """(name, sublayer) pairs: attribute names where the model has them (subclassed models: ``blocks.0``), layer
    names for Sequential / functional models; input layers are left out."""
    import keras

    out: list[tuple[str, Any]] = []
    seen: set[int] = set()

    def add(name, v):
        if isinstance(v, keras.layers.Layer) and id(v) not in seen and not isinstance(v, keras.layers.InputLayer):
            seen.add(id(v))
            out.append((name, v))

    if not _is_graph_model(layer):
        pools = [vars(layer)]
        if isinstance(getattr(layer, "_modules", None), dict):  # the PyTorch backend keeps sublayers here
            pools.append(layer._modules)
        for pool in pools:
            for k, v in list(pool.items()):
                if k.startswith("_"):
                    continue
                if isinstance(v, dict) or type(v).__name__ == "TrackedDict":
                    for kk, x in v.items():
                        add(f"{k}.{kk}", x)
                elif isinstance(v, (list, tuple)) or type(v).__name__ == "TrackedList":
                    for i, x in enumerate(v):
                        add(f"{k}.{i}", x)
                else:
                    add(k, v)
    for v in list(getattr(layer, "_layers", None) or []):  # tracked without an attribute: functional models
        add(v.name, v)
    return out


def _attrs(layer) -> dict:
    out: dict = {}
    try:
        cfg = layer.get_config()
    except Exception:  # noqa: BLE001 - custom layers may not implement it
        cfg = {}
    for k, v in cfg.items():
        if isinstance(v, (int, float, bool, str)) or v is None:
            out[_CONFIG.get(k, k)] = v
        elif isinstance(v, (list, tuple)) and all(isinstance(x, (int, float)) for x in v):
            out[_CONFIG.get(k, k)] = list(v)
        elif isinstance(v, dict) and k == "activation" and "config" in v:  # a serialized activation
            out["activation"] = v.get("config", {}).get("name") or v.get("class_name")
    if "out_features" in out and type(layer).__name__ == "Dense":
        out["units"] = out["out_features"]
    if type(layer).__name__ in ("LSTM", "GRU", "SimpleRNN"):
        out["hidden_size"] = cfg.get("units")
    if type(layer).__name__ == "MultiHeadAttention":
        out["embed_dim"] = cfg.get("num_heads", 1) * cfg.get("key_dim", 0)
    return out


def _op_kind(op) -> str:
    name = type(op).__name__
    return function_kind(name[0].lower() + name[1:] if name else name) if name else "op"


class KerasBackend(Backend):
    framework = "keras"
    function_taps = True
    channels_last = True

    def __init__(self, model, sample_input=None, name: str | None = None):
        import keras

        super().__init__(name or model.name or type(model).__name__)
        self.model = model
        self.keras = keras
        self.ops = KerasOps()
        self.trace_mode = "keras"
        self._sample = sample_input
        self.backend_name = keras.backend.backend()
        self._layers: dict[str, Any] = {}       # path → layer
        self._path_of: dict[int, str] = {}      # id(layer) → path
        self._leaf: set[str] = set()
        whole = not _children(model) or canonical_kind(type(model).__name__, _attrs(model)) in ATOMIC
        self._walk(model, (model.name or "layer") if whole else "")  # a bare layer is one node in the model box
        self._ops: dict[int, tuple[str, Any]] = {}  # functional-graph operations that aren't layers: id → (scope, op)
        self._collect_ops(model, "")
        self._vars: dict[str, Any] = {}
        self._infos: dict[str, ParamInfo] = {}
        self._owner: dict[str, str] = {}        # param path → layer path
        self._register_params()

    # ------------------------------------------------------------------ structure
    def _walk(self, layer, path: str) -> None:
        self._layers[path] = layer
        self._path_of[id(layer)] = path
        kids = _children(layer)
        kind = canonical_kind(type(layer).__name__, _attrs(layer))
        if not kids or kind in ATOMIC:
            self._leaf.add(path)
            return
        for name, sub in kids:
            if id(sub) not in self._path_of:
                self._walk(sub, f"{path}.{name}" if path else name)

    def _collect_ops(self, layer, path: str) -> None:
        if _is_graph_model(layer):
            for op in getattr(layer, "operations", []) or []:
                if not isinstance(op, self.keras.layers.Layer):
                    self._ops[id(op)] = (path, op)
        for p, sub in list(self._layers.items()):
            if p and p.rpartition(".")[0] == path and p not in self._leaf:
                self._collect_ops(sub, p)

    def _own_variables(self, layer) -> list:
        own = list(getattr(layer, "_trainable_variables", []) or []) + \
            list(getattr(layer, "_non_trainable_variables", []) or [])
        if own or not hasattr(layer, "weights"):
            return own
        inner = {id(v) for _, sub in _children(layer) for v in sub.weights}
        return [v for v in layer.weights if id(v) not in inner]

    def _register_params(self) -> None:
        self._vars, self._infos, self._owner = {}, {}, {}
        for path, layer in self._layers.items():
            variables = layer.weights if path in self._leaf else self._own_variables(layer)
            for v in variables:
                local = v.name.rpartition("/")[2] or "weight"
                p = f"{path}.{local}" if path else local
                k = 2
                while p in self._vars:
                    p, k = f"{path}.{local}#{k}" if path else f"{local}#{k}", k + 1
                if any(v is w for w in self._vars.values()):
                    continue  # shared with a layer seen earlier
                self._vars[p] = v
                self._owner[p] = path
                self._infos[p] = ParamInfo(p, local, path, tuple(int(d) for d in v.shape), str(v.dtype),
                                           bool(v.trainable))

    def param_paths(self) -> list[str]:
        return list(self._vars)

    def param_info(self, path: str) -> ParamInfo:
        return self._infos[path]

    def get_param(self, path: str) -> np.ndarray:
        return np.array(_np(self._vars[path].value), copy=True)

    def set_param(self, path: str, value) -> None:
        v = self._vars[path]
        arr = np.asarray(value)
        if tuple(arr.shape) != self._infos[path].shape:
            raise ValueError(f"shape mismatch for {path}: {arr.shape} vs {self._infos[path].shape}")
        v.assign(arr.astype(self._infos[path].dtype, copy=False))

    # ------------------------------------------------------------------ running
    def _call_model(self, args, kwargs):
        model = self.model
        inputs = getattr(model, "inputs", None) if _is_graph_model(model) else None
        if inputs is not None and len(args) > 1 and len(inputs) == len(args):
            return model(list(args), training=False, **(kwargs or {}))  # a functional model takes a list
        return model(*args, training=False, **(kwargs or {}))  # Keras drops `training` for a call() without it

    @contextlib.contextmanager
    def _recording(self, on_call):
        """Route calls of this model's layers (and its functional graph's ops) through ``on_call``."""
        from keras.src.ops.operation import Operation

        Layer = self.keras.layers.Layer

        orig_layer, orig_op = Layer.__call__, Operation.__call__
        stack: list = []
        known = self._path_of
        ops = self._ops

        def wrap(orig):
            def __call__(obj, *args, **kwargs):
                if (id(obj) not in known and id(obj) not in ops) or (stack and stack[-1] is obj):
                    return orig(obj, *args, **kwargs)
                stack.append(obj)
                try:
                    return on_call(obj, lambda *a, **k: orig(obj, *a, **k), args, kwargs)
                finally:
                    stack.pop()
            return __call__

        Layer.__call__, Operation.__call__ = wrap(orig_layer), wrap(orig_op)
        try:
            yield
        finally:
            Layer.__call__, Operation.__call__ = orig_layer, orig_op

    def _node_id(self, obj, calls: dict) -> tuple[str, str]:
        """(node id for this call, static path) of a layer or functional op."""
        if id(obj) in self._path_of:
            path = self._path_of[id(obj)]
        else:
            scope, op = self._ops[id(obj)]
            path = f"{scope}.@{op.name}" if scope else f"@{op.name}"
        k = calls.get(path, 0)
        calls[path] = k + 1
        return (path if k == 0 else f"{path}@{k + 1}"), path

    def _tensors(self, args) -> tuple:
        """Inputs as backend tensors of the dtype the model declares, so its layers see these very objects."""
        declared = [getattr(t, "dtype", None) for t in (getattr(self.model, "inputs", None) or [])]

        def conv(a, k):
            if not isinstance(a, (np.ndarray, list)) and not np.isscalar(a):
                return a
            dtype = declared[k] if k < len(declared) and declared[k] else None
            arr = np.asarray(a)
            if dtype is None and arr.dtype.kind == "f":
                dtype = "float32"
            return self.keras.ops.convert_to_tensor(arr, dtype=dtype)

        return tuple(conv(a, k) for k, a in enumerate(args))

    def _execute(self, args, kwargs, wanted, interventions, build: ModelGraph | None = None):
        """Run the model once: per-call outputs / inputs for the ``wanted`` nodes, interventions applied; with
        ``build``, add a node and its dataflow edges per call."""
        args = self._tensors(args)
        acts: dict[str, Any] = {}
        ins: dict[str, Any] = {}
        producer: dict[int, str] = {}
        keep: list = []
        order: list[str] = []
        calls: dict[str, int] = {}
        input_ids = self._input_ids(args)
        for t, nid in input_ids:
            producer[id(t)] = nid
            keep.append(t)
        g = self._graph if build is None else build

        def on_call(obj, call, a, kw):
            nid, path = self._node_id(obj, calls)
            leaf = path in self._leaf or id(obj) in self._ops
            if build is not None:
                self._add_node(build, nid, path, obj, leaf)
            src_tensors = _flat((a, kw))
            if leaf:
                srcs = [producer[id(t)] for t in src_tensors if id(t) in producer]
                if build is not None:
                    if not srcs:
                        build.add_edge(order[-1] if order else input_ids[0][1] if input_ids else "@in:x", nid,
                                       implicit=True)
                    for s in dict.fromkeys(srcs):
                        build.add_edge(s, nid)
                order.append(nid)
            out = call(*a, **kw)
            if leaf:
                fns = list(interventions.get(nid, [])) + list(self.active_interventions().get(nid, []))
                if fns:
                    first = next(iter(_flat(out)), None)
                    if first is not None:
                        node = g.nodes.get(nid) if g is not None else None
                        arr = _np(first)
                        for fn in fns:
                            arr = fn(arr, InterventionContext(node, self.ops, self.feature_axis(node) if node else -1))
                        out = _replace_first(out, self.keras.ops.convert_to_tensor(np.asarray(arr),
                                                                                   dtype=str(_np(first).dtype)))
                for t in _flat(out):
                    producer[id(t)] = nid
                    keep.append(t)
                if wanted is None or nid in wanted:
                    first = next(iter(_flat(out)), None)
                    if first is not None:
                        acts[nid] = first
                    if src_tensors:
                        ins[nid] = src_tensors[0]
            return out

        with self._recording(on_call):
            out = self._call_model(args, kwargs)
        if build is not None:
            build.add_node(Node("@out", "Output", "Output", "output", parent="", label="output"))
            srcs = [producer[id(t)] for t in _flat(out) if id(t) in producer]
            for s in dict.fromkeys(srcs):
                build.add_edge(s, "@out")
            if not srcs and order:
                build.add_edge(order[-1], "@out", implicit=True)
        return out, acts, ins

    def _input_ids(self, args) -> list[tuple[Any, str]]:
        names = self._input_names(len(_flat(args)))
        return [(t, f"@in:{n}") for t, n in zip(_flat(args), names)]

    def _input_names(self, n: int) -> list[str]:
        """The functional model's own input names (``keras.Input(…, name="img")``); ``x`` / ``x0, x1 …`` when Keras
        named them itself (``keras_tensor_22``)."""
        inputs = getattr(self.model, "inputs", None) if _is_graph_model(self.model) else None
        if inputs is not None and len(inputs) == n:
            names = [getattr(t, "name", "") or "" for t in inputs]
            if all(names) and not any(_AUTO_NAME.match(x) for x in names) and len(set(names)) == n:
                return names
        return ["x"] if n == 1 else [f"x{k}" for k in range(n)]

    def _add_node(self, g: ModelGraph, nid: str, path: str, obj, leaf: bool) -> None:
        if nid in g.nodes:
            return
        parts = path.split(".")
        parent = ""
        for k in range(1, len(parts)):  # the containers this call sits in
            p = ".".join(parts[:k])
            if p in self._layers and p not in g.nodes:
                lay = self._layers[p]
                g.add_node(Node(p, type(lay).__name__, type(lay).__name__, "container", parent=parent,
                                label=parts[k - 1], attrs=_attrs(lay)))
            if p in self._layers:
                parent = p
        if not leaf:
            if path not in g.nodes:
                g.add_node(Node(path, type(obj).__name__, type(obj).__name__, "container", parent=parent,
                                label=parts[-1], attrs=_attrs(obj)))
            return
        if id(obj) in self._ops:
            kind = _op_kind(obj)
            g.add_node(Node(nid, kind, type(obj).__name__, "function", parent=parent, label=kind))
            return
        attrs = _attrs(obj)
        kind = canonical_kind(type(obj).__name__, attrs)
        params = {self._infos[p].name: self._infos[p] for p in self._vars if self._owner[p] == path}
        shapes = {local: info.shape for local, info in params.items()}
        roles = infer_roles(kind, shapes, "keras")
        if kind.startswith("ConvTranspose") and "weight" in roles:  # Keras keeps transposed kernels as (*k, out, in)
            nd = len(shapes[roles["weight"].name])
            roles["weight"] = Role(roles["weight"].name, out_axis=nd - 2, in_axis=nd - 1)
        if "kernel" in shapes and kind == "Linear":
            attrs["in_features"] = shapes["kernel"][0]
        if "kernel" in shapes and kind.startswith("Conv"):
            attrs["in_channels"] = shapes["kernel"][-2] * (attrs.get("groups") or 1)
        if path != nid:
            attrs = dict(attrs, shared=True)
        g.add_node(Node(nid, kind, type(obj).__name__, "module", parent=parent, params=params, roles=roles,
                        attrs=attrs, label=parts[-1].split("@")[0], module=path))

    def build_graph(self, *inputs, **kwargs) -> ModelGraph:
        args = inputs if inputs else self._sample_args()
        if args is None:
            raise ValueError("a Keras model needs a sample input to be drawn: dv.wrap(model, sample_input=x)")
        g = ModelGraph(self.name, "keras")
        g.add_node(Node("", "Model", type(self.model).__name__, "container", label=self.name))
        for t, nid in self._input_ids(args):
            g.add_node(Node(nid, "Input", "Input", "input", parent="", label=nid[4:],
                            out_shapes=[tuple(int(d) for d in t.shape)]))
        _, acts, ins = self._execute(tuple(args), kwargs, None, {}, build=g)
        for nid, v in acts.items():
            if nid in g.nodes:
                g.nodes[nid].out_shapes = [tuple(int(d) for d in v.shape)]
        for nid, v in ins.items():
            if nid in g.nodes:
                g.nodes[nid].in_shapes = [tuple(int(d) for d in v.shape)]
        out = g.nodes["@out"]
        srcs = g.pred("@out")
        if srcs and g.nodes[srcs[0]].out_shapes:
            out.in_shapes = [g.nodes[srcs[0]].out_shapes[0]]
        for e in g.edges:
            s = g.nodes[e.src]
            if s.out_shapes:
                e.shape = tuple(s.out_shapes[0])
        g.link_children()
        self._graph = g
        return g

    def _sample_args(self):
        if self._sample is None:
            return None
        return self._sample if isinstance(self._sample, tuple) else (self._sample,)

    def forward(self, *args, **kwargs):
        return self._call_model(args, kwargs)

    def run(self, args, kwargs=None, record="all", record_inputs=False, interventions=None, grads=False,
            loss_fn=None, capture_attention=False) -> RunResult:
        from ..errors import OptionError

        if grads:
            raise OptionError("gradients aren't available for Keras models yet.",
                              "trace without grads=True; for gradients wrap the model in PyTorch or JAX directly.")
        if self._graph is None:  # interventions are handed their node
            self.build_graph(*args)
        if record in ("all", True):
            wanted = None
        elif record in (None, False):
            wanted = set()
        else:
            wanted = {record} if isinstance(record, str) else set(record)
        out, acts, ins = self._execute(tuple(args), kwargs, wanted, interventions or {})
        res = RunResult(out)
        res.acts = {k: _np(v) for k, v in acts.items()}
        if record_inputs:
            res.inputs = {k: _np(v) for k, v in ins.items()}
        for t, nid in self._input_ids(tuple(args)):
            if wanted is None or nid in wanted:
                res.acts[nid] = _np(t)
                res.inputs.setdefault(nid, None)
        if wanted is None or "@out" in wanted:
            first = next(iter(_flat(out)), None)
            if first is not None:
                res.acts["@out"] = _np(first)
        for nid, fns in self.active_taps().items():
            for fn in fns:
                if nid in res.acts:
                    fn(nid, res.acts[nid], res.inputs.get(nid))
        return res

    def feature_axis(self, node: Node) -> int:
        return 1 if node is not None and node.attrs.get("data_format") == "channels_first" else -1

    def to_native(self, x):
        return self.keras.ops.convert_to_tensor(np.asarray(x))

    def export(self):
        """The live Keras model (every edit is assigned to its variables)."""
        return self.model

    def save(self, path: str) -> str:
        self.model.save(path)
        return path

    def clone(self) -> "KerasBackend":
        m = self.keras.models.clone_model(self.model)
        m.set_weights(self.model.get_weights())
        return KerasBackend(m, self._sample, self.name)


def example_inputs(model) -> tuple | None:
    """Inputs shaped like a functional model's declared ones (unknown sizes → 1; floats ~ N(0, 1), integers → small
    ids); ``None`` for a subclassed model, which declares nothing."""
    inputs = getattr(model, "inputs", None) if _is_graph_model(model) else None
    if not inputs:
        return None
    rng = np.random.default_rng(0)
    out = []
    for t in inputs:
        shape = tuple(1 if d is None else int(d) for d in t.shape)
        dtype = np.dtype(str(t.dtype))
        out.append(rng.standard_normal(shape).astype(dtype) if dtype.kind == "f"
                   else rng.integers(0, 10, shape).astype(dtype))
    return tuple(out) if len(out) > 1 else out[0]
