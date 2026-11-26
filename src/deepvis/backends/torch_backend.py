"""PyTorch backend.

* module tree from ``named_modules``
* dataflow from ``torch.fx`` (a tracer that remembers which container each functional
  op lives in); falls back to a hook-based execution trace for untraceable models
* forward hooks for recording / intervening, tensor hooks for activation gradients
* in-place parameter edits under ``torch.no_grad()`` (device & dtype preserved)
"""

from __future__ import annotations

import contextlib
import math
import operator

import numpy as np
import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

from ..core.graph import ModelGraph, Node, ParamInfo, iter_shapes
from ..core.kinds import CONV_KINDS, Role, canonical_kind, function_kind, infer_roles
from ..core.tensor import first_array, replace_first_array, to_numpy
from .base import ArrayOps, Backend, InterventionContext, RunResult

LEAF_TYPES: tuple[type, ...] = (nn.MultiheadAttention, nn.LSTM, nn.GRU, nn.RNN)


def _library_leaves() -> tuple[type, ...]:
    """Layers from libraries that are clearer as one node than as their internal ops (only if already imported):
    PyTorch Geometric message passing (gather / scatter over edges)."""
    import sys

    out = []
    pyg = sys.modules.get("torch_geometric.nn.conv")
    if pyg is not None and hasattr(pyg, "MessagePassing"):
        out.append(pyg.MessagePassing)
    return tuple(out)
_META_METHODS = {"size", "dim", "numel", "ndimension", "nelement", "type", "get_device"}
_TRANSPARENT = {"contiguous", "to", "float", "double", "half", "type_as", "detach", "clone", "cpu",
                "cuda", "requires_grad_", "__getattr__"}
_ATTR_KEYS = ("in_features", "out_features", "kernel_size", "stride", "padding", "dilation", "groups",
              "in_channels", "out_channels", "num_heads", "embed_dim", "eps", "p", "normalized_shape",
              "num_embeddings", "embedding_dim", "negative_slope", "dim", "approximate", "num_features",
              "momentum", "output_size", "start_dim", "end_dim", "batch_first", "hidden_size",
              "num_layers", "bias", "inplace", "head_dim", "elementwise_affine", "padding_idx")


class TorchOps(ArrayOps):
    framework = "torch"

    def set_index(self, x, idx, value):
        y = x.clone()
        y[idx] = value if not isinstance(value, np.ndarray) else torch.as_tensor(value, dtype=x.dtype,
                                                                                   device=x.device)
        return y

    def asarray(self, value, like):
        return torch.as_tensor(np.array(value), dtype=like.dtype, device=like.device)  # copy: may be read-only

    def clip(self, x, lo, hi):
        return torch.clamp(x, lo, hi)

    def zeros_like(self, x):
        return torch.zeros_like(x)

    def to_numpy(self, x):
        return to_numpy(x)

    def feature_index(self, x, axis, neurons):
        nd = x.ndim
        ax = axis % nd
        sel = torch.as_tensor(np.asarray(neurons), device=x.device) if isinstance(
            neurons, (list, tuple, np.ndarray)) else neurons
        return (slice(None),) * ax + (sel,)


@contextlib.contextmanager
def _no_fastpath():
    """PyTorch's fused transformer fast path skips submodule calls; disable it so hooks/fx see every layer."""
    mha = getattr(torch.backends, "mha", None)
    if mha is None or not hasattr(mha, "set_fastpath_enabled"):
        yield
        return
    prev = mha.get_fastpath_enabled()
    mha.set_fastpath_enabled(False)
    try:
        yield
    finally:
        mha.set_fastpath_enabled(prev)


def _prune_virtual(g: ModelGraph) -> None:
    """Drop the per-output nodes of multi-output ops that nothing consumes."""
    used = {e.src for e in g.edges}
    dead = {n.id for n in g.nodes.values() if n.attrs.get("virtual") and n.kind == "getitem" and n.id not in used}
    if not dead:
        return
    for d in dead:
        del g.nodes[d]
    g.edges = [e for e in g.edges if e.dst not in dead and e.src not in dead]
    g._edge_keys = {(e.src, e.dst) for e in g.edges}
    for n in g.nodes.values():
        if "outputs" in n.attrs:
            n.attrs["outputs"] = [k if k not in dead else None for k in n.attrs["outputs"]]


def _replace_output(out, i: int, new):
    items = list(out)
    items[i] = new
    return type(out)(items) if not isinstance(out, list) else items


_SDPA_ARGS = ("query", "key", "value", "attn_mask", "dropout_p", "is_causal", "scale", "enable_gqa")


def _sdpa_weights(args, kwargs, max_bytes: int = 64 << 20):
    """The attention pattern ``softmax(QKᵀ·scale + mask)`` that ``F.scaled_dot_product_attention`` computes
    but never returns (fused kernels). Handles boolean / additive masks, ``is_causal`` and grouped-query heads."""
    b = dict(zip(_SDPA_ARGS, args))
    b.update(kwargs)
    q, k = b.get("query"), b.get("key")
    if not (isinstance(q, torch.Tensor) and isinstance(k, torch.Tensor)) or q.dim() < 2:
        return None
    if q.shape[-2] * k.shape[-2] * max(1, q.numel() // max(q.shape[-1] * q.shape[-2], 1)) * 4 > max_bytes:
        return None  # a very long sequence: skip rather than allocate gigabytes
    with torch.no_grad():
        q, k = q.detach().float(), k.detach().float()
        if q.dim() == 4 and k.dim() == 4 and k.shape[1] != q.shape[1] and q.shape[1] % k.shape[1] == 0:
            k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)  # GQA: shared k/v heads
        scale = b.get("scale") or 1.0 / math.sqrt(q.shape[-1])
        s = (q @ k.transpose(-2, -1)) * scale
        mask = b.get("attn_mask")
        if b.get("is_causal"):
            L, S = s.shape[-2], s.shape[-1]
            s = s.masked_fill(~torch.ones(L, S, dtype=torch.bool, device=s.device).tril(), float("-inf"))
        if isinstance(mask, torch.Tensor):
            s = s.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else s + mask.float()
        return to_numpy(torch.softmax(s, dim=-1).nan_to_num(0.0))


def _attention_parts(q, k, v, keep=None, add=None, causal=False, scale=None, max_bytes: int = 48 << 20):
    """Every stage of scaled dot-product attention, as numpy (batch, heads, tokens, …) arrays:
    q, k, v · scores = QKᵀ·scale (+ additive mask; blocked cells NaN) · probs = softmax · out = probs @ v."""
    B, H, T, d = q.shape
    S = k.shape[-2]
    if B * H * T * S * 4 * 3 > max_bytes:
        return None
    with torch.no_grad():
        q, k, v = q.detach().float(), k.detach().float(), v.detach().float()
        if k.shape[1] != H and H % k.shape[1] == 0:  # grouped-query attention: shared k/v heads
            k, v = k.repeat_interleave(H // k.shape[1], 1), v.repeat_interleave(H // v.shape[1], 1)
        s = (q @ k.transpose(-2, -1)) * (scale or 1.0 / math.sqrt(d))
        if add is not None:
            s = s + add.float()
        blocked = torch.zeros_like(s, dtype=torch.bool)
        if causal:
            blocked |= ~torch.ones(T, S, dtype=torch.bool, device=s.device).tril()
        if keep is not None:
            blocked |= ~keep
        s = s.masked_fill(blocked, float("-inf"))
        p = torch.softmax(s, -1).nan_to_num(0.0)
        o = p @ v
        scores = s.masked_fill(blocked, float("nan"))
        return {k_: to_numpy(t_) for k_, t_ in (("q", q), ("k", k), ("v", v), ("scores", scores), ("probs", p),
                                              ("out", o))}


def _sdpa_parts(args, kwargs):
    b = dict(zip(_SDPA_ARGS, args))
    b.update(kwargs)
    q, k, v = b.get("query"), b.get("key"), b.get("value")
    if not all(isinstance(t, torch.Tensor) and t.dim() == 4 for t in (q, k, v)):
        return None
    mask = b.get("attn_mask")
    keep = mask if isinstance(mask, torch.Tensor) and mask.dtype == torch.bool else None
    add = mask if isinstance(mask, torch.Tensor) and mask.dtype != torch.bool else None
    return _attention_parts(q, k, v, keep, add, bool(b.get("is_causal")), b.get("scale"))


def _mha_parts(mod: nn.MultiheadAttention, args, kwargs):
    """nn.MultiheadAttention computed step by step from its own projection weights (its masks use True = blocked)."""
    if mod.bias_k is not None or mod.add_zero_attn or len(args) < 3:
        return None
    query, key, value = args[:3]
    if not all(isinstance(t, torch.Tensor) for t in (query, key, value)) or query.dim() != 3:
        return None
    with torch.no_grad():
        if not mod.batch_first:
            query, key, value = (t.transpose(0, 1) for t in (query, key, value))
        E, H = mod.embed_dim, mod.num_heads
        if mod._qkv_same_embed_dim:
            wq, wk, wv = mod.in_proj_weight.chunk(3)
        else:
            wq, wk, wv = mod.q_proj_weight, mod.k_proj_weight, mod.v_proj_weight
        bq, bk, bv = mod.in_proj_bias.chunk(3) if mod.in_proj_bias is not None else (None, None, None)
        heads = lambda t: t.reshape(t.shape[0], t.shape[1], H, E // H).transpose(1, 2)  # noqa: E731
        q, k, v = (heads(F.linear(x, w, b_)) for x, w, b_ in ((query, wq, bq), (key, wk, bk), (value, wv, bv)))
        keep, add = None, None
        B, T, S = q.shape[0], q.shape[2], k.shape[2]
        m = kwargs.get("attn_mask", args[5] if len(args) > 5 else None)
        if isinstance(m, torch.Tensor):
            m = m.reshape(B, H, T, S) if m.dim() == 3 else m
            if m.dtype == torch.bool:
                keep = ~m
            else:
                add = m
        kpm = kwargs.get("key_padding_mask", args[3] if len(args) > 3 else None)
        if isinstance(kpm, torch.Tensor):
            kpm = kpm[:, None, None, :]
            if kpm.dtype == torch.bool:
                keep = ~kpm if keep is None else keep & ~kpm
            else:
                add = kpm if add is None else add + kpm
        causal = bool(kwargs.get("is_causal")) and m is None
        return _attention_parts(q, k, v, keep, add, causal)


def _snap(t):
    """A recorded activation must not change afterwards: later in-place ops (``ReLU(inplace=True)``,
    ``out += identity``) would otherwise rewrite it."""
    return t.detach().clone() if isinstance(t, torch.Tensor) else t


def _module_attrs(m: nn.Module) -> dict:
    attrs = {}
    for k in _ATTR_KEYS:
        if k in m.__dict__:
            v = m.__dict__[k]
            if isinstance(v, (int, float, bool, str, tuple, list)) or v is None:
                attrs[k] = list(v) if isinstance(v, tuple) else v
    if isinstance(m, nn.Linear):
        attrs["bias"] = m.bias is not None
    if hasattr(m, "bias") and isinstance(getattr(m, "bias", None), (torch.Tensor, type(None))) and "bias" not in attrs:
        attrs["bias"] = getattr(m, "bias", None) is not None
    return attrs


# torch.export: an unflattened program keeps the module tree and the weights' names, but every submodule becomes an
# InterpreterModule running its slice of the graph. Its class comes back from the export's nn_module_stack, and its
# settings from the aten call it makes: (argument index, attribute) per op.
_ATEN_ATTRS = {
    **{f"conv{d}d": ((3, "stride"), (4, "padding"), (5, "dilation"), (6, "groups")) for d in (1, 2, 3)},
    **{f"conv_transpose{d}d": ((3, "stride"), (4, "padding"), (5, "output_padding"), (6, "groups"))
       for d in (1, 2, 3)},
    **{f"max_pool{d}d": ((1, "kernel_size"), (2, "stride"), (3, "padding"), (4, "dilation")) for d in (1, 2, 3)},
    **{f"avg_pool{d}d": ((1, "kernel_size"), (2, "stride"), (3, "padding")) for d in (1, 2, 3)},
    **{f"adaptive_avg_pool{d}d": ((1, "output_size"),) for d in (1, 2, 3)},
    **{f"adaptive_max_pool{d}d": ((1, "output_size"),) for d in (1, 2, 3)},
    "batch_norm": ((6, "momentum"), (7, "eps")), "layer_norm": ((1, "normalized_shape"), (4, "eps")),
    "group_norm": ((1, "num_groups"), (4, "eps")), "dropout": ((1, "p"),), "leaky_relu": ((1, "negative_slope"),),
    "hardtanh": ((1, "min_val"), (2, "max_val")), "softmax": ((1, "dim"),), "log_softmax": ((1, "dim"),),
    "gelu": (), "linear": (), "embedding": ((2, "padding_idx"),),
}


def _exported_attrs(m: nn.Module) -> dict:
    """The attributes an unflattened module had, read back from the aten call it makes (stride, padding, eps …)."""
    graph = getattr(m, "graph", None)
    out: dict = {}
    if graph is None:
        return out
    for n in graph.nodes:
        if n.op != "call_function":
            continue
        op = str(getattr(n.target, "_opname", "") or getattr(n.target, "__name__", "")).rstrip("_")
        spec = _ATEN_ATTRS.get(op)
        if spec is None:
            continue
        for idx, key in spec:
            v = n.args[idx] if idx < len(n.args) else n.kwargs.get(key)
            if isinstance(v, (int, float, bool, str)) or (isinstance(v, (list, tuple)) and all(
                    isinstance(x, (int, float)) for x in v)):
                out[key] = list(v) if isinstance(v, tuple) else v
        if op == "linear" or op.startswith("conv"):
            out["bias"] = len(n.args) > 2 and n.args[2] is not None
        break
    return out


def from_exported(ep) -> tuple[nn.Module, tuple | None]:
    """A ``torch.export.ExportedProgram`` as a module deepvis can trace: unflattened, with each submodule's original
    class remembered (from the export's ``nn_module_stack``), and the program's own example inputs."""
    classes: dict[str, str] = {}
    for n in ep.graph.nodes:
        for val in (n.meta.get("nn_module_stack") or {}).values():
            path, cls = val[0], val[1]
            name = cls if isinstance(cls, str) else getattr(cls, "__qualname__", str(cls))
            classes.setdefault(path, name.rsplit(".", 1)[-1])
    module = torch.export.unflatten(ep)
    module._dv_classes = classes
    try:
        module._dv_input_names = [str(n) for n in ep.graph_signature.user_inputs]
    except Exception:  # noqa: BLE001 - names are decoration
        pass
    example = None
    if getattr(ep, "example_inputs", None):
        args, kwargs = ep.example_inputs
        example = tuple(args) if not kwargs else None
    return module, example


class _ScopeTracer(fx.Tracer):
    def __init__(self, leaf_types):
        super().__init__()
        self._leaf_types = leaf_types
        self._dv_scope = [""]

    def is_leaf_module(self, m, qualname):
        return isinstance(m, self._leaf_types) or super().is_leaf_module(m, qualname) or not any(
            True for _ in m.children())

    def call_module(self, m, forward, args, kwargs):
        self._dv_scope.append(self.path_of_module(m))
        try:
            return super().call_module(m, forward, args, kwargs)
        finally:
            self._dv_scope.pop()

    def create_node(self, *args, **kwargs):
        n = super().create_node(*args, **kwargs)
        n.meta["dv_scope"] = self._dv_scope[-1]
        return n


def _fx_flat_nodes(obj) -> list[fx.Node]:
    out = []
    if isinstance(obj, fx.Node):
        out.append(obj)
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            out += _fx_flat_nodes(o)
    elif isinstance(obj, dict):
        for o in obj.values():
            out += _fx_flat_nodes(o)
    elif isinstance(obj, slice):
        out += _fx_flat_nodes([obj.start, obj.stop, obj.step])
    return out


class TorchBackend(Backend):
    framework = "torch"
    @property
    def function_taps(self) -> bool:
        """Persistent monitors / interventions on functional ops (residual ``+``, ``softmax`` …) need the
        dispatch tracer's op numbering."""
        return self.trace_mode == "dispatch"

    def __init__(self, model: nn.Module, name: str | None = None, leaf_modules: tuple = (),
                 sample_input=None):
        super().__init__(name or type(model).__name__)
        self.model = model
        self.ops = TorchOps()
        self._leaf_types = LEAF_TYPES + _library_leaves() + tuple(leaf_modules)
        self._tensors: dict[str, torch.Tensor] = {}
        self._param_infos: dict[str, ParamInfo] = {}
        self._index_params()
        self._fx_nodes: dict[str, str] = {}  # fx node name -> our node id (op nodes)
        self._gm: fx.GraphModule | None = None
        self._module_calls: dict[str, list[str]] = {}  # module path -> node ids per call
        self._native_handles: list = []
        self._running = False  # inside run(): persistent functional-op hooks stand aside
        self._fn_replay: "_Dispatch | None" = None
        self._sample = sample_input
        self.trace_mode = "none"

    # ------------------------------------------------------------ params
    def _index_params(self) -> None:
        self._tensors.clear()
        self._param_infos.clear()
        for path, p in self.model.named_parameters():
            self._add_tensor(path, p, True)
        for path, b in self.model.named_buffers():
            if b is not None and b.is_floating_point():
                self._add_tensor(path, b, False)

    def _add_tensor(self, path: str, t: torch.Tensor, trainable: bool) -> None:
        module, _, name = path.rpartition(".")
        self._tensors[path] = t
        self._param_infos[path] = ParamInfo(path, name, module, tuple(t.shape), str(t.dtype).replace("torch.", ""),
                                            trainable)

    def param_paths(self) -> list[str]:
        return list(self._tensors)

    def param_info(self, path: str) -> ParamInfo:
        return self._param_infos[path]

    @property
    def has_data(self) -> bool:
        """False when the weights live on the meta device (a 70B model instantiated in no time and no memory)."""
        return not any(t.is_meta for t in self._tensors.values())

    def get_param(self, path: str) -> np.ndarray:
        t = self._tensors[path]
        if t.is_meta:
            from ..errors import OptionError

            raise OptionError(f"{path} is on the meta device: it has a shape but no values.",
                              "load real weights (e.g. model.to_empty(device='cpu') then load a state_dict) to see them.")
        return to_numpy(t).copy()

    def get_native_param(self, path: str) -> torch.Tensor:
        return self._tensors[path]

    def set_param(self, path: str, value: np.ndarray) -> None:
        t = self._tensors[path]
        v = np.asarray(value)
        if tuple(v.shape) != tuple(t.shape):
            raise ValueError(f"shape mismatch for {path}: {v.shape} vs {tuple(t.shape)}")
        with torch.no_grad():
            t.copy_(torch.as_tensor(v, dtype=t.dtype, device=t.device))

    # ------------------------------------------------------------- graph
    def build_graph(self, *inputs, **kwargs) -> ModelGraph:
        with _no_fastpath(), self._untouched(rng=True):  # wrapping must not change the model or your random stream
            return self._build_graph(*inputs, **kwargs)

    @contextlib.contextmanager
    def _untouched(self, rng: bool = False):
        """Looking must not change the model: put its buffers back after a pass of ours (a model in training mode
        would otherwise update its BatchNorm running statistics), and with ``rng`` the random-number state too."""
        saved = [(m, k, b, b.detach().clone()) for m in self.model.modules() for k, b in m._buffers.items()
                 if b is not None]
        cpu = torch.get_rng_state() if rng else None
        cuda = torch.cuda.get_rng_state_all() if rng and torch.cuda.is_available() and torch.cuda.is_initialized() \
            else None
        try:
            yield
        finally:
            with torch.no_grad():
                for m, k, b, v in saved:
                    if m._buffers.get(k) is b:
                        b.copy_(v)
                    else:  # the forward replaced the buffer outright
                        m._buffers[k] = v
            if cpu is not None:
                torch.set_rng_state(cpu)
            if cuda is not None:
                torch.cuda.set_rng_state_all(cuda)

    def _build_graph(self, *inputs, **kwargs) -> ModelGraph:
        g = ModelGraph(self.name, "torch")
        modules = dict(self.model.named_modules())
        self._modules = modules
        is_leaf = {p: (isinstance(m, self._leaf_types) or not any(True for _ in m.children()))
                   for p, m in modules.items()}
        # a user-defined model with no submodules still has a forward worth tracing into; only a bare
        # torch.nn layer (wrap(nn.Linear(...))) is a single opaque node
        is_leaf[""] = isinstance(self.model, self._leaf_types) or (
            is_leaf[""] and type(self.model).__module__.startswith("torch."))
        by_id = {id(t): pth for pth, t in self._tensors.items()}
        # a module nested inside a "leaf" (e.g. MHA.out_proj) is hidden
        hidden = {p for p in modules if p and any(
            is_leaf.get(a) for a in _prefixes(p))}
        exported = getattr(self.model, "_dv_classes", None) or {}
        for path, m in modules.items():
            if path in hidden:
                continue
            interp = type(m).__name__ in ("InterpreterModule", "UnflattenedModule") and path in exported
            tname = exported[path] if interp else type(m).__name__
            attrs = _exported_attrs(m) if interp else _module_attrs(m)
            kind = canonical_kind(tname, attrs)
            # HuggingFace GPT-2: a Linear storing W as (in, out)
            hf_conv1d = tname == "Conv1D" and (hasattr(m, "nf") or interp)
            if hf_conv1d:
                kind = "Linear"
            leaf = is_leaf[path]
            params = {}
            if leaf:  # by identity, so tied weights (lm_head.weight is wte.weight) are found under either name
                for local, t in list(m.named_parameters(remove_duplicate=False)) + list(m.named_buffers()):
                    pth = by_id.get(id(t))
                    if pth is not None:
                        params.setdefault(local, self._param_infos[pth])
            else:
                for name, _ in list(m.named_parameters(recurse=False)) + list(m.named_buffers(recurse=False)):
                    full = f"{path}.{name}" if path else name
                    if full in self._param_infos:
                        params[name] = self._param_infos[full]
            roles = infer_roles(kind, {k: v.shape for k, v in params.items()}, "torch")
            if hf_conv1d and "weight" in roles:
                roles["weight"] = Role(roles["weight"].name, out_axis=1, in_axis=0)
            node = Node(
                id=path, kind=kind if leaf else tname, type_name=tname, op="module" if leaf else "container",
                parent=None if path == "" else path.rpartition(".")[0], params=params, roles=roles, attrs=attrs,
                label=path.split(".")[-1] if path else self.name,
            )
            g.add_node(node)
        g.link_children()

        self._gm = None
        self._fx_nodes = {}
        self._module_calls = {}
        self._op_sequence = []
        if not is_leaf[""]:
            sample = inputs if inputs else self._sample_tuple()
            built = False
            if sample is not None:  # eager dispatch trace: follows real control flow, sees functional ops
                try:
                    self._build_dispatch(g, sample, kwargs)
                    self.trace_mode = "dispatch"
                    built = True
                except Exception as exc:  # noqa: BLE001
                    self._dispatch_error = exc
                    g = self._fresh_tree(g)
            if not built:
                try:
                    self._build_fx(g)
                    self.trace_mode = "fx"
                    unused = [n.id for n in g.nodes.values() if n.op == "module" and n.attrs.get("unused")]
                    if unused and sample is not None and self._called_modules(sample, kwargs) & set(unused):
                        raise RuntimeError(f"fx trace skipped modules that run eagerly: {unused[:5]}")
                except Exception as exc:  # noqa: BLE001 - any tracing failure falls back
                    self._fx_error = exc
                    g = self._fresh_tree(g)
                    if sample is not None:
                        self._build_from_hooks(g, sample, kwargs)
                        self.trace_mode = "hooks"
                    else:
                        self._build_sequential(g)
                        self.trace_mode = "sequential"
        else:
            g.add_node(Node("@in:x", "Input", "Input", "input", parent="", label="x"))
            g.add_node(Node("@out", "Output", "Output", "output", parent="", label="output"))
            g.link_children()
            g.nodes[""].op = "module"
            g.changed()
            g.add_edge("@in:x", "")
            g.add_edge("", "@out")
            self._module_calls = {"": [""]}
            self.trace_mode = "single"

        self._graph = g
        sample = inputs if inputs else self._sample_tuple()
        if sample is not None:
            try:
                self._fill_shapes(g, sample, kwargs)
            except Exception as exc:  # noqa: BLE001 - shapes are optional decoration
                self._shape_error = exc
        return g

    def _called_modules(self, args: tuple, kwargs: dict) -> set[str]:
        called: set[str] = set()
        handles = [m.register_forward_hook(lambda mod, i, o, _p=p: called.add(_p))
                   for p, m in self.model.named_modules() if p]
        try:
            with torch.no_grad():
                self.model(*args, **(kwargs or {}))
        finally:
            for h in handles:
                h.remove()
        return called

    def _fresh_tree(self, g: ModelGraph) -> ModelGraph:
        """Drop everything fx added, keeping only the module tree."""
        ng = ModelGraph(g.name, g.framework)
        for n in g.nodes.values():
            if n.op in ("module", "container") and not n.module:
                n.attrs.pop("unused", None)
                n.children = []
                ng.add_node(n)
        ng.link_children()
        self._gm, self._fx_nodes, self._module_calls = None, {}, {}
        return ng

    def _sample_tuple(self):
        if self._sample is None:
            return None
        return self._sample if isinstance(self._sample, tuple) else (self._sample,)

    def _graph_modules(self, g: ModelGraph) -> dict[str, bool]:
        return {n.id: n.op == "module" for n in g.nodes.values()
                if n.op in ("module", "container") and n.id != "" and not n.module}

    def _add_call_node(self, g: ModelGraph, path: str, nid: str, k: int) -> None:
        if k > 0 and nid not in g.nodes:
            base = g.nodes[path]
            g.add_node(Node(nid, base.kind, base.type_name, "module", parent=base.parent, params=dict(base.params),
                            roles=dict(base.roles), attrs=dict(base.attrs, shared=True), module=path,
                            label=f"{base.display} (call {k + 1})"))
        self._module_calls.setdefault(path, []).append(nid)

    def _input_names(self, args: tuple) -> list[str]:
        import inspect

        try:
            names = [p.name for p in inspect.signature(self.model.forward).parameters.values()
                     if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        except (TypeError, ValueError):
            names = []
        names = list(getattr(self.model, "_dv_input_names", None) or names)  # torch.export: the program's names
        out = []
        for i, a in enumerate(args):
            base = names[i] if i < len(names) else f"x{i}"
            ts = _flat_tensors(a)
            out += [base] if len(ts) == 1 else [f"{base}{j}" for j in range(len(ts))]
        return out

    def _build_dispatch(self, g: ModelGraph, args: tuple, kwargs: dict) -> None:
        kwargs = kwargs or {}
        d = _Dispatch(self, g)
        names = self._input_names(args)
        flat = [t for a in args for t in _flat_tensors(a)]
        for i, t in enumerate(flat):
            nm = names[i] if i < len(names) else f"x{i}"
            nid = f"@in:{nm}"
            g.add_node(Node(nid, "Input", "Input", "input", parent="", label=nm, out_shapes=[tuple(t.shape)]))
            d.mark(t, nid)
        for k, v in kwargs.items():
            for j, t in enumerate(_flat_tensors(v)):
                nid = f"@in:{k}" if j == 0 else f"@in:{k}{j}"
                g.add_node(Node(nid, "Input", "Input", "input", parent="", label=k, out_shapes=[tuple(t.shape)]))
                d.mark(t, nid)
        d.attach(self._graph_modules(g))
        try:
            with torch.no_grad(), d:
                out = self.model(*args, **kwargs)
        finally:
            d.detach()
        g.add_node(Node("@out", "Output", "Output", "output", parent="", label="output"))
        for src in d.sources(out):
            g.add_edge(src, "@out")
        self._op_sequence = list(d.op_ids)
        _prune_virtual(g)
        g.link_children()
        self._prune_unused_leaves(g)

    def _build_fx(self, g: ModelGraph) -> None:
        tracer = _ScopeTracer(self._leaf_types)
        fxg = tracer.trace(self.model)
        self._gm = fx.GraphModule(self.model, fxg)
        resolved: dict[str, list[str]] = {}  # fx name -> our producer ids
        meta: set[str] = set()
        calls: dict[str, int] = {}
        for n in fxg.nodes:
            data_in = []
            for a in _fx_flat_nodes((n.args, n.kwargs)):
                if a.name in meta:
                    continue
                data_in += resolved.get(a.name, [])
            data_in = list(dict.fromkeys(data_in))
            scope = n.meta.get("dv_scope", "")
            if scope not in g.nodes or not g.nodes[scope].is_container:
                scope = _nearest_container(g, scope)
            if n.op == "placeholder":
                nid = f"@in:{n.name}"
                g.add_node(Node(nid, "Input", "Input", "input", parent="", label=n.name))
                resolved[n.name] = [nid]
            elif n.op == "get_attr":
                target = str(n.target)
                if target in self._param_infos:
                    info = self._param_infos[target]
                    nid = f"@param:{target}"
                    g.add_node(Node(nid, "Parameter", "Parameter", "param",
                                    parent=_nearest_container(g, info.module), params={info.name: info},
                                    label=target.split(".")[-1], out_shapes=[info.shape]))
                    resolved[n.name] = [nid]
                else:
                    meta.add(n.name)
            elif n.op == "call_module":
                path = str(n.target)
                k = calls.get(path, 0)
                calls[path] = k + 1
                if k == 0:
                    nid = path
                else:
                    base = g.nodes[path]
                    nid = f"{path}@{k + 1}"
                    g.add_node(Node(nid, base.kind, base.type_name, "module", parent=base.parent,
                                    params=dict(base.params), roles=dict(base.roles),
                                    attrs=dict(base.attrs, shared=True), module=path,
                                    label=f"{base.display} (call {k + 1})"))
                self._module_calls.setdefault(path, []).append(nid)
                for s in data_in:
                    g.add_edge(s, nid)
                resolved[n.name] = [nid]
            elif n.op in ("call_function", "call_method"):
                tname = n.target if isinstance(n.target, str) else getattr(n.target, "__name__", str(n.target))
                if n.op == "call_method" and tname in _META_METHODS:
                    meta.add(n.name)
                    continue
                if n.op == "call_function" and n.target is getattr and len(n.args) > 1 and isinstance(
                        n.args[1], str) and n.args[1] in ("shape", "dtype", "device", "ndim"):
                    meta.add(n.name)
                    continue
                if not data_in:  # arithmetic on sizes / constants
                    meta.add(n.name)
                    continue
                if tname in _TRANSPARENT or (n.target is operator.getitem and isinstance(n.args[1], int)):
                    resolved[n.name] = data_in
                    continue
                kind = function_kind(tname)
                if n.target is operator.getitem:
                    kind = "getitem"
                nid = f"@{n.name}"
                attrs = {k: v for k, v in n.kwargs.items() if isinstance(v, (int, float, str, bool))}
                const = [a for a in n.args[1:] if isinstance(a, (int, float))]
                if const:
                    attrs["const"] = const
                g.add_node(Node(nid, kind, tname, "function", parent=scope, attrs=attrs, label=kind))
                for s in data_in:
                    g.add_edge(s, nid)
                resolved[n.name] = [nid]
                self._fx_nodes[n.name] = nid
            elif n.op == "output":
                g.add_node(Node("@out", "Output", "Output", "output", parent="", label="output"))
                for s in data_in:
                    g.add_edge(s, "@out")
        g.link_children()
        self._prune_unused_leaves(g)

    def _prune_unused_leaves(self, g: ModelGraph) -> None:
        """Leaf modules never called in forward (e.g. unused heads) stay in the tree but get flagged."""
        used = {e.src for e in g.edges} | {e.dst for e in g.edges}
        for n in g.nodes.values():
            if n.op == "module" and n.id not in used:
                n.attrs["unused"] = True

    def _build_from_hooks(self, g: ModelGraph, args: tuple, kwargs: dict) -> None:
        producer: dict[int, str] = {}
        keep: list[torch.Tensor] = []
        order: list[str] = []
        calls: dict[str, int] = {}
        g.add_node(Node("@in:x", "Input", "Input", "input", parent="", label="x"))
        for a in args:
            for t in _flat_tensors(a):
                producer[id(t)] = "@in:x"
                keep.append(t)
        handles = []

        def pre(path):
            def hook(mod, inp):
                k = calls.get(path, 0)
                calls[path] = k + 1
                nid = path if k == 0 else f"{path}@{k + 1}"
                if nid not in g.nodes:
                    base = g.nodes[path]
                    g.add_node(Node(nid, base.kind, base.type_name, "module", parent=base.parent,
                                    params=dict(base.params), roles=dict(base.roles),
                                    attrs=dict(base.attrs, shared=True), module=path))
                self._module_calls.setdefault(path, []).append(nid)
                srcs = [producer[id(t)] for t in _flat_tensors(inp) if id(t) in producer]
                if not srcs:
                    g.add_edge(order[-1] if order else "@in:x", nid, implicit=True)
                for s in srcs:
                    g.add_edge(s, nid)
                order.append(nid)
            return hook

        def post(path):
            def hook(mod, inp, out):
                nid = self._module_calls[path][-1]
                for t in _flat_tensors(out):
                    producer[id(t)] = nid
                    keep.append(t)
            return hook

        for path, n in list(g.nodes.items()):
            if n.op == "module":
                m = self._modules[path]
                handles.append(m.register_forward_pre_hook(pre(path)))
                handles.append(m.register_forward_hook(post(path)))
        try:
            with torch.no_grad():
                out = self.model(*args, **(kwargs or {}))
        finally:
            for h in handles:
                h.remove()
        g.add_node(Node("@out", "Output", "Output", "output", parent="", label="output"))
        srcs = [producer[id(t)] for t in _flat_tensors(out) if id(t) in producer]
        if srcs:
            for s in srcs:
                g.add_edge(s, "@out")
        elif order:
            g.add_edge(order[-1], "@out", implicit=True)
        g.link_children()
        self._prune_unused_leaves(g)

    def _build_sequential(self, g: ModelGraph) -> None:
        g.add_node(Node("@in:x", "Input", "Input", "input", parent="", label="x"))
        prev = "@in:x"
        for n in list(g.nodes.values()):
            if n.op == "module":
                g.add_edge(prev, n.id, implicit=True)
                self._module_calls[n.id] = [n.id]
                prev = n.id
        g.add_node(Node("@out", "Output", "Output", "output", parent="", label="output"))
        g.add_edge(prev, "@out", implicit=True)
        g.link_children()

    def _fill_shapes(self, g: ModelGraph, args: tuple, kwargs) -> None:
        res = self.run(args, kwargs, record="all", record_inputs=True)
        for nid, v in res.acts.items():
            if nid in g.nodes:
                g.nodes[nid].out_shapes = list(iter_shapes(v))[:1]
        for nid, v in res.inputs.items():
            if nid in g.nodes:
                g.nodes[nid].in_shapes = list(iter_shapes(v))[:1]
        for inp in g.inputs():
            if inp.id.startswith("@in:"):
                shapes = [s for a in args for s in iter_shapes(a)]
                ids = [n.id for n in g.inputs()]
                i = ids.index(inp.id)
                if i < len(shapes):
                    inp.out_shapes = [shapes[i]]
        out_shapes = list(iter_shapes(res.output))
        for o in g.outputs():
            o.in_shapes = out_shapes[:1]
        for e in g.edges:
            src = g.nodes.get(e.src)
            if src is not None and src.out_shapes:
                e.shape = tuple(src.out_shapes[0])

    # --------------------------------------------------------- execution
    def feature_axis(self, node: Node) -> int:
        if node.kind in CONV_KINDS or node.kind in ("BatchNorm", "MaxPool2d", "AvgPool2d", "AdaptiveAvgPool2d",
                                                     "MaxPool1d", "AvgPool1d", "GroupNorm"):
            return 1
        # activations following a conv keep the channel axis at 1
        if node.out_shapes and len(node.out_shapes[0]) == 4:
            return 1
        return -1

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def _node_for_call(self, path: str, counter: dict[str, int]) -> str:
        k = counter.get(path, 0)
        counter[path] = k + 1
        ids = self._module_calls.get(path) or [path]
        return ids[k] if k < len(ids) else ids[-1]

    def _on_registrations_changed(self) -> None:
        for h in self._native_handles:
            h.remove()
        self._native_handles = []
        if self._fn_replay is not None:
            self._fn_replay.detach()
            self._fn_replay = None
        g = self.graph() if self._graph is None else self._graph
        by_module: dict[str, list] = {}
        fn_int, fn_taps = self._function_registrations(g)
        for nid, fn in self.taps.values():
            if nid not in fn_taps:
                by_module.setdefault(self._module_of(g, nid), []).append(("tap", nid, fn))
        for nid, fn in self.interventions.values():
            if nid not in fn_int:
                by_module.setdefault(self._module_of(g, nid), []).append(("int", nid, fn))
        for path, items in by_module.items():
            module = self.model.get_submodule(path) if path else self.model
            self._native_handles.append(module.register_forward_hook(self._persistent_hook(path, items)))
        if fn_int or fn_taps:
            self._install_function_hooks(g, fn_int, fn_taps)

    def _function_registrations(self, g: ModelGraph) -> tuple[dict, dict]:
        """Persistent interventions / taps that target functional-op nodes, by node id."""
        fn_int: dict[str, list] = {}
        fn_taps: dict[str, list] = {}
        for store, out in ((self.interventions, fn_int), (self.taps, fn_taps)):
            for nid, fn in store.values():
                n = g.nodes.get(nid)
                if n is not None and n.op == "function":
                    if not self.function_taps:
                        raise ValueError(f"{nid!r} is a functional op; this model was traced with {self.trace_mode!r}, "
                                         "so persistent hooks attach to modules only. Use net.trace(x, interventions=...)")
                    out.setdefault(nid, []).append(fn)
        return fn_int, fn_taps

    def _function_op_hook(self, g: ModelGraph, fn_int: dict, fn_taps: dict):
        def op_hook(nid, out, a, kw):
            node = g.nodes[nid]
            kids = [k for k in (node.attrs.get("outputs") or []) if k is not None]
            if nid not in fn_int and nid not in fn_taps and not any(k in fn_int or k in fn_taps for k in kids):
                return out
            primary = first_array(out)
            if primary is None:
                return out
            fns = fn_int.get(nid, [])
            for fn in fns:
                primary = fn(primary, InterventionContext(node, self.ops, self.feature_axis(node)))
            if nid in fn_taps:
                ins = _flat_tensors((a, kw))
                for fn in fn_taps[nid]:
                    fn(nid, primary, ins[0] if ins else None)
            out = replace_first_array(out, primary) if fns else out
            for i, kid in enumerate(node.attrs.get("outputs") or []):
                if kid is None or not isinstance(out, (tuple, list)) or i >= len(out):
                    continue
                t = out[i]
                for fn in fn_int.get(kid, []):
                    t = fn(t, InterventionContext(g.nodes[kid], self.ops, self.feature_axis(g.nodes[kid])))
                if fn_int.get(kid):
                    out = _replace_output(out, i, t)
                for fn in fn_taps.get(kid, []):
                    fn(kid, t, None)
            return out
        return op_hook

    def _install_function_hooks(self, g: ModelGraph, fn_int: dict, fn_taps: dict) -> None:
        """Your own ``model(x)`` calls run under the replay tracer, so functional ops can be tapped / patched."""
        replay = _Dispatch(self, None, self._function_op_hook(g, fn_int, fn_taps))
        replay.enabled = False
        replay.attach(self._graph_modules(g))  # scope tracking (ops inside leaf modules aren't counted)
        self._fn_replay = replay
        active: list = []

        def pre(mod, args, kwargs):
            if self._running or active:  # deepvis's own run() merges these itself; no re-entry
                return None
            replay.reset()
            replay.enabled = True
            for t in _flat_tensors((args, kwargs)):
                replay.mark(t, "@in")
            fp = _no_fastpath()
            fp.__enter__()
            replay.__enter__()
            active.append(fp)
            return None

        def post(mod, args, kwargs, out):
            if active:
                fp = active.pop()
                replay.__exit__(None, None, None)
                fp.__exit__(None, None, None)
                replay.enabled = False
                replay.reset()
            return None

        self._native_handles.append(self.model.register_forward_pre_hook(pre, with_kwargs=True))
        self._native_handles.append(self.model.register_forward_hook(post, with_kwargs=True, always_call=True))

    def _module_of(self, g: ModelGraph, nid: str) -> str:
        n = g.nodes.get(nid)
        if n is None:
            raise KeyError(f"unknown node {nid!r}")
        if n.op == "function":
            raise ValueError(
                f"{nid!r} is a functional op; persistent hooks attach to modules. "
                "Use net.trace(...) / interventions within trace for functional ops.")
        if n.op not in ("module", "container"):
            raise ValueError(f"cannot hook node {nid!r} of type {n.op}")
        return n.module or n.id

    def _persistent_hook(self, path: str, items):
        g = self._graph

        def hook(mod, inp, out):
            primary = first_array(out)
            if primary is None:
                return None
            changed = False
            for typ, nid, fn in items:
                if typ == "int":
                    node = g.nodes[nid]
                    ctx = InterventionContext(node, self.ops, self.feature_axis(node))
                    primary = fn(primary, ctx)
                    changed = True
            for typ, nid, fn in items:
                if typ == "tap":
                    fn(nid, primary, first_array(inp))
            return replace_first_array(out, primary) if changed else None

        return hook

    def run(self, args, kwargs=None, record="all", record_inputs=False, interventions=None, grads=False,
            loss_fn=None, capture_attention=False) -> RunResult:
        kwargs = kwargs or {}
        g = self._graph if self._graph is not None else self.build_graph()
        interventions = interventions or {}
        wanted = _wanted_ids(g, record)
        res = RunResult(output=None)
        handles = []
        counter: dict[str, int] = {}
        grad_store: dict[str, torch.Tensor] = {}
        mha_calls: dict[int, tuple] = {}

        module_nodes = {n.module or n.id for n in g.nodes.values() if n.op == "module"}
        for path in module_nodes:
            module = self.model.get_submodule(path) if path else self.model

            def post(mod, inp, out, _path=path):
                nid = self._node_for_call(_path, counter)
                primary = first_array(out)
                if primary is None:
                    return None
                node = g.nodes.get(nid)
                changed = False
                for fn in interventions.get(nid, []):
                    primary = fn(primary, InterventionContext(node, self.ops, self.feature_axis(node)))
                    changed = True
                if nid in wanted:
                    res.acts[nid] = _snap(primary)
                    if record_inputs:
                        res.inputs[nid] = _snap(first_array(inp))
                    if grads and isinstance(primary, torch.Tensor) and primary.requires_grad:
                        primary.register_hook(lambda gr, _n=nid: grad_store.__setitem__(_n, gr.detach()))
                if capture_attention and isinstance(mod, nn.MultiheadAttention) and isinstance(out, tuple):
                    if len(out) > 1 and out[1] is not None:
                        res.attention[nid] = to_numpy(out[1])
                    call = mha_calls.pop(id(mod), None)
                    if call is not None and nid in wanted:
                        parts = _mha_parts(mod, *call)
                        if parts is not None:
                            res.attention_parts[nid] = parts
                return replace_first_array(out, primary) if changed else None

            handles.append(module.register_forward_hook(post))
            if capture_attention and isinstance(module, nn.MultiheadAttention):
                def pre(mod, a, kw):
                    mha_calls[id(mod)] = (a, dict(kw))  # for the inside view (Q, K, V, scores, …)
                    kw = dict(kw)
                    kw["need_weights"] = True
                    kw["average_attn_weights"] = False
                    return a, kw
                handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))

        fn_int, fn_taps = self._function_registrations(g) if (self.taps or self.interventions) else ({}, {})
        if fn_int:
            interventions = {k: list(v) for k, v in interventions.items()}
            for nid, fns in fn_int.items():
                interventions.setdefault(nid, []).extend(fns)
        op_ids = {nid for nid in g.nodes if g.nodes[nid].op == "function"}
        need_ops = bool(op_ids & (wanted | set(interventions) | set(fn_taps)))
        need_fx = self._gm is not None and need_ops
        replay = None
        if need_ops and not need_fx and self.trace_mode == "dispatch":
            def op_hook(nid, out, a, kw):
                primary = first_array(out)
                if primary is None:
                    return out
                node = g.nodes.get(nid)
                fns = interventions.get(nid, [])
                for fn in fns:
                    primary = fn(primary, InterventionContext(node, self.ops, self.feature_axis(node)))
                if nid in wanted:
                    res.acts[nid] = _snap(primary)
                    if record_inputs:
                        ins = _flat_tensors((a, kw))
                        res.inputs[nid] = _snap(ins[0]) if ins else None
                    if grads and isinstance(primary, torch.Tensor) and primary.requires_grad:
                        primary.register_hook(lambda gr, _n=nid: grad_store.__setitem__(_n, gr.detach()))
                    if capture_attention and node is not None and node.type_name == "scaled_dot_product_attention":
                        att = _sdpa_weights(a, kw)
                        if att is not None:
                            res.attention[nid] = att
                        parts = _sdpa_parts(a, kw)
                        if parts is not None:
                            res.attention_parts[nid] = parts
                for fn in fn_taps.get(nid, []):
                    ins = _flat_tensors((a, kw))
                    fn(nid, primary, ins[0] if ins else None)
                out = replace_first_array(out, primary) if fns else out
                for i, kid in enumerate((node.attrs.get("outputs") or []) if node is not None else []):
                    if kid is None or not isinstance(out, (tuple, list)) or i >= len(out):
                        continue
                    t = out[i]
                    kfns = interventions.get(kid, [])
                    for fn in kfns:
                        t = fn(t, InterventionContext(g.nodes[kid], self.ops, self.feature_axis(g.nodes[kid])))
                    if kfns:
                        out = _replace_output(out, i, t)
                    if kid in wanted:
                        res.acts[kid] = _snap(t)
                        if grads and isinstance(t, torch.Tensor) and t.requires_grad:
                            t.register_hook(lambda gr, _n=kid: grad_store.__setitem__(_n, gr.detach()))
                    for fn in fn_taps.get(kid, []):
                        fn(kid, t, None)
                return out
            replay = _Dispatch(self, None, op_hook)
        saved_grads = {}
        in_leaves: list = []
        if grads:  # float inputs become leaves so ∂L/∂x (saliency) is recorded too
            def _leaf(a):
                if isinstance(a, torch.Tensor) and a.is_floating_point():
                    t = a.detach().requires_grad_(True)
                    in_leaves.append(t)
                    return t
                return a
            args = tuple(_leaf(a) for a in args)
            for p, t in self._tensors.items():
                if t.requires_grad:
                    saved_grads[p] = t.grad
                    t.grad = None
        self._running = True
        try:
            with _no_fastpath(), self._untouched(), torch.set_grad_enabled(True if grads else torch.is_grad_enabled()):
                if need_fx:
                    interp = _RecordingInterpreter(self, g, wanted, interventions, res)
                    interp.grad_store = grad_store if grads else None
                    out = interp.run(*args, **kwargs)
                elif replay is not None:
                    for t in [t for a in args for t in _flat_tensors(a)] + _flat_tensors(kwargs):
                        replay.mark(t, "@in")
                    replay.attach(self._graph_modules(g))
                    try:
                        with replay:
                            out = self.model(*args, **kwargs)
                    finally:
                        replay.detach()
                else:
                    out = self.model(*args, **kwargs)
                res.output = out
                if grads:
                    primary = first_array(out)
                    loss = loss_fn(out) if loss_fn is not None else primary.float().sum()
                    res.loss = float(loss.detach())
                    loss.backward()
                    res.grads = {k: to_numpy(v) for k, v in grad_store.items()}
                    in_ids = [n.id for n in g.inputs()]
                    for i, t in enumerate(in_leaves):
                        if i < len(in_ids) and t.grad is not None:
                            res.grads[in_ids[i]] = to_numpy(t.grad)
                    res.param_grads = {p: to_numpy(t.grad) for p, t in self._tensors.items()
                                       if t.requires_grad and t.grad is not None}
        finally:
            self._running = False
            for h in handles:
                h.remove()
            for p, gsaved in saved_grads.items():
                self._tensors[p].grad = gsaved
        if "@out" in wanted:
            res.acts["@out"] = first_array(res.output)
        for inp in g.inputs():
            if inp.id in wanted:
                ids = [n.id for n in g.inputs()]
                flat = [t for a in args for t in _flat_tensors(a)]
                i = ids.index(inp.id)
                if i < len(flat):
                    res.acts[inp.id] = flat[i]
        return res

    # --------------------------------------------------------------- misc
    def export(self) -> nn.Module:
        return self.model

    @contextlib.contextmanager
    def inspecting(self):
        modes = [(m, m.training) for m in self.model.modules()]
        self.model.eval()  # no BatchNorm running-stat updates, no dropout RNG draws
        try:
            with torch.no_grad():
                yield
        finally:
            for m, t in modes:
                m.training = t

    def load(self, native) -> None:
        sd = native.state_dict() if isinstance(native, nn.Module) else dict(native)
        own = self.model.state_dict(keep_vars=True)
        unknown = [p for p in sd if p not in own]
        if unknown:
            raise KeyError(f"unknown parameters: {unknown[:5]}")
        with torch.no_grad():
            for p, v in sd.items():
                t = self._tensors.get(p, own[p])  # untracked buffers (e.g. num_batches_tracked) too
                if v is not t:
                    t.copy_(torch.as_tensor(v).to(dtype=t.dtype, device=t.device))

    def grads(self, native=None) -> dict[str, np.ndarray]:
        if native is not None:
            src = native.named_parameters() if isinstance(native, nn.Module) else dict(native).items()
            return {p: to_numpy(g.grad if isinstance(g, nn.Parameter) else g) for p, g in src
                    if (g.grad if isinstance(g, nn.Parameter) else g) is not None}
        return {p: to_numpy(t.grad) for p, t in self._tensors.items() if getattr(t, "grad", None) is not None}

    def clone(self) -> "TorchBackend":
        import copy

        return TorchBackend(copy.deepcopy(self.model), self.name, self._leaf_types[len(LEAF_TYPES) + len(_library_leaves()):],
                            self._sample)

    def to_native(self, x):
        if isinstance(x, torch.Tensor):
            return x
        p = next(iter(self._tensors.values()), None)
        dev = p.device if p is not None else "cpu"
        return torch.as_tensor(np.asarray(x), device=dev,
                               dtype=torch.float32 if np.asarray(x).dtype.kind == "f" else None)


_DISPATCH_TRANSPARENT = _TRANSPARENT | {"expand_as", "__get__", "type", "cuda", "requires_grad_"}


def _func_name(func) -> str:
    if getattr(func, "_opname", None):  # an aten op (torch.export): aten.add_.Tensor → add
        return func._opname.rstrip("_") or func._opname
    name = getattr(func, "__name__", None) or str(func)
    if name == "__get__":  # tensor property (x.T, x.mT, ...)
        name = getattr(getattr(func, "__self__", None), "__name__", "attr")
    return name


class _Dispatch(TorchFunctionMode):
    """Eager tracer: module hooks track scope, a TorchFunctionMode sees every functional op.

    Used twice with identical op numbering: to build the graph (``graph`` given) and to
    replay a forward pass that records / intervenes on functional-op nodes (``op_hook``).
    """

    def __init__(self, backend: "TorchBackend", graph: ModelGraph | None = None, op_hook=None):
        super().__init__()
        self.b, self.g, self.op_hook = backend, graph, op_hook
        self.src: dict[int, str] = {}
        self.keep: list = []
        self.stack: list[str] = [""]
        self.leaf_depth = 0
        self.calls: dict[str, int] = {}
        self.cur_leaf: list[str] = []
        self.n_ops = 0
        self.op_ids: list[str] = []
        self.counters: dict[str, int] = {}
        self.param_ids = {id(t): p for p, t in backend._tensors.items()}
        self.handles: list = []
        self.enabled = True  # a persistent replay switches its module hooks off between forward passes

    def mark(self, t, nid: str) -> None:
        self.src[id(t)] = nid
        self.keep.append(t)

    def reset(self) -> None:
        """Fresh numbering for another forward pass (persistent replays are reused)."""
        self.src, self.keep, self.stack = {}, [], [""]
        self.leaf_depth, self.n_ops, self.cur_leaf, self.calls = 0, 0, [], {}

    def attach(self, modules: dict[str, bool]) -> None:
        for path, leaf in modules.items():
            m = self.b.model.get_submodule(path)
            self.handles.append(m.register_forward_pre_hook(self._pre(path, leaf), with_kwargs=True))
            self.handles.append(m.register_forward_hook(self._post(path, leaf)))

    def detach(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []
        self.keep = []

    def _pre(self, path: str, leaf: bool):
        def hook(mod, args, kwargs):
            if not self.enabled:
                return None
            if leaf:
                if self.leaf_depth == 0:
                    k = self.calls.get(path, 0)
                    self.calls[path] = k + 1
                    nid = path if k == 0 else f"{path}@{k + 1}"
                    if self.g is not None:
                        self.b._add_call_node(self.g, path, nid, k)
                        srcs = self.sources((args, kwargs))
                        for s in srcs:
                            self.g.add_edge(s, nid)
                        if not srcs:
                            self._constant_input(nid, (args, kwargs))
                    self.cur_leaf.append(nid)
                self.leaf_depth += 1
            elif self.leaf_depth == 0:
                self.stack.append(path)
            return None
        return hook

    def _post(self, path: str, leaf: bool):
        def hook(mod, args, out):
            if not self.enabled:
                return None
            if leaf:
                self.leaf_depth -= 1
                if self.leaf_depth == 0 and self.cur_leaf:
                    nid = self.cur_leaf.pop()
                    for t in _flat_tensors(out):
                        self.mark(t, nid)
            elif self.leaf_depth == 0 and self.stack and self.stack[-1] == path:
                self.stack.pop()
            return None
        return hook

    def _constant_input(self, nid: str, obj) -> None:
        """A layer fed only by tensors made inside forward (e.g. ``position_ids = arange(T)``) gets a
        ``Constant`` producer, so equations read ``Emb(arange)`` rather than an unnamed input."""
        ts = [t for t in _flat_tensors(obj) if id(t) not in self.param_ids]
        if not ts:
            return
        t = ts[0]
        label = "constant"
        with torch.no_grad():
            flat = t.detach().flatten()
            if not t.is_floating_point() and flat.numel() > 1 and torch.equal(
                    flat - flat[0], torch.arange(flat.numel(), device=flat.device, dtype=flat.dtype)):
                label = "arange"
        cid = f"@const:{nid}"
        self.g.add_node(Node(cid, "Constant", "Constant", "function", parent=self.g.nodes[nid].parent,
                             attrs={"virtual": True, "value": label}, label=label, out_shapes=[tuple(t.shape)]))
        self.g.add_edge(cid, nid)

    def sources(self, obj) -> list[str]:
        out: list[str] = []
        for t in _flat_tensors(obj):
            s = self.src.get(id(t))
            if s is None and id(t) in self.param_ids:
                s = self._param_node(self.param_ids[id(t)])
            if s is not None and s not in out:
                out.append(s)
        return out

    def _param_node(self, path: str) -> str:
        nid = f"@param:{path}"
        if self.g is not None and nid not in self.g.nodes:
            info = self.b._param_infos[path]
            self.g.add_node(Node(nid, "Parameter", "Parameter", "param", parent=_nearest_container(self.g, info.module),
                                 params={info.name: info}, label=path.split(".")[-1], out_shapes=[info.shape]))
        return nid

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        if self.leaf_depth > 0:
            return out
        outs = _flat_tensors(out)
        if not outs:
            return out
        srcs = self.sources((args, kwargs))
        if not srcs:
            return out  # constants (masks, aranges ...)
        name = _func_name(func)
        if name in _DISPATCH_TRANSPARENT:
            for t in outs:
                self.mark(t, srcs[0])
            return out
        if name in ("cat", "concat", "concatenate"):  # cat(empty, x) (e.g. a fresh KV cache) is just x
            full = [t for t in _flat_tensors(args[:1]) if t.numel() > 0]
            if len(full) == 1 and tuple(full[0].shape) == tuple(outs[0].shape) and id(full[0]) in self.src:
                for t in outs:
                    self.mark(t, self.src[id(full[0])])
                return out
        idx = self.n_ops
        self.n_ops += 1
        if self.g is not None:
            kind = function_kind(name)
            if name in ("__getitem__", "getitem"):
                kind = "getitem"
            base = "@" + kind.lower()
            k = self.counters.get(base, 0)
            self.counters[base] = k + 1
            nid = base if k == 0 else f"{base}_{k}"
            attrs = {kk: v for kk, v in kwargs.items() if isinstance(v, (int, float, str, bool))}
            const = [a for a in args if isinstance(a, (int, float)) and not isinstance(a, bool)]
            if const:
                attrs["const"] = const
            parent = self.stack[-1] if self.stack else ""
            self.g.add_node(Node(nid, kind, name, "function", parent=_nearest_container(self.g, parent),
                                 attrs=attrs, label=kind, out_shapes=[tuple(outs[0].shape)]))
            for s in srcs:
                self.g.add_edge(s, nid)
            self.op_ids.append(nid)
            if len(outs) > 1 and isinstance(out, (tuple, list)):  # split / chunk / unbind / max(dim) ...
                kids = []
                for i, t in enumerate(outs):
                    kid = f"{nid}[{i}]"
                    self.g.add_node(Node(kid, "getitem", "getitem", "function", parent=self.g.nodes[nid].parent,
                                         attrs={"index": i, "virtual": True}, label=f"[{i}]",
                                         out_shapes=[tuple(t.shape)]))
                    self.g.add_edge(nid, kid)
                    self.mark(t, kid)
                    kids.append(kid)
                self.g.nodes[nid].attrs["outputs"] = kids
                return out
        else:
            seq = self.b._op_sequence
            nid = seq[idx] if idx < len(seq) else None
            gb = self.b._graph
            if nid is not None and gb is not None and nid in gb.nodes and gb.nodes[nid].type_name != name:
                nid = None  # control flow differs from the traced pass: never patch the wrong op
            if self.op_hook is not None and nid is not None:
                new = self.op_hook(nid, out, args, kwargs)
                if new is not out:
                    out = new
                    outs = _flat_tensors(out)
        for t in outs:
            self.mark(t, nid or "?")
        return out


class _RecordingInterpreter(fx.Interpreter):
    """Runs the fx graph so functional ops (residual adds, softmax ...) can be recorded/patched."""

    def __init__(self, backend: TorchBackend, g: ModelGraph, wanted, interventions, res: RunResult):
        super().__init__(backend._gm)
        self.b, self.g, self.wanted, self.interventions, self.res = backend, g, wanted, interventions, res
        self.grad_store: dict | None = None

    def run_node(self, n):
        out = super().run_node(n)
        nid = self.b._fx_nodes.get(n.name)
        if nid is None:
            return out
        primary = first_array(out)
        if primary is None:
            return out
        node = self.g.nodes[nid]
        fns = self.interventions.get(nid, [])
        for fn in fns:
            primary = fn(primary, InterventionContext(node, self.b.ops, self.b.feature_axis(node)))
        if nid in self.wanted:
            self.res.acts[nid] = _snap(primary)
            if n.args and isinstance(n.args[0], fx.Node) and n.args[0] in self.env:
                self.res.inputs[nid] = _snap(first_array(self.env[n.args[0]]))
            if self.grad_store is not None and isinstance(primary, torch.Tensor) and primary.requires_grad \
                    and nid not in self.grad_store:
                store = self.grad_store
                primary.register_hook(lambda gr, _n=nid: store.__setitem__(_n, gr.detach()))
        return replace_first_array(out, primary) if fns else out


def _wanted_ids(g: ModelGraph, record) -> set[str]:
    if record is None or record is False:
        return set()
    if record == "all" or record is True:
        return {n.id for n in g.nodes.values() if n.is_compute}
    if isinstance(record, str):
        record = [record]
    return set(record)


def _flat_tensors(obj) -> list[torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return [t for o in obj for t in _flat_tensors(o)]
    if isinstance(obj, dict):
        return [t for o in obj.values() for t in _flat_tensors(o)]
    return []


def _prefixes(path: str) -> list[str]:
    parts = path.split(".")
    return [".".join(parts[:i]) for i in range(0, len(parts))]


def _nearest_container(g: ModelGraph, path: str) -> str:
    cur = path
    while cur:
        if cur in g.nodes and g.nodes[cur].is_container:
            return cur
        cur = cur.rpartition(".")[0]
    return ""
