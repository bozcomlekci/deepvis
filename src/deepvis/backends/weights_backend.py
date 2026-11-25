"""Weights without code: a state_dict, ``.safetensors`` or ``.npz`` file, drawn from its tensor names.

* each module path (``layer1.0.conv1`` from ``layer1.0.conv1.weight``) is one layer, its kind guessed from the
  tensors it holds (a 4-D ``weight`` is a 2-D convolution, ``running_mean`` means BatchNorm, ``in_proj_weight``
  attention …); dotted prefixes are the containers;
* there is no computation to trace, so layers are linked in the file's order — drawn as such, not as dataflow;
* tensors load lazily (a 7B checkpoint opens in seconds) and can be edited and saved back.
"""

from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np

from ..core.graph import ModelGraph, Node, ParamInfo
from ..core.kinds import infer_roles
from .base import Backend, RunResult


# where a sibling usually sits in the computation, for files that store names alphabetically (.safetensors):
# embeddings first, then per block norm → attention (q, k, v, o) → norm → MLP (gate, up, down), final norm, head
_RANK = {
    **dict.fromkeys(("embed", "embeddings", "embed_tokens", "wte", "wpe", "tok_embeddings", "word_embeddings",
                     "position_embeddings", "token_type_embeddings", "patch_embed", "patch_embeddings", "stem",
                     "conv1", "cls_token", "pos_embed"), 0),
    **dict.fromkeys(("bn1", "input_layernorm", "ln_1", "norm1", "layernorm_before", "attention_norm"), 1),
    **dict.fromkeys(("self_attn", "attn", "attention", "mixer", "self"), 2),
    **dict.fromkeys(("post_attention_layernorm", "ln_2", "norm2", "layernorm_after", "ffn_norm"), 3),
    **dict.fromkeys(("mlp", "ffn", "feed_forward", "intermediate", "block_sparse_moe"), 4),
    **dict.fromkeys(("output",), 4.5),
    **dict.fromkeys(("norm", "ln_f", "final_layernorm", "final_layer_norm", "layernorm", "norm_f"), 8),
    **dict.fromkeys(("lm_head", "head", "classifier", "fc", "score", "pooler", "heads"), 9),
    # inside attention / MLP
    **dict.fromkeys(("q_proj", "query", "q", "wq", "c_attn", "qkv", "in_proj", "w1", "gate_proj", "fc1", "c_fc",
                     "lin1"), 0),
    **dict.fromkeys(("k_proj", "key", "k", "wk", "up_proj", "w3"), 1),
    **dict.fromkeys(("v_proj", "value", "v", "wv", "down_proj", "w2", "fc2", "lin2"), 2),
    **dict.fromkeys(("o_proj", "out_proj", "wo", "dense", "proj"), 3),
    **dict.fromkeys(("downsample", "shortcut", "skip"), 6),
}
_BASE = {"conv": 0, "bn": 1, "norm": 1, "gn": 1, "ln": 1, "act": 2, "relu": 2}  # conv2 before bn2 before act2


def _order_key(path: str) -> tuple:
    key = []
    for c in path.split("."):
        if c.isdigit():
            key.append((5.0, int(c), 5, ""))
        else:
            head = c.rstrip("0123456789")
            num = int(c[len(head):]) if len(head) < len(c) else 0
            key.append((float(_RANK.get(c, 5)), num, _BASE.get(head, 5), c))  # conv1 bn1 conv2 bn2 …
    return tuple(key)


def _computation_order(names: list[str]) -> list[str]:
    """``names`` as stored when that looks deliberate (a state_dict keeps registration order); alphabetical files
    (.safetensors sorts its keys) are re-ordered by where each module usually sits in the computation."""
    if names != sorted(names):
        return names
    modules = list(dict.fromkeys(n.rpartition(".")[0] for n in names))
    rank = {m: k for k, m in enumerate(sorted(modules, key=_order_key))}
    return sorted(names, key=lambda n: rank[n.rpartition(".")[0]])


def _guess_kind(path: str, shapes: dict[str, tuple]) -> str:
    low = path.lower()
    w = shapes.get("weight")
    if "in_proj_weight" in shapes:
        return "MultiheadAttention"
    ih = next((k for k in shapes if k.startswith("weight_ih")), None)
    hh = next((k for k in shapes if k.startswith("weight_hh")), None)
    if ih and hh:
        ratio = shapes[ih][0] // max(shapes[hh][1], 1)
        return {4: "LSTM", 3: "GRU"}.get(ratio, "RNN")
    if {"running_mean", "running_var"} <= set(shapes):
        return "BatchNorm"
    if w is None:
        return "Parameters"
    if len(w) in (3, 4, 5):
        return f"Conv{len(w) - 2}d"
    if len(w) == 2:
        return "Embedding" if any(t in low for t in ("emb", "wte", "wpe", "tok", "lookup")) else "Linear"
    if len(w) == 1 and any(t in low for t in ("norm", "ln", "layernorm")):
        return "LayerNorm" if "bias" in shapes and "rms" not in low else "RMSNorm"
    return "Parameters"


class WeightsBackend(Backend):
    framework = "weights"
    function_taps = False

    def __init__(self, tensors: dict[str, Any], name: str = "weights", loaders: dict[str, Callable] | None = None,
                 shapes: dict[str, tuple] | None = None, dtypes: dict[str, str] | None = None,
                 source: str | None = None):
        """``tensors``: name → array (anything ``np.asarray`` takes), or pass ``loaders`` (name → callable giving
        the array) with ``shapes`` / ``dtypes`` for lazy loading."""
        super().__init__(name)
        self.trace_mode = "weights"
        self.source = source
        self._arrays: dict[str, np.ndarray] = {k: _to_numpy(v) for k, v in tensors.items()}
        self._loaders = dict(loaders or {})
        self._shapes = {k: tuple(a.shape) for k, a in self._arrays.items()}
        self._dtypes = {k: str(a.dtype) for k, a in self._arrays.items()}
        self._shapes.update({k: tuple(int(d) for d in v) for k, v in (shapes or {}).items()})
        self._dtypes.update(dtypes or {})
        self._order = _computation_order(list(tensors) + [k for k in self._loaders if k not in tensors])
        self._infos = {}
        for p in self._order:
            module, _, local = p.rpartition(".")
            dtype = self._dtypes.get(p, "float32")
            self._infos[p] = ParamInfo(p, local, module, self._shapes[p], dtype,
                                       dtype.startswith(("float", "bfloat", "half")))
        self.model = self  # the "native" object: this backend holds the tensors itself

    # ------------------------------------------------------------------ params
    def param_paths(self) -> list[str]:
        return list(self._order)

    def param_info(self, path: str) -> ParamInfo:
        return self._infos[path]

    def get_param(self, path: str) -> np.ndarray:
        if path not in self._arrays:
            self._arrays[path] = _to_numpy(self._loaders[path]())
        return np.array(self._arrays[path], copy=True)

    def set_param(self, path: str, value) -> None:
        v = np.asarray(value)
        if tuple(v.shape) != self._shapes[path]:
            raise ValueError(f"shape mismatch for {path}: {v.shape} vs {self._shapes[path]}")
        self._arrays[path] = v.copy()

    # ------------------------------------------------------------------ graph
    def build_graph(self, *inputs, **kwargs) -> ModelGraph:
        g = ModelGraph(self.name, "weights")
        g.add_node(Node("", "Weights", "Weights", "container", label=self.name))
        modules: dict[str, dict[str, ParamInfo]] = {}
        for p in self._order:
            info = self._infos[p]
            modules.setdefault(info.module, {})[info.name] = info
        layer_ids = set(modules)
        containers: set[str] = set()
        for path in modules:
            parts = path.split(".") if path else []
            for k in range(1, len(parts)):
                prefix = ".".join(parts[:k])
                if prefix not in layer_ids:
                    containers.add(prefix)
        for c in sorted(containers, key=lambda s: (s.count("."), s)):
            parent = c.rpartition(".")[0]
            while parent and parent not in containers:
                parent = parent.rpartition(".")[0]
            g.add_node(Node(c, "Block", "Block", "container", parent=parent, label=c.split(".")[-1]))
        prev = None
        for path, params in modules.items():
            nid = path or "@root"
            shapes = {k: v.shape for k, v in params.items()}
            kind = _guess_kind(path, shapes)
            parent = path.rpartition(".")[0]
            while parent and parent not in containers:
                parent = parent.rpartition(".")[0]
            attrs = {"untraced": True}
            w = shapes.get("weight")
            if kind == "Linear" and w:
                attrs.update(out_features=w[0], in_features=w[1])
            elif kind.startswith("Conv") and w:
                attrs.update(out_channels=w[0], in_channels=w[1], kernel_size=list(w[2:]))
            elif kind == "Embedding" and w:
                attrs.update(num_embeddings=w[0], embedding_dim=w[1])
            g.add_node(Node(nid, kind, kind, "module", parent=parent, params=params,
                            roles=infer_roles(kind, shapes, "torch"), attrs=attrs,
                            label=path.split(".")[-1] if path else "parameters"))
            if prev is not None:
                g.add_edge(prev, nid, implicit=True)  # the file's order, not traced dataflow
            prev = nid
        g.link_children()
        self._graph = g
        return g

    # ------------------------------------------------------------------ no computation
    def _no_code(self):
        from ..errors import OptionError

        return OptionError("a weights file has no computation: deepvis shows its tensors but can't run it.",
                           "build the model in code, load these weights into it, and wrap that model.")

    def forward(self, *args, **kwargs):
        raise self._no_code()

    def run(self, args, kwargs=None, record="all", record_inputs=False, interventions=None, grads=False,
            loss_fn=None, capture_attention=False) -> RunResult:
        raise self._no_code()

    def export(self) -> dict[str, np.ndarray]:
        """Every tensor (loading the lazy ones), with your edits."""
        return {p: self.get_param(p) for p in self._order}

    def save(self, path: str) -> str:
        """Write the (edited) tensors: ``.safetensors`` or ``.npz`` by extension."""
        tensors = self.export()
        if path.endswith(".safetensors"):
            from safetensors.numpy import save_file

            save_file({k: np.ascontiguousarray(v) for k, v in tensors.items()}, path)
        else:
            np.savez(path, **tensors)
        return path

    def to_native(self, x):
        return np.asarray(x)


def _to_numpy(v) -> np.ndarray:
    if isinstance(v, np.ndarray):
        return v
    if hasattr(v, "detach"):  # torch: bfloat16 has no NumPy dtype, so it is shown as float32
        t = v.detach().cpu()
        if str(t.dtype) in ("torch.bfloat16", "torch.float8_e4m3fn", "torch.float8_e5m2"):
            t = t.float()
        return t.numpy()
    return np.asarray(v)


def from_safetensors(path: str) -> WeightsBackend:
    """A ``.safetensors`` file, read lazily (shapes from its header; tensors when first looked at)."""
    from safetensors import safe_open

    use_torch = False
    try:
        import torch  # noqa: F401 - reads every dtype, bfloat16 included

        use_torch = True
    except ImportError:
        pass
    handle = safe_open(path, framework="pt" if use_torch else "numpy")
    shapes, dtypes, loaders = {}, {}, {}
    for k in handle.keys():
        sl = handle.get_slice(k)
        shapes[k] = tuple(sl.get_shape())
        dt = str(sl.get_dtype()).lower()
        dtypes[k] = {"f32": "float32", "f16": "float16", "bf16": "bfloat16", "f64": "float64", "i64": "int64",
                     "i32": "int32", "i8": "int8", "u8": "uint8", "bool": "bool"}.get(dt, dt)
        loaders[k] = (lambda key=k: handle.get_tensor(key))
    name = os.path.splitext(os.path.basename(path))[0]
    return WeightsBackend({}, name=name, loaders=loaders, shapes=shapes, dtypes=dtypes, source=path)
