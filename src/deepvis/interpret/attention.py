"""Across attention layers: where information flows (rollout) and which heads matter (ablation).

* ``rollout``          Π over layers of ½·I + ½·mean_heads(A) (Abnar & Zuidema, 2020): how much each output
                       position draws on each input token through the whole stack, residual paths included.
* ``head_importance``  switch one head off at a time (its output set to zero) and measure how much the model's
                       output moves: ‖y − y_ablated‖ / ‖y‖ on the shown sample.

Heads are switched off exactly: for ``nn.MultiheadAttention`` by zeroing the head's value projection and bias
(and its extra value token with ``add_bias_kv``), so its P·V is 0 and it adds nothing after the output projection;
for ``F.scaled_dot_product_attention`` by zeroing the head's slice of the result.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from ..monitor.recorder import Trace
    from ..net import Net


def attention_layers(net: "Net", trace: "Trace") -> list[str]:
    """Attention nodes recorded in ``trace``, in execution order."""
    parts = getattr(trace, "attention_parts", None) or {}
    return [n for n in net.graph.topo_order() if n in parts]


def rollout(net: "Net", trace: "Trace", sample: int = 0) -> dict | None:
    """{"matrix": (T, T), "layers": [...]}; None unless every layer is self-attention over the same tokens."""
    layers = attention_layers(net, trace)
    if not layers:
        return None
    R = None
    for nid in layers:
        p = np.asarray(trace.attention_parts[nid]["probs"], dtype=np.float64)
        p = p[min(sample, p.shape[0] - 1)]            # (heads, T, S)
        T, S = p.shape[-2:]
        if T != S or (R is not None and R.shape[0] != T):
            return None
        A = 0.5 * p.mean(0) + 0.5 * np.eye(T)
        A = A / A.sum(-1, keepdims=True)
        R = A if R is None else A @ R
    return {"matrix": R, "layers": layers}


def _mha(net: "Net", nid: str):
    """The ``nn.MultiheadAttention`` module behind attention node ``nid``, or None."""
    import torch

    node = net.graph.nodes.get(nid)
    if node is None:
        return None
    path = node.module if node.module is not None else (node.id if node.op == "module" else None)
    if path is None:
        return None
    try:
        mod = net.backend.model.get_submodule(path) if path else net.backend.model
    except AttributeError:
        return None
    return mod if isinstance(mod, torch.nn.MultiheadAttention) else None


def _value_slices(mod, h: int) -> list[tuple[str, str]]:
    """Where head ``h`` of an ``nn.MultiheadAttention`` gets its values: ``(parameter name, index)`` pairs. Zeroed,
    the head's P·V is exactly 0 — the value projection (packed, or separate when ``kdim``/``vdim`` differ), its bias,
    and the learned extra value token of ``add_bias_kv``."""
    E, d = mod.embed_dim, mod.embed_dim // mod.num_heads
    lo, hi = h * d, (h + 1) * d
    out = [("in_proj_weight", f"{2 * E + lo}:{2 * E + hi}") if mod.in_proj_weight is not None
           else ("v_proj_weight", f"{lo}:{hi}")]
    if mod.in_proj_bias is not None:
        out.append(("in_proj_bias", f"{2 * E + lo}:{2 * E + hi}"))
    if mod.bias_v is not None:
        out.append(("bias_v", f"..., {lo}:{hi}"))
    return out


def _head_switch(net: "Net", nid: str) -> tuple[int, Callable[[int], contextlib.AbstractContextManager]] | None:
    """(number of heads, head → a context in which that head is off) for one attention node, or None."""
    import torch

    from ..core.selectors import normalize_index

    node = net.graph.nodes.get(nid)
    if node is None:
        return None
    mod = _mha(net, nid)
    if mod is not None:

        @contextlib.contextmanager
        def off(h: int):
            where = [(getattr(mod, name), normalize_index(ix)) for name, ix in _value_slices(mod, h)]
            with torch.no_grad():
                saved = [t[ix].clone() for t, ix in where]
                for t, ix in where:
                    t[ix] = 0
            try:
                yield {}
            finally:
                with torch.no_grad():
                    for (t, ix), v in zip(where, saved):
                        t[ix] = v

        return mod.num_heads, off
    if node.op == "function":  # F.scaled_dot_product_attention: (batch, heads, tokens, dim) → zero a head
        from ..manipulate.interventions import ablate

        @contextlib.contextmanager
        def off_fn(h: int):
            yield {nid: [ablate([h], axis=1)]}

        shape = node.out_shapes[0] if node.out_shapes else None
        if shape is not None and len(shape) == 4:
            return int(shape[1]), off_fn
    return None


def _change(a, b) -> float:
    from ..core.tensor import first_array, to_numpy

    a = to_numpy(first_array(a)).astype(np.float64)
    b = to_numpy(first_array(b)).astype(np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) or 1.0))


def head_importance(net: "Net", args: tuple, sample: int = 0, max_heads: int = 256) -> dict | None:
    """{"layers": [node ids], "heads": H, "effect": [[per head] per layer]} or None (no switchable heads / too many)."""
    trace = net.trace(*args, capture_attention=True)
    layers = attention_layers(net, trace) or [n.id for n in net.graph.compute_nodes()
                                              if n.kind in ("MultiheadAttention", "scaled_dot_product_attention")]
    switches = [(nid, s) for nid in layers if (s := _head_switch(net, nid)) is not None]
    if not switches or sum(s[0] for _, s in switches) > max_heads:
        return None
    one = tuple(a[sample:sample + 1] if hasattr(a, "shape") and len(a.shape) and a.shape[0] > sample else a
                for a in args)
    base = net.trace(*one, record=[], capture_attention=False).output
    effect = []
    for nid, (H, off) in switches:
        row = []
        for h in range(H):
            with off(h) as ints:
                out = net.trace(*one, record=[], capture_attention=False, interventions=ints or None).output
            row.append(_change(base, out))
        effect.append(row)
    return {"layers": [nid for nid, _ in switches], "heads": max(s[0] for _, s in switches), "effect": effect}


def head_ablation(net: "Net", nid: str, h: int):
    """How to switch head ``h`` of ``nid`` off until undone: ``("edits", [Edit, …])`` — zeroing its value
    projection, an ordinary undoable weight edit — for ``nn.MultiheadAttention``, or ``("intervention", ablate)``
    on the head's slice for function-style attention."""
    from ..errors import OptionError
    from ..manipulate.edits import Edit
    from ..manipulate.interventions import ablate

    s = _head_switch(net, nid)
    if s is None:
        raise OptionError(f"{nid!r} has no heads deepvis can switch off.",
                          "pick an nn.MultiheadAttention or F.scaled_dot_product_attention node.")
    H = s[0]
    if not 0 <= h < H:
        raise OptionError(f"head {h} doesn't exist: {nid} has {H} heads.", f"pick 0 … {H - 1}.")
    mod = _mha(net, nid)
    if mod is not None:
        paths = {name: info.path for name, info in net.graph.nodes[nid].params.items()}
        slices = _value_slices(mod, h)
        missing = [name for name, _ in slices if name not in paths]
        if missing:
            raise OptionError(f"can't switch head {h} of {nid!r} off: deepvis doesn't see its {', '.join(missing)}.",
                              "switch it off yourself, e.g. with net.intervene on the node's output.")
        zero = [Edit("zero", paths[name], index=ix) for name, ix in slices]
        return "edits", [Edit("group", nid, args={"edits": zero}, label=f"head {h} of {nid} off")]  # one undo step
    return "intervention", ablate([h], axis=1)


def head_off(net: "Net", nid: str, h: int):
    """A context manager with head ``h`` of attention node ``nid`` switched off (yields interventions to pass to
    ``net.trace`` for function-style attention; ``{}`` when the weights themselves were changed)."""
    s = _head_switch(net, nid)
    if s is None:
        from ..errors import OptionError

        raise OptionError(f"{nid!r} has no heads deepvis can switch off.",
                          "pick an nn.MultiheadAttention or F.scaled_dot_product_attention node.")
    if not 0 <= h < s[0]:
        from ..errors import OptionError

        raise OptionError(f"head {h} doesn't exist: {nid} has {s[0]} heads.", f"pick 0 … {s[0] - 1}.")
    return s[1](h)


__all__ = ["rollout", "head_importance", "head_off", "head_ablation", "attention_layers"]
