"""Recurrent layers unrolled over time: every gate at every step, recomputed from the layer's own weights.

For ``nn.LSTM`` / ``nn.GRU`` / ``nn.RNN`` (PyTorch's gate order and equations, every layer and direction), from the
recorded input sequence and zero initial state::

    LSTM  i, f, g, o = σ(·), σ(·), tanh(·), σ(·) of W_ih x_t + b_ih + W_hh h_{t−1} + b_hh
          c_t = f ⊙ c_{t−1} + i ⊙ g        h_t = o ⊙ tanh(c_t)
    GRU   r, z = σ(·) · n = tanh(W_in x_t + b_in + r ⊙ (W_hn h_{t−1} + b_hn)) · h_t = (1 − z) ⊙ n + z ⊙ h_{t−1}
    RNN   h_t = tanh(W_ih x_t + b_ih + W_hh h_{t−1} + b_hh)      (or ReLU)

``match`` is the largest difference between the recomputed hidden states and the layer's real output, so the
page only shows an unrolling that reproduces the layer exactly.
"""

from __future__ import annotations

import numpy as np


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _cell(kind: str, x_t, h, c, Wih, Whh, bih, bhh, relu: bool):
    gi = Wih @ x_t + bih
    gh = Whh @ h + bhh
    if kind == "LSTM":
        i, f, g, o = np.split(gi + gh, 4)
        i, f, g, o = _sig(i), _sig(f), np.tanh(g), _sig(o)
        c = f * c + i * g
        return o * np.tanh(c), c, {"input gate i": i, "forget gate f": f, "candidate g": g, "output gate o": o}
    if kind == "GRU":
        ir, iz, inn = np.split(gi, 3)
        hr, hz, hn = np.split(gh, 3)
        r, z = _sig(ir + hr), _sig(iz + hz)
        n = np.tanh(inn + r * hn)
        return (1 - z) * n + z * h, None, {"reset gate r": r, "update gate z": z, "candidate n": n}
    a = gi + gh
    return (np.maximum(a, 0) if relu else np.tanh(a)), None, {}


def unroll(module, x: np.ndarray, sample: int = 0) -> dict | None:
    """Unroll one sample of ``x`` (the layer's input) through ``module``; None for unsupported layouts."""
    kind = type(module).__name__
    if kind not in ("LSTM", "GRU", "RNN") or getattr(module, "proj_size", 0):
        return None
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 3:
        x = x[sample] if module.batch_first else x[:, sample]
    elif x.ndim != 2:
        return None
    relu = getattr(module, "nonlinearity", "tanh") == "relu"
    H = module.hidden_size
    dirs = 2 if module.bidirectional else 1
    seq = x
    layers = []
    for layer in range(module.num_layers):
        outs = []
        for d in range(dirs):
            sfx = f"_l{layer}" + ("_reverse" if d else "")
            p = lambda n: getattr(module, n + sfx).detach().double().cpu().numpy()  # noqa: E731
            Wih, Whh = p("weight_ih"), p("weight_hh")
            bih = p("bias_ih") if module.bias else np.zeros(Wih.shape[0])
            bhh = p("bias_hh") if module.bias else np.zeros(Wih.shape[0])
            h, c = np.zeros(H), np.zeros(H)
            hs, cs, gates = [], [], {}
            order = range(len(seq) - 1, -1, -1) if d else range(len(seq))
            for t in order:
                h, c, g = _cell(kind, seq[t], h, c, Wih, Whh, bih, bhh, relu)
                hs.append(h)
                cs.append(c)
                for k, v in g.items():
                    gates.setdefault(k, []).append(v)
            if d:  # back to time order
                hs, cs = hs[::-1], cs[::-1]
                gates = {k: v[::-1] for k, v in gates.items()}
            outs.append(np.stack(hs))
            layers.append({"layer": layer, "direction": "←" if d else "→", "h": np.stack(hs),
                           "c": np.stack(cs) if kind == "LSTM" else None,
                           "gates": {k: np.stack(v) for k, v in gates.items()}})
        seq = np.concatenate(outs, axis=-1)
    return {"kind": kind, "layers": layers, "output": seq, "hidden": H, "steps": int(len(x))}


def unroll_node(net, trace, nid: str, sample: int = 0) -> dict | None:
    """Unroll a recorded recurrent node; adds ``match`` (max |recomputed − real| over the output)."""
    import torch

    node = net.graph.nodes.get(nid)
    if node is None or node.op != "module" or nid not in trace.inputs or nid not in trace.acts:
        return None
    mod = net.backend.model.get_submodule(node.module or nid)  # a second call ("lstm@2") uses lstm
    if not isinstance(mod, (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN)):
        return None
    u = unroll(mod, trace.inputs[nid], sample)
    if u is None:
        return None
    real = np.asarray(trace.acts[nid], dtype=np.float64)
    if real.ndim == 3:
        real = real[sample] if mod.batch_first else real[:, sample]
    u["match"] = float(np.max(np.abs(real - u["output"]))) if real.shape == u["output"].shape else None
    return u


__all__ = ["unroll", "unroll_node"]
