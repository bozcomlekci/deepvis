"""What some architectures do that a layer graph alone doesn't show.

* mixture of experts — which experts the router sends each token to (``routing``): a router / gate module next
  to an ``experts`` module (HF Mixtral, Qwen-MoE, Switch, or your own ``gate = nn.Linear(d, n_experts)``);
  probabilities = softmax(router logits), the chosen ones = its top-k.
* selective state spaces (Mamba) — how much each token is written into the state (``ssm``): Δ = softplus(dt_proj(·)),
  per token and channel; large Δ = this token overwrites the state, small Δ = the state carries on.
"""

from __future__ import annotations

import numpy as np


def _module(net, nid: str):
    try:
        return net.backend.model.get_submodule(nid) if nid else net.backend.model
    except AttributeError:
        return None


def find_routers(net) -> list[dict]:
    """[{router, block, experts, k}] for every mixture-of-experts router in a torch network."""
    import torch

    out = []
    for nid, n in net.graph.nodes.items():
        if n.op != "module" or not nid:
            continue
        mod = _module(net, nid)
        parent_id, _, name = nid.rpartition(".")
        parent = _module(net, parent_id)
        experts = getattr(parent, "experts", None) if parent is not None else None
        if mod is None or experts is None or experts is mod:
            continue
        routerish = "router" in type(mod).__name__.lower() or name in ("gate", "router", "w_gate", "gating", "switch")
        if not routerish:
            continue
        E = getattr(mod, "num_experts", None) or (len(experts) if isinstance(experts, (torch.nn.ModuleList, list, tuple))
                                                   else getattr(experts, "num_experts", None))
        if not E and isinstance(mod, torch.nn.Linear):
            E = mod.out_features
        k = (getattr(mod, "top_k", None) or getattr(parent, "top_k", None) or getattr(parent, "num_experts_per_tok", None)
             or getattr(parent, "k", None) or 2)
        if E:
            out.append({"router": nid, "block": parent_id, "experts": int(E), "k": int(min(k, E))})
    return out


def routing(net, trace, sample: int = 0, batch: int = 1) -> list[dict]:
    """Per router: probabilities (tokens × experts), the experts chosen per token, and each expert's load."""
    from ..core.tensor import to_numpy

    res = []
    for r in find_routers(net):
        a = trace.acts.get(r["router"])
        if a is None:
            continue
        L = np.asarray(to_numpy(a), dtype=np.float64)
        E = r["experts"]
        if L.shape[-1] != E:
            continue
        if L.ndim == 3:
            L = L[min(sample, L.shape[0] - 1)]
        elif L.ndim == 2 and batch > 1 and L.shape[0] % batch == 0:
            L = L.reshape(batch, -1, E)[min(sample, batch - 1)]
        elif L.ndim != 2:
            continue
        P = np.exp(L - L.max(-1, keepdims=True))
        P /= P.sum(-1, keepdims=True)
        chosen = np.argsort(-P, axis=-1)[:, : r["k"]]
        load = np.bincount(chosen.ravel(), minlength=E)
        res.append({**r, "probs": P, "chosen": chosen, "load": load})
    return res


def ssm(net, trace, sample: int = 0) -> list[dict]:
    """Per Mamba-style mixer (a module with a ``dt_proj``): Δ = softplus(dt_proj(·)), tokens × channels — read from
    the mixer's softplus (HF's reference path applies dt_proj's weight directly), else from dt_proj's output."""
    from ..core.tensor import to_numpy

    res = []
    g = net.graph
    for nid, n in g.nodes.items():
        dt = g.nodes.get(nid + ".dt_proj")
        if dt is None:
            continue
        width = None
        mod = _module(net, nid + ".dt_proj")
        if mod is not None and hasattr(mod, "out_features"):
            width = int(mod.out_features)
        soft = [k for k, m in g.nodes.items() if m.parent == nid and m.kind == "Softplus" and k in trace.acts]
        if soft:
            D, softplussed = np.asarray(to_numpy(trace.acts[soft[0]]), dtype=np.float64), True
        elif dt.id in trace.acts:
            D, softplussed = np.asarray(to_numpy(trace.acts[dt.id]), dtype=np.float64), False
        else:
            continue
        if D.ndim == 3:
            D = D[min(sample, D.shape[0] - 1)]
        if D.ndim != 2:
            continue
        if width is not None and D.shape[0] == width and D.shape[1] != width:
            D = D.T  # (channels, tokens) → (tokens, channels)
        delta = D if softplussed else np.logaddexp(0.0, D)
        res.append({"mixer": nid, "delta": delta, "per_token": delta.mean(-1)})
    return res


def spring_layout(n: int, edges: np.ndarray, iters: int = 250, seed: int = 0) -> np.ndarray:
    """Force-directed positions in [0, 1]² (Fruchterman–Reingold) for ``n`` nodes and (2, E) ``edges``."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1, 1, (n, 2))
    k = 1.0 / np.sqrt(max(n, 1))
    t = 0.12
    e = edges[:, edges[0] != edges[1]] if edges.size else edges
    for _ in range(iters):
        d = pos[:, None, :] - pos[None, :, :]
        dist2 = (d ** 2).sum(-1) + 1e-6
        disp = ((k * k / dist2)[..., None] * d).sum(1)                      # everyone repels
        if e.size:
            dd = pos[e[0]] - pos[e[1]]
            dl = np.linalg.norm(dd, axis=1) + 1e-9
            f = (dl / k)[:, None] * dd                                        # edges pull together
            np.add.at(disp, e[0], -f)
            np.add.at(disp, e[1], f)
        ln = np.linalg.norm(disp, axis=1) + 1e-9
        pos += disp / ln[:, None] * np.minimum(ln, t)[:, None]
        t *= 0.985
    pos -= pos.min(0)
    span = pos.max(0)
    pos /= span.max() or 1.0
    return pos + (1.0 - span / (span.max() or 1.0)) / 2  # centred in the unit square, aspect kept


def graph_input(net, trace, max_nodes: int = 400, max_edges: int = 3000) -> dict | None:
    """A graph neural network's input graph — node features (N, F) with an edge_index (2, E) — laid out, with each
    layer's per-node output size (‖h_v‖) to colour it by."""
    from ..core.tensor import to_numpy

    arrs = [np.asarray(to_numpy(a)) for a in trace.args if hasattr(a, "shape")]
    feats = next((a for a in arrs if a.ndim == 2 and np.issubdtype(a.dtype, np.floating)), None)
    ei = next((a for a in arrs if a.ndim == 2 and a.shape[0] == 2 and np.issubdtype(a.dtype, np.integer)), None)
    if feats is None or ei is None:
        return None
    N = feats.shape[0]
    if N > max_nodes or ei.size == 0 or ei.max() >= N or ei.min() < 0:
        return None
    edges = ei[:, :max_edges].astype(np.int64)
    pos = spring_layout(N, edges)
    layers = [{"id": "input", "kind": "features", "values": np.linalg.norm(feats, axis=1).tolist()}]
    for nid in net.graph.topo_order():
        n = net.graph.nodes[nid]
        a = trace.acts.get(nid)
        if a is None or n.op not in ("module", "function") or nid.startswith("@in"):
            continue
        a = np.asarray(to_numpy(a), dtype=np.float64)
        if a.ndim == 2 and a.shape[0] == N:
            layers.append({"id": nid, "kind": n.kind, "values": np.linalg.norm(a, axis=1).tolist(),
                           "argmax": a.argmax(1).tolist() if nid == net.graph.topo_order()[-1] or n.op == "output"
                           else None})
    return {"n": int(N), "edges": edges.T.tolist(), "pos": np.round(pos, 4).tolist(), "layers": layers,
            "directed": bool(not np.array_equal(np.sort(edges, 0), np.sort(edges[::-1], 0)))}


__all__ = ["find_routers", "routing", "ssm", "spring_layout", "graph_input"]
