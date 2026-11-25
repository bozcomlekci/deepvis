"""Plain-numpy data for the animations (no manim import, so it is cheap to test)."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from ..core.tensor import first_array, to_numpy
from ..manipulate.edits import Edit
from ..viz.html.layout import layout
from ..viz.html.payload import neuron_view

if TYPE_CHECKING:  # pragma: no cover
    from ..net import Net


_ACT_NAMES = {"Sigmoid": "the sigmoid", "ReLU": "ReLU", "Tanh": "tanh", "GELU": "GELU", "SiLU": "SiLU",
              "Softmax": "a softmax", "LeakyReLU": "a leaky ReLU", "ELU": "ELU", "LogSoftmax": "a log-softmax"}


def colors(net: "Net") -> dict:
    """Node id → its family colour (as in the dashboard), for colour-coded equations; None for inputs / outputs."""
    from .. import theme

    out = {}
    for n in net.graph.nodes.values():
        fam = theme.kind_family(n.kind, n.is_container)
        out[n.id] = None if fam == "io" else theme.FAMILY_COLORS[fam]
    return out


def _f32(b64: str, shape) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype=np.float32).reshape(shape).astype(np.float64)


def dense_data(net: "Net", args: tuple, max_neurons: int = 10, shown=None, trace=None) -> dict:
    """Layers / weights / activations of the longest chain of dense layers."""
    tr = trace if trace is not None else net.trace(*args)
    nv = neuron_view(net, tr, max_neurons=max_neurons, shown=shown, equations=False)
    if not nv or "layers" not in nv:
        raise ValueError("this network has no chain of dense (Linear) layers to draw; try kind='architecture'")
    layers = []
    g = net.graph
    for l_ in nv["layers"]:
        post = g.nodes.get(l_.get("post")) if l_.get("post") else None
        layers.append({"node": l_.get("node"), "size": l_["size"], "shown": l_["shown"],
                       "values": np.asarray(l_["values"]) if l_.get("values") is not None else None,
                       "range": l_.get("range"),
                       "act": _ACT_NAMES.get(post.kind, post.kind) if post is not None else None})
    weights = [_f32(lk["w"], (lk["rows"], lk["cols"])) for lk in nv["links"]]
    image = None
    if nv.get("image"):
        side = nv["image"]["side"]
        image = _f32(nv["image"]["data"], (side, side))
    out = to_numpy(first_array(tr.output)).astype(np.float64)
    out = (out[0] if out.ndim >= 2 else out).reshape(-1)
    probs = bool(np.all(out >= -1e-6) and np.all(out <= 1 + 1e-6) and abs(out.sum() - 1) < 1e-3)
    return {"name": net.name, "layers": layers, "weights": weights, "image": image, "output": out,
            "probabilities": probs, "chain": nv["chain"]}


def equation_data(net: "Net") -> dict:
    eqs = {}
    for mode in ("compact", "system", "expanded"):
        try:
            eqs[mode] = net.equation(mode)
        except Exception:  # noqa: BLE001
            pass
    g = net.graph
    layers = [{"id": n.id, "kind": n.kind, "label": n.label or n.id} for n in g.param_nodes()]
    return {"name": net.name, "equations": eqs, "layers": layers, "colors": colors(net)}


def architecture_data(net: "Net", max_layers: int = 14) -> dict:
    g = net.graph
    choice = g
    for d in range(g.max_depth(), -1, -1):
        gd = g if d == g.max_depth() else g.collapse(d)
        lay = layout(gd)
        n_layers = max((v["layer"] for v in lay["nodes"].values()), default=0) + 1
        choice, chosen_lay = gd, lay
        if n_layers <= max_layers:
            break
    from ..viz.explain import caption

    nodes = []
    for nid, v in chosen_lay["nodes"].items():
        n = choice.nodes[nid]
        shape = n.out_shapes[0] if n.out_shapes and n.out_shapes[0] is not None else None
        nodes.append({"id": nid, "kind": n.type_name if n.attrs.get("collapsed") else n.kind, "op": n.op,
                      "label": n.label or nid, "layer": v["layer"], "x": v["x"], "shape": shape,
                      "params": n.n_params, "caption": caption(n, choice)})
    edges = [{"src": e.src, "dst": e.dst} for e in choice.edges if e.src in chosen_lay["nodes"]
             and e.dst in chosen_lay["nodes"]]
    return {"name": net.name, "nodes": nodes, "edges": edges, "order": [n for n in choice.topo_order()
                                                                      if n in chosen_lay["nodes"]]}


def focus_data(net: "Net", args: tuple, node: str | None = None, neuron: int | None = None,
               max_neurons: int = 10) -> dict:
    from ..equations.unroll import scalar_equation

    tr = net.trace(*args)
    d = dense_data(net, args, max_neurons, trace=tr)
    chain = d["chain"]
    if node is None or node not in chain:
        node = chain[min(1, len(chain) - 1)] if len(chain) > 1 else chain[0]
    li = chain.index(node) + 1
    layer = d["layers"][li]
    if neuron is None:
        vals = layer["values"]
        neuron = int(layer["shown"][int(np.argmax(np.abs(vals)))]) if vals is not None else int(layer["shown"][0])
    eq = scalar_equation(net, node=node, neuron=neuron, top_k=4, trace=tr)
    g = net.graph
    r = g.nodes[node].roles["weight"]
    W = net.backend.get_param(g.nodes[node].params[r.name].path).astype(np.float64)
    if r.in_axis is not None:
        W = np.transpose(W, (r.out_axis % 2, r.in_axis % 2))
    d.update({"focus_layer": li, "focus_node": node, "neuron": neuron, "equation": eq, "matrix": W,
              "colors": colors(net)})
    return d


def edit_data(net: "Net", args: tuple, edit: Any, keep: bool = False, max_neurons: int = 10) -> dict:
    """Dense-chain data before and after an edit (same neurons shown in both)."""
    before = dense_data(net, args, max_neurons)
    shown = [l_["shown"] for l_ in before["layers"]]

    def apply():
        if callable(edit) and not isinstance(edit, Edit):
            edit(net)
            return "custom edit"
        e = edit if isinstance(edit, Edit) else Edit.from_dict(edit)
        net.apply_edit(e)
        return e.describe()

    if keep:
        label = apply()
        after = dense_data(net, args, max_neurons, shown=shown)
    else:
        with net.edits():
            label = apply()
            after = dense_data(net, args, max_neurons, shown=shown)
    return {"name": net.name, "before": before, "after": after, "label": label}


_LIGHT_FAMILIES = {"activation", "norm", "dropout"}


def volumes_data(net: "Net", args: tuple, max_groups: int = 10, face: int = 8) -> dict:
    """Tensors along the main path as 3-D blocks (conv → norm → activation merged, like the dashboard)."""
    from .. import theme
    from ..core.tensor import downsample2d
    from ..viz.html.payload import _is_batched

    tr = net.trace(*args)
    g0 = net.graph
    batched = _is_batched(net, tr)
    choice = g0
    for d in range(g0.max_depth(), -1, -1):  # coarsest depth that still shows enough structure
        gd = g0 if d == g0.max_depth() else g0.collapse(d)
        n = len([x for x in gd.compute_nodes() if x.op not in ("output",)])
        choice = gd
        if n <= max_groups * 2.5:
            break
    g = choice
    order = [nid for nid in g.topo_order() if g.nodes[nid].op not in ("output", "param")]
    groups: list[list[str]] = []
    for nid in order:
        n = g.nodes[nid]
        p = g.pred(nid)
        fam = theme.kind_family(n.kind)
        if (groups and len(p) == 1 and p[0] == groups[-1][-1] and len(g.succ(p[0])) == 1 and fam in _LIGHT_FAMILIES
                and n.out_shapes == g.nodes[p[0]].out_shapes and g.nodes[p[0]].op != "input"):
            groups[-1].append(nid)
        else:
            groups.append([nid])
    if len(groups) > max_groups:  # keep the input, the output and evenly spaced blocks in between
        idx = sorted(set(np.linspace(0, len(groups) - 1, max_groups).round().astype(int).tolist()))
        groups = [groups[i] for i in idx]

    def act_of(nid):
        node = g.nodes[nid]
        key = nid if nid in tr.acts else None
        if key is None and node.attrs.get("collapsed"):
            members = {m for m in g0.descendants(nid) if g0.nodes[m].is_compute} if nid in g0.nodes else set()
            outs = [m for m in g0.topo_order() if m in members and any(s not in members for s in g0.succ(m))]
            key = outs[-1] if outs else None
        a = tr.acts.get(key) if key else None
        if a is None:
            return None
        a = np.asarray(a, dtype=np.float64)
        return a[0] if batched and a.ndim >= 2 else a

    blocks = []
    for grp in groups:
        head = g.nodes[grp[0]]
        a = act_of(grp[-1])
        if a is None or a.ndim == 0:
            continue
        if a.ndim == 1:
            kind, dims, front = "vec", [1, a.shape[0], 1], a.reshape(-1, 1)
        elif a.ndim == 2:
            kind, dims, front = "seq", [1, a.shape[0], a.shape[1]], a
        else:
            ch = (net.backend.feature_axis(head) % (a.ndim + 1)) - 1 if batched else 0
            v = np.moveaxis(a, ch if ch >= 0 else a.ndim - 1, 0)
            v = v.reshape(v.shape[0], -1, v.shape[-1])
            kind, dims, front = "maps", list(v.shape), v.mean(0)
        small, _ = downsample2d(front, face if kind != "vec" else 16, face)
        m = float(np.max(np.abs(small))) or 1.0
        signed = bool(small.min() < 0)
        colors = (theme.colormap(small.ravel(), vmin=-m, vmax=m) if signed else
                  theme.colormap(small.ravel(), stops=theme.SEQUENTIAL_STOPS, vmin=0, vmax=m))
        kinds = list(dict.fromkeys(g.nodes[x].type_name if g.nodes[x].attrs.get("collapsed") else
                                   ("input" if g.nodes[x].op == "input" else g.nodes[x].kind) for x in grp))
        blocks.append({"ids": grp, "kinds": kinds, "kind": kind, "dims": [int(d) for d in dims],
                       "face": {"rows": int(small.shape[0]), "cols": int(small.shape[1]), "colors": colors},
                       "color": theme.kind_color(g.nodes[grp[0]].kind if g.nodes[grp[0]].op != "input" else "Input"),
                       "family_colors": [theme.kind_color(g.nodes[x].kind) for x in grp]})
    return {"name": net.name, "blocks": blocks}


def training_data(net: "Net", log: Any, max_layers: int = 18, max_snapshots: int = 48, max_points: int = 240) -> dict:
    """A training run (:class:`~deepvis.TrainingLog`, or a live server's ``.training``) for the training scene:
    the main metric curve and a layers × snapshots grid of per-step relative updates."""
    from .. import theme

    log = getattr(log, "training", log)
    if log is None or not len(log):
        raise ValueError("the 'training' animation needs log=... (a TrainingLog with recorded steps)")
    d = log.to_dict(max_points=max_points, max_snapshots=max_snapshots)
    names = list(d["metrics"])
    metric = next((k for k in names if "loss" in k.lower()), names[0] if names else None)
    steps = np.asarray(d["steps"], dtype=float)
    curve = None
    if metric is not None:
        ys = np.array([np.nan if v is None else v for v in d["metrics"][metric]], dtype=float)
        ok = np.isfinite(ys)
        log_scale = bool(ok.any() and np.all(ys[ok] > 0) and ys[ok].max() / ys[ok].min() > 20)
        curve = {"name": metric, "steps": steps[ok], "values": ys[ok], "log": log_scale}
    layers = [l_ for l_ in d["layers"] if l_["stats"].get("update")]
    if len(layers) > max_layers:  # keep the first, the last and evenly spaced ones in between
        idx = sorted(set(np.linspace(0, len(layers) - 1, max_layers).round().astype(int).tolist()))
        layers = [layers[i] for i in idx]
    grid = np.array([[np.nan if v is None or v <= 0 else np.log10(v) for v in l_["stats"]["update"]]
                     for l_ in layers], dtype=float) if layers else np.zeros((0, 0))
    colors = None
    if grid.size and np.isfinite(grid).any():
        lo, hi = np.nanpercentile(grid, 2), np.nanpercentile(grid, 98)
        hi = hi if hi > lo else lo + 1
        flat = theme.colormap(np.nan_to_num(grid, nan=lo).ravel(), stops=theme.SEQUENTIAL_STOPS, vmin=lo, vmax=hi)
        colors = np.array(flat, dtype=object).reshape(grid.shape)
        colors[~np.isfinite(grid)] = theme.GREY_E
        rng = (float(10 ** lo), float(10 ** hi))
    else:
        rng = None
    return {"name": net.name, "curve": curve, "snap_steps": np.asarray(d["snapSteps"], dtype=float),
            "layers": [{"id": l_["id"], "kind": l_["kind"]} for l_ in layers], "grid": grid, "colors": colors,
            "range": rng, "last_step": d["lastStep"],
            "last": {k: next((v for v in reversed(vals) if v is not None), None) for k, vals in d["metrics"].items()}}


DATA_BUILDERS: dict[str, Callable] = {
    "forward": dense_data, "equation": equation_data, "architecture": architecture_data, "focus": focus_data,
    "edit": edit_data, "volumes": volumes_data, "training": training_data,
}
