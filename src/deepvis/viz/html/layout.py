"""Layered DAG layout (Sugiyama-style) computed in Python so the browser only draws.

* longest-path layering (sources that aren't inputs are pulled next to their consumers)
* dummy nodes on long edges, barycenter ordering sweeps (siblings of a container kept together)
* iterative x-relaxation towards neighbour medians with non-overlap constraints
* extra vertical room where containers open/close so nested group boxes never collide
"""

from __future__ import annotations

from ...core.graph import ModelGraph

SIZES = {
    "module": (176, 50),
    "collapsed": (190, 56),
    "function": (112, 32),
    "param": (120, 32),
    "input": (104, 32),
    "output": (104, 32),
}
ROW_GAP = 40
COL_GAP = 26
FAR = 8  # an edge longer than this many rows, from a value many far layers read (rotary cos/sin, masks), is drawn
         # as one faint line instead of being routed through every row it passes
MARGIN = 40
HEADER = 22
PAD = 14


def _size(n) -> tuple[int, int]:
    """Node box size; module boxes grow so 'Kind   (out, shape)' never collides."""
    if n.attrs.get("collapsed"):
        w, h = SIZES["collapsed"]
        title = n.type_name
    else:
        w, h = SIZES.get(n.op, SIZES["module"])
        title = n.kind
    if n.op == "module" or n.attrs.get("collapsed"):
        shape = str(tuple(n.out_shapes[0])) if n.out_shapes and n.out_shapes[0] is not None else ""
        need = 14 + 7.8 * len(title) + 14 + 6.7 * len(shape) + 10
        w = max(w, int(need + 0.999))
    elif n.op in ("function", "param", "input", "output"):
        w = max(w, int(26 + 7.2 * min(len(n.label or n.kind), 18)))
    return w, h


def _is_noop(n) -> bool:
    """Placeholders that do nothing at inference (``nn.Identity``, ``Dropout(p=0)``): timm has hundreds."""
    if n.op != "module":
        return False
    rate = n.attrs.get("p", n.attrs.get("rate"))
    return n.kind == "Identity" or (n.kind == "Dropout" and rate is not None and not rate)


def contract_noops(g: ModelGraph) -> ModelGraph:
    """A drawing view of ``g`` without no-op layers; edges are rewired through them (nodes are shared)."""
    skip = {nid for nid, n in g.nodes.items() if _is_noop(n) and len(g.pred(nid)) == 1}
    if not skip:
        return g
    v = ModelGraph(g.name, g.framework)
    v.nodes = {k: n for k, n in g.nodes.items() if k not in skip}

    def real_src(nid: str) -> str:
        while nid in skip:
            nid = g.pred(nid)[0]
        return nid

    for e in g.edges:
        if e.dst in skip:
            continue
        v.add_edge(real_src(e.src), e.dst, e.shape, e.implicit)
    return v


def layout(g: ModelGraph, size_fn=None, row_gap: float | None = None, tensors: bool = False) -> dict:
    """Lay out ``g`` top-to-bottom. ``size_fn(node) -> (w, h)`` overrides node sizes (expressive glyphs);
    ``tensors=True`` adds a 3-D tensor annotation on the first out-edge of every node."""
    g = contract_noops(g)
    ROW_GAP_ = ROW_GAP if row_gap is None else row_gap
    sz = size_fn or _size
    # modules that never ran (e.g. a dropout the fused attention path skips) have no dataflow: leave them out
    order = [nid for nid in g.topo_order()
             if not (g.nodes[nid].attrs.get("unused") and not g.pred(nid) and not g.succ(nid))] or g.topo_order()
    idx = {nid: i for i, nid in enumerate(order)}
    preds = {n: [p for p in g.pred(n) if p in idx] for n in order}
    succs = {n: [s for s in g.succ(n) if s in idx] for n in order}

    # ---------------------------------------------------------------- layering
    layer: dict[str, int] = {}
    for n in order:  # a cycle (a container entered twice at a collapsed level) is drawn with its back edge ignored
        layer[n] = max((layer[p] + 1 for p in preds[n] if p in layer), default=0)
    for n in reversed(order):  # pull non-input sources down next to their first consumer
        if not preds[n] and g.nodes[n].op != "input" and succs[n]:
            layer[n] = max(0, min(layer[s] for s in succs[n]) - 1)
    for n in reversed(order):  # an op whose consumers are all far below sits next to them, not up by its inputs
        if preds[n] and succs[n] and g.nodes[n].op not in ("input", "output"):
            target = min(layer[s] for s in succs[n]) - 1
            if target - layer[n] > FAR // 2:
                layer[n] = target
    out_nodes = [n for n in order if g.nodes[n].op == "output"]
    if out_nodes:
        last = max(layer.values())
        for n in out_nodes:
            layer[n] = last
    long_out: dict[str, int] = {}
    for e in g.edges:
        if e.src in idx and e.dst in idx and layer[e.dst] - layer[e.src] > FAR:
            long_out[e.src] = long_out.get(e.src, 0) + 1

    # ---------------------------------------------------------- dummy nodes
    nodes = {}
    for n in order:
        w_, h_ = sz(g.nodes[n])
        nodes[n] = {"w": w_, "h": h_, "layer": layer[n], "dummy": False}
    chains: list[tuple[str, str, list[str]]] = []
    adj_up: dict[str, list[str]] = {n: [] for n in order}
    adj_dn: dict[str, list[str]] = {n: [] for n in order}
    k = 0
    for e in g.edges:
        if e.src not in idx or e.dst not in idx:
            continue
        a, b = e.src, e.dst
        span = layer[b] - layer[a]
        if span > FAR and long_out.get(a, 0) >= 3:  # broadcast: drawn straight, left out of ordering
            chains.append((a, b, None))
            continue
        chain = []
        prev = a
        for li in range(layer[a] + 1, layer[b]):
            d = f"__d{k}"
            k += 1
            nodes[d] = {"w": 12, "h": 0, "layer": li, "dummy": True}
            adj_up[d], adj_dn[d] = [prev], []
            adj_dn[prev].append(d)
            chain.append(d)
            prev = d
        if span >= 1:
            adj_dn[prev].append(b)
            adj_up[b].append(prev)
        chains.append((a, b, chain))

    n_layers = max((v["layer"] for v in nodes.values()), default=0) + 1
    rows: list[list[str]] = [[] for _ in range(n_layers)]
    for n in sorted(nodes, key=lambda n: (idx.get(n, 1e9), n)):
        rows[nodes[n]["layer"]].append(n)

    # ------------------------------------------------------------- ordering
    group_of: dict[str, str] = {}

    def group_key(n):  # a dummy node belongs to the group of the node its edge starts from (memoized: long chains)
        if n not in group_of:
            if n in g.nodes:
                group_of[n] = g.nodes[n].parent or ""
            else:
                up = adj_up.get(n)
                group_of[n] = group_key(up[0]) if up else ""
        return group_of[n]

    pos = {n: i for row in rows for i, n in enumerate(row)}
    for it in range(12):
        rng = range(1, n_layers) if it % 2 == 0 else range(n_layers - 2, -1, -1)
        nb = adj_up if it % 2 == 0 else adj_dn
        for li in rng:
            row = rows[li]
            if len(row) < 2:  # deep, narrow networks: most rows hold one node, nothing to order
                continue
            bary = {}
            for n in row:
                ps = [pos[p] for p in nb[n] if p in pos]
                bary[n] = sum(ps) / len(ps) if ps else pos[n]
            first_seen: dict[str, float] = {}
            for n in row:
                gk = group_key(n)
                first_seen[gk] = min(first_seen.get(gk, 1e9), bary[n])
            row.sort(key=lambda n: (first_seen[group_key(n)], bary[n]))
            for i, n in enumerate(row):
                pos[n] = i

    # ---------------------------------------------------------- coordinates
    x: dict[str, float] = {}
    for row in rows:
        total = sum(nodes[n]["w"] for n in row) + COL_GAP * max(len(row) - 1, 0)
        cur = -total / 2
        for n in row:
            x[n] = cur + nodes[n]["w"] / 2
            cur += nodes[n]["w"] + COL_GAP

    widths = {n: v["w"] for n, v in nodes.items()}

    def separate(row, target):
        # keep order, enforce min gaps, stay as close to targets as possible (two-pass average)
        k = len(row)
        half = [0.0] + [(widths[row[i - 1]] + widths[row[i]]) / 2 for i in range(1, k)]
        fwd = list(target)
        for i in range(1, k):
            fwd[i] = max(fwd[i], fwd[i - 1] + half[i] + COL_GAP)
        bwd = list(target)
        for i in range(k - 2, -1, -1):
            bwd[i] = min(bwd[i], bwd[i + 1] - half[i + 1] - COL_GAP)
        mid = [(a + b) / 2 for a, b in zip(fwd, bwd)]
        for i in range(1, k):  # final legalisation
            mid[i] = max(mid[i], mid[i - 1] + half[i] + COL_GAP)
        return mid

    for it in range(40):
        down = it % 2 == 0
        rng = range(n_layers) if down else range(n_layers - 1, -1, -1)
        for li in rng:
            row = rows[li]
            target = []
            for n in row:
                nb = (adj_up[n] + adj_dn[n]) if it > 30 else (adj_up[n] if down else adj_dn[n])
                xs = sorted(x[m] for m in nb if m in x)
                if xs:
                    mid = len(xs) // 2
                    med = xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2
                    target.append(0.7 * med + 0.3 * x[n])
                else:
                    target.append(x[n])
            if len(row) == 1:  # one node: nothing to keep apart
                x[row[0]] = target[0]
            else:
                x.update(zip(row, separate(row, target)))

    # ------------------------------------------------------------ vertical
    containers = [c for c in g.nodes.values() if c.is_container and c.id != ""]
    members: dict[str, list[str]] = {}
    for n in order:
        for a in g.ancestors(n):
            if a != "" and a in g.nodes:
                members.setdefault(a, []).append(n)
    opens = [0] * n_layers
    closes = [0] * n_layers
    for c in containers:
        m = members.get(c.id)
        if not m:
            continue
        lo = min(layer[n] for n in m)
        hi = max(layer[n] for n in m)
        opens[lo] += 1
        closes[hi] += 1
    y_row = []
    cur = 0.0
    for li in range(n_layers):
        cur += (HEADER + 6) * opens[li] + (PAD * closes[li - 1] if li else 0)
        h = max((nodes[n]["h"] for n in rows[li]), default=0)
        y_row.append(cur + h / 2)
        cur += h + ROW_GAP_

    out_nodes_xy = {}
    for n, v in nodes.items():
        out_nodes_xy[n] = {"x": x[n], "y": y_row[v["layer"]], "w": v["w"], "h": v["h"], "layer": v["layer"]}

    # --------------------------------------------------------------- groups
    groups = []
    depth_cache = {c.id: g.depth(c.id) for c in containers}
    box: dict[str, list[float]] = {}
    for c in sorted(containers, key=lambda c: -depth_cache[c.id]):
        m = members.get(c.id)
        if not m:
            continue
        x0 = min(out_nodes_xy[n]["x"] - out_nodes_xy[n]["w"] / 2 for n in m)
        x1 = max(out_nodes_xy[n]["x"] + out_nodes_xy[n]["w"] / 2 for n in m)
        y0 = min(out_nodes_xy[n]["y"] - out_nodes_xy[n]["h"] / 2 for n in m)
        y1 = max(out_nodes_xy[n]["y"] + out_nodes_xy[n]["h"] / 2 for n in m)
        for ch in c.children:
            if ch in box:
                bx = box[ch]
                x0, y0, x1, y1 = min(x0, bx[0]), min(y0, bx[1]), max(x1, bx[2]), max(y1, bx[3])
        box[c.id] = [x0 - PAD, y0 - PAD - HEADER, x1 + PAD, y1 + PAD]
    for c in containers:
        if c.id in box:
            b = box[c.id]
            groups.append({"id": c.id, "label": c.label or c.id.split(".")[-1], "type": c.type_name,
                           "x": b[0], "y": b[1], "w": b[2] - b[0], "h": b[3] - b[1], "depth": depth_cache[c.id]})

    # ------------------------------------------------------------ normalise
    xs = [v["x"] - v["w"] / 2 for v in out_nodes_xy.values()] + [gr["x"] for gr in groups]
    ys = [v["y"] - v["h"] / 2 for v in out_nodes_xy.values()] + [gr["y"] for gr in groups]
    x_min, y_min = (min(xs) if xs else 0), (min(ys) if ys else 0)
    dx, dy = MARGIN - x_min, MARGIN - y_min
    for v in out_nodes_xy.values():
        v["x"] += dx
        v["y"] += dy
    for gr in groups:
        gr["x"] += dx
        gr["y"] += dy
    width = max((v["x"] + v["w"] / 2 for v in out_nodes_xy.values()), default=0)
    width = max([width] + [gr["x"] + gr["w"] for gr in groups]) + MARGIN
    height = max((v["y"] + v["h"] / 2 for v in out_nodes_xy.values()), default=0)
    height = max([height] + [gr["y"] + gr["h"] for gr in groups]) + MARGIN

    edges = []
    for a, b, chain in chains:
        pa, pb = out_nodes_xy[a], out_nodes_xy[b]
        pts = [[pa["x"], pa["y"] + pa["h"] / 2]]
        pts += [[out_nodes_xy[d]["x"], out_nodes_xy[d]["y"]] for d in chain or ()]
        if layer[b] <= layer[a]:  # back/flat edge (rare): route to the side
            pts.append([max(pa["x"], pb["x"]) + max(pa["w"], pb["w"]) / 2 + 30, (pa["y"] + pb["y"]) / 2])
        pts.append([pb["x"], pb["y"] - pb["h"] / 2])
        edge = {"src": a, "dst": b, "points": [[round(px, 1), round(py, 1)] for px, py in pts]}
        if chain is None:
            edge["far"] = True
        edges.append(edge)

    tensor_marks = []
    if tensors:
        from ..glyphs import tensor_glyph

        seen = set()
        for e in edges:
            src = g.nodes[e["src"]]
            if e["src"] in seen or src.op == "output" or not src.out_shapes:
                continue
            seen.add(e["src"])
            tg = tensor_glyph(src.out_shapes[0])
            if tg is None:
                continue
            (ax, ay), (bx, by) = e["points"][0], e["points"][1]
            t = 0.4  # nearer the producer than the consumer (whose group header may sit above it)
            tensor_marks.append({"src": e["src"], "x": round(ax + (bx - ax) * t, 1), "y": round(ay + (by - ay) * t, 1),
                                 **tg})
    real = {n: {k2: round(v2, 1) if isinstance(v2, float) else v2 for k2, v2 in v.items()}
            for n, v in out_nodes_xy.items() if not nodes[n]["dummy"]}
    return {"nodes": real, "edges": edges, "groups": groups, "width": round(width), "height": round(height),
            "tensors": tensor_marks}


def transpose_layout(lay: dict) -> dict:
    """Turn a top-to-bottom layout (computed with swapped node sizes) into a left-to-right one."""
    out = {"nodes": {}, "edges": [], "groups": [], "width": lay["height"], "height": lay["width"], "tensors": []}
    for n, v in lay["nodes"].items():
        out["nodes"][n] = dict(v, x=v["y"], y=v["x"], w=v["h"], h=v["w"])
    for e in lay["edges"]:
        out["edges"].append(dict(e, points=[[p[1], p[0]] for p in e["points"]]))
    for gr in lay["groups"]:
        out["groups"].append(dict(gr, x=gr["y"], y=gr["x"], w=gr["h"], h=gr["w"]))
    for t in lay.get("tensors", []):
        out["tensors"].append(dict(t, x=t["y"], y=t["x"]))
    return out
