"""Everything the dashboard shows, as one JSON-able dict.

Arrays travel as base64 float32 (exact and compact). Layouts and symbolic equations depend
only on the architecture and are cached; weight-dependent parts are rebuilt on demand.
"""

from __future__ import annotations

import base64
import hashlib
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from ... import theme
from ...core.graph import ModelGraph, Node
from ...core.kinds import CONV_KINDS
from ...core.tensor import downsample2d, histogram, singular_values, stats, to_numpy
from .layout import layout

if TYPE_CHECKING:  # pragma: no cover
    from ...monitor.recorder import Monitor, Trace
    from ...net import Net

MAX_HEAT = 96
MAX_VEC = 1024
MAX_NEURONS = 16
MAX_LEVEL_NODES = 70


def b64(a: np.ndarray) -> str:
    a = np.nan_to_num(np.asarray(a, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def b64q(a: np.ndarray) -> str:
    """Display-only data as 8-bit levels over its range: ``u8:lo:hi:<base64>`` (4× smaller than float32).

    Colors are 8-bit anyway; hover readouts stay within 0.4 % of the range. The page decodes it transparently.
    """
    a = np.nan_to_num(np.asarray(a, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = (float(a.min()), float(a.max())) if a.size else (0.0, 1.0)
    q = np.zeros(a.shape, np.uint8) if hi <= lo else np.round((a - lo) / (hi - lo) * 255).astype(np.uint8)
    return f"u8:{lo:.9g}:{hi:.9g}:" + base64.b64encode(np.ascontiguousarray(q).tobytes()).decode("ascii")


def _dedupe(svgs: dict, store: list) -> dict:
    """``{id: svg}`` → ``{id: index into store}``, each distinct string stored once."""
    index: dict[str, int] = {}
    out = {}
    for k, v in svgs.items():
        if v not in index:
            index[v] = len(store)
            store.append(v)
        out[k] = index[v]
    return out


def _shared_svgs(levels: list, store: list) -> list:
    """The levels with each edge tensor's SVG replaced by an index into ``store`` (copies: layouts are cached)."""
    index: dict[str, int] = {}
    out = []
    for lv in levels:
        lay = lv.get("layout")
        if isinstance(lay, dict) and lay.get("tensors"):
            tensors = []
            for t in lay["tensors"]:
                svg = t.get("svg")
                if svg not in index:
                    index[svg] = len(store)
                    store.append(svg)
                tensors.append({k: v for k, v in t.items() if k != "svg"} | {"si": index[svg]})
            lv = dict(lv, layout=dict(lay, tensors=tensors))
        out.append(lv)
    return out


def _compute_counts(g) -> dict:
    """Compute nodes under each node (a leaf counts itself)."""
    memo: dict[str, int] = {}

    def count(nid: str) -> int:
        if nid not in memo:
            n = g.nodes[nid]
            memo[nid] = (1 if n.is_compute else 0) + sum(count(c) for c in (n.children or []) if c in g.nodes)
        return memo[nid]

    for nid in g.nodes:
        count(nid)
    return memo


class Budget:
    """Detail levels that keep the page light for big networks (set per build from the node count)."""

    heat = 96          # max heatmap side for parameters
    maps = 32          # feature-map channels shown per activation
    map_side = 40
    vol_side = 48
    block_eq = 150     # a container's own equation, up to this many layers inside (the whole net: Equations tab)

    @classmethod
    def for_nodes(cls, n: int) -> "Budget":
        b = cls()
        if n > 60:
            b.heat, b.maps, b.map_side, b.vol_side = 64, 16, 28, 32
        if n > 200:
            b.heat, b.maps, b.map_side, b.vol_side = 48, 8, 20, 24
        if n > 600:
            b.block_eq = 80
        return b


BUDGET = Budget()


def _finite_stats(a: np.ndarray, finite: np.ndarray | None = None) -> dict:
    s = stats(a, ("mean", "std", "min", "max", "norm", "sparsity"), finite=finite)
    return {k: (None if (v is None or not math.isfinite(v)) else float(v)) for k, v in s.items()}


def _finite_range(a: np.ndarray) -> tuple[float, float]:
    """(min, max) of the finite values; (0, 0) when there are none (a diverged run is all NaN)."""
    f = a[np.isfinite(a)]
    return (float(f.min()), float(f.max())) if f.size else (0.0, 0.0)


def _abs_max(a: np.ndarray) -> float:
    """Largest finite |value|, for a symmetric colour scale (1 when there is none)."""
    lo, hi = _finite_range(np.asarray(a, dtype=np.float64))
    return max(abs(lo), abs(hi)) or 1.0


def _summary(a: np.ndarray) -> tuple[dict, dict]:
    """The stats and the 40-bin histogram of ``a``, from one pass over its finite values."""
    a = np.asarray(a, dtype=np.float64)
    finite = a[np.isfinite(a)]
    st = _finite_stats(a, finite)
    return st, histogram(finite, 40, (st["min"], st["max"]) if st["min"] is not None else None, finite=True)


def heat(m: np.ndarray, diverging: bool | None = None, max_side: int | None = None, row_label: str = "",
         col_label: str = "", quantize: bool = False) -> dict:
    m = np.asarray(m, dtype=np.float64)
    orig = list(m.shape)
    side = max_side or BUDGET.heat
    small, ds = downsample2d(m, side, side)
    lo, hi = _finite_range(small)
    if diverging is None:
        diverging = lo < 0 < hi or (lo < 0 and hi <= 0)
    if diverging:
        vmax = max(abs(lo), abs(hi)) or 1.0
        vmin = -vmax
    else:
        vmin, vmax = lo, (hi if hi > lo else lo + 1.0)
    data = b64q(small) if (ds or quantize) else b64(small)  # exact where a cell is one number (click-to-edit)
    return {"rows": int(small.shape[0]), "cols": int(small.shape[1]), "data": data, "vmin": vmin,
            "vmax": vmax, "diverging": bool(diverging), "downsampled": bool(ds), "orig": orig,
            "rowLabel": row_label, "colLabel": col_label}


def canonical_order(arr: np.ndarray, node: Node | None, local: str) -> tuple[list[int], str, str]:
    """Axis order giving the (out, in, ...) orientation for weights, whatever the framework stored."""
    ident = list(range(arr.ndim))
    if node is None:
        return ident, "", ""
    for role, r in node.roles.items():
        if r.name != local:
            continue
        if r.out_axis is not None and arr.ndim >= 2:
            oa = r.out_axis % arr.ndim
            ia = r.in_axis % arr.ndim if r.in_axis is not None else None
            order = [oa] + ([ia] if ia is not None else []) + [i for i in range(arr.ndim) if i not in (oa, ia)]
            return order, "out", "in" if ia is not None else ""
        if r.out_axis is not None:
            return ident, "", "neuron"
    return ident, "", ""


def canonical(arr: np.ndarray, node: Node | None, local: str) -> tuple[np.ndarray, str, str]:
    order, rl, cl = canonical_order(arr, node, local)
    return np.transpose(arr, order), rl, cl


def param_payload(arr: np.ndarray, node: Node | None, local: str) -> dict:
    order, rl, cl = canonical_order(np.asarray(arr), node, local)
    a = np.transpose(np.asarray(arr, dtype=np.float64), order)
    st, hist = _summary(a)
    out = {"shape": list(np.shape(arr)), "stats": st, "hist": hist, "perm": order, "cshape": [int(d) for d in a.shape]}
    if a.ndim == 4 and node is not None and node.kind in CONV_KINDS:
        n = min(a.shape[0], 48)
        filt = a[:n]
        rgb = a.shape[1] == 3
        if not rgb:
            filt = filt.mean(axis=1)
        vmax = _abs_max(filt)
        out["filters"] = {"n": n, "total": int(a.shape[0]), "kh": int(a.shape[2]), "kw": int(a.shape[3]),
                          "rgb": bool(rgb), "data": b64(filt), "vmax": vmax}
        out["heat"] = heat(a.reshape(a.shape[0], -1), True, row_label="out", col_label="in·k·k")
        nb = min(a.shape[0], 12)
        bm = _abs_max(a[:nb])
        bank = []
        for i in range(nb):  # each output filter is an (in, kh, kw) volume
            vp = volume_payload(a[i], 0)
            bank.append(vp["faces"] if vp else None)
        out["bank"] = {"n": nb, "total": int(a.shape[0]), "dims": [int(d) for d in a.shape[1:]], "filters": bank,
                       "vmin": -bm, "vmax": bm}
    elif a.ndim >= 2:
        out["heat"] = heat(a.reshape(a.shape[0], -1), True, row_label=rl or "rows", col_label=cl or "cols")
        if a.size <= 300_000:  # keep dashboards fast for big models
            out["svd"] = singular_values(a, 64)
    else:
        out["heat"] = heat(a.reshape(1, -1), True, max_side=512, col_label=cl or "index")
    return out


def pick_sample(a: np.ndarray, sample: int) -> np.ndarray:
    """Row ``sample`` of a batched array (row 0 when the batch axis is broadcast / shorter)."""
    return a[min(int(sample), a.shape[0] - 1)]


def batch_size(net: "Net", trace: "Trace | None") -> int:
    """How many samples the traced input holds (1 for unbatched calls)."""
    if trace is None or not _is_batched(net, trace):
        return 1
    ins = [to_numpy(a) for a in trace.args if hasattr(a, "shape")]
    return int(ins[0].shape[0]) if ins and np.ndim(ins[0]) >= 2 else 1


def act_payload(arr: np.ndarray, feature_axis: int = -1, sample: int = 0, batched: bool = True) -> dict:
    a = np.asarray(arr, dtype=np.float64)
    full_shape = list(a.shape)
    if batched and a.ndim >= 2:
        a = pick_sample(a, sample)
        fa = (feature_axis % len(full_shape)) - 1 if feature_axis is not None else -1
    else:
        fa = feature_axis if feature_axis is not None else -1
    st, hist = _summary(a)
    out: dict[str, Any] = {"shape": full_shape, "stats": st, "hist": hist}
    try:
        out["volume"] = volume_payload(a, fa if a.ndim >= 3 else None)
    except Exception:  # noqa: BLE001 - decoration only
        out["volume"] = None
    if a.ndim == 0:
        a = a.reshape(1)
    if a.ndim == 1:
        v = a
        ds = False
        if v.size > MAX_VEC:
            f = math.ceil(v.size / MAX_VEC)
            pad = np.full(f * math.ceil(v.size / f), np.nan)
            pad[: v.size] = v
            v = np.nanmean(pad.reshape(-1, f), axis=1)
            ds = True
        lo, hi = _finite_range(v)
        div = lo < 0 < hi
        vmax = max(abs(lo), abs(hi)) or 1.0
        out["vector"] = {"n": int(a.size), "data": b64(v), "downsampled": ds, "diverging": div,
                         "vmin": -vmax if div else lo, "vmax": vmax if div else (hi if hi > lo else lo + 1)}
        order = np.argsort(-np.abs(np.nan_to_num(a, nan=0.0, posinf=np.inf, neginf=-np.inf)))[:8]  # ±inf first
        out["top"] = [[int(i), float(a[i]) if np.isfinite(a[i]) else None] for i in order]
    elif a.ndim == 3:  # feature maps: move channel axis first
        ch_ax = fa % 3
        maps = np.moveaxis(a, ch_ax, 0)
        C = maps.shape[0]
        k = min(C, BUDGET.maps)
        side = BUDGET.map_side
        small = []
        for c in range(k):
            s, _ = downsample2d(maps[c], side, side)
            small.append(s)
        h = max(s.shape[0] for s in small)
        w = max(s.shape[1] for s in small)
        stack = np.zeros((k, h, w))
        for i, s in enumerate(small):
            stack[i, : s.shape[0], : s.shape[1]] = s
        lo, hi = _finite_range(stack)
        div = lo < 0 < hi
        vmax = max(abs(lo), abs(hi)) or 1.0
        out["maps"] = {"channels": int(C), "n": k, "h": int(h), "w": int(w), "data": b64q(stack),
                       "diverging": div, "vmin": -vmax if div else lo, "vmax": vmax if div else (hi if hi > lo else lo + 1)}
    else:
        m = a.reshape(a.shape[0], -1) if a.ndim >= 2 else a
        out["matrix"] = heat(m, None, row_label="position", col_label="feature")
    return out


VOL_SIDE = 48


def volume_payload(a: np.ndarray, channel_axis: int | None) -> dict | None:
    """A sample's activation as a (C, H, W) volume plus its three face projections.

    maps (C, H, W) — conv feature maps; seq (1, T, D) — sequences; vec (1, F, 1) — vectors.
    Faces (rows × cols): front = H × W (mean over C), side = H × C (mean over W), top = W × C (mean over H).
    """
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 0 or a.size == 0:
        return None
    if a.ndim == 1:
        vol, kind = a.reshape(1, -1, 1), "vec"
    elif a.ndim == 2:
        vol, kind = a.reshape(1, a.shape[0], a.shape[1]), "seq"
    else:
        ax = (channel_axis if channel_axis is not None else 0) % a.ndim
        v = np.moveaxis(a, ax, 0)
        vol, kind = v.reshape(v.shape[0], int(np.prod(v.shape[1:-1])), v.shape[-1]), "maps"
    shown = []

    def face(m):
        side = min(VOL_SIDE, BUDGET.vol_side)
        small, _ = downsample2d(np.asarray(m, dtype=np.float64), side, side)
        shown.append(small[np.isfinite(small)])
        return {"rows": int(small.shape[0]), "cols": int(small.shape[1]), "data": b64q(small)}

    with np.errstate(invalid="ignore"):
        faces = {"front": face(vol.mean(0)), "side": face(vol.mean(2).T), "top": face(vol.mean(1).T)}
    # colour range from what the faces show (1st–99th percentile): a few outliers — GPT-2's residual stream has
    # some — must not turn every face black. Hovering still reads the exact values.
    vals = np.concatenate(shown) if shown else np.zeros(1)
    lo, hi = map(float, np.percentile(vals, [1, 99])) if vals.size else (0.0, 1.0)
    div = lo < 0 < hi
    vmax = max(abs(lo), abs(hi)) or 1.0
    return {"kind": kind, "dims": [int(d) for d in vol.shape], "faces": faces, "diverging": bool(div),
            "vmin": -vmax if div else lo, "vmax": vmax if div else (hi if hi > lo else lo + 1.0)}


def _digest(a: np.ndarray) -> str:
    return hashlib.blake2b(np.ascontiguousarray(a).data, digest_size=12).hexdigest()  # hashes in place, no copy


class PayloadBuilder:
    """Caches architecture-only parts; rebuilds weight/activation parts on demand."""

    def __init__(self, net: "Net", style: str = "expressive", tensor_shapes: bool = True):
        if style not in ("expressive", "compact"):
            raise ValueError("style must be 'expressive' or 'compact'")
        self.net = net
        self.style = style
        self.tensor_shapes = tensor_shapes and style == "expressive"
        self.sample = 0  # which sample of a batched input the views show
        self.tokens: list[str] | None = None  # token strings of that sample (tokenizer= / tokens=)
        self._layouts: dict = {}
        self._equations: dict | None = None
        self._param_cache: dict[str, tuple[str, dict]] = {}
        self._graph_id = None

    # ---------------------------------------------------------------- levels
    def levels(self) -> list[dict]:
        g = self.net.graph
        if self._graph_id != id(g):
            self._layouts, self._equations, self._graph_id = {}, None, id(g)
        if self._layouts:
            return self._layouts["levels"]
        md = g.max_depth()
        depths = list(range(0, md)) + [None]
        if len(depths) > 5:
            depths = depths[:4] + [None]
        levels = []
        for d in depths:
            gd = g if d is None else g.collapse(d)
            n_compute = len(gd.compute_nodes())
            if levels and levels[-1]["n"] == n_compute:
                levels.pop()  # collapsing at this depth changes nothing: keep the deeper one
            if self.style == "expressive":
                from ..glyphs import node_size

                lay = layout(gd, size_fn=node_size, row_gap=74 if self.tensor_shapes else 46,
                             tensors=self.tensor_shapes)
            else:
                lay = layout(gd)
            nodes = []
            for n in gd.nodes.values():
                d_ = n.to_dict()
                d_["collapsed"] = bool(n.attrs.get("collapsed"))
                d_["family"] = theme.kind_family(n.kind, n.is_container or d_["collapsed"])
                d_["color"] = theme.kind_color(n.kind, n.is_container or d_["collapsed"])
                if n.id in lay["nodes"]:
                    d_["layout"] = lay["nodes"][n.id]
                d_["rep"] = self._representative(g, gd, n) if d_["collapsed"] else None
                nodes.append(d_)
            levels.append({"depth": d, "label": "full" if d is None else f"depth {d}", "n": n_compute,
                           "nodes": nodes, "edges": [{"src": e.src, "dst": e.dst,
                                                      "shape": list(e.shape) if e.shape else None,
                                                      "implicit": e.implicit} for e in gd.edges],
                           "layout": {"edges": lay["edges"], "groups": lay["groups"], "width": lay["width"],
                                      "height": lay["height"], "tensors": lay.get("tensors", [])},
                           "_graph": gd})
        # default: the most detailed level that still fits comfortably on screen
        default = 0
        for i, lv in enumerate(levels):
            if lv["n"] <= MAX_LEVEL_NODES:
                default = i
        self._layouts = {"levels": levels, "default": default}
        return levels

    def _pick_level(self, depth) -> int:
        levels = self.levels()
        if depth is None:
            return self._layouts.get("default", len(levels) - 1)
        if depth in ("all", "full"):
            return len(levels) - 1
        for i, lv in enumerate(levels):
            if lv["depth"] == depth:
                return i
        cands = [i for i, lv in enumerate(levels) if lv["depth"] is not None and lv["depth"] <= depth]
        return cands[-1] if cands else len(levels) - 1

    @staticmethod
    def _representative(full: ModelGraph, gd: ModelGraph, n: Node) -> str | None:
        members = {m for m in full.descendants(n.id) if full.nodes[m].is_compute}
        best = None
        for m in full.topo_order():
            if m in members and any(s not in members for s in full.succ(m)):
                best = m
        return best

    # ---------------------------------------------------------------- glyphs
    def glyphs(self, trace) -> dict:
        """Rendered SVG glyph per node id (data-driven: curves, weights, attention, images)."""
        if self.style != "expressive":
            return {}
        from ..glyphs import GlyphData, render

        out: dict[str, str] = {}
        for lv in self.levels():
            gd = lv["_graph"]
            D = GlyphData(self.net, trace, gd, sample=self.sample)
            for n in gd.nodes.values():
                if n.id in out or not n.is_compute:
                    continue
                out[n.id] = render(n, D)
        return out

    def glyph_meta(self) -> dict:
        if self.style != "expressive":
            return {}
        from ..glyphs import glyph_box

        out = {}
        for lv in self.levels():
            for n in lv["_graph"].nodes.values():
                if n.is_compute and n.id not in out:
                    t, box = glyph_box(n)
                    out[n.id] = {"type": t, "box": box}
        return out

    # ------------------------------------------------------------- equations
    def _painter(self):
        """Colour-code the page's equations: each layer's symbols in its family colour, tagged with an index into
        ``eqNodes`` so hovering a term can light up its layer; weighted sums coloured by sign."""
        from ...equations.expr import painted

        g = self.net.graph
        if getattr(self, "_eq_graph", None) != id(g):
            self._eq_graph, self._eq_nodes, self._eq_index = id(g), [], {}
            self._colors = {}
            for lv in self.levels():
                for n in lv["_graph"].nodes.values():
                    fam = theme.kind_family(n.kind, n.is_container or bool(n.attrs.get("collapsed")))
                    self._colors.setdefault(n.id, None if fam == "io" else theme.FAMILY_COLORS[fam])

        def tag(nid):
            if nid not in self._eq_index:
                self._eq_index[nid] = len(self._eq_nodes)
                self._eq_nodes.append(nid)
            return self._eq_index[nid]

        return painted(color_of=self._colors.get, tag_of=tag, signs=(theme.POSITIVE, theme.NEGATIVE))

    def equations(self) -> dict:
        if self._equations is not None:
            return self._equations
        with self._painter():
            out = self._equations_painted()
        self._equations = out
        return out

    def _equations_painted(self) -> dict:
        from ...equations.unroll import node_equation

        net = self.net
        out: dict[str, Any] = {"modes": {}, "byDepth": {}, "perNode": {}}
        for mode in ("system", "expanded", "compact"):
            try:
                out["modes"][mode] = net.equation(mode).to_dict()
            except Exception as exc:  # noqa: BLE001
                out["modes"][mode] = {"error": str(exc)}
        for lv in self.levels():
            if lv["depth"] is None:
                continue
            try:
                out["byDepth"][str(lv["depth"])] = net.equation("system", depth=lv["depth"]).to_dict()
            except Exception:  # noqa: BLE001
                pass
        g = net.graph
        inside = _compute_counts(g)  # a block's equation repeats everything in it: big blocks are left out
        for lv in self.levels():
            gd = lv["_graph"]
            for n in gd.nodes.values():
                if n.id in out["perNode"] or n.op in ("input", "output"):
                    continue
                if inside.get(n.id, 0) > BUDGET.block_eq:
                    continue
                eq = node_equation(net, n.id, gd if n.attrs.get("collapsed") else g)
                if eq is not None:
                    out["perNode"][n.id] = eq.to_dict()
        return out

    def _dataset(self, dataset, tokenizer, labels) -> dict | None:
        if dataset is None:
            return None
        if not hasattr(dataset, "to_dict"):
            from ...errors import OptionError

            raise OptionError(f"dataset= got a {type(dataset).__name__}.",
                              "pass the profile from net.profile(data, labels).")
        key = (id(dataset), dataset.n)
        if getattr(self, "_dataset_cache", (None,))[0] != key:  # live pages: computed once, not per refresh
            self._dataset_cache = (key, dataset.to_dict(tokenizer=tokenizer, class_names=labels))
        return self._dataset_cache[1]

    def _describe(self, trace) -> str | None:
        """The whole network in one plain sentence (see :func:`deepvis.viz.explain.describe`)."""
        from ..explain import describe

        try:
            return describe(self.net.graph, _is_batched(self.net, trace) if trace is not None else True)
        except Exception:  # noqa: BLE001 - words must never break a page
            return None

    def explanations(self, trace) -> dict:
        """Plain-language "what it does / why / look at" per node of every level (see :mod:`deepvis.viz.explain`)."""
        from ..explain import explain

        batched = _is_batched(self.net, trace) if trace is not None else True
        key = (id(self.net.graph), batched)
        if getattr(self, "_explain_cache", (None, None))[0] == key:
            return self._explain_cache[1]
        out: dict[str, dict] = {}
        for lv in self.levels():
            gd = lv["_graph"]
            for n in gd.nodes.values():
                if n.id in out or (not n.is_compute and not n.attrs.get("collapsed") and n.op != "container"):
                    continue
                e = explain(n, gd, batched)
                if e:
                    out[n.id] = e
        self._explain_cache = (key, out)
        return out

    def scalars(self, trace) -> dict:
        """Per convolution: one output value unrolled with the real weights (most active channel & position)."""
        if trace is None:
            return {}
        from ...core.kinds import CONV_KINDS
        from ...equations.unroll import _channels_first, conv_scalar_equation

        net, out = self.net, {}
        for n in net.graph.compute_nodes():
            if n.kind not in CONV_KINDS or "weight" not in n.roles or n.id not in trace.acts:
                continue
            try:
                z = _channels_first(net, n, trace.acts[n.id], self.sample)
                ch = int(np.argmax(np.abs(z).reshape(z.shape[0], -1).max(axis=1)))
                with self._painter():  # the terms coloured by their weight's sign, like the kernel's cells
                    eq = conv_scalar_equation(net, n.id, [], ch, None, top_k=5, trace=trace, sample=self.sample)
                    d = eq.to_dict()
                d["terms"] = len(eq.terms)
                out[n.id] = d
            except Exception:  # noqa: BLE001 - decoration only
                continue
        return out

    # ---------------------------------------------------------------- params
    def params(self) -> dict:
        net = self.net
        g = net.graph
        owner: dict[str, tuple[Node, str]] = {}
        for n in g.compute_nodes():
            for local, info in n.params.items():
                owner.setdefault(info.path, (n, local))
        out = {}
        for p in net.backend.param_paths():
            arr = net.backend.get_param(p)
            dig = _digest(arr) + f"/{BUDGET.heat}"
            cached = self._param_cache.get(p)
            if cached and cached[0] == dig:
                out[p] = cached[1]
                continue
            node, local = owner.get(p, (None, p.split(".")[-1]))
            pay = param_payload(arr, node, local)
            pay["node"] = node.id if node is not None else None
            orig = net.history.original.get(p)
            pay["edited"] = bool(orig is not None and not np.array_equal(orig, arr))
            self._param_cache[p] = (dig, pay)
            out[p] = pay
        return out

    # ------------------------------------------------------------ activations
    def activations(self, trace: "Trace | None") -> dict:
        if trace is None:
            return {}
        net = self.net
        g = net.graph
        batched = _is_batched(net, trace)
        out = {}
        for nid, arr in trace.acts.items():
            node = g.nodes.get(nid)
            fa = net.backend.feature_axis(node) if node is not None else -1
            try:
                pay = act_payload(arr, fa, self.sample, batched)
            except Exception:  # noqa: BLE001
                continue
            if nid in trace.grads:
                pay["grad"] = _finite_stats(trace.grads[nid])
                try:
                    pay["gradView"] = grad_view(trace.grads[nid], fa, self.sample, batched,
                                                saliency=node is not None and node.op == "input")
                except Exception:  # noqa: BLE001 - decoration only
                    pass
            if nid in trace.attention:
                att = np.asarray(trace.attention[nid])
                if att.ndim == 4:
                    att = pick_sample(att, self.sample)
                if att.ndim == 2:
                    att = att[None]
                heads = [heat(att[h], False, 48, "query", "key") for h in range(min(att.shape[0], 8))]
                tok = getattr(self, "tokens", None)
                for hm in heads:
                    if tok and not hm.get("downsampled") and hm["rows"] == hm["cols"] == len(tok):
                        hm["rowNames"] = hm["colNames"] = tok
                pay["attention"] = {"heads": heads, "total": int(att.shape[0])}
            out[nid] = pay
        return out

    def anatomy(self, trace, max_layers: int = 12, max_heads: int = 4, side: int = 48) -> list:
        """Inside every attention layer (up to ``max_layers``): Q, K, V, scores, softmax probabilities and P·V
        per head, for the shown sample — 8-bit maps, so even long models stay light."""
        parts = getattr(trace, "attention_parts", None) if trace is not None else None
        if not parts:
            return []
        order = [n for n in self.net.graph.topo_order() if n in parts]
        if len(order) > max_layers:
            order = [order[i] for i in sorted(set(np.linspace(0, len(order) - 1, max_layers).round().astype(int)))]
        out = []
        for nid in order:
            P = parts[nid]
            pick = lambda a: np.asarray(a)[min(self.sample, np.shape(a)[0] - 1)]  # noqa: E731
            H = int(P["q"].shape[1])
            hs = range(min(H, max_heads))
            T, S, d = int(P["q"].shape[2]), int(P["k"].shape[2]), int(P["q"].shape[3])

            def maps(key, div, rl, cl):
                return [heat(np.nan_to_num(pick(P[key])[h], nan=0.0), div, side, rl, cl, quantize=True) for h in hs]

            sc0 = pick(P["scores"])[0]
            node = self.net.graph.nodes.get(nid)
            owner = node.module or (node.parent if node.op == "function" else nid) if node is not None else nid
            entry = {"id": nid, "name": owner or nid, "heads": H, "shown": len(hs), "T": T, "S": S, "d": d,
                     "q": maps("q", True, "token", "dim"), "k": maps("k", True, "token", "dim"),
                     "v": maps("v", True, "token", "dim"), "scores": maps("scores", True, "query", "key"),
                     "probs": maps("probs", False, "query", "key"), "out": maps("out", True, "token", "dim"),
                     "masked": heat(np.isnan(sc0).astype(float), False, side, quantize=True) if np.isnan(sc0).any()
                     else None}
            tok = getattr(self, "tokens", None)
            if tok and len(tok) == T == S:
                entry["tokens"] = tok
            out.append(entry)
        return out

    def attribution(self, trace, labels=None, tokenizer=None, k: int = 3, steps: int = 24) -> dict | None:
        """Why the model predicted what it did: attribution maps of the input for its top-``k`` outputs (torch)."""
        from ...errors import DeepvisError, warn
        from ...interpret.attribution import attribute

        net = self.net
        if net.framework != "torch":
            warn("attribution=True: attribution maps are available for PyTorch models only; skipped.")
            return None
        from ...core.tensor import first_array

        out = first_array(trace.output)
        if out is None:
            return None
        v = to_numpy(out).astype(np.float64)
        v = pick_sample(v, self.sample) if v.ndim >= 2 else v
        v = v.reshape(-1, v.shape[-1])[-1] if v.ndim >= 2 else v.reshape(-1)
        if v.size < 2:
            return None
        top = [int(i) for i in np.argsort(-v)[:k]]
        names = _class_names(labels, tokenizer, top, int(v.size))
        first = next((a for a in trace.args if hasattr(a, "shape")), None)
        conv = any(n.kind in CONV_KINDS for n in net.graph.compute_nodes())
        tokens = first is not None and not np.issubdtype(to_numpy(first).dtype, np.floating)
        methods = ["integrated_gradients", "saliency"] + (["grad_cam"] if conv and not tokens else [])
        maps: dict[str, list] = {}
        kind = None
        for m in methods:
            row = []
            for t in top:
                try:
                    a = attribute(net, *trace.args, method=m, target=t, sample=self.sample, steps=steps)
                except DeepvisError as exc:
                    row = None
                    warn(f"attribution ({m}) skipped: {exc.what}")
                    break
                kind = a.kind
                if a.kind == "image":
                    row.append(heat(a.map, m == "integrated_gradients", 64, "y", "x", quantize=True))
                else:
                    row.append([float(x) for x in np.asarray(a.map, dtype=np.float64).reshape(-1)[:512]])
            if row is not None:
                maps[m] = row
        if not maps:
            return None
        pay = {"kind": kind, "methods": list(maps), "maps": maps,
               "targets": [{"index": t, "label": (names[i] if names else None), "score": float(v[t])}
                           for i, t in enumerate(top)]}
        if kind == "image":
            from ..glyphs import png_uri

            img = pick_sample(to_numpy(first).astype(np.float64), self.sample)
            img = img[0] if img.shape[0] not in (3,) else img.transpose(1, 2, 0)
            lo, hi = map(float, np.percentile(img, [1, 99]))
            u = (np.clip((img - lo) / ((hi - lo) or 1), 0, 1) * 255).astype(np.uint8)
            u = np.repeat(u[..., None], 3, -1) if u.ndim == 2 else u
            pay["base"] = png_uri(u)
            pay["shape"] = list(u.shape[:2])
        elif kind == "tokens":
            pay["tokens"] = self.tokens
        return pay

    def arch(self, trace, batch: int = 1) -> dict | None:
        """Recurrent layers unrolled over time, mixture-of-experts routing, state-space Δ (torch)."""
        if self.net.framework != "torch":
            return None
        from ...errors import warn
        from ...interpret.arch import routing, ssm
        from ...interpret.recurrent import unroll_node


        out: dict = {}
        try:
            rec = {}
            for nid, n in self.net.graph.nodes.items():
                if n.op != "module" or n.kind not in ("LSTM", "GRU", "RNN"):
                    continue
                u = unroll_node(self.net, trace, nid, self.sample)
                if u is None or u["match"] is None or u["match"] > 1e-3 * max(1.0, float(np.abs(u["output"]).max())):
                    continue  # only show an unrolling that reproduces the layer (e.g. not with a given h0)
                side = 96

                def hm(m, div, lo=None, hi=None, rl="unit"):
                    h_ = heat(m.T, div, side, rl, "time", quantize=True)
                    if lo is not None:
                        h_.update(vmin=lo, vmax=hi)
                    return h_

                rec[nid] = {"kind": u["kind"], "hidden": u["hidden"], "steps": u["steps"], "match": u["match"],
                            "layers": [{"layer": L["layer"], "direction": L["direction"], "h": hm(L["h"], True),
                                        "c": hm(L["c"], True) if L["c"] is not None else None,
                                        "gates": [{"name": k, "heat": hm(v, False, 0.0, 1.0) if "gate" in k
                                                   else hm(v, True)} for k, v in L["gates"].items()]}
                                       for L in u["layers"][:8]]}
            if rec:
                out["recurrent"] = rec
            moe = []
            for r in routing(self.net, trace, self.sample, batch):
                T = min(len(r["probs"]), 128)
                moe.append({"router": r["router"], "block": r["block"], "experts": r["experts"], "k": r["k"],
                            "probs": heat(r["probs"][:T], False, 128, "token", "expert", quantize=True),
                            "chosen": r["chosen"][:T].tolist(), "load": r["load"].tolist(), "tokens": T,
                            "tokenNames": self.tokens[:T] if self.tokens and len(self.tokens) == len(r["probs"]) else None})
            if moe:
                out["moe"] = moe
            sm = [{"mixer": m["mixer"], "delta": heat(m["delta"], False, 96, "token", "channel", quantize=True),
                   "perToken": [float(v) for v in m["per_token"][:512]],
                   "tokenNames": self.tokens if self.tokens and len(self.tokens) == len(m["per_token"]) else None}
                  for m in ssm(self.net, trace, self.sample)]
            if sm:
                out["ssm"] = sm
            from ...interpret.arch import graph_input

            gi = graph_input(self.net, trace)
            if gi is not None:
                out["graph"] = gi
        except Exception as exc:  # noqa: BLE001 - never lose the page over an extra view
            warn(f"an architecture view was skipped: {type(exc).__name__}: {exc}")
        return out or None

    def compare(self, trace, other, names=None, labels=None, tokenizer=None) -> dict:
        """This network against another version of it (a model, a state_dict or a checkpoint path)."""
        from ...interpret.compare import compare, other_net

        net_b = other_net(self.net, other)
        c = compare(self.net, net_b, trace.args, self.sample)
        a, b = (list(names) + ["A", "B"])[:2] if names else ("A", "B")
        if not names and isinstance(other, str):
            b = other.replace("\\", "/").rsplit("/", 1)[-1]
        return {"names": [a, b], "layers": c["layers"],
                "outputs": [_output_payload(c["trace_a"].output, None, self.sample, labels, tokenizer),
                            _output_payload(c["trace_b"].output, None, self.sample, labels, tokenizer)]}

    def interpret(self, trace, heads: bool | None = None) -> dict | None:
        """Across attention layers: rollout (where each position draws from) and head importance (ablation)."""
        from ...interpret.attention import attention_layers, head_importance, rollout

        if not getattr(trace, "attention_parts", None):
            return None
        net = self.net
        out: dict = {}
        try:
            r = rollout(net, trace, self.sample)
        except Exception:  # noqa: BLE001 - an odd attention layout: skip the rollout, keep the rest
            r = None
        if r is not None:
            M = r["matrix"]
            T = M.shape[0]
            cls_grid = not self.tokens and T > 4 and round((T - 1) ** 0.5) ** 2 == T - 1  # a ViT: CLS + patches
            row = 0 if (cls_grid or (self.tokens and self.tokens[0].strip() in ("[CLS]", "<s>", "<|endoftext|>"))) \
                else T - 1
            out["rollout"] = {"matrix": heat(M, False, 64, "position", "token", quantize=True), "row": row,
                              "from": [float(x) for x in M[row]], "layers": len(r["layers"]),
                              "tokens": self.tokens if self.tokens and len(self.tokens) == T else None}
        n_heads = sum(int(np.shape(trace.attention_parts[nid]["q"])[1]) for nid in attention_layers(net, trace))
        if net.framework == "torch" and (heads or (heads is None and n_heads <= 48)) and heads is not False:
            try:
                hi = head_importance(net, trace.args, sample=self.sample)
            except Exception as exc:  # noqa: BLE001 - never lose the page over it
                from ...errors import warn

                warn(f"head importance skipped: {type(exc).__name__}: {exc}")
                hi = None
            if hi is not None:
                out["heads"] = hi
        return out or None

    def _lens(self, trace, tokenizer, lens_fn=None) -> dict | None:
        """Logit lens for language models (an unembedding after repeated blocks, per-position logits): torch models
        find their head; any model with ``lens_fn``."""
        net = self.net
        if trace is None or (net.framework != "torch" and lens_fn is None):
            return None
        from ...core.tensor import first_array
        from ...monitor.lens import find_blocks, find_head, logit_lens

        out = first_array(trace.output)
        if out is None or getattr(out, "ndim", 0) != 3 or (out.shape[-1] <= 64 and lens_fn is None):
            return None
        if (lens_fn is None and find_head(net.graph)[1] is None) or not find_blocks(net.graph):
            return None
        try:
            return logit_lens(net, *trace.args, sample=self.sample, k=5, max_positions=24, keep_probs=False,
                              tokenizer=tokenizer, trace=trace, lens_fn=lens_fn).to_dict()
        except Exception as exc:  # noqa: BLE001 - a bonus view
            if lens_fn is not None:
                from ...errors import warn

                warn(f"the logit lens was skipped: lens_fn failed ({type(exc).__name__}: {exc}).")
            return None

    def _tokens(self, trace, tokenizer, tokens) -> list[str] | None:
        """Token strings of the shown sample: given explicitly, or decoded from integer inputs."""
        if tokens is not None:
            t = list(tokens)
            if t and isinstance(t[0], (list, tuple)):
                t = list(t[min(self.sample, len(t) - 1)])
            return [str(x) for x in t]
        if tokenizer is None or trace is None:
            return None
        ins = [to_numpy(a) for a in trace.args if hasattr(a, "shape")]
        if not ins or ins[0].dtype.kind not in "iu":
            return None
        ids = pick_sample(ins[0], self.sample) if ins[0].ndim >= 2 else ins[0]
        try:
            return token_strings(tokenizer, ids)
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------ neuron view
    def neurons(self, trace: "Trace | None") -> dict | None:
        try:
            with self._painter():  # each term coloured by its weight's sign: the same colours as the lines
                return neuron_view(self.net, trace, sample=self.sample)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    # ------------------------------------------------------------------ all
    def build(self, args=None, trace: "Trace | None" = None, monitor: "Monitor | None" = None,
              focus: str | None = None, highlight=None, live: bool = False, baseline=None,
              interventions: list | None = None, ui: dict | None = None, training=None, sample: int = 0,
              grads: bool = False, loss_fn=None, tokenizer=None, labels=None, tokens=None, diffusion=None,
              attribution: bool = False, heads: bool | None = None, compare=None, compare_names=None,
              lens_fn=None, dataset=None) -> dict:
        global BUDGET
        from ...errors import OptionError

        net = self.net
        if diffusion is not None and not hasattr(diffusion, "to_dict"):
            raise OptionError(f"diffusion= got a {type(diffusion).__name__}.",
                              "pass the run returned by deepvis.diffusion.sample(model, shape, betas).")
        shapes_only = not net.backend.has_data  # a meta-device model: the architecture, no numbers
        if shapes_only:
            args, trace = None, None
        if trace is None and args is not None:
            trace = net.trace(*args, grads=grads, loss_fn=loss_fn)
        n_batch = batch_size(net, trace)
        self.sample = max(0, min(int(sample), n_batch - 1))
        tokenizer = tokenizer if tokenizer is not None else getattr(net, "tokenizer", None)
        labels = labels if labels is not None else getattr(net, "labels", None)
        self.tokens = self._tokens(trace, tokenizer, tokens)
        BUDGET = Budget.for_nodes(len(net.graph.compute_nodes()))
        levels = self.levels()
        n_params = net.num_params()
        acts = self.activations(trace)
        glyph_set: list[str] = []  # each distinct SVG once: a 7B model's 1,300 glyphs are ~70 different drawings
        glyph_ids = _dedupe(self.glyphs(trace), glyph_set)
        tensor_set: list[str] = []
        payload = {
            "version": 1,
            "meta": {"name": net.name, "framework": net.framework, "params": n_params,
                     "nodes": len(net.graph.compute_nodes()), "live": live,
                     "input": [list(a.shape) for a in (args or (net.sample_input if shapes_only else None) or ())
                               if hasattr(a, "shape")],
                     "shapesOnly": shapes_only, "channelsLast": bool(net.backend.channels_last),
                     "edits": len(net.history), "sample": self.sample, "batch": n_batch,
                     "describe": self._describe(trace)},
            "theme": theme.as_dict(),
            "levels": _shared_svgs([{k: v for k, v in lv.items() if k != "_graph"} for lv in levels], tensor_set),
            "tensorSet": tensor_set,
            "defaultLevel": self._pick_level((ui or {}).get("depth")),
            "glyphs": glyph_ids,
            "glyphSet": glyph_set,
            "glyphMeta": self.glyph_meta(),
            "ui": {"style": self.style, "tensorShapes": self.tensor_shapes, **{k: v for k, v in (ui or {}).items()
                                                                               if v is not None}},
            "params": self.params() if not shapes_only else {},
            "acts": acts,
            "equations": self.equations(),
            "explain": self.explanations(trace),
            "scalars": self.scalars(trace),
            "neurons": self.neurons(trace) if not shapes_only else None,
            "monitor": monitor.to_dict() if monitor is not None else None,
            "output": _output_payload(trace.output if trace is not None else None, baseline, self.sample, labels,
                                      tokenizer),
            "tokens": self.tokens,
            "lens": self._lens(trace, tokenizer, lens_fn),
            "anatomy": self.anatomy(trace),
            "diffusion": diffusion.to_dict() if diffusion is not None else None,
            "history": [{"label": e.edit.describe(), "applied": i < net.history.pos}
                        for i, e in enumerate(net.history.entries)],
            "focus": focus,
            "highlight": list(highlight) if highlight else [],
            "trace": {"table": _trace_table(net, trace, acts if n_batch == 1 else None)} if trace is not None else None,
            "input": input_payload(net, args if args is not None else (trace.args if trace is not None else None),
                                   trace, self.sample) if live else None,
            "interventions": interventions or [],
            "training": (training.to_dict(labels=labels, tokenizer=tokenizer, sample=self.sample)
                         if training is not None and len(training) else None),
            "gradflow": grad_flow(net, trace) if trace is not None and (trace.grads or trace.param_grads) else None,
            "attribution": self.attribution(trace, labels, tokenizer) if attribution and trace is not None else None,
            "interpret": self.interpret(trace, heads) if trace is not None else None,
            "compare": (self.compare(trace, compare, compare_names, labels, tokenizer)
                        if compare is not None and trace is not None else None),
            "arch": self.arch(trace, n_batch) if trace is not None else None,
            "inputImage": input_image(trace.args if trace is not None else args, self.sample),
            "dataset": self._dataset(dataset, tokenizer, labels),
        }
        payload["eqNodes"] = list(getattr(self, "_eq_nodes", []))  # \\htmlData{n=k} in any equation → node id
        self._check_request(payload, labels, tokenizer)
        return payload

    @staticmethod
    def _check_request(payload: dict, labels, tokenizer) -> None:
        """Things asked for that this model / input can't show: say why, and open something sensible."""
        from ...errors import warn

        ui = payload["ui"]
        start = ui.get("start")
        if start == "attention" and not payload.get("anatomy"):
            warn("start='attention', but deepvis found no attention to look inside (it reads nn.MultiheadAttention "
                 "and F.scaled_dot_product_attention); opening the graph instead.")
            ui["start"] = None
        if start == "diffusion" and not payload.get("diffusion"):
            warn("start='diffusion' needs a run: pass diffusion=deepvis.diffusion.sample(...); opening the graph.")
            ui["start"] = None
        out = payload.get("output") or {}
        if labels is not None and "labels" not in out and not any(t.get("label") for t in out.get("topk") or []):
            n = len(out.get("values") or []) or out.get("n")
            warn(f"labels has {len(labels)} names but the output has {n} values per example, so they aren't shown; "
                 "pass one name per output class.")
        if tokenizer is not None and payload.get("tokens") is None:
            warn("tokenizer= was given but the input isn't integer token ids (or the tokenizer couldn't decode "
                 "them), so tokens aren't shown; pass the ids, e.g. tokenizer(text, return_tensors='pt')['input_ids'].")


def grad_view(g: np.ndarray, feature_axis: int, sample: int, batched: bool, saliency: bool = False) -> dict:
    """The gradient ∂L/∂(tensor) drawn like the tensor itself (vector / maps / matrix), plus a saliency map
    (max |∂L/∂x| over channels) for image inputs."""
    full = act_payload(g, feature_axis, sample, batched)
    out = {k: full[k] for k in ("vector", "maps", "matrix", "top") if k in full}
    a = np.asarray(g, dtype=np.float64)
    a = pick_sample(a, sample) if batched and a.ndim >= 2 else a
    if saliency and a.ndim == 3:
        ch = (feature_axis % (a.ndim + 1)) - 1 if batched and feature_axis is not None else 0
        ch = ch if ch >= 0 else a.ndim - 1
        if a.shape[ch] > 4:  # channels-last guess failed: take the smallest axis
            ch = int(np.argmin(a.shape))
        out["saliency"] = heat(np.abs(a).max(axis=ch), False, None, "row", "col")
    elif saliency and a.ndim == 2 and min(a.shape) >= 4:
        out["saliency"] = heat(np.abs(a), False, None, "row", "col")
    return out


def grad_flow(net: "Net", trace: "Trace") -> dict:
    """Per layer, in dataflow order: mean / max |∂L/∂output| and mean / max |∂L/∂θ| (vanishing / exploding
    gradients show up as a trend along the network)."""
    g = net.graph
    rows, seen = [], set()
    for nid in g.topo_order():
        n = g.nodes[nid]
        if not n.is_compute or n.op in ("output", "param"):
            continue
        ga = trace.grads.get(nid)
        pg = []
        for i in n.params.values():
            if i.trainable and i.path in trace.param_grads and i.path not in seen:
                seen.add(i.path)
                pg.append(np.abs(np.asarray(trace.param_grads[i.path], dtype=np.float64)).ravel())
        if ga is None and not pg:
            continue
        row: dict[str, Any] = {"id": nid, "kind": n.kind, "op": n.op}
        if ga is not None and np.size(ga):
            a = np.abs(np.asarray(ga, dtype=np.float64))
            row.update(act=_num(a.mean()), actMax=_num(a.max()))
        if pg:
            w = np.concatenate(pg)
            row.update(param=_num(w.mean()), paramMax=_num(w.max()))
        rows.append(row)
    return {"loss": _num(trace.loss) if trace.loss is not None else None, "rows": rows}


def _num(v) -> float | None:
    v = float(v)
    return v if math.isfinite(v) else None


def input_payload(net: "Net", args, trace=None, sample: int = 0) -> dict | None:
    """What the live input playground needs: the first input of one sample, as an image, vector or tokens."""
    if not args or not hasattr(args[0], "shape"):
        return None
    arr = to_numpy(args[0])
    batched = _is_batched(net, trace) if trace is not None else True
    x = pick_sample(arr, sample) if batched and arr.ndim >= 2 else arr
    info: dict[str, Any] = {"shape": list(arr.shape), "dtype": str(arr.dtype), "batched": bool(batched and arr.ndim >= 2),
                            "sample": int(sample)}
    if arr.dtype.kind in "iub":
        flat = x.reshape(-1)
        info.update(kind="tokens", values=[int(v) for v in flat[:512]], n=int(flat.size))
        return info
    x = x.astype(np.float64)
    if x.ndim >= 2:
        if x.ndim >= 3:  # collapse channels: torch / ONNX channel-first, JAX / Keras channel-last
            ch = x.ndim - 1 if net.backend.channels_last else 0
            info["channels"] = int(x.shape[ch])
            info["channel_axis"] = ch
            x = x.mean(axis=ch)
            x = x.reshape(-1, x.shape[-1])
        if min(x.shape) >= 4 and max(x.shape) <= 256:
            info.update(kind="image", rows=int(x.shape[0]), cols=int(x.shape[1]), data=b64(x),
                        vmin=float(x.min()), vmax=float(x.max()))
            return info
    elif x.ndim == 1 and x.size <= 64:
        info.update(kind="vector", values=[float(v) for v in x], n=int(x.size))
        return info
    info.update(kind="other")
    return info


def _is_batched(net: "Net", trace: "Trace") -> bool:
    """Equinox models are usually called per-example (no batch axis); a graph network's input is one graph (its
    first axis is the graph's nodes, not a batch)."""
    arrs = [a for a in trace.args if hasattr(a, "shape")]
    if any(len(a.shape) == 2 and a.shape[0] == 2 and not np.issubdtype(to_numpy(a).dtype, np.floating) for a in arrs) \
            and any(len(a.shape) == 2 and np.issubdtype(to_numpy(a).dtype, np.floating) for a in arrs):
        return False
    if net.framework == "equinox":
        ins = [a for a in trace.args if hasattr(a, "shape")]
        g = net.graph
        inp = g.inputs()
        if ins and inp and inp[0].out_shapes:
            return False
    return True


def token_strings(tokenizer, ids) -> list[str]:
    """Readable strings for token ids from a HuggingFace-style tokenizer (or any ``id -> str`` callable)."""
    ids = [int(i) for i in np.asarray(ids).reshape(-1)]
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        out = [str(t) for t in tokenizer.convert_ids_to_tokens(ids)]
        return [t.replace("Ġ", " ").replace("▁", " ").replace("Ċ", "↵") for t in out]  # byte-level / sentencepiece
    if hasattr(tokenizer, "decode"):
        return [tokenizer.decode([i]) for i in ids]
    if callable(tokenizer):
        return [str(tokenizer(i)) for i in ids]
    return [str(tokenizer[i]) for i in ids]  # a list / dict of names


def _vocab_size(tokenizer) -> int | None:
    for attr in ("__len__",):
        if hasattr(tokenizer, attr):
            try:
                return len(tokenizer)
            except TypeError:
                pass
    return getattr(tokenizer, "vocab_size", None)


def _class_names(labels, tokenizer, ids, n_classes: int | None = None) -> list[str] | None:
    """Names for output classes — only when the output really is over those classes (a base model's hidden
    units must never be labeled with vocabulary tokens)."""
    if labels is not None:
        if n_classes is not None and len(labels) != n_classes:
            return None
        return [str(labels[i]) if i < len(labels) else str(i) for i in ids]
    if tokenizer is not None:
        vs = _vocab_size(tokenizer)
        if n_classes is not None and vs is not None and n_classes not in (vs, getattr(tokenizer, "vocab_size", vs)):
            return None
        try:
            return token_strings(tokenizer, ids)
        except Exception:  # noqa: BLE001
            return None
    return None


def input_image(args, sample: int = 0, max_side: int = 256) -> dict | None:
    """The first input that is a picture (a float (batch, 1 or 3, H, W) tensor) — e.g. CLIP's pixel_values after its
    token ids."""
    def is_image(a):
        sh = getattr(a, "shape", None)
        return sh is not None and len(sh) == 4 and sh[1] in (1, 3) and min(sh[2:]) >= 8 and \
            np.issubdtype(to_numpy(a).dtype, np.floating)

    first = next((a for a in (args or ()) if is_image(a)), None)
    if first is None:
        return None
    v = to_numpy(first)
    from ..glyphs import png_uri

    img = v[min(sample, v.shape[0] - 1)].astype(np.float64)
    img = img[0] if img.shape[0] == 1 else img.transpose(1, 2, 0)
    if max(img.shape[:2]) > max_side:
        f = int(np.ceil(max(img.shape[:2]) / max_side))
        img = img[::f, ::f]
    lo, hi = map(float, np.percentile(img, [1, 99]))
    u = (np.clip((img - lo) / ((hi - lo) or 1), 0, 1) * 255).astype(np.uint8)
    u = np.repeat(u[..., None], 3, -1) if u.ndim == 2 else u
    return {"uri": png_uri(u), "h": int(v.shape[2]), "w": int(v.shape[3])}


def _output_payload(out, baseline, sample: int = 0, labels=None, tokenizer=None, k: int = 10) -> dict | None:
    from ...core.tensor import first_array

    a = first_array(out) if out is not None else None
    if a is None:
        return None
    v = to_numpy(a).astype(np.float64)
    v = pick_sample(v, sample) if v.ndim >= 2 else v
    if v.ndim == 3 and min(v.shape[-2:]) >= 4 and v.shape[0] <= 4:  # an image (C, H, W): show it as one
        from ..glyphs import png_uri

        img = v[0] if v.shape[0] != 3 else v.transpose(1, 2, 0)
        lo, hi = map(float, np.percentile(img, [1, 99]))
        u = (np.clip((img - lo) / ((hi - lo) or 1), 0, 1) * 255).astype(np.uint8)
        u = np.repeat(u[..., None], 3, -1) if u.ndim == 2 else u
        f = max(1, 128 // max(u.shape[:2]))
        return {"n": int(v.size), "values": None, "image": png_uri(np.kron(u, np.ones((f, f, 1), np.uint8))),
                "shape": list(v.shape)}
    named = _class_names(labels, tokenizer, [0], int(v.shape[-1])) is not None if v.ndim >= 1 else False
    if v.ndim >= 2 and (named or v.shape[-1] > 1000):  # (positions, classes), e.g. a language model: last position
        pos = int(np.prod(v.shape[:-1]) - 1)
        v = v.reshape(-1, v.shape[-1])[-1]
    elif v.ndim == 2 and v.size > 64:  # (tokens, features): hidden states, not class scores — a heat map
        return {"n": int(v.size), "values": None, "shape": list(v.shape),
                "heat": heat(v, None, 64, "position", "feature")}
    else:
        pos = None
        v = v.reshape(-1)
    if v.size > 64:  # too many classes for bars: the top k, as probabilities
        prob = bool(np.all(v >= -1e-6) and abs(v.sum() - 1) < 1e-3)
        p = v if prob else np.exp(v - v.max()) / np.exp(v - v.max()).sum()
        top = np.argsort(-p)[:k]
        names = _class_names(labels, tokenizer, top, int(v.size))
        return {"n": int(v.size), "values": None, "position": pos, "softmaxed": not prob,
                "labelKind": "class" if labels is not None else "token",
                "topk": [{"id": int(i), "p": float(p[i]), "label": names[j] if names else None}
                         for j, i in enumerate(top)]}
    prob = bool(np.all(v >= -1e-6) and np.all(v <= 1 + 1e-6) and abs(v.sum() - 1) < 1e-3)
    res = {"n": int(v.size), "values": [float(x) for x in v], "probabilities": prob}
    names = _class_names(labels, tokenizer, range(v.size), int(v.size)) if labels is not None else None
    if names:
        res["labels"] = names
    if baseline is not None:
        b = to_numpy(first_array(baseline)).astype(np.float64)
        b = (pick_sample(b, sample) if b.ndim >= 2 else b).reshape(-1)
        if b.size == v.size:
            res["baseline"] = [float(x) for x in b]
    return res


def _trace_table(net: "Net", trace: "Trace", known: dict | None = None) -> list[dict]:
    """Per-node statistics of the traced activations. ``known``: the page's activation payloads, whose stats are the
    same numbers when the batch holds one example (no second pass over every tensor)."""
    g = net.graph
    rows = []
    for nid in g.topo_order():
        if nid not in trace.acts:
            continue
        s = known[nid]["stats"] if known and nid in known and "stats" in known[nid] else _finite_stats(trace.acts[nid])
        rows.append({"node": nid, "kind": g.nodes[nid].kind, "shape": list(trace.acts[nid].shape), **s})
    return rows


# ---------------------------------------------------------------- neuron view
def _dense_chain(net: "Net") -> list[tuple[str, str, str | None, list[str]]]:
    """Longest chain of Linear layers linked by elementwise ops.

    Returns [(dense_id, source_kind, source_ref, ops_after)] from input side to output side.
    """
    from ...equations.rules import SymbolTable
    from ...equations.unroll import _NeuronNamer

    g = net.graph
    namer = _NeuronNamer(net, g, SymbolTable(g))
    denses = [n for n in g.topo_order() if g.nodes[n].kind == "Linear" and "weight" in g.nodes[n].roles
              and len(net.backend.param_info(g.nodes[n].params[g.nodes[n].roles["weight"].name].path).shape) == 2]
    best: list = []
    for d in reversed(denses):
        chain = []
        cur = d
        while cur is not None:
            kind, ref, ops = namer.source(cur)
            chain.append((cur, kind, ref, ops))
            cur = ref if kind == "neuron" else None
        chain.reverse()
        if len(chain) > len(best):
            best = chain
    return best


def _pick(n: int, values: np.ndarray | None, k: int = MAX_NEURONS) -> list[int]:
    if n <= k:
        return list(range(n))
    if values is not None and values.size == n and np.any(values):
        top = np.argsort(-np.abs(values), kind="stable")[:k]
        return sorted(int(i) for i in top)
    return sorted({int(round(i)) for i in np.linspace(0, n - 1, k)})


def neuron_view(net: "Net", trace: "Trace | None", max_neurons: int = MAX_NEURONS,
                shown: list[list[int]] | None = None, equations: bool = True, sample: int = 0) -> dict | None:
    from ...equations.unroll import scalar_equation

    chain = _dense_chain(net)
    if not chain:
        return None
    g = net.graph
    batched = _is_batched(net, trace) if trace is not None else True

    def vec(nid: str | None) -> np.ndarray | None:
        if trace is None or nid is None or nid not in trace.acts:
            return None
        a = np.asarray(trace.acts[nid], dtype=np.float64)
        if batched and a.ndim >= 2:
            a = pick_sample(a, sample)
        return a.reshape(-1)

    def weight(nid: str) -> np.ndarray:
        node = g.nodes[nid]
        r = node.roles["weight"]
        w = net.backend.get_param(node.params[r.name].path).astype(np.float64)
        return np.transpose(w, (r.out_axis % 2, r.in_axis % 2)) if r.in_axis is not None else w

    first_dense, src_kind, src_ref, _ = chain[0]
    W0 = weight(first_dense)
    layers = []
    src_vals = vec(src_ref)
    in_label = "x" if src_kind == "input" else "h"
    layers.append({"node": src_ref, "label": in_label, "size": int(W0.shape[1]), "values": None, "shown": None})
    for i, (d, kind, ref, ops) in enumerate(chain):
        nxt_ops = chain[i + 1][3] if i + 1 < len(chain) else []
        post = nxt_ops[-1] if nxt_ops else None
        if i == len(chain) - 1:  # final layer: follow elementwise/softmax ops to the output
            from ...equations.unroll import _walk_back

            outs = [e.src for e in g.edges if g.nodes[e.dst].op == "output"]
            if outs:
                p, dd = _walk_back(g, outs[0])
                if dd == d and p:
                    post = p[0]
        layers.append({"node": d, "post": post, "label": f"a^{{({i + 1})}}", "size": int(weight(d).shape[0]),
                       "values": None, "shown": None})
    vals = [src_vals] + [vec(l_["post"] or l_["node"]) for l_ in layers[1:]]
    for i, (l_, v) in enumerate(zip(layers, vals)):
        l_["shown"] = list(shown[i]) if shown is not None else _pick(l_["size"], v, max_neurons)
        if v is not None and v.size == l_["size"]:
            l_["values"] = [float(v[i]) for i in l_["shown"]]
            l_["range"] = [float(np.min(v)), float(np.max(v))]
    links = []
    for i, (d, *_rest) in enumerate(chain):
        W = weight(d)
        rows, cols = layers[i + 1]["shown"], layers[i]["shown"]
        sub = W[np.ix_(rows, cols)]
        links.append({"node": d, "rows": len(rows), "cols": len(cols), "w": b64(sub),
                      "wmax": _abs_max(W)})
    image = None
    n_in = layers[0]["size"]
    side = int(round(math.sqrt(n_in)))
    if side * side == n_in and side >= 8 and src_vals is not None and src_vals.size == n_in:
        image = {"side": side, "data": b64(src_vals)}
    eqs, incoming = {}, {}
    for li in range(1, len(layers) if equations else 1):
        d = layers[li]["node"]
        W = weight(d)
        for j in layers[li]["shown"]:
            try:
                eq = scalar_equation(net, node=layers[li].get("post") or d, neuron=j, top_k=4, trace=trace)
                eqs[f"{li}:{j}"] = eq.to_dict()
            except Exception:  # noqa: BLE001
                pass
            if li == 1 and side * side == n_in and side >= 8:
                incoming[f"{li}:{j}"] = heat(W[j].reshape(side, side), True)
    return {"layers": layers, "links": links, "image": image, "equations": eqs, "incoming": incoming,
            "chain": [c[0] for c in chain]}
