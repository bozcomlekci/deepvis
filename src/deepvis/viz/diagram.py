"""Static architecture diagrams (SVG) with the same expressive glyphs as the dashboard.

A single self-contained SVG — for papers, slides, READMEs (GitHub renders it) — or the same figure wrapped
in a minimal HTML page. Built entirely in Python; no browser needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from xml.sax.saxutils import escape

from .. import theme as T
from .glyphs import GLYPH_CSS, GlyphData, node_size, render
from .html.layout import layout, transpose_layout

if TYPE_CHECKING:  # pragma: no cover
    from ..net import Net


def _pick_graph(net: "Net", depth):
    g = net.graph
    if depth in (None, "auto"):  # the most detailed view that still fits a figure
        best = g
        for d in range(g.max_depth(), -1, -1):
            gd = g if d == g.max_depth() else g.collapse(d)
            best = gd
            if len(gd.compute_nodes()) <= 48:
                break
        return best
    if depth in ("all", "full"):
        return g
    return g.collapse(int(depth))


def diagram_svg(net: "Net", args: tuple | None = None, depth=None, direction: str = "TB", title: str | None = None,
                subtitle: str | None = None, tensor_shapes: bool = True, legend: bool = True,
                background: bool = True, sample: int = 0) -> str:
    """Render the network as one SVG figure.
    ``direction`` "TB" (top → bottom) or "LR" (left → right); ``depth`` collapses containers (None = automatic,
    "all" = every layer); ``args`` (sample inputs) make the glyphs data-driven (activation curves over real inputs,
    input images, attention maps, output bars).
    """
    if direction not in ("TB", "LR"):
        raise ValueError("direction must be 'TB' or 'LR'")
    g = _pick_graph(net, depth)
    trace = net.trace(*args) if args else None
    D = GlyphData(net, trace, g, sample=sample)
    lr = direction == "LR"
    size_fn = (lambda n: node_size(n)[::-1]) if lr else node_size
    lay = layout(g, size_fn=size_fn, row_gap=86 if tensor_shapes else 50, tensors=tensor_shapes)
    if lr:
        lay = transpose_layout(lay)
        from .glyphs import tensor_glyph

        for tm in lay.get("tensors", []):  # labels under the blocks so they never sit on the arrow
            tg = tensor_glyph(g.nodes[tm["src"]].out_shapes[0], label_below=True)
            if tg:
                tm.update(tg)
    head_h = 70 if (title is not False) else 0
    W = max(lay["width"], 420)
    rows: list[list[tuple[str, float]]] = []  # legend items, wrapped to the figure's width
    if legend:
        fams = []
        for n in g.compute_nodes():
            if n.op in ("input", "output"):
                continue
            f = T.kind_family(n.kind, bool(n.attrs.get("collapsed")))
            if f not in fams:
                fams.append(f)
        x = 28.0
        for f in fams:
            w = 26 + len(T.FAMILY_LABELS.get(f, f)) * 6.3
            if not rows or x + w > W - 20:
                rows.append([])
                x = 28.0
            rows[-1].append((f, x))
            x += w
    leg_h = (34 + 18 * (len(rows) - 1)) if rows else 0
    H = lay["height"] + head_h + leg_h
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
             f'font-family="CMU Sans Serif, Inter, system-ui, sans-serif">',
             "<defs><marker id=\"dvarrow\" viewBox=\"0 0 10 10\" refX=\"9\" refY=\"5\" markerWidth=\"7\" "
             f"markerHeight=\"7\" orient=\"auto-start-reverse\"><path d=\"M0 1 L9 5 L0 9z\" fill=\"{T.GREY_C}\"/></marker>"
             "</defs>",
             f"<style>{GLYPH_CSS}.dt{{fill:{T.GREY_A};font:24px 'CMU Serif',Georgia,serif}}"
             f".ds{{fill:{T.GREY_C};font:12px 'CMU Typewriter Text',Menlo,monospace}}"
             f".dg{{fill:{T.GREY_C};font:italic 13px 'CMU Serif',Georgia,serif}}"
             f".dgi{{fill:{T.GREY_D};font:10.5px 'CMU Typewriter Text',Menlo,monospace}}"
             f".dl{{fill:{T.GREY_B};font:11px 'CMU Sans Serif',Inter,system-ui,sans-serif}}</style>"]
    if background:
        parts.append(f'<rect width="{W}" height="{H}" fill="{T.BACKGROUND}"/>')
    if title is not False:
        n_params = net.num_params()
        t = title or net.name
        sub = subtitle if subtitle is not None else f"{net.framework} · {n_params:,} parameters"
        parts.append(f'<text x="28" y="40" class="dt">{escape(t)}</text>')
        parts.append(f'<text x="28" y="60" class="ds">{escape(sub)}</text>')
    parts.append(f'<g transform="translate(0,{head_h})">')
    for gr in sorted(lay["groups"], key=lambda gr: gr["depth"]):
        parts.append(f'<rect x="{gr["x"]:.1f}" y="{gr["y"]:.1f}" width="{gr["w"]:.1f}" height="{gr["h"]:.1f}" rx="14" '
                     f'fill="{T.WHITE}" fill-opacity="0.02" stroke="{T.GREY_D}"/>')
        parts.append(f'<text x="{gr["x"] + 12:.1f}" y="{gr["y"] + 16:.1f}" class="dg">{escape(gr["type"])} '
                     f'<tspan class="dgi">{escape(gr["id"])}</tspan></text>')
    for e in lay["edges"]:
        pts = e["points"]
        d = f"M {pts[0][0]} {pts[0][1]}"
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if lr:
                dx = (x1 - x0) / 2
                d += f" C {x0 + dx} {y0}, {x1 - dx} {y1}, {x1} {y1}"
            else:
                dy = (y1 - y0) / 2
                d += f" C {x0} {y0 + dy}, {x1} {y1 - dy}, {x1} {y1}"
        dash = ' stroke-dasharray="2 6" opacity="0.45"' if e.get("far") else ""
        parts.append(f'<path d="{d}" fill="none" stroke="{T.GREY_D}" stroke-width="1.5"{dash} '
                     'marker-end="url(#dvarrow)"/>')
    for tm in lay.get("tensors", []):
        parts.append(f'<g transform="translate({tm["x"]:.1f},{tm["y"]:.1f})">{tm["svg"]}</g>')
    for nid, v in lay["nodes"].items():
        node = g.nodes[nid]
        parts.append(f'<g transform="translate({v["x"] - v["w"] / 2:.1f},{v["y"] - v["h"] / 2:.1f})">'
                     f'{render(node, D)}</g>')
    parts.append("</g>")
    for r, row in enumerate(rows):
        y = H - 14 - 18 * (len(rows) - 1 - r)
        for f, x in row:
            label = T.FAMILY_LABELS.get(f, f)
            parts.append(f'<rect x="{x:.1f}" y="{y - 9}" width="10" height="10" rx="2.5" fill="{T.FAMILY_COLORS[f]}"/>')
            parts.append(f'<text x="{x + 15:.1f}" y="{y}" class="dl">{escape(label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def diagram_html(svg: str, title: str) -> str:
    return (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>{escape(title)}</title>"
            f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<style>html,body{{margin:0;background:{T.BACKGROUND}}}body{{display:flex;justify-content:center;"
            f"padding:24px}}svg{{max-width:100%;height:auto}}</style></head><body>{svg}</body></html>")


def svg_to_file(svg: str, path: str, scale: float = 2.0) -> str:
    """The figure as a PNG (``scale`` × its size, for sharp slides) or a PDF (vector, for papers), via cairosvg."""
    import os

    try:
        import cairosvg
    except (ImportError, OSError) as exc:  # OSError: cairosvg installed, the cairo library not
        from ..errors import OptionError

        raise OptionError(f"PNG and PDF figures need cairosvg and the cairo library ({type(exc).__name__}: {exc}).",
                          "pip install cairosvg (or deepvis[figures]); cairo itself: brew install cairo / "
                          "apt install libcairo2 / conda install -c conda-forge cairo. Or save .svg and convert it.") \
            from exc
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = svg.encode("utf-8")
    if path.lower().endswith(".pdf"):
        cairosvg.svg2pdf(bytestring=data, write_to=path)
    else:
        cairosvg.svg2png(bytestring=data, write_to=path, scale=scale)
    return path
