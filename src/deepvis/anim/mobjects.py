"""Manim building blocks in the 3Blue1Brown style."""

from __future__ import annotations

import shutil

import numpy as np
from manim import (
    DOWN, Circle, Line, MarkupText, MathTex, Rectangle, RoundedRectangle, Square, Text, VGroup, config,
)

from .. import theme

BG = theme.BACKGROUND
HAS_TEX = bool(shutil.which("latex") and shutil.which("dvisvgm"))


def _pick_font(candidates) -> str | None:
    try:
        import manimpango

        have = set(manimpango.list_fonts())
        for c in candidates:
            if c in have:
                return c
    except Exception:  # noqa: BLE001
        pass
    return None


SERIF = _pick_font(["CMU Serif", "Latin Modern Roman", "Palatino", "Georgia"])
SANS = _pick_font(["CMU Sans Serif", "Helvetica Neue", "Inter"])


def wrap_words(s: str, width: int) -> str:
    """Break a sentence into lines of at most ``width`` characters (at spaces)."""
    lines, cur = [], ""
    for w in s.split():
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    return "\n".join(lines + ([cur] if cur else []))


def caption_text(s: str, size: float = 21, color: str = theme.GREY_A, max_width: float | None = None):
    """One narration sentence for the bottom caption zone: wrapped to two lines at most, centred."""
    limit = max_width or (config.frame_width - 1.6)
    t = text(wrap_words(s, 78), size, color, line_spacing=0.9)
    if t.width > limit:
        t.scale_to_fit_width(limit)
    return t


def text(s: str, size: float = 28, color: str = theme.TEXT, serif: bool = True, **kw) -> Text:
    font = SERIF if serif else SANS
    return Text(s, font=font or "", font_size=size, color=color, **kw) if font else Text(s, font_size=size, color=color, **kw)


def math(latex: str, plain: str, size: float = 34, color: str = theme.TEXT, max_width: float | None = None):
    """Real TeX when ``latex``+``dvisvgm`` are installed; otherwise Pango markup in CMU Serif
    (italic variables, true super/subscripts), and plain Unicode text as a last resort.

    Colour-coded LaTeX (see :func:`deepvis.equations.expr.painted`) keeps its colours on every path: real TeX
    colours the isolated sub-expressions, Pango gets coloured spans."""
    m = None
    if HAS_TEX:
        from .texlite import split_painted

        tex, colors = split_painted(latex) if ("\\textcolor" in latex or "\\htmlData" in latex) else (latex, {})
        for cmap in ((colors, {}) if colors else ({},)):  # coloured first; plain if isolating the parts fails
            try:
                kw = {"substrings_to_isolate": list(cmap), "tex_to_color_map": cmap} if cmap else {}
                m = MathTex(tex, font_size=size, color=color, **kw)
                break
            except Exception:  # noqa: BLE001 - fall back if a TeX package is missing
                m = None
    if m is None:
        try:
            from .texlite import latex_to_pango

            kw = {"font": SERIF} if SERIF else {}
            m = MarkupText(latex_to_pango(latex), font_size=size * 0.82, color=color, **kw)
        except Exception:  # noqa: BLE001
            m = text(plain, size * 0.82, color)
    limit = max_width or (config.frame_width - 1.0)
    if m.width > limit:
        m.scale_to_fit_width(limit)
    return m


def weight_style(w: float, wmax: float, max_stroke: float = 3.2) -> dict:
    t = min(abs(w) / (wmax or 1.0), 1.0)
    return {"color": theme.POSITIVE if w >= 0 else theme.NEGATIVE, "stroke_width": 0.3 + max_stroke * t,
            "stroke_opacity": 0.07 + 0.85 * t}


def neuron_fill(v: float | None, rng) -> tuple[str, float]:
    if v is None or rng is None:
        return BG, 1.0
    lo, hi = rng
    if lo < 0:  # signed: diverging color, brightness by magnitude
        m = max(abs(lo), abs(hi)) or 1.0
        return theme.colormap([v], vmin=-m, vmax=m)[0], 1.0
    m = hi or 1.0
    return theme.WHITE, float(np.clip(v / m, 0, 1)) * 0.95 + 0.03


class NetworkDiagram(VGroup):
    """Columns of neurons joined by synapses colored by weight sign (blue +, red −)."""

    def __init__(self, data: dict, width: float = 9.0, height: float = 5.4, radius: float = 0.13, **kw):
        super().__init__(**kw)
        L = data["layers"]
        n = len(L)
        xs = np.linspace(-width / 2, width / 2, n) if n > 1 else np.array([0.0])
        max_k = max(len(l_["shown"]) for l_ in L)
        gap = min(height / max(max_k - 1, 1), 0.52)
        self.columns = VGroup()
        self.dots = VGroup()
        for li, l_ in enumerate(L):
            k = len(l_["shown"])
            ys = (np.arange(k) - (k - 1) / 2) * -gap
            col = VGroup(*[Circle(radius=radius, stroke_color=theme.GREY_B, stroke_width=2, fill_color=BG,
                                  fill_opacity=1).move_to([xs[li], y, 0]) for y in ys])
            self.columns.add(col)
            if l_["size"] > k:
                dots = text("⋮", 24, theme.GREY_C).next_to(col, DOWN, buff=0.12)
                self.dots.add(dots)
        self.links = VGroup()
        for i, W in enumerate(data["weights"]):
            wmax = float(np.abs(W).max()) or 1.0
            grp = VGroup()
            for r in range(W.shape[0]):
                for c in range(W.shape[1]):
                    a, b = self.columns[i][c], self.columns[i + 1][r]
                    line = Line(a.get_right(), b.get_left(), **weight_style(W[r, c], wmax))
                    line.w, line.rc = float(W[r, c]), (r, c)
                    grp.add(line)
            self.links.add(grp)
        self.add(self.links, self.columns, self.dots)
        self.data = data

    def fill_anims(self, li: int, values=None, rng=None):
        l_ = self.data["layers"][li]
        values = l_["values"] if values is None else values
        rng = l_.get("range") if rng is None else rng
        anims = []
        for k, c in enumerate(self.columns[li]):
            col, op = neuron_fill(None if values is None else float(values[k]), rng)
            anims.append(c.animate.set_fill(col, opacity=op))
        return anims

    def contributions(self, i: int, top: int = 40) -> list[Line]:
        """Synapses into layer i+1 carrying the largest |w·a| for the traced input."""
        vals = self.data["layers"][i]["values"]
        lines = list(self.links[i])
        if vals is None:
            key = [abs(ln.w) for ln in lines]
        else:
            key = [abs(ln.w * vals[ln.rc[1]]) for ln in lines]
        order = np.argsort(key)[::-1][:top]
        return [lines[j] for j in order]


class WeightGrid(VGroup):
    """A weight matrix as a grid of diverging-colored squares (rows = output neurons)."""

    def __init__(self, W: np.ndarray, cell: float = 0.16, max_rows: int = 24, max_cols: int = 36, **kw):
        super().__init__(**kw)
        W = np.asarray(W, dtype=float)
        W = W[:max_rows, :max_cols]
        m = float(np.abs(W).max()) or 1.0
        colors = theme.colormap(W.ravel(), vmin=-m, vmax=m)
        self.cells = VGroup()
        for r in range(W.shape[0]):
            for c in range(W.shape[1]):
                sq = Square(cell, stroke_width=0, fill_color=colors[r * W.shape[1] + c], fill_opacity=1)
                sq.move_to([c * cell, -r * cell, 0])
                self.cells.add(sq)
        self.shape_ = W.shape
        self.add(self.cells)
        self.center()

    def row(self, r: int) -> VGroup:
        n = self.shape_[1]
        return VGroup(*self.cells[r * n:(r + 1) * n])


class Block(VGroup):
    """A layer block for architecture diagrams."""

    def __init__(self, kind: str, label: str, family_color: str, width: float = 1.9, height: float = 0.62,
                 pill: bool = False, **kw):
        super().__init__(**kw)
        body = RoundedRectangle(corner_radius=height / 2 if pill else 0.12, width=width, height=height,
                                stroke_color=family_color, stroke_width=2, fill_color=family_color, fill_opacity=0.14)
        t = text(kind, 20 if not pill else 17, theme.TEXT, serif=False)
        if t.width > width * 0.86:
            t.scale_to_fit_width(width * 0.86)
        t.move_to(body)
        self.body, self.label = body, t
        self.add(body, t)
        if label and not pill:
            sub = text(label, 12, theme.GREY_C, serif=False)
            if sub.width > width * 0.9:
                sub.scale_to_fit_width(width * 0.9)
            sub.next_to(body, DOWN, buff=0.06)
            self.add(sub)


def bar_chart(values, width: float = 2.4, bar_h: float = 0.2, gap: float = 0.08, probs: bool = True,
              highlight: bool = True) -> VGroup:
    vals = np.asarray(values, dtype=float)
    top = int(np.argmax(vals))
    m = max(float(np.max(np.abs(vals))), 1e-9) if not probs else 1.0
    g = VGroup()
    for i, v in enumerate(vals):
        w = max(width * abs(v) / m, 0.02)
        r = Rectangle(width=w, height=bar_h, stroke_width=0,
                      fill_color=theme.POSITIVE if (highlight and i == top) else theme.GREY_D, fill_opacity=1)
        r.move_to([w / 2, -i * (bar_h + gap), 0])
        g.add(r)
    return g


__all__ = ["NetworkDiagram", "WeightGrid", "Block", "bar_chart", "math", "text", "weight_style", "HAS_TEX"]
