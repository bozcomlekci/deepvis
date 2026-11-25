"""Expressive, data-driven glyphs for network components (SVG fragments).

Every component type has its own shape, drawn from the model's actual numbers:

* activations plot their real transfer curve over a histogram of the inputs they received; the input
  range the data actually occupies is drawn bold (see saturation at a glance);
* dense layers are trapezoids whose top/bottom widths follow in/out features, with a tiny 3b1b-style
  network inside drawn from sampled real weights (blue +, red −);
* convolutions are 3-D slabs showing their first kernel; normalizations show the input distribution
  collapsing onto the normalized one; softmax shows its output distribution; attention shows its
  (head-averaged) attention map; pooling is a funnel; arithmetic is ⊕ ⊖ ⊙ ⊘ ⊗; …
* tensors are 3-D cuboids proportioned by their shape (inputs show the actual image on their face).

The same fragments are used by the interactive dashboard and by the static diagram export, so they look
identical everywhere. Geometry is structure-only (sizes never depend on data); only colors and curves
come from the data, which keeps layouts stable while you edit a network live.
"""

from __future__ import annotations

import base64
import math
import struct
import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable
from xml.sax.saxutils import escape

import numpy as np

from .. import theme
from ..core.graph import Node
from ..core.kinds import CONV_KINDS, ELEMENTWISE_KINDS, NORM_KINDS, TRANSPOSED_CONV_KINDS

if TYPE_CHECKING:  # pragma: no cover
    from ..net import Net

PAD = 8
GAP = 10
MIN_H = 58
T = theme


def _f(v: float) -> str:
    return f"{v:.1f}".rstrip("0").rstrip(".") if abs(v - round(v)) > 1e-9 else str(int(round(v)))


def _num(n: float) -> str:
    n = float(n)
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e4:
        return f"{n / 1e3:.0f}K"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return f"{int(n)}"


# ----------------------------------------------------------------------- png
def png_uri(rgb: np.ndarray) -> str:
    """Encode an (h, w, 3) uint8 array as a PNG data URI (no Pillow needed)."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    raw = b"".join(b"\x00" + rgb[i].tobytes() for i in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _rgb(hexes: list[str], shape) -> np.ndarray:
    arr = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in hexes], dtype=np.uint8)
    return arr.reshape(*shape, 3)


def heat_png(m: np.ndarray, diverging: bool | None = None, max_side: int = 24) -> str:
    from ..core.tensor import downsample2d

    m = np.nan_to_num(np.asarray(m, dtype=np.float64))
    if m.ndim == 1:
        m = m.reshape(1, -1)
    m, _ = downsample2d(m, max_side, max_side)
    m = np.nan_to_num(m)
    if diverging is None:
        diverging = bool(m.min() < 0 < m.max())
    if diverging:
        v = float(np.abs(m).max()) or 1.0
        cols = T.colormap(m.ravel(), T.DIVERGING_STOPS, -v, v)
    else:
        lo, hi = float(m.min()), float(m.max())
        cols = T.colormap(m.ravel(), T.SEQUENTIAL_STOPS, lo, hi if hi > lo else lo + 1)
    return png_uri(_rgb(cols, m.shape))


def image_png(img: np.ndarray, channel_axis: int | None = 0) -> str:
    """An input image (H, W) or (C, H, W)/(H, W, C) as a PNG — RGB when it has three channels."""
    a = np.nan_to_num(np.asarray(img, dtype=np.float64))
    if a.ndim == 3:
        ax = channel_axis if channel_axis is not None else 0
        a = np.moveaxis(a, ax, -1)
        if a.shape[-1] != 3:
            a = a.mean(-1)
    lo, hi = float(a.min()), float(a.max())
    a = (a - lo) / (hi - lo if hi > lo else 1.0)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=-1)
    return png_uri((a * 255).clip(0, 255).astype(np.uint8))


# -------------------------------------------------------------- activations
def _erf(x):
    return np.vectorize(math.erf)(x)


def activation_fn(node: Node) -> Callable[[np.ndarray], np.ndarray] | None:
    k, a = node.kind, node.attrs
    slope = a.get("negative_slope", 0.01)
    const = a.get("const") or []
    expo = float(const[0]) if const else None
    alpha = a.get("alpha", 1.0)
    beta = a.get("beta", 1.0)
    fns: dict[str, Callable] = {
        "ReLU": lambda x: np.maximum(x, 0),
        "ReLU6": lambda x: np.clip(x, 0, 6),
        "LeakyReLU": lambda x: np.where(x > 0, x, slope * x),
        "ELU": lambda x: np.where(x > 0, x, alpha * (np.exp(np.minimum(x, 0)) - 1)),
        "CELU": lambda x: np.where(x > 0, x, alpha * (np.exp(np.minimum(x, 0) / alpha) - 1)),
        "SELU": lambda x: 1.0507 * np.where(x > 0, x, 1.67326 * (np.exp(np.minimum(x, 0)) - 1)),
        "GELU": lambda x: 0.5 * x * (1 + _erf(x / math.sqrt(2))),
        "SiLU": lambda x: x / (1 + np.exp(-x)),
        "Sigmoid": lambda x: 1 / (1 + np.exp(-x)),
        "Tanh": np.tanh,
        "Softplus": lambda x: np.log1p(np.exp(beta * x)) / beta,
        "Mish": lambda x: x * np.tanh(np.log1p(np.exp(x))),
        "Hardtanh": lambda x: np.clip(x, a.get("min_val", -1.0), a.get("max_val", 1.0)),
        "Hardswish": lambda x: x * np.clip(x + 3, 0, 6) / 6,
        "Hardsigmoid": lambda x: np.clip(x + 3, 0, 6) / 6,
        "Softsign": lambda x: x / (1 + np.abs(x)),
        "exp": np.exp,
        "abs": np.abs,
        "square": np.square,
        "neg": np.negative,
        "erf": _erf,
        "sqrt": lambda x: np.sqrt(np.maximum(x, 0)),
        "sin": np.sin,
        "cos": np.cos,
        "log": lambda x: np.log(np.maximum(x, 1e-3)),
        "rsqrt": lambda x: 1 / np.sqrt(np.maximum(x, 1e-3)),
        "reciprocal": lambda x: 1 / np.where(np.abs(x) < 1e-3, 1e-3, x),
        "pow": (lambda x: np.power(np.abs(x) if expo % 1 else x, expo)) if expo is not None else None,
        "Identity": lambda x: x,
    }
    return fns.get(k)  # None for unknown kinds (and pow without a recorded exponent)


# ---------------------------------------------------------------- the spec
@dataclass
class Spec:
    kind: str  # glyph type
    gw: int
    gh: int
    title: str
    sub: str
    detail: str

    @property
    def text_w(self) -> float:
        return max(len(self.title) * 7.6, len(self.sub) * 6.2, len(self.detail) * 6.2, 24.0)

    @property
    def size(self) -> tuple[int, int]:
        w = PAD + self.gw + GAP + self.text_w + PAD
        h = max(self.gh + 2 * PAD, MIN_H if self.kind not in ("op",) else 44)
        return int(math.ceil(w)), int(math.ceil(h))


_GLYPH_SIZE = {
    "linear": (64, 46), "conv": (64, 50), "activation": (76, 50), "softmax": (64, 44), "norm": (72, 54),
    "attention": (52, 52), "pool": (56, 44), "op": (30, 30), "concat": (44, 40), "dropout": (52, 38),
    "reshape": (62, 38), "transpose": (42, 38), "slice": (42, 38), "embedding": (56, 44), "recurrent": (60, 44),
    "input": (64, 52), "output": (64, 48), "param": (40, 40), "block": (62, 50), "other": (40, 38),
    "split": (52, 40), "constant": (46, 36), "rotary": (46, 46), "matmul": (76, 40), "mask": (40, 40),
    "expand": (52, 40), "graph": (54, 46),
}
_OP_SYMBOL = {"add": "+", "sub": "−", "mul": "·", "div": "/", "matmul": "×", "neg": "−"}


def glyph_type(n: Node) -> str:
    if n.op == "input":
        return "input"
    if n.op == "output":
        return "output"
    if n.attrs.get("collapsed") or n.op == "container":
        return "block"
    if n.op == "param":
        return "param"
    k = n.kind
    fam = T.kind_family(k)
    if k in ("Linear", "Bilinear"):
        return "linear"
    if k in CONV_KINDS or k in TRANSPOSED_CONV_KINDS:
        return "conv"
    if k in ("Softmax", "LogSoftmax"):
        return "softmax"
    if k in NORM_KINDS or k == "Normalize":
        return "norm"
    if k in ("MultiheadAttention", "Attention"):
        return "attention"
    if k == "Dropout":
        return "dropout"
    if k in ("matmul", "einsum", "bmm"):
        return "matmul"
    if k in _OP_SYMBOL:
        return "op"
    if k in ("split", "chunk", "unbind", "tensor_split"):
        return "split"
    if k == "Constant" or k in ("arange", "linspace"):
        return "constant"
    if k == "RotaryEmbedding":
        return "rotary"
    if k in ("masked_fill", "where", "tril", "triu", "masked_fill_"):
        return "mask"
    if k in ("expand", "expand_as", "repeat", "broadcast_to", "tile", "repeat_interleave"):
        return "expand"
    if activation_fn(n) is not None and k not in ("Identity",):
        return "activation"
    if k == "Concat":
        return "concat"
    if k in ("Flatten", "Reshape", "Unflatten", "PixelShuffle", "unsqueeze", "squeeze", "view_as", "contiguous"):
        return "reshape"
    if k == "Transpose":
        return "transpose"
    if k == "getitem":
        return "slice"
    if k == "Embedding":
        return "embedding"
    if k in ("LSTM", "GRU", "RNN", "Scan"):
        return "recurrent"
    if fam == "graph":
        return "graph"
    if fam == "pool" or k in ("Upsample",):
        return "pool"
    if k in ELEMENTWISE_KINDS or fam == "activation":
        return "activation" if activation_fn(n) is not None or n.op == "module" else "other"
    return "other"


def _dims(shape) -> tuple:
    if not shape:
        return ()
    return tuple(int(s) for s in (shape[1:] if len(shape) >= 2 else shape))


def _shape_str(shape) -> str:
    d = _dims(shape)
    return "×".join(str(v) for v in d) if d else ""


def _canon_weight_shape(n: Node) -> tuple | None:
    r = n.roles.get("weight")
    if not r or r.name not in n.params:
        return None
    s = tuple(n.params[r.name].shape)
    if r.out_axis is None or len(s) < 2:
        return s
    oa, ia = r.out_axis % len(s), (r.in_axis % len(s) if r.in_axis is not None else None)
    order = [oa] + ([ia] if ia is not None else []) + [i for i in range(len(s)) if i not in (oa, ia)]
    return tuple(s[i] for i in order)


def spec(n: Node) -> Spec:
    gt = glyph_type(n)
    gw, gh = _GLYPH_SIZE[gt]
    out = n.out_shapes[0] if n.out_shapes else None
    title = n.kind
    sub = n.label or n.id
    if gt == "input":
        title, sub = "input", (n.label if n.label not in ("x", None) else "")
    elif gt == "output":
        title, sub = "output", ""
        out = n.in_shapes[0] if n.in_shapes else None
    elif gt == "block":
        title = n.type_name
    elif gt == "param":
        title = "parameter"
    elif n.op == "function":
        sub = ""
    detail = _shape_str(out)
    ws = _canon_weight_shape(n)
    a = n.attrs
    if gt == "linear" and ws and len(ws) == 2:
        detail = f"{ws[1]} → {ws[0]}"
    elif gt == "conv" and ws and len(ws) >= 3:
        k = "×".join(str(v) for v in ws[2:])
        st = a.get("stride") or a.get("strides")
        st = st[0] if isinstance(st, (list, tuple)) and st else st
        detail = f"{ws[1]}→{ws[0]} · {k}" + (f" · s{st}" if st and st != 1 else "")
    elif gt == "attention":
        h = a.get("num_heads")
        detail = (f"{h} heads · " if h else "") + _shape_str(out)
    elif gt == "dropout":
        p = a.get("p", a.get("rate"))
        detail = f"p = {p}" if p is not None else detail
    elif gt == "embedding" and ws:
        full = tuple(n.params[n.roles["weight"].name].shape)
        detail = f"{full[0]} × {full[1]}" if len(full) == 2 else detail
    elif gt == "block":
        detail = _shape_str(out)
    if n.params and gt not in ("param",) and n.n_params:
        detail = (detail + " · " if detail else "") + f"{_num(n.n_params)} params"
    if sub == title:
        sub = ""
    if sub.isdigit():  # an nn.Sequential index: say it is one
        sub = f"[{sub}]"
    if len(sub) > 24:
        sub = "…" + sub[-23:]
    return Spec(gt, gw, gh, title, sub, detail)


def node_size(n: Node) -> tuple[int, int]:
    return spec(n).size


def glyph_box(n: Node) -> tuple[str, list[float]]:
    """(glyph type, [x, y, w, h]) of the glyph area inside the node's box — for enlarged views."""
    sp = spec(n)
    w, h = sp.size
    return sp.kind, [PAD - 2, (h - sp.gh) / 2 - 2, sp.gw + 4, sp.gh + 4]


# ---------------------------------------------------------- data context
class GlyphData:
    """Access to the numbers behind a node (one sample of a trace, weights in canonical orientation)."""

    def __init__(self, net: "Net", trace=None, graph=None, sample: int = 0):
        self.net = net
        self.trace = trace
        self.sample = int(sample)
        self.graph = graph or net.graph
        self.batched = True
        if trace is not None:
            from .html.payload import _is_batched

            self.batched = _is_batched(net, trace)

    def _sample(self, a):
        if a is None:
            return None
        a = np.asarray(a, dtype=np.float64)
        return a[min(self.sample, a.shape[0] - 1)] if self.batched and a.ndim >= 2 else a

    def act(self, nid: str):
        if self.trace is None:
            return None
        return self._sample(self.trace.acts.get(nid))

    def inp(self, nid: str):
        if self.trace is None:
            return None
        v = self.trace.inputs.get(nid)
        if v is None:
            preds = self.graph.pred(nid) if nid in self.graph.nodes else []
            if preds:
                v = self.trace.acts.get(preds[0])
        return self._sample(v)

    def weight(self, n: Node):
        r = n.roles.get("weight")
        if not r or r.name not in n.params or not self.net.backend.has_data:
            return None
        w = self.net.backend.get_param(n.params[r.name].path).astype(np.float64)
        if r.out_axis is None or w.ndim < 2:
            return w
        oa, ia = r.out_axis % w.ndim, (r.in_axis % w.ndim if r.in_axis is not None else None)
        order = [oa] + ([ia] if ia is not None else []) + [i for i in range(w.ndim) if i not in (oa, ia)]
        return np.transpose(w, order)

    def attention(self, nid: str):
        if self.trace is None or nid not in self.trace.attention:
            return None
        a = np.asarray(self.trace.attention[nid], dtype=np.float64)
        if a.ndim == 4:
            a = a[min(self.sample, a.shape[0] - 1)]
        while a.ndim > 2:
            a = a.mean(0)
        return a

    def channel_axis(self, n: Node) -> int:
        fa = self.net.backend.feature_axis(n)
        return 0 if fa == 1 else -1

    def module_curve(self, n: Node, xs: np.ndarray):
        """Probe an unknown elementwise torch module with a 1-D ramp."""
        if self.net.framework != "torch" or n.op != "module":
            return None
        try:
            import torch

            m = self.net.backend.model.get_submodule(n.module or n.id)
            with torch.no_grad():
                ys = m(torch.as_tensor(xs, dtype=torch.float32))
            ys = ys.numpy() if hasattr(ys, "numpy") else None
            return ys if ys is not None and ys.shape == xs.shape else None
        except Exception:  # noqa: BLE001
            return None


# ------------------------------------------------------------ primitives
def _fop(fop) -> str:
    return f' fill-opacity="{fop:.3g}"' if fop is not None else ""


def _poly(pts, stroke, width=1.5, fill="none", opacity=1.0, extra="", fop=None) -> str:
    d = " ".join(f"{_f(x)},{_f(y)}" for x, y in pts)
    return (f'<polygon points="{d}" fill="{fill}"{_fop(fop)} stroke="{stroke}" stroke-width="{width}" '
            f'stroke-linejoin="round" opacity="{opacity}" {extra}/>')


def _line(x1, y1, x2, y2, stroke, width=1.0, opacity=1.0, extra="") -> str:
    return (f'<line x1="{_f(x1)}" y1="{_f(y1)}" x2="{_f(x2)}" y2="{_f(y2)}" stroke="{stroke}" '
            f'stroke-width="{width}" opacity="{opacity:.3g}" stroke-linecap="round" {extra}/>')


def _rect(x, y, w, h, fill="none", stroke="none", width=1.0, rx=0, opacity=1.0, extra="", fop=None) -> str:
    return (f'<rect x="{_f(x)}" y="{_f(y)}" width="{_f(w)}" height="{_f(h)}" rx="{_f(rx)}" fill="{fill}"{_fop(fop)} '
            f'stroke="{stroke}" stroke-width="{width}" opacity="{opacity:.3g}" {extra}/>')


def _path(d, stroke, width=1.5, fill="none", opacity=1.0, extra="") -> str:
    return (f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" stroke-linejoin="round" '
            f'stroke-linecap="round" opacity="{opacity:.3g}" {extra}/>')


def _image(uri, x, y, w, h, extra="") -> str:
    return (f'<image href="{uri}" x="{_f(x)}" y="{_f(y)}" width="{_f(w)}" height="{_f(h)}" '
            f'preserveAspectRatio="none" style="image-rendering:pixelated" {extra}/>')


def _text(x, y, s, cls, anchor="start") -> str:
    return f'<text x="{_f(x)}" y="{_f(y)}" class="{cls}" text-anchor="{anchor}">{escape(s)}</text>'


def cuboid(dims, max_side: float = 34.0, image: str | None = None, color: str = T.BLUE_C, face: str | None = None
           ) -> tuple[str, float, float]:
    """An oblique-projected tensor: width ∝ W, height ∝ H, depth ∝ C (log scale). Returns (svg, w, h) with the
    front face's top-left corner at (0, depth offset)."""
    d = [int(v) for v in dims if v]
    s = lambda v: 3.0 + 4.2 * math.log2(v + 1)  # noqa: E731
    if len(d) == 0:
        W, H, D = 8.0, 8.0, 3.0
    elif len(d) == 1:
        W, H, D = 5.0, s(d[0]), 3.0
    elif len(d) == 2:
        W, H, D = s(d[1]), s(d[0]), 3.0
    else:
        C, H_, W_ = d[0], int(np.prod(d[1:-1])), d[-1]
        W, H, D = s(W_), s(H_), s(C) * 0.6
    k = min(1.0, max_side / max(W + D * 0.55, H + D * 0.45, 1e-6))
    W, H, D = W * k, H * k, D * k
    ox, oy = D * 0.55, D * 0.45
    front = [(0, oy), (W, oy), (W, oy + H), (0, oy + H)]
    top = [(0, oy), (ox, 0), (W + ox, 0), (W, oy)]
    side = [(W, oy), (W + ox, 0), (W + ox, H), (W, oy + H)]
    parts = [
        _poly(top, color, 1.0, fill=color, fop=0.28),
        _poly(side, color, 1.0, fill=T.BLUE_E, fop=0.55),
    ]
    if image:
        parts.append(_image(image, 0, oy, W, H))
        parts.append(_poly(front, color, 1.0))
    else:
        parts.append(_poly(front, color, 1.0, fill=face or T.BLUE_E, fop=0.42))
    return "".join(parts), W + ox, H + oy


def tensor_glyph(shape, label_below: bool = False) -> dict | None:
    """Edge annotation: the tensor flowing along an edge as a small 3-D block plus its dims."""
    dims = _dims(shape)
    if not dims:
        return None
    svg, w, h = cuboid(dims, max_side=26)
    label = "×".join(str(v) for v in dims)
    body = f'<g transform="translate({_f(-w / 2)},{_f(-h / 2)})">{svg}</g>'
    if label_below:
        body += _text(0, h / 2 + 10, label, "gt", "middle")
    else:
        body += _text(w / 2 + 6, 4, label, "gt")
    return {"svg": body, "w": w + 8 + len(label) * 6.0, "h": h}


# -------------------------------------------------------------- the glyphs
def _g_linear(n, sp, D: GlyphData, x0, y0):
    ws = _canon_weight_shape(n) or (1, 1)
    out_f, in_f = (ws[0], ws[1]) if len(ws) >= 2 else (ws[0], ws[0])
    width = lambda v: min(sp.gw, max(16.0, 12 + 6.5 * math.log2(v + 1)))  # noqa: E731
    top, bot = width(in_f), width(out_f)
    cx, h = x0 + sp.gw / 2, sp.gh
    pts = [(cx - top / 2, y0), (cx + top / 2, y0), (cx + bot / 2, y0 + h), (cx - bot / 2, y0 + h)]
    color = T.kind_color(n.kind)
    parts = [_poly(pts, color, 1.5, fill=color, fop=0.12, extra='class=\"body\"')]
    W = D.weight(n)
    k_in, k_out = min(in_f, 6), min(out_f, 5)
    xi = [cx - top / 2 + top * (i + 0.5) / k_in for i in range(k_in)]
    xo = [cx - bot / 2 + bot * (j + 0.5) / k_out for j in range(k_out)]
    if W is not None and W.ndim == 2:
        ri = np.linspace(0, W.shape[0] - 1, k_out).round().astype(int)
        ci = np.linspace(0, W.shape[1] - 1, k_in).round().astype(int)
        sub = W[np.ix_(ri, ci)]
        m = float(np.abs(sub).max()) or 1.0
        for j in range(k_out):
            for i in range(k_in):
                w = sub[j, i]
                t = abs(w) / m
                parts.append(_line(xi[i], y0 + 6, xo[j], y0 + h - 6, T.POSITIVE if w >= 0 else T.NEGATIVE,
                                   0.4 + 1.6 * t, 0.15 + 0.8 * t))
    for x in xi:
        parts.append(f'<circle cx="{_f(x)}" cy="{_f(y0 + 6)}" r="2.2" fill="{T.GREY_A}"/>')
    for x in xo:
        parts.append(f'<circle cx="{_f(x)}" cy="{_f(y0 + h - 6)}" r="2.2" fill="{T.GREY_A}"/>')
    return "".join(parts)


def _g_conv(n, sp, D, x0, y0):
    color = T.kind_color(n.kind)
    ws = _canon_weight_shape(n) or (8, 3, 3, 3)
    cout = ws[0]
    depth = min(20.0, 6 + 2.2 * math.log2(cout + 1))
    s = sp.gh - depth * 0.7 - 2
    fx, fy = x0 + 2, y0 + depth * 0.7
    ox, oy = depth, -depth * 0.7
    front = [(fx, fy), (fx + s, fy), (fx + s, fy + s), (fx, fy + s)]
    top = [(fx, fy), (fx + ox, fy + oy), (fx + s + ox, fy + oy), (fx + s, fy)]
    side = [(fx + s, fy), (fx + s + ox, fy + oy), (fx + s + ox, fy + s + oy), (fx + s, fy + s)]
    parts = [_poly(top, color, 1.2, fill=color, fop=0.22), _poly(side, color, 1.2, fill=color, fop=0.1),
             _poly(front, color, 1.5, fill=color, fop=0.12, extra='class=\"body\"')]
    W = D.weight(n)
    kh, kw = (ws[2], ws[3]) if len(ws) >= 4 else ((ws[2], 1) if len(ws) == 3 else (3, 3))
    kh, kw = min(kh, 7), min(kw, 7)
    if W is not None and W.ndim >= 3:
        k = W[0].mean(0)
        k = k.reshape(k.shape[0], -1)[:kh, :kw]
        v = float(np.abs(k).max()) or 1.0
        cols = T.colormap(k.ravel(), T.DIVERGING_STOPS, -v, v)
    else:
        cols = [T.GREY_D] * (kh * kw)
    cell = (s - 8) / max(kh, kw)
    gx, gy = fx + (s - cell * kw) / 2, fy + (s - cell * kh) / 2
    for r in range(kh):
        for c in range(kw):
            parts.append(_rect(gx + c * cell, gy + r * cell, cell - 1, cell - 1, fill=cols[r * kw + c]))
    if n.kind in TRANSPOSED_CONV_KINDS:
        parts.append(_text(fx + s + ox - 2, fy + s + oy + 10, "ᵀ", "gd"))
    return "".join(parts)


def _data_range(x) -> float:
    if x is None or np.size(x) == 0:
        return 4.0
    v = np.asarray(x, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 4.0
    lo, hi = np.percentile(v, [1, 99])
    return float(min(10.0, max(3.0, abs(lo), abs(hi)) * 1.15))


def _g_activation(n, sp, D, x0, y0):
    color = T.kind_color(n.kind)
    w, h = sp.gw, sp.gh
    parts = [_rect(x0, y0, w, h, fill=T.BLACK, stroke=color, width=1.5, rx=6, fop=0.001, extra='class="body"')]
    xin = D.inp(n.id)
    R = _data_range(xin)
    xs = np.linspace(-R, R, 96)
    fn = activation_fn(n)
    ys = None
    if n.type_name != n.kind:  # a third-party class recognised by name (e.g. QuickGELU): probe the real module
        ys = D.module_curve(n, xs)
    if ys is None and fn is not None:
        with np.errstate(all="ignore"):
            ys = np.asarray(fn(xs), dtype=np.float64)
    if ys is None:
        ys = D.module_curve(n, xs)
    if ys is None or not np.all(np.isfinite(ys)):
        parts.append(_text(x0 + w / 2, y0 + h / 2 + 5, "f(x)", "gd", "middle"))
        return "".join(parts)
    ylo, yhi = min(float(ys.min()), 0.0), max(float(ys.max()), 0.0)
    if yhi - ylo < 1e-9:
        yhi = ylo + 1
    pad = 5
    X = lambda v: x0 + pad + (v + R) / (2 * R) * (w - 2 * pad)  # noqa: E731
    Y = lambda v: y0 + h - pad - (v - ylo) / (yhi - ylo) * (h - 2 * pad)  # noqa: E731
    # the inputs this layer actually saw
    if xin is not None and np.size(xin):
        v = np.asarray(xin).ravel()
        v = v[np.isfinite(v)]
        counts, edges = np.histogram(np.clip(v, -R, R), bins=24, range=(-R, R))
        cm = counts.max() or 1
        bw = (w - 2 * pad) / 24
        for i, c in enumerate(counts):
            if c:
                bh = (h - 2 * pad) * 0.5 * c / cm
                parts.append(_rect(X(edges[i]), y0 + h - pad - bh, bw - 0.6, bh, fill=T.GREY_C, opacity=0.35))
        p5, p95 = np.percentile(v, [5, 95])
    else:
        p5, p95 = -R, R
    parts.append(_line(X(-R), Y(0), X(R), Y(0), T.GREY_D, 1))
    parts.append(_line(X(0), y0 + pad, X(0), y0 + h - pad, T.GREY_D, 1))
    pts = " ".join(f"{_f(X(a))},{_f(Y(b))}" for a, b in zip(xs, ys))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.2" opacity="0.55"/>')
    sel = (xs >= p5) & (xs <= p95)
    if sel.sum() >= 2:  # where the data lives on the curve
        pts = " ".join(f"{_f(X(a))},{_f(Y(b))}" for a, b in zip(xs[sel], ys[sel]))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.6" '
                     f'stroke-linecap="round"/>')
    return "".join(parts)


def _g_softmax(n, sp, D, x0, y0):
    color = T.kind_color(n.kind)
    w, h = sp.gw, sp.gh
    parts = [_rect(x0, y0, w, h, fill=T.BLACK, stroke=color, width=1.5, rx=6, fop=0.001, extra='class="body"')]
    a = D.act(n.id)
    if a is not None and np.size(a):
        v = np.asarray(a, dtype=np.float64)
        v = v.reshape(-1, v.shape[-1])[-1] if v.ndim >= 2 else v.ravel()  # last row: all keys visible
        if n.kind == "LogSoftmax":
            v = np.exp(v)
    else:
        v = np.array([0.08, 0.14, 0.46, 0.2, 0.12])
    v = v[:12]
    m = float(v.max()) or 1.0
    top = int(np.argmax(v))
    bw = (w - 10) / len(v)
    for i, p in enumerate(v):
        bh = (h - 12) * max(p, 0) / m
        parts.append(_rect(x0 + 5 + i * bw + 0.8, y0 + h - 5 - bh, max(bw - 1.6, 1), bh,
                           fill=T.POSITIVE if i == top else T.GREY_C, rx=1))
    parts.append(_line(x0 + 4, y0 + h - 5, x0 + w - 4, y0 + h - 5, T.GREY_D, 1))
    return "".join(parts)


def _gauss(xs, mu, sd):
    sd = max(sd, 1e-6)
    return np.exp(-0.5 * ((xs - mu) / sd) ** 2) / sd


def _g_norm(n, sp, D, x0, y0):
    color = T.kind_color(n.kind)
    w, h = sp.gw, sp.gh
    parts = [_rect(x0, y0, w, h, fill=T.BLACK, stroke=color, width=1.5, rx=10, fop=0.001, extra='class="body"')]
    xin, yout = D.inp(n.id), D.act(n.id)
    if xin is not None and np.size(xin) and yout is not None and np.size(yout):
        mi, si = float(np.mean(xin)), float(np.std(xin))
        mo, so = float(np.mean(yout)), float(np.std(yout))
    else:
        mi, si, mo, so = 1.5, 2.2, 0.0, 1.0
    R = max(abs(mi) + 3 * si, abs(mo) + 3 * so, 3.0)
    xs = np.linspace(-R, R, 80)
    a, b = _gauss(xs, mi, si), _gauss(xs, mo, so)
    a, b = a / (a.max() or 1), b / (b.max() or 1)  # equal peaks: the eye compares widths (spread)
    X = lambda v: x0 + 5 + (v + R) / (2 * R) * (w - 10)  # noqa: E731
    Y = lambda v: y0 + h - 5 - v * (h - 12)  # noqa: E731
    hc = h - 16  # curves on top, the normalization grid below
    Y = lambda v: y0 + hc - 3 - v * (hc - 8)  # noqa: E731
    for ys, col, sw, op in ((a, T.GREY_C, 1.4, 0.8), (b, color, 2.2, 1.0)):
        pts = " ".join(f"{_f(X(x))},{_f(Y(y))}" for x, y in zip(xs, ys))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="{sw}" opacity="{op}"/>')
    parts.append(_line(x0 + 4, y0 + hc - 3, x0 + w - 4, y0 + hc - 3, T.GREY_D, 1))
    parts.append(_norm_grid(n.kind, x0 + 8, y0 + hc + 1, w - 16, 12, color))
    return "".join(parts)


def _norm_grid(kind: str, x, y, w, h, color) -> str:
    """Rows = samples / tokens, columns = features / channels; lit cells are normalized together."""
    rows, cols = 2, 8
    cw, ch = w / cols, h / rows
    out = []
    for r in range(rows):
        for c in range(cols):
            if kind == "BatchNorm":
                lit = c == 2  # one channel, across the batch
            elif kind == "GroupNorm":
                lit = r == 0 and c in (2, 3)  # a group of channels, one sample
            elif kind in ("InstanceNorm2d", "InstanceNorm1d"):
                lit = r == 0 and c == 2
            else:  # LayerNorm / RMSNorm: all features of one token
                lit = r == 0
            hollow = lit and kind == "RMSNorm"  # scaled by the root-mean-square, not centered
            fill = "none" if hollow else (color if lit else T.GREY_E)
            out.append(_rect(x + c * cw + 0.6, y + r * ch + 0.6, cw - 1.2, ch - 1.2, fill=fill,
                             stroke=color if hollow else "none", width=1.1, rx=1, opacity=0.9 if lit else 0.8))
    return "".join(out)


def _g_attention(n, sp, D, x0, y0):
    color = T.kind_color(n.kind)
    s = sp.gh
    att = D.attention(n.id)
    if att is None:
        k = 8
        att = np.exp(-np.abs(np.arange(k)[:, None] - np.arange(k)[None]) * 0.8)
    parts = [_image(heat_png(att / (att.max() or 1), diverging=False, max_side=24), x0 + 4, y0 + 4, s - 8, s - 8),
             _rect(x0, y0, s, s, stroke=color, width=1.5, rx=4, extra='class="body"')]
    return "".join(parts)


def _g_pool(n, sp, D, x0, y0):
    color = T.kind_color(n.kind)
    w, h = sp.gw, sp.gh
    glob = "Adaptive" in n.kind or n.kind in ("mean", "sum", "max")
    up = n.kind == "Upsample"
    top_w, bot_w = (w - 8, 12 if glob else (w - 8) / 2) if not up else ((w - 8) / 2, w - 8)
    cx = x0 + w / 2
    pts = [(cx - top_w / 2, y0), (cx + top_w / 2, y0), (cx + bot_w / 2, y0 + h), (cx - bot_w / 2, y0 + h)]
    parts = [_poly(pts, color, 1.5, fill=color, fop=0.1, extra='class=\"body\"')]
    nt = 4 if not up else 2
    nb = 1 if glob else (2 if not up else 4)
    ct, cb = top_w / nt, bot_w / nb
    is_max = "Max" in n.kind or n.kind == "max"
    for r in range(nt):
        for c in range(nt):
            hot = is_max and (r % 2 == c % 2 == 0)
            parts.append(_rect(cx - top_w / 2 + c * ct + 1, y0 + 2 + r * (ct * 0.35), ct - 2, ct * 0.35 - 1,
                               fill=T.POSITIVE if hot else T.GREY_D, opacity=0.9))
    for r in range(nb):
        for c in range(nb):
            parts.append(_rect(cx - bot_w / 2 + c * cb + 1, y0 + h - 2 - (nb - r) * (cb * 0.35), cb - 2,
                               cb * 0.35 - 1, fill=T.POSITIVE if is_max else T.BLUE_E))
    return "".join(parts)


def _g_op(n, sp, D, x0, y0):
    color = T.GREY_B
    r = sp.gw / 2
    cx, cy = x0 + r, y0 + r
    parts = [f'<circle cx="{_f(cx)}" cy="{_f(cy)}" r="{_f(r - 1)}" fill="{color}" fill-opacity="0.08" '
             f'stroke="{color}" stroke-width="1.6" class="body"/>']
    k = n.kind
    a = r * 0.5
    if k in ("add", "sub", "neg"):
        parts.append(_line(cx - a, cy, cx + a, cy, T.GREY_A, 2))
        if k == "add":
            parts.append(_line(cx, cy - a, cx, cy + a, T.GREY_A, 2))
    elif k == "mul":
        parts.append(f'<circle cx="{_f(cx)}" cy="{_f(cy)}" r="3" fill="{T.GREY_A}"/>')
    elif k == "div":
        parts.append(_line(cx - a * 0.8, cy + a, cx + a * 0.8, cy - a, T.GREY_A, 2))
    elif k == "matmul":
        parts.append(_line(cx - a * 0.8, cy - a * 0.8, cx + a * 0.8, cy + a * 0.8, T.GREY_A, 2))
        parts.append(_line(cx - a * 0.8, cy + a * 0.8, cx + a * 0.8, cy - a * 0.8, T.GREY_A, 2))
    return "".join(parts)


def _g_concat(n, sp, D, x0, y0, n_in=2):
    color = T.kind_color(n.kind)
    w, h = sp.gw, sp.gh
    k = max(2, min(n_in, 4))
    shades = [T.BLUE_C, T.TEAL_C, T.GOLD_C, T.MAROON_C]
    bw = (w - 6) / (k * 1.6)
    parts = []
    for i in range(k):
        parts.append(_rect(x0 + 3 + i * bw * 1.6, y0, bw, h * 0.38, fill=shades[i], opacity=0.8, rx=1.5))
    seg = (w - 6) / k
    for i in range(k):
        parts.append(_rect(x0 + 3 + i * seg, y0 + h * 0.62, seg - 0.5, h * 0.38, fill=shades[i], opacity=0.8))
    parts.append(_path(f"M{_f(x0 + w / 2 - 6)},{_f(y0 + h * 0.43)} L{_f(x0 + w / 2)},{_f(y0 + h * 0.56)} "
                       f"L{_f(x0 + w / 2 + 6)},{_f(y0 + h * 0.43)}", T.GREY_B, 1.5))
    parts.append(_rect(x0 + 1, y0 + h * 0.62 - 1, w - 2, h * 0.38 + 2, stroke=color, width=1.2, rx=2,
                       extra='class="body"'))
    return "".join(parts)


def _g_dropout(n, sp, D, x0, y0):
    color = T.kind_color(n.kind)
    p = float(n.attrs.get("p", n.attrs.get("rate", 0.5)) or 0.0)
    rows, cols = 3, 5
    dead = set(np.random.default_rng(7).permutation(rows * cols)[: int(round(p * rows * cols))].tolist())
    parts = [_rect(x0, y0, sp.gw, sp.gh, stroke=color, width=1.2, rx=6, extra='class="body"')]
    for r in range(rows):
        for c in range(cols):
            cx, cy = x0 + 7 + c * (sp.gw - 14) / (cols - 1), y0 + 8 + r * (sp.gh - 16) / (rows - 1)
            if r * cols + c in dead:
                parts.append(f'<circle cx="{_f(cx)}" cy="{_f(cy)}" r="3" fill="none" stroke="{T.GREY_C}" '
                             f'stroke-width="1"/>')
            else:
                parts.append(f'<circle cx="{_f(cx)}" cy="{_f(cy)}" r="3" fill="{T.GREY_A}"/>')
    return "".join(parts)


def _g_reshape(n, sp, D, x0, y0):
    color = T.kind_color(n.kind)
    h = sp.gh
    cube, cw, ch = cuboid((4, 4, 4), max_side=24, color=T.GREY_B, face=T.GREY_D)
    parts = [f'<g transform="translate({_f(x0)},{_f(y0 + (h - ch) / 2)})">{cube}</g>']
    ax = x0 + cw + 4
    parts.append(_path(f"M{_f(ax)},{_f(y0 + h / 2)} L{_f(ax + 12)},{_f(y0 + h / 2)} M{_f(ax + 8)},{_f(y0 + h / 2 - 4)} "
                       f"L{_f(ax + 12)},{_f(y0 + h / 2)} L{_f(ax + 8)},{_f(y0 + h / 2 + 4)}", T.GREY_B, 1.4))
    bx = ax + 18
    parts.append(_rect(bx, y0 + 2, 7, h - 4, fill=T.GREY_D, stroke=color, width=1.2, rx=1.5, extra='class="body"'))
    for i in range(1, 6):
        parts.append(_line(bx, y0 + 2 + i * (h - 4) / 6, bx + 7, y0 + 2 + i * (h - 4) / 6, T.GREY_C, 0.8))
    return "".join(parts)


def _g_transpose(n, sp, D, x0, y0):
    color = T.kind_color(n.kind)
    s = min(sp.gw, sp.gh) - 4
    parts = [_rect(x0 + 2, y0 + 2, s, s, stroke=color, width=1.4, rx=2, extra='class="body"')]
    for i in range(1, 3):
        parts.append(_line(x0 + 2 + i * s / 3, y0 + 2, x0 + 2 + i * s / 3, y0 + 2 + s, T.GREY_D, 1))
        parts.append(_line(x0 + 2, y0 + 2 + i * s / 3, x0 + 2 + s, y0 + 2 + i * s / 3, T.GREY_D, 1))
    parts.append(_line(x0 + 2, y0 + 2, x0 + 2 + s, y0 + 2 + s, T.YELLOW_D, 1.4))
    parts.append(_rect(x0 + 2 + s / 3, y0 + 2, s / 3, s / 3, fill=T.POSITIVE, opacity=0.8))
    parts.append(_rect(x0 + 2, y0 + 2 + s / 3, s / 3, s / 3, fill=T.POSITIVE, opacity=0.35))
    return "".join(parts)


def _g_slice(n, sp, D, x0, y0):
    color = T.kind_color(n.kind)
    s = min(sp.gw, sp.gh) - 4
    parts = [_rect(x0 + 2, y0 + 2, s, s, stroke=color, width=1.4, rx=2, extra='class="body"')]
    for i in range(4):
        parts.append(_rect(x0 + 2 + i * s / 4 + 1, y0 + 3, s / 4 - 2, s - 2, fill=T.POSITIVE if i == 0 else T.GREY_E,
                           opacity=0.85))
    return "".join(parts)


def _g_embedding(n, sp, D, x0, y0):
    color = T.kind_color(n.kind)
    w, h = sp.gw, sp.gh
    rows, cols = 5, 4
    tw, th = w - 14, h
    tx = x0 + 14
    parts = [_rect(tx, y0, tw, th, stroke=color, width=1.5, rx=2, extra='class="body"')]
    for r in range(rows):
        for c in range(cols):
            hot = r == 2
            parts.append(_rect(tx + 1.5 + c * (tw - 3) / cols, y0 + 1.5 + r * (th - 3) / rows, (tw - 3) / cols - 1.5,
                               (th - 3) / rows - 1.5, fill=color if hot else T.GREY_E, opacity=0.9 if hot else 1))
    ry = y0 + 1.5 + 2.5 * (th - 3) / rows
    parts.append(_path(f"M{_f(x0)},{_f(ry)} L{_f(tx - 2)},{_f(ry)} M{_f(tx - 6)},{_f(ry - 3.5)} L{_f(tx - 2)},{_f(ry)} "
                       f"L{_f(tx - 6)},{_f(ry + 3.5)}", T.GREY_A, 1.4))
    return "".join(parts)


def _g_recurrent(n, sp, D, x0, y0):
    color = T.kind_color(n.kind)
    w, h = sp.gw, sp.gh
    cw, ch = w * 0.5, h * 0.55
    cx, cy = x0 + 4, y0 + (h - ch) / 2
    parts = [_rect(cx, cy, cw, ch, fill=color, stroke=color, width=1.5, rx=6, fop=0.12, extra='class="body"')]
    r = ch * 0.62
    ax, ay = cx + cw, cy + ch / 2
    parts.append(_path(f"M{_f(ax)},{_f(ay - ch * 0.3)} C{_f(ax + r * 1.6)},{_f(ay - r * 1.1)} {_f(ax + r * 1.6)},"
                       f"{_f(ay + r * 1.1)} {_f(ax + 1)},{_f(ay + ch * 0.3)}", color, 1.6))
    parts.append(_path(f"M{_f(ax + 6)},{_f(ay + ch * 0.3 - 2)} L{_f(ax + 1)},{_f(ay + ch * 0.3)} "
                       f"L{_f(ax + 6)},{_f(ay + ch * 0.3 + 4)}", color, 1.6))
    parts.append(_text(cx + cw / 2, cy + ch / 2 + 4, "hₜ", "gq", "middle"))
    return "".join(parts)


def _g_graph(n, sp, D, x0, y0):
    """Message passing: a node (filled) gathering from its neighbours along the edges."""
    color = T.kind_color(n.kind)
    w, h = sp.gw, sp.gh
    cx, cy = x0 + w / 2, y0 + h / 2
    ring = [(cx + 0.36 * w * np.cos(a), cy + 0.36 * h * np.sin(a)) for a in np.linspace(-2.6, 2.9, 5)]
    parts = [_rect(x0, y0, w, h, stroke=color, width=1.3, rx=6, fop=0.06, fill=color, extra='class="body"')]
    for i, (px, py) in enumerate(ring):
        parts.append(_path(f"M{_f(px)},{_f(py)} L{_f(cx)},{_f(cy)}", color, 1.2))
        if i:
            qx, qy = ring[i - 1]
            parts.append(_path(f"M{_f(qx)},{_f(qy)} L{_f(px)},{_f(py)}", T.GREY_D, 0.8))
    for px, py in ring:
        parts.append(f'<circle cx="{_f(px)}" cy="{_f(py)}" r="3.2" fill="{T.GREY_B}" stroke="#000" stroke-width="0.8"/>')
    parts.append(f'<circle cx="{_f(cx)}" cy="{_f(cy)}" r="5" fill="{color}" stroke="#000" stroke-width="1"/>')
    return "".join(parts)


def _g_io(n, sp, D, x0, y0, output=False):
    shape = (n.in_shapes[0] if output else n.out_shapes[0]) if (n.in_shapes if output else n.out_shapes) else None
    dims = _dims(shape)
    if output:
        a = D.act("@out") if D.trace is not None and "@out" in D.trace.acts else None
        if a is not None and 1 < np.size(a) <= 16:
            v = np.asarray(a).reshape(-1)
            m = float(np.abs(v).max()) or 1.0
            bw = (sp.gw - 8) / len(v)
            top = int(np.argmax(v))
            base = y0 + sp.gh - 6 if v.min() >= 0 else y0 + sp.gh / 2
            parts = [_rect(x0, y0, sp.gw, sp.gh, stroke=T.GREY_A, width=1.4, rx=6, extra='class="body"')]
            for i, x in enumerate(v):
                bh = (sp.gh - 12) * abs(x) / m * (1 if v.min() >= 0 else 0.5)
                parts.append(_rect(x0 + 4 + i * bw + 0.5, base - bh if x >= 0 else base, max(bw - 1, 1), bh,
                                   fill=T.POSITIVE if i == top else T.GREY_C, rx=1))
            return "".join(parts)
    image = None
    if not output and D.trace is not None and len(dims) >= 2:
        a = D.act(n.id)
        ch = D.channel_axis(n) if a is not None and a.ndim == 3 else None
        spatial = (a.shape[:-1] if ch == -1 else a.shape[1:]) if ch is not None else (a.shape if a is not None else ())
        if a is not None and a.ndim in (2, 3) and min(spatial) >= 4:
            try:
                image = image_png(a, ch)
            except Exception:  # noqa: BLE001
                image = None
    svg, w, h = cuboid(dims, max_side=min(sp.gw, sp.gh) - 2, image=image, color=T.GREY_A)
    body = _rect(x0, y0, sp.gw, sp.gh, fill=T.BLACK, fop=0.001, extra='class="body"')
    return body + f'<g transform="translate({_f(x0 + (sp.gw - w) / 2)},{_f(y0 + (sp.gh - h) / 2)})">{svg}</g>'


def _g_param(n, sp, D, x0, y0):
    color = T.kind_color("Parameter")
    info = next(iter(n.params.values()), None)
    parts = []
    if info is not None:
        try:
            w = np.asarray(D.net.backend.get_param(info.path), dtype=np.float64)
            w = w.reshape([d for d in w.shape if d != 1] or [1])  # (1, 197, 192) -> (197, 192)
            w = w.reshape(-1, w.shape[-1]) if w.ndim >= 2 else w.reshape(1, -1)
            if w.shape[0] == 1:  # a vector: a strip across the middle
                parts.append(_image(heat_png(w, True, 16), x0 + 3, y0 + sp.gh / 2 - 4, sp.gw - 6, 8))
            else:
                parts.append(_image(heat_png(w, True, 16), x0 + 3, y0 + 3, sp.gw - 6, sp.gh - 6))
        except Exception:  # noqa: BLE001 - decoration only
            pass
    parts.append(_rect(x0, y0, sp.gw, sp.gh, stroke=color, width=1.5, rx=3, extra='class="body"'))
    return "".join(parts)


def _g_block(n, sp, D, x0, y0):
    color = T.GREY_B
    parts = []
    for i in (2, 1, 0):
        parts.append(_rect(x0 + i * 5, y0 + (2 - i) * 5, sp.gw - 12, sp.gh - 12, fill=T.BLACK, stroke=color,
                           width=1.3, rx=6, extra='class="body"' if i == 0 else ""))
    fams = [T.BLUE_C, T.PURPLE_C, T.GREEN_C]
    for j, c in enumerate(fams):
        parts.append(_rect(x0 + 8, y0 + 18 + j * 9, sp.gw - 28, 5, fill=c, rx=2, opacity=0.8))
    return "".join(parts)


def _g_split(n, sp, D, x0, y0):
    """One tensor breaking into k chunks (split / chunk / unbind): k = the outputs actually used."""
    color = T.kind_color(n.kind)
    w, h = sp.gw, sp.gh
    k = max(2, min(len(n.attrs.get("outputs") or []) or 3, 6))
    shades = [T.BLUE_C, T.TEAL_C, T.GOLD_C, T.MAROON_C, T.PURPLE_C, T.GREEN_C]
    parts = [_rect(x0 + 2, y0 + 1, w - 4, h * 0.3, fill=T.GREY_D, stroke=color, width=1.2, rx=2, extra='class="body"')]
    seg = (w - 4) / k
    for i in range(k):
        parts.append(_rect(x0 + 2 + i * seg, y0 + 1, seg, h * 0.3, fill=shades[i], opacity=0.35))
        cx = x0 + 2 + (i + 0.5) * seg
        gap = (i - (k - 1) / 2) * 3.2
        parts.append(_rect(x0 + 2 + i * seg + gap + 1, y0 + h * 0.62, seg - 2, h * 0.36, fill=shades[i], opacity=0.9,
                           rx=1.5))
        parts.append(_line(cx, y0 + h * 0.33, cx + gap, y0 + h * 0.6, shades[i], 1.2, 0.8))
    return "".join(parts)


def _g_constant(n, sp, D, x0, y0):
    """A tensor made inside forward: ``arange`` as a staircase, other constants as a flat block."""
    color = T.kind_color(n.kind)
    w, h = sp.gw, sp.gh
    ramp = n.attrs.get("value") == "arange" or n.kind in ("arange", "linspace")
    k = 6
    bw = (w - 4) / k
    parts = []
    for i in range(k):
        bh = (h - 4) * ((i + 1) / k if ramp else 0.55)
        parts.append(_rect(x0 + 2 + i * bw + 0.8, y0 + h - 2 - bh, bw - 1.6, bh, fill=T.BLUE_D if ramp else T.GREY_C,
                           opacity=0.35 + 0.65 * (i + 1) / k if ramp else 0.7, rx=1))
    parts.append(_line(x0 + 1, y0 + h - 1.5, x0 + w - 1, y0 + h - 1.5, color, 1.4, extra='class="body"'))
    return "".join(parts)


def _g_rotary(n, sp, D, x0, y0):
    """Rotary position embedding: feature pairs rotate by position × frequency — phasors at several speeds."""
    color = T.kind_color(n.kind)
    s = min(sp.gw, sp.gh)
    cx, cy, r = x0 + s / 2, y0 + s / 2, s / 2 - 3
    parts = [f'<circle cx="{_f(cx)}" cy="{_f(cy)}" r="{_f(r)}" fill="{color}" fill-opacity="0.06" stroke="{color}" '
             f'stroke-width="1.4" class="body"/>']
    for i, (ang, col, frac) in enumerate(((0.35, T.BLUE_C, 0.92), (1.25, T.TEAL_C, 0.72), (2.6, T.GOLD_C, 0.52))):
        ex, ey = cx + r * frac * math.cos(ang), cy - r * frac * math.sin(ang)
        parts.append(_line(cx, cy, ex, ey, col, 2))
        parts.append(f'<circle cx="{_f(ex)}" cy="{_f(ey)}" r="2.2" fill="{col}"/>')
        a0, a1 = ang - 0.55 + 0.1 * i, ang - 0.1
        rr = r * frac * 0.55
        parts.append(_path(f"M{_f(cx + rr * math.cos(a0))},{_f(cy - rr * math.sin(a0))} A{_f(rr)},{_f(rr)} 0 0 0 "
                           f"{_f(cx + rr * math.cos(a1))},{_f(cy - rr * math.sin(a1))}", col, 1, opacity=0.6))
    parts.append(f'<circle cx="{_f(cx)}" cy="{_f(cy)}" r="1.8" fill="{T.GREY_A}"/>')
    return "".join(parts)


def _g_matmul(n, sp, D, x0, y0):
    """A @ B: a row of A meets a column of B in one cell of the product."""
    color = T.kind_color(n.kind)
    w, h = sp.gw, sp.gh
    m = min(h - 6, (w - 14) / 3)
    c = m / 4
    ax, bx, ox = x0 + 1, x0 + m + 5, x0 + w - m - 1
    ay = y0 + 3
    parts = []
    for gx, lit in ((ax, "row"), (bx, "col"), (ox, "cell")):
        parts.append(_rect(gx, ay, m, m, stroke=color, width=1.1, rx=1.5, extra='class="body"' if lit == "cell" else ""))
        for i in range(4):
            for j in range(4):
                on = (lit == "row" and i == 1) or (lit == "col" and j == 2) or (lit == "cell" and i == 1 and j == 2)
                parts.append(_rect(gx + j * c + 0.7, ay + i * c + 0.7, c - 1.4, c - 1.4,
                                   fill=T.POSITIVE if on else T.GREY_E, opacity=0.95 if on else 0.8))
    mid = ay + m / 2
    parts.append(f'<circle cx="{_f(bx - 2)}" cy="{_f(mid)}" r="1.4" fill="{T.GREY_A}"/>')
    parts.append(_line(bx + m + 2, mid - 1.5, ox - 2, mid - 1.5, T.GREY_B, 1))
    parts.append(_line(bx + m + 2, mid + 1.5, ox - 2, mid + 1.5, T.GREY_B, 1))
    return "".join(parts)


def _g_mask(n, sp, D, x0, y0):
    """masked_fill / where / tril: the causal triangle — lower cells kept, upper cells masked."""
    color = T.kind_color(n.kind)
    s = min(sp.gw, sp.gh) - 2
    k = 5
    c = s / k
    parts = [_rect(x0 + 1, y0 + 1, s, s, stroke=color, width=1.3, rx=2, extra='class="body"')]
    upper = n.kind == "triu"
    for i in range(k):
        for j in range(k):
            keep = (j >= i) if upper else (j <= i)
            parts.append(_rect(x0 + 1 + j * c + 0.8, y0 + 1 + i * c + 0.8, c - 1.6, c - 1.6,
                               fill=T.POSITIVE if keep else T.GREY_E, opacity=0.9 if keep else 0.7))
            if not keep:
                parts.append(_line(x0 + 1 + j * c + 2, y0 + 1 + i * c + 2, x0 + 1 + (j + 1) * c - 2,
                                   y0 + 1 + (i + 1) * c - 2, T.GREY_D, 0.8))
    return "".join(parts)


def _g_expand(n, sp, D, x0, y0):
    """Broadcast / repeat: one row copied into several."""
    color = T.kind_color(n.kind)
    w, h = sp.gw, sp.gh
    cols = 5
    cw = (w * 0.42) / cols
    parts = []
    for j in range(cols):
        parts.append(_rect(x0 + 1 + j * cw, y0 + h / 2 - cw / 2, cw - 1, cw, fill=T.BLUE_C, opacity=0.9))
    rx = x0 + w * 0.58
    for r in range(3):
        for j in range(cols):
            parts.append(_rect(rx + j * cw * 0.85, y0 + 3 + r * (h - 6) / 3, cw * 0.85 - 1, (h - 6) / 3 - 1.5,
                               fill=T.BLUE_C, opacity=0.9 - 0.25 * r))
    ax = x0 + w * 0.44
    for r, dy in enumerate((-h * 0.3, 0, h * 0.3)):
        parts.append(_line(ax, y0 + h / 2, rx - 2, y0 + h / 2 + dy, T.GREY_B, 1, 0.8))
    parts.append(_rect(rx - 1, y0 + 1, w - (rx - x0), h - 2, stroke=color, width=1.1, rx=2, extra='class="body"'))
    return "".join(parts)


def _g_other(n, sp, D, x0, y0):
    color = T.kind_color(n.kind)
    return (_rect(x0, y0, sp.gw, sp.gh, stroke=color, width=1.4, rx=8, fill=color, fop=0.08, extra='class=\"body\"')
            + _text(x0 + sp.gw / 2, y0 + sp.gh / 2 + 6, "ƒ", "gf", "middle"))


_RENDER = {
    "linear": _g_linear, "conv": _g_conv, "activation": _g_activation, "softmax": _g_softmax, "norm": _g_norm,
    "attention": _g_attention, "pool": _g_pool, "op": _g_op, "dropout": _g_dropout, "reshape": _g_reshape,
    "transpose": _g_transpose, "slice": _g_slice, "embedding": _g_embedding, "recurrent": _g_recurrent, "graph": _g_graph,
    "param": _g_param, "block": _g_block, "other": _g_other, "split": _g_split, "constant": _g_constant,
    "rotary": _g_rotary, "matmul": _g_matmul, "mask": _g_mask, "expand": _g_expand,
}


def render(n: Node, D: GlyphData) -> str:
    """The whole node: hit area + glyph + text column, in a box of ``node_size(n)`` with origin top-left."""
    sp = spec(n)
    w, h = sp.size
    x0, y0 = PAD, (h - sp.gh) / 2
    try:
        if sp.kind == "input":
            g = _g_io(n, sp, D, x0, y0)
        elif sp.kind == "output":
            g = _g_io(n, sp, D, x0, y0, output=True)
        elif sp.kind == "concat":
            g = _g_concat(n, sp, D, x0, y0, len(D.graph.pred(n.id)) if n.id in D.graph.nodes else 2)
        else:
            g = _RENDER[sp.kind](n, sp, D, x0, y0)
    except Exception:  # noqa: BLE001 - a glyph must never break the page
        g = _g_other(n, sp, D, x0, y0)
    tx = PAD + sp.gw + GAP
    lines = [(sp.title, "gk")] + [(t, c) for t, c in ((sp.sub, "gs"), (sp.detail, "gd")) if t]
    total = 16 + 14 * (len(lines) - 1)
    ty = (h - total) / 2 + 12
    text = "".join(_text(tx, ty + i * 14 + (2 if i else 0), t, c) for i, (t, c) in enumerate(lines))
    hit = f'<rect class="hit" x="0" y="0" width="{w}" height="{h}" rx="10" fill="transparent"/>'
    return hit + g + text


# CSS shared by the dashboard and standalone SVG exports
GLYPH_CSS = f"""
.gk {{ fill: {T.GREY_A}; font: 600 13px "CMU Sans Serif", Inter, system-ui, sans-serif; }}
.gs {{ fill: {T.GREY_C}; font: 10.5px "CMU Typewriter Text", "SF Mono", Menlo, monospace; }}
.gd {{ fill: {T.GREY_B}; font: 10.5px "CMU Typewriter Text", "SF Mono", Menlo, monospace; }}
.gt {{ fill: {T.GREY_C}; font: 9.5px "CMU Typewriter Text", "SF Mono", Menlo, monospace; dominant-baseline: middle; }}
.gq {{ fill: {T.GREY_B}; font: italic 10px "CMU Serif", Georgia, serif; }}
.gf {{ fill: {T.GREY_A}; font: italic 18px "CMU Serif", Georgia, serif; }}
"""
