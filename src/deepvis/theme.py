"""Visual theme: the 3Blue1Brown palette (exact values from manimlib/default_config.yml).

Conventions borrowed from the 3b1b neural-network series:
positive weights are blue, negative weights are red, and opacity / stroke width
scale with magnitude; neuron brightness scales with activation.
"""

from __future__ import annotations

import numpy as np

# --- manim palette -----------------------------------------------------------
BLUE_E, BLUE_D, BLUE_C, BLUE_B, BLUE_A = "#1C758A", "#29ABCA", "#58C4DD", "#9CDCEB", "#C7E9F1"
TEAL_E, TEAL_D, TEAL_C, TEAL_B, TEAL_A = "#49A88F", "#55C1A7", "#5CD0B3", "#76DDC0", "#ACEAD7"
GREEN_E, GREEN_D, GREEN_C, GREEN_B, GREEN_A = "#699C52", "#77B05D", "#83C167", "#A6CF8C", "#C9E2AE"
YELLOW_E, YELLOW_D, YELLOW_C, YELLOW_B, YELLOW_A = "#E8C11C", "#F4D345", "#FFFF00", "#FFEA94", "#FFF1B6"
GOLD_E, GOLD_D, GOLD_C, GOLD_B, GOLD_A = "#C78D46", "#E1A158", "#F0AC5F", "#F9B775", "#F7C797"
RED_E, RED_D, RED_C, RED_B, RED_A = "#CF5044", "#E65A4C", "#FC6255", "#FF8080", "#F7A1A3"
MAROON_E, MAROON_D, MAROON_C, MAROON_B, MAROON_A = "#94424F", "#A24D61", "#C55F73", "#EC92AB", "#ECABC1"
PURPLE_E, PURPLE_D, PURPLE_C, PURPLE_B, PURPLE_A = "#644172", "#715582", "#9A72AC", "#B189C6", "#CAA3E8"
GREY_E, GREY_D, GREY_C, GREY_B, GREY_A = "#222222", "#444444", "#888888", "#BBBBBB", "#DDDDDD"
WHITE, BLACK = "#FFFFFF", "#000000"
PINK, LIGHT_PINK, ORANGE, GREY_BROWN = "#D147BD", "#DC75CD", "#FF862F", "#736357"

BLUE, TEAL, GREEN, YELLOW, GOLD, RED, MAROON, PURPLE, GREY = (
    BLUE_C, TEAL_C, GREEN_C, YELLOW_C, GOLD_C, RED_C, MAROON_C, PURPLE_C, GREY_C,
)

# Every color below is a manim constant (or a manim color with transparency applied at draw time).
BACKGROUND = BLACK      # manim's canvas
PANEL = BLACK           # side panels share the canvas; separated by GREY_E rules
PANEL_2 = GREY_E        # tiles, inputs, raised surfaces
PANEL_EDGE = GREY_D     # borders and separators
GRID = GREY_E           # faint NumberPlane-like background grid
TEXT = GREY_A
TEXT_2 = GREY_B
MUTED = GREY_C
HIGHLIGHT = YELLOW_C    # selection / focus / "look here" (3b1b's highlight yellow)
POSITIVE = BLUE_C       # positive weights (3b1b neural-network convention)
NEGATIVE = RED_C        # negative weights

# --- layer kinds -> color families (manim hues; related kinds share a hue family) --------------
# Color is a secondary cue: every node, block and tensor also prints its kind as text.
FAMILY_COLORS = {
    "linear": BLUE_C,
    "conv": TEAL_C,
    "pool": TEAL_E,
    "attention": GOLD_C,
    "embedding": MAROON_C,
    "recurrent": PINK,
    "graph": ORANGE,
    "norm": PURPLE_C,
    "activation": GREEN_C,
    "dropout": GREY_C,
    "reshape": GREY_C,
    "op": GREY_B,
    "io": GREY_A,
    "container": GREY_B,
    "other": GREY_BROWN,
}
FAMILY_LABELS = {
    "linear": "dense", "conv": "convolution", "pool": "pooling", "attention": "attention",
    "embedding": "embedding", "recurrent": "recurrent", "norm": "normalization", "activation": "nonlinearity",
    "dropout": "dropout", "reshape": "reshape", "op": "arithmetic", "io": "input / output", "other": "other",
    "container": "block", "graph": "message passing",
}
WEIGHTS_HUE = FAMILY_COLORS["linear"]  # kept for backwards compatibility

_FAMILY_OF_KIND = {
    "Linear": "linear", "Bilinear": "linear", "matmul": "op",
    "Conv1d": "conv", "Conv2d": "conv", "Conv3d": "conv", "ConvTranspose1d": "conv",
    "ConvTranspose2d": "conv", "ConvTranspose3d": "conv",
    "LayerNorm": "norm", "RMSNorm": "norm", "BatchNorm": "norm", "BatchNorm1d": "norm",
    "BatchNorm2d": "norm", "BatchNorm3d": "norm", "GroupNorm": "norm", "InstanceNorm2d": "norm",
    "ReLU": "activation", "LeakyReLU": "activation", "GELU": "activation", "SiLU": "activation",
    "Sigmoid": "activation", "Tanh": "activation", "Softmax": "activation", "LogSoftmax": "activation",
    "ELU": "activation", "Softplus": "activation", "Mish": "activation", "Hardtanh": "activation",
    "exp": "activation", "log": "activation", "abs": "activation", "square": "activation",
    "Hardswish": "activation", "Hardsigmoid": "activation", "ReLU6": "activation", "PReLU": "activation",
    "SELU": "activation", "CELU": "activation", "Softsign": "activation", "GLU": "activation",
    "Upsample": "pool", "PixelShuffle": "reshape", "expand": "reshape", "Normalize": "norm",
    "MultiheadAttention": "attention", "Attention": "attention",
    "MaxPool1d": "pool", "MaxPool2d": "pool", "AvgPool1d": "pool", "AvgPool2d": "pool",
    "AdaptiveAvgPool1d": "pool", "AdaptiveAvgPool2d": "pool", "AdaptiveMaxPool2d": "pool",
    "mean": "pool", "sum": "pool", "max": "pool",
    "Embedding": "embedding", "Parameter": "embedding",
    "LSTM": "recurrent", "GRU": "recurrent", "RNN": "recurrent", "Scan": "recurrent",
    "Dropout": "dropout", "Identity": "dropout",
    "Flatten": "reshape", "Reshape": "reshape", "Unflatten": "reshape", "Transpose": "reshape",
    "Concat": "reshape", "getitem": "reshape",
    "add": "op", "sub": "op", "mul": "op", "div": "op", "neg": "op",
    "Input": "io", "Output": "io",
    "sin": "activation", "cos": "activation", "sqrt": "activation", "rsqrt": "activation", "pow": "activation",
    "reciprocal": "activation", "erf": "activation",
    "split": "reshape", "unbind": "reshape", "unsqueeze": "reshape", "squeeze": "reshape", "repeat": "reshape",
    "Constant": "io", "RotaryEmbedding": "embedding",
    "masked_fill": "attention", "where": "op", "tril": "attention", "triu": "attention",
}


def kind_family(kind: str, is_container: bool = False) -> str:
    if is_container:
        return "container"
    fam = _FAMILY_OF_KIND.get(kind)
    if fam is not None:
        return fam
    if kind.endswith("Conv") and len(kind) > 4:  # PyG: GCNConv, GATConv, SAGEConv, GINConv, TransformerConv …
        return "graph"
    if "Mixer" in kind or kind.startswith("Mamba"):  # selective state-space blocks
        return "recurrent"
    if "Router" in kind or "Experts" in kind or "Moe" in kind or "MoE" in kind:
        return "linear"
    return "other"


def kind_color(kind: str, is_container: bool = False) -> str:
    return FAMILY_COLORS[kind_family(kind, is_container)]


# --- colormaps ----------------------------------------------------------------
def _hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0


def _rgb_to_hex(rgb) -> str:
    r, g, b = (int(round(float(np.clip(c, 0, 1)) * 255)) for c in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


# Diverging (3b1b weight convention): negative -> red, zero -> dark grey, positive -> blue.
# On the black canvas larger magnitude = brighter, on both arms.
DIVERGING_STOPS = [RED_A, RED_C, RED_E, GREY_E, BLUE_E, BLUE_C, BLUE_A]
# Sequential (magnitude): one hue, dark -> light.
SEQUENTIAL_STOPS = [GREY_E, BLUE_E, BLUE_D, BLUE_C, BLUE_B, BLUE_A]


def colormap(values, stops=DIVERGING_STOPS, vmin=None, vmax=None) -> list[str]:
    """Map numeric values to hex colors along piecewise-linear color stops."""
    v = np.asarray(values, dtype=float)
    lo = np.nanmin(v) if vmin is None else vmin
    hi = np.nanmax(v) if vmax is None else vmax
    t = np.zeros_like(v) if hi == lo else (v - lo) / (hi - lo)
    t = np.clip(np.nan_to_num(t, nan=0.5), 0, 1)
    rgb = np.stack([_hex_to_rgb(s) for s in stops])
    pos = t * (len(stops) - 1)
    i = np.minimum(pos.astype(int), len(stops) - 2)
    f = (pos - i)[..., None]
    out = (1 - f) * rgb[i] + f * rgb[i + 1]
    return [_rgb_to_hex(c) for c in out.reshape(-1, 3)]


def as_dict() -> dict:
    """Theme exported to the HTML dashboard (it sets the page's CSS variables from this)."""
    return {
        "background": BACKGROUND,
        "panel": PANEL,
        "panel2": PANEL_2,
        "panelEdge": PANEL_EDGE,
        "grid": GRID,
        "text": TEXT,
        "text2": TEXT_2,
        "muted": MUTED,
        "highlight": HIGHLIGHT,
        "positive": POSITIVE,
        "negative": NEGATIVE,
        "families": FAMILY_COLORS,
        "familyLabels": FAMILY_LABELS,
        "diverging": DIVERGING_STOPS,
        "sequential": SEQUENTIAL_STOPS,
        "palette": {k: v for k, v in globals().items() if k.isupper() and isinstance(v, str) and v.startswith("#")
                    and "_" in k and k.split("_")[0] in ("BLUE", "TEAL", "GREEN", "YELLOW", "GOLD", "RED",
                                                            "MAROON", "PURPLE", "GREY")},
    }
