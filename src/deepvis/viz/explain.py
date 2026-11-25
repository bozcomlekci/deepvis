"""Plain-language explanations of layers, filled in with each layer's own numbers.

``explain(node, graph)`` → ``{"what": …, "why": …, "look": …}`` (``why`` / ``look`` may be missing): what the layer
does, why networks have it, and where to look in the dashboard. The dashboard shows it at the top of a layer's panel;
the videos use ``what`` (and ``why``) as captions. Written for someone meeting the layer for the first time: words
first, a formula only where it is the shortest way to say it, and this layer's real sizes instead of placeholders.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

from ..core.kinds import ACTIVATION_KINDS, CONV_KINDS, NORM_KINDS, TRANSPOSED_CONV_KINDS

if TYPE_CHECKING:  # pragma: no cover
    from ..core.graph import ModelGraph, Node

WHY_NONLINEAR = ("A bend between layers: without one, any stack of layers collapses into a single linear map, "
                 "however deep it is.")


def _n(v) -> str:
    return f"{int(v):,}"


def _dims(shape, batched: bool) -> str:
    s = [int(d) for d in (shape or ())]
    if batched and len(s) >= 2:
        s = s[1:]
    return " × ".join(str(d) for d in s) if s else "a single number"


def _count(shape, batched: bool) -> int:
    s = [int(d) for d in (shape or ())]
    if batched and len(s) >= 2:
        s = s[1:]
    return math.prod(s) if s else 1


def _params(node: "Node") -> int:
    return node.n_params


def _weight(node: "Node"):
    """(out, in, spatial dims) of the layer's weight, from its roles — the same in every framework."""
    r = node.roles.get("weight")
    info = node.params.get(r.name) if r is not None else None
    if info is None or r.out_axis is None:
        return None
    shape = tuple(int(d) for d in info.shape)
    if not shape:
        return None
    oa = r.out_axis % len(shape)
    ia = r.in_axis % len(shape) if r.in_axis is not None else None
    rest = tuple(d for i, d in enumerate(shape) if i not in (oa, ia))
    return shape[oa], (shape[ia] if ia is not None else None), rest


def _ints(v) -> tuple:
    if v is None:
        return ()
    if isinstance(v, (list, tuple)):
        return tuple(int(x) for x in v if isinstance(x, (int, float)))
    return (int(v),) if isinstance(v, (int, float)) else ()


def _rank(node: "Node") -> int:
    """Spatial rank of a 1d / 2d / 3d layer (from its name, else its input)."""
    for r in (1, 2, 3):
        if node.kind.endswith(f"{r}d"):
            return r
    s = node.in_shapes[0] if node.in_shapes and node.in_shapes[0] is not None else ()
    return max(1, len(s) - 2) if len(s) >= 3 else 1


def _window(v, rank: int = 1) -> str:
    k = _ints(v)
    if len(k) == 1 and rank > 1:
        k = k * rank
    return " × ".join(str(x) for x in k) if k else ""


def _spatial(shape, g: "ModelGraph | None") -> str:
    """The spatial sizes of a (batch, channels, …) tensor — channels last for JAX."""
    s = [int(d) for d in (shape or ())]
    if len(s) < 3:
        return ""
    sp = s[1:-1] if g is not None and g.framework not in ("torch", "onnx", "") else s[2:]
    return " × ".join(map(str, sp))


def _stride(node: "Node") -> str:
    s = _ints(node.attrs.get("stride", node.attrs.get("strides")))
    return f" in steps of {s[0]}" if s and max(s) > 1 and len(set(s)) == 1 else (
        f" in steps of {' × '.join(map(str, s))}" if s and max(s) > 1 else "")


def _io(node: "Node", batched: bool) -> tuple[str, str]:
    i = _dims(node.in_shapes[0], batched) if node.in_shapes and node.in_shapes[0] is not None else ""
    o = _dims(node.out_shapes[0], batched) if node.out_shapes and node.out_shapes[0] is not None else ""
    return i, o


def _then(node: "Node") -> str:
    """A built-in activation (Keras: ``Dense(10, activation="relu")``), said after the layer's own work."""
    act = node.attrs.get("activation")
    if not isinstance(act, str) or act in ("linear", "identity", "None"):
        return ""
    from ..core.kinds import function_kind

    return f" Then {function_kind(act)} is applied to each output."


# ------------------------------------------------------------------------------------------ by kind
def _linear(node, g, batched):
    w = _weight(node)
    out = w[0] if w else node.attrs.get("out_features") or node.attrs.get("features")
    inn = w[1] if w else node.attrs.get("in_features")
    if not out or not inn:
        return None
    bias = "bias" in node.roles
    return {
        "what": f"Each of its {_n(out)} outputs is a weighted sum of all {_n(inn)} inputs"
                + (" plus a bias" if bias else "") + f" — {_n(_params(node))} learned numbers in all." + _then(node),
        "why": "It lets every output combine any of the inputs; training decides which combinations matter.",
        "look": "Weights: one row per output — blue pushes that output up, red pulls it down. "
                "The Neurons tab draws the same weights as lines.",
    }


def _conv(node, g, batched):
    w = _weight(node)
    if not w or not w[1]:
        return None
    c_out, c_in, k = w
    groups = node.attrs.get("groups") or 1
    k_txt = " × ".join(map(str, k)) if k else "1"
    over = f"all {_n(c_in * groups)} input channels" if groups == 1 else f"{_n(c_in)} input channels (in {groups} groups)"
    return {
        "what": f"{_n(c_out)} small filters, each {k_txt} across {over}, slide over the input{_stride(node)}; at every "
                f"position a filter's weighted sum becomes one value of its output map — {_n(c_out)} maps in all."
                + _then(node),
        "why": "The same pattern detector is reused at every position, so it finds an edge or a texture wherever it "
               f"appears — with only {_n(_params(node))} weights.",
        "look": "Kernel: each square is one filter's weights. Activation: one feature map per filter, bright where "
                "its pattern shows up.",
    }


def _conv_t(node, g, batched):
    i, o = _io(node, batched)
    w = _weight(node)
    k = " × ".join(map(str, w[2])) if w and w[2] else ""
    return {
        "what": f"Grows its input ({i} → {o}): each value is spread out through learned {k + ' ' if k else ''}filters"
                f"{_stride(node)}, and overlapping contributions add up.",
        "why": "It upsamples: decoders and image generators use it to go from a small feature map back to a full-size "
               "picture.",
    }


def _pool(node, g, batched):
    kind = node.kind
    i, o = _io(node, batched)
    k = _window(node.attrs.get("kernel_size", node.attrs.get("window_shape")), _rank(node))
    si = _spatial(node.in_shapes[0], g) if node.in_shapes and node.in_shapes[0] is not None else ""
    so = _spatial(node.out_shapes[0], g) if node.out_shapes and node.out_shapes[0] is not None else ""
    shrink = f", shrinking each map from {si} to {so}" if si and so and si != so else ""
    if kind.startswith("Adaptive"):
        how = "largest value" if "Max" in kind else "average"
        return {"what": f"Takes the {how} of each channel's values in a grid of regions, giving {o} whatever the input "
                        "size was.",
                "why": "The layers after it can then work on images of any size; with a 1 × 1 output it is 'global "
                       "pooling': one number per channel."}
    if "Max" in kind or kind == "max":
        return {"what": f"Keeps the largest value in each {k + ' ' if k else ''}window{_stride(node)}{shrink}.",
                "why": "Fewer numbers to process downstream, and a pattern that shifts by a pixel still gives the "
                       "same result.",
                "look": "The glyph lights the cells a max-pool keeps."}
    if "Avg" in kind:
        return {"what": f"Averages each {k + ' ' if k else ''}window{_stride(node)}{shrink}.",
                "why": "Fewer numbers to process downstream, keeping the overall level of each region."}
    if kind == "Upsample":
        return {"what": f"Enlarges each map ({i} → {o}) by repeating or interpolating its values.",
                "why": "Brings a coarse feature map back to a finer resolution."}
    verb = {"mean": "Averages", "sum": "Adds up"}.get(kind, "Reduces")
    return {"what": f"{verb} values along some axes: {i} → {o}." if i and o else f"{verb} values along some axes."}


def _norm(node, g, batched):
    kind = node.kind
    sc = node.role_param("scale") if "scale" in node.roles else None
    d = int(sc.shape[0]) if sc is not None and sc.shape else None
    learned = f" ({_n(_params(node))} learned numbers)" if _params(node) else ""
    why = "Keeps the numbers flowing through a deep network in a steady range, which makes training stable and fast."
    look = "The glyph: grey before, colour after — the spread is evened out."
    if kind == "BatchNorm":
        c = node.attrs.get("num_features") or d
        return {"what": f"Normalizes each of its {_n(c) + ' ' if c else ''}channels with a mean and variance gathered "
                        "over training batches (when you use the model: the stored running averages), then applies a "
                        f"learned scale and shift{learned}.", "why": why, "look": look}
    if kind == "RMSNorm":
        return {"what": f"Divides each position's {_n(d) + ' ' if d else ''}features by their root-mean-square, then "
                        f"multiplies by a learned scale{learned}.", "why": why, "look": look}
    if kind == "GroupNorm":
        gr = node.attrs.get("num_groups")
        return {"what": f"Normalizes channels in {gr} groups" if gr else "Normalizes channels in groups",
                "why": why + " Unlike BatchNorm it doesn't depend on the batch.", "look": look}
    if kind == "InstanceNorm2d":
        return {"what": "Normalizes each channel of each example on its own (mean 0, spread 1).", "why": why}
    return {"what": f"For each position it shifts and rescales the {_n(d) + ' ' if d else ''}features to mean 0 and "
                    f"spread 1, then applies a learned scale and shift{learned}.", "why": why, "look": look}


_ACT = {
    "ReLU": "Keeps positive values and turns negative ones into 0: max(0, x).",
    "ReLU6": "Keeps values between 0 and 6: negatives become 0, anything above 6 becomes 6.",
    "GELU": "A smooth ReLU: values well below 0 become 0, values well above pass unchanged, with a soft curve in "
            "between.",
    "SiLU": "A smooth ReLU: x · sigmoid(x) — negative values are mostly, but not completely, switched off.",
    "Sigmoid": "Squashes any number into the range 0 to 1: 1 / (1 + e^(−x)).",
    "Tanh": "Squashes any number into the range −1 to 1.",
    "ELU": "Like ReLU for positive values; negative ones bend smoothly towards −1 instead of stopping at 0.",
    "Softplus": "A smooth ReLU: log(1 + e^x), always positive.",
    "Mish": "A smooth ReLU: x · tanh(softplus(x)).",
    "Hardtanh": "Clips values to the range −1 to 1.",
    "Hardswish": "A cheap, piecewise version of SiLU.",
    "Hardsigmoid": "A cheap, piecewise version of the sigmoid: values between 0 and 1.",
    "SELU": "A scaled ELU, chosen so activations keep mean 0 and spread 1 through many layers.",
    "CELU": "A smooth ELU variant: like ReLU for positive values, a soft curve below 0.",
    "Softsign": "Squashes any number into −1 to 1: x / (1 + |x|).",
    "PReLU": "Like ReLU, but negative values pass through scaled by a learned slope.",
    "GLU": "Splits its input in two halves and lets one half gate the other: a · sigmoid(b).",
}


def _activation(node, g, batched):
    kind = node.kind
    if kind == "Softmax":
        _, o = _io(node, batched)
        n = int(node.out_shapes[0][-1]) if node.out_shapes and node.out_shapes[0] else None
        return {"what": f"Turns {_n(n) + ' ' if n else ''}scores into probabilities that add up to 1: exponentiate "
                        "each, then divide by the total.",
                "why": "The scores become readable as the network's confidence in each option; the biggest score "
                       "gets the biggest share.",
                "look": "The Output panel shows the probabilities."}
    if kind == "LogSoftmax":
        return {"what": "The logarithm of a softmax: log-probabilities that add up (after exponentiating) to 1.",
                "why": "The form a cross-entropy loss works with; it is more precise than log(softmax(x))."}
    if kind == "LeakyReLU":
        s = node.attrs.get("negative_slope")
        return {"what": "Keeps positive values; negative ones pass through scaled down"
                        + (f" by {s}" if s is not None else "") + ".",
                "why": WHY_NONLINEAR + " The small slope keeps neurons from going permanently silent.",
                "look": "The glyph: its curve, with bars where this input's values landed."}
    text = _ACT.get(kind)
    if text is None:
        if kind in ("exp", "log", "abs", "square", "sqrt", "rsqrt", "pow", "erf", "sin", "cos", "reciprocal"):
            return {"what": f"Applies {kind} to every value on its own."}
        text = f"Applies {kind} to every value on its own."
    look = ("The glyph: its curve, with bars where this input's values landed"
            + ("; the stats show how many came out 0." if kind in ("ReLU", "ReLU6") else "."))
    why = WHY_NONLINEAR + (" ReLU is the simplest bend and the cheapest to compute." if kind == "ReLU" else "")
    if kind in ("Sigmoid", "Hardsigmoid"):
        why = "Useful as a probability or as a gate that decides how much of something gets through."
    return {"what": text, "why": why, "look": look}


def _attention(node, g, batched):
    heads = node.attrs.get("num_heads")
    e = node.attrs.get("embed_dim")
    d = node.attrs.get("head_dim") or (e // heads if e and heads else None)
    shape = node.in_shapes[0] if node.in_shapes and node.in_shapes[0] else None
    t = None
    if shape is not None and len(shape) >= 2:
        if len(shape) == 3:
            t = shape[1] if node.attrs.get("batch_first", True) else shape[0]
        elif len(shape) == 4:  # (batch, heads, tokens, dim): scaled_dot_product_attention
            heads, t, d = heads or shape[1], shape[2], d or shape[3]
    split = f", in {heads} heads of {d} numbers each" if heads and d else (f", in {heads} heads" if heads else "")
    grid = f" the {t} × {t} scores" if t else " the scores"
    return {
        "what": f"Every token asks a question (its query) and compares it with every token's key{split}; a softmax "
                f"turns{grid} into weights that mix the tokens' values.",
        "why": "Each position can pull in information from any other position, with weights computed from the "
               "content itself — the core of a transformer.",
        "look": "Attention tab: per head, queries (rows) × keys (columns); a bright cell means 'this token looks at "
                "that one'.",
    }


def _embedding(node, g, batched):
    w = _weight(node)
    info = node.role_param("weight") if "weight" in node.roles else None
    if info is not None and len(info.shape) == 2:
        rows, dim = int(info.shape[0]), int(info.shape[1])
    elif w and w[1]:
        rows, dim = w[1], w[0]
    else:
        return {"what": "A lookup table: every token id picks out its own learned vector."}
    return {
        "what": f"A table of {_n(rows)} rows with {_n(dim)} numbers each; every token id picks out its row.",
        "why": "It turns symbols the network can't do arithmetic on into vectors it can; training pulls related "
               "tokens' rows together.",
        "look": "Weights: the table itself, one row per token.",
    }


def _recurrent(node, g, batched):
    kind = node.kind
    if kind == "Scan":
        return {"what": "A loop over the sequence (jax.lax.scan) that carries a state from each step to the next."}
    h = node.attrs.get("hidden_size")
    layers = node.attrs.get("num_layers") or 1
    bi = node.attrs.get("bidirectional") or any("_reverse" in k for k in node.params)  # torch: weight_ih_l0_reverse
    mem = f"a memory of {_n(h)} numbers" if h else "a memory"
    extra = {"LSTM": " Its gates decide what to forget, what to write and what to output.",
             "GRU": " Its gates decide how much of the old memory to keep and what to write."}.get(kind, "")
    stack = f" ({layers} stacked layers)" if layers > 1 else ""
    return {
        "what": f"Reads the sequence one step at a time, carrying {mem} from each step to the next{stack}"
                + (", once forwards and once backwards" if bi else "") + "." + extra,
        "why": "Earlier steps change how later ones are read — a memory for sequences.",
        "look": "Panel: every gate at every time step, recomputed from the layer's own weights.",
    }


def _dropout(node, g, batched):
    if node.kind == "Identity":
        return {"what": "Passes its input through unchanged — a placeholder, often where an optional layer was "
                        "switched off."}
    p = node.attrs.get("p", node.attrs.get("rate"))
    if p is not None and not p:
        return {"what": "With p = 0 it passes everything through unchanged — a placeholder."}
    share = f"{p:.0%} of the values" if isinstance(p, (int, float)) else "some of the values"
    return {
        "what": f"While training it switches off {share} at random (and scales the rest up to compensate); when you "
                "use the model it passes everything through unchanged.",
        "why": "No unit can rely on any single neighbour, which reduces overfitting.",
    }


def _reshape(node, g, batched):
    kind = node.kind
    i, o = _io(node, batched)
    if kind == "Flatten":
        n = _count(node.out_shapes[0], batched) if node.out_shapes and node.out_shapes[0] else None
        where = f"the {i} values out in one row of {_n(n)}" if i and n else "its input out in one row"
        return {"what": f"Lays {where} — no arithmetic, only a new shape.",
                "why": "Dense layers expect a flat list of numbers."}
    if kind == "Reshape":
        if o.isdigit():
            return {"what": f"Lays the numbers out in one row of {_n(int(o))} — no arithmetic, only a new shape."}
        return {"what": f"Rearranges the same numbers into the shape {o} — no arithmetic." if o else
                        "Rearranges the same numbers into a new shape — no arithmetic."}
    if kind == "Transpose":
        return {"what": "Reorders its axes: the same numbers, read in a different order."}
    if kind == "Concat":
        k = len(g.pred(node.id)) if g is not None and node.id in g.nodes else None
        inputs = f"its {k} inputs" if k and k > 1 else "its inputs"
        return {"what": f"Joins {inputs} side by side" + (f" into one tensor of {o}." if o else ".")}
    if kind == "getitem":
        return {"what": "Picks out part of a tensor" + (f" ({i} → {o})." if i and o and i != o else ".")}
    if kind in ("split", "unbind"):
        return {"what": "Cuts one tensor into pieces, one per output."}
    if kind == "expand":
        return {"what": "Repeats values to fill a larger shape (broadcasting) — no new numbers."}
    if kind == "PixelShuffle":
        return {"what": f"Moves channels into space: {i} → {o}, trading depth for resolution."}
    return {"what": f"Changes the tensor's shape ({i} → {o}) without arithmetic." if i and o else
                    "Changes the tensor's shape without arithmetic."}


_PREDS: dict = {}


def _preds(g: "ModelGraph") -> dict:
    """Predecessors by node, built once per graph (``g.pred`` scans every edge)."""
    key = (id(g), len(g.edges))
    if key not in _PREDS:
        if len(_PREDS) > 16:
            _PREDS.clear()
        out: dict[str, list[str]] = {}
        for e in g.edges:
            out.setdefault(e.dst, []).append(e.src)
        _PREDS[key] = out
    return _PREDS[key]


def _reaches(g: "ModelGraph", src: str, dst: str, limit: int = 400) -> bool:
    """Whether ``dst`` depends on ``src`` (walking predecessors from ``dst``)."""
    preds = _preds(g)
    stack, seen = [dst], set()
    while stack and len(seen) < limit:
        n = stack.pop()
        if n == src:
            return True
        if n in seen:
            continue
        seen.add(n)
        stack.extend(preds.get(n, ()))
    return False


def _ancestors(g: "ModelGraph", nid: str, limit: int = 80) -> set:
    preds, out, stack = _preds(g), set(), [nid]
    while stack and len(out) < limit:
        n = stack.pop()
        if n not in out:
            out.add(n)
            stack.extend(preds.get(n, ()))
    return out


def _residual(g: "ModelGraph | None", node: "Node") -> bool:
    """An add joining two paths that forked from the same recent tensor: a skip connection, whether the shortcut is
    the identity or a projection (ResNet's downsampling blocks)."""
    if g is None or node.kind != "add":
        return False
    p = _preds(g).get(node.id, [])
    return len(p) >= 2 and bool(_ancestors(g, p[0]) & _ancestors(g, p[1]))


def _op(node, g, batched):
    kind = node.kind
    if kind == "add":
        if _residual(g, node):
            return {"what": "Adds a block's input back onto its output — a residual (skip) connection.",
                    "why": "The block only has to learn a correction to its input, and gradients reach the early "
                           "layers directly: this is what makes very deep networks trainable."}
        return {"what": "Adds its inputs together, value by value."}
    if kind == "mul":
        return {"what": "Multiplies its inputs value by value — often one of them is a gate that decides how much "
                        "of the other gets through."}
    if kind == "matmul":
        return {"what": "A matrix product: each row of the first meets each column of the second in one dot product."}
    return {"what": {"sub": "Subtracts its inputs, value by value.", "div": "Divides its inputs, value by value.",
                     "neg": "Flips the sign of every value.",
                     "where": "Chooses, value by value, between two inputs according to a condition."}.get(
        kind, f"A {kind} operation.")}


def _mask(node, g, batched):
    return {"what": "Hides some positions — for a language model, the future — by setting their attention scores to "
                    "−∞, so the softmax gives them weight 0.",
            "why": "A token mustn't peek at the tokens it is supposed to predict."}


def _io_node(node, g, batched):
    kind = node.kind
    if node.op == "input":
        o = _dims(node.out_shapes[0], batched) if node.out_shapes and node.out_shapes[0] else ""
        o = f"{o} values" if o.isdigit() else o
        return {"what": f"The data you passed in: {o} per example." if o and batched else
                        (f"The data you passed in: {o}." if o else "The data you passed in.")}
    if node.op == "output":
        return {"what": "What the network returns.", "look": "The Output panel shows it for the shown example."}
    if kind == "Constant":
        return {"what": "A tensor made inside the network's forward pass (for example the positions 0, 1, 2, …), "
                        "not passed in and not learned."}
    if kind == "Parameter" or node.op == "param":
        n = sum(p.numel for p in node.params.values())
        return {"what": f"A learned tensor of {_n(n)} numbers used directly in a calculation." if n else
                        "A learned tensor used directly in a calculation."}
    return None


def _graph(node, g, batched):
    return {"what": "Each node of the input graph gathers its neighbours' features along the edges, combines them "
                    f"with its own and transforms the result with learned weights ({_n(_params(node))}).",
            "why": "Networks can then work on graphs — molecules, meshes, social networks — where 'neighbour' isn't "
                   "a fixed grid.",
            "look": "Panel: the input graph and how features spread along it."}


def _container(node, g, batched):
    kids = 0
    if g is not None and node.id in g.nodes:
        kids = sum(1 for d in g.descendants(node.id) if g.nodes[d].is_compute)
    p = node.n_params
    inside = " and ".join(x for x in (f"{kids} layers" if kids else "", f"{_n(p)} learned numbers" if p else "") if x)
    what = f"A {node.type_name} block" + (f" with {inside} inside." if inside else ".")
    if node.attrs.get("collapsed"):
        what += " Pick a deeper level (or 'all layers') to open it."
    return {"what": what}


_RULES: dict[str, Callable] = {"Linear": _linear, "Bilinear": _linear, "Embedding": _embedding,
                               "MultiheadAttention": _attention, "Attention": _attention, "Dropout": _dropout,
                               "Identity": _dropout, "masked_fill": _mask, "tril": _mask, "triu": _mask,
                               "RotaryEmbedding": lambda n, g, b: {
                                   "what": "Rotates pairs of features by an angle that grows with the token's "
                                           "position, so attention can tell how far apart two tokens are."},
                               "Input": _io_node, "Output": _io_node, "Constant": _io_node, "Parameter": _io_node}
for _k in CONV_KINDS:
    _RULES[_k] = _conv
for _k in TRANSPOSED_CONV_KINDS:
    _RULES[_k] = _conv_t
for _k in NORM_KINDS | {"Normalize"}:
    _RULES[_k] = _norm
for _k in ACTIVATION_KINDS | set(_ACT) | {"sin", "cos", "reciprocal"}:
    _RULES[_k] = _activation
for _k in ("MaxPool1d", "MaxPool2d", "AvgPool1d", "AvgPool2d", "AdaptiveAvgPool1d", "AdaptiveAvgPool2d",
           "AdaptiveMaxPool2d", "Upsample", "mean", "sum", "max"):
    _RULES[_k] = _pool
for _k in ("LSTM", "GRU", "RNN", "Scan"):
    _RULES[_k] = _recurrent
for _k in ("Flatten", "Reshape", "Unflatten", "Transpose", "Concat", "getitem", "split", "unbind", "expand",
           "PixelShuffle", "unsqueeze", "squeeze", "repeat"):
    _RULES[_k] = _reshape
for _k in ("add", "sub", "mul", "div", "neg", "matmul", "where"):
    _RULES[_k] = _op


def explain(node: "Node", graph: "ModelGraph | None" = None, batched: bool = True) -> dict | None:
    """``{"what", "why"?, "look"?}`` for ``node`` (see the module docstring), or None when there is nothing useful
    to say. ``graph`` lets residual connections and containers be recognized; ``batched`` drops the batch axis
    from the sizes quoted."""
    if node.op == "container" or node.attrs.get("collapsed"):
        return _container(node, graph, batched)
    rule = _RULES.get(node.kind)
    if rule is None and node.op in ("input", "output", "param"):
        rule = _io_node
    if rule is None:
        from .. import theme

        fam = theme.kind_family(node.kind)
        if fam == "graph":
            rule = _graph
        elif "Router" in node.kind or "Experts" in node.kind:
            return {"what": "Scores each token for every expert and sends it to the best few — only part of the "
                            "network runs for each token.",
                    "why": "More learned capacity without more computation per token (a mixture of experts)."}
        elif "Mixer" in node.kind or node.kind.startswith("Mamba"):
            return {"what": "A selective state-space layer: a fast, recurrent-style memory whose update speed (Δ) "
                            "depends on each token.",
                    "why": "Long sequences at a cost that grows linearly with their length."}
    if rule is not None:
        try:
            out = rule(node, graph, batched)
        except Exception:  # noqa: BLE001 - an explanation must never break a page
            out = None
        if out:
            return {k: v for k, v in out.items() if v}
    i, o = _io(node, batched)
    p = node.n_params
    if not (i and o) and not p:
        return None
    sizes = f"{i} → {o}" if i and o else ""
    learned = f"{_n(p)} learned numbers" if p else ""
    return {"what": f"A {node.type_name} layer" + (": " if sizes or learned else ".")
                    + ", ".join(x for x in (sizes, learned) if x) + ("." if sizes or learned else "")}


_NOUNS = {  # family → (one, many), in the words a newcomer uses
    "embedding": ("embedding", "embeddings"), "conv": ("convolution", "convolutions"),
    "attention": ("attention layer", "attention layers"), "recurrent": ("recurrent layer", "recurrent layers"),
    "graph": ("message-passing layer", "message-passing layers"), "norm": ("normalization", "normalizations"),
    "activation": ("nonlinearity", "nonlinearities"), "pool": ("pooling", "poolings"),
    "linear": ("dense layer", "dense layers"),
}


def describe(graph: "ModelGraph", batched: bool = True, detail: bool = True) -> str:
    """The whole network in one sentence: what kind it is, what it turns into what, its layers by kind and its size.
    "A convolutional network that turns a 1 × 28 × 28 input into 10 values, through 7 layers — 1 convolution, …".
    ``detail=False`` leaves out the layer counts (a two-line video caption)."""
    from collections import Counter

    from .. import theme

    order = [graph.nodes[i] for i in graph.topo_order()]
    layers = [n for n in order if n.op in ("module", "function") and not n.attrs.get("unused")]
    fams = Counter()
    for n in layers:
        f = theme.kind_family(n.kind)
        if f in _NOUNS:
            fams[f] += 1
    residual = sum(1 for n in layers if _residual(graph, n))
    if fams["attention"]:
        what = "A transformer-style network" if residual and fams["norm"] else "An attention network"
    elif fams["recurrent"]:
        what = "A recurrent network"
    elif fams["graph"]:
        what = "A graph neural network"
    elif fams["conv"]:
        what = "A convolutional network"
    elif fams["linear"] and set(fams) <= {"linear", "activation", "norm"}:
        what = "A fully connected network (an MLP)"
    else:
        what = "A network"
    ins = [n for n in order if n.op == "input"]
    outs = [n for n in order if n.op == "output"]
    near = set()  # what the first input feeds within two steps (an Embedding there: the input is token ids)
    for s1 in (_succ(graph, ins[0].id) if ins else []):
        near.add(s1)
        near.update(_succ(graph, s1))
    emb = next((graph.nodes[i] for i in near if i in graph.nodes and graph.nodes[i].kind == "Embedding"), None)

    def dims(shape) -> list[int]:
        d = [int(v) for v in (shape or ())]
        return d[1:] if batched and len(d) >= 2 else d

    def tensor(shape, noun: str) -> str:
        d = dims(shape)
        if not d:
            return "a single number"
        if len(d) == 1:
            return f"{_n(d[0])} values"
        txt = " × ".join(f"{v:,}" for v in d)
        return f"{_article(txt)} {txt} {noun}"

    inp = out = ""
    if ins and ins[0].out_shapes and ins[0].out_shapes[0]:
        src = ins[0].out_shapes[0]
        if emb is not None:
            inp = f"a sequence of {_n(_count(src, batched))} token ids"
        else:
            inp = tensor(src, "input") if len(ins) == 1 else f"{len(ins)} inputs"
    last = next((n for n in reversed(layers)), None)
    if outs and outs[0].in_shapes and outs[0].in_shapes[0]:
        shape = outs[0].in_shapes[0]
        d = dims(shape)
        info = emb.role_param("weight") if emb is not None and "weight" in emb.roles else None
        vocab = int(info.shape[0]) if info is not None and info.shape else None
        if vocab and len(d) == 2 and d[-1] == vocab:  # a language model's head
            out = f"a score for each of its {_n(vocab)} tokens at each of the {_n(d[0])} positions"
        else:
            out = tensor(shape, "tensor")
            if last is not None and last.kind in ("Softmax", "LogSoftmax") and out.endswith(" values"):
                out = out.replace(" values", " probabilities" if last.kind == "Softmax" else " log-probabilities")
    n_layers = sum(fams.values())
    parts = [f"{c} {_NOUNS[f][0] if c == 1 else _NOUNS[f][1]}" for f, c in fams.items()]
    seen, params = set(), 0
    for n in graph.compute_nodes():
        for p in n.params.values():
            if p.trainable and p.path not in seen:
                seen.add(p.path)
                params += p.numel
    s = what
    if inp and out:
        s += f" that turns {inp} into {out}"
    if n_layers and detail:
        s += f", through {n_layers} layer{'s' if n_layers != 1 else ''} — " + ", ".join(parts) + " —"
    elif inp and out:
        s += ","
    if residual and detail:
        s += f" with {residual} residual connection{'s' if residual != 1 else ''} and"
    s += f" {'with ' if not (residual and detail) else ''}{_n(params)} learned numbers."
    if graph.framework == "weights":  # a file of tensors: no code ran, so the layers are only in name order
        s = s.replace(what, "A weights file", 1).replace(", through", ": its tensors make", 1)
        s += " The layers are drawn in name order: there is no code here, so no dataflow was traced."
    return s.replace(" — with", " — with").replace(",  ", ", ")


def _article(word: str) -> str:
    """"a" or "an" before a number written out: an 8, an 11, an 18, an 80…"""
    head = word.split()[0].replace(",", "")
    return "an" if head.startswith("8") or head in ("11", "18") or (head.startswith("18") and len(head) in (5, 6)) \
        or (head.startswith("11") and len(head) in (5, 6)) else "a"


def _succ(g: "ModelGraph", nid: str) -> list[str]:
    return [e.dst for e in g.edges if e.src == nid]


def caption(node: "Node", graph: "ModelGraph | None" = None, batched: bool = True) -> str | None:
    """One sentence for a video caption: the ``what`` of :func:`explain`."""
    e = explain(node, graph, batched)
    return e["what"] if e else None


__all__ = ["explain", "caption", "describe"]
