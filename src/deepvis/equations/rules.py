"""Per-kind equation rules, function definitions and the parameter symbol table.

Register your own::

    @dv.equation_rule("MyGate")          # canonical kind or class name
    def my_gate(node, args, ctx):
        W = ctx.param(node, "weight")
        return Apply(r"\\mathrm{gate}", "gate", [Prod(W, args[0])])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..core.graph import ModelGraph, Node
from ..core.kinds import ACTIVATION_KINDS, CONV_KINDS, NORM_KINDS
from .expr import Add, Apply, Expr, Frac, Postfix, Prod, Raw, Sym, paint, sub, sup, tex_escape

RULES: dict[str, Callable] = {}

_ROLE_LETTER = {"weight": ("W", "W"), "bias": ("b", "b"), "scale": (r"\gamma", "γ"),
                "running_mean": (r"\mu", "μ"), "running_var": (r"\sigma^2", "σ²"),
                "mean": (r"\mu", "μ"), "var": (r"\sigma^2", "σ²")}
_KIND_LETTER = {"Embedding": ("E", "E")}


def equation_rule(kind: str | type):
    """Decorator registering ``fn(node, args, ctx) -> Expr`` for a kind or class name."""
    key = kind if isinstance(kind, str) else kind.__name__

    def deco(fn):
        RULES[key] = fn
        return fn

    return deco


@dataclass
class LegendEntry:
    latex: str
    text: str
    path: str
    shape: tuple
    transposed: bool = False

    desc: bool = False  # ``path`` is a free-text description
    node: str | None = None  # the layer the symbol belongs to (its colour when painted)

    def latex_line(self) -> str:
        sym = paint(self.latex, self.node)
        if self.desc:
            return rf"{sym}:\ \text{{{tex_escape(self.path)}}}"
        dims = r"\times ".join(str(s) for s in self.shape) if self.shape else "1"
        path = tex_escape(self.path)
        t = r"^{\top}" if self.transposed else ""
        return rf"{sym} = \texttt{{{path}}}{t} \in \mathbb{{R}}^{{{dims}}}"

    def text_line(self) -> str:
        if self.desc:
            return f"{self.text}: {self.path}"
        dims = "×".join(str(s) for s in self.shape) if self.shape else "1"
        return f"{self.text} = {self.path}{'ᵀ' if self.transposed else ''} ∈ ℝ^{dims}"


class SymbolTable:
    """Assigns W⁽ℓ⁾, b⁽ℓ⁾ ... to parameters (layer index ℓ in topological order)."""

    def __init__(self, graph: ModelGraph, style: str = "index", numbering: ModelGraph | None = None):
        self.graph = graph
        self.style = style
        self.layer_of: dict[str, int] = {}
        self.legend: dict[str, LegendEntry] = {}
        k = 0
        for n in (numbering or graph).param_nodes():
            owner = n.module or n.id
            if owner not in self.layer_of:
                k += 1
                self.layer_of[owner] = k

    def layer(self, node: Node) -> int:
        return self.layer_of.get(node.module or node.id, 0)

    def _tag(self, node: Node) -> tuple[str, str]:
        if self.style == "path":
            p = tex_escape(node.module or node.id)
            return rf"_{{\texttt{{{p}}}}}", sub(node.module or node.id)
        ell = self.layer(node)
        return rf"^{{({ell})}}", sup(f"({ell})")

    def param(self, node: Node, role: str) -> Expr | None:
        r = node.roles.get(role)
        if r is None:
            return None
        info = node.params.get(r.name)
        if info is None:
            return None
        for e in self.legend.values():  # a tied weight keeps the symbol it already has (lm_head.weight is E)
            if not e.desc and e.path == info.path and not e.transposed and (r.out_axis in (None, 0)):
                return Sym(e.latex, e.text, node.id)
        letter = _KIND_LETTER.get(node.kind) if role == "weight" else None
        base_l, base_t = letter or _ROLE_LETTER.get(role, (rf"\theta_{{{role}}}", f"θ_{role}"))
        if node.kind in NORM_KINDS and role == "bias":
            base_l, base_t = r"\beta", "β"
        tl, tt = self._tag(node)
        sym_l, sym_t = base_l + tl, base_t + tt
        shape = tuple(info.shape)
        transposed = False
        if r.out_axis is not None and r.in_axis is not None and len(shape) >= 2:
            oa, ia = r.out_axis % len(shape), r.in_axis % len(shape)
            rest = [shape[i] for i in range(len(shape)) if i not in (oa, ia)]
            shape = (shape[oa], shape[ia], *rest)
            transposed = oa > ia and len(info.shape) == 2
            if node.kind == "Embedding":  # E[x] indexes rows as stored: keep (vocab, dim)
                shape, transposed = tuple(info.shape), False
        self.legend.setdefault(sym_l, LegendEntry(sym_l, sym_t, info.path, shape, transposed, node=node.id))
        return Sym(sym_l, sym_t, node.id)

    def free_param(self, node: Node) -> Expr:
        info = next(iter(node.params.values()))
        name = info.path.split(".")[-1]
        sym_l = rf"P_{{\texttt{{{tex_escape(name)}}}}}"
        self.legend.setdefault(sym_l, LegendEntry(sym_l, f"P_{name}", info.path, tuple(info.shape), node=node.id))
        return Sym(sym_l, f"P_{name}", node.id)


# ------------------------------------------------------------------ rules
def _fn(latex: str, text: str):
    def rule(node, args, ctx):
        return Apply(latex, text, args[:1] or [Sym("z")], node.id)

    return rule


_ACTS = {
    "ReLU": (r"\mathrm{ReLU}", "ReLU"), "LeakyReLU": (r"\mathrm{LReLU}", "LReLU"),
    "GELU": (r"\mathrm{GELU}", "GELU"), "SiLU": (r"\mathrm{SiLU}", "SiLU"), "Sigmoid": (r"\sigma", "σ"),
    "Tanh": (r"\tanh", "tanh"), "Softmax": (r"\mathrm{softmax}", "softmax"),
    "LogSoftmax": (r"\log\mathrm{softmax}", "log softmax"), "ELU": (r"\mathrm{ELU}", "ELU"),
    "Softplus": (r"\mathrm{softplus}", "softplus"), "Mish": (r"\mathrm{Mish}", "Mish"),
    "Hardtanh": (r"\mathrm{hardtanh}", "hardtanh"), "exp": (r"\exp", "exp"), "log": (r"\log", "log"),
    "abs": (r"\mathrm{abs}", "abs"), "sqrt": (r"\sqrt", "√"), "rsqrt": (r"\mathrm{rsqrt}", "rsqrt"),
    "erf": (r"\mathrm{erf}", "erf"), "Flatten": (r"\mathrm{vec}", "vec"), "Reshape": (r"\mathrm{reshape}", "reshape"),
    "mean": (r"\mathrm{mean}", "mean"), "sum": (r"\textstyle\sum", "Σ"), "max": (r"\max", "max"),
    "MaxPool1d": (r"\mathrm{MaxPool}", "MaxPool"), "MaxPool2d": (r"\mathrm{MaxPool}", "MaxPool"),
    "AvgPool1d": (r"\mathrm{AvgPool}", "AvgPool"), "AvgPool2d": (r"\mathrm{AvgPool}", "AvgPool"),
    "AdaptiveAvgPool1d": (r"\mathrm{GAP}", "GAP"), "AdaptiveAvgPool2d": (r"\mathrm{GAP}", "GAP"),
    "AdaptiveMaxPool2d": (r"\mathrm{GMP}", "GMP"),
    "Hardswish": (r"\mathrm{hswish}", "hswish"), "Hardsigmoid": (r"\mathrm{hsigmoid}", "hsigmoid"),
    "ReLU6": (r"\mathrm{ReLU6}", "ReLU6"), "PReLU": (r"\mathrm{PReLU}", "PReLU"), "SELU": (r"\mathrm{SELU}", "SELU"),
    "CELU": (r"\mathrm{CELU}", "CELU"), "Softsign": (r"\mathrm{softsign}", "softsign"),
    "Upsample": (r"\mathrm{Upsample}", "Upsample"), "PixelShuffle": (r"\mathrm{PixelShuffle}", "PixelShuffle"),
    "LSTM": (r"\mathrm{LSTM}", "LSTM"), "GRU": (r"\mathrm{GRU}", "GRU"), "RNN": (r"\mathrm{RNN}", "RNN"),
}
for _k, (_l, _t) in _ACTS.items():
    RULES[_k] = _fn(_l, _t)


def _passthrough(node, args, ctx):
    return args[0] if args else Sym("z")


RULES["Dropout"] = _passthrough
RULES["Identity"] = _passthrough
RULES["getitem"] = lambda node, args, ctx: (
    Postfix(args[0], rf"_{{[{node.attrs['index']}]}}", f"[{node.attrs['index']}]") if "index" in node.attrs
    else Postfix(args[0], r"_{[\ldots]}", "[…]"))
RULES["Constant"] = lambda node, args, ctx: Sym(rf"\mathrm{{{tex_escape(node.attrs.get('value', 'c'))}}}",
                                               node.attrs.get("value", "c"), node.id)
RULES["Transpose"] = lambda node, args, ctx: Postfix(args[0], r"^{\top}", "ᵀ")
RULES["square"] = lambda node, args, ctx: Postfix(args[0], "^{2}", "²")
RULES["pow"] = lambda node, args, ctx: Postfix(args[0], "^{p}", "ᵖ")
RULES["neg"] = lambda node, args, ctx: Add([args[0]], [-1])


@equation_rule("Linear")
def _linear(node, args, ctx):
    x = args[0] if args else Sym("x")
    W, b = ctx.symbols.param(node, "weight"), ctx.symbols.param(node, "bias")
    if W is None:
        return Apply(r"\mathrm{Linear}", "Linear", [x], node.id)
    term = Prod(W, x)
    return Add([term, b]) if b is not None else term


def _conv(node, args, ctx):
    x = args[0] if args else Sym("x")
    W, b = ctx.symbols.param(node, "weight"), ctx.symbols.param(node, "bias")
    if W is None:
        return Apply(r"\mathrm{Conv}", "Conv", [x], node.id)
    term = Prod(W, x, r"\star", "⋆")
    return Add([term, b]) if b is not None else term


for _k in CONV_KINDS:
    RULES[_k] = _conv


def _conv_t(node, args, ctx):
    x = args[0] if args else Sym("x")
    W, b = ctx.symbols.param(node, "weight"), ctx.symbols.param(node, "bias")
    if W is None:
        return Apply(r"\mathrm{ConvT}", "ConvT", [x], node.id)
    term = Prod(W, x, r"\star^{\top}", "⋆ᵀ")
    return Add([term, b]) if b is not None else term


for _k in ("ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d"):
    RULES[_k] = _conv_t


def _norm(abbr_l: str, abbr_t: str):
    def rule(node, args, ctx):
        g, b = ctx.symbols.param(node, "scale"), ctx.symbols.param(node, "bias")
        subs = [s for s in (g, b) if s is not None]
        fn_l = abbr_l + (r"_{" + ",".join(s.latex() for s in subs) + "}" if subs else "")
        fn_t = abbr_t + ("_{" + ",".join(s.text() for s in subs) + "}" if subs else "")
        return Apply(fn_l, fn_t, args[:1] or [Sym("z")], node.id)

    return rule


RULES["LayerNorm"] = _norm(r"\mathrm{LN}", "LN")
RULES["RMSNorm"] = _norm(r"\mathrm{RMSN}", "RMSN")
RULES["BatchNorm"] = _norm(r"\mathrm{BN}", "BN")
RULES["GroupNorm"] = _norm(r"\mathrm{GN}", "GN")


@equation_rule("Embedding")
def _embedding(node, args, ctx):
    E = ctx.symbols.param(node, "weight")
    x = args[0] if args else Sym("x")
    if E is None:
        return Apply(r"\mathrm{Embed}", "Embed", [x], node.id)
    return Raw(rf"{E.latex()}\left[{x.latex()}\right]", f"{E.text()}[{x.text()}]")


@equation_rule("MultiheadAttention")
def _mha(node, args, ctx):
    tl, tt = ctx.symbols._tag(node)
    seen, uniq = set(), []
    for a in args:  # self-attention passes the same tensor as q, k and v
        if a.latex() not in seen:
            seen.add(a.latex())
            uniq.append(a)
    return Apply(r"\mathrm{MHA}" + tl, "MHA" + tt, uniq or [Sym("z")], node.id)


RULES["Attention"] = lambda node, args, ctx: Apply(r"\mathrm{Attn}", "Attn", args or [Sym("z")], node.id)


@equation_rule("add")
def _add(node, args, ctx):
    const = node.attrs.get("const")
    terms = list(args) + ([Raw(str(const[0]), str(const[0]))] if const else [])
    return Add(terms) if len(terms) > 1 else (terms[0] if terms else Sym("z"))


@equation_rule("sub")
def _sub(node, args, ctx):
    if len(args) >= 2:
        return Add([args[0], args[1]], [1, -1])
    return args[0]


@equation_rule("mul")
def _mul(node, args, ctx):
    const = node.attrs.get("const")
    if len(args) >= 2:
        return Prod(args[0], args[1], r"\odot", "⊙")
    if const:
        return Prod(Raw(str(const[0]), str(const[0])), args[0], r"\cdot", "·")
    return args[0]


@equation_rule("div")
def _div(node, args, ctx):
    const = node.attrs.get("const")
    if len(args) >= 2:
        return Frac(args[0], args[1])
    if const:
        return Frac(args[0], Raw(str(const[0]), str(const[0])))
    return args[0]


@equation_rule("matmul")
def _matmul(node, args, ctx):
    if len(args) >= 2:
        return Prod(args[0], args[1])
    return Apply(r"\mathrm{matmul}", "matmul", args, node.id)


@equation_rule("Concat")
def _concat(node, args, ctx):
    return Raw(r"\left[" + r";\ ".join(a.latex() for a in args) + r"\right]",
               "[" + "; ".join(a.text() for a in args) + "]")


@equation_rule("Parameter")
def _param(node, args, ctx):
    info = next(iter(node.params.values()), None)
    if info is not None:  # a weight used functionally (e.g. tied embeddings) keeps its layer's symbol
        for e in ctx.symbols.legend.values():
            if e.path == info.path:
                return Sym(e.latex, e.text, node.id)
    return ctx.symbols.free_param(node)


@equation_rule("Scan")
def _scan(node, args, ctx):
    for r in ("weight", "bias") if node.roles else ():
        ctx.symbols.param(node, r)  # puts the scan's weights in the legend
    names = [p.name for p in node.params.values()]
    sub = r"_{\theta}" if names else ""
    return Apply(r"\mathrm{scan}" + sub, "scan" + ("_θ" if names else ""), args or [Sym("x")], node.id)


@equation_rule("Affine")
def _affine(node, args, ctx):
    x = args[0] if args else Sym("x")
    g, b = ctx.symbols.param(node, "weight"), ctx.symbols.param(node, "bias")
    t = Prod(g, x, r"\odot", "⊙") if g is not None else x
    return Add([t, b]) if b is not None else t


@equation_rule("Scope")
def _scope(node, args, ctx):
    """A named region of a pure-JAX function (``dv.scope("readout")``) reads as a function of that name."""
    lab = tex_escape((node.label or node.id).split(".")[-1])
    return Apply(rf"\mathrm{{{lab}}}", node.label or node.id, args or [Sym("z")], node.id)


def container_rule(node, args, ctx):
    name = tex_escape(node.type_name)
    lab = tex_escape((node.label or node.id).split(".")[-1])
    return Apply(rf"\mathrm{{{name}}}_{{\texttt{{{lab}}}}}", f"{node.type_name}_{lab}", args or [Sym("z")], node.id)


def opaque_rule(node, args, ctx):
    kind = tex_escape(node.kind)
    return Apply(rf"\operatorname{{{kind}}}", node.kind, args or [Sym("z")], node.id)


def rule_for(node: Node) -> Callable:
    if node.attrs.get("collapsed"):
        return container_rule
    rule = RULES.get(node.type_name) or RULES.get(node.kind) or opaque_rule
    act = _builtin_activation(node)
    if act is None:
        return rule
    latex, text = _ACTS[act]
    return lambda node, args, ctx: Apply(latex, text, [rule(node, args, ctx)], node.id)


def _builtin_activation(node: Node) -> str | None:
    """The activation a layer applies itself (Keras: ``Dense(10, activation="relu")``), as a kind with a symbol."""
    act = node.attrs.get("activation")
    if not isinstance(act, str) or not (node.kind == "Linear" or node.kind in CONV_KINDS
                                        or node.kind.startswith("ConvTranspose")):
        return None
    from ..core.kinds import function_kind

    kind = function_kind(act)
    return kind if kind in _ACTS else None


def fn_name(node: Node, ctx) -> tuple[str, str]:
    """Function symbol used by the compact (composition) form."""
    tl, tt = ctx.symbols._tag(node) if node.params else ("", "")
    if node.kind in _ACTS:
        return _ACTS[node.kind]
    if node.attrs.get("collapsed"):
        return rf"\mathrm{{{node.type_name}}}", node.type_name
    short = {"Linear": ("A", "A"), "LayerNorm": (r"\mathrm{LN}", "LN"), "BatchNorm": (r"\mathrm{BN}", "BN"),
             "RMSNorm": (r"\mathrm{RMSN}", "RMSN"), "Embedding": (r"\mathrm{Emb}", "Emb"),
             "MultiheadAttention": (r"\mathrm{MHA}", "MHA")}
    if node.kind in CONV_KINDS:
        return rf"\mathrm{{Conv}}{tl}", f"Conv{tt}"
    if node.kind in short:
        l, t = short[node.kind]
        return l + tl, t + tt
    k = tex_escape(node.kind)
    return rf"\operatorname{{{k}}}{tl}", node.kind + tt


# ------------------------------------------------------------ definitions
DEFINITIONS: dict[str, tuple[str, str]] = {
    "ReLU": (r"\mathrm{ReLU}(z) = \max(0, z)", "ReLU(z) = max(0, z)"),
    "LeakyReLU": (r"\mathrm{LReLU}(z) = \max(\alpha z, z)", "LReLU(z) = max(αz, z)"),
    "GELU": (r"\mathrm{GELU}(z) = z\,\Phi(z) = \tfrac{z}{2}\left(1 + \mathrm{erf}(z/\sqrt{2})\right)",
             "GELU(z) = z·Φ(z)"),
    "SiLU": (r"\mathrm{SiLU}(z) = z\,\sigma(z)", "SiLU(z) = z·σ(z)"),
    "Sigmoid": (r"\sigma(z) = \frac{1}{1 + e^{-z}}", "σ(z) = 1/(1 + e^(−z))"),
    "Tanh": (r"\tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}", "tanh(z) = (eᶻ − e⁻ᶻ)/(eᶻ + e⁻ᶻ)"),
    "Softmax": (r"\mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}", "softmax(z)ᵢ = e^(zᵢ) / Σⱼ e^(zⱼ)"),
    "LogSoftmax": (r"\log\mathrm{softmax}(z)_i = z_i - \log\sum_j e^{z_j}", "log softmax(z)ᵢ = zᵢ − log Σⱼ e^(zⱼ)"),
    "ELU": (r"\mathrm{ELU}(z) = \begin{cases} z & z > 0 \\ \alpha(e^{z} - 1) & z \le 0\end{cases}", "ELU"),
    "Softplus": (r"\mathrm{softplus}(z) = \log(1 + e^{z})", "softplus(z) = log(1 + eᶻ)"),
    "LayerNorm": (r"\mathrm{LN}_{\gamma,\beta}(z) = \gamma \odot \frac{z - \mu(z)}{\sqrt{\mathrm{Var}(z) + \epsilon}}"
                  r" + \beta", "LN(z) = γ ⊙ (z − μ(z))/√(Var(z) + ε) + β"),
    "RMSNorm": (r"\mathrm{RMSN}_{\gamma}(z) = \gamma \odot \frac{z}{\sqrt{\tfrac{1}{d}\sum_i z_i^2 + \epsilon}}",
                "RMSN(z) = γ ⊙ z / RMS(z)"),
    "BatchNorm": (r"\mathrm{BN}_{\gamma,\beta}(z) = \gamma \odot \frac{z - \hat\mu}{\sqrt{\hat\sigma^2 + \epsilon}}"
                  r" + \beta", "BN(z) = γ ⊙ (z − μ̂)/√(σ̂² + ε) + β"),
    "GroupNorm": (r"\mathrm{GN}_{\gamma,\beta}(z) = \gamma \odot \frac{z - \mu_g(z)}{\sqrt{\mathrm{Var}_g(z)"
                  r" + \epsilon}} + \beta", "GN(z) = γ ⊙ (z − μ_g)/σ_g + β"),
    "Conv": (r"(W \star x)_{c,i,j} = \sum_{k}\sum_{u,v} W_{c,k,u,v}\; x_{k,\; s i + u,\; s j + v}",
             "(W ⋆ x)_{c,i,j} = Σₖ Σᵤᵥ W_{c,k,u,v} x_{k, si+u, sj+v}"),
    "MultiheadAttention": (
        r"\mathrm{MHA}(z) = \left[\mathrm{head}_1; \dots; \mathrm{head}_h\right] W_O,\quad"
        r" \mathrm{head}_i = \mathrm{softmax}\!\left(\frac{(z W_Q^{i})(z W_K^{i})^{\top}}{\sqrt{d_k}}\right) z W_V^{i}",
        "MHA(z) = [head₁; …; headₕ] W_O,  headᵢ = softmax((zW_Qⁱ)(zW_Kⁱ)ᵀ/√d_k) zW_Vⁱ"),
    "Attention": (r"\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V",
                  "Attn(Q,K,V) = softmax(QKᵀ/√d_k) V"),
    "Flatten": (r"\mathrm{vec}(z)\ \text{stacks the entries of } z \text{ into a vector}", "vec(z) flattens z"),
    "AdaptiveAvgPool2d": (r"\mathrm{GAP}(z)_c = \frac{1}{HW}\sum_{i,j} z_{c,i,j}", "GAP(z)_c = mean over i,j"),
    "MaxPool2d": (r"\mathrm{MaxPool}(z)_{c,i,j} = \max_{u,v < k} z_{c,\, s i+u,\, s j+v}", "MaxPool: max over k×k"),
    "Embedding": (r"E[x]\ \text{selects row } x \text{ of the embedding table } E", "E[x] = row x of E"),
    "Hardswish": (r"\mathrm{hswish}(z) = z\,\frac{\mathrm{ReLU6}(z + 3)}{6}", "hswish(z) = z·ReLU6(z+3)/6"),
    "Hardsigmoid": (r"\mathrm{hsigmoid}(z) = \frac{\mathrm{ReLU6}(z + 3)}{6}", "hsigmoid(z) = ReLU6(z+3)/6"),
    "ReLU6": (r"\mathrm{ReLU6}(z) = \min(\max(0, z), 6)", "ReLU6(z) = min(max(0, z), 6)"),
    "ConvTranspose2d": (r"W \star^{\top} x\ \text{: transposed convolution (the adjoint of } W \star \cdot\text{)}",
                        "W ⋆ᵀ x: transposed convolution"),
    "LSTM": (r"c_t = f_t \odot c_{t-1} + i_t \odot \tanh(W_c[h_{t-1}; x_t] + b_c),\ h_t = o_t \odot \tanh(c_t)",
             "c_t = f_t⊙c_{t−1} + i_t⊙tanh(W_c[h_{t−1}; x_t] + b_c),  h_t = o_t⊙tanh(c_t)"),
    "Scan": (r"\mathrm{scan}_{\theta}(x)\ \text{iterates } h_t = f_{\theta}(h_{t-1}, x_t) \text{ over } t",
             "scan_θ(x) iterates h_t = f_θ(h_{t−1}, x_t) over t"),
    "GRU": (r"h_t = (1 - z_t) \odot n_t + z_t \odot h_{t-1},\ n_t = \tanh(W_n x_t + r_t \odot (U_n h_{t-1}))",
            "h_t = (1−z_t)⊙n_t + z_t⊙h_{t−1},  n_t = tanh(W_n x_t + r_t⊙(U_n h_{t−1}))"),
}


def definition_for(node: Node) -> tuple[str, str] | None:
    if node.kind in CONV_KINDS:
        return DEFINITIONS["Conv"]
    return DEFINITIONS.get(node.kind)


def is_elementwise_free(node: Node) -> bool:
    """Parameterless op that doesn't need its own line."""
    return not node.params and node.op == "function" or (node.kind in ACTIVATION_KINDS | {"Dropout", "Identity"}
                                                          and not node.params)
