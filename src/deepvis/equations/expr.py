"""A tiny expression tree that renders to both LaTeX and Unicode with correct parenthesization."""

from __future__ import annotations

import contextlib
import contextvars
import math
from typing import Callable

# precedence: higher binds tighter
P_SUM, P_PROD, P_UNARY, P_ATOM = 0, 1, 2, 3

_SUP = str.maketrans("0123456789+-()=ijklmnℓ", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁽⁾⁼ⁱʲᵏˡᵐⁿˡ")
_SUB = str.maketrans("0123456789+-()=ijklmnoprstuvx", "₀₁₂₃₄₅₆₇₈₉₊₋₍₎₌ᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ")


_TEX_SPECIAL = {"_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#", "$": r"\$", "{": r"\{", "}": r"\}",
                "\\": r"\backslash{}", "^": r"\hat{}", "~": r"\sim{}"}


def tex_escape(s) -> str:
    """Make an arbitrary identifier (module path, param name) safe inside ``\\text{}`` / ``\\texttt{}``."""
    return "".join(_TEX_SPECIAL.get(c, c) for c in str(s))


def sup(s: str) -> str:
    t = str(s).translate(_SUP)
    return t if all(c not in t for c in "abcdefghopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") else f"^({s})"


def sub(s: str) -> str:
    t = str(s).translate(_SUB)
    return t if t != str(s) or not str(s) else f"_{s}"


# ------------------------------------------------------------------------------------------ colour coding
class Painter:
    """How :meth:`Expr.latex` colours what it renders while :func:`painted` is active.

    ``color_of(node_id) -> "#rrggbb" | None`` colours a layer's symbols and function names (the same hue as the layer
    in the drawings: 3b1b's "colour as meaning"); ``tag_of(node_id) -> int | None`` wraps them in ``\\htmlData{n=…}``
    so a page can link a term to its layer; ``signs=(positive, negative)`` colours each term of a weighted sum by its
    weight's sign, like the lines of a neuron diagram."""

    def __init__(self, color_of: Callable | None = None, tag_of: Callable | None = None, signs=None):
        self.color_of, self.tag_of, self.signs = color_of, tag_of, signs


_PAINT: contextvars.ContextVar = contextvars.ContextVar("deepvis_paint", default=None)


@contextlib.contextmanager
def painted(color_of: Callable | None = None, tag_of: Callable | None = None, signs=None):
    """Render LaTeX colour-coded inside this block (see :class:`Painter`). Plain LaTeX everywhere else, so
    ``net.equation()`` stays clean; the dashboard and the videos opt in."""
    token = _PAINT.set(Painter(color_of, tag_of, signs))
    try:
        yield
    finally:
        _PAINT.reset(token)


def paint(latex: str, node: str | None) -> str:
    p = _PAINT.get()
    if p is None or not node or latex.startswith((r"\htmlData", r"\textcolor")):
        return latex
    c = p.color_of(node) if p.color_of else None
    t = p.tag_of(node) if p.tag_of else None
    if c:
        latex = rf"\textcolor{{{c}}}{{{latex}}}"
    if t is not None:
        latex = rf"\htmlData{{n={t}}}{{{latex}}}"
    return latex


def paint_sign(latex: str, sign: int) -> str:
    p = _PAINT.get()
    if p is None or not p.signs:
        return latex
    return rf"\textcolor{{{p.signs[0] if sign >= 0 else p.signs[1]}}}{{{latex}}}"


class Expr:
    prec = P_ATOM

    def latex(self) -> str:  # pragma: no cover
        raise NotImplementedError

    def text(self) -> str:  # pragma: no cover
        raise NotImplementedError

    def size(self) -> int:
        return 1

    def wrap_latex(self, min_prec: int) -> str:
        s = self.latex()
        return rf"\left({s}\right)" if self.prec < min_prec else s

    def wrap_text(self, min_prec: int) -> str:
        s = self.text()
        return f"({s})" if self.prec < min_prec else s

    def __repr__(self) -> str:
        return self.text()


class Sym(Expr):
    def __init__(self, latex: str, text: str | None = None, node: str | None = None):
        self._l, self._t, self.node = latex, text if text is not None else latex, node

    def latex(self):
        return paint(self._l, self.node)

    def text(self):
        return self._t


class Num(Expr):
    def __init__(self, value: float, precision: int = 3):
        self.value, self.precision = float(value), precision
        self.prec = P_ATOM if value >= 0 else P_UNARY

    def fmt(self) -> str:
        v = self.value
        if v == 0:
            return "0"
        if abs(v) >= 1e4 or abs(v) < 10 ** (-self.precision):
            m, e = f"{v:.{self.precision - 1}e}".split("e")
            return f"{m}e{int(e)}"
        digits = max(self.precision - int(math.floor(math.log10(abs(v)))) - 1, 0)
        return f"{v:.{digits}f}"

    def latex(self):
        s = self.fmt()
        if "e" in s:
            m, e = s.split("e")
            return rf"{m}\times 10^{{{e}}}"
        return s

    def text(self):
        return self.fmt()


class Raw(Expr):
    def __init__(self, latex: str, text: str, prec: int = P_ATOM):
        self._l, self._t, self.prec = latex, text, prec

    def latex(self):
        return self._l

    def text(self):
        return self._t


class Apply(Expr):
    """``f(args)``; ``fn`` is LaTeX, ``fn_text`` Unicode."""

    def __init__(self, fn: str, fn_text: str, args: list[Expr], node: str | None = None):
        self.fn, self.fn_text, self.args, self.node = fn, fn_text, args, node

    def latex(self):
        inner = ",\\, ".join(a.latex() for a in self.args)
        return rf"{paint(self.fn, self.node)}\!\left({inner}\right)"

    def text(self):
        return f"{self.fn_text}(" + ", ".join(a.text() for a in self.args) + ")"

    def size(self):
        return 1 + sum(a.size() for a in self.args)


class Add(Expr):
    prec = P_SUM

    def __init__(self, terms: list[Expr], signs: list[int] | None = None, weighted: bool = False):
        self.terms = terms
        self.signs = signs or [1] * len(terms)
        self.weighted = weighted  # a weighted sum: each term's sign is its weight's (coloured when painted)

    def latex(self):
        out = []
        for i, (t, s) in enumerate(zip(self.terms, self.signs)):
            body = t.wrap_latex(P_PROD if s < 0 else P_SUM)
            sign = -s if isinstance(t, Num) and t.value < 0 else s
            if isinstance(t, Num) and t.value < 0:
                body, s = Num(-t.value, t.precision).latex(), -s
            if self.weighted and not isinstance(t, Ellipsis_):
                body = paint_sign(body, sign)
            if i == 0:
                out.append(("-" if s < 0 else "") + body)
            else:
                out.append((" - " if s < 0 else " + ") + body)
        return "".join(out)

    def text(self):
        out = []
        for i, (t, s) in enumerate(zip(self.terms, self.signs)):
            body = t.wrap_text(P_PROD if s < 0 else P_SUM)
            if isinstance(t, Num) and t.value < 0:
                body, s = Num(-t.value, t.precision).text(), -s
            if i == 0:
                out.append(("−" if s < 0 else "") + body)
            else:
                out.append((" − " if s < 0 else " + ") + body)
        return "".join(out)

    def size(self):
        return 1 + sum(t.size() for t in self.terms)


class Prod(Expr):
    """Juxtaposition / binary product: ``op`` in {"", "\\odot", "\\cdot", "\\star"}."""

    prec = P_PROD

    def __init__(self, a: Expr, b: Expr, op: str = "", op_text: str = ""):
        self.a, self.b, self.op, self.op_text = a, b, op, op_text

    def latex(self):
        sep = rf" {self.op} " if self.op else r"\,"
        return self.a.wrap_latex(P_PROD) + sep + self.b.wrap_latex(P_UNARY if not self.op else P_PROD + 1)

    def text(self):
        sep = f" {self.op_text} " if self.op_text else ""
        return self.a.wrap_text(P_PROD) + sep + self.b.wrap_text(P_UNARY if not self.op_text else P_PROD + 1)

    def size(self):
        return 1 + self.a.size() + self.b.size()


class Frac(Expr):
    prec = P_ATOM

    def __init__(self, num: Expr, den: Expr):
        self.num, self.den = num, den

    def latex(self):
        return rf"\frac{{{self.num.latex()}}}{{{self.den.latex()}}}"

    def text(self):
        return f"{self.num.wrap_text(P_ATOM)}/{self.den.wrap_text(P_ATOM)}"

    def size(self):
        return 1 + self.num.size() + self.den.size()


class Postfix(Expr):
    """``x^T``, ``x_{[...]}`` etc."""

    prec = P_UNARY

    def __init__(self, a: Expr, latex_suffix: str, text_suffix: str):
        self.a, self.ls, self.ts = a, latex_suffix, text_suffix

    def latex(self):
        base = self.a.wrap_latex(P_ATOM)
        if self.ls.startswith(("_", "^")):  # brace the base so E^{(1)} + ^{\top} never double-scripts
            base = "{" + base + "}"
        return base + self.ls

    def text(self):
        return self.a.wrap_text(P_ATOM) + self.ts

    def size(self):
        return 1 + self.a.size()


class Compose(Expr):
    """``(f_L ∘ ⋯ ∘ f_1)(x)``."""

    prec = P_ATOM

    def __init__(self, fns: list[tuple[str, str]], arg: Expr):
        self.fns, self.arg = fns, arg

    def latex(self):
        inner = r" \circ ".join(f for f, _ in reversed(self.fns))
        return rf"\left({inner}\right)\!\left({self.arg.latex()}\right)"

    def text(self):
        return "(" + " ∘ ".join(t for _, t in reversed(self.fns)) + f")({self.arg.text()})"

    def size(self):
        return len(self.fns) + self.arg.size()


class Ellipsis_(Expr):
    prec = P_ATOM

    def __init__(self, n: int):
        self.n = n

    def latex(self):
        return rf"\underbrace{{\cdots}}_{{{self.n}\text{{ more}}}}"

    def text(self):
        return f"…({self.n} more)"
