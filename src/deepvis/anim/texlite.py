"""Render the LaTeX subset deepvis emits as Pango markup (for manim's MarkupText).

Used when no TeX toolchain (latex + dvisvgm) is available: variables become italic,
``^{…}``/``_{…}`` become real super/subscripts, Greek letters and operators map to Unicode.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε", "zeta": "ζ",
    "eta": "η", "theta": "θ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π", "rho": "ρ",
    "sigma": "σ", "tau": "τ", "phi": "φ", "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω", "ell": "ℓ",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
}
SYMBOLS = {
    "cdot": "·", "odot": "⊙", "star": "⋆", "circ": "∘", "times": "×", "in": "∈", "top": "⊤", "ldots": "…",
    "cdots": "⋯", "vdots": "⋮", "sum": "Σ", "prod": "Π", "infty": "∞", "le": "≤", "leq": "≤", "ge": "≥",
    "geq": "≥", "approx": "≈", "to": "→", "mapsto": "↦", "partial": "∂", "nabla": "∇", "pm": "±", "neq": "≠",
    "quad": " ", "qquad": "  ", ",": " ", ";": " ", ":": " ", " ": " ", "!": "",
    "{": "{", "}": "}", "%": "%", "_": "_", "&": "&amp;", "#": "#", "lvert": "|", "rvert": "|", "|": "‖",
}
UPRIGHT_FUNCS = {"tanh", "exp", "log", "max", "min", "sin", "cos", "arg", "det", "sgn"}
BLACKBOARD = {"R": "ℝ", "N": "ℕ", "Z": "ℤ", "Q": "ℚ", "C": "ℂ"}
_DROP = {"textstyle", "displaystyle", "scriptstyle", "limits", "nolimits", "big", "Big", "bigl", "bigr"}


class _P:
    def __init__(self, s: str):
        self.s, self.i = s, 0

    def peek(self) -> str:
        return self.s[self.i] if self.i < len(self.s) else ""

    def command(self) -> str:
        self.i += 1  # backslash
        if self.i < len(self.s) and not self.s[self.i].isalpha():
            c = self.s[self.i]
            self.i += 1
            return c
        j = self.i
        while j < len(self.s) and self.s[j].isalpha():
            j += 1
        name = self.s[self.i:j]
        self.i = j
        if self.peek() == " ":  # a space after a control word only terminates it
            self.i += 1
        return name

    def arg(self, upright: bool) -> str:
        while self.peek() == " ":
            self.i += 1
        c = self.peek()
        if c == "{":
            self.i += 1
            out = self.seq(upright)
            if self.peek() == "}":
                self.i += 1
            return out
        if c == "\\":
            return self.atom(upright)
        self.i += 1
        return self.char(c, upright)

    def char(self, c: str, upright: bool) -> str:
        if c.isalpha() and c.isascii() and not upright:
            return f"<i>{escape(c)}</i>"
        return escape(c)

    def atom(self, upright: bool) -> str:
        c = self.peek()
        if c == "\\":
            name = self.command()
            if name in ("mathrm", "operatorname", "text", "textrm", "mathsf", "textsf"):
                return self.arg(True)
            if name in ("mathit", "textit"):
                return f"<i>{self.arg(True)}</i>"
            if name in ("mathbf", "textbf", "boldsymbol"):
                return f"<b>{self.arg(upright)}</b>"
            if name == "texttt":
                return f"<tt>{self.arg(True)}</tt>"
            if name == "textcolor":  # \\textcolor{#58C4DD}{W} — a layer's colour (deepvis.equations.expr.painted)
                color = _strip(self.arg(True)).strip()
                body = self.arg(upright)
                return f'<span foreground="{escape(color)}">{body}</span>' if color.startswith("#") else body
            if name == "htmlData":  # a page's link to the layer: nothing to draw in a video
                self.arg(True)
                return self.arg(upright)
            if name == "mathbb":
                inner = self.arg(True)
                return BLACKBOARD.get(inner, inner)
            if name == "hat":
                return self.arg(upright) + "̂"
            if name in ("bar", "overline"):
                return self.arg(upright) + "̄"
            if name == "frac":
                a, b = self.arg(upright), self.arg(upright)
                wrap = lambda t: t if len(_strip(t)) <= 1 else f"({t})"  # noqa: E731
                return f"{wrap(a)}/{wrap(b)}"
            if name == "tfrac":
                a, b = self.arg(upright), self.arg(upright)
                return f"{a}/{b}"
            if name == "sqrt":
                return f"√({self.arg(upright)})"
            if name in ("left", "right", "bigl", "bigr"):
                d = self.peek()
                if d == "\\":
                    sym = self.command()
                    return SYMBOLS.get(sym, "")
                self.i += 1
                return "" if d == "." else escape(d)
            if name == "underbrace":
                a = self.arg(upright)
                if self.peek() == "_":
                    self.i += 1
                    b = self.arg(True)
                    return f"{a} ({b})"
                return a
            if name in ("begin", "end"):
                self.arg(True)
                return ""
            if name in GREEK:
                g = GREEK[name]
                return g if upright or name[0].isupper() else f"<i>{g}</i>"
            if name in SYMBOLS:
                return SYMBOLS[name]
            if name in UPRIGHT_FUNCS:
                return name
            if name in _DROP:
                return ""
            if name == "\\":
                return "\n"
            return escape(name)
        if c == "{":
            self.i += 1
            out = self.seq(upright)
            if self.peek() == "}":
                self.i += 1
            return out
        self.i += 1
        if c == "&":
            return ""
        if c == "~":
            return " "
        return self.char(c, upright)

    def seq(self, upright: bool = False) -> str:
        out = []
        while self.i < len(self.s) and self.peek() != "}":
            c = self.peek()
            if c == "^":
                self.i += 1
                out.append(f"<sup>{self.arg(upright)}</sup>")
            elif c == "_":
                self.i += 1
                out.append(f"<sub>{self.arg(upright)}</sub>")
            elif c == " ":
                self.i += 1
                if upright or (out and not out[-1].endswith(" ")):
                    out.append(" ")
            else:
                out.append(self.atom(upright))
        return "".join(out)


def _strip(markup: str) -> str:
    import re

    return re.sub(r"<[^>]+>", "", markup)


def split_painted(latex: str) -> tuple[str, dict[str, str]]:
    """Painted LaTeX → (plain LaTeX, {coloured sub-expression: "#rrggbb"}), for real TeX: manim's MathTex colours
    isolated substrings itself (``tex_to_color_map``) but ignores ``\\textcolor``; ``\\htmlData`` is dropped."""
    out, colors, i, n = [], {}, 0, len(latex)

    def group(j: int) -> tuple[str, int]:  # the {…} starting at latex[j] (balanced), and the index after it
        depth, k = 0, j
        while k < n:
            if latex[k] == "\\":
                k += 2
                continue
            if latex[k] == "{":
                depth += 1
            elif latex[k] == "}":
                depth -= 1
                if depth == 0:
                    return latex[j + 1:k], k + 1
            k += 1
        return latex[j + 1:], n

    while i < n:
        if latex.startswith("\\textcolor{", i):
            color, j = group(i + len("\\textcolor"))
            body, i = group(j)
            plain, inner = split_painted(body)
            colors.update(inner)
            if color.startswith("#"):
                colors[plain] = color
            out.append(plain)
        elif latex.startswith("\\htmlData{", i):
            _, j = group(i + len("\\htmlData"))
            body, i = group(j)
            plain, inner = split_painted(body)
            colors.update(inner)
            out.append(plain)
        else:
            out.append(latex[i])
            i += 1
    return "".join(out), colors


def latex_to_pango(latex: str) -> str:
    """``W^{(1)}\\,x + b^{(1)}`` → ``<i>W</i><sup>(1)</sup> <i>x</i> + <i>b</i><sup>(1)</sup>``."""
    p = _P(latex)
    out = []
    while p.i < len(p.s):
        out.append(p.seq())
        if p.peek() == "}":  # unbalanced brace: skip it
            p.i += 1
    s = "".join(out)
    return s.replace("-", "−") if "<tt>" not in s else s  # typographic minus
