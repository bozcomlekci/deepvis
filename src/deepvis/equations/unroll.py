"""Unroll a network into mathematics.

Modes
-----
``system``    one line per layer:  h₁ = ReLU(W⁽¹⁾x + b⁽¹⁾), …, ŷ = softmax(W⁽³⁾h₂ + b⁽³⁾)
``expanded``  substitute everything that is used once:  ŷ = softmax(W⁽³⁾ReLU(W⁽²⁾ReLU(W⁽¹⁾x + b⁽¹⁾) + b⁽²⁾) + b⁽³⁾)
``compact``   function composition:  ŷ = (softmax ∘ A⁽³⁾ ∘ ReLU ∘ A⁽²⁾ ∘ …)(x)
``scalar``    one neuron with real numbers:  z₃ = 0.82·a₁₂ − 0.61·a₇ + … + 0.10
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..core.graph import ModelGraph, Node
from ..core.kinds import ACTIVATION_KINDS, CONV_KINDS
from .expr import Add, Apply, Compose, Ellipsis_, Expr, Num, Prod, Raw, Sym, sub, sup, tex_escape
from .rules import _ACTS, DEFINITIONS, LegendEntry, SymbolTable, _builtin_activation, definition_for, fn_name, rule_for

if TYPE_CHECKING:  # pragma: no cover
    from ..net import Net

_PASSTHROUGH = {"Dropout", "Identity"}
_STRUCTURAL = {"Flatten", "Reshape", "Transpose", "getitem", "Concat"}
_ELEMENTWISE = (ACTIVATION_KINDS - {"Softmax", "LogSoftmax"}) | _PASSTHROUGH


@dataclass
class Line:
    lhs: Expr
    rhs: Expr
    node: str | None = None
    note: str | None = None


@dataclass
class Equation:
    lines: list[Line]
    mode: str
    definitions: list[tuple[str, str]] = field(default_factory=list)
    legend: list[LegendEntry] = field(default_factory=list)
    title: str = ""

    @property
    def latex(self) -> str:
        rows = []
        for ln in self.lines:
            note = rf"\qquad \text{{{ln.note}}}" if ln.note else ""
            rows.append(rf"{ln.lhs.latex()} &= {ln.rhs.latex()}{note}")
        return r"\begin{aligned}" + r" \\ ".join(rows) + r"\end{aligned}"

    @property
    def text(self) -> str:
        lhs_w = max((len(ln.lhs.text()) for ln in self.lines), default=0)
        out = [f"{ln.lhs.text().rjust(lhs_w)} = {ln.rhs.text()}" + (f"    ({ln.note})" if ln.note else "")
               for ln in self.lines]
        if self.definitions:
            out.append("where")
            out += [f"  {t}" for _, t in self.definitions]
        if self.legend:
            out += [f"  {e.text_line()}" for e in self.legend]
        return "\n".join(out)

    def full_latex(self) -> str:
        parts = [self.latex]
        if self.definitions or self.legend:
            extra = [d for d, _ in self.definitions] + [e.latex_line() for e in self.legend]
            parts.append(r"\begin{aligned}&\text{where}\\ " + r" \\ ".join("&" + x for x in extra) + r"\end{aligned}")
        return r"\\[6pt]".join(parts)

    def _repr_latex_(self) -> str:
        return "$$" + self.full_latex() + "$$"

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"Equation(mode={self.mode!r}, lines={len(self.lines)})\n{self.text}"

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "title": self.title,
            "latex": self.latex,
            "lines": [{"lhs": ln.lhs.latex(), "rhs": ln.rhs.latex(), "lhs_text": ln.lhs.text(),
                       "rhs_text": ln.rhs.text(), "node": ln.node, "note": ln.note} for ln in self.lines],
            "definitions": [{"latex": d, "text": t} for d, t in self.definitions],
            "legend": [{"latex": e.latex_line(), "text": e.text_line(), "symbol": e.latex, "path": e.path}
                       for e in self.legend],
        }

    def show(self):
        from ..viz.html.dashboard import equation_html

        html = equation_html(self)
        try:
            from IPython.display import HTML

            return HTML(html)
        except ImportError:  # pragma: no cover
            return html


class _Ctx:
    def __init__(self, graph: ModelGraph, symbols: str, numbering: ModelGraph | None = None):
        self.graph = graph
        self.symbols = SymbolTable(graph, symbols, numbering)


def select_graph(net: "Net", focus: str | None, depth: int | None) -> tuple[ModelGraph, ModelGraph]:
    """Returns (graph to unroll, graph that defines global layer numbering)."""
    g = net.graph
    if depth is not None:
        base = (g.depth(focus) + 1) if focus else 0
        g = g.collapse(base + depth)
    full = g
    if focus:
        if focus not in g.nodes:
            raise KeyError(f"focus {focus!r} not in graph (collapsed at depth={depth}?)")
        if g.nodes[focus].is_compute:
            return _single_node_graph(g, focus), full
        g = g.subgraph(focus)
    return g, full


def _single_node_graph(g: ModelGraph, nid: str) -> ModelGraph:
    from ..core.graph import Node as N

    sg = ModelGraph(f"{g.name}.{nid}", g.framework)
    node = g.nodes[nid]
    sg.add_node(N("@in:z", "Input", "Input", "input", label="z", out_shapes=node.in_shapes))
    c = N(node.id, node.kind, node.type_name, node.op, params=node.params, roles=node.roles, attrs=node.attrs,
          in_shapes=node.in_shapes, out_shapes=node.out_shapes, module=node.module, label=node.label)
    sg.add_node(c)
    sg.add_node(N("@out", "Output", "Output", "output", label="out"))
    n_in = max(1, len(g.pred(nid)))
    for _ in range(n_in):
        sg.add_edge("@in:z", nid)
    sg.add_edge(nid, "@out")
    return sg


def _input_symbol(node: Node, n_inputs: int) -> Sym:
    lab = node.label or node.id.split(":", 1)[-1]
    if lab in ("x", "z") or n_inputs == 1:
        base = lab if lab in ("x", "z") else "x"
        return Sym(base, base, node.id)
    safe = tex_escape(lab)
    return Sym(rf"x_{{\text{{{safe}}}}}", f"x_{lab}", node.id)


def unroll(net: "Net", mode: str = "system", focus: str | None = None, depth: int | None = None,
           symbols: str = "index", max_size: int = 40, **kw) -> Equation:
    if mode == "scalar":
        return scalar_equation(net, **kw)
    g, full = select_graph(net, focus, depth)
    ctx = _Ctx(g, symbols, full)
    out_sym = ("y", "y") if focus else (r"\hat{y}", "ŷ")
    if mode == "compact":
        eq = _compact(g, ctx)
    elif mode in ("system", "expanded"):
        eq = _system(g, ctx, mode, max_size, out_sym)
    else:
        raise ValueError(f"unknown mode {mode!r}: use compact | system | expanded | scalar")
    eq.title = f"{net.name}" + (f" › {focus}" if focus else "")
    return eq


def _definitions(g: ModelGraph, nodes) -> list[tuple[str, str]]:
    seen, out = set(), []
    for nid in nodes:
        act = _builtin_activation(g.nodes[nid])
        for d in (definition_for(g.nodes[nid]), DEFINITIONS.get(act) if act else None):
            if d and d[0] not in seen:
                seen.add(d[0])
                out.append(d)
    return out


def _system(g: ModelGraph, ctx: _Ctx, mode: str, max_size: int, out_sym=(r"\hat{y}", "ŷ")) -> Equation:
    order = g.topo_order()
    ref: dict[str, Expr] = {}
    lines: list[Line] = []
    inputs = [n for n in order if g.nodes[n].op == "input"]
    outputs_src = [e.src for e in g.edges if g.nodes[e.dst].op == "output"]
    multi_out = len(set(outputs_src)) > 1
    k = 0
    used = []
    for nid in order:
        node = g.nodes[nid]
        if node.op == "input":
            ref[nid] = _input_symbol(node, len(inputs))
            continue
        if node.op == "output":
            continue
        args = [ref[p] for p in g.pred(nid) if p in ref]
        e = rule_for(node)(node, args, ctx)
        used.append(nid)
        if args and e is args[0]:  # passthrough (dropout / identity)
            ref[nid] = e
            continue
        succ = g.succ(nid)
        feeds_out = nid in outputs_src
        if mode == "system":
            name = (feeds_out or len(succ) >= 2 or any(g.nodes[s].params or g.nodes[s].attrs.get("collapsed")
                                                        for s in succ) or e.size() > max_size)
        else:
            name = feeds_out or len(succ) >= 2 or e.size() > max_size * 4
        if node.op == "param" or (node.kind in _STRUCTURAL and not node.params and len(succ) < 2 and not feeds_out):
            name = False
        if name:
            if feeds_out:
                i = outputs_src.index(nid)
                lhs = (Sym(rf"{out_sym[0]}_{{{i + 1}}}", f"{out_sym[1]}{sub(i + 1)}", nid) if multi_out
                       else Sym(*out_sym, nid))
            else:
                k += 1
                lhs = Sym(f"h_{{{k}}}", f"h{sub(k)}", nid)
            lines.append(Line(lhs, e, nid))
            ref[nid] = Sym(lhs.latex(), lhs.text(), nid)
        else:
            ref[nid] = e
    return Equation(lines, mode, _definitions(g, used), list(ctx.symbols.legend.values()))


def _compact(g: ModelGraph, ctx: _Ctx) -> Equation:
    order = [n for n in g.topo_order() if g.nodes[n].op not in ("input", "output")]
    chain = all(len(g.pred(n)) == 1 and len(g.succ(n)) == 1 for n in order)
    inputs = [n for n in g.topo_order() if g.nodes[n].op == "input"]
    if chain and len(inputs) == 1:
        fns, defs = [], []
        for nid in order:
            node = g.nodes[nid]
            if node.kind in _PASSTHROUGH:
                continue
            fns.append(fn_name(node, ctx))
            if node.params:
                fl, ft = fn_name(node, ctx)
                body = rule_for(node)(node, [Sym("z")], ctx)
                defs.append(Line(Raw(rf"{fl}(z)", f"{ft}(z)"), body, nid))
        x = _input_symbol(g.nodes[inputs[0]], 1)
        lines = [Line(Sym(r"\hat{y}", "ŷ"), Compose(fns, x), None)] + defs
        return Equation(lines, "compact", _definitions(g, order), list(ctx.symbols.legend.values()))
    # DAG: opaque function applications, naming only multi-use values
    ref: dict[str, Expr] = {}
    lines, k = [], 0
    outputs_src = [e.src for e in g.edges if g.nodes[e.dst].op == "output"]
    for nid in g.topo_order():
        node = g.nodes[nid]
        if node.op == "input":
            ref[nid] = _input_symbol(node, len(inputs))
            continue
        if node.op == "output":
            continue
        args = [ref[p] for p in g.pred(nid) if p in ref]
        if node.kind in _PASSTHROUGH and args:
            ref[nid] = args[0]
            continue
        if node.kind in ("add", "sub", "mul", "div", "matmul", "Concat", "Transpose", "getitem") or node.op == "param":
            e = rule_for(node)(node, args, ctx)
        else:
            fl, ft = fn_name(node, ctx)
            e = _with_builtin(node, Apply(fl, ft, args or [Sym("z")], nid))
        if nid in outputs_src:
            lines.append(Line(Sym(r"\hat{y}", "ŷ"), e, nid))
        elif len(g.succ(nid)) >= 2:
            k += 1
            lines.append(Line(Sym(f"h_{{{k}}}", f"h{sub(k)}"), e, nid))
            e = Sym(f"h_{{{k}}}", f"h{sub(k)}", nid)
        ref[nid] = e
    return Equation(lines, "compact", _definitions(g, [n for n in g.nodes]), list(ctx.symbols.legend.values()))


# ------------------------------------------------------------------ scalar
def _weight_matrix(net: "Net", node: Node) -> tuple[np.ndarray, np.ndarray | None]:
    r = node.roles["weight"]
    w = net.backend.get_param(node.params[r.name].path)
    if w.ndim == 2 and r.out_axis is not None and r.in_axis is not None:
        w = np.transpose(w, (r.out_axis % 2, r.in_axis % 2))
    b = None
    if "bias" in node.roles:
        b = net.backend.get_param(node.params[node.roles["bias"].name].path).reshape(-1)
    return w, b


def _walk_back(g: ModelGraph, nid: str) -> tuple[list[str], str | None]:
    """From ``nid`` walk producers through elementwise ops until a dense layer. Returns (post_ops, dense)."""
    post, cur = [], nid
    while cur is not None:
        n = g.nodes[cur]
        if n.kind == "Linear" and "weight" in n.roles:
            return post, cur
        if n.kind in _ELEMENTWISE or n.kind in ("Softmax", "LogSoftmax", "Flatten", "Reshape"):
            post.append(cur)
            p = g.pred(cur)
            cur = p[0] if len(p) == 1 else None
            continue
        if n.op == "output":
            p = g.pred(cur)
            cur = p[0] if len(p) == 1 else None
            continue
        return post, None
    return post, None


class _NeuronNamer:
    def __init__(self, net: "Net", g: ModelGraph, symbols: SymbolTable):
        self.net, self.g, self.sym = net, g, symbols

    def source(self, dense: str) -> tuple[str, str | None, list[str]]:
        """What feeds ``dense``: ('input', input_id, []) or ('neuron', prev_dense, elementwise ops between)."""
        g = self.g
        p = g.pred(dense)
        if len(p) != 1:
            return "opaque", None, []
        src = p[0]
        post, prev = _walk_back(g, src)
        if prev is not None and all(g.nodes[o].kind not in ("Softmax", "LogSoftmax") for o in post):
            return "neuron", prev, list(reversed(post))
        cur = src
        while g.nodes[cur].kind in _PASSTHROUGH | {"Flatten", "Reshape"} and len(g.pred(cur)) == 1:
            cur = g.pred(cur)[0]
        if g.nodes[cur].op == "input":
            return "input", cur, []
        return "opaque", cur, []

    def symbol(self, kind: str, ref: str | None, i: int) -> Sym:
        if kind == "input":
            return Sym(f"x_{{{i}}}", f"x{sub(i)}")
        if kind == "neuron":
            ell = self.sym.layer(self.g.nodes[ref])
            return Sym(f"a^{{({ell})}}_{{{i}}}", f"a{sup(f'({ell})')}{sub(i)}")
        lab = tex_escape(ref or "h")
        return Sym(rf"h^{{\texttt{{{lab}}}}}_{{{i}}}", f"h[{ref}]{sub(i)}")


def scalar_equation(net: "Net", node: str | None = None, neuron: Any = 0, top_k: int = 6, levels: int = 1,
                    precision: int = 3, input: Any = None, sample: int = 0, trace=None, position=None,
                    **_) -> Equation:
    """Explicit formula for one neuron with the network's actual weights.

    ``node`` may be a Linear layer or an activation that follows one (default: the output), or a
    convolution (optionally through BatchNorm / activations): then ``neuron`` is the output channel and
    ``position`` the output location (default: where that channel responds most to the input), or pass
    ``neuron=(channel, i, j)``. With ``input`` the terms are ranked by contribution |wᵢ·aᵢ| and evaluated.
    """
    g = net.graph
    target = node
    if target is None:
        outs = [e.src for e in g.edges if g.nodes[e.dst].op == "output"]
        if not outs:
            raise ValueError("graph has no output")
        target = outs[0]
    if trace is None and input is not None:
        trace = net.trace(input)
    cpost, conv = _walk_back_conv(g, target)
    if conv is not None:
        return conv_scalar_equation(net, conv, cpost, neuron, position, top_k, precision, trace, sample)
    post, dense = _walk_back(g, target)
    if dense is None:
        raise ValueError(f"scalar mode needs a Linear layer at or before {target!r} (through elementwise ops); "
                         f"got {g.nodes[target].kind}. Pick node=… explicitly.")
    symbols = SymbolTable(g)
    namer = _NeuronNamer(net, g, symbols)

    def acts_of(kind, ref, ops) -> np.ndarray | None:
        if trace is None:
            return None
        key = ops[-1] if ops else ref
        if kind == "input":
            key = ref
        if key in trace.acts:
            a = trace.acts[key]
            return a.reshape(a.shape[0], -1)[sample] if a.ndim > 1 else a.reshape(-1)
        return None

    def pre_expr(dense_id: str, j: int, depth: int) -> Expr:
        dn = g.nodes[dense_id]
        W, b = _weight_matrix(net, dn)
        row = W[j].reshape(-1)
        kind, ref, ops = namer.source(dense_id)
        a = acts_of(kind, ref, ops)
        score = np.abs(row * a) if a is not None and a.shape[0] == row.shape[0] else np.abs(row)
        idx = np.argsort(-score, kind="stable")[:top_k]
        terms: list[Expr] = []
        signs: list[int] = []
        for i in idx:
            w = float(row[i])
            if kind == "neuron" and depth > 1:
                inner = _with_builtin(g.nodes[ref], pre_expr(ref, int(i), depth - 1))
                for op in ops:
                    inner = _apply_elementwise(g.nodes[op], inner)
                sym = inner
            else:
                sym = namer.symbol(kind, ref, int(i))
            terms.append(Prod(Num(abs(w), precision), sym, r"\,", "·"))
            signs.append(1 if w >= 0 else -1)
        rest = row.shape[0] - len(idx)
        if rest > 0:
            terms.append(Ellipsis_(rest))
            signs.append(1)
        if b is not None and float(b[j]) != 0.0:
            terms.append(Num(abs(float(b[j])), precision))
            signs.append(1 if b[j] >= 0 else -1)
        return Add(terms, signs, weighted=True)

    ell = symbols.layer(g.nodes[dense])
    z_sym = Sym(f"z^{{({ell})}}_{{{neuron}}}", f"z{sup(f'({ell})')}{sub(neuron)}")
    lines = [Line(z_sym, pre_expr(dense, neuron, levels), dense)]
    ops_after = list(reversed(post))
    builtin = _builtin_activation(g.nodes[dense])  # Keras Dense(…, activation=…): recorded after its activation
    val_pre = None
    if trace is not None and dense in trace.acts and builtin is None:
        a = trace.acts[dense]
        val_pre = float(a.reshape(a.shape[0], -1)[sample][neuron] if a.ndim > 1 else a.reshape(-1)[neuron])
    elif trace is not None:
        W, b = _weight_matrix(net, g.nodes[dense])
        a_in = acts_of(*namer.source(dense))
        if a_in is not None and a_in.shape[0] == W.shape[1]:
            val_pre = float(W[neuron].reshape(-1) @ a_in + (float(b[neuron]) if b is not None else 0.0))
    if val_pre is not None:
        lines[0].note = f"= {val_pre:.{precision}g} for this input"
    elementwise = [o for o in ops_after if g.nodes[o].kind in _ELEMENTWISE - _PASSTHROUGH]
    if builtin in ("Softmax", "LogSoftmax") or any(g.nodes[o].kind in ("Softmax", "LogSoftmax") for o in ops_after):
        out = Sym(rf"\hat{{y}}_{{{neuron}}}", f"ŷ{sub(neuron)}")
        rhs: Expr = Raw(rf"\frac{{e^{{z_{{{neuron}}}}}}}{{\sum_k e^{{z_k}}}}", f"e^(z{sub(neuron)}) / Σₖ e^(zₖ)")
        lines.append(Line(out, rhs, ops_after[-1] if ops_after else dense))
    elif elementwise or builtin:
        rhs = _with_builtin(g.nodes[dense], z_sym)
        for o in elementwise:
            rhs = _apply_elementwise(g.nodes[o], rhs)
        out_l = f"a^{{({ell})}}_{{{neuron}}}"
        lines.append(Line(Sym(out_l, f"a{sup(f'({ell})')}{sub(neuron)}"), rhs, elementwise[-1] if elementwise else dense))
    if trace is not None and len(lines) > 1:
        key = ops_after[-1] if ops_after else dense
        if key in trace.acts:
            a = trace.acts[key]
            v = float(a.reshape(a.shape[0], -1)[sample][neuron] if a.ndim > 1 else a.reshape(-1)[neuron])
            lines[-1].note = f"= {v:.{precision}g}"
    defs = [DEFINITIONS[k] for k in ([builtin] if builtin else []) + [g.nodes[o].kind for o in ops_after]
            if k in DEFINITIONS]
    legend = [LegendEntry(r"a^{(\ell)}_i", "a⁽ˡ⁾ᵢ", "output i of layer ℓ (after its activation)", (), desc=True)]
    eq = Equation(lines, "scalar", list(dict.fromkeys(defs)), [])
    eq.title = f"{net.name} › {dense}[{neuron}]"
    eq.legend = legend if levels > 1 or namer.source(dense)[0] == "neuron" else []
    return eq


# ---------------------------------------------------------------- convolutions, one output value
def _walk_back_conv(g: ModelGraph, nid: str) -> tuple[list[str], str | None]:
    """From ``nid`` back through elementwise ops and BatchNorm to a convolution. Returns (post_ops, conv)."""
    post, cur = [], nid
    while cur is not None:
        n = g.nodes[cur]
        if n.kind in CONV_KINDS and "weight" in n.roles:
            return post, cur
        if n.kind in _ELEMENTWISE or n.kind == "BatchNorm" or n.op == "output":
            if n.op != "output":
                post.append(cur)
            p = g.pred(cur)
            cur = p[0] if len(p) == 1 else None
            continue
        return post, None
    return post, None


def _tuple(v, nd: int, default: int) -> tuple:
    if v is None:
        return (default,) * nd
    if isinstance(v, (int, float)):
        return (int(v),) * nd
    v = list(v)
    return tuple(int(x) for x in v) if len(v) == nd else (int(v[0]),) * nd


def _conv_geometry(node: Node, in_sp: tuple, out_sp: tuple, ks: tuple) -> tuple[tuple, tuple, tuple]:
    """(stride, dilation, low padding) per spatial dim, from the layer's attributes or, failing that, its shapes."""
    nd = len(ks)
    a = node.attrs
    stride = _tuple(a.get("stride", a.get("strides")), nd, 1)
    dil = _tuple(a.get("dilation", a.get("kernel_dilation", a.get("rhs_dilation"))), nd, 1)
    pad = a.get("padding")
    if isinstance(pad, str) and pad.lower() == "valid":
        lo = (0,) * nd
    elif isinstance(pad, (int, float)) or (isinstance(pad, (list, tuple)) and pad and
                                           all(isinstance(p, (int, float)) for p in pad)):
        lo = _tuple(pad, nd, 0)
    else:  # "same" / unknown: the padding that makes the shapes work out (low side gets the smaller half)
        lo = tuple(max((o - 1) * s + (k - 1) * d + 1 - i, 0) // 2
                   for i, o, s, k, d in zip(in_sp, out_sp, stride, ks, dil))
    return stride, dil, lo


def _channels_first(net: "Net", node: Node, a, sample: int) -> np.ndarray | None:
    """One sample of a traced tensor with the channel axis first."""
    if a is None:
        return None
    a = np.asarray(a, dtype=np.float64)
    fa = net.backend.feature_axis(node)
    unbatched = net.framework == "equinox"
    if not unbatched:
        a = a[min(sample, a.shape[0] - 1)]
        fa = (fa % (a.ndim + 1)) - 1
    return np.moveaxis(a, fa % a.ndim, 0)


def conv_scalar_equation(net: "Net", conv: str, post: list[str], neuron: Any = 0, position=None, top_k: int = 6,
                         precision: int = 3, trace=None, sample: int = 0) -> Equation:
    """``z_{c,i,j} = Σ_{k,u,v} W_{c,k,u,v} · x_{k, s·i−p+d·u, s·j−p+d·v} + b_c`` with the real numbers, the
    ``top_k`` largest contributions shown, then any BatchNorm / activation applied to that value."""
    g = net.graph
    n = g.nodes[conv]
    r = n.roles["weight"]
    w = net.backend.get_param(n.params[r.name].path).astype(np.float64)
    w = np.moveaxis(w, (r.out_axis % w.ndim, r.in_axis % w.ndim), (0, 1))  # (out, in / groups, *kernel)
    b = net.backend.get_param(n.params[n.roles["bias"].name].path).reshape(-1) if "bias" in n.roles else None
    c_out, c_in_g, ks = w.shape[0], w.shape[1], tuple(w.shape[2:])
    nd = len(ks)
    if isinstance(neuron, (tuple, list)):
        ch, pos = int(neuron[0]), tuple(int(v) for v in neuron[1:])
    else:
        ch, pos = int(neuron), tuple(position) if position is not None else None
    if not 0 <= ch < c_out:
        raise ValueError(f"{conv} has {c_out} output channels; got channel {ch}")
    x = _channels_first(net, n, trace.inputs.get(conv) if trace is not None else None, sample)
    if x is None and trace is not None:
        p = g.pred(conv)
        x = _channels_first(net, g.nodes[p[0]], trace.acts.get(p[0]), sample) if len(p) == 1 else None
    z = _channels_first(net, n, trace.acts.get(conv) if trace is not None else None, sample)
    out_sp = tuple(z.shape[1:]) if z is not None else tuple(
        (n.out_shapes[0][1:-1] if net.backend.channels_last else n.out_shapes[0][2:]) if n.out_shapes else ks)
    in_sp = tuple(x.shape[1:]) if x is not None else out_sp
    if pos is None:  # where this channel responds most (or the center without an input)
        pos = (tuple(int(v) for v in np.unravel_index(np.argmax(np.abs(z[ch])), z[ch].shape)) if z is not None
               else tuple(o // 2 for o in out_sp))
    if len(pos) != nd:
        raise ValueError(f"{conv} has {nd} spatial dims; got position {pos}")
    stride, dil, lo = _conv_geometry(n, in_sp, out_sp, ks)
    groups = max(1, int(n.attrs.get("groups", n.attrs.get("feature_group_count", 1)) or 1))
    k0 = (ch // max(c_out // groups, 1)) * c_in_g  # first input channel this output channel sees
    terms_raw = []
    for k in range(c_in_g):
        for off in itertools.product(*(range(kk) for kk in ks)):
            loc = tuple(p_ * s_ - l_ + d_ * u for p_, s_, l_, d_, u in zip(pos, stride, lo, dil, off))
            if any(q < 0 or q >= m for q, m in zip(loc, in_sp)):
                continue  # zero padding
            wv = float(w[(ch, k) + off])
            xv = float(x[(k0 + k,) + loc]) if x is not None else None
            terms_raw.append((wv, xv, k0 + k, loc))
    score = [abs(t[0] * t[1]) if t[1] is not None else abs(t[0]) for t in terms_raw]
    order = sorted(range(len(terms_raw)), key=lambda i: -score[i])[:top_k]
    src = g.pred(conv)
    src_node = g.nodes[src[0]] if len(src) == 1 else None
    is_input = src_node is None or src_node.op == "input"
    idx_l = lambda kk, loc: ",".join(str(v) for v in (kk, *loc))  # noqa: E731
    sym = lambda kk, loc: Sym(f"x_{{{idx_l(kk, loc)}}}", f"x[{idx_l(kk, loc)}]")  # noqa: E731 - this layer's input
    terms: list[Expr] = []
    signs: list[int] = []
    for i in order:
        wv, _, kk, loc = terms_raw[i]
        terms.append(Prod(Num(abs(wv), precision), sym(kk, loc), r"\,", "·"))
        signs.append(1 if wv >= 0 else -1)
    rest = len(terms_raw) - len(order)
    if rest > 0:
        terms.append(Ellipsis_(rest))
        signs.append(1)
    if b is not None and float(b[ch]) != 0.0:
        terms.append(Num(abs(float(b[ch])), precision))
        signs.append(1 if b[ch] >= 0 else -1)
    at = ",".join(str(v) for v in (ch, *pos))
    z_sym = Sym(f"z_{{{at}}}", f"z[{at}]")
    lines = [Line(z_sym, Add(terms, signs, weighted=True), conv)]

    def value(nid: str) -> float | None:
        if trace is None or nid not in trace.acts:
            return None
        a = _channels_first(net, g.nodes[nid], trace.acts[nid], sample)
        return float(a[(ch, *pos)]) if a is not None and a.ndim == nd + 1 else None

    v = value(conv)
    if v is not None:
        lines[0].note = f"= {v:.{precision}g} for this input"
    cur: Expr = z_sym
    defs = []
    for o in reversed(post):
        on = g.nodes[o]
        if on.kind == "BatchNorm":
            p_ = {role: net.backend.get_param(on.params[rr.name].path).reshape(-1) for role, rr in on.roles.items()}
            if "running_mean" in p_ or "mean" in p_:
                mu = p_.get("running_mean", p_.get("mean"))[ch]
                var = p_.get("running_var", p_.get("var"))[ch]
                eps = float(on.attrs.get("eps", on.attrs.get("epsilon", 1e-5)) or 1e-5)
                gam = p_["scale"][ch] if "scale" in p_ else 1.0
                bet = p_["bias"][ch] if "bias" in p_ else 0.0
                scale = float(gam) / float(np.sqrt(var + eps))
                shift = float(bet) - scale * float(mu)
                terms_bn: list[Expr] = [Prod(Num(abs(scale), precision), cur, r"\,", "·")]
                signs_bn = [1 if scale >= 0 else -1]
                if abs(shift) >= 0.5 * 10 ** -precision:  # an untrained BN has no shift worth printing
                    terms_bn.append(Num(abs(shift), precision))
                    signs_bn.append(1 if shift >= 0 else -1)
                rhs = Add(terms_bn, signs_bn)
                cur = Sym(rf"\hat{{z}}_{{{at}}}", f"ẑ[{at}]")
                lines.append(Line(cur, rhs, o, None))
                defs.append((r"\mathrm{BN}(z) = \gamma\,\frac{z-\mu}{\sqrt{\sigma^2+\epsilon}} + \beta"
                             r"\ \text{folds into one affine map per channel}", "BN(z) = γ(z−μ)/√(σ²+ε)+β"))
                v = value(o)
                if v is not None:
                    lines[-1].note = f"= {v:.{precision}g}"
            continue
        if on.kind in _PASSTHROUGH:
            continue
        rhs = _apply_elementwise(on, cur)
        cur = Sym(f"a_{{{at}}}", f"a[{at}]")
        lines.append(Line(cur, rhs, o))
        if on.kind in DEFINITIONS:
            defs.append(DEFINITIONS[on.kind])
        v = value(o)
        if v is not None:
            lines[-1].note = f"= {v:.{precision}g}"
    names = {1: ["position"], 2: ["row", "col"], 3: ["depth", "row", "col"]}.get(nd, [f"dim{d}" for d in range(nd)])
    geo = ", ".join(f"{names[d]} = {'' if stride[d] == 1 else stride[d]}{'ijl'[d]}{f' − {lo[d]}' if lo[d] else ''}"
                    f" + {'' if dil[d] == 1 else dil[d]}{'uvw'[d]}" for d in range(nd))
    who = "the network input" if is_input else f"the output of {src_node.kind} {src_node.label or src_node.id}"
    legend = [LegendEntry(r"x_{k,\ldots}", "x", f"this layer's input ({who}): channel k at {geo}, "
                          f"(u, v) running over the kernel; {len(terms_raw)} terms fall inside the input", (),
                          desc=True)]
    eq = Equation(lines, "scalar", list(dict.fromkeys(defs)), legend)
    eq.title = f"{net.name} › {conv}[{at}]"
    # every term, for programmatic use: z = Σ w·x + bias
    eq.terms = [{"w": wv, "x": xv, "index": (kk, *loc)} for wv, xv, kk, loc in terms_raw]
    eq.bias = float(b[ch]) if b is not None else 0.0
    eq.value = value(conv)
    return eq


def _apply_elementwise(node: Node, e: Expr) -> Expr:
    rule = rule_for(node)
    return rule(node, [e], None)


def _with_builtin(node: Node, e: Expr) -> Expr:
    """``e`` passed through the activation ``node`` applies itself, if any (Keras ``activation=``)."""
    act = _builtin_activation(node)
    if act is None:
        return e
    latex, text = _ACTS[act]
    return Apply(latex, text, [e], node.id)


def node_equation(net: "Net", node_id: str, graph: ModelGraph | None = None) -> Equation | None:
    """Single-layer equation ``out = f(z)`` for the dashboard's focus panel."""
    g = graph or net.graph
    node = g.nodes[node_id]
    if node.op in ("input", "output"):
        return None
    try:
        sg = _single_node_graph(g, node_id) if node.is_compute else g.subgraph(node_id)
        ctx = _Ctx(sg, "index", g)
        if node.is_compute:
            n_in = len(g.pred(node_id))
            args = [Sym("z", "z")] if n_in <= 1 else [Sym(f"z_{{{i + 1}}}", f"z{sub(i + 1)}") for i in range(n_in)]
            e = rule_for(node)(node, args, ctx)
            eq = Equation([Line(Sym("h", "h"), e, node_id)], "layer", _definitions(g, [node_id]),
                          list(ctx.symbols.legend.values()))
            return eq
        return _system(sg, ctx, "system", 40, ("y", "y"))
    except Exception:  # noqa: BLE001 - equations are best-effort decoration in the UI
        return None


def to_sympy(net: "Net", input_symbols: int | None = None):
    """Exact closed form for tiny dense networks (Linear + elementwise activations) as a sympy Matrix."""
    import sympy as sp

    g = net.graph
    order = g.topo_order()
    ins = [n for n in order if g.nodes[n].op == "input"]
    if len(ins) != 1:
        raise ValueError("to_sympy supports single-input networks")
    shape = g.nodes[ins[0]].out_shapes[0] if g.nodes[ins[0]].out_shapes else None
    n_in = input_symbols or (shape[-1] if shape else None)
    if n_in is None:
        raise ValueError("unknown input width; pass input_symbols=")
    xs = sp.symbols(f"x0:{n_in}")
    val: dict[str, Any] = {ins[0]: sp.Matrix(xs)}
    funcs = {"ReLU": lambda z: sp.Max(0, z), "Tanh": sp.tanh, "Sigmoid": lambda z: 1 / (1 + sp.exp(-z)),
             "SiLU": lambda z: z / (1 + sp.exp(-z)), "exp": sp.exp, "log": sp.log,
             "Softplus": lambda z: sp.log(1 + sp.exp(z))}
    for nid in order:
        node = g.nodes[nid]
        if node.op in ("input", "output"):
            continue
        p = g.pred(nid)
        a = val[p[0]] if p else None
        if node.kind == "Linear":
            W, b = _weight_matrix(net, node)
            if W.size > 4096:
                raise ValueError("to_sympy is meant for tiny networks")
            z = sp.Matrix(np.round(W, 6).tolist()) * a
            if b is not None:
                z = z + sp.Matrix(np.round(b, 6).tolist())
            val[nid] = z
        elif node.kind in funcs:
            val[nid] = a.applyfunc(funcs[node.kind])
        elif node.kind in _PASSTHROUGH | {"Flatten", "Reshape"}:
            val[nid] = a
        elif node.kind == "add" and len(p) == 2:
            val[nid] = val[p[0]] + val[p[1]]
        elif node.kind == "Softmax":
            e = a.applyfunc(sp.exp)
            val[nid] = e / sum(e)
        else:
            raise ValueError(f"to_sympy: unsupported layer kind {node.kind!r}")
    outs = [e.src for e in g.edges if g.nodes[e.dst].op == "output"]
    return val[outs[0]]


__all__ = ["Equation", "Line", "unroll", "scalar_equation", "node_equation", "to_sympy", "CONV_KINDS"]
