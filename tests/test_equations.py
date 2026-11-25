import numpy as np
import pytest
import torch

import deepvis as dv
from deepvis.equations.expr import Apply, Sym
from deepvis.equations.unroll import to_sympy


def test_modes(mlp):
    model, x = mlp
    net = dv.wrap(model, sample_input=x)
    sys = net.equation()
    assert [ln.lhs.text() for ln in sys.lines] == ["h₁", "h₂", "ŷ"]
    assert sys.lines[0].rhs.text() == "tanh(W⁽¹⁾x + b⁽¹⁾)"
    assert "\\begin{aligned}" in sys.latex and sys._repr_latex_().startswith("$$")
    comp = net.equation("compact")
    assert comp.lines[0].rhs.text().startswith("(A⁽³⁾ ∘ tanh ∘ A⁽²⁾")
    exp = net.equation("expanded")
    assert len(exp.lines) == 1 and exp.lines[0].rhs.text().count("tanh") == 2
    legend = {e.path for e in sys.legend}
    assert legend == {"net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias", "net.4.weight", "net.4.bias"}


def test_residual_depth_and_focus(resnet):
    model, x = resnet
    net = dv.wrap(model, sample_input=x)
    txt = net.equation().text
    assert "h₃ = h₁ + ReLU(LN" in txt
    collapsed = net.equation(depth=1)
    assert any("Block_0" in ln.rhs.text() for ln in collapsed.lines)
    focus = net["blocks.1"].equation()
    assert focus.lines[-1].lhs.text() == "y" and "W⁽⁴⁾" in focus.text  # global layer numbering


def test_scalar_mode_uses_real_weights(mlp):
    model, x = mlp
    net = dv.wrap(model, sample_input=x)
    eq = net.equation("scalar", node="net.2", neuron=3, top_k=8, input=x)
    w = model.net[2].weight[3].detach().numpy()
    for v in w:
        assert f"{abs(v):.3g}".rstrip("0") in eq.text
    with torch.no_grad():
        z = model.net[2](torch.tanh(model.net[0](x)))[0, 3].item()
    assert eq.lines[0].note.startswith(f"= {z:.3g}")


def test_to_sympy_matches_forward():
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.ReLU(), torch.nn.Linear(3, 1))
    net = dv.wrap(m, sample_input=torch.randn(1, 2))
    expr = to_sympy(net)
    import sympy as sp

    x0, x1 = sp.symbols("x0:2")
    for a, b in [(0.3, -1.2), (2.0, 0.5)]:
        val = float(expr[0].subs({x0: a, x1: b}))
        assert abs(val - m(torch.tensor([[a, b]]))[0, 0].item()) < 1e-4


def test_custom_rule():
    class Gate(torch.nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)

    @dv.equation_rule("Gate")
    def gate(node, args, ctx):
        return Apply(r"\mathrm{gate}", "gate", args)

    m = torch.nn.Sequential(torch.nn.Linear(3, 3), Gate())
    net = dv.wrap(m, sample_input=torch.randn(1, 3))
    assert "gate(W⁽¹⁾x + b⁽¹⁾)" in net.equation("expanded").text
    assert Sym("x").text() == "x"


@pytest.mark.parametrize("kw", [dict(stride=2, padding=1), dict(padding="same", dilation=2),
                                dict(groups=2, padding=2, stride=3), dict(padding="valid")])
def test_conv_scalar_equation_is_exact(kw):
    nn = torch.nn
    torch.manual_seed(0)
    m = nn.Sequential(nn.Conv2d(4, 6, 3, **kw), nn.BatchNorm2d(6), nn.Tanh()).eval()
    with torch.no_grad():
        m[1].running_mean.uniform_(-0.5, 0.5)
        m[1].running_var.uniform_(0.5, 2.0)
    x = torch.randn(2, 4, 11, 10)
    net = dv.wrap(m, sample_input=x)
    tr = net.trace(x)
    z = tr["0"][1]
    for pos in [(0, 0), tuple(s - 1 for s in z.shape[1:])]:  # corners exercise the padding
        eq = net.equation("scalar", node="2", neuron=(5, *pos), trace=tr, sample=1)
        assert np.isclose(sum(t["w"] * t["x"] for t in eq.terms) + eq.bias, z[(5, *pos)], atol=1e-5)
        assert [ln.node for ln in eq.lines] == ["0", "1", "2"]  # conv, folded BN, tanh
        assert eq.lines[-1].note == f"= {tr['2'][(1, 5, *pos)]:.3g}"
    auto = net.equation("scalar", node="0", neuron=2, trace=tr)  # position: where channel 2 peaks
    peak = np.unravel_index(np.argmax(np.abs(tr["0"][0, 2])), z.shape[1:])
    assert auto.title.endswith(f"[2,{peak[0]},{peak[1]}]")


def test_conv_scalar_in_dashboard_payload(cnn):
    from deepvis.viz.html.payload import PayloadBuilder

    model, x = cnn
    net = dv.wrap(model, sample_input=x)
    sc = PayloadBuilder(net).build((x,))["scalars"]
    convs = [n.id for n in net.graph.compute_nodes() if n.kind.startswith("Conv")]
    assert convs and set(convs) <= set(sc)
    assert all(d["lines"][0]["note"].startswith("= ") and d["terms"] > 0 for d in sc.values())
