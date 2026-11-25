"""The videos' narration and colour-coded maths, without rendering (manim not needed)."""

import pytest

import deepvis as dv
from deepvis.anim.texlite import latex_to_pango, split_painted

torch = pytest.importorskip("torch")
nn = torch.nn

PAINTED = (r"\htmlData{n=0}{\textcolor{#83C167}{\sigma}}\!\left(\htmlData{n=1}{\textcolor{#58C4DD}{W^{(1)}}}\,x"
           r" + \textcolor{#FC6255}{0.3\, x_{1}}\right)")


def test_pango_fallback_keeps_the_colours():
    out = latex_to_pango(PAINTED)
    assert '<span foreground="#58C4DD"><i>W</i><sup>(1)</sup></span>' in out
    assert '<span foreground="#FC6255">' in out and "htmlData" not in out and "textcolor" not in out


def test_real_tex_gets_plain_latex_and_a_colour_map():
    plain, colors = split_painted(PAINTED)
    assert plain == r"\sigma\!\left(W^{(1)}\,x + 0.3\, x_{1}\right)"
    assert colors == {r"\sigma": "#83C167", "W^{(1)}": "#58C4DD", r"0.3\, x_{1}": "#FC6255"}
    assert split_painted(r"W x + b") == (r"W x + b", {})


@pytest.mark.torch
def test_animation_data_carries_narration_and_colours():
    from deepvis.anim.data import architecture_data, dense_data, equation_data, focus_data

    m = nn.Sequential(nn.Flatten(), nn.Linear(16, 8), nn.Sigmoid(), nn.Linear(8, 3), nn.Softmax(-1))
    x = torch.rand(1, 1, 4, 4)
    net = dv.wrap(m, sample_input=x)
    d = dense_data(net, (x,))
    assert [l_["act"] for l_ in d["layers"][1:]] == ["the sigmoid", "a softmax"]
    assert equation_data(net)["colors"]["1"] == "#58C4DD"  # a dense layer's symbols in BLUE_C
    assert focus_data(net, (x,))["colors"]["2"] == "#83C167"  # the sigmoid's in GREEN_C
    caps = {n["kind"]: n["caption"] for n in architecture_data(net)["nodes"]}
    assert "8 outputs" in caps["Linear"] or "3 outputs" in caps["Linear"]
    assert caps["Softmax"].startswith("Turns 3 scores into probabilities")


@pytest.mark.torch
def test_videos_open_with_the_whole_network():
    from deepvis.anim.render import build_data

    m = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 3))
    net = dv.wrap(m, sample_input=torch.randn(1, 6))
    d = build_data(net, "architecture", None)
    assert d["describe"] == "A fully connected network (an MLP) that turns 6 values into 3 values, with 83 learned numbers."
