"""Expressive glyphs and the export API (interactive HTML and static SVG diagrams)."""

import json
import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch
import torch.nn as nn

import deepvis as dv
from deepvis.viz.glyphs import GlyphData, activation_fn, glyph_type, node_size, render, tensor_glyph

SVG = '<svg xmlns="http://www.w3.org/2000/svg">{}</svg>'


def _payload(path):
    html = open(path, encoding="utf-8").read()
    m = re.search(r'<script id="dv-payload" type="application/json">(.*?)</script>', html, re.S)
    return json.loads(m.group(1).replace("<\\/", "</")), html


def test_every_glyph_is_valid_svg(resnet, transformer, cnn, mlp):
    seen = set()
    for model, x in (resnet, transformer, cnn, mlp):
        net = dv.wrap(model, sample_input=x)
        tr = net.trace(x)
        for g in (net.graph, net.graph.collapse(0)):
            D = GlyphData(net, tr, g)
            for n in g.compute_nodes():
                ET.fromstring(SVG.format(render(n, D)))  # strict XML: no duplicate attributes etc.
                seen.add(glyph_type(n))
    assert {"input", "output", "linear", "activation", "norm", "op", "attention", "conv", "pool", "embedding",
            "softmax", "block"} <= seen


def test_glyph_sizes_do_not_depend_on_data(mlp):
    model, x = mlp
    net = dv.wrap(model, sample_input=x)
    before = {n.id: node_size(n) for n in net.graph.compute_nodes()}
    net.select(kind="Linear").scale(-3.0)
    assert {n.id: node_size(n) for n in net.graph.compute_nodes()} == before


def test_activation_curves_are_exact():
    from deepvis.core.graph import Node

    xs = np.linspace(-3, 3, 13)
    for kind, torch_fn in (("ReLU", torch.relu), ("GELU", nn.functional.gelu), ("Sigmoid", torch.sigmoid),
                           ("Tanh", torch.tanh), ("SiLU", nn.functional.silu), ("Hardswish", nn.functional.hardswish)):
        fn = activation_fn(Node("a", kind, kind, "module"))
        assert np.allclose(fn(xs), torch_fn(torch.tensor(xs)).numpy(), atol=1e-5), kind
    leaky = activation_fn(Node("a", "LeakyReLU", "LeakyReLU", "module", attrs={"negative_slope": 0.2}))
    assert leaky(np.array([-1.0]))[0] == pytest.approx(-0.2)


def test_activation_glyph_uses_the_real_inputs(mlp):
    model, x = mlp
    net = dv.wrap(model, sample_input=x)
    tr = net.trace(x)
    svg = render(net.graph.nodes["net.1"], GlyphData(net, tr))
    assert svg.count("<polyline") == 2          # full curve + the range the data actually occupies
    assert svg.count('opacity="0.35"') >= 3     # the input histogram bars


def test_tensor_glyph():
    t = tensor_glyph((1, 16, 8, 8))
    assert "16×8×8" in t["svg"] and t["w"] > 0
    assert tensor_glyph(()) is None


def test_export_html_options(tmp_path, cnn):
    model, x = cnn
    path = dv.export_html(model, x, str(tmp_path / "m.html"), title="My CNN", note="demo", views=("graph", "volumes"),
                          start="volumes", depth="all")
    p, html = _payload(path)
    assert p["ui"]["views"] == ["graph", "volumes"] and p["ui"]["start"] == "volumes"
    assert p["ui"]["title"] == "My CNN" and p["ui"]["style"] == "expressive"
    assert p["defaultLevel"] == len(p["levels"]) - 1
    assert p["glyphs"] and all("<" in p["glyphSet"][v] for v in p["glyphs"].values())  # each drawing shipped once
    assert len(p["glyphSet"]) == len(set(p["glyphSet"])) <= len(p["glyphs"])
    tensors = p["levels"][-1]["layout"]["tensors"]
    assert tensors and all("<" in p["tensorSet"][t["si"]] for t in tensors), "3-D tensor annotations on edges"
    assert "<title>My CNN · deepvis</title>" in html
    compact, _ = _payload(dv.export_html(model, x, str(tmp_path / "c.html"), style="compact"))
    assert compact["glyphs"] == {} and compact["ui"]["style"] == "compact"
    with pytest.raises(ValueError):
        dv.export_html(model, x, str(tmp_path / "bad.html"), views=("nope",))


@pytest.mark.parametrize("direction", ["TB", "LR"])
def test_export_svg(tmp_path, resnet, direction):
    model, x = resnet
    path = dv.export_svg(model, x, str(tmp_path / "m.svg"), direction=direction, depth="all")
    root = ET.parse(path).getroot()
    assert root.tag.endswith("svg")
    net = dv.wrap(model, sample_input=x)
    n_nodes = len(net.graph.compute_nodes())
    assert open(path, encoding="utf-8").read().count('class="hit"') == n_nodes
    html = dv.export_svg(model, x, str(tmp_path / "m.html"), direction=direction)
    assert open(html, encoding="utf-8").read().startswith("<!doctype html>")


def test_export_jax_models(tmp_path):
    jax = pytest.importorskip("jax")
    nnl = pytest.importorskip("flax.linen")

    class M(nnl.Module):
        @nnl.compact
        def __call__(self, x):
            x = nnl.relu(nnl.Conv(4, (3, 3))(x)).reshape((x.shape[0], -1))
            return nnl.softmax(nnl.Dense(3)(nnl.gelu(nnl.LayerNorm()(nnl.Dense(8)(x)))))

    x = jax.random.normal(jax.random.PRNGKey(0), (1, 6, 6, 3))
    m = M()
    v = m.init(jax.random.PRNGKey(1), x)
    svg = dv.export_svg(m, x, str(tmp_path / "f.svg"), params=v)
    text = open(svg, encoding="utf-8").read()
    ET.fromstring(text)
    assert "data:image/png;base64," in text  # the channels-last input image is painted on the input tensor
    p, _ = _payload(dv.export_html(m, x, str(tmp_path / "f.html"), params=v))
    kinds = {glyph_type(n) for n in dv.wrap(m, v, sample_input=x).graph.compute_nodes()}
    assert {"conv", "activation", "norm", "linear", "softmax"} <= kinds and p["glyphs"]


def test_noop_layers_are_not_drawn_but_stay_hookable():
    from deepvis.viz.html.layout import layout

    m = nn.Sequential(nn.Linear(4, 4), nn.Identity(), nn.Dropout(0.0), nn.ReLU(), nn.Dropout(0.3), nn.Linear(4, 2))
    net = dv.wrap(m, sample_input=torch.randn(2, 4))
    lay = layout(net.graph)
    assert {"1", "2"}.isdisjoint(lay["nodes"]) and "4" in lay["nodes"]  # Dropout(0.3) matters in training
    assert ("0", "3") in {(e["src"], e["dst"]) for e in lay["edges"]}  # rewired through the no-ops
    assert "1" in net.trace(torch.randn(2, 4)).acts  # still recorded / hookable


def test_parameter_glyph_shows_values_for_any_rank():
    from deepvis.viz.glyphs import GlyphData, render

    class Tok(nn.Module):
        def __init__(self):
            super().__init__()
            self.cls = nn.Parameter(torch.randn(1, 1, 8))
            self.pos = nn.Parameter(torch.randn(1, 5, 8))

        def forward(self, x):
            return torch.cat([self.cls.expand(x.shape[0], -1, -1), x], 1) + self.pos

    net = dv.wrap(Tok(), sample_input=torch.randn(2, 4, 8))
    D = GlyphData(net, None, net.graph)
    for nid in ("@param:cls", "@param:pos"):
        assert "<image" in render(net.graph.nodes[nid], D)
