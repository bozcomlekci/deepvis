"""Plain-language layer explanations (deepvis.viz.explain) and colour-coded equations (deepvis.equations.expr.painted)."""

import pytest

import deepvis as dv
from deepvis.core.graph import Node, ParamInfo
from deepvis.core.kinds import ACTIVATION_KINDS, CONV_KINDS, NORM_KINDS, Role
from deepvis.viz.explain import caption, explain

torch = pytest.importorskip("torch")
nn = torch.nn


def _whats(net):
    return {n.id: (explain(n, net.graph) or {}).get("what", "") for n in net.graph.compute_nodes()}


@pytest.mark.torch
def test_explanations_quote_this_layers_numbers():
    m = nn.Sequential(nn.Conv2d(3, 8, 3, stride=2, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
                      nn.Flatten(), nn.Dropout(0.25), nn.Linear(128, 10), nn.Softmax(-1))
    w = _whats(dv.wrap(m, sample_input=torch.randn(2, 3, 16, 16)))
    assert "8 small filters, each 3 × 3 across all 3 input channels" in w["0"] and "in steps of 2" in w["0"]
    assert "each of its 8 channels" in w["1"]
    assert "max(0, x)" in w["2"]
    assert "each 2 × 2 window" in w["3"] and "from 8 × 8 to 4 × 4" in w["3"]
    assert "one row of 128" in w["4"]
    assert "25% of the values" in w["5"]
    assert "10 outputs" in w["6"] and "all 128 inputs" in w["6"] and "1,290 learned numbers" in w["6"]
    assert "10 scores into probabilities" in w["7"]
    assert "3 × 16 × 16 per example" in w["@in:input"]


@pytest.mark.torch
def test_residual_attention_and_recurrent_layers():
    layer = nn.TransformerEncoderLayer(32, 4, 64, batch_first=True, dropout=0.0)
    enc = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False).eval()
    w = _whats(dv.wrap(enc, sample_input=torch.randn(1, 12, 32)))
    assert "4 heads of 8 numbers each" in w["layers.0.self_attn"] and "12 × 12 scores" in w["layers.0.self_attn"]
    assert "residual (skip) connection" in w["@add"]

    class R(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(100, 16)
            self.lstm = nn.LSTM(16, 24, num_layers=2, bidirectional=True, batch_first=True)

        def forward(self, x):
            return self.lstm(self.emb(x))[0]

    w = _whats(dv.wrap(R(), sample_input=torch.randint(0, 100, (1, 7))))
    assert "100 rows with 16 numbers each" in w["emb"]
    assert "memory of 24 numbers" in w["lstm"] and "2 stacked layers" in w["lstm"] and "backwards" in w["lstm"]
    assert "7 values per example" in w["@in:x"]


def test_every_known_kind_has_words():
    def node(kind):
        return Node("n", kind, kind, "module", in_shapes=[(1, 4, 6, 6)], out_shapes=[(1, 4, 6, 6)])

    for kind in sorted(ACTIVATION_KINDS | NORM_KINDS):
        e = explain(node(kind))
        assert e and e["what"], kind
    for kind in CONV_KINDS:  # a convolution needs its weight to say anything concrete
        n = node(kind)
        n.params = {"weight": ParamInfo("n.weight", "weight", "n", (4, 4, 3, 3), "float32")}
        n.roles = {"weight": Role("weight", out_axis=0, in_axis=1)}
        assert "4 small filters" in explain(n)["what"], kind
    assert caption(node("ReLU")).startswith("Keeps positive values")


@pytest.mark.jax
def test_flax_explanations_read_channels_last():
    jax = pytest.importorskip("jax")
    fnn = pytest.importorskip("flax.linen")

    class CNN(fnn.Module):
        @fnn.compact
        def __call__(self, x):
            x = fnn.max_pool(fnn.relu(fnn.Conv(8, (3, 3), strides=(2, 2))(x)), (2, 2), strides=(2, 2))
            return fnn.Dense(10)(x.reshape(x.shape[0], -1))

    x = jax.numpy.ones((2, 16, 16, 3))
    m = CNN()
    net = dv.wrap(m, m.init(jax.random.PRNGKey(0), x), sample_input=x)
    w = {n.kind: (explain(n, net.graph) or {}).get("what", "") for n in net.graph.compute_nodes()}
    assert "8 small filters, each 3 × 3 across all 3 input channels" in w["Conv2d"]
    assert "from 8 × 8 to 4 × 4" in w["MaxPool2d"]
    assert "10 outputs" in w["Linear"] and "all 128 inputs" in w["Linear"]


@pytest.mark.torch
def test_page_equations_are_colour_coded_and_linked_but_the_api_stays_plain():
    from deepvis.viz.html.payload import PayloadBuilder

    m = nn.Sequential(nn.Linear(4, 3), nn.Sigmoid(), nn.Linear(3, 2))
    x = torch.randn(1, 4)
    net = dv.wrap(m, sample_input=x)
    pay = PayloadBuilder(net).build((x,))
    rhs = pay["equations"]["modes"]["system"]["lines"][0]["rhs"]
    assert r"\textcolor{#58C4DD}" in rhs and r"\htmlData{n=" in rhs  # W⁽¹⁾ in the dense-layer blue, tagged
    tags = {int(t) for t in __import__("re").findall(r"\\htmlData\{n=(\d+)\}", rhs)}
    assert all(0 <= t < len(pay["eqNodes"]) for t in tags) and "0" in pay["eqNodes"]
    neuron = next(iter(pay["neurons"]["equations"].values()))["lines"][0]["rhs"]
    assert r"\textcolor{#58C4DD}" in neuron or r"\textcolor{#FC6255}" in neuron  # terms by their weight's sign
    assert set(pay["explain"]) >= {n.id for n in net.graph.compute_nodes() if n.op != "output"} - {""}
    plain = net.equation("system").latex  # the Python API is unchanged
    assert r"\textcolor" not in plain and r"\htmlData" not in plain


@pytest.mark.torch
def test_the_whole_network_in_one_sentence(capsys):
    from deepvis.viz.explain import describe

    mlp = nn.Sequential(nn.Flatten(), nn.Linear(784, 16), nn.Sigmoid(), nn.Linear(16, 10), nn.Softmax(-1))
    net = dv.wrap(mlp, sample_input=torch.rand(1, 1, 28, 28))
    s = describe(net.graph)
    assert s.startswith("A fully connected network (an MLP) that turns a 1 × 28 × 28 input into 10 probabilities")
    assert "2 dense layers, 2 nonlinearities" in s and "12,730 learned numbers" in s
    assert describe(net.graph, detail=False).endswith("into 10 probabilities, with 12,730 learned numbers.")
    net.summary()
    assert "A fully connected network" in capsys.readouterr().out
    enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(32, 4, 64, batch_first=True, dropout=0.0), 2,
                                enable_nested_tensor=False).eval()
    s = describe(dv.wrap(enc, sample_input=torch.randn(1, 12, 32)).graph)
    assert s.startswith("A transformer-style network that turns a 12 × 32 input into a 12 × 32 tensor")
    assert "2 attention layers" in s and "4 residual connections" in s


@pytest.mark.torch
def test_projection_shortcuts_count_as_residual_and_token_ids_are_recognised():
    from deepvis.viz.explain import describe

    tv = pytest.importorskip("torchvision")
    s = describe(dv.wrap(tv.models.resnet18().eval(), sample_input=torch.rand(1, 3, 64, 64)).graph)
    assert "into 1,000 values" in s and "8 residual connections" in s  # 3 of the 8 shortcuts are projections

    class Clf(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb, self.lstm, self.head = nn.Embedding(100, 16), nn.LSTM(16, 24, batch_first=True), nn.Linear(24, 3)

        def forward(self, x):
            return self.head(self.lstm(self.emb(x))[0][:, -1])

    s = describe(dv.wrap(Clf(), sample_input=torch.randint(0, 100, (1, 7))).graph)
    assert s.startswith("A recurrent network that turns a sequence of 7 token ids into 3 values")
