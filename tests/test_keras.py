"""Keras 3 models — Sequential, functional and subclassed — on the backend KERAS_BACKEND names (default: JAX if it is
installed, else PyTorch). Layers are the nodes; the dataflow follows the tensors from call to call."""

import importlib.util
import os

if "KERAS_BACKEND" not in os.environ:  # must be set before Keras is imported anywhere
    os.environ["KERAS_BACKEND"] = "jax" if importlib.util.find_spec("jax") else "torch"

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import deepvis as dv  # noqa: E402

keras = pytest.importorskip("keras")
layers = keras.layers
pytestmark = pytest.mark.keras


def _np(x):
    return np.asarray(keras.ops.convert_to_numpy(x[0] if isinstance(x, (list, tuple)) else x))


def _cnn():
    keras.utils.set_random_seed(0)
    return keras.Sequential([keras.Input((16, 16, 3)), layers.Conv2D(8, 3, activation="relu", name="conv2d"),
                             layers.BatchNormalization(name="batch_normalization"),
                             layers.MaxPooling2D(name="max_pooling2d"), layers.Flatten(name="flatten"),
                             layers.Dropout(0.2, name="dropout"), layers.Dense(10, activation="softmax", name="dense")],
                            name="cnn")


def test_a_sequential_model_draws_its_layers_and_matches(tmp_path):
    from deepvis.viz.explain import explain

    m = _cnn()
    x = np.random.default_rng(0).standard_normal((2, 16, 16, 3)).astype("float32")
    net = dv.wrap(m, sample_input=x)
    assert net.framework == "keras"
    np.testing.assert_allclose(_np(net.trace(x).output), _np(m(x, training=False)), rtol=1e-5, atol=1e-6)
    nodes = {n.id: n for n in net.graph.compute_nodes()}
    assert [nodes[k].kind for k in ("conv2d", "batch_normalization", "max_pooling2d", "flatten", "dropout", "dense")] \
        == ["Conv2d", "BatchNorm", "MaxPool2d", "Flatten", "Dropout", "Linear"]
    assert "@in:x" in nodes and nodes["conv2d"].out_shapes[0] == (2, 14, 14, 8)  # channels last
    dense = nodes["dense"]
    assert dense.params["kernel"].path == "dense.kernel" and dense.params["kernel"].shape == (392, 10)
    assert dense.roles["weight"].out_axis == 1 and dense.roles["weight"].in_axis == 0  # Keras kernels are (in, out)
    assert nodes["conv2d"].roles["weight"].out_axis == 3
    assert set(nodes["batch_normalization"].roles) >= {"scale", "bias"}
    what = explain(dense, net.graph)["what"]
    assert "Each of its 10 outputs is a weighted sum of all 392 inputs" in what and "Softmax" in what
    assert "from 14 × 14 to 7 × 7" in explain(nodes["max_pooling2d"], net.graph)["what"]
    assert net.summary(show=False).startswith("A convolutional network")
    assert net.export_html(str(tmp_path / "p.html"), x)


def test_functional_models_keep_their_skips_ops_and_nested_models():
    keras.utils.set_random_seed(1)
    inp = keras.Input((8, 8, 3), name="img")
    x = layers.Conv2D(4, 3, padding="same", name="stem")(inp)
    block = keras.Sequential([layers.Conv2D(4, 3, padding="same", activation="relu", name="conv"),
                              layers.BatchNormalization(name="bn")], name="block")
    z = keras.ops.relu(layers.Add(name="add")([x, block(x)]))
    out = layers.Dense(5, name="head")(layers.GlobalAveragePooling2D(name="gap")(z))
    m = keras.Model(inp, out)
    xs = np.random.default_rng(1).standard_normal((1, 8, 8, 3)).astype("float32")
    net = dv.wrap(m, sample_input=xs)
    np.testing.assert_allclose(_np(net(xs)), _np(m(xs)), rtol=1e-5, atol=1e-6)
    g = net.graph
    edges = {(e.src, e.dst) for e in g.edges}
    assert {("@in:img", "stem"), ("stem", "block.conv"), ("block.conv", "block.bn"), ("stem", "add"),
            ("block.bn", "add"), ("add", "@relu"), ("@relu", "gap"), ("gap", "head")} <= edges
    assert g.nodes["block"].op == "container" and g.nodes["block"].kind == "Sequential"
    assert g.nodes["block.conv"].parent == "block" and g.nodes["@relu"].kind == "ReLU"
    assert g.nodes["gap"].kind == "AdaptiveAvgPool2d"


def test_subclassed_models_use_attribute_names_and_shared_layers_are_drawn_per_call():
    class Net(keras.Model):
        def __init__(self):
            super().__init__()
            self.d1 = layers.Dense(16, activation="relu")
            self.blocks = [layers.Dense(16) for _ in range(2)]
            self.out = layers.Dense(3)

        def call(self, x):
            h = self.d1(x)
            for b in self.blocks:
                h = b(h)
            return self.out(self.blocks[0](h))  # blocks.0 called twice

    keras.utils.set_random_seed(2)
    m = Net()
    x = np.random.default_rng(2).standard_normal((2, 6)).astype("float32")
    m(x)
    net = dv.wrap(m, sample_input=x)
    ids = [n.id for n in net.graph.compute_nodes() if n.op == "module"]
    assert ids == ["d1", "blocks.0", "blocks.1", "blocks.0@2", "out"]
    assert net.graph.nodes["blocks.0@2"].attrs.get("shared")
    np.testing.assert_allclose(_np(net(x)), _np(m(x)), rtol=1e-5, atol=1e-6)
    assert sorted(net.backend.param_paths())[:2] == ["blocks.0.bias", "blocks.0.kernel"]


def test_two_inputs_two_outputs():
    a, b = keras.Input((5,), name="a"), keras.Input((7,), name="b")
    h = layers.Concatenate(name="cat")([layers.Dense(8, name="fa")(a), layers.Dense(8, name="fb")(b)])
    m = keras.Model([a, b], [layers.Dense(2, name="y1")(h), layers.Dense(3, name="y2")(h)])
    xa, xb = np.ones((2, 5), "float32"), np.full((2, 7), 0.5, "float32")
    net = dv.wrap(m, sample_input=(xa, xb))
    out = net.trace(xa, xb).output
    want = m([xa, xb])
    assert len(out) == 2
    np.testing.assert_allclose(_np(out[1]), _np(want[1]), rtol=1e-5, atol=1e-6)
    edges = {(e.src, e.dst) for e in net.graph.edges}
    assert {("@in:a", "fa"), ("@in:b", "fb"), ("fa", "cat"), ("fb", "cat"), ("y1", "@out"), ("y2", "@out")} <= edges


def test_weight_edits_and_interventions_reach_the_keras_model():
    from deepvis.errors import OptionError

    keras.utils.set_random_seed(3)
    m = keras.Sequential([keras.Input((4,)), layers.Dense(6, name="a"), layers.Dense(3, name="b")])
    x = np.random.default_rng(3).standard_normal((2, 4)).astype("float32")
    net = dv.wrap(m, sample_input=x)
    bias = net.backend.get_param("b.bias")
    zeroed = net.trace(x, interventions={"a": lambda v, ctx: v * 0}).output  # a's output → 0: b gives its bias
    np.testing.assert_allclose(_np(zeroed), np.broadcast_to(bias, (2, 3)), rtol=1e-6, atol=1e-6)
    net.backend.set_param("b.kernel", np.zeros((6, 3), "float32"))  # assigned into the live model
    np.testing.assert_allclose(_np(m(x)), np.broadcast_to(bias, (2, 3)), rtol=1e-6, atol=1e-6)
    with pytest.raises(OptionError, match="gradients"):
        net.trace(x, grads=True)


def test_recurrent_and_attention_layers_are_single_nodes():
    tok = keras.Input((9,), dtype="int32")
    e = layers.LSTM(12, return_sequences=True, name="lstm")(layers.Embedding(50, 16, name="emb")(tok))
    att = layers.MultiHeadAttention(num_heads=2, key_dim=6, name="att")(e, e)
    m = keras.Model(tok, layers.Dense(4, name="head")(layers.GlobalAveragePooling1D()(e + att)))
    ids = np.random.default_rng(4).integers(0, 50, (1, 9)).astype("int32")
    net = dv.wrap(m, sample_input=ids)
    np.testing.assert_allclose(_np(net(ids)), _np(m(ids)), rtol=1e-5, atol=1e-6)
    g = net.graph
    assert g.nodes["emb"].kind == "Embedding" and g.nodes["lstm"].kind == "LSTM"
    assert g.nodes["att"].kind == "MultiheadAttention" and g.nodes["att"].op == "module"
    assert g.nodes["lstm"].attrs["hidden_size"] == 12
    assert not any(n.startswith("att.") for n in g.nodes)  # its projections stay inside the one node


def test_equations_show_the_activation_a_layer_applies_itself():
    keras.utils.set_random_seed(5)
    m = keras.Sequential([keras.Input((4,)), layers.Dense(5, activation="relu", name="h"),
                          layers.Dense(3, activation="tanh", name="y")])
    x = np.array([[1.0, -2.0, 0.5, 3.0]], "float32")
    net = dv.wrap(m, sample_input=x)
    system = net.equation("system").text
    assert "ReLU(W⁽¹⁾x + b⁽¹⁾)" in system and "tanh(W⁽²⁾h₁ + b⁽²⁾)" in system
    eq = net.equation("scalar", node="h", neuron=0, input=x)
    k, b = net.backend.get_param("h.kernel"), net.backend.get_param("h.bias")
    z = float(x[0] @ k[:, 0] + b[0])  # the note on z is before the activation, the one on a after it
    assert eq.lines[0].note == f"= {z:.3g} for this input" and "ReLU(z⁽¹⁾₀)" in eq.text
    assert eq.lines[1].note == f"= {max(z, 0.0):.3g}"
