"""ONNX files: layers from the exporter's scopes, parity with onnxruntime and the PyTorch original, weight edits."""

import numpy as np
import pytest

import deepvis as dv

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
nn = torch.nn


def _export(m, args, path, dynamo=False):
    torch.onnx.export(m.eval(), args, str(path), dynamo=dynamo)
    return str(path)


def _cnn():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
                         nn.Linear(8 * 8 * 8, 10)).eval()


@pytest.mark.torch
def test_a_torchscript_export_draws_the_modules_and_matches(tmp_path):
    m = _cnn()
    x = torch.randn(2, 3, 16, 16)
    path = _export(m, (x,), tmp_path / "cnn.onnx")
    net = dv.wrap(path)  # the file is the model; its declared inputs make a sample
    tr = net.trace(x.numpy())
    with torch.no_grad():
        np.testing.assert_allclose(tr.output, m(x).numpy(), rtol=1e-4, atol=1e-5)
    nodes = {n.id: n for n in net.graph.compute_nodes()}
    assert nodes["0"].kind == "Conv2d" and nodes["5"].kind == "Linear" and nodes["2"].kind == "ReLU"
    conv = nodes["0"]  # BatchNorm folded into the convolution by the exporter: the weight keeps the module's name
    assert set(conv.params) == {"weight", "bias"} and conv.params["weight"].path == "0.weight"
    assert conv.roles["weight"].out_axis == 0 and conv.attrs["out_channels"] == 8
    lin = nodes["5"]
    assert lin.params["weight"].shape == (10, 512) and lin.roles["weight"].out_axis == 0  # Gemm, transB=1
    assert net.framework == "onnx" and net.summary(show=False).startswith("A convolutional network")
    assert net.export_html(str(tmp_path / "p.html"), x.numpy())


@pytest.mark.torch
def test_weight_edits_reach_onnxruntime_and_match_the_same_edit_in_torch(tmp_path):
    m = _cnn()
    x = torch.randn(1, 3, 16, 16)
    net = dv.wrap(_export(m, (x,), tmp_path / "cnn.onnx"), sample_input=x.numpy())
    w = net.backend.get_param("5.weight")
    w[3] = 0.0
    net.backend.set_param("5.weight", w)
    with torch.no_grad():
        m[5].weight[3] = 0.0
        want = m(x).numpy()
    np.testing.assert_allclose(net(x.numpy()), want, rtol=1e-4, atol=1e-5)
    assert net(x.numpy())[0, 3] == pytest.approx(float(m[5].bias[3]))
    out = net.backend.save(str(tmp_path / "edited.onnx"))  # the edit is saved in the file
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    np.testing.assert_allclose(sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0], want, rtol=1e-4, atol=1e-5)


@pytest.mark.torch
def test_dynamo_export_attention_folds_back_into_one_node(tmp_path):
    pytest.importorskip("onnxscript")
    torch.manual_seed(0)
    enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(32, 4, 64, batch_first=True, dropout=0.0), 2,
                                enable_nested_tensor=False).eval()
    x = torch.randn(1, 10, 32)
    net = dv.wrap(_export(enc, (x,), tmp_path / "enc.onnx", dynamo=True), sample_input=x.numpy())
    with torch.no_grad():
        np.testing.assert_allclose(net.trace(x.numpy()).output, enc(x).numpy(), rtol=1e-4, atol=1e-5)
    g = net.graph
    attn = [n for n in g.compute_nodes() if n.kind == "Attention"]
    assert len(attn) == 2  # scaled_dot_product_attention's Mul/MatMul/Softmax/MatMul: one node each
    assert g.nodes["layers.0.self_attn"].op == "container" and g.nodes["layers.0.self_attn"].kind == "MultiheadAttention"
    assert {"layers.0.self_attn.in_proj", "layers.0.self_attn.out_proj", "layers.1.norm2"} <= set(g.nodes)
    assert g.nodes["layers.0"].kind == "TransformerEncoderLayer"
    # both layers started as copies, so the exporter stored their weights once: an edit stays in its layer
    p = "layers.1.self_attn.out_proj.weight"
    w = net.backend.get_param(p)
    net.backend.set_param(p, w * 0.0)
    with torch.no_grad():
        enc.layers[1].self_attn.out_proj.weight.zero_()
        np.testing.assert_allclose(net(x.numpy()), enc(x).numpy(), rtol=1e-4, atol=1e-5)
    assert np.abs(net.backend.get_param("layers.0.self_attn.out_proj.weight")).sum() > 0
    assert net.export_html(str(tmp_path / "p.html"), x.numpy())


@pytest.mark.torch
def test_two_inputs_two_outputs_and_a_recurrent_layer(tmp_path):
    class TwoIO(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b, self.head = nn.Linear(4, 8), nn.Linear(6, 8), nn.Linear(8, 3)

        def forward(self, x, y):
            h = torch.tanh(self.a(x) + self.b(y))
            return self.head(h), h.mean(-1)

    torch.manual_seed(0)
    m = TwoIO().eval()
    x, y = torch.randn(2, 4), torch.randn(2, 6)
    net = dv.wrap(_export(m, (x, y), tmp_path / "two.onnx"), sample_input=(x.numpy(), y.numpy()))
    out = net.trace(x.numpy(), y.numpy()).output
    with torch.no_grad():
        want = m(x, y)
    assert isinstance(out, tuple) and len(out) == 2
    np.testing.assert_allclose(out[1], want[1].numpy(), rtol=1e-4, atol=1e-5)
    assert len([n for n in net.graph.nodes.values() if n.op == "input"]) == 2

    class Clf(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb, self.lstm, self.head = nn.Embedding(100, 16), nn.LSTM(16, 24, batch_first=True), nn.Linear(24, 3)

        def forward(self, t):
            return self.head(self.lstm(self.emb(t))[0][:, -1])

    c = Clf().eval()
    ids = torch.randint(0, 100, (1, 7))
    net = dv.wrap(_export(c, (ids,), tmp_path / "lstm.onnx"), sample_input=ids.numpy())
    with torch.no_grad():
        np.testing.assert_allclose(net.trace(ids.numpy()).output, c(ids).numpy(), rtol=1e-4, atol=1e-5)
    lstm = net.graph.nodes["lstm"]
    assert lstm.kind == "LSTM" and lstm.attrs["hidden_size"] == 24 and set(lstm.params) >= {"weight_ih", "weight_hh"}
    assert net.graph.nodes["emb"].kind == "Embedding"
    from deepvis.viz.explain import explain

    assert "memory of 24 numbers" in explain(lstm, net.graph)["what"]


@pytest.mark.torch
def test_onnx_explains_what_it_cannot_do(tmp_path):
    from deepvis.errors import OptionError

    m = _cnn()
    x = torch.randn(1, 3, 16, 16)
    net = dv.wrap(_export(m, (x,), tmp_path / "cnn.onnx"), sample_input=x.numpy())
    with pytest.raises(OptionError, match="gradients"):
        net.trace(x.numpy(), grads=True)
    with pytest.raises(OptionError, match="onnxruntime"):
        with net.intervene("2", lambda v, ctx: v * 0):
            net.trace(x.numpy())
    with pytest.raises(ValueError, match="takes 1 input"):
        net(x.numpy(), x.numpy())


def test_scope_names_from_both_exporters():
    from deepvis.backends.onnx_backend import _common, _rebase, _scope_levels

    assert _scope_levels(["layer1", "layer1.0", "conv1"])[-1] == "layer1.0.conv1"
    assert _scope_levels(["layer2", "layer2.0", "downsample", "downsample.0"])[-1] == "layer2.0.downsample.0"
    assert _scope_levels(["features", "features.1", "block", "block.0", "block.0.0"])[-1] == "features.1.block.0.0"
    assert _scope_levels(["m", "transformer", "h.0", "attn"])[-1] == "m.transformer.h.0.attn"
    assert _scope_levels(["", "m", "m.transformer", "m.transformer.h.0"])[-1] == "m.transformer.h.0"
    assert _rebase("layers.0.self_attn.out_proj", "layers.1.self_attn") == "layers.1.self_attn.out_proj"
    assert _rebase("layer1.0.conv1", "layer1.0.bn1") == "layer1.0.conv1"
    assert _rebase("m.lm_head", "m.transformer.wte") is None
    assert _common(["a.b.c", "a.b", "a.b.d.e"]) == "a.b" and _common(["a.b", ""]) == ""


def test_a_scikit_learn_model_exported_to_onnx():
    sk = pytest.importorskip("sklearn.neural_network")
    skl2onnx = pytest.importorskip("skl2onnx")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 6)).astype(np.float32)
    clf = sk.MLPClassifier(hidden_layer_sizes=(8,), max_iter=50, random_state=0).fit(X, (X[:, 0] > 0).astype(int))
    onx = skl2onnx.to_onnx(clf, X[:1], options={id(clf): {"zipmap": False}})
    net = dv.wrap(onx, sample_input=X[:3])
    labels, probs = net.trace(X[:3]).output
    np.testing.assert_array_equal(labels, clf.predict(X[:3]))
    np.testing.assert_allclose(probs, clf.predict_proba(X[:3]), rtol=1e-4, atol=1e-5)
    layers = [n for n in net.graph.compute_nodes() if n.op == "module"]
    assert [(n.id, n.kind, set(n.params)) for n in layers] == [("linear", "Linear", {"weight", "bias"}),
                                                              ("linear_1", "Linear", {"weight", "bias"})]
