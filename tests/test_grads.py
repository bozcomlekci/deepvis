"""Gradient flow: activation / parameter / input gradients in traces, payloads and pages."""

import json
import urllib.request

import numpy as np
import pytest

import deepvis as dv
from deepvis.viz.html.payload import PayloadBuilder

torch = pytest.importorskip("torch")
nn = torch.nn


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.b = nn.Linear(4, 4), nn.Linear(4, 2)

    def forward(self, x):
        return self.b(torch.relu(self.a(x)) + x)


@pytest.mark.torch
def test_torch_grads_reach_functional_ops_and_inputs():
    torch.manual_seed(0)
    m, x = Residual(), torch.randn(3, 4)
    net = dv.wrap(m, sample_input=x)
    tr = net.trace(x, grads=True, loss_fn=lambda y: y[:, 1].sum())
    xx = x.clone().requires_grad_(True)
    m(xx)[:, 1].sum().backward()
    inp = net.graph.inputs()[0].id
    assert {"@add", "@relu", "a", "b", inp} <= set(tr.grads)
    assert np.allclose(tr.grads[inp], xx.grad.numpy(), atol=1e-6)
    assert np.allclose(tr.param_grads["a.weight"], m.a.weight.grad.numpy(), atol=1e-6)
    m.zero_grad()
    assert m.a.weight.grad is None or not m.a.weight.grad.any()  # the trace restored the user's .grad


@pytest.mark.torch
def test_gradient_flow_payload_shows_vanishing_gradients():
    torch.manual_seed(0)
    layers = [nn.Linear(8, 16), nn.Sigmoid()]
    for _ in range(6):
        layers += [nn.Linear(16, 16), nn.Sigmoid()]
    m = nn.Sequential(*layers, nn.Linear(16, 1))
    x = torch.randn(4, 8)
    net = dv.wrap(m, sample_input=x)
    pay = PayloadBuilder(net).build((x,), grads=True)
    rows = pay["gradflow"]["rows"]
    assert rows[0]["op"] == "input" and rows[-1]["id"] == str(len(layers))
    acts = [r["act"] for r in rows if r.get("act")]
    assert acts[-1] / acts[0] > 1e3  # sigmoids shrink the gradient on its way back
    linear = [r for r in rows if r["kind"] == "Linear"]
    assert all(r["param"] > 0 and r["paramMax"] >= r["param"] for r in linear)
    assert "gradView" in pay["acts"]["0"] and "vector" in pay["acts"]["0"]["gradView"]
    json.dumps(pay["gradflow"], allow_nan=False)
    assert PayloadBuilder(net).build((x,))["gradflow"] is None  # only on request


@pytest.mark.torch
def test_input_saliency_and_export(tmp_path, cnn):
    model, x = cnn
    net = dv.wrap(model, sample_input=x)
    pay = PayloadBuilder(net).build((x,), grads=True, loss_fn=lambda y: y[:, 0].sum())
    inp = net.graph.inputs()[0].id
    sal = pay["acts"][inp]["gradView"]["saliency"]
    assert sal["orig"] == list(x.shape[-2:])
    path = net.export_html(tmp_path / "g.html", x, grads=True, start="monitor")
    assert '"gradflow":{' in open(path, encoding="utf-8").read()


@pytest.mark.torch
def test_live_server_backpropagates_without_touching_training_grads():
    torch.manual_seed(0)
    m, x = Residual(), torch.randn(3, 4)
    net = dv.wrap(m, sample_input=x)
    srv = net.serve(x, open_browser=False, grads=True, min_interval=0.0)
    try:
        m(x).pow(2).sum().backward()
        g = m.a.weight.grad.clone()
        srv.update(0, loss=1.0)
        assert torch.equal(g, m.a.weight.grad)
        state = json.load(urllib.request.urlopen(srv.url + "api/state"))
        assert state["gradflow"]["rows"]
    finally:
        srv.stop()


@pytest.mark.jax
def test_jax_gradient_flow():
    jax = pytest.importorskip("jax")
    lnn = pytest.importorskip("flax.linen")

    class M(lnn.Module):
        @lnn.compact
        def __call__(self, x):
            return lnn.Dense(2)(lnn.tanh(lnn.Dense(4)(x)))

    x = jax.random.normal(jax.random.PRNGKey(0), (3, 5))
    m = M()
    net = dv.wrap(m, m.init(jax.random.PRNGKey(1), x), sample_input=x)
    rows = PayloadBuilder(net).build((x,), grads=True)["gradflow"]["rows"]
    assert [r["kind"] for r in rows if "param" in r] == ["Linear", "Linear"]
    assert rows[0]["op"] == "input" and rows[0]["act"] > 0
