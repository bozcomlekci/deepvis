"""General-purpose wrapping: real architectures, recurrent nets, multi-input/output, skips, tied weights."""

import json

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import deepvis as dv
from deepvis.viz.html.payload import PayloadBuilder

tv = pytest.importorskip("torchvision")


class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb, self.lstm, self.head = nn.Embedding(50, 8), nn.LSTM(8, 16, batch_first=True), nn.Linear(16, 3)

    def forward(self, ids):
        out, _ = self.lstm(self.emb(ids))
        return self.head(out[:, -1])


class MultiIO(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.meta, self.head, self.aux = nn.Linear(5, 4), nn.Linear(8, 3), nn.Linear(8, 1)

    def forward(self, img, meta):
        z = torch.cat([self.cnn(img), F.relu(self.meta(meta))], dim=1)
        return {"logits": self.head(z), "aux": torch.sigmoid(self.aux(z))}


class UNetTiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.ReLU())
        self.d2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(4, 8, 3, padding=1), nn.ReLU())
        self.up, self.out = nn.ConvTranspose2d(8, 4, 2, stride=2), nn.Conv2d(8, 2, 1)

    def forward(self, x):
        a = self.d1(x)
        return self.out(torch.cat([a, self.up(self.d2(a))], dim=1))


class Tied(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb, self.block = nn.Embedding(20, 8), nn.Linear(8, 8)

    def forward(self, ids):
        h = self.emb(ids)
        h = h + torch.tanh(self.block(h))
        h = h + torch.tanh(self.block(h))  # the same module twice
        return h @ self.emb.weight.T


def zoo():
    torch.manual_seed(0)
    vit = tv.models.VisionTransformer(image_size=16, patch_size=8, num_layers=1, num_heads=2, hidden_dim=16,
                                      mlp_dim=32, num_classes=4)
    return {
        "resnet18": (tv.models.resnet18(num_classes=4).eval(), (torch.randn(1, 3, 32, 32),)),
        "mobilenet": (tv.models.mobilenet_v3_small(num_classes=4).eval(), (torch.randn(1, 3, 32, 32),)),
        "vit": (vit.eval(), (torch.randn(1, 3, 16, 16),)),
        "lstm": (LSTMClassifier().eval(), (torch.randint(0, 50, (2, 6)),)),
        "multi_io": (MultiIO().eval(), (torch.randn(2, 3, 8, 8), torch.randn(2, 5))),
        "unet": (UNetTiny().eval(), (torch.randn(1, 1, 16, 16),)),
        "tied": (Tied().eval(), (torch.randint(0, 20, (2, 5)),)),
    }


@pytest.mark.parametrize("name", list(zoo()))
def test_any_network(name):
    model, args = zoo()[name]
    net = dv.wrap(model, sample_input=args)
    assert net.backend.trace_mode == "dispatch"
    out, ref = net(*args), model(*args)
    for a, b in zip(torch.utils._pytree.tree_leaves(out), torch.utils._pytree.tree_leaves(ref)):
        assert torch.allclose(a, b)
    eq = net.equation()
    assert eq.lines and "Parameter(" not in eq.text
    tr = net.trace(*args, grads=True, loss_fn=lambda y: sum(t.float().sum() for t in torch.utils._pytree.tree_leaves(y)))
    assert tr.param_grads
    p = PayloadBuilder(net).build(args)
    json.dumps(p)
    vols = [a["volume"] for a in p["acts"].values() if a.get("volume")]
    assert vols, "activations carry 3-D volume faces"
    # edits work anywhere
    first = next(n for n in net.graph.param_nodes() if "weight" in n.roles)
    net[first.id].noise(0.01, seed=0)
    net.undo()


def test_equation_details():
    model, args = zoo()["tied"]
    assert "E⁽¹⁾ᵀ" in dv.wrap(model, sample_input=args).equation().text  # tied weights keep their symbol
    model, args = zoo()["unet"]
    assert "⋆ᵀ" in dv.wrap(model, sample_input=args).equation().text     # transposed convolution
    model, args = zoo()["mobilenet"]
    assert "hswish" in dv.wrap(model, sample_input=args).equation().text


def test_jax_scan_rnn():
    jax = pytest.importorskip("jax")
    jnp = jax.numpy

    def rnn(p, xs):
        def cell(h, x):
            h = jnp.tanh(x @ p["Wx"] + h @ p["Wh"])
            return h, h
        _, hs = jax.lax.scan(cell, jnp.zeros((xs.shape[0], 4)), jnp.swapaxes(xs, 0, 1))
        return hs[-1] @ p["Wo"]

    k = jax.random.split(jax.random.PRNGKey(0), 3)
    p = {"Wx": jax.random.normal(k[0], (3, 4)), "Wh": jax.random.normal(k[1], (4, 4)), "Wo": jax.random.normal(k[2], (4, 2))}
    xs = jnp.ones((2, 5, 3))
    net = dv.wrap(rnn, p, sample_input=xs)
    assert "Scan" in [n.kind for n in net.graph.compute_nodes()]
    assert np.allclose(net(xs), rnn(p, xs), atol=1e-5)
    assert "scan_θ" in net.equation().text
