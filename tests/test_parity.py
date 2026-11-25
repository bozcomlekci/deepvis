"""The same MLP in five frameworks gives the same graph kinds, equations and activations."""

import numpy as np
import pytest
import torch

import deepvis as dv

jax = pytest.importorskip("jax")
jnp = jax.numpy
nn = pytest.importorskip("flax.linen")
eqx = pytest.importorskip("equinox")
from flax import nnx  # noqa: E402

rng = np.random.default_rng(0)
W1, b1 = rng.normal(size=(5, 4)).astype(np.float32), rng.normal(size=5).astype(np.float32)
W2, b2 = rng.normal(size=(3, 5)).astype(np.float32), rng.normal(size=3).astype(np.float32)
X = rng.normal(size=(2, 4)).astype(np.float32)


def torch_net():
    m = torch.nn.Sequential(torch.nn.Linear(4, 5), torch.nn.Tanh(), torch.nn.Linear(5, 3))
    with torch.no_grad():
        m[0].weight.copy_(torch.tensor(W1)); m[0].bias.copy_(torch.tensor(b1))
        m[2].weight.copy_(torch.tensor(W2)); m[2].bias.copy_(torch.tensor(b2))
    return dv.wrap(m, sample_input=torch.tensor(X)), torch.tensor(X)


def linen_net():
    class M(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.Dense(3, name="Dense_1")(jnp.tanh(nn.Dense(5, name="Dense_0")(x)))

    v = {"params": {"Dense_0": {"kernel": W1.T, "bias": b1}, "Dense_1": {"kernel": W2.T, "bias": b2}}}
    return dv.wrap(M(), v, sample_input=jnp.asarray(X)), jnp.asarray(X)


def nnx_net():
    class M(nnx.Module):
        def __init__(self):
            r = nnx.Rngs(0)
            self.a, self.b = nnx.Linear(4, 5, rngs=r), nnx.Linear(5, 3, rngs=r)

        def __call__(self, x):
            return self.b(jnp.tanh(self.a(x)))

    m = M()
    m.a.kernel[...], m.a.bias[...] = jnp.asarray(W1.T), jnp.asarray(b1)
    m.b.kernel[...], m.b.bias[...] = jnp.asarray(W2.T), jnp.asarray(b2)
    return dv.wrap(m, sample_input=jnp.asarray(X)), jnp.asarray(X)


def eqx_net():
    k = jax.random.PRNGKey(0)
    layers = [eqx.nn.Linear(4, 5, key=k), eqx.nn.Lambda(jnp.tanh), eqx.nn.Linear(5, 3, key=k)]
    layers[0] = eqx.tree_at(lambda l: (l.weight, l.bias), layers[0], (jnp.asarray(W1), jnp.asarray(b1)))
    layers[2] = eqx.tree_at(lambda l: (l.weight, l.bias), layers[2], (jnp.asarray(W2), jnp.asarray(b2)))
    seq = eqx.nn.Sequential(layers)
    return dv.wrap(seq, sample_input=jnp.asarray(X[0])), jnp.asarray(X[0])


def pure_net():
    p = {"l1": {"w": jnp.asarray(W1.T), "b": jnp.asarray(b1)}, "l2": {"w": jnp.asarray(W2.T), "b": jnp.asarray(b2)}}
    return dv.wrap(lambda p, x: jnp.tanh(x @ p["l1"]["w"] + p["l1"]["b"]) @ p["l2"]["w"] + p["l2"]["b"], p,
                   sample_input=jnp.asarray(X)), jnp.asarray(X)


BUILDERS = [torch_net, linen_net, nnx_net, eqx_net, pure_net]


@pytest.mark.parametrize("build", BUILDERS, ids=lambda b: b.__name__)
def test_parity(build):
    net, x = build()
    lin = [n for n in net.graph.param_nodes()]
    assert [n.kind for n in lin] == ["Linear", "Linear"]
    tr = net.trace(x)
    h1 = tr[lin[0].id].reshape(-1, 5)[0]
    y = tr[lin[1].id].reshape(-1, 3)[0]
    assert np.allclose(h1, W1 @ X[0] + b1, atol=1e-5)
    assert np.allclose(y, W2 @ np.tanh(W1 @ X[0] + b1) + b2, atol=1e-5)
    eq = net.equation("expanded")
    assert eq.lines[0].rhs.text() == "W⁽²⁾tanh(W⁽¹⁾x + b⁽¹⁾) + b⁽²⁾"
    # canonical (out, in) orientation regardless of storage layout
    assert np.allclose(net[lin[0].id].weight_matrix(), W1)
    s = net.equation("scalar", neuron=1, top_k=5)
    assert f"{abs(W2[1]).max():.3g}"[:4] in s.text
