import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

import deepvis as dv  # noqa: E402

nn = pytest.importorskip("flax.linen")


class Block(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x + nn.relu(nn.LayerNorm(name="ln")(nn.Dense(8, name="fc")(x)))


class LinenNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(8, name="inp")(x))
        x = Block(name="b0")(x)
        return nn.softmax(nn.Dense(3, name="out")(x))


@pytest.fixture
def linen():
    x = jax.random.normal(jax.random.PRNGKey(1), (4, 5))
    m = LinenNet()
    return m, m.init(jax.random.PRNGKey(0), x), x


def test_linen_graph_trace_grads(linen):
    m, v, x = linen
    net = dv.wrap(m, v, sample_input=x)
    kinds = [n.kind for n in net.graph.compute_nodes()]
    assert "add" in kinds and "Softmax" in kinds and kinds.count("Linear") == 3
    assert np.allclose(net(x), m.apply(v, x), atol=1e-6)
    _, inter = m.apply(v, x, capture_intermediates=True, mutable=["intermediates"])
    tr = net.trace(x, grads=True)
    assert np.allclose(tr["b0.fc"], inter["intermediates"]["b0"]["fc"]["__call__"][0], atol=1e-6)
    g = jax.grad(lambda vv: jnp.sum(m.apply(vv, x)))(v)
    assert np.allclose(tr.param_grads["b0.fc.kernel"], g["params"]["b0"]["fc"]["kernel"], atol=1e-5)


def test_linen_edits_interventions_export(linen):
    m, v, x = linen
    net = dv.wrap(m, v, sample_input=x)
    y0 = np.asarray(net(x))
    net["inp"].neurons([0, 2]).zero()
    k = net["inp.kernel"].numpy()
    assert np.all(k[:, [0, 2]] == 0), "flax kernels are (in, out): neurons are columns"
    _, variables = net.export()
    assert np.allclose(m.apply(variables, x), net(x))
    net.undo()
    assert np.allclose(net(x), y0)
    with net.intervene("@ReLU", dv.ablate()):
        tr = net.trace(x)
        assert np.all(tr["@ReLU"] == 0)
    with net.monitor("b0.fc", "@add") as mon:
        net(x)
    assert mon["b0.fc"].calls == 1 and mon["@add"].calls == 1


def test_linen_permutation_invariance():
    class M(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.Dense(2, name="b")(jnp.tanh(nn.Dense(6, name="a")(x)))

    x = jax.random.normal(jax.random.PRNGKey(2), (3, 4))
    m = M()
    net = dv.wrap(m, m.init(jax.random.PRNGKey(0), x), sample_input=x)
    y0 = np.asarray(net(x))
    net.permute_neurons("a", [5, 4, 3, 2, 1, 0])
    assert np.allclose(net(x), y0, atol=1e-6)


def test_pure_function_inferred_layers():
    def f(p, x):
        h = jnp.tanh(x @ p["W1"] + p["b1"])
        return h @ p["W2"] + p["b2"]

    k = jax.random.split(jax.random.PRNGKey(0), 2)
    p = {"W1": jax.random.normal(k[0], (4, 6)), "b1": jnp.zeros(6), "W2": jax.random.normal(k[1], (6, 2)),
         "b2": jnp.ones(2)}
    x = jnp.ones((3, 4))
    net = dv.wrap(f, p, sample_input=x)
    linear = [n for n in net.graph.compute_nodes() if n.kind == "Linear"]
    assert [n.id for n in linear] == ["W1", "W2"]
    assert linear[0].roles["weight"].out_axis == 1 and "bias" in linear[0].roles
    assert "tanh(W⁽¹⁾x + b⁽¹⁾)" in net.equation("expanded").text
    net.focus("W1").neurons(0).zero()  # net["W1"] is the parameter; focus() gives the node
    assert np.all(np.asarray(net.export()["W1"])[:, 0] == 0)


def test_pure_named_scopes():
    def f(p, x):
        with dv.scope("encoder"):
            h = jax.nn.relu(x @ p["w"])
        with dv.scope("decoder"):
            return h @ p["v"]

    p = {"w": jnp.ones((3, 4)), "v": jnp.ones((4, 2))}
    net = dv.wrap(f, p, sample_input=jnp.ones((2, 3)))
    assert {"encoder", "decoder"} <= set(net.graph.nodes)
    tr = net.trace(jnp.ones((2, 3)))
    assert tr["encoder"].shape == (2, 4)


def test_equinox():
    eqx = pytest.importorskip("equinox")
    mlp = eqx.nn.MLP(4, 2, 6, 2, key=jax.random.PRNGKey(0))
    x = jnp.ones(4)
    net = dv.wrap(mlp, sample_input=x)
    assert [n.kind for n in net.graph.compute_nodes() if n.op == "module"] == ["Linear"] * 3
    assert np.allclose(net(x), mlp(x), atol=1e-6)
    net["layers.0.weight"][0] = 0.0
    assert np.all(np.asarray(net.export().layers[0].weight[0]) == 0)


def test_nnx_live_sync():
    nnx = pytest.importorskip("flax.nnx")

    class M(nnx.Module):
        def __init__(self, rngs):
            self.fc1 = nnx.Linear(4, 6, rngs=rngs)
            self.drop = nnx.Dropout(0.5, rngs=rngs)
            self.fc2 = nnx.Linear(6, 2, rngs=rngs)

        def __call__(self, x):
            return self.fc2(self.drop(nnx.gelu(self.fc1(x))))

    m = M(nnx.Rngs(0))
    x = jnp.ones((2, 4))
    net = dv.wrap(m, sample_input=x)
    m.eval()
    assert np.allclose(net(x), m(x), atol=1e-6)
    net["fc1"].scale(0.0)
    assert float(jnp.abs(m.fc1.kernel[...]).max()) == 0.0


def test_an_anonymous_softmax_call_is_named_softmax():
    """jax ≥ 0.12 wraps jax.nn.softmax in a custom_jvp_call without a name: it is recognized by what it computes."""
    from types import SimpleNamespace as NS

    from deepvis.backends.jax.interp import _call_name

    def eq(prim, **params):
        return NS(primitive=NS(name=prim), params=params)

    softmax = NS(eqns=[eq(p) for p in ("reduce_max", "max", "broadcast_in_dim", "sub", "exp", "reduce_sum", "div")])
    assert _call_name(eq("custom_jvp_call", call_jaxpr=softmax)) == "softmax"
    more = NS(eqns=[eq(p) for p in ("exp", "reduce_sum", "div", "dot_general")])  # computes more than a softmax
    assert _call_name(eq("custom_jvp_call", call_jaxpr=more)) is None
    assert _call_name(eq("jit", name="relu", jaxpr=softmax)) == "relu"  # a named call keeps its name
