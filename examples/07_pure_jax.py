"""Plain JAX functions: layers are inferred from x @ W + b, or named with dv.scope."""

import jax
import jax.numpy as jnp

import deepvis as dv


def mlp(params, x):
    h = jnp.tanh(x @ params["W1"] + params["b1"])
    with dv.scope("readout"):                        # a named region becomes one node
        logits = h @ params["W2"] + params["b2"]
        return jax.nn.log_softmax(logits)


k1, k2 = jax.random.split(jax.random.PRNGKey(0))
params = {"W1": jax.random.normal(k1, (3, 5)), "b1": jnp.zeros(5), "W2": jax.random.normal(k2, (5, 2)),
          "b2": jnp.zeros(2)}
x = jnp.ones((2, 3))
net = dv.wrap(mlp, params, sample_input=x)
net.summary()
print(net.equation())
tr = net.trace(x)
print(tr.table())
net["W1"][:, 0] = 0.0                              # same slice-assignment API as torch
print("edited params:", net.export()["W1"][:, 0])
