"""Equinox: modules are pytrees; deepvis returns an edited copy with export()."""

import jax
import jax.numpy as jnp
import equinox as eqx

import deepvis as dv

mlp = eqx.nn.MLP(in_size=4, out_size=2, width_size=8, depth=2, activation=jax.nn.tanh, key=jax.random.PRNGKey(0))
x = jnp.ones(4)                                     # equinox models usually take one example
net = dv.wrap(mlp, sample_input=x)
net.summary()
print(net.equation("expanded"))
print(net.equation("scalar", neuron=1, top_k=3, input=x))
net.edit("layers.1.weight", "clamp", min=-0.2, max=0.2)
edited = net.export()
print("max |w| after clamp:", float(jnp.abs(edited.layers[1].weight).max()))
