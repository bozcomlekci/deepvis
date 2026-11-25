"""flax.linen: same API — graph, math, edits, interventions, gradients."""

import os

import jax
import jax.numpy as jnp
import flax.linen as nn

import deepvis as dv

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


class Block(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x + nn.gelu(nn.LayerNorm(name="ln")(nn.Dense(32, name="fc")(x)))


class Model(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Conv(8, (3, 3), name="conv")(x))
        x = nn.avg_pool(x, (2, 2), strides=(2, 2)).reshape((x.shape[0], -1))
        x = nn.relu(nn.Dense(32, name="proj")(x))
        for i in range(2):
            x = Block(name=f"block{i}")(x)
        return nn.softmax(nn.Dense(4, name="head")(x))


x = jax.random.normal(jax.random.PRNGKey(0), (2, 16, 16, 3))
model = Model()
variables = model.init(jax.random.PRNGKey(1), x)
net = dv.wrap(model, variables, sample_input=x)
net.summary()
print(net.equation())

# activation + parameter gradients via jax.grad, for the log-probability of class 0
tr = net.trace(x, grads=True, loss_fn=lambda y: jnp.log(y[:, 0]).sum())
print("‖∂L/∂block0.fc‖ =", float(jnp.linalg.norm(tr.grads["block0.fc"])))
net["block1.fc"].low_rank(4)                       # arbitrary surgery; export() gives the new variables
module, new_vars = net.export()
print("rank of edited kernel:", int(jnp.linalg.matrix_rank(new_vars["params"]["block1"]["fc"]["kernel"])))
with net.intervene("@GELU_1", dv.clamp(0, 0.5)):   # functional-op nodes can be intervened on in JAX
    print("clamped output:", net(x)[0])
print("saved", net.save_html(os.path.join(OUT, "04_flax.html"), x))
