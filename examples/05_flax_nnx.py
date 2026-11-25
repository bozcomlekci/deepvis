"""flax.nnx: edits made through deepvis update your live nnx module in place."""

import jax.numpy as jnp
from flax import nnx

import deepvis as dv


class MLP(nnx.Module):
    def __init__(self, rngs):
        self.fc1 = nnx.Linear(8, 16, rngs=rngs)
        self.drop = nnx.Dropout(0.2, rngs=rngs)
        self.fc2 = nnx.Linear(16, 3, rngs=rngs)

    def __call__(self, x):
        return self.fc2(self.drop(nnx.relu(self.fc1(x))))


model = MLP(nnx.Rngs(0))
x = jnp.ones((4, 8))
net = dv.wrap(model, sample_input=x)
net.summary()
print(net.equation("expanded"))
net["fc1"].neurons([0, 1, 2]).zero()
print("model.fc1.kernel[:, :3] is now", model.fc1.kernel[...][:, :3].sum(), "(edited in place)")
with net.monitor("fc1", "@ReLU") as mon:
    for _ in range(5):
        net(x)
print(mon.table())
