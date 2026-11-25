"""Train a Flax network from its HTML page, the JAX way: ▶ Train calls a jitted optax step, the page follows.

    python examples/17_jax_train_in_browser.py

JAX arrays are immutable, so the step reads the current weights from deepvis (``net.export()`` — which also picks
up edits you make on the page, and Reset) and hands the new ones back as ``"params"``; everything else is the same
page as the PyTorch example (15): Graph, Volumes, Monitor with the loss curves and the replay of how every layer's
weights and the outputs changed.
"""

import sys

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

import deepvis as dv
from _data import strokes

X, y = strokes(1200)
X = jnp.asarray(X)[..., None]            # (N, 28, 28, 1): flax convolutions are channels-last
y = jnp.asarray(y)
X_train, y_train = X[:1000], y[:1000]
probe_x = X[1000:1008]                   # the 8 images the page shows (switch with ‹ ›)
CLASSES = ["vertical bar", "horizontal bar", "ring"]


class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.max_pool(nn.relu(nn.Conv(8, (3, 3))(x)), (2, 2), (2, 2))
        x = nn.max_pool(nn.relu(nn.Conv(16, (3, 3))(x)), (2, 2), (2, 2))
        x = x.reshape(x.shape[0], -1)
        return nn.Dense(3)(nn.tanh(nn.Dense(32)(x)))


model = CNN()
variables = model.init(jax.random.PRNGKey(0), probe_x)
tx = optax.adam(3e-3)
opt_state = tx.init(variables["params"])
rng = np.random.default_rng(0)


@jax.jit
def update(params, opt_state, xb, yb):
    def loss_fn(p):
        logits = model.apply({"params": p}, xb)
        return optax.softmax_cross_entropy_with_integer_labels(logits, yb).mean(), logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = tx.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss, (logits.argmax(-1) == yb).mean()


def train_step(step: int) -> dict:
    """One optax step on a mini-batch, from the weights deepvis holds now (so page edits and Reset count)."""
    global opt_state
    _, current = net.export()
    idx = rng.choice(len(X_train), 64, replace=False)
    params, opt_state, loss, acc = update(current["params"], opt_state, X_train[idx], y_train[idx])
    return {"loss": float(loss), "accuracy": float(acc), "params": {**current, "params": params}}


def train_reset() -> None:
    global opt_state
    opt_state = tx.init(net.export()[1]["params"])


net = dv.wrap(model, variables, sample_input=probe_x)
net.labels = CLASSES
srv = net.serve(probe_x, train_step=train_step, train_steps=500, train_reset=train_reset, train_delay=0.02,
                min_interval=0.4, snapshot_every=10, title="Train a Flax CNN")
print(f"open {srv.url} and press ▶ Train")
if "--no-block" not in sys.argv:
    try:
        input("press Enter to stop\n")
    except (EOFError, KeyboardInterrupt):
        pass
    srv.stop()
