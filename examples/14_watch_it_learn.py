"""Watch a network learn: a live dashboard that follows your training loop, plus a static replay.

    python examples/14_watch_it_learn.py            # opens the live page, trains, then writes out/14_run.html
    python examples/14_watch_it_learn.py --jax      # the same with Flax (JAX weights are immutable: pass params=)

The page polls the Python process: loss curves, a layers × time heat strip of how fast every layer moves
(relative update per step), gradient and weight norms, and every other view (activation curves, weights,
3-D volumes) refreshes in place while you look at it. Pause it any time from the header.
"""

import os
import sys
import time

import numpy as np

import deepvis as dv
from _data import spirals

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
X, Y = spirals(600)
headless = "--headless" in sys.argv  # CI: no browser, no waiting


def torch_run():
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    x, y = torch.tensor(X, dtype=torch.float32), torch.tensor(Y).long()
    model = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 3))
    net = dv.wrap(model, sample_input=x[:32])
    srv = net.serve(x[:32], open_browser=not headless, start="monitor", min_interval=0.5, snapshot_every=25,
                    title="Spirals — training live")
    print("live dashboard:", srv.url)
    opt = torch.optim.Adam(model.parameters(), 3e-3)
    for step in range(1500):
        opt.zero_grad()
        out = model(x)
        loss = nn.functional.cross_entropy(out, y)
        loss.backward()
        opt.step()
        srv.update(step, loss=loss.item(), accuracy=(out.argmax(1) == y).float().mean().item())
        if not headless:
            time.sleep(0.01)  # slow enough to watch
    srv.update(step, render=True)  # final state
    # the same log, as a self-contained page anyone can open later
    print(net.export_html(f"{OUT}/14_run.html", x[:32], training=srv.training, start="monitor",
                          title="Spirals — training replay"))
    try:  # and as a video (needs manim)
        print(net.animate("training", log=srv.training, path=f"{OUT}/14_learning.mp4"))
    except ImportError:
        pass
    return srv


def jax_run():
    import jax
    import jax.numpy as jnp
    import flax.linen as fnn
    import optax

    class MLP(fnn.Module):
        @fnn.compact
        def __call__(self, x):
            x = jnp.tanh(fnn.Dense(32)(x))
            x = jnp.tanh(fnn.Dense(32)(x))
            return fnn.Dense(3)(x)

    x, y = jnp.asarray(X, dtype=jnp.float32), jnp.asarray(Y)
    model = MLP()
    params = model.init(jax.random.PRNGKey(0), x[:1])["params"]
    net = dv.wrap(model, {"params": params}, sample_input=x[:32])
    srv = net.serve(x[:32], open_browser=not headless, start="monitor", min_interval=0.5, snapshot_every=25)
    print("live dashboard:", srv.url)
    tx = optax.adam(3e-3)
    state = tx.init(params)

    @jax.jit
    def step_fn(params, state):
        def loss_fn(p):
            logits = model.apply({"params": p}, x)
            return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean(), logits
        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, state = tx.update(grads, state)
        return optax.apply_updates(params, updates), state, loss, (logits.argmax(1) == y).mean(), grads

    for step in range(1500):
        params, state, loss, acc, grads = step_fn(params, state)
        # JAX weights are immutable: hand the new ones (and optionally the grads) to the dashboard
        srv.update(step, params=params, grads=grads, loss=float(loss), accuracy=float(acc))
        if not headless:
            time.sleep(0.01)
    srv.update(step, render=True)
    return srv


if __name__ == "__main__":
    srv = jax_run() if "--jax" in sys.argv else torch_run()
    print(srv.training)
    if not headless:
        input("training done — press Enter to stop the server ")
    srv.stop()
