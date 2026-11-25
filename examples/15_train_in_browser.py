"""Train a network from its HTML page: press ▶ Train and watch inputs, intermediates and outputs change.

    python examples/15_train_in_browser.py

deepvis wraps the model and serves an interactive page of it — the input images, every intermediate tensor
(convolution feature maps, activations, pooled features), the class probabilities — plus Train / Pause /
Step / Reset buttons. The buttons drive the training loop defined below, which runs in this Python process;
the page refreshes as the weights change:

* Graph: each layer drawn from its current numbers (activation curves, kernels, weight fans, output bars)
* Volumes: the intermediate tensors of the selected sample as 3-D blocks
* Monitor: loss / accuracy curves, how fast each layer learns, and the gradient flow of the probe batch
* side panel: click any layer for its activations, weights and gradients; ‹ › switches between probe samples;
  draw on the input to see what the network predicts for your own stroke
"""

import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import deepvis as dv
from _data import strokes

torch.manual_seed(0)
X, y = strokes(1200)
X, y = torch.tensor(X).unsqueeze(1), torch.tensor(y)
X_train, y_train = X[:1000], y[:1000]
probe_x, probe_y = X[1000:1008], y[1000:1008]  # the 8 images the page shows (switch with ‹ ›)
CLASSES = ["vertical bar", "horizontal bar", "ring"]

model = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),   # 8 × 14 × 14 feature maps
    nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),                     # 16 × 7 × 7
    nn.Flatten(), nn.Linear(16 * 7 * 7, 32), nn.Tanh(), nn.Linear(32, 3),
)
opt = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
rng = np.random.default_rng(0)


def train_step(step: int) -> dict:
    """One optimization step on a mini-batch; the page calls this while ▶ Train is on."""
    model.train()
    idx = torch.as_tensor(rng.choice(len(X_train), 64, replace=False))
    logits = model(X_train[idx])
    loss = F.cross_entropy(logits, y_train[idx])
    opt.zero_grad()
    loss.backward()
    opt.step()
    acc = (logits.argmax(1) == y_train[idx]).float().mean()
    return {"loss": loss.item(), "accuracy": acc.item()}


def train_reset() -> None:
    """Reset restores the initial weights; start the optimizer fresh too."""
    global opt
    opt = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9)


net = dv.wrap(model.eval(), sample_input=probe_x)
net.labels = CLASSES
srv = net.serve(
    probe_x,
    train_step=train_step, train_steps=600, train_reset=train_reset,
    train_delay=0.02,           # slow enough to watch; 0 trains as fast as possible
    min_interval=0.4,           # refresh the page at most every 0.4 s
    snapshot_every=10,          # per-layer learning dynamics every 10 steps
    grads=True, loss_fn=lambda out: F.cross_entropy(out, probe_y),   # gradient flow of the probe batch
    title="Strokes CNN — train me", open_browser="--no-browser" not in sys.argv,
    note="Press ▶ Train. The dashboard shows the 8 probe images (‹ › to switch); draw your own stroke in the "
         "input panel. Monitor: loss, accuracy, how fast each layer learns, gradient flow.",
)
print("open", srv.url)
if "--no-browser" in sys.argv:  # headless check: train from Python through the same buttons' code path
    srv.train("start")
    while srv.trainer_state()["running"]:
        import time

        time.sleep(0.2)
    print(srv.training)
    srv.stop()
else:
    input("press Enter to stop the server ")
    srv.stop()
