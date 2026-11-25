"""Monitor intermediate layers while training (dead ReLUs, activation drift, gradient norms)."""

import os

import torch
import torch.nn as nn

import deepvis as dv
from _data import spirals

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)
X, y = spirals(400)
X, y = torch.tensor(X), torch.tensor(y)
model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))
net = dv.wrap(model, sample_input=X[:8])
mon = net.monitor("1", "3", stats=("mean", "std", "sparsity"), grads=True)   # the two ReLUs

opt = torch.optim.SGD(model.parameters(), lr=0.5)
for step in range(150):
    loss = nn.functional.cross_entropy(model(X), y)                           # call the model as usual
    opt.zero_grad(); loss.backward(); opt.step()
    mon.step()
print(f"final loss {loss.item():.3f}")
print(mon.table())
print("dead-ReLU fraction, first → last:", mon["3"].series("sparsity")[[0, -1]])
print("saved", net.save_html(os.path.join(OUT, "09_monitor.html"), X[:8], monitor=mon))
mon.stop()
