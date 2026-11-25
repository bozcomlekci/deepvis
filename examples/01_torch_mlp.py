"""Train a small MLP on synthetic strokes, then look inside it the 3Blue1Brown way."""

import os

import torch
import torch.nn as nn

import deepvis as dv
from _data import strokes

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)

X, y = strokes(600)
X, y = torch.tensor(X).unsqueeze(1), torch.tensor(y)
model = nn.Sequential(nn.Flatten(), nn.Linear(784, 16), nn.Sigmoid(), nn.Linear(16, 16), nn.Sigmoid(),
                      nn.Linear(16, 3), nn.Softmax(dim=-1))
opt = torch.optim.Adam(model.parameters(), 1e-2)
for epoch in range(60):
    loss = nn.functional.nll_loss(torch.log(model(X) + 1e-9), y)
    opt.zero_grad(); loss.backward(); opt.step()
print(f"train loss {loss.item():.3f}, accuracy {(model(X).argmax(1) == y).float().mean():.1%}")

net = dv.wrap(model.eval(), sample_input=X[:1])
net.summary()
print(net.equation())                               # layer by layer
print(net.equation("expanded"))                     # one formula
print(net.equation("scalar", node="3", neuron=0, top_k=4, input=X[:1]))  # one neuron, real numbers

tr = net.trace(X[:4])
print(tr.table())
print("saved", net.save_html(os.path.join(OUT, "01_mlp.html"), X[:1]))  # net.show(X[:1]) opens it directly
