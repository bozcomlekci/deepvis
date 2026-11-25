"""A CNN: filters, feature maps, and structured (channel) pruning."""

import os

import torch
import torch.nn as nn

import deepvis as dv

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)

model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 10),
).eval()
x = torch.randn(2, 3, 32, 32)
net = dv.wrap(model, sample_input=x)
net.summary()
print(net.equation())

conv = net["0"]                                     # focus on the first conv layer
print(conv, conv.weight)
print(conv.equation())
before = net(x).detach()
conv.prune_neurons(0.5)                             # zero the 8 weakest filters (by L2 norm)
print("zeroed filters:", int((abs(conv.weight_matrix().reshape(16, -1)).sum(1) == 0).sum()))
print("output change:", (net(x) - before).norm().item())
net.diff()
print("saved", net.save_html(os.path.join(OUT, "02_cnn.html"), x))
