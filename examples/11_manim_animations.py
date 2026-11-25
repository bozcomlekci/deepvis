"""Manim videos: forward pass, equation unrolling, layer focus, a weight edit, an architecture tour, 3-D tensors.

Needs `pip install manim`. With `latex` + `dvisvgm` installed the math is typeset by TeX;
otherwise it is rendered in Computer Modern via Pango.
"""

import os

import torch
import torch.nn as nn

import deepvis as dv
from _data import strokes

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)
X, _ = strokes(2)
x = torch.tensor(X[:1]).unsqueeze(1)
model = nn.Sequential(nn.Flatten(), nn.Linear(784, 16), nn.Sigmoid(), nn.Linear(16, 16), nn.Sigmoid(),
                      nn.Linear(16, 10), nn.Softmax(-1)).eval()
net = dv.wrap(model, sample_input=x)

print(net.animate("forward", x, path=f"{OUT}/forward.mp4"))
print(net.animate("equation", path=f"{OUT}/equation.mp4"))
print(net.animate("focus", x, focus="3", path=f"{OUT}/focus.mp4"))
print(net.animate("edit", x, edit={"op": "prune", "target": "3", "args": {"amount": 0.8}}, path=f"{OUT}/edit.mp4"))
print(net.animate("architecture", path=f"{OUT}/architecture.mp4"))  # fmt="gif" also works (larger files)

cnn = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                    nn.Linear(32, 10)).eval()
img = torch.randn(1, 3, 32, 32)
print(dv.wrap(cnn, sample_input=img).animate("volumes", img, path=f"{OUT}/volumes.mp4"))  # a ThreeDScene
