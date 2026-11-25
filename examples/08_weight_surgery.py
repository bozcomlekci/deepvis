"""A tour of weight manipulation: every edit is recorded, undoable, and replayable."""

import json

import numpy as np
import torch
import torch.nn as nn

import deepvis as dv

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 4))
x = torch.randn(8, 10)
net = dv.wrap(model, sample_input=x)
y0 = net(x).detach()

net["0.weight"][:, 0] = 0                                   # numpy-style slices
net["0.weight"][4:8] *= 2                                   # in-place arithmetic on regions
net.edit("2.weight", "zero", where="abs(w) < 0.05")          # masks as safe expressions
net["2"].neurons([3, 7, 11]).zero()                         # knock out hidden units (rows + biases)
net.select(kind="Linear").noise(0.01, seed=0)               # every Linear layer at once
net["4"].low_rank(2)                                        # SVD truncation
net.edit("0.bias", lambda b: np.sign(b) * np.sqrt(np.abs(b)))  # any python function
print(net.history)
net.diff()

log = net.history.export()                                  # JSON, replayable on another copy
print(json.dumps(log[:2], indent=1))

with net.edits():                                           # temporary: reverted on exit
    net["2"].scale(0.0)
    print("inside: output norm", net(x).norm().item())
print("after : output norm", net(x).norm().item())

net.reset()                                                 # back to the original weights
assert torch.allclose(net(x), y0)
net.permute_neurons("0", torch.randperm(32).numpy())        # a symmetry: weights change, function doesn't
print("function preserved under neuron permutation:", torch.allclose(net(x), y0, atol=1e-6))

twin = net.clone()
twin.select(kind="Linear").reinit(seed=1)
net.interpolate(twin, alpha=0.5)                            # halfway between two networks
print("after interpolation, Δoutput:", (net(x) - y0).norm().item())
