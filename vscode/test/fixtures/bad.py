import torch
import torch.nn as nn
import deepvis as dv

mlp = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2))
x = torch.randn(3, 5)
dv.visualize(mlp, x)
