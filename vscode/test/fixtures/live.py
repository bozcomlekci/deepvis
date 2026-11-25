import torch
import deepvis as dv

model = torch.nn.Linear(4, 2)
srv = dv.visualize(model, torch.randn(3, 4), live=True, title="live fixture")
for s in range(5):
    srv.update(s, loss=1 / (s + 1))
