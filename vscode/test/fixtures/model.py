import torch
import torch.nn as nn
import deepvis as dv


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, 3, padding=1)
        self.head = nn.Linear(8 * 8 * 8, 10)

    def forward(self, x):
        return self.head(torch.relu(self.conv(x)).flatten(1))


net = Net()
x = torch.randn(2, 1, 8, 8)
dv.visualize(net, x, title="first")
dv.visualize(net, x,
             title="second")
raise SystemExit("the rest of the script must not run")
