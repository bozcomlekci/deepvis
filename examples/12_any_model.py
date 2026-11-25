"""deepvis is a general-purpose wrapper: throw any architecture at it.

ResNet-18 and a Vision Transformer from torchvision, an LSTM, a U-Net with skip concatenations and a
model with two inputs and a dict of outputs — same API, same dashboard, same editing tools.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import deepvis as dv

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)


class LSTMTagger(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb, self.lstm, self.head = nn.Embedding(100, 16), nn.LSTM(16, 32, batch_first=True), nn.Linear(32, 5)

    def forward(self, ids):
        out, _ = self.lstm(self.emb(ids))
        return self.head(out)


class UNetTiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU())
        self.mid = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(8, 16, 3, padding=1), nn.ReLU())
        self.up, self.out = nn.ConvTranspose2d(16, 8, 2, stride=2), nn.Conv2d(16, 2, 1)

    def forward(self, x):
        a = self.down(x)
        return self.out(torch.cat([a, self.up(self.mid(a))], dim=1))


class TwoInputs(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.meta, self.head, self.aux = nn.Linear(4, 8), nn.Linear(16, 3), nn.Linear(16, 1)

    def forward(self, image, metadata):
        z = torch.cat([self.cnn(image), F.relu(self.meta(metadata))], dim=1)
        return {"class": self.head(z), "score": torch.sigmoid(self.aux(z))}


models = {
    "resnet18": (torchvision.models.resnet18(num_classes=10), (torch.randn(1, 3, 64, 64),)),
    "vit": (torchvision.models.VisionTransformer(image_size=32, patch_size=8, num_layers=2, num_heads=4,
                                                 hidden_dim=64, mlp_dim=128, num_classes=10), (torch.randn(1, 3, 32, 32),)),
    "lstm": (LSTMTagger(), (torch.randint(0, 100, (2, 12)),)),
    "unet": (UNetTiny(), (torch.randn(1, 1, 32, 32),)),
    "two_inputs": (TwoInputs(), (torch.randn(2, 3, 16, 16), torch.randn(2, 4))),
}
for name, (model, args) in models.items():
    net = dv.wrap(model.eval(), sample_input=args)
    kinds = sorted({n.kind for n in net.graph.compute_nodes() if n.op in ("module", "function")})
    print(f"\n{name}: {len(net.graph.compute_nodes())} nodes — {', '.join(kinds[:10])}{' …' if len(kinds) > 10 else ''}")
    print("  ", net.equation(depth=1).lines[-1].rhs.text()[:120])
    print("   saved", net.save_html(os.path.join(OUT, f"12_{name}.html"), *args))
