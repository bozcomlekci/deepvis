"""Export expressive visualizations: interactive HTML pages and static SVG figures.

Every component is drawn as its own data-driven glyph: activations plot their curve over the inputs they
actually received, dense layers show a fan of their real weights, convolutions a 3-D slab with a real kernel,
normalizations the spread before/after, softmax its output distribution, attention its attention map —
and each edge carries the 3-D shape of the tensor flowing along it.
"""

import os

import torch
import torch.nn as nn

import deepvis as dv
from _data import strokes

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)

# a small classifier, trained for a few seconds so the glyphs show learned structure
X, y = strokes(600)
X, y = torch.tensor(X).unsqueeze(1), torch.tensor(y)
mlp = nn.Sequential(nn.Flatten(), nn.Linear(784, 16), nn.Sigmoid(), nn.Linear(16, 16), nn.GELU(), nn.Dropout(0.2),
                    nn.Linear(16, 3), nn.Softmax(dim=-1))
opt = torch.optim.Adam(mlp.parameters(), 1e-2)
for _ in range(60):
    loss = nn.functional.nll_loss(torch.log(mlp(X) + 1e-9), y)
    opt.zero_grad(); loss.backward(); opt.step()
mlp.eval()
ring = X[y == 2][:1]

# one call: model + input -> a self-contained interactive page
print(dv.export_html(mlp, ring, f"{OUT}/13_mlp.html", title="Stroke classifier",
                     note="784 → 16 → 16 → 3, trained on synthetic bars and rings. Click any layer."))
# the same glyphs as a static figure for papers / READMEs (GitHub renders SVG)
print(dv.export_svg(mlp, ring, f"{OUT}/13_mlp.svg", title="Stroke classifier"))

cnn = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                    nn.Linear(32, 10)).eval()
img = torch.randn(1, 3, 32, 32)
net = dv.wrap(cnn, sample_input=img)
# choose the tabs, the start view, the detail level; "compact" gives plain boxes instead of glyphs
print(net.export_html(f"{OUT}/13_cnn.html", img, views=("graph", "volumes", "equations"), start="graph"))
print(net.export_svg(f"{OUT}/13_cnn_lr.svg", img, direction="LR"))          # left-to-right figure
print(net.export_html(f"{OUT}/13_cnn_compact.html", img, style="compact", tensor_shapes=False))
