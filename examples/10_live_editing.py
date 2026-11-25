"""Live dashboard: edit weights in the browser, backed by this Python process.

Run it and open the printed URL. Then play:
  * draw on the input canvas (right-click erases) and watch the output respond,
  * click a layer: scale / prune / add noise / low-rank / quantize its weights, click a weight cell to set it,
  * intervene on any node's activations (ablate, amplify, clamp, noise) and remove interventions again,
  * rotate the Volumes tab to see every tensor as a 3-D block.
Press Ctrl+C to stop. Every browser edit lands in net.history (undo from Python too).
"""

import sys
import time

import torch
import torch.nn as nn

import deepvis as dv
from _data import strokes

torch.manual_seed(0)
X, y = strokes(600)
X, y = torch.tensor(X).unsqueeze(1), torch.tensor(y)
model = nn.Sequential(nn.Flatten(), nn.Linear(784, 16), nn.Sigmoid(), nn.Linear(16, 16), nn.Sigmoid(),
                      nn.Linear(16, 3), nn.Softmax(-1))
opt = torch.optim.Adam(model.parameters(), 1e-2)
for _ in range(60):  # a quick fit: vertical bar / horizontal bar / ring
    loss = nn.functional.nll_loss(torch.log(model(X) + 1e-9), y)
    opt.zero_grad(); loss.backward(); opt.step()
net = dv.wrap(model.eval(), sample_input=torch.zeros(1, 1, 28, 28))   # start from a blank canvas
server = net.serve(open_browser="--no-browser" not in sys.argv)
print("live dashboard at", server.url)
try:
    for _ in range(int(sys.argv[sys.argv.index("--seconds") + 1]) if "--seconds" in sys.argv else 10 ** 9):
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    server.stop()
    print(net.history)
