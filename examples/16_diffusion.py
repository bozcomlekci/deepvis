"""A diffusion model, inside and out: train a small DDPM on bars and rings, sample, and open the page.

    python examples/16_diffusion.py          # ~1-2 min of CPU training the first time (weights are cached)

Tabs to look at:
* Diffusion — play the reverse process: noisy xₜ, the network's predicted clean image x̂₀ and predicted noise ε̂,
  the whole trajectory, the forward (noising) process, the schedule
* Graph — the U-Net: timestep embedding (sin / cos), residual blocks, GroupNorm, SiLU, self-attention,
  down/up sampling, skip connections — each drawn from its numbers; ‹ › steps the dashboard through timesteps
* Attention — inside the bottleneck self-attention: pixels as tokens, Q, Kᵀ, scores, softmax, P·V
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

import deepvis as dv
from _data import strokes
from _diffusion import TinyUNet
from deepvis import diffusion as dd

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
WEIGHTS = os.path.join(OUT, "tinyunet_strokes.pt")
torch.manual_seed(0)

T = 200
betas = dd.linear_betas(T, 1e-4, 0.02)
model = TinyUNet(ch=32, tdim=64)

if os.path.exists(WEIGHTS):
    model.load_state_dict(torch.load(WEIGHTS))
else:
    X, _ = strokes(3000, side=20, seed=1)
    X = torch.tensor(X).unsqueeze(1) * 2 - 1  # images in [-1, 1]
    opt = torch.optim.Adam(model.parameters(), 2e-3)
    rng = np.random.default_rng(0)
    steps = int(os.environ.get("DV_DIFFUSION_STEPS", 1500))
    for step in range(steps):
        x0 = X[torch.as_tensor(rng.choice(len(X), 64, replace=False))]
        t = torch.randint(0, T, (64,))
        xt, noise = dd.q_sample(x0, t, betas)
        loss = F.mse_loss(model(xt, t), noise)  # the DDPM objective: predict the noise
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 250 == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}")
    torch.save(model.state_dict(), WEIGHTS)

run = dd.sample(model, shape=(6, 1, 20, 20), betas=betas, sampler="ddpm", frames=60, seed=3)
print(run)
x_t, t = run.probe(k=8)                    # one sample at 8 timesteps: ‹ › in the page steps through them
net = dv.wrap(model.eval(), sample_input=(x_t, t))
path = net.export_html(f"{OUT}/16_diffusion.html", x_t, t, diffusion=run, start="diffusion",
                       title="Diffusion — a tiny DDPM")
print(path)
if "--open" in sys.argv:
    import webbrowser

    webbrowser.open("file://" + os.path.abspath(path))
