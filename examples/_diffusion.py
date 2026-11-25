"""A small DDPM denoiser (a U-Net) and a noise schedule — everything a diffusion example needs, offline.

The U-Net has the parts real diffusion models have: sinusoidal timestep embeddings fed into every residual
block, GroupNorm + SiLU, a self-attention block at the bottleneck, down/upsampling and skip connections.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Transformer-style sinusoidal features of the diffusion step t (DDPM, Ho et al. 2020)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, tdim: int, groups: int = 8):
        super().__init__()
        self.norm1, self.conv1 = nn.GroupNorm(groups, cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.temb = nn.Linear(tdim, cout)
        self.norm2, self.conv2 = nn.GroupNorm(groups, cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(F.silu(temb))[:, :, None, None]  # the timestep steers every block
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    """Self-attention over the pixels of a feature map."""

    def __init__(self, ch: int, heads: int = 2):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        s = self.norm(x).flatten(2).transpose(1, 2)  # (b, h·w, c): one token per pixel
        a, _ = self.attn(s, s, s, need_weights=False)
        return x + a.transpose(1, 2).reshape(b, c, h, w)


class TinyUNet(nn.Module):
    def __init__(self, ch: int = 32, tdim: int = 64, in_ch: int = 1):
        super().__init__()
        self.tdim = tdim
        self.time_mlp = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.inp = nn.Conv2d(in_ch, ch, 3, padding=1)
        self.down1 = ResBlock(ch, ch, tdim)
        self.pool1 = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
        self.down2 = ResBlock(ch, 2 * ch, tdim)
        self.pool2 = nn.Conv2d(2 * ch, 2 * ch, 3, stride=2, padding=1)
        self.mid = ResBlock(2 * ch, 2 * ch, tdim)
        self.attn = AttnBlock(2 * ch)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2 = ResBlock(4 * ch, ch, tdim)
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1 = ResBlock(2 * ch, ch, tdim)
        self.out = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, in_ch, 3, padding=1))

    def forward(self, x, t):
        temb = self.time_mlp(timestep_embedding(t, self.tdim))
        h0 = self.inp(x)
        h1 = self.down1(h0, temb)
        h2 = self.down2(self.pool1(h1), temb)
        m = self.attn(self.mid(self.pool2(h2), temb))
        d2 = self.dec2(torch.cat([self.up2(m), h2], 1), temb)   # skip connections
        d1 = self.dec1(torch.cat([self.up1(d2), h1], 1), temb)
        return self.out(d1)  # predicted noise ε̂(x_t, t)


def linear_schedule(T: int = 200, beta_1: float = 1e-4, beta_T: float = 0.02):
    betas = torch.linspace(beta_1, beta_T, T)
    alphas = 1 - betas
    return betas, alphas, torch.cumprod(alphas, 0)
