"""Diffusion models: record the denoising process and show it next to the network.

>>> from deepvis import diffusion as dd
>>> run = dd.sample(unet, shape=(4, 1, 16, 16), betas=betas)          # DDPM (or sampler="ddim", steps=50)
>>> net = dv.wrap(unet, sample_input=run.probe())                       # the U-Net at 8 timesteps of sample 0
>>> net.export_html("diffusion.html", *run.probe(), diffusion=run)     # adds the Diffusion tab

The run keeps, at up to ``frames`` evenly spaced steps of the reverse process, the noisy image xₜ, the
predicted noise ε̂(xₜ, t) and the predicted clean image x̂₀ = (xₜ − √(1−ᾱₜ) ε̂) / √ᾱₜ, plus the forward process
q(xₜ | x₀) of the final samples. ``run.probe(k)`` stacks one sample at k timesteps as a batch, so the dashboard's
sample switcher steps through time and every layer shows how the network's work changes as noise decreases.

The denoiser is any ``model(x, t) -> ε̂`` (``call=`` adapts other signatures, e.g. diffusers' ``.sample``).
Torch only for now.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


def linear_betas(T: int = 1000, beta_1: float = 1e-4, beta_T: float = 0.02) -> np.ndarray:
    return np.linspace(beta_1, beta_T, T, dtype=np.float64)


def cosine_betas(T: int = 1000, s: float = 0.008) -> np.ndarray:
    """Nichol & Dhariwal (2021)."""
    t = np.linspace(0, T, T + 1) / T
    f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    return np.clip(1 - f[1:] / f[:-1], 0, 0.999)


@dataclass
class DiffusionRun:
    ts: np.ndarray                    # (frames,) timestep of each recorded frame, T-1 → 0
    x: np.ndarray                     # (frames, B, C, H, W) noisy image before the step at t
    eps: np.ndarray                   # (frames, B, C, H, W) predicted noise
    x0: np.ndarray                    # (frames, B, C, H, W) predicted clean image
    final: np.ndarray                 # (B, C, H, W) the generated samples
    alphas_bar: np.ndarray            # (T,) ᾱ_t
    sampler: str = "ddpm"
    forward_ts: np.ndarray | None = None
    forward_x: np.ndarray | None = None   # (k, C, H, W) q(x_t | x_0) of sample 0
    meta: dict = field(default_factory=dict)
    _probe: tuple | None = None

    # ------------------------------------------------------------------ inputs for the dashboard
    def probe(self, k: int = 8, sample: int = 0) -> tuple:
        """(x_t, t) of ``sample`` at ``k`` recorded timesteps, as one batch (noisiest first)."""
        import torch

        if self._probe is not None and self._probe[0].shape[0] == min(k, len(self.ts)):
            return self._probe
        idx = np.unique(np.linspace(0, len(self.ts) - 1, min(k, len(self.ts))).round().astype(int))
        x = torch.as_tensor(self.x[idx, sample], dtype=torch.float32)
        t = torch.as_tensor(self.ts[idx], dtype=torch.long)
        self._probe = (x, t)
        self.meta["probe_ts"] = [int(v) for v in self.ts[idx]]
        return self._probe

    # ------------------------------------------------------------------ payload
    def to_dict(self, max_frames: int = 60, side: int = 96) -> dict:
        from .viz.glyphs import png_uri

        n = len(self.ts)
        idx = np.unique(np.linspace(0, n - 1, min(max_frames, n)).round().astype(int))
        B = self.x.shape[1]

        def img(a) -> str:  # (C, H, W) → PNG; channels 1 or 3; range from the data's own scale
            a = np.asarray(a, dtype=np.float64)
            a = a[0] if a.shape[0] == 1 else a[:3].transpose(1, 2, 0)
            lo, hi = -1.0, 1.0
            if a.min() < -1.5 or a.max() > 1.5:
                lo, hi = float(np.percentile(a, 1)), float(np.percentile(a, 99))
            u = np.clip((a - lo) / ((hi - lo) or 1), 0, 1)
            u = (u * 255).astype(np.uint8)
            if u.ndim == 2:
                u = np.repeat(u[..., None], 3, -1)
            f = max(1, side // max(u.shape[:2]))  # upscale small images (crisp pixels in every browser)
            return png_uri(np.kron(u, np.ones((f, f, 1), dtype=np.uint8)))

        frames = [{"t": int(self.ts[i]),
                   "x": [img(self.x[i, b]) for b in range(B)],
                   "x0": [img(self.x0[i, b]) for b in range(B)],
                   "eps": [img(self.eps[i, b]) for b in range(B)]} for i in idx]
        T = len(self.alphas_bar)
        ab = self.alphas_bar[np.unique(np.linspace(0, T - 1, min(T, 200)).round().astype(int))]
        out = {"T": T, "sampler": self.sampler, "frames": frames, "final": [img(self.final[b]) for b in range(B)],
               "alphasBar": [float(v) for v in ab], "probeTs": self.meta.get("probe_ts")}
        if self.forward_x is not None:
            out["forward"] = [{"t": int(t), "x": img(x)} for t, x in zip(self.forward_ts, self.forward_x)]
        return out

    def __repr__(self) -> str:
        return (f"DiffusionRun({self.sampler}, {len(self.alphas_bar)} steps, {len(self.ts)} frames, "
                f"samples {tuple(self.final.shape)})")


def sample(model, shape, betas, steps: int | None = None, sampler: str = "ddpm", frames: int = 60, seed: int = 0,
           call: Callable | None = None, device=None, clip_x0: float | None = 1.0) -> DiffusionRun:
    """Run the reverse process and record it (see the module docstring).

    ``betas`` the noise schedule (length T); ``sampler`` "ddpm" (all T steps, stochastic) or "ddim" (``steps``
    deterministic steps); ``clip_x0`` clips x̂₀ to ±clip (None: no clipping); ``call(model, x, t) -> ε̂``.
    """
    import torch

    betas = np.asarray(betas, dtype=np.float64)
    T = len(betas)
    alphas = 1 - betas
    ab = np.cumprod(alphas)
    call = call or (lambda m, x, t: m(x, t))
    dev = device or next((p.device for p in model.parameters()), torch.device("cpu"))
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(shape, generator=g).to(dev)
    if sampler == "ddpm":
        seq = list(range(T - 1, -1, -1))
    elif sampler == "ddim":
        n = steps or 50
        seq = sorted({int(round(v)) for v in np.linspace(0, T - 1, n)}, reverse=True)
    else:
        from .errors import OptionError, suggest

        hint = suggest(str(sampler), ["ddpm", "ddim"])
        raise OptionError(f"unknown sampler {sampler!r}.", (hint + " " if hint else "") + "Use 'ddpm' or 'ddim'.")
    keep = set(np.unique(np.linspace(0, len(seq) - 1, min(frames, len(seq))).round().astype(int)).tolist())
    rec_t, rec_x, rec_e, rec_0 = [], [], [], []
    modes = [(m, m.training) for m in model.modules()] if hasattr(model, "modules") else []  # each layer's own
    if hasattr(model, "eval"):
        model.eval()
    try:
        with torch.no_grad():
            for i, t in enumerate(seq):
                tt = torch.full((shape[0],), t, dtype=torch.long, device=dev)
                try:
                    eps = call(model, x, tt)
                except Exception as exc:  # noqa: BLE001
                    from .errors import RunError

                    first = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
                    raise RunError(f"the denoiser failed on x of shape {tuple(shape)}, t = {t} ({type(exc).__name__}: "
                                   f"{first[:200]}).",
                                   "it is called as model(x, t) with t a (batch,) LongTensor and must return the "
                                   "predicted noise, shaped like x. Check `shape` matches what the model was trained "
                                   "on; adapt other signatures with call=, e.g. call=lambda m, x, t: m(x, t).sample "
                                   "(diffusers).") from exc
                if not hasattr(eps, "shape") or tuple(eps.shape) != tuple(x.shape):
                    from .errors import RunError

                    raise RunError(f"the denoiser returned {type(eps).__name__} "
                                   f"{tuple(getattr(eps, 'shape', ()))}, but the noise must be shaped like x "
                                   f"{tuple(x.shape)}.",
                                   "return the predicted noise ε̂; for diffusers use call=lambda m, x, t: m(x, t).sample.")
                a_t = float(ab[t])
                x0 = (x - np.sqrt(1 - a_t) * eps) / np.sqrt(a_t)
                if clip_x0 is not None:
                    x0 = x0.clamp(-clip_x0, clip_x0)
                if i in keep:
                    rec_t.append(t)
                    rec_x.append(x.cpu().numpy())
                    rec_e.append(eps.cpu().numpy())
                    rec_0.append(x0.cpu().numpy())
                t_prev = seq[i + 1] if i + 1 < len(seq) else -1
                ab_prev = float(ab[t_prev]) if t_prev >= 0 else 1.0
                if sampler == "ddpm":
                    # posterior q(x_{t-1} | x_t, x̂0) mean and variance (Ho et al., eq. 6-7)
                    beta_t = float(betas[t])
                    mean = (np.sqrt(ab_prev) * beta_t / (1 - a_t)) * x0 + \
                           (np.sqrt(alphas[t]) * (1 - ab_prev) / (1 - a_t)) * x
                    if t > 0:
                        var = beta_t * (1 - ab_prev) / (1 - a_t)
                        x = mean + np.sqrt(var) * torch.randn(shape, generator=g).to(dev)
                    else:
                        x = mean
                else:  # DDIM, η = 0
                    eps_eff = (x - np.sqrt(a_t) * x0) / np.sqrt(1 - a_t)
                    x = np.sqrt(ab_prev) * x0 + np.sqrt(1 - ab_prev) * eps_eff
    finally:
        for m, was_training in modes:
            m.training = was_training
    final = x.cpu().numpy()
    run = DiffusionRun(np.asarray(rec_t), np.stack(rec_x), np.stack(rec_e), np.stack(rec_0), final, ab, sampler)
    # the forward process on the generated sample 0: q(x_t | x_0) = √ᾱ x0 + √(1−ᾱ) ε
    fts = np.unique(np.linspace(0, T - 1, 8).round().astype(int))
    noise = np.random.default_rng(seed).standard_normal((len(fts),) + tuple(final.shape[1:]))
    run.forward_ts = fts
    run.forward_x = np.stack([np.sqrt(ab[t]) * final[0] + np.sqrt(1 - ab[t]) * noise[j] for j, t in enumerate(fts)])
    return run


def q_sample(x0, t: int, betas, noise=None):
    """x_t ~ q(x_t | x_0) for training loops: √ᾱ_t x0 + √(1−ᾱ_t) ε (returns x_t, ε)."""
    import torch

    ab = float(np.cumprod(1 - np.asarray(betas, dtype=np.float64))[t]) if np.isscalar(t) else None
    noise = torch.randn_like(x0) if noise is None else noise
    if ab is not None:
        return np.sqrt(ab) * x0 + np.sqrt(1 - ab) * noise, noise
    abs_ = torch.as_tensor(np.cumprod(1 - np.asarray(betas, dtype=np.float64)), dtype=x0.dtype, device=x0.device)[t]
    shape = (-1,) + (1,) * (x0.dim() - 1)
    return abs_.sqrt().view(shape) * x0 + (1 - abs_).sqrt().view(shape) * noise, noise


__all__ = ["sample", "DiffusionRun", "linear_betas", "cosine_betas", "q_sample"]
