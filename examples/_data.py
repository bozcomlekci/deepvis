"""Tiny synthetic datasets so every example runs offline in seconds."""

import numpy as np


def strokes(n: int = 600, side: int = 28, seed: int = 0):
    """28×28 images of three classes: vertical bar, horizontal bar, ring (+ noise)."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, side, side), dtype=np.float32)
    y = rng.integers(0, 3, n)
    yy, xx = np.mgrid[:side, :side]
    for i, c in enumerate(y):
        cx, cy = rng.integers(9, side - 9, 2)
        if c == 0:
            X[i, 5:side - 5, cx - 1:cx + 2] = 1
        elif c == 1:
            X[i, cy - 1:cy + 2, 5:side - 5] = 1
        else:
            r = np.hypot(yy - side / 2, xx - side / 2)
            X[i][(r > 6) & (r < 9)] = 1
    X += rng.normal(0, 0.15, X.shape).astype(np.float32)
    return X.clip(0, 1), y


def spirals(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.3, 3.5 * np.pi, n)
    c = rng.integers(0, 2, n)
    r = t / (3.5 * np.pi)
    X = np.stack([r * np.cos(t + np.pi * c), r * np.sin(t + np.pi * c)], 1) + rng.normal(0, 0.03, (n, 2))
    return X.astype(np.float32), c
