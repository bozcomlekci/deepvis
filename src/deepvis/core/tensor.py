"""Array helpers: conversion to numpy, summary statistics, display downsampling."""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

DEFAULT_STATS = ("mean", "std", "min", "max", "norm", "sparsity")


def to_numpy(x: Any) -> np.ndarray:
    """Convert torch / jax / numpy / python values to a float-friendly numpy array."""
    if isinstance(x, np.ndarray):
        return x
    mod = type(x).__module__
    if mod.startswith("torch"):
        t = x.detach()
        if t.is_complex():
            return t.cpu().numpy()
        if t.dtype.is_floating_point and t.dtype not in (__import__("torch").float32, __import__("torch").float64):
            t = t.float()
        return t.cpu().numpy()
    try:
        arr = np.asarray(x)
    except Exception:  # noqa: BLE001
        return np.asarray(float("nan"))
    if arr.dtype.kind == "V" or str(arr.dtype) == "bfloat16":
        arr = arr.astype(np.float32)
    return arr


def is_array(x: Any) -> bool:
    return hasattr(x, "shape") and hasattr(x, "dtype") and not isinstance(x, type)


def first_array(x: Any):
    """Primary array of a module output (first array leaf of tuples/lists/dicts)."""
    if is_array(x):
        return x
    if isinstance(x, (list, tuple)):
        for v in x:
            a = first_array(v)
            if a is not None:
                return a
    if isinstance(x, dict):
        for v in x.values():
            a = first_array(v)
            if a is not None:
                return a
    return None


def replace_first_array(x: Any, new: Any):
    """Rebuild a (nested) output structure with its primary array replaced."""
    if is_array(x):
        return new
    if isinstance(x, (list, tuple)):
        items, done = [], False
        for v in x:
            if not done and first_array(v) is not None:
                items.append(replace_first_array(v, new))
                done = True
            else:
                items.append(v)
        return type(x)(items) if not hasattr(x, "_fields") else type(x)(*items)
    if isinstance(x, dict):
        out = copy.copy(x)  # keeps the mapping's type: a HuggingFace ModelOutput must stay one (callers use .logits)
        for k, v in x.items():
            if first_array(v) is not None:
                out[k] = replace_first_array(v, new)
                break
        return out
    return x


def stats(x: Any, which=DEFAULT_STATS, finite: np.ndarray | None = None) -> dict[str, float]:
    """Summary numbers over the finite values of ``x``. ``finite``: those values as a float64 vector, when the
    caller has them already (one pass and one copy less on a big tensor)."""
    a = to_numpy(x)
    size = a.size
    if finite is None:
        a = a.astype(np.float64, copy=False)
        finite = a[np.isfinite(a)] if size else a.ravel()
    has = finite.size > 0
    nan = float("nan")
    out: dict[str, float] = {}
    for s in which:
        if size == 0:
            out[s] = nan
        elif s == "mean":
            out[s] = float(finite.mean()) if has else nan
        elif s == "std":
            out[s] = float(finite.std()) if has else nan
        elif s == "min":
            out[s] = float(finite.min()) if has else nan
        elif s == "max":
            out[s] = float(finite.max()) if has else nan
        elif s == "norm":
            out[s] = float(np.sqrt(np.dot(finite, finite)))
        elif s in ("abs_mean", "absmean"):
            out[s] = float(np.abs(finite).mean()) if has else nan
        elif s == "sparsity":  # non-finite values are never ~0
            out[s] = float(np.count_nonzero(np.abs(finite) <= 1e-8) / size)
        elif s == "nan":
            out[s] = float(size - finite.size)
        elif s == "rms":
            out[s] = float(np.sqrt(np.dot(finite, finite) / finite.size)) if has else nan
        else:
            raise ValueError(f"unknown stat {s!r}")
    return out


def histogram(x: Any, bins: int = 40, range_=None, finite: bool = False) -> dict:
    """``finite=True``: ``x`` is already a float64 vector of finite values (skips the copy and the filter)."""
    a = x if finite else to_numpy(x).astype(np.float64).ravel()
    if not finite:
        a = a[np.isfinite(a)]
    if a.size == 0:
        return {"edges": [0.0, 1.0], "counts": [0]}
    lo, hi = (a.min(), a.max()) if range_ is None else range_
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    counts, edges = np.histogram(a, bins=bins, range=(lo, hi))
    return {"edges": edges.tolist(), "counts": counts.tolist()}


def downsample2d(m: np.ndarray, max_rows: int = 96, max_cols: int = 96) -> tuple[np.ndarray, bool]:
    """Block-average a matrix down to at most ``max_rows x max_cols`` (NaN cells are left out of their block)."""
    r, c = m.shape
    fr, fc = math.ceil(r / max_rows), math.ceil(c / max_cols)
    if fr <= 1 and fc <= 1:
        return m, False
    fr, fc = max(fr, 1), max(fc, 1)
    R, C = math.ceil(r / fr) * fr, math.ceil(c / fc) * fc
    if not np.isfinite(m).all():
        return _downsample2d_nan(m, fr, fc, R, C), True
    # all finite: zero-pad, sum each block, divide by its real cell count (the edge blocks are partly padding)
    pad = m if (R, C) == (r, c) else np.pad(m, ((0, R - r), (0, C - c)))
    sums = pad.reshape(R // fr, fr, C // fc, fc).sum(axis=(1, 3), dtype=np.float64)
    rows = np.full(R // fr, fr)
    rows[-1] = r - (R // fr - 1) * fr
    cols = np.full(C // fc, fc)
    cols[-1] = c - (C // fc - 1) * fc
    return sums / np.outer(rows, cols), True


def _downsample2d_nan(m: np.ndarray, fr: int, fc: int, R: int, C: int) -> np.ndarray:
    r, c = m.shape
    pad = np.full((R, C), np.nan)
    pad[:r, :c] = m
    blocks = pad.reshape(R // fr, fr, C // fc, fc)
    with np.errstate(invalid="ignore"):
        return np.nanmean(blocks, axis=(1, 3))


def as_matrix(a: np.ndarray) -> np.ndarray:
    """View any array as a 2-D matrix for heatmap display."""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 0:
        return a.reshape(1, 1)
    if a.ndim == 1:
        return a.reshape(1, -1)
    if a.ndim == 2:
        return a
    return a.reshape(a.shape[0], -1)


def singular_values(a: np.ndarray, k: int = 64) -> list[float]:
    m = as_matrix(a)
    if m.shape[0] < 2 or m.shape[1] < 2 or m.size > 4_000_000:
        return []
    try:
        s = np.linalg.svd(m, compute_uv=False)
    except np.linalg.LinAlgError:
        return []
    return s[:k].tolist()
