"""Weight-editing operations.

Every op is a pure numpy function ``fn(region, **args) -> new_region``. Ops are
framework-agnostic: the backend reads a parameter as numpy, the op transforms the
selected region, and the backend writes it back (torch in-place, jax functionally).

Register your own with :func:`register_op`.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class OpDef:
    name: str
    fn: Callable
    default_roles: tuple[str, ...] | None  # which params of a *node* an op touches by default
    doc: str = ""


OPS: dict[str, OpDef] = {}


def register_op(name: str, default_roles: tuple[str, ...] | None = None):
    """Decorator: ``@register_op("my_op")`` then ``net["fc1"].edit("my_op", k=3)``."""

    def deco(fn):
        OPS[name] = OpDef(name, fn, default_roles, (fn.__doc__ or "").strip())
        return fn

    return deco


def _rng(seed):
    return np.random.default_rng(seed)


@register_op("set")
def op_set(w, value=0.0):
    """Assign a scalar or broadcastable array."""
    return np.broadcast_to(np.asarray(value, dtype=w.dtype), w.shape).copy()


@register_op("scale")
def op_scale(w, factor=1.0):
    """Multiply by a factor."""
    return w * factor


@register_op("add")
def op_add(w, value=0.0):
    """Add a scalar or broadcastable array."""
    return w + np.asarray(value, dtype=w.dtype)


@register_op("zero")
def op_zero(w):
    """Set to zero."""
    return np.zeros_like(w)


@register_op("noise")
def op_noise(w, std=0.01, seed=None, relative=False, dist="normal"):
    """Add Gaussian (or uniform) noise; ``relative=True`` scales std by the region's std."""
    rng = _rng(seed)
    s = std * (float(np.std(w)) if relative else 1.0)
    n = rng.normal(0, s, w.shape) if dist == "normal" else rng.uniform(-s, s, w.shape)
    return w + n.astype(w.dtype)


@register_op("clamp")
def op_clamp(w, min=None, max=None):  # noqa: A002 - mirrors torch.clamp
    """Clip values into [min, max]."""
    return np.clip(w, min, max)


@register_op("prune", default_roles=("weight",))
def op_prune(w, amount=0.5, method="magnitude", seed=None):
    """Zero out a fraction of entries (smallest |w| for ``magnitude``, or ``random``)."""
    k = int(round(amount * w.size))
    if k <= 0:
        return w.copy()
    flat = w.ravel().copy()
    if method == "magnitude":
        idx = np.argsort(np.abs(flat), kind="stable")[:k]
    elif method == "random":
        idx = _rng(seed).choice(flat.size, size=k, replace=False)
    else:
        raise ValueError(f"unknown prune method {method!r}")
    flat[idx] = 0
    return flat.reshape(w.shape)


@register_op("quantize", default_roles=("weight",))
def op_quantize(w, bits=8, symmetric=True):
    """Uniform fake-quantization to ``bits`` bits."""
    levels = 2 ** int(bits) - 1
    if symmetric:
        m = float(np.max(np.abs(w))) or 1.0
        step = 2 * m / levels
        return (np.round(w / step) * step).astype(w.dtype)
    lo, hi = float(w.min()), float(w.max())
    step = (hi - lo) / levels if hi > lo else 1.0
    return (np.round((w - lo) / step) * step + lo).astype(w.dtype)


def _as_2d(w):
    return w.reshape(w.shape[0], -1) if w.ndim > 2 else w


@register_op("low_rank", default_roles=("weight",))
def op_low_rank(w, rank=1):
    """Best rank-``rank`` approximation via truncated SVD (convs are flattened per output channel)."""
    if w.ndim < 2:
        return w.copy()
    m = _as_2d(w)
    u, s, vt = np.linalg.svd(m, full_matrices=False)
    r = int(rank)
    approx = (u[:, :r] * s[:r]) @ vt[:r]
    return approx.reshape(w.shape).astype(w.dtype)


@register_op("orthogonalize", default_roles=("weight",))
def op_orthogonalize(w):
    """Replace with the nearest (semi-)orthogonal matrix U Vᵀ, keeping the overall scale."""
    if w.ndim < 2:
        return w.copy()
    m = _as_2d(w)
    u, s, vt = np.linalg.svd(m, full_matrices=False)
    return ((u @ vt) * float(np.mean(s))).reshape(w.shape).astype(w.dtype)


@register_op("normalize", default_roles=("weight",))
def op_normalize(w, axis=-1, norm=1.0):
    """Rescale slices along ``axis`` to L2 norm ``norm``."""
    n = np.linalg.norm(w, axis=axis, keepdims=True)
    return (w / np.where(n == 0, 1, n) * norm).astype(w.dtype)


@register_op("reinit")
def op_reinit(w, method="kaiming", seed=None, fan_in=None, std=None):
    """Fresh random init (``kaiming`` uniform, ``xavier`` uniform, ``normal`` or ``zeros``)."""
    rng = _rng(seed)
    if method == "zeros":
        return np.zeros_like(w)
    if fan_in is None:
        fan_in = int(np.prod(w.shape[1:])) if w.ndim > 1 else w.shape[0]
    fan_out = w.shape[0] if w.ndim > 1 else w.shape[0]
    if method == "kaiming":
        b = np.sqrt(1.0 / max(fan_in, 1)) * np.sqrt(3.0)
        return rng.uniform(-b, b, w.shape).astype(w.dtype)
    if method == "xavier":
        b = np.sqrt(6.0 / max(fan_in + fan_out, 1))
        return rng.uniform(-b, b, w.shape).astype(w.dtype)
    if method == "normal":
        return rng.normal(0, 0.02 if std is None else std, w.shape).astype(w.dtype)
    raise ValueError(f"unknown init {method!r}")


@register_op("sign")
def op_sign(w):
    """Keep only signs (±1, 0)."""
    return np.sign(w).astype(w.dtype)


@register_op("abs")
def op_abs(w):
    """Absolute value."""
    return np.abs(w)


@register_op("negate")
def op_negate(w):
    """Flip sign."""
    return -w


@register_op("shuffle")
def op_shuffle(w, seed=None, axis=None):
    """Randomly permute entries (all, or along one axis)."""
    rng = _rng(seed)
    if axis is None:
        flat = w.ravel().copy()
        rng.shuffle(flat)
        return flat.reshape(w.shape)
    return np.take(w, rng.permutation(w.shape[axis]), axis=axis)


@register_op("permute")
def op_permute(w, perm, axis=0):
    """Reorder along ``axis`` by index list ``perm``."""
    return np.take(w, np.asarray(perm), axis=axis)


@register_op("mask")
def op_mask(w, mask):
    """Multiply elementwise by a (broadcastable) mask."""
    return (w * np.asarray(mask, dtype=w.dtype)).astype(w.dtype)


@register_op("apply")
def op_apply(w, fn=None):
    """Arbitrary python callable ``fn(np.ndarray) -> np.ndarray``."""
    out = np.asarray(fn(w.copy()))
    return np.broadcast_to(out, w.shape).astype(w.dtype).copy()


# --------------------------------------------------------- where-expressions
_WHERE_FUNCS = {
    "abs": np.abs, "sign": np.sign, "sqrt": np.sqrt, "exp": np.exp, "log": np.log,
    "mean": np.mean, "std": np.std, "max": np.max, "min": np.min, "median": np.median,
    "quantile": np.quantile, "percentile": np.percentile, "argsort": np.argsort,
}
_WHERE_NODES = (ast.Expression, ast.Compare, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Name, ast.Load,
                ast.Constant, ast.Call, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq, ast.Add,
                ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.BitAnd, ast.BitOr,
                ast.Invert, ast.Not, ast.And, ast.Or, ast.Mod, ast.keyword, ast.Tuple)


def eval_where(expr, w: np.ndarray) -> np.ndarray:
    """Evaluate a boolean mask over region ``w``.

    ``expr`` is a callable ``f(w) -> bool array`` or a safe expression string such as
    ``"abs(w) < 0.01"`` or ``"(w > 0) & (abs(w) > quantile(abs(w), 0.9))"``.
    """
    if callable(expr):
        return np.asarray(expr(w), dtype=bool)
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _WHERE_NODES):
            raise ValueError(f"disallowed syntax in where-expression: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in _WHERE_FUNCS and node.id != "w":
            raise ValueError(f"unknown name {node.id!r} in where-expression (use 'w')")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in _WHERE_FUNCS):
            raise ValueError("only whitelisted functions may be called in where-expressions")
    ns = dict(_WHERE_FUNCS, w=w)
    return np.asarray(eval(compile(tree, "<where>", "eval"), {"__builtins__": {}}, ns), dtype=bool)  # noqa: S307
