"""Activation-level interventions (applied to a node's output during the forward pass).

>>> with net.intervene("relu1", dv.ablate([3, 7])):
...     y = net(x)
>>> h = net.intervene("block2", dv.steer(direction, alpha=4.0))   # persistent
>>> h.remove()

Interventions are framework-agnostic callables ``fn(x, ctx) -> x'`` where ``x`` is the
native tensor/array and ``ctx.ops`` provides functional array primitives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from ..backends.base import InterventionContext

if TYPE_CHECKING:  # pragma: no cover
    from ..net import Net


class Intervention:
    name = "intervention"

    def __call__(self, x, ctx: InterventionContext):  # pragma: no cover
        raise NotImplementedError

    def _axis(self, ctx: InterventionContext) -> int:
        ax = getattr(self, "axis", None)
        return ctx.feature_axis if ax is None else ax

    def __repr__(self) -> str:
        fields = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        body = ", ".join(f"{k}={_short(v)}" for k, v in fields.items())
        return f"{self.name}({body})"


def _short(v):
    if isinstance(v, np.ndarray):
        return f"array{v.shape}"
    return repr(v)


class ablate(Intervention):
    """Set neurons (features along the channel axis) to ``value`` — all of them if ``neurons`` is None."""

    name = "ablate"

    def __init__(self, neurons=None, value: float = 0.0, axis: int | None = None):
        self.neurons, self.value, self.axis = neurons, value, axis

    def __call__(self, x, ctx):
        if self.neurons is None:
            return ctx.ops.zeros_like(x) + self.value
        idx = ctx.ops.feature_index(x, self._axis(ctx), self.neurons)
        return ctx.ops.set_index(x, idx, self.value)


class mean_ablate(Intervention):
    """Replace neurons by their mean over every other axis (batch/space) of the current input."""

    name = "mean_ablate"

    def __init__(self, neurons=None, axis: int | None = None):
        self.neurons, self.axis = neurons, axis

    def __call__(self, x, ctx):
        ax = self._axis(ctx) % len(x.shape)
        other = tuple(i for i in range(len(x.shape)) if i != ax)
        mean = x.mean(axis=other, keepdims=True) if ctx.ops.framework != "torch" else x.mean(dim=other, keepdim=True)
        full = mean + ctx.ops.zeros_like(x)
        if self.neurons is None:
            return full
        idx = ctx.ops.feature_index(x, ax, self.neurons)
        return ctx.ops.set_index(x, idx, full[idx])


class amplify(Intervention):
    """Multiply activations (optionally only some neurons) by ``factor``."""

    name = "amplify"

    def __init__(self, factor: float, neurons=None, axis: int | None = None):
        self.factor, self.neurons, self.axis = factor, neurons, axis

    def __call__(self, x, ctx):
        if self.neurons is None:
            return x * self.factor
        idx = ctx.ops.feature_index(x, self._axis(ctx), self.neurons)
        return ctx.ops.set_index(x, idx, x[idx] * self.factor)


class clamp(Intervention):
    """Clip activations into [min, max]."""

    name = "clamp"

    def __init__(self, min: float | None = None, max: float | None = None):  # noqa: A002
        self.min, self.max = min, max

    def __call__(self, x, ctx):
        return ctx.ops.clip(x, self.min, self.max)


class steer(Intervention):
    """Add ``alpha * vector`` along the feature axis (activation steering)."""

    name = "steer"

    def __init__(self, vector, alpha: float = 1.0, axis: int | None = None):
        self.vector, self.alpha, self.axis = np.asarray(vector, dtype=np.float32), alpha, axis

    def __call__(self, x, ctx):
        v = ctx.ops.broadcast_feature(self.vector * self.alpha, x, self._axis(ctx))
        return x + v


class add_noise(Intervention):
    """Gaussian noise on activations (fixed seed → reproducible)."""

    name = "add_noise"

    def __init__(self, std: float = 0.1, seed: int = 0):
        self.std, self.seed = std, seed

    def __call__(self, x, ctx):
        n = np.random.default_rng(self.seed).normal(0, self.std, tuple(x.shape)).astype(np.float32)
        return x + ctx.ops.asarray(n, x)


class replace(Intervention):
    """Replace the activation by a constant/array, or by ``fn(x)``; optionally only some neurons."""

    name = "replace"

    def __init__(self, value: Any, neurons=None, axis: int | None = None):
        self.value, self.neurons, self.axis = value, neurons, axis

    def __call__(self, x, ctx):
        new = self.value(x) if callable(self.value) else self.value
        if not callable(self.value):
            from ..core.tensor import to_numpy

            arr = to_numpy(new)  # detaches torch tensors, copies jax arrays
            new = ctx.ops.asarray(np.broadcast_to(arr, tuple(x.shape)) if self.neurons is None else arr, x)
        if self.neurons is None:
            return new
        idx = ctx.ops.feature_index(x, self._axis(ctx), self.neurons)
        src = new[idx] if tuple(new.shape) == tuple(x.shape) else new
        return ctx.ops.set_index(x, idx, src)


class custom(Intervention):
    """Wrap any ``fn(x, ctx) -> x`` (or ``fn(x) -> x``)."""

    name = "custom"

    def __init__(self, fn: Callable):
        self.fn = fn
        try:
            import inspect

            self._nargs = len(inspect.signature(fn).parameters)
        except (TypeError, ValueError):
            self._nargs = 1

    def __call__(self, x, ctx):
        return self.fn(x, ctx) if self._nargs >= 2 else self.fn(x)


def intervention(fn: Callable) -> custom:
    """Decorator turning a function into an intervention."""
    return custom(fn)


def as_intervention(obj) -> Intervention:
    if isinstance(obj, Intervention):
        return obj
    if callable(obj):
        return custom(obj)
    raise TypeError(f"not an intervention: {obj!r}")


class InterventionHandle:
    """Returned by :meth:`Net.intervene`; a context manager or call ``.remove()``."""

    def __init__(self, net: "Net", handles: list[int], description: str):
        self.net, self.handles, self.description = net, handles, description

    def remove(self) -> None:
        for h in self.handles:
            self.net.backend.remove(h)
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()

    def __repr__(self) -> str:
        state = "active" if self.handles else "removed"
        return f"InterventionHandle({self.description}, {state})"


class PatchResult:
    def __init__(self, clean, corrupted, patched, at):
        self.clean, self.corrupted, self.patched, self.at = clean, corrupted, patched, at

    def recovery(self, metric: Callable | None = None) -> float:
        """Fraction of the clean-vs-corrupted gap recovered by patching (1.0 = fully restored)."""
        from ..core.tensor import to_numpy

        m = metric or (lambda y: float(np.asarray(to_numpy(y)).ravel()[int(np.argmax(to_numpy(self.clean)))]))
        c, k, p = m(self.clean), m(self.corrupted), m(self.patched)
        return float((p - k) / (c - k)) if c != k else float("nan")

    def __repr__(self) -> str:
        return f"PatchResult(at={self.at!r})"


def patch(net: "Net", source, target, at, neurons=None) -> PatchResult:
    """Activation patching: run ``target`` but splice in ``at``'s activations from ``source``.

    ``source`` / ``target`` are inputs (a tensor/array or a tuple of positional args).
    """
    src_args = source if isinstance(source, tuple) else (source,)
    tgt_args = target if isinstance(target, tuple) else (target,)
    ids = net._resolve_nodes(at)
    clean = net.backend.run(src_args, {}, record=ids)
    corrupted = net.backend.run(tgt_args, {}, record=None).output
    fns = {nid: [replace(clean.acts[nid], neurons=neurons)] for nid in ids}
    patched = net.backend.run(tgt_args, {}, record=None, interventions=fns).output
    return PatchResult(clean.output, corrupted, patched, at)
