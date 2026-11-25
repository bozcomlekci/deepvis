"""User-facing handles for editing: :class:`ParamRef` and the shared edit-method mixin."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from ..core.selectors import format_index, normalize_index
from ..core.tensor import histogram, stats
from .edits import Edit

if TYPE_CHECKING:  # pragma: no cover
    from ..net import Net


class EditMethods:
    """Weight-editing verbs shared by ParamRef, NodeView and Selection.

    Subclasses implement ``_edit(op, args, where=None, params=None, label=None)``.
    Every call is recorded in ``net.history`` (undo/redo) and returns ``self``
    so edits can be chained: ``net["fc1"].prune(0.5).noise(0.01)``.
    """

    def _edit(self, op: str, args: dict, where=None, params=None, label=None):  # pragma: no cover
        raise NotImplementedError

    def edit(self, op: str | Callable, where=None, params=None, **args):
        """Generic entry point: a registered op name or any ``fn(np.ndarray) -> np.ndarray``."""
        if callable(op):
            return self._edit("apply", {"fn": op}, where, params)
        return self._edit(op, args, where, params)

    def set(self, value, where=None, params=None):
        return self._edit("set", {"value": value}, where, params)

    def scale(self, factor: float, where=None, params=None):
        return self._edit("scale", {"factor": factor}, where, params)

    def add(self, value, where=None, params=None):
        return self._edit("add", {"value": value}, where, params)

    def zero(self, where=None, params=None):
        return self._edit("zero", {}, where, params)

    def noise(self, std: float = 0.01, seed: int | None = None, relative: bool = False, where=None, params=None):
        return self._edit("noise", {"std": std, "seed": seed, "relative": relative}, where, params)

    def clamp(self, min: float | None = None, max: float | None = None, where=None, params=None):  # noqa: A002
        return self._edit("clamp", {"min": min, "max": max}, where, params)

    def prune(self, amount: float = 0.5, method: str = "magnitude", seed: int | None = None, where=None,
              params=None):
        return self._edit("prune", {"amount": amount, "method": method, "seed": seed}, where, params)

    def quantize(self, bits: int = 8, symmetric: bool = True, params=None):
        return self._edit("quantize", {"bits": bits, "symmetric": symmetric}, None, params)

    def low_rank(self, rank: int = 1, params=None):
        return self._edit("low_rank", {"rank": rank}, None, params)

    def orthogonalize(self, params=None):
        return self._edit("orthogonalize", {}, None, params)

    def normalize(self, axis: int = -1, norm: float = 1.0, params=None):
        return self._edit("normalize", {"axis": axis, "norm": norm}, None, params)

    def reinit(self, method: str = "kaiming", seed: int | None = None, params=None):
        return self._edit("reinit", {"method": method, "seed": seed}, None, params)

    def shuffle(self, seed: int | None = None, axis: int | None = None, params=None):
        return self._edit("shuffle", {"seed": seed, "axis": axis}, None, params)

    def negate(self, where=None, params=None):
        return self._edit("negate", {}, where, params)

    def sign(self, params=None):
        return self._edit("sign", {}, None, params)

    def mask(self, mask, params=None):
        return self._edit("mask", {"mask": np.asarray(mask)}, None, params)

    def apply(self, fn: Callable[[np.ndarray], np.ndarray], where=None, params=None):
        return self._edit("apply", {"fn": fn}, where, params)


class ParamRef(EditMethods):
    """A parameter (optionally a numpy-style region of it) that can be read and edited.

    >>> w = net["fc1.weight"]          # whole tensor
    >>> w[:, 3] = 0                    # slice assignment, recorded in history
    >>> w[0:8].noise(0.05, seed=0)     # region edits
    >>> w *= 0.5                       # in-place arithmetic
    """

    def __init__(self, net: "Net", path: str, index: Any = None):
        self.net = net
        self.path = path
        self.index = index

    # ------------------------------------------------------------- reading
    @property
    def info(self):
        return self.net.backend.param_info(self.path)

    def numpy(self) -> np.ndarray:
        full = self.net.backend.get_param(self.path)
        return full if self.index is None else np.asarray(full[normalize_index(self.index)]).copy()

    @property
    def value(self) -> np.ndarray:
        return self.numpy()

    @property
    def shape(self) -> tuple:
        return self.numpy().shape if self.index is not None else tuple(self.info.shape)

    @property
    def dtype(self):
        return self.info.dtype

    def __array__(self, dtype=None, copy=None):
        a = self.numpy()
        return a.astype(dtype) if dtype is not None else a

    def stats(self) -> dict:
        return stats(self.numpy(), ("mean", "std", "min", "max", "norm", "sparsity"))

    def histogram(self, bins: int = 40) -> dict:
        return histogram(self.numpy(), bins)

    @property
    def native(self):
        """The framework-native tensor/array (whole parameter)."""
        getter = getattr(self.net.backend, "get_native_param", None)
        return getter(self.path) if getter else self.net.backend.get_param(self.path)

    # ------------------------------------------------------------ indexing
    def __getitem__(self, idx) -> "ParamRef":
        if self.index is not None:
            raise IndexError("chained indexing is not supported; use a single index expression")
        return ParamRef(self.net, self.path, idx)

    def __setitem__(self, idx, value) -> None:
        if self.index is not None:
            raise IndexError("chained indexing is not supported; use a single index expression")
        if isinstance(value, ParamRef) and value.path == self.path and _same_index(value.index, idx):
            return  # ``w[i] *= 2`` — the in-place operator already applied (and recorded) the edit
        ParamRef(self.net, self.path, idx).set(value)

    # -------------------------------------------------------------- edits
    def _edit(self, op, args, where=None, params=None, label=None):
        self.net.apply_edit(Edit(op, self.path, index=self.index, args=args, where=where, label=label))
        return self

    def __imul__(self, other):
        return self.scale(other)

    def __itruediv__(self, other):
        return self.scale(1.0 / other)

    def __iadd__(self, other):
        return self.add(other)

    def __isub__(self, other):
        return self.add(-np.asarray(other))

    # ------------------------------------------------------------- display
    def __repr__(self) -> str:
        s = self.stats()
        idx = format_index(self.index) if self.index is not None else ""
        return (f"ParamRef({self.path}{idx or ''}, shape={self.shape}, mean={s['mean']:.4g}, "
                f"std={s['std']:.4g}, min={s['min']:.4g}, max={s['max']:.4g})")

    def show(self, **kw):
        return self.net.show(focus=self.info.module, **kw)


def _same_index(a, b) -> bool:
    if a is b:
        return True
    fa, fb = format_index(a), format_index(b)
    return fa is not None and fa == fb


def axis_index(ndim: int, axis: int, sel) -> str:
    """Index string selecting ``sel`` along ``axis`` of an ``ndim``-array, e.g. ``[:, [1, 2]]``."""
    ax = axis % ndim
    s = repr([int(i) for i in np.atleast_1d(sel)])
    return "[" + ", ".join([":"] * ax + [s]) + "]"


class NeuronSelection(EditMethods):
    """Output neurons of a layer: edits touch the matching weight rows (and bias entries).

    >>> net["fc1"].neurons([3, 7]).zero()        # knock out two hidden units
    >>> net["conv1"].neurons(0).scale(2.0)       # amplify one channel
    """

    def __init__(self, net: "Net", node_id: str, neurons):
        self.net = net
        self.node_id = node_id
        self.neurons = [int(i) for i in np.atleast_1d(neurons)]

    @property
    def node(self):
        return self.net.graph.nodes[self.node_id]

    def _edit(self, op, args, where=None, params=None, label=None):
        node = self.node
        targets = []
        for role, r in node.roles.items():
            if r.out_axis is None:
                continue
            if params is not None and role not in params and r.name not in params:
                continue
            info = node.params[r.name]
            targets.append((info.path, axis_index(len(info.shape), r.out_axis, self.neurons)))
        if not targets:
            raise ValueError(f"{self.node_id!r} has no parameters with a neuron (output) axis")
        lbl = label or f"{op} neurons {self.neurons} of {self.node_id}"
        edits = [Edit(op, p, index=i, args=args, where=where) for p, i in targets]
        self.net.apply_edit(edits[0] if len(edits) == 1 else _group(edits, lbl))
        return self

    def incoming(self) -> np.ndarray:
        """Weights feeding these neurons, shaped (len(neurons), fan_in...)."""
        node = self.node
        r = node.roles["weight"]
        w = self.net.backend.get_param(node.params[r.name].path)
        return np.moveaxis(np.take(w, self.neurons, axis=r.out_axis), r.out_axis, 0)

    def __repr__(self) -> str:
        return f"NeuronSelection({self.node_id}, neurons={self.neurons})"


def _group(edits: list[Edit], label: str) -> Edit:
    return Edit("group", [e.target for e in edits], args={"edits": edits}, label=label)
