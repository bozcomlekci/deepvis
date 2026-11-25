"""Equinox: modules are pytrees; submodules get wrapped in a scope-marking proxy for tracing."""

from __future__ import annotations

import dataclasses
from typing import Any

import equinox as eqx
import jax

from .common import JaxBackend, keypath_str
from .interp import MARK, ModuleMeta


class _Scoped(eqx.Module):
    inner: Any
    path: str = eqx.field(static=True)

    def __call__(self, *args, **kwargs):
        with jax.named_scope(MARK + self.path):
            return self.inner(*args, **kwargs)

    def __getattr__(self, name):
        if name in ("inner", "path") or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.inner, name)


def _join(a: str, b: str) -> str:
    return f"{a}.{b}" if a and b else (a or b)


def _scoped(m, path: str):
    kp_leaves, td = jax.tree_util.tree_flatten_with_path(
        m, is_leaf=lambda x: x is not m and isinstance(x, eqx.Module))
    new = []
    for kp, leaf in kp_leaves:
        if isinstance(leaf, eqx.Module):
            p = _join(path, keypath_str(kp))
            new.append(_Scoped(_scoped(leaf, p), p))
        else:
            new.append(leaf)
    return jax.tree_util.tree_unflatten(td, new)


def _attrs(mod) -> dict:
    out = {}
    for f in dataclasses.fields(mod):
        v = getattr(mod, f.name, None)
        if isinstance(v, (int, float, bool, str)) or v is None:
            out[f.name] = v
        elif isinstance(v, (tuple, list)) and all(isinstance(x, (int, float, str)) for x in v):
            out[f.name] = list(v)
        elif callable(v) and not isinstance(v, eqx.Module):
            nm = getattr(v, "__name__", None) or getattr(getattr(v, "fun", None), "__name__", None)
            if nm:
                out["fn_name" if f.name == "fn" else f.name] = nm
    return out


def _walk(obj, path: str, out: dict):
    if isinstance(obj, eqx.Module):
        out[path] = ModuleMeta(path, type(obj).__name__, _attrs(obj))
        for f in dataclasses.fields(obj):
            _walk(getattr(obj, f.name, None), _join(path, f.name), out)
    elif isinstance(obj, (list, tuple)):
        for i, o in enumerate(obj):
            _walk(o, _join(path, str(i)), out)
    elif isinstance(obj, dict):
        for k, o in obj.items():
            _walk(o, _join(path, str(k)), out)


class EquinoxBackend(JaxBackend):
    framework = "equinox"

    def __init__(self, model: eqx.Module, sample_input=None, name: str | None = None):
        super().__init__(name or type(model).__name__, sample_input)
        kp_leaves, treedef = jax.tree_util.tree_flatten_with_path(model)
        self._set_state([leaf for _, leaf in kp_leaves], treedef, [keypath_str(kp) for kp, _ in kp_leaves])
        self._meta: dict[str, ModuleMeta] = {}
        _walk(model, "", self._meta)

    @property
    def model(self):
        return jax.tree_util.tree_unflatten(self._treedef, self._leaves)

    def _call(self, params, args, kwargs, scoped):
        model = jax.tree_util.tree_unflatten(self._treedef, self.leaves_with(params))
        if not scoped:
            return model(*args, **kwargs)
        with jax.named_scope(MARK):
            return _scoped(model, "")(*args, **kwargs)

    def _module_meta(self):
        return self._meta

    def export(self):
        """The edited equinox model."""
        return self.model

    def clone(self) -> "EquinoxBackend":
        return EquinoxBackend(self.model, self._sample, self.name)
