"""flax.nnx: split/merge for parameters; module boundaries marked by temporarily wrapping
``__call__`` of every module class present in the model (identity-checked, restored after)."""

from __future__ import annotations

import contextlib

import jax
from flax import nnx

from .common import JaxBackend, keypath_str
from .interp import MARK, ModuleMeta


def _strip_value(kp):
    kp = list(kp)
    if kp and type(kp[-1]).__name__ == "GetAttrKey" and kp[-1].name in ("value", "raw_value"):
        kp = kp[:-1]
    return kp


def _iter_modules(m):
    it = nnx.iter_modules(m) if hasattr(nnx, "iter_modules") else m.iter_modules()
    for path, sub in it:
        yield ".".join(str(p) for p in path), sub


def _attrs(mod) -> dict:
    out = {}
    for k, v in vars(mod).items():
        if k.startswith("_"):
            continue
        if isinstance(v, (int, float, bool, str)) or v is None:
            out[k] = v
        elif isinstance(v, (tuple, list)) and all(isinstance(x, (int, float, str)) for x in v):
            out[k] = list(v)
        elif callable(v) and not isinstance(v, nnx.Module) and hasattr(v, "__name__"):
            out[k] = v.__name__
    return out


@contextlib.contextmanager
def _scoped_calls(ids: dict[int, str]):
    """Patch __call__ on each module class so calls on *known instances* enter a named scope."""
    patched = []
    for cls in {type(m) for m in _objs(ids)}:
        own = cls.__dict__.get("__call__")
        orig = cls.__call__

        def wrapper(self, *a, __orig=orig, **k):
            p = ids.get(id(self))
            if p is None:
                return __orig(self, *a, **k)
            with jax.named_scope(MARK + p):
                return __orig(self, *a, **k)

        cls.__call__ = wrapper
        patched.append((cls, own))
    try:
        yield
    finally:
        for cls, own in patched:
            if own is None:
                del cls.__call__
            else:
                cls.__call__ = own


_LIVE: dict[int, object] = {}


def _objs(ids):
    return [_LIVE[i] for i in ids if i in _LIVE]


class NNXBackend(JaxBackend):
    framework = "nnx"

    def __init__(self, model: nnx.Module, sample_input=None, name: str | None = None, eval_mode: bool = True):
        super().__init__(name or type(model).__name__, sample_input)
        self.model = model
        self.eval_mode = eval_mode
        self._graphdef, self._pstate, self._rest = nnx.split(model, (nnx.Param, nnx.BatchStat), ...)
        trainable_paths = {keypath_str(_strip_value(kp)) for kp, _ in
                           jax.tree_util.tree_flatten_with_path(nnx.state(model, nnx.Param))[0]}
        kp_leaves, treedef = jax.tree_util.tree_flatten_with_path(self._pstate)
        paths = [keypath_str(_strip_value(kp)) for kp, _ in kp_leaves]
        self._set_state([leaf for _, leaf in kp_leaves], treedef, paths, [p in trainable_paths for p in paths])
        self._meta = {p: ModuleMeta(p, type(sub).__name__, _attrs(sub)) for p, sub in _iter_modules(model)}

    def _module(self, params):
        state = jax.tree_util.tree_unflatten(self._treedef, self.leaves_with(params))
        m = nnx.merge(self._graphdef, state, self._rest)
        if self.eval_mode and hasattr(m, "eval"):
            m.eval()
        return m

    def _call(self, params, args, kwargs, scoped):
        m = self._module(params)
        if not scoped:
            return m(*args, **kwargs)
        ids = {}
        for p, sub in _iter_modules(m):
            ids[id(sub)] = p
            _LIVE[id(sub)] = sub
        try:
            with _scoped_calls(ids):
                return m(*args, **kwargs)
        finally:
            for i in ids:
                _LIVE.pop(i, None)

    def _module_meta(self):
        return self._meta

    def _on_params_changed(self) -> None:
        nnx.update(self.model, jax.tree_util.tree_unflatten(self._treedef, self._leaves))

    def _native_tree(self, native):
        if isinstance(native, nnx.Module):
            native = nnx.split(native, (nnx.Param, nnx.BatchStat), ...)[1]
        return native

    def _grad_path(self, kp) -> str:
        return keypath_str(_strip_value(kp))

    def sync(self) -> None:
        """Re-read the live module (after an optimizer updated it in place)."""
        self.load(self.model)

    def export(self):
        """The live nnx module (kept in sync with every edit)."""
        return self.model

    def clone(self) -> "NNXBackend":
        return NNXBackend(nnx.clone(self.model), self._sample, self.name, self.eval_mode)
