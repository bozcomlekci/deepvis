"""flax.linen: interception of every module method call marks module boundaries."""

from __future__ import annotations

import dataclasses

import flax.linen as nn
import jax

from .common import JaxBackend, keypath_str
from .interp import MARK, ModuleMeta


def _attrs(mod) -> dict:
    out = {}
    for f in dataclasses.fields(mod):
        if f.name in ("parent", "name") or f.name.startswith("_"):
            continue
        try:
            v = getattr(mod, f.name)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(v, (int, float, bool, str)) or v is None:
            out[f.name] = v
        elif isinstance(v, (tuple, list)) and all(isinstance(x, (int, float, str)) for x in v):
            out[f.name] = list(v)
        elif callable(v) and not isinstance(v, nn.Module) and hasattr(v, "__name__"):
            out[f.name] = v.__name__
    return out


class LinenBackend(JaxBackend):
    framework = "flax"

    def __init__(self, module: nn.Module, variables, sample_input=None, name: str | None = None, method=None,
                 apply_kwargs: dict | None = None):
        super().__init__(name or type(module).__name__, sample_input)
        self.module = module
        self.method = method
        self.apply_kwargs = dict(apply_kwargs or {})
        if "params" not in variables:
            variables = {"params": variables}
        kp_leaves, treedef = jax.tree_util.tree_flatten_with_path(variables)
        paths, trainable = [], []
        for kp, _ in kp_leaves:
            full = keypath_str(kp)
            col, _, rest = full.partition(".")
            paths.append(rest if col == "params" else rest)
            trainable.append(col == "params")
        if len(set(paths)) != len(paths):  # collision across collections -> prefix non-params
            paths = [p if t else f"{keypath_str(kp).split('.')[0]}:{p}" for (kp, _), p, t in
                     zip(kp_leaves, paths, trainable)]
        self._set_state([leaf for _, leaf in kp_leaves], treedef, paths, trainable)
        self._meta: dict[str, ModuleMeta] = {}

    def _native_tree(self, native):
        if isinstance(native, tuple) and len(native) == 2 and isinstance(native[0], nn.Module):
            native = native[1]
        native = dict(native)
        if "params" not in native:
            native = {"params": native}
        return {**dict(self._variables(self.param_arrays())), **native}  # keep batch_stats etc. if not given

    def _grad_path(self, kp) -> str:
        p = keypath_str(kp)
        return p.partition(".")[2] if p.startswith("params.") else p

    def _variables(self, params):
        return jax.tree_util.tree_unflatten(self._treedef, self.leaves_with(params))

    def _call(self, params, args, kwargs, scoped):
        variables = self._variables(params)
        kw = dict(self.apply_kwargs, **kwargs)
        if self.method is not None:
            kw["method"] = self.method
        if not scoped:
            return self.module.apply(variables, *args, **kw)

        def interceptor(next_fun, a, k, context):
            mod = context.module
            path = ".".join(str(p) for p in mod.path)
            if path not in self._meta:
                self._meta[path] = ModuleMeta(path, type(mod).__name__, _attrs(mod))
            with jax.named_scope(MARK + path):
                return next_fun(*a, **k)

        with nn.intercept_methods(interceptor):
            return self.module.apply(variables, *args, **kw)

    def _module_meta(self):
        # filled by the interceptor while build_program traces (it copies the dict afterwards)
        return self._meta

    def export(self):
        """``(module, variables)`` with the edited variables."""
        return self.module, self._variables(self.param_arrays())

    def clone(self) -> "LinenBackend":
        return LinenBackend(self.module, self._variables(self.param_arrays()), self._sample, self.name, self.method,
                            self.apply_kwargs)
