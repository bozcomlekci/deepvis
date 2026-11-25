"""Pure JAX: any ``apply_fn(params, *inputs)`` with a params pytree.

Nodes come from ``jax.named_scope`` regions (``dv.scope("attn")``) when present;
otherwise parameter groups are inferred (``x @ W + b`` becomes one Linear node).
"""

from __future__ import annotations

from typing import Callable

import jax

from .common import JaxBackend, keypath_str


class PureBackend(JaxBackend):
    framework = "jax"
    marker_mode = False

    def __init__(self, apply_fn: Callable, params, sample_input=None, name: str | None = None):
        super().__init__(name or getattr(apply_fn, "__name__", "fn"), sample_input)
        self.apply_fn = apply_fn
        kp_leaves, treedef = jax.tree_util.tree_flatten_with_path(params)
        self._set_state([leaf for _, leaf in kp_leaves], treedef, [keypath_str(kp) for kp, _ in kp_leaves])

    def _call(self, params, args, kwargs, scoped):
        tree = jax.tree_util.tree_unflatten(self._treedef, self.leaves_with(params))
        return self.apply_fn(tree, *args, **kwargs)

    def export(self):
        """The edited params pytree."""
        return jax.tree_util.tree_unflatten(self._treedef, self._leaves)

    def clone(self) -> "PureBackend":
        return PureBackend(self.apply_fn, self.export(), self._sample, self.name)
