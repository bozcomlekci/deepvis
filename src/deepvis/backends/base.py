"""Backend contract every framework adapter implements."""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..core.graph import ModelGraph, Node, ParamInfo


class ArrayOps:
    """Minimal functional array API so interventions work on torch and jax arrays alike."""

    framework = "numpy"

    def set_index(self, x, idx, value):
        y = np.array(x, copy=True)
        y[idx] = value
        return y

    def asarray(self, value, like):
        return np.asarray(value, dtype=np.asarray(like).dtype)

    def clip(self, x, lo, hi):
        return np.clip(x, lo, hi)

    def zeros_like(self, x):
        return np.zeros_like(x)

    def to_numpy(self, x) -> np.ndarray:
        return np.asarray(x)

    def feature_index(self, x, axis: int, neurons) -> tuple:
        nd = len(x.shape)
        ax = axis % nd
        sel = np.asarray(neurons) if isinstance(neurons, (list, tuple, np.ndarray)) else neurons
        return (slice(None),) * ax + (sel,)

    def broadcast_feature(self, vec, x, axis: int):
        """Reshape a 1-D feature vector so it broadcasts against ``x`` along ``axis``."""
        nd = len(x.shape)
        ax = axis % nd
        shape = [1] * nd
        shape[ax] = -1
        return self.asarray(vec, x).reshape(shape)


@dataclass
class InterventionContext:
    node: Node
    ops: ArrayOps
    feature_axis: int
    call_index: int = 0


@dataclass
class RunResult:
    output: Any
    acts: dict[str, Any] = field(default_factory=dict)  # node id -> native output (primary array)
    inputs: dict[str, Any] = field(default_factory=dict)  # node id -> native primary input
    grads: dict[str, np.ndarray] = field(default_factory=dict)  # node id -> d loss / d output
    param_grads: dict[str, np.ndarray] = field(default_factory=dict)
    attention: dict[str, np.ndarray] = field(default_factory=dict)
    # inside attention: {"q", "k", "v", "scores", "probs", "out"} per attention node (batch, heads, tokens, …)
    attention_parts: dict[str, dict] = field(default_factory=dict)
    loss: float | None = None


Tap = Callable[[str, Any, Any], None]  # (node_id, output, input) -> None
InterventionFn = Callable[[Any, InterventionContext], Any]


class Backend(ABC):
    framework: str = ""
    function_taps = True  # can persistent monitors/interventions attach to functional-op nodes?
    channels_last = False  # images as (…, H, W, C) — JAX and Keras — rather than (…, C, H, W)
    has_data = True  # False for a model on the meta device: shapes but no numbers (pages show the architecture)

    def __init__(self, name: str):
        self.name = name
        self._graph: ModelGraph | None = None
        self._ids = itertools.count(1)
        # persistent registrations: handle -> (node_id, fn)
        self.taps: dict[int, tuple[str, Tap]] = {}
        self.interventions: dict[int, tuple[str, InterventionFn]] = {}
        self.ops: ArrayOps = ArrayOps()

    # --------------------------------------------------------------- params
    @abstractmethod
    def param_paths(self) -> list[str]: ...

    @abstractmethod
    def param_info(self, path: str) -> ParamInfo: ...

    @abstractmethod
    def get_param(self, path: str) -> np.ndarray: ...

    @abstractmethod
    def set_param(self, path: str, value: np.ndarray) -> None: ...

    def inspecting(self):
        """Context for side-effect-free looks at the model mid-training (torch: eval mode, no autograd)."""
        import contextlib

        return contextlib.nullcontext()

    def load(self, native) -> None:
        """Replace every parameter with the values in ``native`` (whatever ``export()`` returns)."""
        raise NotImplementedError(f"{type(self).__name__} cannot load native parameters")

    def grads(self, native=None) -> dict[str, np.ndarray]:
        """Gradients by param path: from the framework (torch ``.grad``) or a grads tree shaped like the params."""
        return {}

    def get_params(self) -> dict[str, np.ndarray]:
        return {p: self.get_param(p) for p in self.param_paths()}

    # ---------------------------------------------------------------- graph
    @abstractmethod
    def build_graph(self, *inputs, **kwargs) -> ModelGraph: ...

    def graph(self, *inputs, **kwargs) -> ModelGraph:
        if self._graph is None or inputs:
            self._graph = self.build_graph(*inputs, **kwargs)
        return self._graph

    # ------------------------------------------------------------ execution
    @abstractmethod
    def forward(self, *args, **kwargs): ...

    @abstractmethod
    def run(
        self,
        args: tuple,
        kwargs: dict | None = None,
        record: Any = "all",
        record_inputs: bool = False,
        interventions: dict[str, list[InterventionFn]] | None = None,
        grads: bool = False,
        loss_fn: Callable | None = None,
        capture_attention: bool = False,
    ) -> RunResult: ...

    def feature_axis(self, node: Node) -> int:
        return -1

    # -------------------------------------------------- persistent hooks
    def add_tap(self, node_id: str, fn: Tap) -> int:
        h = next(self._ids)
        self.taps[h] = (node_id, fn)
        self._on_registrations_changed()
        return h

    def add_intervention(self, node_id: str, fn: InterventionFn) -> int:
        h = next(self._ids)
        self.interventions[h] = (node_id, fn)
        self._on_registrations_changed()
        return h

    def remove(self, handle: int) -> None:
        self.taps.pop(handle, None)
        self.interventions.pop(handle, None)
        self._on_registrations_changed()

    def _on_registrations_changed(self) -> None:  # backends with native hooks override
        pass

    def active_interventions(self) -> dict[str, list[InterventionFn]]:
        out: dict[str, list[InterventionFn]] = {}
        for nid, fn in self.interventions.values():
            out.setdefault(nid, []).append(fn)
        return out

    def active_taps(self) -> dict[str, list[Tap]]:
        out: dict[str, list[Tap]] = {}
        for nid, fn in self.taps.values():
            out.setdefault(nid, []).append(fn)
        return out

    # ----------------------------------------------------------- misc
    @abstractmethod
    def export(self): ...

    def to_native(self, x):
        return x
