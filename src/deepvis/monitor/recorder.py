"""Monitoring intermediate layers.

* :class:`Trace` — one forward pass, every intermediate (plus optional gradients).
* :class:`Monitor` — persistent taps that accumulate per-call statistics, e.g. across
  a training loop, while you call the model normally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..core.tensor import DEFAULT_STATS, histogram, stats, to_numpy

if TYPE_CHECKING:  # pragma: no cover
    from ..net import Net


class Trace:
    """Everything recorded during one forward pass."""

    def __init__(self, net: "Net", args: tuple, kwargs: dict, result):
        self.net = net
        self.args, self.kwargs = args, kwargs
        self.output = result.output
        self.loss = result.loss
        self._acts_native = result.acts
        self.acts: dict[str, np.ndarray] = {k: to_numpy(v) for k, v in result.acts.items() if v is not None}
        self.inputs: dict[str, np.ndarray] = {k: to_numpy(v) for k, v in result.inputs.items() if v is not None}
        self.grads: dict[str, np.ndarray] = dict(result.grads)
        self.param_grads: dict[str, np.ndarray] = dict(result.param_grads)
        self.attention: dict[str, np.ndarray] = dict(result.attention)
        self.attention_parts: dict[str, dict] = dict(getattr(result, "attention_parts", {}) or {})

    def __getitem__(self, node_id: str) -> np.ndarray:
        if node_id in self.acts:
            return self.acts[node_id]
        from ..core.selectors import filter_paths

        m = filter_paths(node_id, list(self.acts))
        if len(m) == 1:
            return self.acts[m[0]]
        raise KeyError(f"{node_id!r} not recorded; available: {list(self.acts)[:12]}…")

    def __contains__(self, node_id: str) -> bool:
        return node_id in self.acts

    def keys(self):
        return self.acts.keys()

    def stats(self, node_id: str, which=DEFAULT_STATS) -> dict[str, float]:
        return stats(self.acts[node_id], which)

    def table(self, which=("mean", "std", "min", "max", "sparsity")) -> str:
        g = self.net.graph
        order = [n for n in g.topo_order() if n in self.acts]
        head = f"{'node':<32} {'kind':<14} {'shape':<18}" + "".join(f"{w:>10}" for w in which)
        lines = [head, "─" * len(head)]
        for nid in order:
            s = stats(self.acts[nid], which)
            kind = g.nodes[nid].kind if nid in g.nodes else ""
            lines.append(f"{nid[:32]:<32} {kind[:14]:<14} {str(self.acts[nid].shape)[:18]:<18}"
                         + "".join(f"{s[w]:>10.4g}" for w in which))
        return "\n".join(lines)

    def __repr__(self) -> str:
        extra = f", grads={len(self.grads)}" if self.grads else ""
        return f"Trace(nodes={len(self.acts)}{extra})"

    def show(self, **kw):
        return self.net.show(*self.args, trace=self, **kw)


@dataclass
class Record:
    node: str
    calls: int = 0
    history: dict[str, list[float]] = field(default_factory=dict)
    grad_history: dict[str, list[float]] = field(default_factory=dict)
    steps: list[int] = field(default_factory=list)
    last: np.ndarray | None = None
    last_input: np.ndarray | None = None
    all: list[np.ndarray] = field(default_factory=list)

    def series(self, stat: str) -> np.ndarray:
        return np.asarray(self.history.get(stat, []))

    def histogram(self, bins: int = 40) -> dict:
        return histogram(self.last, bins) if self.last is not None else {"edges": [0, 1], "counts": [0]}


class Monitor:
    """Persistent activation monitor.

    >>> mon = net.monitor("fc*", "relu*", stats=("mean", "std", "sparsity"))
    >>> for x, y in loader:                  # call the model as usual
    ...     loss = loss_fn(model(x), y); ...
    ...     mon.step()                       # optional: mark training steps
    >>> mon["relu1"].series("sparsity")      # dead-ReLU fraction over time
    >>> mon.stop()
    """

    def __init__(self, net: "Net", node_ids: list[str], stats_: tuple = DEFAULT_STATS, keep: str = "last",
                 grads: bool = False, every: int = 1):
        if keep not in ("last", "all", "none"):
            raise ValueError("keep must be 'last', 'all' or 'none'")
        self.net = net
        self.node_ids = node_ids
        self.stats = tuple(stats_)
        self.keep = keep
        self.grads = grads
        self.every = max(1, int(every))
        self.records: dict[str, Record] = {nid: Record(nid) for nid in node_ids}
        self._handles: list[int] = []
        self.step_count = 0

    # -------------------------------------------------------------- control
    def start(self) -> "Monitor":
        if not self._handles:
            self._handles = [self.net.backend.add_tap(nid, self._observe) for nid in self.node_ids]
            self.net._monitors.append(self)
        return self

    def stop(self) -> "Monitor":
        for h in self._handles:
            self.net.backend.remove(h)
        self._handles = []
        if self in self.net._monitors:
            self.net._monitors.remove(self)
        return self

    @property
    def active(self) -> bool:
        return bool(self._handles)

    def __enter__(self) -> "Monitor":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def step(self, n: int = 1) -> None:
        self.step_count += n

    def reset(self) -> None:
        self.records = {nid: Record(nid) for nid in self.node_ids}
        self.step_count = 0

    # ------------------------------------------------------------ recording
    def _observe(self, nid: str, out: Any, inp: Any) -> None:
        rec = self.records[nid]
        rec.calls += 1
        if (rec.calls - 1) % self.every:
            return
        arr = to_numpy(out)
        for k, v in stats(arr, self.stats).items():
            rec.history.setdefault(k, []).append(v)
        rec.steps.append(self.step_count)
        if self.keep == "last":
            rec.last = np.array(arr, copy=True)
            rec.last_input = to_numpy(inp).copy() if inp is not None and hasattr(inp, "shape") else None
        elif self.keep == "all":
            rec.last = np.array(arr, copy=True)
            rec.all.append(rec.last)
        if self.grads and getattr(out, "requires_grad", False) and hasattr(out, "register_hook"):
            def on_grad(g, _rec=rec):
                for k, v in stats(to_numpy(g), ("mean", "std", "norm")).items():
                    _rec.grad_history.setdefault(k, []).append(v)
            out.register_hook(on_grad)

    # -------------------------------------------------------------- access
    def __getitem__(self, node_id: str) -> Record:
        if node_id in self.records:
            return self.records[node_id]
        from ..core.selectors import filter_paths

        m = filter_paths(node_id, list(self.records))
        if len(m) == 1:
            return self.records[m[0]]
        raise KeyError(node_id)

    def __iter__(self):
        return iter(self.records.values())

    def table(self) -> str:
        which = self.stats
        head = f"{'node':<32} {'calls':>6}" + "".join(f"{w:>11}" for w in which)
        lines = [head, "─" * len(head)]
        for nid, rec in self.records.items():
            vals = "".join(f"{rec.history[w][-1]:>11.4g}" if rec.history.get(w) else f"{'—':>11}" for w in which)
            lines.append(f"{nid[:32]:<32} {rec.calls:>6}{vals}")
        return "\n".join(lines)

    def to_dict(self, max_points: int = 400) -> dict:
        out = {}
        for nid, rec in self.records.items():
            hist = {}
            for k, v in rec.history.items():
                arr = np.asarray(v, dtype=float)
                if len(arr) > max_points:
                    idx = np.linspace(0, len(arr) - 1, max_points).astype(int)
                    arr = arr[idx]
                hist[k] = np.nan_to_num(arr).round(6).tolist()
            out[nid] = {"calls": rec.calls, "history": hist,
                        "grad_history": {k: np.nan_to_num(np.asarray(v[-max_points:])).round(6).tolist()
                                         for k, v in rec.grad_history.items()},
                        "hist": rec.histogram() if rec.last is not None else None}
        return out

    def __repr__(self) -> str:
        state = "active" if self.active else "stopped"
        return f"Monitor({len(self.node_ids)} nodes, {state}, steps={self.step_count})"

    def show(self, **kw):
        return self.net.show(monitor=self, **kw)
