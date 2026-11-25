"""Watching a network learn: scalar metrics plus per-layer weight dynamics across training steps.

>>> log = dv.TrainingLog(net)
>>> for step, (x, y) in enumerate(loader):
...     loss = train_step(x, y)
...     log.record(step, loss=float(loss))       # metrics every call; layer snapshots are throttled
>>> net.export_html("run.html", x, training=log)

Per layer (every node that owns parameters) each snapshot stores

* ``norm``   — ‖θ‖ over the layer's parameters,
* ``update`` — ‖θ_t − θ_prev‖ / ‖θ_prev‖ per step: the relative change since the previous snapshot divided by
  the steps in between, so it doesn't depend on how often you snapshot (healthy training usually sits
  around 1e-3; much larger means unstable, much smaller means stuck),
* ``grad``   — ‖∇θ‖ when gradients are available (torch ``.grad``, or a grads tree passed in),
* ``act``    — the standard deviation of the layer's output when a trace is given.

and, for replaying training in the page, up to ``frames`` snapshots also keep a small picture of each layer's
weights (``frame_side``² cells, 8-bit) and the network's output at that step.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from ..net import Net
    from .recorder import Trace

LAYER_STATS = ("update", "norm", "grad", "act")


def _finite(v) -> float | None:
    v = float(v)
    return v if math.isfinite(v) else None


def _bucket(steps: list, values: list, n: int) -> tuple[list, list]:
    """Down-sample to ``n`` points by averaging buckets (keeps a noisy loss curve's shape)."""
    if len(values) <= n:
        return list(steps), [_finite(v) if v is not None else None for v in values]
    idx = np.array_split(np.arange(len(values)), n)
    vals = np.array([np.nan if v is None else float(v) for v in values], dtype=np.float64)
    out_s, out_v = [], []
    for b in idx:
        chunk = vals[b]
        ok = chunk[np.isfinite(chunk)]
        out_s.append(steps[b[-1]])
        out_v.append(float(ok.mean()) if ok.size else None)
    return out_s, out_v


class TrainingLog:
    """Metrics and layer dynamics recorded during training (see the module docstring)."""

    def __init__(self, net: "Net", every: float = 0.0, every_steps: int | None = None, max_snapshots: int = 2000,
                 frames: int = 48, frame_side: int = 16):
        """When :meth:`record` takes a layer snapshot (metrics are always kept): every ``every_steps`` steps if
        given, otherwise at most every ``every`` seconds. ``frames`` snapshots (evenly thinned) keep weight
        pictures and outputs for the page's replay; 0 turns that off."""
        self.net = net
        self.every = float(every)
        self.every_steps = int(every_steps) if every_steps else None
        self.max_snapshots = max_snapshots
        self.steps: list[int] = []
        self.metrics: dict[str, list] = {}
        self.snap_steps: list[int] = []
        self.layers: dict[str, dict[str, list]] = {}
        self._prev: dict[str, np.ndarray] = {}
        self._last_snap = -1e18
        self._count = 0
        self._groups = self._param_groups()
        self.frames = int(frames)
        self.frame_side = int(frame_side)
        self._frames: list[dict] = []

    def clear(self) -> None:
        """Forget everything recorded (settings stay): training starts over."""
        self.steps, self.metrics, self.snap_steps, self.layers = [], {}, [], {}
        self._prev, self._frames = {}, []
        self._last_snap = -1e18
        self._count = 0

    # ------------------------------------------------------------------ recording
    def _param_groups(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        self._main: dict[str, str] = {}  # each layer's largest parameter: the one replay frames picture
        seen: set[str] = set()
        for n in self.net.graph.param_nodes():
            infos = [i for i in n.params.values()
                     if i.trainable and i.path not in seen and np.prod(i.shape or (1,)) > 0]
            if not infos:
                continue
            seen.update(i.path for i in infos)
            groups[n.id] = [i.path for i in infos]
            self._main[n.id] = max(infos, key=lambda i: int(np.prod(i.shape or (1,)))).path
        return groups

    def record(self, step: int | None = None, grads=None, trace: "Trace | None" = None, snapshot: bool | None = None,
               **metrics) -> bool:
        """Log ``metrics`` for ``step``; also take a layer snapshot if ``every`` seconds passed (or ``snapshot=True``).

        Returns whether a snapshot was taken.
        """
        step = self._count if step is None else int(step)
        self._count += 1
        if metrics or not self.steps or step != self.steps[-1]:  # a bare refresh of the same step logs nothing new
            self.log(step, **metrics)
        if (snapshot or (snapshot is None and self.due(step))) and not (self.snap_steps and step == self.snap_steps[-1]):
            self.snapshot(step, grads=grads, trace=trace)
            return True
        return False

    def due(self, step: int) -> bool:
        """Whether a layer snapshot is due at ``step`` under this log's schedule."""
        if self.every_steps:
            return not self.snap_steps or step - self.snap_steps[-1] >= self.every_steps
        return (time.monotonic() - self._last_snap) >= self.every

    def log(self, step: int, **metrics) -> None:
        """Scalars only (cheap): ``log(step, loss=0.3, acc=0.9)``. Logging the latest step again adds to it."""
        step = int(step)
        if not self.steps or step != self.steps[-1]:
            self.steps.append(step)
        n = len(self.steps)
        for k, v in metrics.items():
            try:
                val = float(v.item() if hasattr(v, "item") else v)
            except (TypeError, ValueError):
                continue
            series = self.metrics.setdefault(k, [])
            series.extend([None] * (n - len(series)))
            series[-1] = val
        for series in self.metrics.values():
            series.extend([None] * (n - len(series)))

    def snapshot(self, step: int | None = None, grads=None, trace: "Trace | None" = None) -> None:
        """Per-layer ‖θ‖, relative update since the last snapshot, ‖∇θ‖ and output spread."""
        b = self.net.backend
        step = (self.steps[-1] if self.steps else 0) if step is None else int(step)
        g = b.grads(grads)
        first = not self.snap_steps
        gap = max(1, step - self.snap_steps[-1]) if self.snap_steps else 1
        self.snap_steps.append(step)
        for nid, paths in self._groups.items():
            rec = self.layers.setdefault(nid, {k: [] for k in LAYER_STATS})
            sq = dsq = psq = gsq = 0.0
            has_grad = False
            for p in paths:
                w = b.get_param(p).astype(np.float64, copy=False)
                sq += float(np.sum(w * w))
                prev = self._prev.get(p)
                if prev is not None and prev.shape == w.shape:
                    d = w - prev
                    dsq += float(np.sum(d * d))
                    psq += float(np.sum(prev * prev))
                self._prev[p] = w.astype(np.float32)
                if p in g:
                    has_grad = True
                    gg = np.asarray(g[p], dtype=np.float64)
                    gsq += float(np.sum(gg * gg))
            rec["norm"].append(_finite(math.sqrt(sq)))
            rec["update"].append(None if first or psq == 0 else _finite(math.sqrt(dsq) / math.sqrt(psq) / gap))
            rec["grad"].append(_finite(math.sqrt(gsq)) if has_grad else None)
            a = trace.acts.get(nid) if trace is not None else None
            rec["act"].append(_finite(np.std(a)) if a is not None and np.size(a) else None)
        self._last_snap = time.monotonic()
        if self.frames > 0:
            self._keep_frame(step, trace)
        if len(self.snap_steps) > self.max_snapshots:  # thin out the older half
            keep = list(range(0, len(self.snap_steps) // 2, 2)) + list(range(len(self.snap_steps) // 2,
                                                                          len(self.snap_steps)))
            self.snap_steps = [self.snap_steps[i] for i in keep]
            for rec in self.layers.values():
                for k in rec:
                    rec[k] = [rec[k][i] for i in keep]

    def _keep_frame(self, step: int, trace) -> None:
        """A small picture of every layer's main weight, and the output, for the replay slider."""
        from ..core.tensor import first_array, to_numpy
        from ..viz.html.payload import heat

        b = self.net.backend
        weights = {}
        for nid in self._groups:
            w = np.asarray(b.get_param(self._main[nid]), dtype=np.float64)
            w2 = w.reshape(1, -1) if w.ndim < 2 else w.reshape(w.shape[0], -1)
            weights[nid] = heat(w2, True, self.frame_side, quantize=True)
        out = None
        if trace is not None and trace.output is not None:
            a = first_array(trace.output)
            if a is not None:
                a = to_numpy(a)
                out = a.astype(np.float32) if a.size <= 200_000 else None
        self._frames.append({"step": int(step), "weights": weights, "output": out})
        if len(self._frames) > 2 * self.frames:  # keep the first, thin the rest evenly
            self._frames = self._frames[:1] + self._frames[2::2]

    def frame_list(self, labels=None, tokenizer=None, sample: int = 0) -> list[dict]:
        """The kept frames, at most ``frames`` of them evenly spaced, as page payload."""
        if not self._frames:
            return []
        from ..viz.html.payload import _output_payload

        idx = np.unique(np.linspace(0, len(self._frames) - 1, min(self.frames, len(self._frames))).round().astype(int))
        out = []
        for i in idx:
            f = self._frames[int(i)]
            o = None
            if f["output"] is not None:
                try:
                    o = _output_payload(f["output"], None, sample, labels, tokenizer)
                except Exception:  # noqa: BLE001 - an output we can't draw: the weights still replay
                    o = None
            out.append({"step": f["step"], "weights": f["weights"], "output": o})
        return out

    # ------------------------------------------------------------------ export
    def __len__(self) -> int:
        return len(self.steps)

    @property
    def last_step(self) -> int | None:
        return self.steps[-1] if self.steps else None

    def history(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """``(steps, values)`` of a metric, or of a layer stat as ``"<node>:<stat>"``."""
        if name in self.metrics:
            return np.asarray(self.steps), np.array([np.nan if v is None else v for v in self.metrics[name]])
        nid, _, stat = name.rpartition(":")
        vals = self.layers[nid][stat]
        return np.asarray(self.snap_steps), np.array([np.nan if v is None else v for v in vals])

    def to_dict(self, max_points: int = 600, max_snapshots: int = 240, labels=None, tokenizer=None,
                sample: int = 0) -> dict:
        m_steps = _bucket(self.steps, [0.0] * len(self.steps), max_points)[0]
        metrics = {k: _bucket(self.steps, v, max_points)[1] for k, v in self.metrics.items()}
        s_steps = _bucket(self.snap_steps, [0.0] * len(self.snap_steps), max_snapshots)[0]
        g = self.net.graph
        layers = []
        for nid, rec in self.layers.items():
            n = g.nodes.get(nid)
            stats = {k: _bucket(self.snap_steps, vals, max_snapshots)[1] for k, vals in rec.items()
                     if any(x is not None for x in vals)}
            layers.append({"id": nid, "kind": n.kind if n else "", "stats": stats})
        return {"steps": m_steps, "metrics": metrics, "snapSteps": s_steps, "layers": layers,
                "lastStep": self.last_step, "count": len(self.steps),
                "frames": self.frame_list(labels, tokenizer, sample)}

    def __repr__(self) -> str:
        last = {k: v[-1] for k, v in self.metrics.items() if v and v[-1] is not None}
        return (f"TrainingLog(steps={len(self.steps)}, snapshots={len(self.snap_steps)}, "
                f"last={ {k: round(v, 4) for k, v in last.items()} })")
