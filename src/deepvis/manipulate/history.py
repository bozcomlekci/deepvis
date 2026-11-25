"""Undo / redo, snapshots and diffs for parameter edits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .edits import Edit

if TYPE_CHECKING:  # pragma: no cover
    from ..net import Net


@dataclass
class Entry:
    edit: Edit
    before: dict[str, np.ndarray]
    after: dict[str, np.ndarray]
    changed: dict[str, "np.ndarray | bool"] = field(default_factory=dict)  # per param: which entries it changed


@dataclass
class Snapshot:
    values: dict[str, np.ndarray]
    label: str = ""

    def __repr__(self) -> str:
        n = sum(v.size for v in self.values.values())
        return f"Snapshot({self.label!r}, params={len(self.values)}, numel={n:,})"


class History:
    def __init__(self, net: "Net", limit: int = 200):
        self.net = net
        self.limit = limit
        self.entries: list[Entry] = []
        self.pos = 0  # entries[:pos] are applied
        self.original: dict[str, np.ndarray] = {}  # first-seen value of every touched param
        self.dropped: dict[str, np.ndarray | bool] = {}  # what edits pushed out past ``limit`` changed

    def record(self, edit: Edit, before: dict, after: dict) -> None:
        for p, v in before.items():
            self.original.setdefault(p, v.copy())
        changed = {p: _changed(b, after[p]) if p in after and np.shape(b) == np.shape(after[p]) else True
                   for p, b in before.items()}
        del self.entries[self.pos:]
        self.entries.append(Entry(edit, before, after, changed))
        if len(self.entries) > self.limit:
            for p, m in self.entries.pop(0).changed.items():
                self.dropped[p] = _union(self.dropped.get(p, False), m)
        self.pos = len(self.entries)

    def undo(self, n: int = 1) -> list[Edit]:
        done = []
        for _ in range(n):
            if self.pos == 0:
                break
            self.pos -= 1
            e = self.entries[self.pos]
            for p, v in e.before.items():
                self.net.backend.set_param(p, self._revert(p, e.changed.get(p, True), v))
            done.append(e.edit)
        self.net._invalidate()
        return done

    def redo(self, n: int = 1) -> list[Edit]:
        done = []
        for _ in range(n):
            if self.pos >= len(self.entries):
                break
            e = self.entries[self.pos]
            for p, v in e.after.items():
                self.net.backend.set_param(p, self._revert(p, e.changed.get(p, True), v))
            self.pos += 1
            done.append(e.edit)
        self.net._invalidate()
        return done

    def reset(self) -> None:
        """Restore every touched parameter to its original value (recorded as an undoable edit)."""
        changed = self.restore_values()
        if changed:
            self.net.apply_edit(Edit("load", list(changed), args={"values": changed}, label="reset to original"))

    def restore_values(self, paths=None) -> dict[str, np.ndarray]:
        """``{param: value}`` putting back the original of every entry the applied edits changed, for the params
        (default: all touched) that would change. Weights that moved since — training between an edit and
        the reset — keep their progress everywhere else."""
        out = {}
        for p in (self.original if paths is None else [q for q in paths if q in self.original]):
            cur, orig = self.net.backend.get_param(p), self.original[p]
            mask = self.dropped.get(p, False)
            for e in self.entries[: self.pos]:
                if p in e.changed:
                    mask = _union(mask, e.changed[p])
            fits = np.shape(cur) == np.shape(orig) and np.shape(mask) in ((), np.shape(cur))
            value = np.where(mask, orig, cur) if fits else orig
            if not np.array_equal(cur, value, equal_nan=value.dtype.kind in "fc"):
                out[p] = value
        return out

    def clear(self) -> None:
        """Forget every edit (the current weights become the new original), e.g. after re-initialising."""
        self.rollback(([], {}, {}))

    def checkpoint(self) -> tuple:
        """The applied edits as they stand, to :meth:`rollback` to when the weights go back to this point."""
        return list(self.entries[: self.pos]), dict(self.original), dict(self.dropped)

    def rollback(self, checkpoint: tuple) -> None:
        entries, original, dropped = checkpoint
        self.entries, self.original, self.dropped = list(entries), dict(original), dict(dropped)
        self.pos = len(self.entries)

    def _revert(self, p: str, mask: "np.ndarray | bool", value: np.ndarray) -> np.ndarray:
        """``value`` written back only where the edit changed something: weights that moved since — training
        between the edit and the undo — keep their progress everywhere else. (Without training, that's ``value``.)"""
        if mask is True:
            return value
        cur = self.net.backend.get_param(p)
        if not np.shape(cur) == np.shape(mask) == np.shape(value):
            return value
        return np.where(mask, value, cur)

    @property
    def applied(self) -> list[Edit]:
        return [e.edit for e in self.entries[: self.pos]]

    def export(self) -> list[dict]:
        return [e.to_dict() for e in self.applied]

    def __len__(self) -> int:
        return self.pos

    def __repr__(self) -> str:
        lines = [f"History({self.pos} applied, {len(self.entries) - self.pos} redoable)"]
        for i, e in enumerate(self.entries):
            mark = "✓" if i < self.pos else "·"
            lines.append(f"  {mark} {i + 1:>3}. {e.edit.describe()}")
        return "\n".join(lines)


def _union(a: "np.ndarray | bool", b: "np.ndarray | bool") -> "np.ndarray | bool":
    """a | b for change masks, where True means the whole parameter (its shape changed)."""
    if a is True or b is True:
        return True
    if a is False:
        return b
    return (a | b) if np.shape(a) == np.shape(b) else True


def _changed(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise a != b, with NaN equal to NaN."""
    a, b = np.asarray(a), np.asarray(b)
    ne = a != b
    if a.dtype.kind in "fc" and b.dtype.kind in "fc":
        ne &= ~(np.isnan(a) & np.isnan(b))
    return ne


def diff(net: "Net", reference: dict[str, np.ndarray] | Snapshot | None = None) -> list[dict]:
    """Per-parameter change report against a snapshot (default: original values)."""
    ref = reference.values if isinstance(reference, Snapshot) else reference
    if ref is None:
        ref = net.history.original
    rows = []
    for p, old in ref.items():
        new = net.backend.get_param(p)
        d = new.astype(np.float64) - old.astype(np.float64)
        if not np.any(d):
            continue
        rows.append({
            "param": p,
            "delta_norm": float(np.linalg.norm(d)),
            "rel_change": float(np.linalg.norm(d) / (np.linalg.norm(old) + 1e-12)),
            "frac_changed": float(np.mean(d != 0)),
            "max_abs_delta": float(np.max(np.abs(d))),
        })
    return rows


def format_diff(rows: list[dict]) -> str:
    if not rows:
        return "no parameter changes"
    head = f"{'param':<40} {'‖Δ‖':>10} {'‖Δ‖/‖w‖':>10} {'changed':>8} {'max|Δ|':>10}"
    lines = [head, "─" * len(head)]
    for r in rows:
        lines.append(f"{r['param']:<40} {r['delta_norm']:>10.4g} {r['rel_change']:>10.3%} "
                     f"{r['frac_changed']:>8.1%} {r['max_abs_delta']:>10.4g}")
    return "\n".join(lines)
