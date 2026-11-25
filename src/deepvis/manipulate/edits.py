"""Serializable edit records and their application.

An :class:`Edit` is a small JSON-able record::

    {"op": "prune", "target": "fc1", "args": {"amount": 0.3}}
    {"op": "scale", "target": "layers.*.weight", "index": "[:, :16]", "args": {"factor": 0.5}}
    {"op": "zero",  "target": "fc2.weight", "where": "abs(w) < 0.01"}

Targets may be parameter paths, node paths (op applies to that node's parameters,
defaulting to the roles the op declares) or glob patterns over either.
Every edit goes through :meth:`Net.apply_edit`, which records undo information.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..core.selectors import filter_paths, format_index, is_pattern, normalize_index
from .ops import OPS, eval_where

if TYPE_CHECKING:  # pragma: no cover
    from ..net import Net


@dataclass
class Edit:
    op: str
    target: str | list[str]
    index: Any = None
    args: dict = field(default_factory=dict)
    where: Any = None
    params: list[str] | None = None  # restrict node targets to these roles / local names
    label: str | None = None

    @property
    def replayable(self) -> bool:
        if callable(self.where) or (self.index is not None and format_index(self.index) is None):
            return False
        return all(_jsonable_arg(v) for v in self.args.values())

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"op": self.op, "target": self.target}
        if self.index is not None:
            d["index"] = format_index(self.index) or "<unserializable>"
        if self.args:
            d["args"] = {k: _serialize_arg(v) for k, v in self.args.items()}
        if self.where is not None:
            d["where"] = self.where if isinstance(self.where, str) else "<callable>"
        if self.params:
            d["params"] = list(self.params)
        if self.label:
            d["label"] = self.label
        if not self.replayable:
            d["replayable"] = False
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Edit":
        if d.get("replayable") is False:
            raise ValueError(f"edit {d.get('op')!r} on {d.get('target')!r} is not replayable")
        args = dict(d.get("args", {}))
        if d["op"] == "group":
            args["edits"] = [cls.from_dict(e) for e in args.get("edits", [])]
        return cls(op=d["op"], target=d["target"], index=d.get("index"), args=args, where=d.get("where"),
                   params=d.get("params"), label=d.get("label"))

    def describe(self) -> str:
        if self.label:
            return self.label
        tgt = self.target if isinstance(self.target, str) else ", ".join(self.target[:3]) + (
            "…" if len(self.target) > 3 else "")
        idx = format_index(self.index) if self.index is not None else ""
        a = ", ".join(f"{k}={_short(v)}" for k, v in self.args.items())
        w = f" where {self.where}" if isinstance(self.where, str) else ""
        return f"{self.op}({a}) → {tgt}{idx or ''}{w}"


def _short(v) -> str:
    if isinstance(v, np.ndarray):
        return f"array{v.shape}"
    if callable(v):
        return getattr(v, "__name__", "fn")
    s = repr(v)
    return s if len(s) < 30 else s[:27] + "…"


MAX_SERIALIZED = 250_000  # elements; bigger arrays make an edit non-replayable


def _jsonable_arg(v) -> bool:
    if isinstance(v, list) and v and all(isinstance(e, Edit) for e in v):
        return all(e.replayable for e in v)
    if isinstance(v, dict):
        return all(_jsonable_arg(x) for x in v.values())
    if callable(v) or hasattr(v, "backend"):
        return False
    if isinstance(v, np.ndarray):
        return v.size <= MAX_SERIALIZED
    return True


def _serialize_arg(v):
    if isinstance(v, list) and v and all(isinstance(e, Edit) for e in v):
        return [e.to_dict() for e in v]
    if isinstance(v, dict):
        return {k: _serialize_arg(x) for k, x in v.items()}
    if hasattr(v, "backend"):
        return f"<net {getattr(v, 'name', '')}>"
    if isinstance(v, np.ndarray):
        return v.tolist() if v.size <= MAX_SERIALIZED else "<array>"
    if isinstance(v, np.generic):
        return v.item()
    if callable(v):
        return f"<callable {getattr(v, '__name__', '')}>"
    return v


# ------------------------------------------------------------------ targets
def resolve_param_targets(net: "Net", edit: Edit) -> list[str]:
    targets = [edit.target] if isinstance(edit.target, str) else list(edit.target)
    params = net.backend.param_paths()
    pset = set(params)
    graph = net.graph
    out: list[str] = []
    opdef = OPS.get(edit.op)
    for t in targets:
        if t in pset:
            out.append(t)
            continue
        node_ids = [t] if t in graph.nodes else (filter_paths(t, list(graph.nodes)) if is_pattern(t) else [])
        node_ids = [n for n in node_ids if graph.nodes[n].params]
        if node_ids:
            for nid in node_ids:
                node = graph.nodes[nid]
                if node.module and node.module != nid:
                    continue  # shared call instance; owner node covers it
                wanted = edit.params or (list(opdef.default_roles) if opdef and opdef.default_roles else None)
                for local, info in node.params.items():
                    if wanted is None or local in wanted or any(
                            r in wanted and node.roles[r].name == local for r in node.roles):
                        out.append(info.path)
            continue
        if is_pattern(t):
            matched = filter_paths(t, params)
            if matched:
                out.extend(matched)
                continue
        raise KeyError(f"edit target {t!r} matches no parameter or node")
    return list(dict.fromkeys(out))


def compute_edit(net: "Net", edit: Edit, overlay: dict | None = None) -> dict[str, np.ndarray]:
    """Return ``{param_path: new_full_value}`` without applying anything.

    ``overlay`` holds pending values that take precedence over the backend (used by groups).
    """
    overlay = overlay if overlay is not None else {}
    if edit.op == "group":
        pending = dict(overlay)
        changed: dict[str, np.ndarray] = {}
        for sub in edit.args["edits"]:
            out = compute_edit(net, sub, pending)
            pending.update(out)
            changed.update(out)
        return changed
    if edit.op in COMPOUND:
        return COMPOUND[edit.op](net, edit, lambda p: overlay[p].copy() if p in overlay else net.backend.get_param(p))
    if edit.op not in OPS:
        raise KeyError(f"unknown edit op {edit.op!r}; available: {sorted(OPS) + sorted(COMPOUND)}")
    fn = OPS[edit.op].fn
    idx = normalize_index(edit.index)
    new: dict[str, np.ndarray] = {}
    for path in resolve_param_targets(net, edit):
        full = overlay[path].copy() if path in overlay else net.backend.get_param(path)
        region = full if idx is None else full[idx]
        region = np.asarray(region)
        result = np.asarray(fn(region.copy(), **edit.args), dtype=full.dtype)
        if edit.where is not None:
            mask = eval_where(edit.where, region)
            result = np.where(mask, result, region)
        out = full.copy()
        if idx is None:
            out[...] = result
        else:
            out[idx] = result
        new[path] = out
    return new


# ------------------------------------------------------- compound (multi-param) ops
def _permute_neurons(net: "Net", edit: Edit, get) -> dict[str, np.ndarray]:
    """Permute a layer's output neurons and the matching inputs of the layers that consume them.

    The network function is unchanged (up to float rounding) — a classic symmetry demo.
    """
    from ..core.kinds import ACTIVATION_KINDS, NORM_KINDS

    g = net.graph
    nid = edit.target if isinstance(edit.target, str) else edit.target[0]
    perm = np.asarray(edit.args["perm"])
    node = g.nodes[nid]
    new: dict[str, np.ndarray] = {}

    def permute_out(n):
        for role, r in n.roles.items():
            if r.out_axis is None:
                continue
            info = n.params[r.name]
            w = new[info.path] if info.path in new else get(info.path)
            new[info.path] = np.take(w, perm, axis=r.out_axis)

    permute_out(node)
    frontier = list(g.succ(nid))
    seen = set()
    while frontier:
        s = frontier.pop()
        if s in seen:
            continue
        seen.add(s)
        sn = g.nodes[s]
        if sn.kind in ACTIVATION_KINDS - {"Softmax", "LogSoftmax"} or sn.kind in ("Dropout", "Identity"):
            frontier.extend(g.succ(s))
        elif sn.kind in NORM_KINDS and sn.kind != "GroupNorm":
            permute_out(sn)
            frontier.extend(g.succ(s))
        elif "weight" in sn.roles and sn.roles["weight"].in_axis is not None:
            r = sn.roles["weight"]
            info = sn.params[r.name]
            w = new[info.path] if info.path in new else get(info.path)
            new[info.path] = np.take(w, perm, axis=r.in_axis)
        elif sn.op == "output":
            raise ValueError(f"{nid!r} feeds the network output directly; permuting it changes the output order")
        else:
            raise ValueError(
                f"cannot propagate neuron permutation through {s!r} ({sn.kind}); "
                "supported: elementwise activations, dropout, norms, Linear/Conv consumers")
    return new


def _interpolate(net: "Net", edit: Edit, get) -> dict[str, np.ndarray]:
    other = edit.args["other"]
    alpha = float(edit.args.get("alpha", 0.5))
    pattern = edit.args.get("select", "**")
    src = other.backend if hasattr(other, "backend") else other
    new = {}
    for p in filter_paths(pattern, net.backend.param_paths()):
        a = get(p)
        b = src.get_param(p)
        new[p] = ((1 - alpha) * a + alpha * b).astype(a.dtype)
    return new


def _transplant(net: "Net", edit: Edit, get) -> dict[str, np.ndarray]:
    other = edit.args["other"]
    mapping = edit.args.get("mapping")  # {dst_path: src_path} or None for same names
    src = other.backend if hasattr(other, "backend") else other
    dst_paths = resolve_param_targets(net, Edit("set", edit.target, params=edit.params))
    new = {}
    for p in dst_paths:
        sp = mapping.get(p, p) if mapping else p
        v = src.get_param(sp)
        cur = get(p)
        if v.shape != cur.shape:
            raise ValueError(f"transplant shape mismatch {sp} {v.shape} -> {p} {cur.shape}")
        new[p] = v.astype(cur.dtype)
    return new


def _load(net: "Net", edit: Edit, get) -> dict[str, np.ndarray]:
    values = edit.args["values"]
    return {p: np.asarray(v).astype(get(p).dtype) for p, v in values.items()}


COMPOUND = {
    "permute_neurons": _permute_neurons,
    "interpolate": _interpolate,
    "transplant": _transplant,
    "load": _load,
}


def save_edits(edits: list[Edit], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([e.to_dict() for e in edits], f, indent=2)


def load_edits(path_or_json) -> list[Edit]:
    if isinstance(path_or_json, str) and path_or_json.strip().startswith("["):
        data = json.loads(path_or_json)
    elif isinstance(path_or_json, (list, tuple)):
        data = path_or_json
    else:
        with open(path_or_json, encoding="utf-8") as f:
            data = json.load(f)
    return [d if isinstance(d, Edit) else Edit.from_dict(d) for d in data]
