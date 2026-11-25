"""A network over a dataset, not one input: ``net.profile(data, labels)``.

Streams the data through the network once and keeps, per layer:

* per-unit statistics (a unit is a channel / feature: Welford mean and spread of its per-sample activation), how often
  each unit is active, and its **dead units** (never active after a ReLU-like layer, or constant);
* the **inputs that excite each unit most** (top-k samples, with where in the input it peaked);
* a **2-D map of the samples** (PCA of the layer's pooled representation; UMAP if ``umap-learn`` is installed and
  asked for) — do the classes separate here?
* **how similar layers are** (linear CKA between their representations of the same samples).

``net.export_html(path, x, dataset=profile)`` adds a *Dataset* tab, and the node panel shows each layer's most
exciting inputs. The map and CKA use each sample's whole activation, count-sketched to 256 numbers when it is larger
(a random signed hashing that keeps inner products, so both stay faithful without storing a projection matrix).
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from ..core.tensor import first_array, to_numpy

PROJ = 256  # wide layers: count-sketched to this many dimensions for the map and CKA


class _Sketch:
    """Count sketch of a D-dimensional vector into PROJ: element i adds ±x[i] to bucket h(i). The same D always gets
    the same hashing, so a convolution's output and its flattened copy sketch identically."""

    _cache: dict[int, "_Sketch"] = {}

    def __init__(self, d: int, seed: int = 0):
        rng = np.random.default_rng(seed + d)
        h = rng.integers(0, PROJ, d)
        self.order = np.argsort(h, kind="stable")
        self.sign = rng.choice(np.array([-1.0, 1.0], np.float32), d)[self.order]
        sorted_h = h[self.order]
        self.starts = np.searchsorted(sorted_h, np.arange(PROJ))
        self.present = np.zeros(PROJ, bool)
        self.present[np.unique(sorted_h)] = True

    @classmethod
    def of(cls, d: int) -> "_Sketch":
        if d not in cls._cache:
            cls._cache[d] = cls(d)
        return cls._cache[d]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y = x[:, self.order] * self.sign
        sums = np.add.reduceat(y, np.minimum(self.starts, y.shape[1] - 1), axis=1)
        return np.where(self.present, sums, 0.0).astype(np.float32)


class DatasetProfile:
    """What :func:`profile` found; ``to_dict()`` is what the page draws."""

    def __init__(self, layers: list[str], top_k: int):
        self.layers = layers
        self.top_k = top_k
        self.n = 0
        self.labels: list | None = None
        self.pred: list | None = None
        self.units: dict[str, dict] = {}       # layer → mean, std, active rate, dead mask …
        self.top: dict[str, dict] = {}         # layer → values (C, k), index (C, k), pos (C, k)
        self.embed: dict[str, np.ndarray] = {}  # layer → (n_embed, 2)
        self.embed_index: list[int] = []
        self.cka: np.ndarray | None = None
        self.thumbs: dict[int, np.ndarray] = {}
        self.tokens: np.ndarray | None = None  # token-id inputs: (n, T), for the snippets around a unit's peak

    def __repr__(self) -> str:
        dead = sum(u["dead"] for u in self.units.values())
        return f"DatasetProfile({self.n} samples, {len(self.layers)} layers, {dead} dead units)"

    def dead_units(self) -> dict[str, list[int]]:
        """Units that never fire (after a ReLU-like layer) or never change, per layer."""
        return {k: np.flatnonzero(u["dead_mask"]).tolist() for k, u in self.units.items() if u["dead"]}

    def top_inputs(self, layer: str, unit: int) -> list[tuple[int, float]]:
        """(sample index, activation) of the inputs that excite ``unit`` of ``layer`` most, best first."""
        t = self.top[layer]
        order = np.argsort(-t["values"][unit])
        return [(int(t["index"][unit, i]), float(t["values"][unit, i])) for i in order if t["index"][unit, i] >= 0]

    def to_dict(self, max_units: int = 24, max_bars: int = 512, tokenizer=None, class_names=None) -> dict:
        from ..viz.glyphs import png_uri

        units, top, used = {}, {}, set()
        for L in self.layers:
            u = self.units[L]
            C = len(u["mean"])
            idx = np.arange(C) if C <= max_bars else np.linspace(0, C - 1, max_bars).astype(int)
            units[L] = {"C": C, "dead": int(u["dead"]), "nonneg": bool(u["nonneg"]),
                        "mean": _q(u["mean"][idx]), "std": _q(u["std"][idx]), "active": _q(u["active"][idx]),
                        "shown": idx.tolist() if C > max_bars else None}
            t = self.top[L]
            peak = t["values"].max(axis=1)
            best = np.argsort(-np.where(np.isfinite(peak), peak, -np.inf))[:max_units]
            rows = []
            for unit in best:
                items = []
                for i in np.argsort(-t["values"][unit]):
                    s = int(t["index"][unit, i])
                    if s < 0:
                        continue
                    items.append([s, round(float(t["values"][unit, i]), 4), int(t["pos"][unit, i])])
                    used.add(s)
                rows.append({"unit": int(unit), "items": items, "snippets": self._snippets(items, tokenizer)})
            top[L] = rows
        used.update(self.embed_index)
        thumbs = {str(s): png_uri(self.thumbs[s]) for s in sorted(used) if s in self.thumbs}
        out = {"n": self.n, "layers": self.layers, "units": units, "top": top, "thumbs": thumbs,
               "labels": self.labels, "pred": self.pred, "classNames": list(class_names) if class_names else None,
               "embed": {L: {"xy": np.round(xy, 3).tolist()} for L, xy in self.embed.items()},
               "embedIndex": self.embed_index,
               "cka": {"layers": self.layers, "m": np.round(self.cka, 3).tolist()} if self.cka is not None else None}
        return out

    def _snippets(self, items, tokenizer) -> list[str] | None:
        if self.tokens is None:
            return None
        out = []
        for s, _, pos in items:
            if s >= len(self.tokens):
                out.append("")
                continue
            row = self.tokens[s]
            lo, hi = max(0, pos - 3), min(len(row), pos + 4)
            if tokenizer is not None:
                try:
                    out.append("".join(f"⟦{tokenizer.decode([int(t)])}⟧" if k == pos else tokenizer.decode([int(t)])
                                       for k, t in zip(range(lo, hi), row[lo:hi])))
                    continue
                except Exception:  # noqa: BLE001 - ids then
                    pass
            out.append(" ".join(f"[{int(t)}]" if k == pos else str(int(t)) for k, t in zip(range(lo, hi), row[lo:hi])))
        return out


def _q(a) -> list:
    return [None if not np.isfinite(v) else round(float(v), 4) for v in np.asarray(a, dtype=np.float64)]


# ------------------------------------------------------------------------------------------- streaming
def _batches(data, labels, batch_size: int, max_samples: int) -> Iterable[tuple[tuple, Any]]:
    """(positional inputs, labels or None) per batch: arrays are sliced, iterables (a DataLoader, a generator, a
    list of batches) are taken as they come — ``x``, ``(x, y)`` or a dict with ``labels`` / ``label``."""
    def sized(a):
        return hasattr(a, "shape") and len(getattr(a, "shape", ())) >= 1

    if isinstance(data, tuple) and len(data) == 2 and sized(data[0]) and sized(data[1]) and labels is None \
            and len(data[0]) == len(data[1]):
        data, labels = data  # (X, y)
    if sized(data):
        n = min(len(data), max_samples)
        for i in range(0, n, batch_size):
            yield (data[i:min(i + batch_size, n)],), (None if labels is None else labels[i:min(i + batch_size, n)])
        return
    seen = 0
    for item in data:
        y = None
        if isinstance(item, dict) or (hasattr(item, "keys") and not hasattr(item, "shape")):
            item = dict(item)
            y = item.pop("labels", item.pop("label", None))
            args = (item,)
        elif isinstance(item, (tuple, list)) and len(item) == 2 and sized(item[1]) and len(np.shape(item[1])) == 1:
            args, y = (item[0],), item[1]  # (inputs, class labels), as a DataLoader yields them
        elif isinstance(item, (tuple, list)):
            args = tuple(item)
        else:
            args = (item,)
        b = _batch_len(args)
        if seen + b > max_samples:
            keep = max_samples - seen
            args = tuple(_cut(a, keep) for a in args)
            y = None if y is None else y[:keep]
            b = keep
        if b <= 0:
            return
        yield args, y
        seen += b
        if seen >= max_samples:
            return


def _cut(a, k):
    if isinstance(a, dict):
        return {key: v[:k] for key, v in a.items()}
    return a[:k]


def _batch_len(args) -> int:
    a = args[0]
    if isinstance(a, dict):
        a = next(iter(a.values()))
    return int(np.shape(to_numpy(a))[0]) if a is not None else 0


def _default_layers(net) -> list[str]:
    skip = {"Dropout", "Identity"}
    g = net.graph
    ids = [i for i in g.topo_order() if g.nodes[i].op == "module" and g.nodes[i].kind not in skip]
    if len(ids) > 96:  # very deep networks: evenly spaced, the last one kept
        keep = set(np.linspace(0, len(ids) - 1, 96).astype(int))
        ids = [i for k, i in enumerate(ids) if k in keep]
    return ids


def profile(net, data, labels=None, *, max_samples: int = 2048, layers=None, top_k: int = 8, batch_size: int = 64,
            embed_samples: int = 512, embed: str = "pca", seed: int = 0) -> DatasetProfile:
    """Run ``data`` through ``net`` and profile every layer (see the module docstring)."""
    from ..errors import OptionError

    if net.framework == "equinox":
        raise OptionError("profile() runs batches; an equinox model takes one example at a time.",
                          "wrap jax.vmap(model) or profile the examples with net.trace one by one.")
    if not net.backend.has_data:
        raise OptionError("this model is on the meta device: it has no numbers to run.", "load real weights first.")
    ids = list(layers) if layers is not None else _default_layers(net)
    if not ids:
        raise OptionError("no layers to profile.", "pass layers=[...] with node ids from net.graph.")
    prof = DatasetProfile(ids, top_k)
    acc: dict[str, dict] = {}
    reps: dict[str, list] = {L: [] for L in ids}
    labels_out: list = []
    pred_out: list = []
    tokens: list = []
    n = 0
    for args, y in _batches(data, labels, batch_size, max_samples):
        tr = net.trace(*args, record=ids)
        b = _batch_len(args)
        for L in ids:
            a = tr.acts.get(L)
            if a is None:
                continue
            node = net.graph.nodes[L]
            arr = np.asarray(to_numpy(a), dtype=np.float32)
            if arr.ndim == 0 or arr.shape[0] != b:
                continue  # not per sample (a position table computed once): nothing to profile over the data
            m, mx, arg, mn = _per_unit(arr, net.backend.feature_axis(node), b)
            st = acc.get(L)
            if st is None:
                C = m.shape[1]
                st = acc[L] = {"n": 0, "mean": np.zeros(C), "M2": np.zeros(C), "active": np.zeros(C),
                               "max": np.full(C, -np.inf), "min": np.full(C, np.inf),
                               "tv": np.full((C, top_k), -np.inf, np.float32),
                               "ti": np.full((C, top_k), -1, np.int64), "tp": np.zeros((C, top_k), np.int64)}
            _welford(st, m)
            st["active"] += (mx > 0).sum(0)
            st["max"] = np.maximum(st["max"], mx.max(0))
            st["min"] = np.minimum(st["min"], mn.min(0))  # over every sample and position
            _merge_top(st, mx, arg, n)
            have = sum(len(r) for r in reps[L])
            if have < embed_samples:  # the whole activation of each sample, for the map and CKA
                flat = np.nan_to_num(arr.reshape(arr.shape[0], -1)[: embed_samples - have] if arr.ndim > 1
                                     else arr.reshape(-1, 1)[: embed_samples - have])
                reps[L].append(_Sketch.of(flat.shape[1])(flat) if flat.shape[1] > PROJ else flat)
        if y is not None:
            labels_out.extend(_labels_list(y))
        out = first_array(tr.output)
        if out is not None:
            o = np.asarray(to_numpy(out))
            if o.ndim == 2 and o.shape[0] == b and 2 <= o.shape[1] <= 1000:
                pred_out.extend(int(v) for v in o.argmax(1))
        _keep_inputs(prof, args, n, tokens)
        n += b
    prof.n = n
    if not acc:
        raise OptionError("none of the profiled layers produced an activation.",
                          "pass layers=[...] with node ids that run, e.g. from net.summary().")
    prof.layers = [L for L in ids if L in acc]
    for L in prof.layers:
        st = acc[L]
        std = np.sqrt(st["M2"] / max(st["n"] - 1, 1))
        nonneg = bool(np.nanmin(st["min"]) >= 0)  # after a ReLU-like layer: a unit that never fires is dead
        # otherwise dead = the same value for every sample and position (a position embedding varies by position)
        dead = (st["max"] <= 0) if nonneg else (st["max"] - st["min"] <= 1e-9 * (np.abs(st["max"]) + 1))
        prof.units[L] = {"mean": st["mean"], "std": std, "active": st["active"] / max(st["n"], 1),
                         "dead_mask": dead, "dead": int(dead.sum()), "nonneg": nonneg}
        order = np.argsort(-st["tv"], axis=1)
        prof.top[L] = {"values": np.take_along_axis(st["tv"], order, 1), "index": np.take_along_axis(st["ti"], order, 1),
                       "pos": np.take_along_axis(st["tp"], order, 1)}
    prof.labels = labels_out[:n] or None
    prof.pred = pred_out[:n] or None
    if tokens:
        width = max(len(t) for t in tokens)
        prof.tokens = np.array([np.pad(t, (0, width - len(t)), constant_values=-1) for t in tokens])
    X = {L: np.concatenate(reps[L]) for L in prof.layers if reps[L]}
    if X:
        prof.embed_index = list(range(min(len(next(iter(X.values()))), n)))
        umap_layers = set(prof.layers[-1:]) if embed == "umap" else set()
        for L, x in X.items():
            prof.embed[L] = _embed2d(x, "umap" if L in umap_layers else "pca", seed)
        prof.cka = _cka_matrix([X[L] for L in prof.layers if L in X])
    return prof


def _per_unit(a: np.ndarray, feature_axis: int, b: int) -> tuple[np.ndarray, ...]:
    """Per sample and unit: the mean over positions, the max, where it peaked, and the min — (b, C) each."""
    if a.ndim == 1:
        a = a.reshape(b, -1) if a.shape[0] == b else a.reshape(1, -1)
    if a.ndim == 2:
        return a, a, np.zeros(a.shape, np.int64), a
    fa = feature_axis % a.ndim
    if fa == 0:
        fa = a.ndim - 1
    x = np.moveaxis(a, fa, 1).reshape(a.shape[0], a.shape[fa], -1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.mean(2), x.max(2), x.argmax(2), x.min(2)


def _welford(st: dict, m: np.ndarray) -> None:
    """Chan et al.'s parallel update of the running mean / M2 with a batch of per-sample values."""
    nb = m.shape[0]
    mb = m.mean(0)
    m2b = ((m - mb) ** 2).sum(0)
    tot = st["n"] + nb
    delta = mb - st["mean"]
    st["mean"] = st["mean"] + delta * nb / tot
    st["M2"] = st["M2"] + m2b + delta ** 2 * st["n"] * nb / tot
    st["n"] = tot


def _merge_top(st: dict, mx: np.ndarray, arg: np.ndarray, offset: int) -> None:
    k = st["tv"].shape[1]
    vals = np.concatenate([st["tv"], mx.T.astype(np.float32)], 1)
    idx = np.concatenate([st["ti"], np.broadcast_to(np.arange(offset, offset + mx.shape[0]), mx.T.shape)], 1)
    pos = np.concatenate([st["tp"], arg.T], 1)
    keep = np.argpartition(-vals, k - 1, axis=1)[:, :k] if vals.shape[1] > k else np.arange(vals.shape[1])[None]
    keep = np.broadcast_to(keep, (vals.shape[0], keep.shape[1]))
    st["tv"] = np.take_along_axis(vals, keep, 1)
    st["ti"] = np.take_along_axis(idx, keep, 1)
    st["tp"] = np.take_along_axis(pos, keep, 1)


def _labels_list(y) -> list:
    v = np.asarray(to_numpy(y))
    if v.ndim > 1 and v.shape[-1] > 1:  # one-hot
        v = v.argmax(-1)
    return [x.item() if hasattr(x, "item") else x for x in v.reshape(len(v), -1)[:, 0]]


def _keep_inputs(prof: DatasetProfile, args, offset: int, tokens: list) -> None:
    """Small thumbnails of image inputs, the ids of token inputs (for the snippets)."""
    a = args[0]
    if isinstance(a, dict):
        a = a.get("input_ids", a.get("pixel_values", next(iter(a.values()))))
    v = np.asarray(to_numpy(a))
    if v.ndim == 2 and v.dtype.kind in "iu":
        tokens.extend(list(v))
        return
    if v.ndim != 4 or v.dtype.kind != "f":
        return
    ch_first = v.shape[1] in (1, 3) and v.shape[-1] not in (1, 3)
    imgs = v if not ch_first else v.transpose(0, 2, 3, 1)
    if imgs.shape[-1] not in (1, 3) or min(imgs.shape[1:3]) < 4:
        return
    f = max(1, int(np.ceil(max(imgs.shape[1:3]) / 32)))
    small = imgs[:, ::f, ::f].astype(np.float64)
    for i, img in enumerate(small):
        lo, hi = np.percentile(img, [1, 99])
        u = (np.clip((img - lo) / ((hi - lo) or 1), 0, 1) * 255).astype(np.uint8)
        prof.thumbs[offset + i] = np.repeat(u, 3, -1) if u.shape[-1] == 1 else u


def _embed2d(x: np.ndarray, method: str, seed: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 3:
        return np.zeros((len(x), 2))
    if method == "umap":
        try:
            import umap

            return umap.UMAP(n_components=2, random_state=seed).fit_transform(x)
        except ImportError:
            pass
    c = x - x.mean(0)
    u, s, _ = np.linalg.svd(c, full_matrices=False)
    xy = u[:, :2] * s[:2]
    if xy.shape[1] < 2:
        xy = np.pad(xy, ((0, 0), (0, 2 - xy.shape[1])))
    span = np.abs(xy).max() or 1.0
    return xy / span


def _cka_matrix(reps: list[np.ndarray], max_n: int = 384) -> np.ndarray:
    """Linear CKA between every pair of representations of the same samples (Kornblith et al., 2019)."""
    n = min(min(len(r) for r in reps), max_n)
    vecs = []
    for r in reps:
        x = np.asarray(r[:n], dtype=np.float64)
        x = x - x.mean(0)
        k = x @ x.T
        k = k - k.mean(0) - k.mean(1)[:, None] + k.mean()
        norm = np.linalg.norm(k)
        vecs.append((k / norm).ravel() if norm > 0 else np.zeros(n * n))
    v = np.stack(vecs).astype(np.float32)
    return np.clip(v @ v.T, 0.0, 1.0)
