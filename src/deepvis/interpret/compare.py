"""Two versions of one network side by side — before / after fine-tuning, two checkpoints, an edited copy.

>>> dv.compare(model, "ckpt_step5000.pt", x)            # or another model object, or a state_dict
>>> dv.visualize(model, x, compare=finetuned)            # the same, as an option

Per layer: how far the weights moved (‖W_b − W_a‖ / ‖W_a‖) and how far the activations on the same input moved
(‖h_b − h_a‖ / ‖h_a‖, and their cosine similarity); and both outputs.
"""

from __future__ import annotations

import copy
import os
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import ModelError, OptionError, PathError

if TYPE_CHECKING:  # pragma: no cover
    from ..net import Net


def _load_state(path: str) -> dict:
    if not os.path.exists(path):
        raise PathError(f"compare={path!r}: no such file.", "pass the checkpoint's path, a model, or a state_dict.")
    if path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ModelError("reading .safetensors needs the safetensors package.", "pip install safetensors") from exc
        return load_file(path)
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.nn.Module):
        return obj.state_dict()
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "module"):  # common training-checkpoint layouts
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        return obj
    raise OptionError(f"compare={path!r} holds a {type(obj).__name__}, not weights.",
                      "save the model's state_dict (torch.save(model.state_dict(), path)).")


def other_net(net: "Net", other: Any) -> "Net":
    """The second network: a model / Net as given, or ``net``'s model with a state_dict / checkpoint loaded."""
    from ..net import Net, wrap

    if isinstance(other, Net):
        return other
    if net.framework != "torch":
        if isinstance(other, (str, dict)):
            raise OptionError("compare= with a checkpoint or params is available for PyTorch models.",
                              "pass the second model itself.")
        return wrap(other, sample_input=net.sample_input)
    import torch

    if isinstance(other, torch.nn.Module):
        return wrap(other, sample_input=net.sample_input)
    if isinstance(other, (str, os.PathLike)):
        other = _load_state(os.fspath(other))
    if isinstance(other, dict):
        model = copy.deepcopy(net.backend.model)
        state = {k.removeprefix("module."): v for k, v in other.items()}  # DataParallel / DDP prefixes
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing and len(missing) == len(model.state_dict()):
            raise OptionError("compare=: none of the checkpoint's weights match this model's layers.",
                              f"it has keys like {list(state)[:3]}, the model {list(model.state_dict())[:3]} — "
                              "compare checkpoints of the same architecture.")
        return wrap(model, sample_input=net.sample_input)
    raise OptionError(f"compare= got a {type(other).__name__}.",
                      "pass a model, a state_dict, or the path of a checkpoint (.pt / .pth / .safetensors).")


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(b - a) / (np.linalg.norm(a) or 1.0))


def compare(net_a: "Net", net_b: "Net", args: tuple, sample: int = 0) -> dict:
    """Per-layer weight and activation changes from ``net_a`` to ``net_b`` on ``args``, plus both traces."""
    from ..core.tensor import to_numpy

    ga, gb = net_a.graph, net_b.graph
    if set(ga.nodes) != set(gb.nodes):
        only = sorted(set(ga.nodes) ^ set(gb.nodes))[:4]
        raise OptionError("the two networks have different layers, so they can't be compared layer by layer.",
                          f"compare two versions of the same architecture (differs at {only}).")
    ta, tb = net_a.trace(*args), net_b.trace(*args)
    layers = []
    for nid in ga.topo_order():
        n = ga.nodes[nid]
        if not n.is_compute:
            continue
        w = None
        paths = [i.path for i in n.params.values()]
        if paths:
            try:
                A = np.concatenate([np.ravel(net_a.backend.get_param(p)).astype(np.float64) for p in paths])
                B = np.concatenate([np.ravel(net_b.backend.get_param(p)).astype(np.float64) for p in paths])
                w = _rel(A, B) if A.shape == B.shape else None
            except (KeyError, ValueError):
                w = None
        act = cos = None
        if nid in ta.acts and nid in tb.acts and np.shape(ta.acts[nid]) == np.shape(tb.acts[nid]):
            a = np.asarray(to_numpy(ta.acts[nid]), dtype=np.float64)
            b = np.asarray(to_numpy(tb.acts[nid]), dtype=np.float64)
            if a.ndim >= 1 and a.shape[0] > sample:
                a, b = a[sample].ravel(), b[sample].ravel()
            act = _rel(a, b)
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            cos = float(a @ b / (na * nb)) if na and nb else None
        if w is None and act is None:
            continue
        layers.append({"id": nid, "kind": n.kind, "weight": w, "act": act, "cos": cos})
    return {"layers": layers, "trace_a": ta, "trace_b": tb}


__all__ = ["compare", "other_net"]
