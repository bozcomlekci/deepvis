"""Which parts of the input made the model say what it said (PyTorch).

* ``saliency``              |∂y/∂x| · |x| — fast, noisy
* ``integrated_gradients``  (x − x₀) · mean over α of ∂y/∂x at x₀ + α(x − x₀) — sums to ≈ y(x) − y(x₀)
                            (completeness; the midpoint rule over ``steps`` points)
* ``grad_cam``              ReLU(Σ_c mean(∂y/∂A_c) · A_c) on the last convolutional feature map, upsampled — images

Token inputs (integer ids) are attributed through the embedding: the same formulas on the embedding output,
summed over the embedding dimension, one number per token.

>>> a = net.attribute(x, method="ig")          # the predicted class of sample 0
>>> a.map.shape                                  # (H, W) for an image, (T,) for tokens, (F,) for features
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import InputError, ModelError, OptionError

if TYPE_CHECKING:  # pragma: no cover
    from ..net import Net

METHODS = {"saliency": "saliency", "ig": "integrated_gradients", "integrated_gradients": "integrated_gradients",
           "gradients": "saliency", "grad_cam": "grad_cam", "gradcam": "grad_cam", "cam": "grad_cam"}


@dataclass
class Attribution:
    map: np.ndarray                # per pixel (H, W) · per token (T,) · per feature (F,)
    method: str
    target: int                    # the output index explained
    score: float                   # the output value explained
    kind: str                      # "image" | "tokens" | "features"
    sample: int = 0
    extra: dict = field(default_factory=dict)  # e.g. {"baseline_score": …} for integrated gradients

    def __repr__(self) -> str:
        return f"Attribution({self.method}, target={self.target}, {self.kind} {tuple(self.map.shape)})"


def _torch():
    import torch

    return torch


def _first_tensor_index(args) -> int:
    torch = _torch()
    for i, a in enumerate(args):
        if isinstance(a, torch.Tensor):
            return i
    raise InputError("attribution needs a tensor input.", "pass the model's input tensor, e.g. net.attribute(x).")


def _score_fn(out, target: int | None, sample: int):
    """(the scalar to explain, the target index) from the model output for one sample."""
    torch = _torch()
    from ..core.tensor import first_array

    y = first_array(out)
    if not isinstance(y, torch.Tensor) or not y.is_floating_point():
        raise ModelError("the model's output isn't a float tensor, so there's nothing to attribute.",
                         "attribute a model whose output is scores / logits.")
    ys = y[sample] if y.dim() >= 2 else y
    if ys.dim() >= 2:  # (positions, classes): explain the last position (a language model's next token)
        ys = ys.reshape(-1, ys.shape[-1])[-1]
    ys = ys.reshape(-1)
    t = int(ys.detach().argmax()) if target is None else int(target)
    if not 0 <= t < ys.numel():
        raise OptionError(f"target={target} is out of range: the output has {ys.numel()} values.",
                          f"pick 0 … {ys.numel() - 1}, or leave target=None for the predicted one.")
    return ys[t], t


def _embedding(model):
    torch = _torch()
    for m in model.modules():
        if isinstance(m, torch.nn.Embedding):
            return m
    return None


def _last_conv(model, name: str | None):
    torch = _torch()
    if name:
        try:
            return model.get_submodule(name)
        except AttributeError as exc:
            raise OptionError(f"node={name!r} is not a submodule of the model.",
                              "pass a module path, e.g. 'layer4' or 'features.28'.") from exc
    last = None
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv1d)):
            last = m
    return last


def attribute(net: "Net", *inputs, method: str = "integrated_gradients", target: int | None = None, sample: int = 0,
              steps: int = 32, baseline: Any = None, node: str | None = None) -> Attribution:
    """Explain output ``target`` (default: the predicted one) of ``sample`` in the batch; see the module docstring."""
    torch = _torch()
    from .. import errors as E

    if net.framework != "torch":
        raise ModelError("attribution is available for PyTorch models for now.",
                         "wrap the torch version of the model, or compute gradients with jax.grad directly.")
    m = METHODS.get(str(method).lower())
    if m is None:
        raise OptionError(f"unknown attribution method {method!r}.", "use 'saliency', 'integrated_gradients' "
                          "(or 'ig') or 'grad_cam'.")
    model = net.backend.model
    args = E.prepare_inputs(model, inputs if inputs else net.sample_input, "torch")
    i = _first_tensor_index(args)
    x_all = args[i]
    sample = max(0, min(int(sample), x_all.shape[0] - 1)) if x_all.dim() >= 1 else 0
    x = x_all[sample:sample + 1]
    others = [a[sample:sample + 1] if isinstance(a, torch.Tensor) and a.dim() >= 1 and a.shape[0] == x_all.shape[0]
              else a for a in args]
    tokens = not x.is_floating_point()
    modes = [(mod, mod.training) for mod in model.modules()]  # each layer's own: a frozen BatchNorm stays frozen
    model.eval()
    try:
        if m == "grad_cam":
            return _grad_cam(model, others, i, x, target, node)
        if tokens:
            return _token_attr(model, others, i, x, target, m, steps)
        return _input_attr(model, others, i, x, target, m, steps, baseline)
    finally:
        for mod, was_training in modes:
            mod.training = was_training


def _call(model, others, i, x):
    a = list(others)
    a[i] = x
    return model(*a)


def _input_attr(model, others, i, x, target, m, steps, baseline) -> Attribution:
    torch = _torch()
    x0 = torch.zeros_like(x) if baseline is None else torch.as_tensor(baseline, dtype=x.dtype,
                                                                      device=x.device).expand_as(x)
    with torch.enable_grad():
        xi = x.detach().clone().requires_grad_(True)
        score, t = _score_fn(_call(model, others, i, xi), target, 0)
        if m == "saliency":
            (g,) = torch.autograd.grad(score, xi)
            a = (g * xi).abs()
            extra = {}
        else:
            total = torch.zeros_like(x)
            for k in range(steps):  # midpoint rule along the straight path x0 → x
                xa = (x0 + ((k + 0.5) / steps) * (x - x0)).detach().requires_grad_(True)
                sk, _ = _score_fn(_call(model, others, i, xa), t, 0)
                (g,) = torch.autograd.grad(sk, xa)
                total += g
            a = (x - x0) * total / steps
            with torch.no_grad():
                s0, _ = _score_fn(_call(model, others, i, x0), t, 0)
            extra = {"baseline_score": float(s0), "sum": float(a.sum())}
    a = a.detach()[0]
    kind = "image" if a.dim() == 3 else "features"
    amap = a.sum(0) if a.dim() == 3 else a.reshape(-1) if a.dim() <= 1 else a.sum(-1)
    return Attribution(amap.float().cpu().numpy(), "saliency" if m == "saliency" else "integrated_gradients", t,
                       float(score.detach()), kind, extra=extra)


def _token_attr(model, others, i, x, target, m, steps) -> Attribution:
    torch = _torch()
    emb = _embedding(model)
    if emb is None:
        raise ModelError("the input is integer ids but the model has no nn.Embedding to attribute through.",
                         "attribute a float input, or a model that starts with an embedding.")
    state: dict = {"alpha": 1.0}

    def hook(mod, inp, out):
        out = out * state["alpha"]
        if torch.is_grad_enabled():
            out.retain_grad()
        state["e"] = out
        return out

    h = emb.register_forward_hook(hook)
    try:
        with torch.enable_grad():
            score, t = _score_fn(_call(model, others, i, x), target, 0)
            e1 = state["e"]
            if m == "saliency":
                (g,) = torch.autograd.grad(score, e1)
                a = (g * e1).detach()
                extra = {}
            else:
                total = None
                for k in range(steps):
                    state["alpha"] = (k + 0.5) / steps
                    sk, _ = _score_fn(_call(model, others, i, x), t, 0)
                    (g,) = torch.autograd.grad(sk, state["e"])
                    total = g if total is None else total + g
                state["alpha"] = 1.0
                a = e1.detach() * total / steps  # (x − 0) · mean gradient, x = the embedding output
                state["alpha"] = 0.0
                with torch.no_grad():
                    s0, _ = _score_fn(_call(model, others, i, x), t, 0)
                extra = {"baseline_score": float(s0), "sum": float(a.sum())}
    finally:
        h.remove()
    amap = a[0].sum(-1)
    return Attribution(amap.float().cpu().numpy(), "saliency" if m == "saliency" else "integrated_gradients", t,
                       float(score.detach()), "tokens", extra=extra)


def _grad_cam(model, others, i, x, target, node) -> Attribution:
    torch = _torch()
    F = torch.nn.functional
    conv = _last_conv(model, node)
    if conv is None:
        raise ModelError("Grad-CAM needs a convolutional layer, and this model has none.",
                         "use method='integrated_gradients' or 'saliency' instead.")
    state: dict = {}

    def hook(mod, inp, out):
        out.retain_grad()
        state["a"] = out

    h = conv.register_forward_hook(hook)
    try:
        with torch.enable_grad():
            xi = x.detach().clone().requires_grad_(x.is_floating_point())
            score, t = _score_fn(_call(model, others, i, xi), target, 0)
            A = state["a"]
            (g,) = torch.autograd.grad(score, A)
    finally:
        h.remove()
    w = g.mean(dim=tuple(range(2, g.dim())), keepdim=True)
    cam = F.relu((w * A).sum(1, keepdim=True)).detach()
    if x.dim() == 4 and cam.dim() == 4:
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        kind = "image"
    else:
        cam = F.interpolate(cam, size=x.shape[-1:], mode="linear", align_corners=False) if cam.dim() == 3 else cam
        kind = "features"
    return Attribution(cam[0, 0].float().cpu().numpy(), "grad_cam", t, float(score.detach()), kind)


__all__ = ["Attribution", "attribute", "METHODS"]
