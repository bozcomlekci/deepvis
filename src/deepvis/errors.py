"""Errors that say what went wrong and how to fix it.

Every deepvis error reads::

    deepvis: <what went wrong>
      → fix: <what to do about it>

and subclasses the matching built-in (``ValueError``, ``TypeError``, ``RuntimeError``, ``OSError``), so existing
``except`` clauses keep working. When the model itself raised, the original exception is chained (``raise … from``)
and its traceback still points at the failing line of the model's code.
"""

from __future__ import annotations

import difflib
import re
import warnings
from typing import Any, Iterable


class DeepvisError(Exception):
    """A problem deepvis can explain."""

    def __init__(self, what: str, fix: str | None = None):
        self.what, self.fix = what, fix
        super().__init__(f"{what}" + (f"\n  → fix: {fix}" if fix else ""))


class ModelError(DeepvisError, TypeError, ValueError):
    """``model`` is not something deepvis can wrap, or lacks what was asked of it."""


class InputError(DeepvisError, ValueError):
    """``inputs`` can't be fed to the model as given."""


class RunError(DeepvisError, RuntimeError):
    """Running the model on the inputs failed (the model's own error is chained)."""


class OptionError(DeepvisError, TypeError, ValueError):
    """An option name or value is not one deepvis knows."""


class PathError(DeepvisError, OSError):
    """The output file can't be written."""


class Captured(BaseException):
    """Stops a script once the page it was run for is written (``$DEEPVIS_CAPTURE_STOP``; see deepvis.capture). A
    BaseException, so a script's own ``except Exception`` can't swallow it."""

    def __init__(self, page: str):
        super().__init__(page)
        self.page = page


class DeepvisWarning(UserWarning):
    """Something was skipped or adjusted; the page is still made."""


def warn(msg: str) -> None:
    warnings.warn(msg, DeepvisWarning, stacklevel=3)


def suggest(name: str, choices: Iterable[str]) -> str:
    close = difflib.get_close_matches(name, list(choices), n=1, cutoff=0.6)
    return f"did you mean {close[0]!r}?" if close else ""


# ----------------------------------------------------------------------------------------------- model
def check_model(model: Any) -> None:
    if model is None:
        raise ModelError("`model` is None.",
                         "pass the network object, e.g. dv.visualize(my_model, x) where my_model = MyNet().")
    if isinstance(model, (str, bytes)):
        raise ModelError(f"`model` is the string {model!r}, not a network.",
                         "pass the network object itself; a saved file opens with dv.open(path) (.onnx .keras .pt2 "
                         ".pt .safetensors .npz); to load one by name use the command line, e.g. "
                         "`deepvis export torchvision:resnet18` or `deepvis export hf:gpt2`.")
    if isinstance(model, (int, float, bool)):
        raise ModelError(f"`model` is a {type(model).__name__}, not a network.", "pass the network object itself.")
    if isinstance(model, type):
        raise ModelError(f"`model` is the class {model.__name__}, not a network built from it.",
                         f"create it first, e.g. {model.__name__}(...) — note the parentheses.")


def unknown_model(model: Any) -> ModelError:
    return ModelError(
        f"deepvis doesn't know how to wrap a {type(model).__module__}.{type(model).__name__}.",
        "pass a torch.nn.Module, a flax.linen module with its variables (dv.visualize(module, x, params=variables)), "
        "an nnx or equinox model, or a pure JAX function with its params (dv.visualize((apply_fn, params), x)).")


# ----------------------------------------------------------------------------------------------- inputs
def prepare_inputs(model: Any, inputs: Any, framework: str) -> tuple:
    """``inputs`` as the tuple of positional arguments for the model; harmless mismatches fixed quietly."""
    if inputs is None:
        raise InputError("`inputs` is None.",
                         "pass what you would call the model with, e.g. dv.visualize(model, torch.randn(1, 3, 32, 32)).")
    if isinstance(inputs, tuple) and len(inputs) == 1 and _is_mapping(inputs[0]):
        inputs = inputs[0]
    if _is_mapping(inputs):
        bound = _bind_dict(model, inputs, framework)
        if bound is None:
            keys = list(inputs.keys())
            hint = "inputs['input_ids']" if "input_ids" in keys else f"inputs[{keys[0]!r}]" if keys else "the tensor"
            raise InputError(f"`inputs` is a dict (keys {keys[:6]}) whose names don't match the model's arguments.",
                             f"pass the tensor(s) the model's forward takes, in order — e.g. {hint}, or a tuple of "
                             "them.")
        inputs = bound
    args = inputs if isinstance(inputs, tuple) else (inputs,)
    if framework == "torch":
        args = tuple(_to_torch(a, model) for a in args)
    for a in args:
        shape = getattr(a, "shape", None)
        if shape is not None and len(shape) >= 1 and shape[0] == 0:
            raise InputError(f"an input has an empty batch: shape {tuple(shape)}.",
                             "give at least one example, e.g. x[:1] rather than x[:0].")
    return args


def _is_mapping(x: Any) -> bool:
    return isinstance(x, dict) or (hasattr(x, "keys") and hasattr(x, "__getitem__") and not hasattr(x, "shape"))


def _bind_dict(model: Any, inputs: Any, framework: str) -> tuple | None:
    """A dict of inputs (a Hugging Face tokenizer's output) as the positional arguments of the model's forward:
    ``{"input_ids": …, "attention_mask": …}`` → ``(input_ids, attention_mask)``. Keras functional models take the
    dict itself. ``None`` if the names don't match."""
    import inspect

    if framework == "keras":
        return (dict(inputs),)
    fn = getattr(model, "forward", None) if framework == "torch" else model
    try:
        params = [p for p in inspect.signature(fn).parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    except (TypeError, ValueError):
        return None
    names = [p.name for p in params]
    keys = list(inputs.keys())
    if not keys or any(k not in names for k in keys):
        return None
    last = max(names.index(k) for k in keys)
    args = []
    for p in params[: last + 1]:
        if p.name in inputs:
            args.append(inputs[p.name])
        elif p.default is not inspect.Parameter.empty:
            args.append(p.default)  # an argument the dict skips: its default, positionally
        else:
            return None
    return tuple(args)


def _to_torch(a: Any, model: Any):
    import numpy as np
    import torch

    p = next(iter(model.parameters()), None) if hasattr(model, "parameters") else None
    if isinstance(a, np.ndarray) or (isinstance(a, list) and a and all(isinstance(v, (int, float)) for v in a)):
        t = torch.as_tensor(np.asarray(a))
        if t.is_floating_point() and p is not None and p.is_floating_point():
            t = t.to(p.dtype)
        a = t
    if isinstance(a, torch.Tensor) and p is not None and a.device != p.device:
        a = a.to(p.device)  # the model's device: a cpu tensor for a model on mps / cuda just works
    return a


# ----------------------------------------------------------------------------------------------- running
def _shape(a) -> str:
    s = getattr(a, "shape", None)
    return str(tuple(s)) if s is not None else type(a).__name__


def explain(err: BaseException, args: tuple = (), generic: bool = True) -> str:
    """A concrete fix for common failures of a model called on the wrong inputs ("" if nothing specific and not
    ``generic``)."""
    msg = str(err)
    shapes = ", ".join(_shape(a) for a in args) or "?"
    m = re.search(r"mat1 and mat2 shapes cannot be multiplied \((\d+)x(\d+) and (\d+)x(\d+)\)", msg)
    if m:
        return (f"a Linear layer expects {m.group(3)} features in the last dimension but got {m.group(2)} "
                f"(inputs: {shapes}). Reshape the input, or check the first layer's in_features.")
    m = re.search(r"expected to generate shape \(([\d, ]+)\), but the existing parameter it received has shape "
                  r"\(([\d, ]+)\)", msg)
    if m:  # flax: weights were created (module.init) for a different input size
        return (f"the weights were created for inputs of another size (a weight is ({m.group(2)}); these inputs would "
                f"need ({m.group(1)}); inputs: {shapes}). Pass inputs shaped like the ones given to module.init.")
    m = re.search(r"dot_general requires contracting dimensions to have the same shape, got \((\d+),?\) and "
                  r"\((\d+),?\)", msg)
    if m:  # jax: x @ W with mismatched inner sizes
        return (f"a dense layer expects {m.group(2)} features but the input has {m.group(1)} (inputs: {shapes}). "
                "Reshape the input, or check the layer's size.")
    m = re.search(r"expected input\[([\d, ]+)\] to have (\d+) channels, but got (\d+) channels", msg)
    if m:
        return (f"the first convolution expects {m.group(2)} channels but the input has {m.group(3)} "
                f"(inputs: {shapes}). Channels come right after the batch: (batch, channels, height, width).")
    if "Expected 3D (unbatched) or 4D (batched) input" in msg or "Expected 2D (unbatched) or 3D (batched)" in msg:
        return f"the input has the wrong number of dimensions ({shapes}); add a batch dimension, e.g. x.unsqueeze(0)."
    if "indices" in msg and ("Long, Int" in msg or "IntTensor" in msg or "LongTensor" in msg):
        return ("this model starts with an embedding: pass integer token ids, e.g. torch.randint(0, vocab_size, "
                "(1, 16)), or convert with x.long().")
    if "index out of range in self" in msg:
        return ("a token id is larger than the embedding table: keep ids in [0, vocab_size), e.g. "
                "torch.randint(0, model.config.vocab_size, (1, 16)).")
    if re.search(r"(same dtype|expected scalar type|found dtype).*(Long|Int)|(Long|Int).*(Float|Half)", msg):
        return "the input is integers but the model computes in floats: convert with x.float()."
    if "Half" in msg and "Float" in msg or "BFloat16" in msg:
        return "the input's precision differs from the model's: convert with x.to(next(model.parameters()).dtype)."
    if re.search(r"on (cpu|cuda|mps).*(expected|but).*(cpu|cuda|mps)|Expected all tensors to be on the same device",
                 msg):
        return "the model and the input are on different devices: move one of them, e.g. x.to('cuda')."
    if "missing" in msg and "required positional argument" in msg:
        return (f"the model's forward takes more arguments than given ({len(args)}): pass all of them as a tuple, "
                "e.g. dv.visualize(model, (x, t)).")
    if "takes" in msg and "positional argument" in msg and "given" in msg:
        return "the number of inputs doesn't match the model's forward signature: pass exactly the arguments it takes."
    if "must be Tensor, not" in msg:
        return "an input is not a tensor: convert it, e.g. torch.as_tensor(x)."
    if not generic:
        return ""
    if isinstance(err, (NameError, AttributeError, KeyError, ImportError)):
        return "this is an error in the model's own code (see the traceback above); fix it there."
    return (f"check that the model runs on these inputs by itself — model(*inputs) with inputs {shapes} — "
            "the traceback above shows where it fails.")


def run_error(err: BaseException, args: tuple) -> RunError:
    first = str(err).strip().splitlines()[0] if str(err).strip() else ""
    return RunError(f"the model raised {type(err).__name__} when run on the inputs: {first[:300]}", explain(err, args))


def trace_error(err: BaseException) -> RunError:
    first = str(err).strip().splitlines()[0] if str(err).strip() else ""
    return RunError(f"the model runs, but deepvis couldn't trace it ({type(err).__name__}: {first[:200]}).",
                    "this is a deepvis limitation — please report it with the model's code. Meanwhile "
                    "dv.wrap(model, sample_input=x).export_svg(...) or a smaller part of the model may work.")


def probe(model: Any, args: tuple, framework: str) -> BaseException | None:
    """Run the model once on its own: None if it works, else its exception (so errors can be attributed)."""
    if framework != "torch":
        return None
    import torch

    try:
        with torch.no_grad():
            model(*args)
    except Exception as exc:  # noqa: BLE001
        return exc
    return None


__all__ = ["DeepvisError", "ModelError", "InputError", "RunError", "OptionError", "PathError", "DeepvisWarning",
           "explain", "check_model", "prepare_inputs"]
