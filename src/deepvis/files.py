"""``dv.open(path)``: a saved model file as a :class:`~deepvis.Net`, whatever wrote it.

=====================  ==============================================================================
``.onnx``              any exported network, run in onnxruntime (PyTorch, TensorFlow via tf2onnx, …)
``.keras`` / ``.h5``   a Keras 3 model (on the backend KERAS_BACKEND names)
``.pt2``               a ``torch.export`` program: the module tree, classes and example inputs it carries
``.pt`` / ``.pth`` …   ``torch.load`` — a state_dict becomes a weights-only view; a pickled module needs
                       ``trusted=True``, because unpickling runs code from the file
``.safetensors``       a weights-only view, read lazily (a 7B checkpoint opens in seconds)
``.npz``               a weights-only view of its arrays
=====================  ==============================================================================

A weights-only view draws one node per module path (``layer1.0.conv1``), its kind guessed from the tensors it holds,
linked in file order — there is no code to trace — with every tensor's values, statistics and histograms.
"""

from __future__ import annotations

import os
from typing import Any

from .errors import ModelError

EXTENSIONS = (".onnx", ".keras", ".h5", ".hdf5", ".pt2", ".pt", ".pth", ".bin", ".ckpt", ".safetensors", ".npz")


def is_model_file(path: Any) -> bool:
    return isinstance(path, (str, os.PathLike)) and str(path).lower().endswith(EXTENSIONS)


def open_file(path: str | os.PathLike, sample_input=None, *, trusted: bool = False, name: str | None = None):
    """Open a saved model or weights file (see the module docstring for the formats)."""
    from .net import Net, wrap

    path = os.fspath(path)
    if not os.path.exists(path):
        raise ModelError(f"no such file: {path}", "check the path (relative paths start from the current directory).")
    low = path.lower()
    stem = name or os.path.splitext(os.path.basename(path))[0]
    if low.endswith(".onnx"):
        return wrap(path, sample_input=sample_input, name=stem)
    if low.endswith((".keras", ".h5", ".hdf5")):
        keras = _keras()
        model = keras.models.load_model(path, compile=False)
        if sample_input is None:
            from .backends.keras_backend import example_inputs

            sample_input = example_inputs(model)
        return wrap(model, sample_input=sample_input, name=stem)
    if low.endswith(".pt2"):
        import torch

        return wrap(torch.export.load(path), sample_input=sample_input, name=stem)
    if low.endswith(".safetensors"):
        from .backends.weights_backend import from_safetensors

        return Net(from_safetensors(path))
    if low.endswith(".npz"):
        import numpy as np

        from .backends.weights_backend import WeightsBackend

        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        return Net(WeightsBackend(arrays, name=stem, source=path))
    if low.endswith((".pt", ".pth", ".bin", ".ckpt")):
        return _open_torch(path, sample_input, trusted, stem)
    raise ModelError(f"deepvis doesn't know the file type of {path!r}.",
                     f"it opens {', '.join(EXTENSIONS)}; or build the model in code and use dv.wrap(model).")


def _keras():
    """Keras, on an installed backend: KERAS_BACKEND if set, else TensorFlow, JAX or PyTorch — whichever is there
    (Keras itself defaults to TensorFlow and fails without it)."""
    import importlib.util
    import sys

    if "keras" not in sys.modules and "KERAS_BACKEND" not in os.environ:
        for backend, module in (("tensorflow", "tensorflow"), ("jax", "jax"), ("torch", "torch")):
            if importlib.util.find_spec(module) is not None:
                os.environ["KERAS_BACKEND"] = backend
                break
    import keras

    return keras


def _state_dict(obj) -> dict | None:
    """The tensors of a checkpoint: a state_dict itself, or one stored under a usual key."""
    import torch

    if isinstance(obj, dict) and obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
        return obj
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "module", "net", "params", "weights"):
            sd = _state_dict(obj.get(key)) if key in obj else None
            if sd is not None:
                return sd
    return None


def _open_torch(path: str, sample_input, trusted: bool, stem: str):
    import torch

    from .backends.weights_backend import WeightsBackend
    from .net import Net, wrap

    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - a TorchScript archive, or a pickled module
        if "TorchScript" in str(exc) or "torch.jit.load" in str(exc):
            script = torch.jit.load(path, map_location="cpu")
            return Net(WeightsBackend(dict(script.state_dict()), name=stem, source=path))
        if not trusted:
            raise ModelError(f"{os.path.basename(path)} holds pickled Python objects (a whole model, not only its "
                             "weights), and loading those runs code from the file.",
                             "if you trust the file: dv.open(path, trusted=True) (command line: --trust). Better: "
                             "save model.state_dict() and open that, or export the model with torch.export.") \
                from exc
        obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.nn.Module):
        return wrap(obj.eval(), sample_input=sample_input, name=stem)
    sd = _state_dict(obj)
    if sd is None:
        raise ModelError(f"{os.path.basename(path)} holds a {type(obj).__name__}, not a model or a state_dict.",
                         "save the model (torch.export / ONNX) or its state_dict.")
    return Net(WeightsBackend(dict(sd), name=stem, source=path))
