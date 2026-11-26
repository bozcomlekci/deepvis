"""deepvis — wrap PyTorch & JAX networks; focus, monitor, unroll into math, edit, and visualize.

>>> import deepvis as dv
>>> dv.visualize(model, x)            # the one call: wrap, run on x, show every layer, tensor and output
>>> net = dv.wrap(model)              # or keep the wrapper to edit, trace, unroll … (net.show(x) to look)
>>> net = dv.open("model.onnx")        # a saved file: .onnx .keras .pt2 .pt .safetensors .npz
"""

import os

from . import theme
from .core.graph import ModelGraph, Node, ParamInfo
from .manipulate.edits import Edit, load_edits, save_edits
from .manipulate.interventions import (
    Intervention,
    ablate,
    add_noise,
    amplify,
    clamp,
    custom,
    intervention,
    mean_ablate,
    patch,
    replace,
    steer,
)
from .manipulate.ops import register_op
from .manipulate.params import ParamRef
from .monitor.recorder import Monitor, Trace
from .monitor.training import TrainingLog
from .errors import DeepvisError, DeepvisWarning
from .net import Net, NodeView, Selection, wrap
from .files import open_file as open  # noqa: A001 - dv.open(path), like the builtin but for model files
from . import callbacks
from . import integrations  # dv.integrations.log_wandb / log_tensorboard
from .callbacks import watch


def _wrap_for_export(model, inputs, params=None, wrap_kw=None):
    from . import errors as E

    if isinstance(model, Net):
        args = E.prepare_inputs(model.backend.model if model.framework == "torch" else model, inputs, model.framework)
        return model, args
    from .files import is_model_file

    if is_model_file(model):  # dv.visualize("model.onnx"): the file's own inputs when none are given
        net = open(model, sample_input=inputs)
        return net, net.sample_input or ()
    E.check_model(model)
    if inputs is None:
        E.prepare_inputs(model, None, "")  # raises the explained InputError
    net = wrap(model, params, sample_input=inputs, **(wrap_kw or {}))
    return net, net.sample_input


def _check_options(options: dict, where: str, live: bool = False) -> None:
    import inspect

    from . import errors as E

    known = [p for p in inspect.signature(Net.export_html).parameters if p not in ("self", "path", "inputs")]
    if live:
        from .viz.server import LiveServer

        known += [p for p in inspect.signature(LiveServer.__init__).parameters
                  if p not in ("self", "net", "args", "ui", "host") and p not in known] + ["views", "start", "port"]
    for k in options:
        if k not in known:
            hint = E.suggest(k, known)
            raise E.OptionError(f"{where}() got an unknown option {k!r}.",
                                (hint + " " if hint else "") + "Options: " + ", ".join(known) + ".")


def export_html(model, inputs, path: str, params=None, wrap_kw: dict | None = None, **kw) -> str:
    """One call from any model to a self-contained interactive HTML file.

    >>> dv.export_html(torch_model, x, "model.html", title="My CNN", views=("graph", "volumes"))
    >>> dv.export_html(flax_module, x, "model.html", params=variables)
    """
    _check_options(kw, "export_html")
    net, args = _wrap_for_export(model, inputs, params, wrap_kw)
    return net.export_html(path, *args, **kw)


CAPTURE_ENV = "DEEPVIS_CAPTURE"  # when set, visualize() writes its page to this file (the VS Code extension)
CAPTURE_LINES_ENV = "DEEPVIS_CAPTURE_LINES"  # "FILE:FIRST-LAST": only the call written on those lines is captured
CAPTURE_STOP_ENV = "DEEPVIS_CAPTURE_STOP"  # stop the script once the page is written
CAPTURE_OPTIONS_ENV = "DEEPVIS_CAPTURE_OPTIONS"  # JSON options added to the captured call (e.g. {"compare": "ckpt.pt"})


def _capture_wanted() -> bool:
    """Is this visualize() call the one being captured? (Without $DEEPVIS_CAPTURE_LINES, every call is.)"""
    import sys

    spec = os.environ.get(CAPTURE_LINES_ENV)
    if not spec:
        return True
    file, _, span = spec.rpartition(":")
    first, _, last = span.partition("-")
    lo, hi = int(first), int(last or first)
    f = sys._getframe(2)  # the caller of visualize()
    while f is not None:
        if os.path.abspath(f.f_code.co_filename) == os.path.abspath(file):
            return lo <= f.f_lineno <= hi
        f = f.f_back
    return False


def visualize(model, inputs, params=None, *, path: str | None = None, open_browser: bool | None = None,
              live: bool = False, wrap_kw: dict | None = None, **options) -> Net:
    """The one call: wrap ``model``, run it on ``inputs`` and show the interactive page — the inputs, every
    intermediate tensor, the outputs, each layer drawn from its own numbers.

    model    a torch module · a flax module (``params=variables``) · an nnx / equinox model · ``(apply_fn, params)``
    inputs   what you would call the model with: one tensor, or a tuple of them (``model(*inputs)``)
    options  anything :meth:`Net.export_html` takes — ``title``, ``start`` (the tab to open), ``views``,
             ``tokenizer``, ``labels``, ``diffusion``, ``grads``, ``loss_fn``, ``sample``, ``depth`` …

    The page goes to the file named by ``$DEEPVIS_CAPTURE`` (the VS Code extension), else to ``path``, else inline
    in Jupyter, else your browser. Returns the wrapped :class:`Net` (``net.page`` is the file written, if any), so
    you can keep going: ``net.equation()``, ``net.trace(x)``, ``net["fc1"].prune(0.5)`` …

    ``live=True`` serves the page from this process instead, and returns the
    :class:`~deepvis.viz.server.LiveServer`: call ``srv.update(step, loss=...)`` from a training loop and the page
    follows along, or pass ``train_step=fn`` for Train / Pause / Step buttons (see :meth:`Net.serve`). Run from VS
    Code with the deepvis extension, the live page opens beside the code.
    """
    _check_options(options, "visualize", live=live)
    if live:
        net, args = _wrap_for_export(model, inputs, params, wrap_kw)
        from .viz.server import LiveServer

        srv = LiveServer(net, args, **options)
        return srv.start(open_browser=True if open_browser is None else open_browser)
    capture = os.environ.get(CAPTURE_ENV)
    if capture and path and os.path.abspath(path) == os.path.abspath(capture):
        capture = None  # deepvis's own capture call: it handles the page itself
    if capture and not _capture_wanted():
        if not path:
            return _wrap_for_export(model, inputs, params, wrap_kw)[0]  # another call is being captured: no page
        capture = None
    net, args = _wrap_for_export(model, inputs, params, wrap_kw)
    target = capture or path
    if capture and os.environ.get(CAPTURE_OPTIONS_ENV):
        import json

        options = {**options, **json.loads(os.environ[CAPTURE_OPTIONS_ENV])}
        _check_options(options, "visualize")
    if target:
        net.page = net.export_html(target, *args, **options)
        if capture and os.environ.get(CAPTURE_STOP_ENV):
            from .errors import Captured

            raise Captured(net.page)  # the page is all that was asked for: skip the rest of the script
        return net
    from .viz.html.dashboard import Dashboard, in_notebook

    d = Dashboard(net, args, **options)
    if in_notebook() and not open_browser:
        from IPython.display import display

        display(d)
        net.page = None
    else:
        d.display(open_browser=True)
        net.page = d.path
    return net


def compare(model_a, model_b, inputs, params=None, *, names=("A", "B"), **options) -> Net:
    """Two versions of one network side by side: ``model_b`` is another model, a state_dict or a checkpoint path
    (.pt / .pth / .safetensors). The page opens on the Compare tab — how far each layer's weights and activations
    moved, both outputs — and has every other view of ``model_a``. Options as :func:`visualize`."""
    options.setdefault("start", "compare")
    return visualize(model_a, inputs, params, compare=model_b, compare_names=list(names), **options)


def export_svg(model, inputs, path: str, params=None, wrap_kw: dict | None = None, **kw) -> str:
    """One call from any model to a static SVG (or .html) architecture figure with expressive glyphs."""
    net, args = _wrap_for_export(model, inputs, params, wrap_kw)
    return net.export_svg(path, *args, **kw)


def equation_rule(kind):
    """Register a custom LaTeX rule for a layer kind / class: see :mod:`deepvis.equations.rules`."""
    from .equations.rules import equation_rule as _er

    return _er(kind)


def scope(name: str):
    """Name a region of a pure-JAX function so it shows up as one node (alias of ``jax.named_scope``)."""
    import jax

    return jax.named_scope(name)


__all__ = [
    "visualize", "compare", "wrap", "open", "watch", "callbacks", "integrations", "export_html", "export_svg", "DeepvisError", "DeepvisWarning", "Net", "NodeView", "Selection", "ParamRef", "Monitor", "Trace", "TrainingLog", "ModelGraph", "Node", "ParamInfo",
    "Edit", "load_edits", "save_edits", "register_op", "equation_rule", "scope", "theme",
    "Intervention", "ablate", "mean_ablate", "amplify", "clamp", "steer", "add_noise", "replace", "custom",
    "intervention", "patch",
]
__version__ = "0.1.0"
