"""deepvis inside the tools you already log to.

Weights & Biases — the interactive page in your run, next to the curves::

    run = wandb.init(project="mnist")
    dv.integrations.log_wandb(net, x, step=epoch)              # wandb.Html: the whole page, interactive
    dv.integrations.log_wandb(net, x, figure=True)             # + the architecture figure as an image

TensorBoard — per-layer weight histograms and statistics, activation statistics on an input, a TrainingLog's
curves::

    writer = SummaryWriter("runs/exp1")                         # torch.utils.tensorboard or tensorboardX
    dv.integrations.log_tensorboard(writer, net, step, x)       # weights/…, activations/… and stats/…
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from .errors import OptionError


def page_html(net, *inputs, **options) -> str:
    """The interactive page as one HTML string (what ``net.export_html`` would write)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = net.export_html(os.path.join(tmp, "page.html"), *inputs, **options)
        with open(path, encoding="utf-8") as f:
            return f.read()


def log_wandb(net, *inputs, run=None, key: str = "deepvis", step: int | None = None, figure: bool = False,
              **options) -> dict:
    """Log ``net``'s page to a Weights & Biases run (default: the active one) as ``wandb.Html``; with ``figure``
    the static architecture figure too (``wandb.Image``). ``options`` are :meth:`~deepvis.Net.export_html`'s.
    Returns what was logged."""
    try:
        import wandb
    except ImportError as exc:
        raise OptionError("log_wandb needs wandb.", "pip install wandb") from exc
    run = run if run is not None else wandb.run
    if run is None:
        raise OptionError("there is no active wandb run.", "call wandb.init(...) first, or pass run=.")
    data = {key: wandb.Html(page_html(net, *inputs, **options), inject=False)}
    with tempfile.TemporaryDirectory() as tmp:  # wandb reads the image when it logs: keep the file until then
        if figure:
            try:
                png = net.export_svg(os.path.join(tmp, "figure.png"), *inputs)
                data[f"{key}/figure"] = wandb.Image(png)
            except OptionError:  # no cairo: the SVG, which W&B shows as HTML
                svg = net.export_svg(os.path.join(tmp, "figure.svg"), *inputs)
                with open(svg, encoding="utf-8") as f:
                    data[f"{key}/figure"] = wandb.Html(f.read(), inject=False)
        run.log(data, step=step)
    return data


def log_tensorboard(writer, net, step: int, *inputs, weights: bool = True, activations: bool = True,
                    training=None, prefix: str = "") -> None:
    """Write ``net``'s per-layer weight histograms and norms (``weights/…``), with an input its per-layer activation
    histograms and mean / std (``activations/…``), and a :class:`~deepvis.TrainingLog`'s latest metrics (``train/…``)
    to a TensorBoard ``SummaryWriter``."""
    if not hasattr(writer, "add_histogram") or not hasattr(writer, "add_scalar"):
        raise OptionError(f"writer is a {type(writer).__name__}, not a TensorBoard SummaryWriter.",
                          "from torch.utils.tensorboard import SummaryWriter; writer = SummaryWriter('runs/x')")
    p = prefix.rstrip("/") + "/" if prefix else ""
    if weights and net.backend.has_data:
        for path in net.backend.param_paths():
            w = np.asarray(net.backend.get_param(path), dtype=np.float64)
            if w.size == 0 or not np.issubdtype(w.dtype, np.floating):
                continue
            finite = w[np.isfinite(w)]
            if finite.size:
                writer.add_histogram(f"{p}weights/{path}", finite, step)
            writer.add_scalar(f"{p}stats/{path}/norm", float(np.linalg.norm(finite)) if finite.size else 0.0, step)
    if activations and inputs:
        tr = net.trace(*inputs)
        for nid, a in tr.acts.items():
            node = net.graph.nodes.get(nid)
            if node is None or node.op not in ("module", "function"):
                continue
            v = np.asarray(a, dtype=np.float64).reshape(-1)
            v = v[np.isfinite(v)]
            if not v.size:
                continue
            name = nid.lstrip("@") or "model"
            writer.add_histogram(f"{p}activations/{name}", v, step)
            writer.add_scalar(f"{p}activations/{name}/mean", float(v.mean()), step)
            writer.add_scalar(f"{p}activations/{name}/std", float(v.std()), step)
    if training is not None and training.steps:
        for k, series in training.metrics.items():
            if series and series[-1] is not None and np.isfinite(series[-1]):
                writer.add_scalar(f"{p}train/{k}", float(series[-1]), training.steps[-1])
    if hasattr(writer, "flush"):
        writer.flush()
