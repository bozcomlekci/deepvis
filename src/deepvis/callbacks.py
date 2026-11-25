"""Watch a network learn from any training loop: the live page follows along (loss curves, per-layer dynamics,
the replay of weights and outputs), in the browser — or beside your code in VS Code.

Plain loop::

    with dv.watch(model, x_sample, every=50) as w:      # snapshot the layers every 50 steps
        for step, (x, y) in enumerate(loader):
            loss = train_step(x, y)
            w.step(step, loss=loss)

PyTorch Lightning / HuggingFace Trainer::

    trainer = pl.Trainer(callbacks=[dv.callbacks.lightning(x_sample)])
    trainer = transformers.Trainer(..., callbacks=[dv.callbacks.huggingface(batch["input_ids"][:1])])
"""

from __future__ import annotations

from typing import Any

from .errors import ModelError


class Watch:
    """A live page for a model being trained; :meth:`step` pushes progress to it."""

    def __init__(self, model, inputs, params=None, *, every: int | None = 50, open_browser: bool | None = None,
                 **options):
        import deepvis as dv

        options.setdefault("start", "monitor")
        self.server = dv.visualize(model, inputs, params, live=True, open_browser=open_browser,
                                   snapshot_every=every, **options)
        self._count = 0

    @property
    def url(self) -> str:
        return self.server.url

    def step(self, step: int | None = None, *, params=None, grads=None, **metrics) -> None:
        """Log ``metrics`` (``loss=…``, ``acc=…``; tensors are fine) for ``step``. The page refreshes at most every
        ``min_interval`` seconds and snapshots the layers every ``every`` steps. JAX: pass the new ``params=``."""
        s = self._count if step is None else int(step)
        self._count = s + 1
        self.server.update(s, params=params, grads=grads, **metrics)

    def close(self) -> None:
        """Stop serving the page."""
        self.server.stop()

    def __enter__(self) -> "Watch":
        return self

    def __exit__(self, *exc) -> None:
        # a last refresh so the page shows the final weights; the page keeps being served until close() or exit
        self.server.update(render=True)

    def __repr__(self) -> str:
        return f"Watch({self.url})"


def watch(model, inputs, params=None, *, every: int | None = 50, **options) -> Watch:
    """Open a live page for ``model`` on ``inputs`` (a sample batch) and return a :class:`Watch`: call
    ``w.step(step, loss=...)`` from the training loop. ``every``: snapshot the layers every N steps. Other options
    are :func:`deepvis.visualize`'s (``title``, ``labels``, ``tokenizer``, ``min_interval``, ``port`` …)."""
    return Watch(model, inputs, params, every=every, **options)


def _scalars(d: Any) -> dict:
    out = {}
    for k, v in (d or {}).items():
        try:
            out[str(k).replace("/", "_")] = float(v.item() if hasattr(v, "item") else v)
        except (TypeError, ValueError):
            continue
    return out


def lightning(inputs, *, every: int | None = 50, **options):
    """A PyTorch Lightning callback that opens a live page when training starts and logs every batch."""
    try:
        import lightning.pytorch as pl
    except ImportError:
        try:
            import pytorch_lightning as pl
        except ImportError as exc:
            raise ModelError("PyTorch Lightning isn't installed.",
                             "pip install lightning — or use dv.watch(model, x) in a plain training loop.") from exc

    class DeepvisCallback(pl.Callback):
        watch: Watch | None = None

        def on_train_start(self, trainer, pl_module):
            self.watch = Watch(pl_module, inputs, every=every, **options)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
            metrics = _scalars(trainer.callback_metrics)
            if loss is not None:
                metrics["loss"] = float(loss.detach().item() if hasattr(loss, "detach") else loss)
            self.watch.step(trainer.global_step, **metrics)

    return DeepvisCallback()


def huggingface(inputs, *, every: int | None = 50, **options):
    """A HuggingFace ``Trainer`` callback that opens a live page when training starts and logs what the trainer
    logs (``loss``, ``learning_rate``, ``grad_norm`` …)."""
    try:
        from transformers import TrainerCallback
    except ImportError as exc:
        raise ModelError("transformers isn't installed.", "pip install transformers.") from exc

    class DeepvisCallback(TrainerCallback):
        watch: Watch | None = None

        def on_train_begin(self, args, state, control, model=None, **kw):
            self.watch = Watch(model, inputs, every=every, **options)

        def on_step_end(self, args, state, control, **kw):
            if self.watch is not None:
                self.watch.step(state.global_step)

        def on_log(self, args, state, control, logs=None, **kw):
            if self.watch is not None and logs:
                self.watch.step(state.global_step, **_scalars(logs))

    return DeepvisCallback()


__all__ = ["Watch", "watch", "lightning", "huggingface"]
