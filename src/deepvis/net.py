"""The user-facing wrapper: :class:`Net`, :class:`NodeView` (focus) and :class:`Selection`."""

from __future__ import annotations

import contextlib
import difflib
import os
from typing import Any, Callable, Iterable

import numpy as np

from .backends.base import Backend
from .core.graph import ModelGraph, Node
from .core.selectors import filter_paths, is_pattern, match, split_index
from .core.tensor import DEFAULT_STATS
from .manipulate.edits import Edit, compute_edit, load_edits
from .manipulate.history import History, Snapshot, diff, format_diff
from .manipulate.interventions import InterventionHandle, as_intervention
from .manipulate.params import EditMethods, NeuronSelection, ParamRef
from .monitor.recorder import Monitor, Trace


def _as_args(inputs: tuple, net: "Net | None" = None) -> tuple:
    if len(inputs) == 1 and isinstance(inputs[0], tuple):
        return inputs[0]
    if len(inputs) == 1 and net is not None:
        from . import errors as E

        if E._is_mapping(inputs[0]):  # a tokenizer's output: bound to the forward's arguments by name
            model = getattr(net.backend, "model", None)
            bound = E._bind_dict(model, inputs[0], net.framework) if model is not None else None
            if bound is not None:
                return bound
    return inputs


class Net:
    """A wrapped PyTorch / JAX network.

    Build with :func:`deepvis.wrap`. Highlights::

        net.summary()                                   # module tree
        net["encoder.block2"]                           # focus on a sub-network (NodeView)
        net["fc1.weight"][:, 3] = 0                     # arbitrary weight surgery (undoable)
        net.select("layers.*.fc*", kind="Linear").prune(0.3)
        tr = net.trace(x)                               # every intermediate activation
        with net.monitor("relu*") as mon: net(x)        # persistent stats
        with net.intervene("relu1", dv.ablate([2])): net(x)
        net.equation(mode="expanded")                   # unroll into math
        net.show(x)                                     # interactive 3b1b-style dashboard
        net.animate("forward", x, path="fwd.mp4")       # manim video
    """

    def __init__(self, backend: Backend, sample_input: Any = None):
        self.backend = backend
        self.history = History(self)
        self._monitors: list[Monitor] = []
        self._sample = None if sample_input is None else (
            sample_input if isinstance(sample_input, tuple) else (sample_input,))
        self._last_args: tuple | None = None
        self._version = 0
        self.tokenizer = None  # optional: token strings / next-token predictions in pages (see export_html)
        self.labels = None  # optional: class names for the output
        self.page: str | None = None  # the file dv.visualize() wrote

    # --------------------------------------------------------------- basics
    @property
    def framework(self) -> str:
        return self.backend.framework

    @property
    def name(self) -> str:
        return self.backend.name

    @property
    def graph(self) -> ModelGraph:
        if self.backend._graph is None:
            if self._sample is not None:
                self.backend.build_graph(*self._sample)
            else:
                self.backend.build_graph()
        return self.backend._graph

    def rebuild_graph(self, *inputs) -> ModelGraph:
        """Re-trace the graph (e.g. with a new sample input to fill in shapes)."""
        args = _as_args(inputs, self) if inputs else self._sample
        if args:
            self._sample = args
        return self.backend.build_graph(*(args or ()))

    def __call__(self, *args, **kwargs):
        if len(args) == 1:
            from . import errors as E

            if E._is_mapping(args[0]):  # net(tokenizer_output): bound to the forward's arguments by name
                args = _as_args(args, self)
        self._last_args = args
        if self._sample is None and args:
            self._sample = args
        return self.backend.forward(*args, **kwargs)

    @property
    def sample_input(self) -> tuple | None:
        return self._last_args or self._sample

    def _invalidate(self) -> None:
        self._version += 1

    def export(self):
        """The (edited) model in native form: torch Module, (module, variables), eqx/nnx model, or params."""
        return self.backend.export()

    def load(self, native) -> "Net":
        """Load new weights in native form, e.g. after an optimizer step in a JAX training loop.

        Accepts what :meth:`export` returns: flax ``variables`` (or ``params``), an nnx / equinox model, a params
        pytree, or a torch ``state_dict`` / module. Not recorded as an edit.
        """
        self.backend.load(native)
        self._invalidate()
        return self

    def clone(self) -> "Net":
        """Independent copy (weights copied); history is not carried over."""
        return Net(self.backend.clone(), self._sample)

    # ------------------------------------------------------------ navigation
    def __getitem__(self, key: str):
        path, idx = split_index(key)
        params = self.backend.param_paths()
        if path in params:
            return ParamRef(self, path, idx)
        g = self.graph
        if path in g.nodes and idx is None:
            return NodeView(self, path)
        if is_pattern(path):
            return self.select(path)
        close = difflib.get_close_matches(path, list(g.nodes) + params, n=5)
        raise KeyError(f"{key!r} is neither a node nor a parameter. Did you mean: {close}?")

    def __setitem__(self, key: str, value) -> None:
        if isinstance(value, (ParamRef, NodeView, Selection)):
            return  # result of an in-place operator (e.g. net["w"] *= 2) — already applied
        target = self[key]
        if not isinstance(target, ParamRef):
            raise TypeError("assignment is supported on parameters (e.g. net['fc1.weight'][:, 0] = 0)")
        target.set(value)

    def __contains__(self, key: str) -> bool:
        return key in self.graph.nodes or key in self.backend.param_paths()

    def focus(self, path: str) -> "NodeView":
        return NodeView(self, path)

    @property
    def root(self) -> "NodeView":
        return NodeView(self, "")

    def params(self, pattern: str | None = None) -> dict[str, ParamRef]:
        paths = self.backend.param_paths()
        if pattern:
            paths = filter_paths(pattern, paths)
        return {p: ParamRef(self, p) for p in paths}

    def select(self, pattern: str | Iterable[str] | None = None, kind: str | Iterable[str] | None = None,
               where: Callable[[Node], bool] | None = None, leaves: bool = True) -> "Selection":
        """Select nodes by path glob/regex, canonical kind and/or predicate."""
        g = self.graph
        kinds = {kind} if isinstance(kind, str) else set(kind) if kind else None
        pats = [pattern] if isinstance(pattern, str) else list(pattern) if pattern else None
        ids = []
        for n in g.nodes.values():
            if leaves and not n.is_compute:
                continue
            if n.op in ("input", "output") and pats is None:
                continue
            if pats and not any(match(p, n.id) for p in pats):
                continue
            if kinds and n.kind not in kinds and n.type_name not in kinds:
                continue
            if where and not where(n):
                continue
            ids.append(n.id)
        return Selection(self, ids)

    def _resolve_nodes(self, targets) -> list[str]:
        if isinstance(targets, (NodeView,)):
            return [targets.path]
        if isinstance(targets, Selection):
            return list(targets.ids)
        if isinstance(targets, str):
            targets = [targets]
        g = self.graph
        out = []
        for t in targets:
            if isinstance(t, NodeView):
                out.append(t.path)
            elif t in g.nodes:
                out.append(t)
            else:
                m = [n for n in filter_paths(t, list(g.nodes)) if g.nodes[n].is_compute]
                if not m:
                    close = difflib.get_close_matches(t, list(g.nodes), n=5)
                    raise KeyError(f"no node matches {t!r}. Did you mean: {close}?")
                out.extend(m)
        return list(dict.fromkeys(out))

    def summary(self, show: bool = True) -> str:
        """The network in one plain sentence, then a table of its layers, shapes and parameter counts."""
        from .viz.explain import describe

        s = describe(self.graph, batched=self.framework != "equinox") + "\n" + self.graph.summary()
        if show:
            print(s)
        return s

    def num_params(self, trainable_only: bool = True) -> int:
        """Parameter count; ``trainable_only=False`` also counts buffers (e.g. BatchNorm running statistics)."""
        infos = (self.backend.param_info(p) for p in self.backend.param_paths())
        return sum(int(np.prod(i.shape)) for i in infos if i.trainable or not trainable_only)

    def __repr__(self) -> str:
        return (f"deepvis.Net({self.name}, framework={self.framework}, params={self.num_params():,}, "
                f"edits={len(self.history)})")

    def _repr_html_(self) -> str | None:
        """Jupyter: evaluating ``net`` shows its expressive architecture figure (``net.show(x)`` is interactive)."""
        import base64
        from html import escape

        cached = getattr(self, "_repr_cache", None)
        if cached and cached[0] == self._version:
            return cached[1]
        try:
            from .viz.diagram import diagram_svg

            args = self.sample_input
            svg = diagram_svg(self, args if args else None)
        except Exception:  # noqa: BLE001 - fall back to the text repr
            return None
        # an <img> keeps the figure's CSS from leaking into the notebook
        b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
        out = (f'<div style="max-height:640px;overflow:auto;border:1px solid #333;border-radius:8px;background:#000">'
               f'<img alt="{escape(self.name)} architecture" src="data:image/svg+xml;base64,{b64}" '
               f'style="display:block;max-width:none"></div>'
               f'<div style="font:12px monospace;color:#888;margin-top:4px">{escape(repr(self))} · '
               f'net.show(x) opens the interactive dashboard</div>')
        self._repr_cache = (self._version, out)
        return out

    # ----------------------------------------------------------- manipulation
    def apply_edit(self, edit: Edit | dict) -> "Net":
        """Apply one edit (recorded for undo/redo)."""
        if isinstance(edit, dict):
            edit = Edit.from_dict(edit)
        new = compute_edit(self, edit)
        before = {p: self.backend.get_param(p) for p in new}
        for p, v in new.items():
            self.backend.set_param(p, v)
        self.history.record(edit, before, {p: v.copy() for p, v in new.items()})
        self._invalidate()
        return self

    def apply_edits(self, edits) -> "Net":
        """Apply a list of edits / dicts, or a JSON file / string of them (see ``net.history.export()``)."""
        for e in load_edits(edits):
            self.apply_edit(e)
        return self

    def edit(self, target: str, op: str | Callable, where=None, index=None, params=None, **args) -> "Net":
        """``net.edit("fc1.weight", "scale", factor=2)`` or ``net.edit("fc*", lambda w: w.clip(-1, 1))``."""
        if callable(op):
            args = {"fn": op}
            op = "apply"
        path, idx = split_index(target)
        return self.apply_edit(Edit(op, path, index=index if index is not None else idx, args=args,
                                    where=where, params=params))

    @contextlib.contextmanager
    def edits(self):
        """Temporary edits: everything applied inside the block is undone on exit."""
        start = self.history.pos
        try:
            yield self
        finally:
            self.history.undo(self.history.pos - start)
            del self.history.entries[start:]

    def undo(self, n: int = 1):
        return self.history.undo(n)

    def redo(self, n: int = 1):
        return self.history.redo(n)

    def reset(self) -> None:
        """Restore all edited parameters to their original values."""
        self.history.reset()

    def snapshot(self, label: str = "") -> Snapshot:
        return Snapshot({p: self.backend.get_param(p) for p in self.backend.param_paths()}, label)

    def restore(self, snap: Snapshot) -> "Net":
        changed = {p: v for p, v in snap.values.items() if not np.array_equal(self.backend.get_param(p), v)}
        if changed:
            self.apply_edit(Edit("load", list(changed), args={"values": changed},
                                 label=f"restore snapshot {snap.label}".strip()))
        return self

    def diff(self, reference: Snapshot | None = None, show: bool = True) -> list[dict]:
        rows = diff(self, reference)
        if show:
            print(format_diff(rows))
        return rows

    def permute_neurons(self, node: str, perm) -> "Net":
        """Permute a layer's output neurons and its consumers' inputs (function-preserving)."""
        return self.apply_edit(Edit("permute_neurons", node, args={"perm": np.asarray(perm).tolist()}))

    def interpolate(self, other: "Net", alpha: float = 0.5, select: str = "**") -> "Net":
        """Move parameters towards ``other``'s: θ ← (1-α)θ + αθ_other (same architecture)."""
        return self.apply_edit(Edit("interpolate", select, args={"other": other, "alpha": alpha,
                                                                 "select": select},
                                    label=f"interpolate α={alpha} towards {other.name}"))

    def transplant(self, other: "Net", target: str, mapping: dict | None = None) -> "Net":
        """Copy parameters of ``target`` (node or param pattern) from ``other``."""
        return self.apply_edit(Edit("transplant", target, args={"other": other, "mapping": mapping},
                                    label=f"transplant {target} from {other.name}"))

    # ---------------------------------------------------------- activations
    def intervene(self, targets, *interventions) -> InterventionHandle:
        """Attach activation interventions (persistent until ``.remove()`` / end of ``with``)."""
        ids = self._resolve_nodes(targets)
        fns = [as_intervention(i) for i in interventions]
        handles = [self.backend.add_intervention(nid, fn) for nid in ids for fn in fns]
        return InterventionHandle(self, handles, f"{', '.join(map(repr, fns))} @ {ids}")

    def profile(self, data, labels=None, **kw):
        """The network over a dataset: per-unit statistics, dead units, the inputs that excite each unit most, a 2-D
        map of the samples per layer and layer similarity (CKA) — :mod:`deepvis.monitor.profile`. Show it with
        ``net.export_html(path, x, dataset=profile)``."""
        from .monitor.profile import profile

        return profile(self, data, labels, **kw)

    def monitor(self, *targets, stats=DEFAULT_STATS, keep: str = "last", grads: bool = False,
                every: int = 1, start: bool = True) -> Monitor:
        """Persistent activation monitor on nodes matching ``targets`` (default: all compute nodes)."""
        if targets:
            ids = self._resolve_nodes(list(targets))
        else:
            ids = [n.id for n in self.graph.compute_nodes() if n.op in ("module", "function")]
        ops = ("module", "function") if self.backend.function_taps else ("module",)
        if targets and not self.backend.function_taps:
            bad = [i for i in ids if self.graph.nodes[i].op == "function"]
            if bad:
                raise ValueError(f"{self.framework} monitors attach to modules; {bad} are functional ops. "
                                 "Use net.trace(x) to record those.")
        ids = [i for i in ids if self.graph.nodes[i].op in ops]
        mon = Monitor(self, ids, tuple(stats), keep=keep, grads=grads, every=every)
        return mon.start() if start else mon

    def trace(self, *inputs, record: Any = "all", grads: bool = False, loss_fn: Callable | None = None,
              capture_attention: bool = True, interventions: dict | None = None, **kwargs) -> Trace:
        """Run once and record intermediates (``record`` = "all", node ids or patterns)."""
        args = _as_args(inputs, self) if inputs else self.sample_input
        if args is None:
            raise ValueError("trace() needs an input")
        if not self.backend.has_data:
            from .errors import OptionError

            raise OptionError("this model is on the meta device: it has shapes but no numbers to run.",
                              "its page shows the architecture, shapes and equations (net.export_html(path)); load "
                              "real weights to trace it.")
        before = self._sample  # a run that fails leaves the sample input as it was
        if self._sample is None:
            self._sample = args
        try:
            return self._trace(args, record, grads, loss_fn, capture_attention, interventions, kwargs)
        except BaseException:
            self._sample = before
            raise

    def _trace(self, args, record, grads, loss_fn, capture_attention, interventions, kwargs) -> Trace:
        g = self.graph
        if not any(n.out_shapes for n in g.compute_nodes()):
            self.rebuild_graph(*args)
        rec = record if record in ("all", None, True, False) else self._resolve_nodes(record)
        ints = {}
        for k, v in (interventions or {}).items():
            for nid in self._resolve_nodes(k):
                ints.setdefault(nid, []).extend(as_intervention(i) for i in (v if isinstance(v, (list, tuple))
                                                                              else [v]))
        from . import errors as E

        if self.framework == "torch":
            args = E.prepare_inputs(self.backend.model, args, "torch")
        try:
            res = self.backend.run(args, kwargs, record=rec, record_inputs=True, interventions=ints, grads=grads,
                                   loss_fn=loss_fn, capture_attention=capture_attention)
        except E.DeepvisError:
            raise
        except Exception as exc:  # noqa: BLE001
            failed = E.probe(self.backend.model, args, "torch") if self.framework == "torch" else exc
            if failed is not None:
                raise E.run_error(failed, args) from failed
            raise E.trace_error(exc) from exc
        self._last_args = args
        return Trace(self, args, kwargs, res)

    def attribute(self, *inputs, method: str = "integrated_gradients", target: int | None = None, sample: int = 0,
                  **kw):
        """Which parts of the input made the model predict ``target`` (default: its prediction): an
        :class:`~deepvis.interpret.Attribution` with ``.map`` shaped like the input (see :mod:`deepvis.interpret`)."""
        from .interpret.attribution import attribute

        return attribute(self, *inputs, method=method, target=target, sample=sample, **kw)

    def head_importance(self, *inputs, sample: int = 0, max_heads: int = 256):
        """How much the output moves when each attention head is switched off (see :mod:`deepvis.interpret`)."""
        from .interpret.attention import head_importance

        return head_importance(self, _as_args(inputs, self) if inputs else self.sample_input, sample=sample,
                               max_heads=max_heads)

    def logit_lens(self, *inputs, **kw):
        """What a language model would predict after each block (see :mod:`deepvis.monitor.lens`)."""
        from .monitor.lens import logit_lens

        return logit_lens(self, *(inputs or self.sample_input or ()), **kw)

    # -------------------------------------------------------------- equations
    def equation(self, mode: str = "system", **kw):
        """Unroll the network into math. Modes: compact | system | expanded | scalar."""
        from .equations.unroll import unroll

        return unroll(self, mode=mode, **kw)

    # ------------------------------------------------------------------ viz
    def show(self, *inputs, focus: str | None = None, trace: Trace | None = None,
             monitor: Monitor | None = None, open_browser: bool | None = None, **kw):
        """Interactive dashboard (inline in Jupyter, otherwise opens a browser tab).

        Accepts the :meth:`export_html` options (views, start, style, title, note, depth, tensor_shapes).
        """
        from .viz.html.dashboard import Dashboard

        args = _as_args(inputs, self) if inputs else None
        return Dashboard(self, args, focus=focus, trace=trace, monitor=monitor, **kw).display(open_browser)

    def export_html(self, path: str, *inputs, views=None, start: str | None = None, style: str = "expressive",
                    title: str | None = None, note: str | None = None, depth=None, tensor_shapes: bool = True,
                    trace: Trace | None = None, monitor: Monitor | None = None, focus: str | None = None,
                    training=None, sample: int = 0, grads: bool = False, loss_fn: Callable | None = None,
                    tokenizer=None, labels=None, tokens=None, diffusion=None, attribution: bool = False,
                    heads: bool | None = None, compare=None, compare_names=None, lens_fn=None, dataset=None,
                    offline: bool | None = None, open_browser: bool = False) -> str:
        """Write one self-contained, interactive HTML file and return its path.

        views          which tabs to include: any of "graph", "volumes", "neurons", "equations", "monitor"
        start          the tab the page opens on
        style          "expressive" (data-driven glyphs: activation curves, weight fans, 3-D tensors …) or "compact"
        title, note    header title and a short description shown in the overview panel
        depth          initial detail level: an int (collapse containers deeper than it) or "all"
        tensor_shapes  annotate edges with 3-D tensor cuboids
        training       a :class:`~deepvis.TrainingLog`: loss curves and per-layer learning dynamics (Monitor tab)
        sample         which sample of a batched input the views show (activations, glyphs, outputs)
        grads          also backpropagate ``loss_fn(output)`` (default: the output's sum): gradient flow per layer
                       (Monitor tab), ∂L/∂(output) of every layer and an input saliency map (side panel)
        tokenizer      a HuggingFace tokenizer (or ``id -> str``): token strings on the input and attention maps,
                       and the top next-token predictions of a language model (default: ``net.tokenizer``)
        labels         class names for the output (default: ``net.labels``); ``tokens`` labels the input directly
        diffusion      a :class:`~deepvis.diffusion.DiffusionRun`: the Diffusion tab (denoising process, x̂₀, ε̂)
        attribution    explain the top predictions: integrated gradients, saliency and Grad-CAM maps of the input
                       (PyTorch; the Output panel)
        heads          head importance in the Attention tab (each head switched off in turn): None = when the model
                       has ≤ 48 heads, True / False to force it
        compare        another version of this network — a model, a state_dict or a checkpoint path — for the
                       Compare tab: how far each layer's weights and activations moved, both outputs;
                       ``compare_names`` = ("before", "after") labels them
        lens_fn        ``h -> logits`` for the logit lens of a language model whose unembedding deepvis can't run
                       itself (JAX): e.g. ``lambda h: model.apply(variables, h, method=Model.unembed)``
        dataset        a :class:`~deepvis.monitor.profile.DatasetProfile` from ``net.profile(data, labels)``: the
                       Dataset tab (a map of the samples per layer, layer similarity, dead units) and each layer's
                       most exciting inputs in its panel
        offline        inline the bundled KaTeX (~0.7 MB) so the page makes no network requests; default:
                       ``$DEEPVIS_OFFLINE`` (otherwise math loads from a CDN, with a plain-text fallback)
        """
        import os
        import webbrowser

        from .viz.html.dashboard import Dashboard

        args = _as_args(inputs, self) if inputs else (trace.args if trace is not None else self.sample_input)
        d = Dashboard(self, args, focus=focus, trace=trace, monitor=monitor, style=style, tensor_shapes=tensor_shapes,
                      views=views, start=start, title=title, note=note, depth=depth, training=training, sample=sample,
                      grads=grads, loss_fn=loss_fn, tokenizer=tokenizer, labels=labels, tokens=tokens,
                      diffusion=diffusion, attribution=attribution, heads=heads, compare=compare,
                      compare_names=compare_names, lens_fn=lens_fn, dataset=dataset, offline=offline)
        out = d.save(path)
        if open_browser:
            webbrowser.open("file://" + os.path.abspath(out))
        return out

    def save_html(self, path: str, *inputs, **kw) -> str:
        """Alias of :meth:`export_html`."""
        return self.export_html(path, *inputs, **kw)

    def export_svg(self, path: str, *inputs, depth=None, direction: str = "TB", title: str | None = None,
                   subtitle: str | None = None, tensor_shapes: bool = True, legend: bool = True) -> str:
        """Write a static architecture figure with the expressive glyphs: ``.svg``, ``.png`` or ``.pdf`` (for papers
        and slides; needs ``cairosvg``), or ``.html`` wrapping the SVG.

        With inputs the glyphs are data-driven (real activation ranges, input images, attention maps, outputs).
        """
        from .viz.diagram import diagram_html, diagram_svg, svg_to_file

        args = _as_args(inputs, self) if inputs else self.sample_input
        svg = diagram_svg(self, args, depth=depth, direction=direction, title=title, subtitle=subtitle,
                          tensor_shapes=tensor_shapes, legend=legend)
        low = str(path).lower()
        if low.endswith((".png", ".pdf")):
            return svg_to_file(svg, str(path))
        text = diagram_html(svg, title or self.name) if low.endswith((".html", ".htm")) else svg
        from .viz.html.dashboard import write_text

        return write_text(path, text)

    def widget(self, *inputs, height: int = 820, **kw):
        """The live page as a notebook widget — weight edits, interventions, the input playground, training — in
        JupyterLab, Jupyter, VS Code notebooks and Colab, with no web server (needs ``anywidget``). Takes
        :meth:`serve`'s options; ``widget.update(step, loss=…)`` records training progress."""
        from .viz.server import LiveServer
        from .viz.widget import make_widget

        args = _as_args(inputs, self) if inputs else self.sample_input
        return make_widget(LiveServer(self, args, **kw), height=height)

    def serve(self, *inputs, port: int = 0, open_browser: bool = True, **kw):
        """Live dashboard with in-browser weight editing backed by this Python process.

        Returns a :class:`~deepvis.viz.server.LiveServer`; call ``srv.update(step, loss=...)`` from a training
        loop and the open page follows along (loss curves, per-layer learning dynamics, refreshed views).

        Or let the page drive training: ``train_step=fn`` (``fn(step) -> {"loss": ..., ...}``, one optimization
        step) adds Train / Pause / Step / Reset buttons; ``train_steps`` caps the run, ``train_reset()`` is called
        after Reset restores the initial weights (e.g. to rebuild the optimizer), ``train_delay`` slows it down.
        """
        from .viz.server import LiveServer

        args = _as_args(inputs, self) if inputs else self.sample_input
        return LiveServer(self, args, port=port, **kw).start(open_browser=open_browser)

    def animate(self, kind: str = "forward", *inputs, path: str | None = None, quality: str = "low", **kw):
        """Render a manim animation: forward | equation | focus | edit | architecture | volumes | training.

        The videos narrate themselves along the bottom of the frame (plain words, real numbers); ``captions=False``
        turns that off."""
        from .anim.render import animate

        args = _as_args(inputs, self) if inputs else self.sample_input
        return animate(self, kind, args, path=path, quality=quality, **kw)


class NodeView(EditMethods):
    """A focused sub-network / layer. Most :class:`Net` verbs are scoped to it."""

    def __init__(self, net: Net, path: str):
        if path not in net.graph.nodes:
            raise KeyError(path)
        self.net = net
        self.path = path

    @property
    def node(self) -> Node:
        return self.net.graph.nodes[self.path]

    @property
    def kind(self) -> str:
        return self.node.kind

    @property
    def children(self) -> list["NodeView"]:
        return [NodeView(self.net, c) for c in self.node.children]

    @property
    def graph(self) -> ModelGraph:
        return self.net.graph.subgraph(self.path)

    def __getitem__(self, key: str):
        return self.net[f"{self.path}.{key}" if self.path else key]

    def __setitem__(self, key: str, value) -> None:
        self.net[f"{self.path}.{key}" if self.path else key] = value

    def params(self) -> dict[str, ParamRef]:
        if self.node.is_compute:
            return {k: ParamRef(self.net, v.path) for k, v in self.node.params.items()}
        prefix = f"{self.path}." if self.path else ""
        return {p[len(prefix):]: ParamRef(self.net, p) for p in self.net.backend.param_paths()
                if p.startswith(prefix)}

    def _role(self, role: str) -> ParamRef:
        info = self.node.role_param(role)
        if info is None:
            raise AttributeError(f"{self.path!r} ({self.kind}) has no {role!r} parameter")
        return ParamRef(self.net, info.path)

    @property
    def weight(self) -> ParamRef:
        return self._role("weight")

    @property
    def bias(self) -> ParamRef:
        return self._role("bias")

    def weight_matrix(self) -> np.ndarray:
        """Weight in canonical (out, in, ...) orientation regardless of framework."""
        r = self.node.roles["weight"]
        w = self.weight.numpy()
        if r.out_axis is None:
            return w
        order = [r.out_axis % w.ndim] + ([r.in_axis % w.ndim] if r.in_axis is not None else [])
        order += [i for i in range(w.ndim) if i not in order]
        return np.transpose(w, order)

    def neurons(self, idx) -> NeuronSelection:
        return NeuronSelection(self.net, self.path, idx)

    def neuron(self, i: int) -> NeuronSelection:
        return NeuronSelection(self.net, self.path, [i])

    def prune_neurons(self, amount: float = 0.5, by: str = "l2") -> "NodeView":
        """Structured pruning: remove (zero) the output neurons with the smallest weight norm."""
        w = self.weight_matrix()
        flat = w.reshape(w.shape[0], -1)
        score = np.linalg.norm(flat, axis=1) if by == "l2" else np.abs(flat).sum(1)
        k = int(round(amount * len(score)))
        if k:
            self.neurons(np.argsort(score)[:k].tolist()).zero()
        return self

    def _edit(self, op, args, where=None, params=None, label=None):
        if self.node.is_container:
            target = f"{self.path}.**" if self.path else "**"
        else:
            target = self.path
        self.net.apply_edit(Edit(op, target, args=args, where=where, params=params, label=label))
        return self

    def intervene(self, *interventions) -> InterventionHandle:
        return self.net.intervene(self.path, *interventions)

    def monitor(self, **kw) -> Monitor:
        ids = [self.path] if self.node.is_compute else [
            d for d in self.net.graph.descendants(self.path) if self.net.graph.nodes[d].op in ("module", "function")]
        return self.net.monitor(*ids, **kw)

    def trace(self, *inputs, **kw) -> Trace:
        ids = [self.path] if self.node.is_compute else [
            d for d in self.net.graph.descendants(self.path) if self.net.graph.nodes[d].is_compute]
        return self.net.trace(*inputs, record=ids, **kw)

    def equation(self, mode: str = "system", **kw):
        return self.net.equation(mode=mode, focus=self.path, **kw)

    def show(self, *inputs, **kw):
        return self.net.show(*inputs, focus=self.path, **kw)

    def animate(self, kind: str = "focus", *inputs, **kw):
        return self.net.animate(kind, *inputs, focus=self.path, **kw)

    def __repr__(self) -> str:
        n = self.node
        shp = f" → {tuple(n.out_shapes[0])}" if n.out_shapes and n.out_shapes[0] else ""
        return f"NodeView({self.path or '<root>'}: {n.kind}{shp}, params={n.n_params:,})"


class Selection(EditMethods):
    """Several nodes at once; edits and interventions apply to each."""

    def __init__(self, net: Net, ids: list[str]):
        self.net = net
        self.ids = ids

    def __len__(self) -> int:
        return len(self.ids)

    def __iter__(self):
        return (NodeView(self.net, i) for i in self.ids)

    def __getitem__(self, i):
        return NodeView(self.net, self.ids[i])

    @property
    def paths(self) -> list[str]:
        return list(self.ids)

    def params(self) -> dict[str, ParamRef]:
        out = {}
        for i in self.ids:
            out.update({v.path: ParamRef(self.net, v.path) for v in self.net.graph.nodes[i].params.values()})
        return out

    def _edit(self, op, args, where=None, params=None, label=None):
        targets = [i for i in self.ids if self.net.graph.nodes[i].params]
        if not targets:
            raise ValueError("selection contains no parameterized nodes")
        self.net.apply_edit(Edit(op, targets, args=args, where=where, params=params, label=label))
        return self

    def intervene(self, *interventions) -> InterventionHandle:
        return self.net.intervene(self.ids, *interventions)

    def monitor(self, **kw) -> Monitor:
        return self.net.monitor(*self.ids, **kw)

    def show(self, *inputs, **kw):
        return self.net.show(*inputs, highlight=self.ids, **kw)

    def __repr__(self) -> str:
        return f"Selection({len(self.ids)} nodes: {self.ids[:6]}{'…' if len(self.ids) > 6 else ''})"


# ---------------------------------------------------------------------- wrap
def wrap(model, params=None, *, sample_input=None, name: str | None = None, framework: str | None = None,
         **kw) -> Net:
    """Wrap a network.

    * ``wrap(torch_module)``
    * ``wrap(flax_linen_module, variables, sample_input=x)``
    * ``wrap(nnx_module, sample_input=x)``
    * ``wrap(equinox_module, sample_input=x)``
    * ``wrap(apply_fn, params, sample_input=x)``  — any ``apply_fn(params, *inputs)``; ``wrap((apply_fn, params))``
      and ``wrap((linen_module, variables))`` work too
    * ``wrap(keras_model, sample_input=x)``  — Keras 3, on the JAX, PyTorch or TensorFlow backend
    * ``wrap("model.onnx")``  — an ONNX file (or ``onnx.ModelProto``), run in onnxruntime
    """
    from . import errors as E

    if params is None and isinstance(model, tuple) and len(model) == 2 and callable(model[0]):
        model, params = model
    if isinstance(model, (str, os.PathLike)) and str(model).lower().endswith(".onnx"):
        framework = framework or "onnx"  # an ONNX file: its path is the model
    else:
        from .files import is_model_file, open_file

        if is_model_file(model):  # .keras / .pt2 / .safetensors …: dv.open knows how to read it
            return open_file(model, sample_input=sample_input, name=name)
        E.check_model(model)
    if type(model).__name__ == "ExportedProgram" and type(model).__module__.startswith("torch.export"):
        from .backends.torch_backend import from_exported

        model, example = from_exported(model)  # torch.export: its module tree, classes and example inputs
        if sample_input is None:
            sample_input = example
        name = name or "exported"
    fw = framework or _detect(model, params)
    if sample_input is not None and fw != "unknown":
        sample_input = E.prepare_inputs(model, sample_input, fw)
    if fw == "torch":
        from .backends.torch_backend import TorchBackend

        backend = TorchBackend(model, name=name, sample_input=sample_input, **kw)
    elif fw == "flax":
        from .backends.jax.flax_linen import LinenBackend

        if params is None:
            raise E.ModelError("a flax.linen module has no weights of its own; deepvis needs its variables.",
                               "pass them: dv.visualize(module, x, params=variables), where "
                               "variables = module.init(jax.random.PRNGKey(0), x).")
        backend = LinenBackend(model, params, sample_input=sample_input, name=name, **kw)
    elif fw == "nnx":
        from .backends.jax.flax_nnx import NNXBackend

        backend = NNXBackend(model, sample_input=sample_input, name=name, **kw)
    elif fw == "equinox":
        from .backends.jax.equinox_backend import EquinoxBackend

        backend = EquinoxBackend(model, sample_input=sample_input, name=name, **kw)
    elif fw == "keras":
        from .backends.keras_backend import KerasBackend

        backend = KerasBackend(model, sample_input=sample_input, name=name)
    elif fw == "onnx":
        from .backends.onnx_backend import OnnxBackend

        backend = OnnxBackend(model, name=name, sample_input=sample_input)
        if sample_input is None:  # an ONNX model declares its inputs: make some, so the page has data
            sample_input = backend._sample = backend.example_inputs()
    elif fw == "jax":
        from .backends.jax.pure import PureBackend

        if params is None:
            raise E.ModelError("a plain function needs its parameters to be wrapped.",
                               "pass them: dv.visualize((apply_fn, params), x) — apply_fn(params, x) must run.")
        backend = PureBackend(model, params, sample_input=sample_input, name=name, **kw)
    else:
        raise E.unknown_model(model)
    try:
        net = Net(backend, sample_input)
        if sample_input is not None:
            net.graph  # noqa: B018 - build eagerly so shapes are known
    except E.DeepvisError:
        raise
    except Exception as exc:  # noqa: BLE001 - attribute the failure: the model's, or deepvis's tracing
        if sample_input is None:
            raise
        failed = E.probe(model, sample_input, fw)
        if failed is not None:
            raise E.run_error(failed, sample_input) from failed
        if fw != "torch":
            raise E.run_error(exc, sample_input) from exc
        raise E.trace_error(exc) from exc
    return net


def _detect(model, params) -> str:
    mro = [f"{c.__module__}.{c.__name__}" for c in type(model).__mro__]
    if any(m.startswith("keras.") and m.endswith((".Layer", ".Model")) for m in mro):
        return "keras"  # before torch: on the PyTorch backend a Keras layer is also an nn.Module
    if any(m.startswith("torch.nn.modules.module.Module") for m in mro):
        return "torch"
    if any(m.startswith("flax.linen") for m in mro):
        return "flax"
    if any(m.startswith("flax.nnx") for m in mro):
        return "nnx"
    if any(m.startswith("equinox") for m in mro):
        return "equinox"
    if any(m.startswith("onnx") and m.endswith("ModelProto") for m in mro):
        return "onnx"
    if callable(model) and params is not None:
        return "jax"
    return "unknown"
