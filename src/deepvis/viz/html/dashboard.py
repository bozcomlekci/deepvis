"""Self-contained HTML dashboard (one file). Math is typeset by KaTeX: from a CDN by default (a plain-text fallback
offline), or inlined from the bundled copy with ``offline=True`` / ``$DEEPVIS_OFFLINE=1`` (the VS Code extension)."""

from __future__ import annotations

import html as _html
import json
import math
import os
import tempfile
import base64
import functools
import re
import webbrowser
from importlib import resources
from typing import TYPE_CHECKING

from .payload import PayloadBuilder

if TYPE_CHECKING:  # pragma: no cover
    from ...net import Net


def _asset(name: str) -> str:
    return resources.files("deepvis.viz.html").joinpath("assets").joinpath(name).read_text(encoding="utf-8")


OFFLINE_ENV = "DEEPVIS_OFFLINE"  # "1": pages inline the bundled KaTeX instead of loading it from the CDN
KATEX_VERSION = "0.16.11"
KATEX_CDN = f"https://cdn.jsdelivr.net/npm/katex@{KATEX_VERSION}/dist"


def offline_default() -> bool:
    return os.environ.get(OFFLINE_ENV, "").strip().lower() not in ("", "0", "false", "no")


@functools.lru_cache(maxsize=1)
def katex_inline() -> str:
    """The bundled KaTeX as one ``<style>`` + ``<script>``, its fonts as ``data:`` URIs (woff2): no network."""
    root = resources.files("deepvis.viz.html").joinpath("assets").joinpath("katex")

    def font(m: re.Match) -> str:
        data = base64.b64encode(root.joinpath("fonts").joinpath(m.group(1) + ".woff2").read_bytes()).decode("ascii")
        return f'src:url(data:font/woff2;base64,{data}) format("woff2")'

    css = re.sub(r"src:url\(fonts/([\w-]+)\.woff2\)[^;}]*", font, root.joinpath("katex.min.css").read_text("utf-8"))
    js = root.joinpath("katex.min.js").read_text("utf-8").replace("</script", "<\\/script")
    return f"<style>{css}</style>\n<script>{js}</script>"


def katex_tags(offline: bool | None = None) -> str:
    """The ``<head>`` tags that load KaTeX: the CDN, or (``offline``) the bundled copy inlined."""
    if offline is None:
        offline = offline_default()
    if offline:
        return katex_inline()
    return (f'<link rel="stylesheet" href="{KATEX_CDN}/katex.min.css" crossorigin="anonymous">\n'
            f'<script src="{KATEX_CDN}/katex.min.js" crossorigin="anonymous"></script>')


def render_page(payload: dict, api: str | None = None, title: str = "", bridge: bool = False,
                widget: bool = False, offline: bool | None = None) -> str:
    """The page for ``payload``. ``bridge``: it will be shown framed in an editor panel (VS Code) or a notebook
    widget, which saves its downloads and opens its links. ``widget``: its server requests go to the notebook
    widget around it (see :mod:`deepvis.viz.widget`). ``offline``: inline the bundled KaTeX rather than load it
    from the CDN (default: ``$DEEPVIS_OFFLINE``); the page then makes no network requests, at ~0.7 MB more."""
    data = dumps(payload, separators=(",", ":"))
    data = data.replace("</", "<\\/")  # never terminate the <script> early
    page = _asset("template.html")
    if widget:
        page = page.replace("<head>", "<head>\n<script>" + _asset("widget.js") + "</script>", 1)
    if bridge:
        page = page.replace("<head>", "<head>\n<script>" + _asset("bridge.js") + "</script>", 1)
    page = page.replace("__TITLE__", _html.escape(title or payload["meta"]["name"]))
    from ..glyphs import GLYPH_CSS

    page = page.replace("__CSS__", _asset("style.css") + GLYPH_CSS)
    page = page.replace("__API__", json.dumps(api))
    page = page.replace("__VOL__", _asset("volume3d.js"))
    page = page.replace("__JS__", _asset("app.js"))
    page = page.replace("__KATEX__", katex_tags(offline), 1)  # after the other markers: KaTeX's text is never scanned
    return page.replace("__PAYLOAD__", data)  # last: payload may contain the other markers


def dumps(obj, **kw) -> str:
    """Strict JSON (what browsers parse): NaN / ±inf — a diverged run, a bad input — become null."""
    try:
        return json.dumps(obj, allow_nan=False, default=_json_default, **kw)
    except ValueError:
        return json.dumps(_finite(obj), allow_nan=False, default=_json_default, **kw)


def _finite(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    if type(o).__module__ == "numpy" and hasattr(o, "tolist"):  # numpy scalars and arrays
        return _finite(o.tolist())
    return o


def _json_default(o):
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    return str(o)


def write_text(path, text: str) -> str:
    """Write a page, creating missing folders; a clear error when the place isn't writable."""
    from ...errors import PathError

    path = os.path.abspath(os.path.expanduser(str(path)))
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    except OSError as exc:
        raise PathError(f"can't write {path}: {exc.strerror or exc}.",
                        "choose a folder you can write to, e.g. path='~/deepvis/page.html'.") from exc
    return path


def in_notebook() -> bool:
    try:
        from IPython import get_ipython

        ip = get_ipython()
        return ip is not None and ip.__class__.__name__ in ("ZMQInteractiveShell", "Shell")
    except ImportError:  # pragma: no cover
        return False


class Dashboard:
    """``net.show(x)`` returns one of these. In Jupyter it renders inline; elsewhere it opens a browser."""

    def __init__(self, net: "Net", args: tuple | None = None, focus: str | None = None, trace=None, monitor=None,
                 highlight=None, height: int = 820, builder: PayloadBuilder | None = None, style: str = "expressive",
                 tensor_shapes: bool = True, views=None, start: str | None = None, title: str | None = None,
                 note: str | None = None, depth=None, training=None, sample: int = 0, grads: bool = False,
                 loss_fn=None, tokenizer=None, labels=None, tokens=None, diffusion=None, attribution: bool = False,
                 heads: bool | None = None, compare=None, compare_names=None, lens_fn=None, dataset=None,
                 offline: bool | None = None):
        self.net = net
        self.offline = offline
        self.vocab = {"tokenizer": tokenizer, "labels": labels, "tokens": tokens, "diffusion": diffusion,
                      "attribution": attribution, "heads": heads, "compare": compare, "compare_names": compare_names,
                      "lens_fn": lens_fn, "dataset": dataset}
        self.grads, self.loss_fn = grads, loss_fn
        self.training = training
        self.sample = sample
        self.args = args if args is not None else (trace.args if trace is not None else net.sample_input)
        self.focus = focus
        self.trace = trace
        self.monitor = monitor if monitor is not None else (net._monitors[-1] if net._monitors else None)
        self.highlight = highlight
        self.height = height
        self.builder = builder or PayloadBuilder(net, style=style, tensor_shapes=tensor_shapes)
        known = ("graph", "volumes", "neurons", "equations", "monitor", "attention", "diffusion", "compare", "dataset")
        if views is not None:
            views = [views] if isinstance(views, str) else list(views)
            bad = [v for v in views if v not in known]
            if bad:
                from ...errors import OptionError, suggest

                hint = suggest(bad[0], known)
                raise OptionError(f"unknown view {bad[0]!r}.", (hint + " " if hint else "") + f"Views: {', '.join(known)}.")
        if start is not None and start not in known:
            from ...errors import OptionError, suggest

            hint = suggest(start, known)
            raise OptionError(f"unknown start view {start!r}.", (hint + " " if hint else "") + f"Views: {', '.join(known)}.")
        self.ui = {"views": views, "start": start, "title": title, "note": note, "depth": depth}
        self.path: str | None = None
        self._html: str | None = None

    def payload(self) -> dict:
        return self.builder.build(self.args if self.trace is None else None, trace=self.trace, monitor=self.monitor,
                                  focus=self.focus, highlight=self.highlight, ui=self.ui, training=self.training,
                                  sample=self.sample, grads=self.grads, loss_fn=self.loss_fn, **self.vocab)

    def html(self) -> str:
        if self._html is None:
            self._html = render_page(self.payload(), None, self.ui.get("title") or self.net.name, offline=self.offline)
        return self._html

    def save(self, path: str) -> str:
        self.path = write_text(path, self.html())
        return self.path

    def display(self, open_browser: bool | None = None):
        if open_browser is None:
            open_browser = not in_notebook()
        if open_browser:
            fd, path = tempfile.mkstemp(prefix=f"deepvis-{self.net.name}-", suffix=".html")
            os.close(fd)
            self.save(path)
            webbrowser.open("file://" + self.path)
        return self

    def _repr_html_(self) -> str:
        doc = _html.escape(self.html(), quote=True)
        return (f'<iframe srcdoc="{doc}" style="width:100%;height:{self.height}px;border:1px solid #444444;'
                f'border-radius:10px;background:#000000" sandbox="allow-scripts allow-downloads"></iframe>')

    def __repr__(self) -> str:
        where = f" saved to {self.path}" if self.path else ""
        return f"Dashboard({self.net.name}{where})"


def equation_html(eq, offline: bool | None = None) -> str:
    """Minimal standalone page rendering an :class:`~deepvis.equations.unroll.Equation` with KaTeX."""
    tex = json.dumps(eq.full_latex())
    return f"""<div class="dv-eq" style="background:#000000;color:#DDDDDD;padding:16px 20px;border-radius:10px;overflow-x:auto">
{katex_tags(offline)}
<div id="dv-eq-target" style="font-size:16px"><pre>{_html.escape(eq.text)}</pre></div>
<script>try{{katex.render({tex}, document.getElementById("dv-eq-target"), {{displayMode:true, throwOnError:false}});}}catch(e){{}}</script>
</div>"""
