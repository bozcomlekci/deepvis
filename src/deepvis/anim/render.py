"""Render deepvis scenes with ManimCE."""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import TYPE_CHECKING, Any

from .. import theme
from . import data as D

if TYPE_CHECKING:  # pragma: no cover
    from ..net import Net

QUALITY = {"low": "low_quality", "medium": "medium_quality", "high": "high_quality", "production": "production_quality",
           "4k": "fourk_quality"}


def render(scene_cls, data: dict, path: str | None = None, quality: str = "low", fmt: str = "mp4",
           name: str = "deepvis") -> str:
    """Render ``scene_cls(data)`` and return the path of the video (or GIF)."""
    try:
        import manim
    except ImportError as exc:  # pragma: no cover
        raise ImportError("animations need ManimCE: pip install 'deepvis[anim]'") from exc
    media = tempfile.mkdtemp(prefix="deepvis-manim-")
    cfg = {"quality": QUALITY.get(quality, quality), "media_dir": media, "disable_caching": True,
           "verbosity": "ERROR", "progress_bar": "none", "background_color": theme.BACKGROUND,
           "output_file": name, "format": "gif" if fmt == "gif" else "mp4", "write_to_movie": True}
    with manim.tempconfig(cfg):
        scene = scene_cls(data)
        scene.render()
        fw = scene.renderer.file_writer
        produced = str(getattr(fw, "gif_file_path", None) if fmt == "gif" else fw.movie_file_path)
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        shutil.copy(produced, path)
        shutil.rmtree(media, ignore_errors=True)
        return os.path.abspath(path)
    return produced


def build_data(net: "Net", kind: str, args: tuple | None, **kw) -> dict:
    from ..viz.explain import describe

    data = _build_data(net, kind, args, **kw)
    data["describe"] = describe(net.graph, batched=net.framework != "equinox", detail=False)  # the opening caption
    return data


def _build_data(net: "Net", kind: str, args: tuple | None, **kw) -> dict:
    if kind not in D.DATA_BUILDERS:
        raise ValueError(f"unknown animation {kind!r}; choose from {sorted(D.DATA_BUILDERS)}")
    if kind in ("forward", "focus", "edit", "volumes") and args is None:
        raise ValueError(f"the {kind!r} animation needs an input: net.animate({kind!r}, x)")
    if kind == "forward":
        return D.dense_data(net, args, max_neurons=kw.get("max_neurons", 10))
    if kind == "focus":
        return D.focus_data(net, args, node=kw.get("focus") or kw.get("node"), neuron=kw.get("neuron"),
                            max_neurons=kw.get("max_neurons", 10))
    if kind == "edit":
        if "edit" not in kw:
            raise ValueError("the 'edit' animation needs edit=... (an Edit, a dict, or fn(net))")
        return D.edit_data(net, args, kw["edit"], keep=kw.get("keep", False), max_neurons=kw.get("max_neurons", 10))
    if kind == "equation":
        return D.equation_data(net)
    if kind == "volumes":
        return D.volumes_data(net, args, max_groups=kw.get("max_blocks", 10))
    if kind == "training":
        return D.training_data(net, kw.get("log", kw.get("training")), max_layers=kw.get("max_layers", 18))
    return D.architecture_data(net, max_layers=kw.get("max_layers", 14))


def animate(net: "Net", kind: str, args: tuple | None, path: str | None = None, quality: str = "low",
            fmt: str = "mp4", **kw) -> Any:
    from .scenes import SCENES

    data = build_data(net, kind, args, **kw)
    data["captions"] = bool(kw.get("captions", True))  # narration in the caption zone (captions=False: none)
    out = render(SCENES[kind], data, path=path, quality=quality, fmt=fmt, name=f"{net.name}-{kind}")
    try:
        from ..viz.html.dashboard import in_notebook

        if in_notebook():
            from IPython.display import Image, Video

            return Image(out) if fmt == "gif" else Video(out, embed=True, html_attributes="controls loop autoplay")
    except ImportError:  # pragma: no cover
        pass
    return out
