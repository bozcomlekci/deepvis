"""``deepvis`` command line: export a model's interactive page or figure without writing any code.

    deepvis export torchvision:resnet18 -o resnet18.html
    deepvis export torchvision:vit_b_16 -o vit.svg --direction LR
    deepvis export hf:gpt2 --random -o gpt2.html              # architecture from the config, random weights
    deepvis export hf:gpt2 --text "The capital of France is" -o gpt2.html   # tokens + next-token predictions
    deepvis export torchvision:resnet50 --pretrained --image cat.jpg -o r50.html   # ImageNet class names
    deepvis export my_models.py:build -o net.html              # build() returns a module, (module, input) or a dv.Net
    deepvis export mypkg.models:MLP --shape 8,784 -o mlp.html  # importable class / factory
    deepvis summary torchvision:mobilenet_v3_small
    deepvis export hf:meta-llama/Llama-3.1-70B --meta -o 70b.html   # config only, on the meta device: seconds
    deepvis open model.onnx                                    # any saved model file, straight into the browser
    deepvis export llama.safetensors -o weights.html           # a weights-only view (no code needed)

A model spec is one of
  torchvision:<name>          torchvision.models.<name>() (random init; add --pretrained for default weights)
  timm:<name>                 timm.create_model(<name>)
  hf:<name-or-path>           transformers AutoModel (``--random``: from the config only, no weight download)
  <file.py>:<attr>            an attribute of a Python file: a module, or a callable returning one of
                              module | (module, sample_input) | deepvis.Net  (use a Net for JAX models)
  <package.module>:<attr>     the same, imported by name
  <file>                      a saved model: .onnx .keras .h5 .pt2 (torch.export) — or weights only: a
                              state_dict .pt/.pth/.bin/.ckpt, .safetensors, .npz. A .pt holding a whole pickled
                              module runs code from the file when loaded: pass --trust, only for files you trust

Output format follows the extension: .html (interactive dashboard) or .svg (static figure; .svg.html wraps it).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import ntpath
import os
import sys


def _shape(s: str | None):
    if not s:
        return None
    return tuple(int(v) for v in s.replace("x", ",").split(",") if v.strip())


def _load_attr(spec: str):
    target, _, attr = spec.rpartition(":")
    if target.endswith(".py"):
        path = os.path.abspath(target)
        if not os.path.exists(path):
            raise SystemExit(f"no such file: {target}")
        sys.path.insert(0, os.path.dirname(path))
        mod_spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(path))[0], path)
        mod = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(target)
    obj = getattr(mod, attr)
    return obj() if callable(obj) and not _is_model(obj) else obj


def _is_model(obj) -> bool:
    """A model instance (torch, flax, nnx, equinox) or a dv.Net: used as is, never called to build one."""
    from .net import _detect

    return _detect(obj, None) in ("torch", "flax", "nnx", "equinox") or type(obj).__name__ == "Net"


_META: dict = {}  # tokenizer / class names found while loading (used for labels in the page)


def _hf_auto(tf, config):
    """The head that matches the checkpoint: a causal LM keeps its vocabulary logits, a classifier its classes."""
    arch = " ".join(getattr(config, "architectures", None) or [])
    if "ForCausalLM" in arch or "LMHeadModel" in arch:
        return tf.AutoModelForCausalLM, "tokens"
    if "ForImageClassification" in arch:
        return tf.AutoModelForImageClassification, "image"
    if "ForSequenceClassification" in arch:
        return tf.AutoModelForSequenceClassification, "tokens"
    return tf.AutoModel, "image" if getattr(config, "num_channels", None) and not getattr(config, "vocab_size",
                                                                                           None) else "tokens"


def load_model(spec: str, pretrained: bool = False, random: bool = False, trust: bool = False, meta: bool = False,
               _meta_device: bool = False):
    """Returns ``(model_or_net, sample_input_or_None, kind)``. ``meta``: built on the meta device — the architecture
    and shapes of any size of model, without weights or memory."""
    if meta and not any(spec.startswith(p) for p in ("hf:", "torchvision:", "timm:")):
        raise SystemExit("--meta builds hf:, torchvision: and timm: models from their configs")
    if meta:
        import torch

        with torch.device("meta"):
            return load_model(spec, pretrained=False, random=True, trust=trust, _meta_device=True)
    _META.clear()
    from . import files

    if files.is_model_file(spec) and (os.path.exists(spec) or ":" not in ntpath.splitdrive(spec)[1]):
        return files.open_file(spec, trusted=trust), None, "given"  # C:\models\net.pt is a file, not module:attr
    if spec.startswith("torchvision:"):
        import torchvision

        name = spec.split(":", 1)[1]
        if name not in torchvision.models.list_models():
            from .errors import ModelError, suggest

            hint = suggest(name, torchvision.models.list_models())
            raise ModelError(f"torchvision has no model called {name!r}.",
                             (hint + " " if hint else "") + "See them all with: python -c \"import torchvision; "
                             "print(torchvision.models.list_models())\"")
        fn = getattr(torchvision.models, name)
        if pretrained:
            weights = torchvision.models.get_model_weights(name).DEFAULT
            _META["labels"] = list(weights.meta.get("categories") or []) or None
            _META["preprocess"] = lambda img: weights.transforms()(img).unsqueeze(0)
            m = fn(weights=weights)
        else:
            m = fn()
        return m.eval(), None, "image"
    if spec.startswith("timm:"):
        import timm

        name = spec.split(":", 1)[1]
        if not timm.list_models(name):
            from .errors import ModelError, suggest

            hint = suggest(name, timm.list_models())
            raise ModelError(f"timm has no model called {name!r}.",
                             (hint + " " if hint else "") + f"Search with: python -c \"import timm; "
                             f"print(timm.list_models('*{name.split('_')[0]}*'))\"")
        m = timm.create_model(name, pretrained=pretrained)
        return m.eval(), None, "image"
    if spec.startswith("hf:"):
        import transformers as tf

        name = spec.split(":", 1)[1]
        try:
            config = tf.AutoConfig.from_pretrained(name)
        except OSError as exc:
            from .errors import ModelError

            raise ModelError(f"couldn't load {name!r} from the HuggingFace Hub.",
                             "check the name on huggingface.co (e.g. 'gpt2', 'google/vit-base-patch16-224') and your "
                             "internet connection; private models need `huggingface-cli login`.") from exc
        auto, kind = _hf_auto(tf, config)
        if _meta_device:  # mixture-of-experts: the batched kernel runs on shapes alone (eager routing needs data)
            try:
                config._experts_implementation = "batched_mm"
            except Exception:  # noqa: BLE001 - older transformers: nothing to choose
                pass
        m = auto.from_config(config) if random else auto.from_pretrained(name)
        labels = getattr(config, "id2label", None) or {}
        if len(labels) > 2 and not all(str(v).startswith("LABEL_") for v in labels.values()):
            _META["labels"] = [labels[i] for i in sorted(labels)]
        if kind == "tokens":
            try:
                _META["tokenizer"] = tf.AutoTokenizer.from_pretrained(name)
            except Exception:  # noqa: BLE001 - labels are a nicety
                pass
        else:
            try:
                proc = tf.AutoImageProcessor.from_pretrained(name)
                _META["preprocess"] = lambda img: proc(images=img, return_tensors="pt")["pixel_values"]
            except Exception:  # noqa: BLE001
                pass
        return m.eval(), None, kind
    if ":" not in spec:
        raise SystemExit(f"can't interpret model spec {spec!r} (see `deepvis export --help`)")
    obj = _load_attr(spec)
    if isinstance(obj, tuple) and len(obj) == 2 and _is_model(obj[0]):
        return obj[0], obj[1], "given"
    return obj, None, "unknown"


def _sample(model, kind: str, shape, dtype: str | None, high: int | None):
    import torch

    if shape is None:
        if kind == "image":
            shape = (1, 3, 224, 224)
        elif kind == "tokens":
            shape = (1, 16)
        else:
            return None
    if dtype in ("long", "int", "int64") or (dtype is None and kind == "tokens"):
        vocab = high or getattr(getattr(model, "config", None), "vocab_size", None) or 1000
        return torch.randint(0, int(vocab), shape, generator=torch.Generator().manual_seed(0))
    return torch.randn(shape, generator=torch.Generator().manual_seed(0))


def _load_image(path: str, preprocess=None, shape=None):
    """An image file as the model's input: its own preprocessing when known, else resize + ImageNet normalization."""
    import numpy as np
    import torch
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if preprocess is not None:
        return preprocess(img)
    h, w = (shape[-2], shape[-1]) if shape and len(shape) >= 2 else (224, 224)
    a = np.asarray(img.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
    a = (a - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    return torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0)


def _wrap(args):
    import deepvis as dv

    model, x, kind = load_model(args.model, pretrained=args.pretrained, random=args.random,
                                trust=getattr(args, "trust", False), meta=getattr(args, "meta", False))
    if type(model).__name__ == "Net":
        return model, (model.sample_input or ())
    if x is None and getattr(args, "text", None):
        tok = _META.get("tokenizer")
        if tok is None:
            raise SystemExit("--text needs a tokenizer (an hf: model with one)")
        x = tok(args.text, return_tensors="pt")["input_ids"]
    if x is None and getattr(args, "image", None):
        x = _load_image(args.image, _META.get("preprocess"), _shape(args.shape))
    if x is None:
        x = _sample(model, kind, _shape(args.shape), args.dtype, args.high)
    if x is None:
        raise SystemExit("this model needs an input shape: pass --shape, e.g. --shape 1,3,32,32")
    if getattr(args, "meta", False):
        x = x.to("meta")  # shapes flow through a meta model; no numbers
    net = dv.wrap(model, sample_input=x)
    net.tokenizer, net.labels = _META.get("tokenizer"), _META.get("labels")
    return net, (x,)


def cmd_export(args) -> None:
    net, inputs = _wrap(args)
    outs = args.output or [f"{net.name}.html"]
    for out in outs:
        low = out.lower()
        if low.endswith((".svg", ".svg.html", ".png", ".pdf")):
            p = net.export_svg(out, *inputs, depth=_depth(args.depth), direction=args.direction, title=args.title)
        elif low.endswith((".html", ".htm")):
            views = tuple(v.strip() for v in args.views.split(",")) if args.views else None
            p = net.export_html(out, *inputs, views=views, start=args.start, style=args.style, title=args.title,
                                note=args.note, depth=_depth(args.depth), tensor_shapes=not args.no_tensors,
                                sample=args.sample, open_browser=args.open)
        else:
            raise SystemExit(f"unknown output format for {out!r}: use .html, .svg, .png, .pdf or .svg.html")
        print(p)


def cmd_open(args) -> None:
    """Write the page into a temporary folder and open it in the browser."""
    import tempfile

    net, inputs = _wrap(args)
    out = os.path.join(tempfile.mkdtemp(prefix="deepvis-"), f"{net.name}.html")
    print(net.export_html(out, *inputs, title=args.title, open_browser=True))


def cmd_summary(args) -> None:
    net, _ = _wrap(args)
    print(repr(net))
    net.summary()


def _depth(d):
    if d in (None, "auto"):
        return None  # automatic: the most detailed view that stays readable
    if d in ("all", "full"):
        return "all"
    return int(d)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="deepvis", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("model", help="model spec, e.g. torchvision:resnet18, hf:gpt2, file.py:build")
        p.add_argument("--shape", help="sample input shape, e.g. 1,3,224,224 (defaults: images 1,3,224,224; "
                                       "token models 1,16)")
        p.add_argument("--dtype", choices=["float", "long"], help="sample input dtype (token ids: long)")
        p.add_argument("--high", type=int, help="token ids are drawn from [0, high) (default: config.vocab_size)")
        p.add_argument("--pretrained", action="store_true", help="torchvision/timm: load default weights")
        p.add_argument("--random", action="store_true", help="hf: build from the config, random weights")
        p.add_argument("--text", help="hf language models: the prompt to run (tokenized as the input)")
        p.add_argument("--image", help="vision models: an image file as the input (the model's own preprocessing "
                                       "when known, else resized and ImageNet-normalized)")
        p.add_argument("--trust", action="store_true", help="files: allow a .pt holding a whole pickled module "
                                                             "(loading it runs code from the file)")
        p.add_argument("--meta", action="store_true", help="hf/torchvision/timm: build on the meta device from the "
                                                            "config — any size, in seconds: architecture and shapes, "
                                                            "no weights")

    ex = sub.add_parser("export", help="write an interactive .html page and/or a static .svg figure",
                        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common(ex)
    ex.add_argument("-o", "--output", action="append", help="output file(s): .html or .svg (repeatable)")
    ex.add_argument("--title")
    ex.add_argument("--note", help="a short description shown in the page's overview")
    ex.add_argument("--views", help="tabs to include: graph,volumes,neurons,equations,monitor")
    ex.add_argument("--start", help="tab the page opens on")
    ex.add_argument("--style", default="expressive", choices=["expressive", "compact"])
    ex.add_argument("--depth", help="detail level: an int, 'auto' or 'all'")
    ex.add_argument("--direction", default="TB", choices=["TB", "LR"], help="SVG figures: top-bottom or left-right")
    ex.add_argument("--no-tensors", action="store_true", help="hide the 3-D tensor shapes on edges")
    ex.add_argument("--sample", type=int, default=0, help="which sample of a batched input the page shows")
    ex.add_argument("--open", action="store_true", help="open the page in a browser")
    ex.set_defaults(fn=cmd_export)

    op = sub.add_parser("open", help="open a model (or a saved file) in the browser",
                        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common(op)
    op.add_argument("--title")
    op.set_defaults(fn=cmd_open)

    sm = sub.add_parser("summary", help="print the module tree and dataflow summary")
    common(sm)
    sm.set_defaults(fn=cmd_summary)

    args = ap.parse_args(argv)
    from .errors import DeepvisError

    try:
        args.fn(args)
    except DeepvisError as exc:  # a clear message, not a traceback
        print(f"deepvis: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":  # pragma: no cover
    main()
