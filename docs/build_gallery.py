"""Build the deepvis site: one interactive page per showcase model, a thumbnail of each, an index, the API
reference, the README / tour / model zoo as pages (docs/*.html) and the pictures they show (media/).

    python docs/build_gallery.py [OUT_DIR]          # default docs/site

Needs torch; transformers, torch_geometric, pdoc, markdown and playwright (for thumbnails) are used when installed.
Every model here is small and trains in seconds on a CPU, so the whole gallery builds in about a minute. pages.yml
publishes it to GitHub Pages, adding the videos from the repository's ``media`` release.
"""

from __future__ import annotations

import copy
import html
import os
import re
import shutil
import sys
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "examples"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import deepvis as dv  # noqa: E402
from _data import strokes  # noqa: E402

warnings.filterwarnings("ignore")
SHOWS: list[dict] = []


def show(name, title, blurb):
    def deco(fn):
        SHOWS.append({"name": name, "title": title, "blurb": blurb, "make": fn})
        return fn
    return deco


def strokes_data(n=900, side=20):
    X, y = strokes(n, side=side, seed=1)
    return torch.tensor(X).unsqueeze(1), torch.tensor(y).long()


def fit(model, X, y, steps=200, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr)
    g = torch.Generator().manual_seed(0)
    for _ in range(steps):
        i = torch.randint(0, len(X), (64,), generator=g)
        loss = F.cross_entropy(model(X[i]), y[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


CLASSES = ["vertical bar", "horizontal bar", "ring"]


def cnn():
    return nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(8, 16, 3, padding=1),
                         nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 3))


# ------------------------------------------------------------------------------------------ the showcase
@show("mlp", "A multilayer perceptron", "Every layer drawn from its own numbers: weight fans, transfer curves, outputs.")
def _mlp(path):
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(8, 32), nn.GELU(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 4), nn.Softmax(-1))
    return dv.visualize(m, torch.randn(16, 8), path=path, title="MLP")


@show("why", "Why this prediction?", "Integrated gradients, saliency and Grad-CAM for the top predictions.")
def _why(path):
    torch.manual_seed(0)
    X, y = strokes_data()
    m = fit(cnn(), X, y)
    return dv.visualize(m, X[:4], path=path, attribution=True, labels=CLASSES, title="Why — a small CNN")


@show("attention", "Inside self-attention", "Q, Kᵀ, scores, softmax, P·V — which heads matter, where information flows.")
def _attention(path):
    torch.manual_seed(0)

    class TinyLM(nn.Module):
        def __init__(self, V=40, d=32):
            super().__init__()
            self.emb = nn.Embedding(V, d)
            layer = nn.TransformerEncoderLayer(d, 4, 64, batch_first=True, dropout=0.0)
            self.enc = nn.TransformerEncoder(layer, 3, enable_nested_tensor=False)
            self.head = nn.Linear(d, V)

        def forward(self, t):
            return self.head(self.enc(self.emb(t)))

    words = "the cat sat on a mat and it purred".split()
    return dv.visualize(TinyLM().eval(), torch.randint(0, 40, (1, 9)), path=path, tokens=words, attribution=True,
                        start="attention", title="A tiny transformer")


@show("lstm", "An LSTM, unrolled", "Every gate at every time step, recomputed from the layer's weights.")
def _lstm(path):
    torch.manual_seed(0)

    class Seq(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm, self.head = nn.LSTM(2, 16, batch_first=True), nn.Linear(16, 3)

        def forward(self, x):
            return self.head(self.lstm(x)[0][:, -1])

    t = torch.linspace(0, 6.28, 48)
    x = torch.stack([torch.sin(t), torch.cos(2 * t)], -1)[None]
    return dv.visualize(Seq().eval(), x, path=path, focus="lstm", title="LSTM over a signal")


@show("moe", "Mixture of experts", "Which experts the router sends each token to, and how the load spreads.")
def _moe(path):
    torch.manual_seed(0)

    class MoE(nn.Module):
        def __init__(self, d=24, E=8, k=2):
            super().__init__()
            self.gate, self.experts, self.top_k = nn.Linear(d, E, bias=False), nn.ModuleList(
                nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)) for _ in range(E)), k

        def forward(self, x):
            w, idx = self.gate(x).softmax(-1).topk(self.top_k, -1)
            out = torch.zeros_like(x)
            for j in range(self.top_k):
                for e, ex in enumerate(self.experts):
                    out = out + (idx[..., j] == e).unsqueeze(-1) * w[..., j:j + 1] * ex(x)
            return out

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb, self.moe, self.head = nn.Embedding(50, 24), MoE(), nn.Linear(24, 50)

        def forward(self, t):
            return self.head(self.moe(self.emb(t)))

    return dv.visualize(Net().eval(), torch.randint(0, 50, (1, 16)), path=path, focus="moe",
                        title="Routing in a mixture of experts")


@show("gnn", "A graph neural network", "Message passing on the input graph: features spread along the edges.")
def _gnn(path):
    from torch_geometric.nn import GCNConv

    torch.manual_seed(0)
    A, B = list(range(15)), list(range(15, 30))
    g = torch.Generator().manual_seed(1)
    edges = [(i, j) for grp in (A, B) for i in grp for j in grp
             if i < j and torch.rand((), generator=g) < 0.25] + [(0, 15)]
    ei = torch.tensor(edges).T
    ei = torch.cat([ei, ei.flip(0)], 1)
    x = torch.randn(30, 4) * 0.1
    x[A[:3], 0] = 3.0

    class GCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1, self.c2, self.c3 = GCNConv(4, 8), GCNConv(8, 8), GCNConv(8, 2)

        def forward(self, x, ei):
            return self.c3(torch.relu(self.c2(torch.relu(self.c1(x, ei)), ei)), ei)

    return dv.visualize(GCN().eval(), (x, ei), path=path, focus="c2", title="A GCN on two communities")


@show("compare", "Before and after fine-tuning", "How far each layer's weights and activations moved.")
def _compare(path):
    torch.manual_seed(0)
    X, y = strokes_data()
    before = cnn().eval()
    after = fit(copy.deepcopy(before), X, y, steps=150)
    return dv.compare(before, after, X[:4], names=("untrained", "trained"), path=path, labels=CLASSES,
                      title="Before / after training")


@show("training", "Watch it learn", "Loss curves, per-layer learning dynamics, and a replay of the weights.")
def _training(path):
    torch.manual_seed(0)
    X = torch.randn(256, 2)
    y = ((X[:, 0] * X[:, 1]) > 0).long()
    m = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 2))
    net = dv.wrap(m, sample_input=X[:8])
    log = dv.TrainingLog(net, every_steps=10)
    opt = torch.optim.Adam(m.parameters(), 0.03)
    for s in range(400):
        loss = F.cross_entropy(m(X), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        log.record(s, loss=loss, trace=net.trace(X[:8]) if s % 10 == 0 else None)
    return net.export_html(path, X[:8], training=log, start="monitor", title="Learning XOR")


@show("diffusion", "A diffusion model", "The reverse process: noisy xₜ, the predicted x̂₀ and ε̂, step by step.")
def _diffusion(path):
    from _diffusion import TinyUNet
    from deepvis import diffusion as dd

    weights = os.path.join(ROOT, "examples", "tinyunet_strokes.pt")
    model = TinyUNet(ch=32, tdim=64)
    model.load_state_dict(torch.load(weights))
    run = dd.sample(model.eval(), shape=(6, 1, 20, 20), betas=dd.linear_betas(200, 1e-4, 0.02), sampler="ddpm",
                    frames=40, seed=3)
    x_t, t = run.probe(k=6)
    return dv.visualize(model, (x_t, t), path=path, diffusion=run, start="diffusion", heads=False,
                        title="A tiny DDPM")


@show("vit", "A vision transformer", "Image patches as tokens; attention and rollout drawn on the picture.")
def _vit(path):
    import transformers as T

    torch.manual_seed(0)
    X, _ = strokes(2, side=32, seed=2)
    x = torch.tensor(X[:1]).unsqueeze(1).repeat(1, 3, 1, 1)
    vit = T.ViTForImageClassification(T.ViTConfig(image_size=32, patch_size=8, num_channels=3, hidden_size=32,
                                                  num_hidden_layers=2, num_attention_heads=4,
                                                  intermediate_size=64, num_labels=3)).eval()
    return dv.visualize(vit, x, path=path, start="attention", title="A tiny ViT")


# ------------------------------------------------------------------------------------------ building
def thumbnails(items, site):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  (playwright not installed: no thumbnails)")
        return
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1280, "height": 800}, device_scale_factor=1)
        for it in items:
            pg.goto("file://" + os.path.join(site, "pages", it["name"] + ".html"))
            pg.wait_for_timeout(1400)
            pg.screenshot(path=os.path.join(site, "thumbs", it["name"] + ".png"))
        b.close()


MARKETPLACE = "https://marketplace.visualstudio.com/items?itemName=batuoz.deepvis"
DOCS = {"README.md": "readme", "docs/tour.md": "tour", "docs/zoo.md": "zoo"}  # source → site/docs/<name>.html

DOC_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{title} · deepvis</title>
<style>
body {{ margin: 0; background: #000; color: #DDD; font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Inter,
  sans-serif; }}
article {{ max-width: 920px; margin: 0 auto; padding: 32px 18px 80px; }}
nav {{ max-width: 920px; margin: 0 auto; padding: 18px 18px 0; color: #888; font-size: 14px; }}
a {{ color: #58C4DD; }}
h1, h2, h3 {{ font-family: "CMU Serif", Georgia, serif; font-weight: 400; }}
h1 {{ font-size: 40px; }} h2 {{ font-size: 28px; border-bottom: 1px solid #2a2a2a; padding-bottom: 4px; }}
img {{ max-width: 100%; border-radius: 8px; }}
pre {{ background: #111; border: 1px solid #2a2a2a; border-radius: 8px; padding: 12px 14px; overflow-x: auto;
  font-size: 13.5px; line-height: 1.45; }}
code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.92em; }}
:not(pre) > code {{ background: #161616; border-radius: 4px; padding: 1px 4px; }}
table {{ border-collapse: collapse; display: block; overflow-x: auto; font-size: 14px; }}
th, td {{ border: 1px solid #2a2a2a; padding: 5px 9px; vertical-align: top; }}
th {{ background: #111; }}
</style></head><body>
<nav><a href="../index.html">gallery</a> · <a href="readme.html">overview</a> · <a href="tour.html">tour</a> ·
<a href="zoo.html">model zoo</a> · <a href="../api/deepvis.html">API</a> · <a href="{marketplace}">VS Code</a></nav>
<article>{body}</article>
</body></html>"""


def _site_links(md: str, source: str) -> str:
    """Repository-relative links → the site's: pictures to media/, markdown files to their pages."""
    md = re.sub(r"\]\((?:\./)?(?:docs/)?(?:images|video/out)/", "](../media/", md)
    md = re.sub(r"\]\((?:\./)?vscode/README\.md\)", f"]({MARKETPLACE})", md)
    in_docs = source.startswith("docs/")  # README links docs/x.md, docs/*.md link x.md and ../README.md
    for src, name in DOCS.items():
        local = ("../" + src if not src.startswith("docs/") else src.removeprefix("docs/")) if in_docs else src
        md = md.replace(f"]({local})", f"]({name}.html)")
    return md


def doc_pages(site: str) -> list[str]:
    """README, the tour and the zoo table as pages of the site; the pictures they use copied to media/."""
    media = os.path.join(site, "media")
    os.makedirs(media, exist_ok=True)
    for name in os.listdir(os.path.join(ROOT, "docs", "images")):
        shutil.copy2(os.path.join(ROOT, "docs", "images", name), media)
    try:
        import markdown
    except ImportError:
        print("  (markdown not installed: no docs pages)")
        return []
    os.makedirs(os.path.join(site, "docs"), exist_ok=True)
    made = []
    for src, name in DOCS.items():
        md = _site_links(open(os.path.join(ROOT, src), encoding="utf-8").read(), src)
        body = markdown.markdown(md, extensions=["tables", "fenced_code", "toc"])
        m = re.search(r"^# (.+)$", md, re.M)
        with open(os.path.join(site, "docs", name + ".html"), "w", encoding="utf-8") as f:
            f.write(DOC_PAGE.format(title=html.escape(m.group(1) if m else name), body=body, marketplace=MARKETPLACE))
        made.append(name)
    return made


INDEX = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>deepvis gallery</title>
<style>
:root {{ --bg: #000; --panel: #111; --edge: #2a2a2a; --text: #DDD; --muted: #888; --accent: #58C4DD; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--text);
  font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif; }}
header {{ max-width: 1180px; margin: 0 auto; padding: 48px 16px 12px; }}
h1 {{ font: 400 40px/1.1 "CMU Serif", Georgia, serif; margin: 0 0 8px; }}
header p {{ color: var(--muted); margin: 0; max-width: 720px; }}
header a {{ color: var(--accent); }}
main {{ max-width: 1180px; margin: 0 auto; padding: 24px 16px 64px; display: grid; gap: 20px;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); }}
a.card {{ display: block; background: var(--panel); border: 1px solid var(--edge); border-radius: 12px;
  overflow: hidden; color: inherit; text-decoration: none; transition: border-color .15s, transform .15s; }}
a.card:hover, a.card:focus-visible {{ border-color: var(--accent); transform: translateY(-2px); outline: none; }}
a.card img {{ display: block; width: 100%; aspect-ratio: 16 / 10; object-fit: cover; background: #000;
  border-bottom: 1px solid var(--edge); }}
a.card div {{ padding: 12px 14px 14px; }}
a.card b {{ display: block; font-weight: 600; margin-bottom: 2px; }}
a.card span {{ color: var(--muted); font-size: 13.5px; }}
footer {{ text-align: center; color: var(--muted); font-size: 13px; padding: 0 16px 40px; }}
</style></head><body>
<header><h1>deepvis</h1>
<p>Each card is a real, interactive page made by one <code>dv.visualize(model, inputs)</code> call — click to
explore it. {api} {docs}</p></header>
<main>{cards}</main>
<footer>built {when} · {n} pages</footer>
</body></html>"""


def main(out: str = os.path.join(ROOT, "docs", "site")) -> None:
    site = os.path.abspath(out)
    os.makedirs(os.path.join(site, "pages"), exist_ok=True)
    os.makedirs(os.path.join(site, "thumbs"), exist_ok=True)
    made = []
    for it in SHOWS:
        t0 = time.time()
        path = os.path.join(site, "pages", it["name"] + ".html")
        try:
            it["make"](path)
        except ImportError as exc:
            print(f"  skip {it['name']}: {exc.name} not installed")
            continue
        made.append(it)
        print(f"  {it['name']:<10} {time.time() - t0:5.1f}s")
    thumbnails(made, site)
    api = ""
    try:
        import pdoc  # noqa: F401
        import subprocess

        subprocess.run([sys.executable, "-m", "pdoc", "deepvis", "-o", os.path.join(site, "api"), "--docformat",
                        "restructuredtext"], check=True, capture_output=True)
        api = 'The <a href="api/deepvis.html">API reference</a> documents every function.'
    except Exception as exc:  # noqa: BLE001 - the gallery is still useful without it
        print(f"  (no API reference: {type(exc).__name__})")
    docs = ""
    if doc_pages(site):
        docs = ('Read the <a href="docs/readme.html">overview</a>, <a href="docs/tour.html">the tour</a> and the '
                f'<a href="docs/zoo.html">model zoo</a>, or get the <a href="{MARKETPLACE}">VS Code extension</a>.')
    cards = "\n".join(
        f'<a class="card" href="pages/{it["name"]}.html"><img src="thumbs/{it["name"]}.png" alt="" loading="lazy">'
        f'<div><b>{html.escape(it["title"])}</b><span>{html.escape(it["blurb"])}</span></div></a>' for it in made)
    with open(os.path.join(site, "index.html"), "w", encoding="utf-8") as f:
        f.write(INDEX.format(cards=cards, api=api, docs=docs, when=time.strftime("%Y-%m-%d"), n=len(made)))
    print(f"gallery: {os.path.join(site, 'index.html')}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
