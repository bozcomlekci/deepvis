"""Inside attention, diffusion sampling, the new component symbols, output display rules."""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import deepvis as dv
from deepvis import diffusion as dd
from deepvis.viz.glyphs import GlyphData, glyph_type, render
from deepvis.viz.html.payload import PayloadBuilder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------ inside attention
class Masked(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(16, 4, batch_first=True)

    def forward(self, x):
        T = x.shape[1]
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)  # MHA convention: True = blocked
        pad = torch.zeros(x.shape[0], T, dtype=torch.bool)
        pad[:, -1] = True
        return self.attn(x, x, x, attn_mask=causal, key_padding_mask=pad)[0]


def test_mha_parts_rebuild_the_output():
    torch.manual_seed(0)
    m, x = Masked().eval(), torch.randn(2, 6, 16)
    net = dv.wrap(m, sample_input=x)
    tr = net.trace(x)
    P = tr.attention_parts["attn"]
    assert P["q"].shape == (2, 4, 6, 4) and P["probs"].shape == (2, 4, 6, 6)
    heads = torch.as_tensor(P["out"]).transpose(1, 2).reshape(2, 6, 16)
    assert np.allclose(m.attn.out_proj(heads).detach().numpy(), tr["attn"], atol=1e-5)
    assert np.allclose(P["probs"], tr.attention["attn"], atol=1e-5)  # = torch's own attention weights
    assert np.isnan(P["scores"][0, 0, 0, 1]) and np.isnan(P["scores"][0, 0, 4, 5])  # causal, padding
    assert np.allclose(P["probs"].sum(-1), 1, atol=1e-5)


class SDPA(nn.Module):
    def __init__(self, d=16, h=2):
        super().__init__()
        self.h, self.qkv = h, nn.Linear(d, 3 * d)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        sp = lambda t: t.view(B, T, self.h, D // self.h).transpose(1, 2)  # noqa: E731
        return F.scaled_dot_product_attention(sp(q), sp(k), sp(v), is_causal=True)


def test_sdpa_parts_and_anatomy_payload():
    torch.manual_seed(0)
    m, x = SDPA().eval(), torch.randn(1, 5, 16)
    net = dv.wrap(m, sample_input=x)
    tr = net.trace(x)
    nid = next(iter(tr.attention_parts))
    P = tr.attention_parts[nid]
    assert np.allclose(P["out"], tr[nid], atol=1e-5)
    an = PayloadBuilder(net).build((x,))["anatomy"]
    assert len(an) == 1 and an[0]["heads"] == 2 and an[0]["T"] == 5 and an[0]["masked"] is not None
    assert {"q", "k", "v", "scores", "probs", "out"} <= set(an[0]) and len(an[0]["probs"]) == 2


# ------------------------------------------------------------------ diffusion
class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.t = nn.Embedding(50, 8)
        self.c1, self.c2 = nn.Conv2d(1, 8, 3, padding=1), nn.Conv2d(8, 1, 3, padding=1)

    def forward(self, x, t):
        return self.c2(F.silu(self.c1(x) + self.t(t)[:, :, None, None]))


def test_diffusion_sampling_records_the_process():
    torch.manual_seed(0)
    m = Denoiser()
    betas = dd.linear_betas(50)
    run = dd.sample(m, (3, 1, 8, 8), betas, sampler="ddpm", frames=10, clip_x0=None)
    assert run.x.shape == (10, 3, 1, 8, 8) and run.ts[0] == 49 and run.ts[-1] == 0
    ab = np.cumprod(1 - betas)
    i, t = 3, run.ts[3]  # x̂0 = (x_t − √(1−ᾱ) ε̂) / √ᾱ
    assert np.allclose(run.x0[i], (run.x[i] - np.sqrt(1 - ab[t]) * run.eps[i]) / np.sqrt(ab[t]), atol=1e-5)
    a = dd.sample(m, (2, 1, 8, 8), betas, sampler="ddim", steps=10, seed=1)
    b = dd.sample(m, (2, 1, 8, 8), betas, sampler="ddim", steps=10, seed=1)
    assert np.array_equal(a.final, b.final) and len(a.ts) == 10  # DDIM (η = 0) is deterministic
    x, tt = run.probe(k=4)
    assert x.shape == (4, 1, 8, 8) and tt.tolist() == sorted(tt.tolist(), reverse=True)
    d = run.to_dict()
    assert len(d["frames"]) == 10 and len(d["frames"][0]["x"]) == 3 and d["forward"] and d["T"] == 50
    net = dv.wrap(m, sample_input=(x, tt))
    pay = PayloadBuilder(net).build((x, tt), diffusion=run)
    assert pay["diffusion"]["sampler"] == "ddpm" and pay["output"]["image"].startswith("data:image/png")


def test_q_sample_matches_the_closed_form():
    betas = dd.linear_betas(100)
    x0 = torch.ones(1000, 1)
    xt, eps = dd.q_sample(x0, torch.full((1000,), 60), betas)
    ab = float(np.cumprod(1 - betas)[60])
    assert torch.allclose(xt, np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps, atol=1e-6)


# ------------------------------------------------------------------ symbols & outputs
def test_every_component_gets_its_own_symbol():
    class Zoo(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln, self.rms, self.fc = nn.LayerNorm(8), nn.RMSNorm(8), nn.Linear(8, 24)
            self.conv, self.bn, self.gn = nn.Conv2d(2, 4, 3, padding=1), nn.BatchNorm2d(4), nn.GroupNorm(2, 4)

        def forward(self, x, img):
            q, k, v = self.fc(self.rms(self.ln(x))).chunk(3, -1)
            s = torch.einsum("btd,bsd->bts", q, k)
            s = s.masked_fill(torch.ones(4, 4, dtype=torch.bool).triu(1), float("-inf"))
            o = torch.softmax(s, -1) @ v
            y = torch.sin(o) * torch.exp(-o.pow(2))
            g = self.gn(self.bn(self.conv(img)))
            return y.sum() + g.mean(0).sum() + g.mean((2, 3)).expand(3, -1, -1).sum()

    net = dv.wrap(Zoo().eval(), sample_input=(torch.randn(2, 4, 8), torch.randn(2, 2, 6, 6)))
    kinds = {n.kind: glyph_type(n) for n in net.graph.compute_nodes()}
    assert kinds["split"] == "split" and kinds["masked_fill"] == "mask" and kinds["matmul"] == "matmul"
    assert kinds["expand"] == "expand" and kinds["sin"] == kinds["exp"] == kinds["pow"] == "activation"
    assert "other" not in kinds.values()
    D = GlyphData(net, net.trace(torch.randn(2, 4, 8), torch.randn(2, 2, 6, 6)), net.graph)
    norms = {n.kind: render(n, D) for n in net.graph.compute_nodes() if glyph_type(n) == "norm"}
    assert len(norms) == 4 and len(set(norms.values())) == 4  # LayerNorm, RMSNorm, BatchNorm, GroupNorm differ


def test_output_display_follows_what_the_output_is():
    enc = nn.TransformerEncoderLayer(16, 2, 32, dropout=0.0, batch_first=True).eval()
    x = torch.randn(1, 6, 16)
    out = PayloadBuilder(dv.wrap(enc, sample_input=x)).build((x,))["output"]
    assert "heat" in out and "topk" not in out  # hidden states: a heat map, not "classes"
    clf = nn.Linear(4, 1000)
    out = PayloadBuilder(dv.wrap(clf, sample_input=torch.randn(1, 4))).build((torch.randn(1, 4),))["output"]
    assert len(out["topk"]) == 10  # a 1000-way classifier: its top k


def test_visualize_is_the_one_call(tmp_path, monkeypatch):
    model, x = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)), torch.randn(3, 4)
    net = dv.visualize(model, x, path=str(tmp_path / "a.html"), title="Mine", start="monitor")
    assert isinstance(net, dv.Net) and net.page == str(tmp_path / "a.html")
    page = open(net.page, encoding="utf-8").read()
    assert '"title":"Mine"' in page and '"start":"monitor"' in page
    monkeypatch.setenv("DEEPVIS_CAPTURE", str(tmp_path / "b.html"))  # how the VS Code extension captures the page
    net = dv.visualize(model, (x,))
    assert net.page == str(tmp_path / "b.html") and os.path.getsize(net.page) > 50_000
    assert net.equation().text  # and the wrapper keeps working
