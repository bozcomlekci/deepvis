"""Interpretability: attribution, attention rollout, head importance / switching heads off, comparing versions."""

import copy
import json
import re

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
F = torch.nn.functional
pytestmark = pytest.mark.torch

import deepvis as dv  # noqa: E402
from deepvis.errors import ModelError, OptionError  # noqa: E402
from deepvis.interpret import attribute, head_importance, rollout  # noqa: E402
from deepvis.interpret.attention import head_ablation  # noqa: E402


def payload_of(path):
    html = open(path, encoding="utf-8").read()
    return json.loads(re.search(r'<script id="dv-payload" type="application/json">(.*?)</script>', html, re.S).group(1)
                      .replace("<\\/", "</"))


def encoder(layers=2, heads=4, d=16):
    torch.manual_seed(0)
    layer = nn.TransformerEncoderLayer(d, heads, 2 * d, batch_first=True, dropout=0.0)
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False).eval()


class SDPA(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv, self.o = nn.Linear(16, 48), nn.Linear(16, 16)

    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).reshape(B, T, 3, 4, 4).permute(2, 0, 3, 1, 4)
        return self.o(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, T, 16))


# ------------------------------------------------------------------------------------------ attribution
def test_integrated_gradients_is_complete():
    torch.manual_seed(0)
    mlp = nn.Sequential(nn.Linear(6, 16), nn.Tanh(), nn.Linear(16, 3))
    x = torch.randn(4, 6)
    a = attribute(dv.wrap(mlp, sample_input=x), x, method="ig", sample=2, steps=32)
    assert a.kind == "features" and a.map.shape == (6,)
    assert a.map.sum() == pytest.approx(a.score - a.extra["baseline_score"], rel=0.02, abs=1e-4)
    assert a.target == int(mlp(x[2:3]).argmax())


def test_grad_cam_finds_the_bright_patch():
    conv = nn.Conv2d(1, 4, 3, padding=1)
    with torch.no_grad():
        conv.weight.fill_(0.1)
        conv.bias.zero_()
    head = nn.Linear(4, 2)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[1.0] * 4, [-1.0] * 4]))
        head.bias.zero_()
    model = nn.Sequential(conv, nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), head)
    img = torch.zeros(1, 1, 16, 16)
    img[..., 10:14, 2:6] = 1.0
    cam = attribute(dv.wrap(model, sample_input=img), img, method="grad_cam", target=0)
    assert cam.kind == "image" and cam.map.shape == (16, 16)
    r, c = np.unravel_index(cam.map.argmax(), cam.map.shape)
    assert 9 <= r <= 14 and 1 <= c <= 6


def test_token_attribution_through_the_embedding():
    class Tok(nn.Module):
        def __init__(self):
            super().__init__()
            self.e, self.l = nn.Embedding(20, 8), nn.Linear(8, 20)

        def forward(self, t):
            return self.l(self.e(t).mean(1))

    torch.manual_seed(0)
    t = torch.randint(0, 20, (2, 7))
    a = attribute(dv.wrap(Tok(), sample_input=t), t, method="ig")
    assert a.kind == "tokens" and a.map.shape == (7,)
    assert a.map.sum() == pytest.approx(a.score - a.extra["baseline_score"], rel=0.02, abs=1e-4)
    s = attribute(dv.wrap(Tok(), sample_input=t), t, method="saliency")
    assert s.map.shape == (7,)


def test_attribution_errors_are_explained():
    mlp = nn.Sequential(nn.Linear(3, 2))
    net = dv.wrap(mlp, sample_input=torch.randn(1, 3))
    with pytest.raises(OptionError, match="unknown attribution method"):
        attribute(net, torch.randn(1, 3), method="shap")
    with pytest.raises(OptionError, match="out of range"):
        attribute(net, torch.randn(1, 3), target=7)
    with pytest.raises(ModelError, match="Grad-CAM needs a convolutional layer"):
        attribute(net, torch.randn(1, 3), method="grad_cam")


# ------------------------------------------------------------------------------------------ attention
def test_rollout_rows_are_distributions():
    enc = encoder()
    x = torch.randn(2, 5, 16)
    net = dv.wrap(enc, sample_input=x)
    r = rollout(net, net.trace(x))
    assert r["matrix"].shape == (5, 5) and np.allclose(r["matrix"].sum(-1), 1) and len(r["layers"]) == 2


def test_head_importance_finds_a_dead_head_and_restores_weights():
    enc = encoder()
    m = enc.layers[1].self_attn
    with torch.no_grad():  # head 3 of layer 1 contributes nothing
        m.in_proj_weight[32 + 12:32 + 16] = 0
        m.in_proj_bias[32 + 12:32 + 16] = 0
    x = torch.randn(1, 5, 16)
    w0 = [p.clone() for p in enc.parameters()]
    hi = head_importance(dv.wrap(enc, sample_input=x), (x,))
    assert hi["layers"] == ["layers.0.self_attn", "layers.1.self_attn"] and hi["heads"] == 4
    assert hi["effect"][1][3] == pytest.approx(0.0, abs=1e-7)
    assert min(hi["effect"][0]) > 1e-4
    assert all(torch.equal(a, b) for a, b in zip(w0, enc.parameters()))


def test_function_attention_heads():
    x = torch.randn(1, 5, 16)
    torch.manual_seed(0)
    hi = head_importance(dv.wrap(SDPA(), sample_input=x), (x,))
    assert hi is not None and hi["heads"] == 4 and all(v > 0 for v in hi["effect"][0])


def test_switching_a_head_off_is_one_undoable_edit():
    enc = encoder()
    x = torch.randn(1, 5, 16)
    net = dv.wrap(enc, sample_input=x)
    w0 = enc.layers[0].self_attn.in_proj_weight.clone()
    kind, edits = head_ablation(net, "layers.0.self_attn", 2)
    assert kind == "edits" and len(edits) == 1
    net.apply_edit(edits[0])
    assert float(enc.layers[0].self_attn.in_proj_weight[32 + 8:32 + 12].abs().sum()) == 0
    net.undo()
    assert torch.equal(enc.layers[0].self_attn.in_proj_weight, w0)
    with pytest.raises(OptionError, match="head 9 doesn't exist"):
        head_ablation(net, "layers.0.self_attn", 9)


class CrossAttn(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        torch.manual_seed(0)
        self.attn = nn.MultiheadAttention(16, 4, batch_first=True, **kw)

    def forward(self, x, kv):
        return self.attn(x, kv, kv)[0]


@pytest.mark.parametrize("kw,kv_dim", [({}, 16), ({"kdim": 8, "vdim": 8}, 8), ({"add_bias_kv": True}, 16)],
                         ids=["packed", "kdim-vdim", "add-bias-kv"])
def test_a_switched_off_head_adds_nothing(kw, kv_dim):
    m = CrossAttn(**kw).eval()
    x, kv = torch.randn(2, 3, 16), torch.randn(2, 5, kv_dim)
    ref = copy.deepcopy(m)
    with torch.no_grad():
        ref.attn.out_proj.weight[:, 4:8] = 0  # head 1's contribution removed after the fact
        expected = ref(x, kv)
    net = dv.wrap(m, sample_input=(x, kv))
    w0 = {k: v.clone() for k, v in m.state_dict().items()}
    kind, edits = head_ablation(net, "attn", 1)
    assert kind == "edits" and len(edits) == 1
    net.apply_edit(edits[0])
    with torch.no_grad():
        assert torch.allclose(m(x, kv), expected, atol=1e-6)
    net.undo()
    assert all(torch.equal(m.state_dict()[k], v) for k, v in w0.items())
    from deepvis.interpret.attention import head_off

    with head_off(net, "attn", 1), torch.no_grad():
        assert torch.allclose(m(x, kv), expected, atol=1e-6)
    assert all(torch.equal(m.state_dict()[k], v) for k, v in w0.items())
    assert head_importance(net, (x, kv))["layers"] == ["attn"]


# ------------------------------------------------------------------------------------------ compare
def small_cnn():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(), nn.Flatten(), nn.Linear(4 * 8 * 8, 3))


def test_compare_identical_is_zero_and_a_checkpoint_loads(tmp_path):
    from deepvis.interpret.compare import compare, other_net

    a = small_cnn()
    x = torch.randn(2, 1, 8, 8)
    net = dv.wrap(a, sample_input=x)
    same = compare(net, other_net(net, copy.deepcopy(a)), (x,))
    assert all((l["weight"] or 0) == 0 and (l["act"] or 0) == 0 for l in same["layers"])
    b = copy.deepcopy(a)
    with torch.no_grad():
        b[3].weight.mul_(1.5)
    torch.save({"state_dict": {"module." + k: v for k, v in b.state_dict().items()}}, tmp_path / "b.pt")
    diff = {l["id"]: l for l in compare(net, other_net(net, str(tmp_path / "b.pt")), (x,))["layers"]}
    assert diff["0"]["weight"] == 0 and diff["3"]["weight"] > 0.1 and diff["3"]["act"] > 0.1
    with pytest.raises(OptionError, match="different layers"):
        compare(net, other_net(net, nn.Sequential(nn.Flatten(), nn.Linear(64, 3))), (x,))
    with pytest.raises(dv.errors.PathError, match="no such file"):
        other_net(net, str(tmp_path / "missing.pt"))


def test_pages_carry_the_new_views(tmp_path):
    enc = encoder()
    x = torch.randn(1, 5, 16)
    p = payload_of(dv.visualize(enc, x, path=str(tmp_path / "enc.html")).page)
    assert p["interpret"]["rollout"]["matrix"]["rows"] == 5 and len(p["interpret"]["heads"]["effect"]) == 2
    assert p["attribution"] is None and p["compare"] is None
    p = payload_of(dv.visualize(enc, x, path=str(tmp_path / "noheads.html"), heads=False).page)
    assert "heads" not in (p["interpret"] or {})

    cnn = small_cnn()
    img = torch.randn(2, 1, 8, 8)
    p = payload_of(dv.visualize(cnn, img, path=str(tmp_path / "attr.html"), attribution=True,
                                labels=["a", "b", "c"]).page)
    A = p["attribution"]
    assert A["kind"] == "image" and A["methods"] == ["integrated_gradients", "saliency", "grad_cam"]
    assert [t["label"] for t in A["targets"]] and len(A["maps"]["grad_cam"]) == 3 and A["base"].startswith("data:")

    b = copy.deepcopy(cnn)
    with torch.no_grad():
        b[0].bias.add_(0.5)
    p = payload_of(dv.compare(cnn, b, img, names=("before", "after"), path=str(tmp_path / "cmp.html")).page)
    C = p["compare"]
    assert C["names"] == ["before", "after"] and p["ui"]["start"] == "compare"
    assert {l["id"] for l in C["layers"] if (l["weight"] or 0) > 0} == {"0"} and len(C["outputs"]) == 2
