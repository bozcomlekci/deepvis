"""Architecture views: recurrent layers unrolled, mixture-of-experts routing, Mamba Δ, ViT / CLIP images, graph
networks, and JAX parity (attention interior, logit lens, training from the page)."""

import json
import re

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

import deepvis as dv  # noqa: E402


def payload_of(path):
    html = open(path, encoding="utf-8").read()
    return json.loads(re.search(r'id="dv-payload" type="application/json">(.*?)</script>', html, re.S).group(1)
                      .replace("<\\/", "</"))


# ------------------------------------------------------------------------------------------ recurrent
@pytest.mark.parametrize("kind,kw,batch_first", [
    ("LSTM", {"bidirectional": True, "num_layers": 2}, True),
    ("GRU", {"num_layers": 2}, False),
    ("RNN", {"nonlinearity": "relu"}, True),
])
def test_recurrent_unrolling_reproduces_the_layer(kind, kw, batch_first):
    from deepvis.interpret.recurrent import unroll_node

    torch.manual_seed(0)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = getattr(nn, kind)(5, 7, batch_first=batch_first, **kw)

        def forward(self, x):
            return self.rnn(x)[0]

    x = torch.randn(2, 6, 5) if batch_first else torch.randn(6, 2, 5)
    net = dv.wrap(M(), sample_input=x)
    u = unroll_node(net, net.trace(x), "rnn", sample=1)
    assert u["match"] < 1e-5 and u["steps"] == 6
    assert len(u["layers"]) == kw.get("num_layers", 1) * (2 if kw.get("bidirectional") else 1)
    gates = {"LSTM": 4, "GRU": 3, "RNN": 0}[kind]
    assert len(u["layers"][0]["gates"]) == gates


def test_recurrent_payload(tmp_path):
    torch.manual_seed(0)

    class Seq(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm, self.head = nn.LSTM(2, 8, batch_first=True), nn.Linear(8, 3)

        def forward(self, x):
            return self.head(self.lstm(x)[0][:, -1])

    p = payload_of(dv.visualize(Seq(), torch.randn(2, 12, 2), path=str(tmp_path / "s.html")).page)
    R = p["arch"]["recurrent"]["lstm"]
    assert R["kind"] == "LSTM" and R["steps"] == 12 and [g["name"] for g in R["layers"][0]["gates"]][0] == "input gate i"
    assert R["layers"][0]["h"]["cols"] == 12 and R["layers"][0]["c"] is not None


# ------------------------------------------------------------------------------------------ mixture of experts
class TinyMoE(nn.Module):
    def __init__(self, d=16, E=6, k=2):
        super().__init__()
        self.gate = nn.Linear(d, E, bias=False)
        self.experts = nn.ModuleList(nn.Linear(d, d) for _ in range(E))
        self.top_k = k

    def forward(self, x):
        p = self.gate(x).softmax(-1)
        w, idx = p.topk(self.top_k, -1)
        out = torch.zeros_like(x)
        for j in range(self.top_k):
            for e, ex in enumerate(self.experts):
                m = (idx[..., j] == e).unsqueeze(-1)
                out = out + m * w[..., j:j + 1] * ex(x)
        return out


def test_moe_routing_matches_the_router(tmp_path):
    from deepvis.interpret.arch import find_routers, routing

    torch.manual_seed(0)
    model = nn.Sequential(TinyMoE())
    x = torch.randn(2, 9, 16)
    net = dv.wrap(model, sample_input=x)
    assert find_routers(net) == [{"router": "0.gate", "block": "0", "experts": 6, "k": 2}]
    r = routing(net, net.trace(x), sample=1, batch=2)[0]
    own = model[0].gate(x[1]).softmax(-1).topk(2, -1).indices
    assert np.array_equal(np.sort(r["chosen"], -1), np.sort(own.numpy(), -1))
    assert r["load"].sum() == 9 * 2
    p = payload_of(dv.visualize(model, x, path=str(tmp_path / "m.html")).page)
    assert p["arch"]["moe"][0]["experts"] == 6 and len(p["arch"]["moe"][0]["chosen"]) == 9


def test_mixtral_routing():
    T = pytest.importorskip("transformers")
    from deepvis.interpret.arch import routing

    torch.manual_seed(0)
    c = T.MixtralConfig(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                        num_attention_heads=4, num_key_value_heads=2, num_local_experts=4, num_experts_per_tok=2)
    m = T.MixtralForCausalLM(c).eval()
    ids = torch.randint(0, 100, (2, 8))
    net = dv.wrap(m, sample_input=ids)
    r = routing(net, net.trace(ids), sample=1, batch=2)[0]
    got = {}
    h = m.model.layers[0].mlp.gate.register_forward_hook(lambda mod, i, o: got.__setitem__("idx", o[2]) or None)
    with torch.no_grad():
        m(ids)
    h.remove()
    assert np.array_equal(np.sort(r["chosen"], -1), got["idx"].reshape(2, 8, 2)[1].sort(-1).values.numpy())


# ------------------------------------------------------------------------------------------ state spaces, images
def test_mamba_delta(tmp_path):
    T = pytest.importorskip("transformers")
    m = T.MambaForCausalLM(T.MambaConfig(vocab_size=100, hidden_size=32, state_size=8, num_hidden_layers=2,
                                         expand=2)).eval()
    ids = torch.randint(0, 100, (1, 10))
    p = payload_of(dv.visualize(m, ids, path=str(tmp_path / "mb.html")).page)
    S = p["arch"]["ssm"]
    assert [s["mixer"] for s in S] == ["backbone.layers.0.mixer", "backbone.layers.1.mixer"]
    assert len(S[0]["perToken"]) == 10 and min(S[0]["perToken"]) > 0 and S[0]["delta"]["rows"] == 10


def test_vit_and_clip_images(tmp_path):
    T = pytest.importorskip("transformers")
    torch.manual_seed(0)
    vit = T.ViTForImageClassification(T.ViTConfig(image_size=32, patch_size=8, num_channels=3, hidden_size=32,
                                                  num_hidden_layers=2, num_attention_heads=4, intermediate_size=64,
                                                  num_labels=5)).eval()
    p = payload_of(dv.visualize(vit, torch.randn(1, 3, 32, 32), path=str(tmp_path / "v.html"), heads=False).page)
    assert p["inputImage"]["w"] == 32 and p["interpret"]["rollout"]["row"] == 0  # the CLS token
    tc = dict(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
              max_position_embeddings=16, bos_token_id=0, eos_token_id=1)
    vc = dict(image_size=32, patch_size=8, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
              num_attention_heads=4)
    clip = T.CLIPModel(T.CLIPConfig(text_config=tc, vision_config=vc, projection_dim=16)).eval()
    ids, px = torch.randint(2, 100, (3, 6)), torch.randn(2, 3, 32, 32)
    p = payload_of(dv.visualize(clip, (ids, px), path=str(tmp_path / "c.html"), labels=["a", "b", "c"],
                                heads=False).page)
    assert p["inputImage"] is not None and p["output"]["labels"] == ["a", "b", "c"]


# ------------------------------------------------------------------------------------------ graph networks
def test_graph_network(tmp_path):
    pyg = pytest.importorskip("torch_geometric.nn")
    torch.manual_seed(0)

    class GCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1, self.c2 = pyg.GCNConv(4, 8), pyg.GCNConv(8, 2)

        def forward(self, x, ei):
            return self.c2(torch.relu(self.c1(x, ei)), ei)

    x = torch.randn(12, 4)
    ei = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])
    net = dv.wrap(GCN(), sample_input=(x, ei))
    kinds = {n.kind for n in net.graph.compute_nodes()}
    assert {"GCNConv", "ReLU"} <= kinds and len(net.graph.compute_nodes()) <= 8  # one node per message passing
    assert dv.theme.kind_family("GCNConv") == "graph"
    p = payload_of(dv.visualize(GCN(), (x, ei), path=str(tmp_path / "g.html")).page)
    G = p["arch"]["graph"]
    assert G["n"] == 12 and len(G["edges"]) == 11 and [l["id"] for l in G["layers"]][:2] == ["input", "c1"]
    assert p["meta"]["batch"] == 1  # the graph's nodes are not a batch
    assert all(0 <= c <= 1 for xy in G["pos"] for c in xy)


# ------------------------------------------------------------------------------------------ JAX parity
@pytest.fixture
def flax_lm():
    pytest.importorskip("flax")
    import flax.linen as fnn
    import jax
    import jax.numpy as jnp

    class Block(fnn.Module):
        @fnn.compact
        def __call__(self, x):
            h = fnn.LayerNorm()(x)
            x = x + fnn.MultiHeadDotProductAttention(num_heads=4, qkv_features=32)(
                h, mask=fnn.make_causal_mask(jnp.ones(x.shape[:2])))
            return x + fnn.Dense(32)(fnn.gelu(fnn.Dense(64)(fnn.LayerNorm()(x))))

    class LM(fnn.Module):
        def setup(self):
            self.emb, self.blocks, self.ln = fnn.Embed(100, 32), [Block() for _ in range(2)], fnn.LayerNorm()

        def __call__(self, ids):
            x = self.emb(ids)
            for b in self.blocks:
                x = b(x)
            return self.unembed(x)

        def unembed(self, h):
            return self.emb.attend(self.ln(h))

    lm = LM()
    ids = jnp.array([[1, 5, 9, 3, 7, 2]])
    return lm, lm.init(jax.random.PRNGKey(0), ids), ids


@pytest.mark.jax
def test_flax_attention_interior_and_lens(flax_lm, tmp_path):
    lm, variables, ids = flax_lm
    net = dv.wrap(lm, variables, sample_input=ids)
    tr = net.trace(ids)
    parts = tr.attention_parts[next(iter(tr.attention_parts))]
    q, k = parts["q"], parts["k"]
    s = q @ np.swapaxes(k, -1, -2) / np.sqrt(q.shape[-1])
    s = np.where(np.tril(np.ones(s.shape[-2:])) > 0, s, -np.inf)
    mine = np.exp(s - s.max(-1, keepdims=True))
    assert np.allclose(mine / mine.sum(-1, keepdims=True), parts["probs"], atol=1e-5)
    assert np.isnan(parts["scores"][0, 0, 0, 1])  # the causal mask
    with pytest.raises(dv.errors.ModelError, match="lens_fn"):
        net.logit_lens(ids)
    lens_fn = lambda h: lm.apply(variables, h, method=type(lm).unembed)  # noqa: E731
    assert len(net.logit_lens(ids, lens_fn=lens_fn).layers) == 3
    p = payload_of(dv.visualize(lm, ids, variables, path=str(tmp_path / "f.html"), lens_fn=lens_fn).page)
    assert len(p["anatomy"]) == 2 and len(p["lens"]["layers"]) == 3


@pytest.mark.jax
def test_jax_training_from_the_page(flax_lm):
    import jax

    lm, variables, ids = flax_lm
    net = dv.wrap(lm, variables, sample_input=ids)

    def step(i):  # "train": nudge every weight; hand the new tree back as params
        _, cur = net.export()
        return {"loss": 1.0 / (i + 1), "params": jax.tree_util.tree_map(lambda a: a * 0.9, cur)}

    srv = net.serve(ids, open_browser=False, train_step=step, min_interval=0.0)
    try:
        w0 = np.asarray(net.export()[1]["params"]["emb"]["embedding"]).copy()
        srv.train("step")
        srv.train("step")
        w1 = np.asarray(net.export()[1]["params"]["emb"]["embedding"])
        assert np.allclose(w1, w0 * 0.81, atol=1e-6)
        srv.train("reset")
        assert np.allclose(np.asarray(net.export()[1]["params"]["emb"]["embedding"]), w0)
    finally:
        srv.stop()
