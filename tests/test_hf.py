"""HuggingFace transformers (tiny random configs, nothing downloaded)."""

import os
import re

import numpy as np
import pytest
import torch

import deepvis as dv
from deepvis.viz.html.layout import layout

tf = pytest.importorskip("transformers")
pytestmark = pytest.mark.torch


def _gpt2():
    cfg = tf.GPT2Config(n_layer=2, n_head=4, n_embd=32, vocab_size=100, n_positions=32, bos_token_id=0, eos_token_id=0)
    return tf.GPT2LMHeadModel(cfg).eval()


def _llama():
    cfg = tf.LlamaConfig(num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, hidden_size=32,
                         intermediate_size=64, vocab_size=100, bos_token_id=0, eos_token_id=0)
    return tf.LlamaForCausalLM(cfg).eval()


def _bert():
    cfg = tf.BertConfig(num_hidden_layers=2, num_attention_heads=4, hidden_size=32, intermediate_size=64, vocab_size=100)
    return tf.BertForSequenceClassification(cfg).eval()


IDS = torch.randint(0, 100, (1, 8), generator=torch.Generator().manual_seed(0))


@pytest.mark.parametrize("build,expect", [
    (_gpt2, {"GELU", "LayerNorm", "Linear", "Embedding", "Attention"}),
    (_llama, {"RMSNorm", "SiLU", "Linear", "RotaryEmbedding", "Attention"}),
    (_bert, {"GELU", "LayerNorm", "Linear", "Tanh", "Attention"}),
])
def test_hf_models_wrap_with_canonical_kinds(build, expect, tmp_path):
    torch.manual_seed(0)
    model = build()
    net = dv.wrap(model, sample_input=IDS)
    kinds = {n.kind for n in net.graph.compute_nodes()}
    assert expect <= kinds, expect - kinds
    tr = net.trace(IDS)
    ref = model(IDS).logits
    assert torch.allclose(tr.output.logits, ref)
    for mode in ("compact", "system"):  # module names like layer_scale / class_token must be TeX-escaped
        latex = net.equation(mode).latex
        assert not re.search(r"\\texttt\{[^}]*[^\\]_", latex)
    path = net.export_html(tmp_path / "m.html", IDS)
    assert os.path.getsize(path) > 10_000


def test_gpt2_conv1d_is_a_transposed_linear():
    torch.manual_seed(0)
    m = _gpt2()
    net = dv.wrap(m, sample_input=IDS)
    node = net.graph.nodes["transformer.h.0.mlp.c_fc"]
    assert node.kind == "Linear" and node.type_name == "Conv1D"
    r = node.roles["weight"]
    assert (r.out_axis, r.in_axis) == (1, 0)
    # neuron-level edits use the (in, out) orientation: ablating output unit 3 zeroes that activation
    with net.edits():
        net["transformer.h.0.mlp.c_fc"].neurons([3]).zero()
        a = net.trace(IDS).acts["transformer.h.0.mlp.c_fc"]
        assert np.allclose(np.asarray(a)[..., 3], 0)


def test_unused_modules_and_empty_cache_concat_are_hidden():
    torch.manual_seed(0)
    net = dv.wrap(_gpt2(), sample_input=IDS)
    g = net.graph
    unused = [n.id for n in g.compute_nodes() if n.attrs.get("unused")]
    assert any("attn_dropout" in u for u in unused)  # SDPA never calls it
    lay = layout(g)
    assert not set(unused) & set(lay["nodes"])
    # cat(empty KV cache, k) is not a node: attention reads k/v straight from the projections
    assert not any(n.kind == "Concat" for n in g.compute_nodes())


def test_gpt2_reads_as_math():
    torch.manual_seed(0)
    m = _gpt2()
    net = dv.wrap(m, sample_input=IDS)
    text = net.equation("system").text
    assert "= arange" in text  # position ids made inside forward are named, not an unnamed input
    assert all(f"[{i}])ᵀ" in text for i in range(3))  # q, k, v are the three outputs of split
    assert "ŷ = E⁽¹⁾" in text  # lm_head is tied to the token embedding: same symbol
    split = [n.id for n in net.graph.compute_nodes() if n.kind == "split"][0]
    tr = net.trace(IDS)
    q, k, v = (tr[f"{split}[{i}]"] for i in range(3))
    assert np.allclose(np.concatenate([q, k, v], axis=-1), tr["transformer.h.0.attn.c_attn"])
    ref = m(IDS).logits
    with net.intervene(f"{split}[2]", dv.ablate()):  # values only
        assert not torch.allclose(m(IDS).logits, ref)
    assert torch.allclose(m(IDS).logits, ref)


@pytest.mark.parametrize("build", [_gpt2, _llama])
def test_sdpa_attention_patterns_match_eager(build):
    torch.manual_seed(0)
    m = build()
    net = dv.wrap(m, sample_input=IDS)
    att = net.trace(IDS).attention
    assert att, "fused attention exposes no weights; deepvis recomputes them"
    first = att[sorted(att)[0]]
    eager = type(m)(m.config).eval()
    eager.load_state_dict(m.state_dict())
    eager.config._attn_implementation = "eager"
    ref = eager(IDS, output_attentions=True).attentions[0].detach().numpy()
    assert np.allclose(first, ref, atol=1e-5)  # causal mask (and grouped k/v heads for Llama) included


class _Tok:
    """Stands in for a HuggingFace tokenizer (byte-level 'Ġ' marks a leading space)."""

    def __len__(self):
        return 100

    def convert_ids_to_tokens(self, ids):
        return [f"Ġt{i}" for i in ids]


def test_tokenizer_names_tokens_attention_and_predictions():
    from deepvis.viz.html.payload import PayloadBuilder

    torch.manual_seed(0)
    m = _gpt2()
    net = dv.wrap(m, sample_input=IDS)
    net.tokenizer = _Tok()
    pay = PayloadBuilder(net).build((IDS,))
    assert pay["tokens"] == [f" t{i}" for i in IDS[0].tolist()]
    out = pay["output"]
    assert out["position"] == IDS.shape[1] - 1 and out["labelKind"] == "token"
    logits = m(IDS).logits[0, -1].detach()
    top = int(torch.argmax(logits))
    assert out["topk"][0]["id"] == top and out["topk"][0]["label"] == f" t{top}"
    assert np.isclose(out["topk"][0]["p"], float(torch.softmax(logits, -1)[top]), atol=1e-6)
    heads = next(a["attention"]["heads"] for a in pay["acts"].values() if "attention" in a)
    assert heads[0]["rowNames"] == pay["tokens"] and heads[0]["rowLabel"] == "query"
    base = dv.wrap(m.transformer, sample_input=IDS)  # hidden states, not vocabulary logits
    base.tokenizer = _Tok()
    out_base = PayloadBuilder(base).build((IDS,))["output"]
    assert "heat" in out_base and "topk" not in out_base  # hidden states: a heat map, never named as tokens


@pytest.mark.parametrize("build,blocks", [(_gpt2, 2), (_llama, 2)])
def test_logit_lens(build, blocks):
    from deepvis.viz.html.payload import PayloadBuilder

    torch.manual_seed(0)
    m = build()
    net = dv.wrap(m, sample_input=IDS)
    lens = net.logit_lens(IDS, tokenizer=_Tok(), k=3)
    assert lens.layers[0] == "embeddings" and len(lens.layers) == blocks + 1
    assert lens.meta["head"] == "lm_head"
    real = torch.softmax(m(IDS).logits[0], -1).detach().numpy()
    assert np.allclose(lens.probs_all[-1], real[lens.positions], atol=1e-5)  # last row = the real prediction
    assert np.allclose(lens.probs(5)[-1], real[-1, 5], atol=1e-6)
    assert lens.top_ids.shape == (blocks + 1, IDS.shape[1], 3) and "t" in lens.table()
    pay = PayloadBuilder(net).build((IDS,))["lens"]
    assert pay is not None and len(pay["top"]) == blocks + 1 and pay["nodes"][1:] == lens.meta["blocks"][1:]


def test_logit_lens_needs_a_language_model(mlp):
    model, x = mlp
    with pytest.raises(ValueError):  # no repeated blocks before the head
        dv.wrap(model, sample_input=x).logit_lens(x)
