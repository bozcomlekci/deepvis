"""Huge models: meta-device pages (architecture and shapes, no numbers), `--meta` on the command line, layouts that
stay fast when a value feeds every layer, and lighter pages and live updates."""

import json

import pytest

import deepvis as dv

torch = pytest.importorskip("torch")
nn = torch.nn
pytestmark = pytest.mark.torch


def _meta_llama(layers=2):
    tf = pytest.importorskip("transformers")
    cfg = tf.LlamaConfig(vocab_size=1000, hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
                         num_attention_heads=4, num_key_value_heads=2)
    with torch.device("meta"):
        return tf.LlamaForCausalLM(cfg).eval(), torch.randint(0, 100, (1, 8), device="meta")


def test_a_meta_device_model_gets_an_architecture_page(tmp_path):
    from deepvis.errors import OptionError
    from deepvis.viz.html.payload import PayloadBuilder

    m, ids = _meta_llama()
    net = dv.wrap(m, sample_input=ids)
    assert not net.backend.has_data and net.num_params() == sum(p.numel() for p in m.parameters())
    assert net.summary(show=False).startswith("A transformer-style network that turns a sequence of 8 token ids")
    with pytest.raises(OptionError, match="meta device"):
        net.trace(ids)
    with pytest.raises(OptionError, match="no values"):
        net.backend.get_param("lm_head.weight")
    pay = PayloadBuilder(net).build(None)
    assert pay["meta"]["shapesOnly"] and pay["meta"]["input"] == [[1, 8]]
    assert pay["params"] == {} and not pay["acts"] and pay["neurons"] is None
    assert pay["equations"]["modes"]["system"]["lines"] and pay["explain"]
    page = net.export_html(str(tmp_path / "p.html"))
    assert "shapesOnly" in open(page, encoding="utf-8").read()


def test_the_command_line_builds_any_config_on_the_meta_device(tmp_path, capsys):
    tf = pytest.importorskip("transformers")
    from deepvis.cli import main

    tf.LlamaConfig(vocab_size=1000, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                   num_key_value_heads=2, architectures=["LlamaForCausalLM"]).save_pretrained(str(tmp_path / "cfg"))
    out = tmp_path / "p.html"
    main(["export", f"hf:{tmp_path / 'cfg'}", "--meta", "-o", str(out)])
    assert out.exists()
    main(["summary", "torchvision:resnet18", "--meta"])
    assert "11,689,512 learned numbers" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["summary", "my_model.py:build", "--meta"])


def test_a_value_every_layer_reads_does_not_route_through_every_row():
    from deepvis.core.graph import ModelGraph, Node
    from deepvis.viz.html.layout import layout

    g = ModelGraph("deep", "torch")
    g.add_node(Node("", "Model", "Model", "container"))
    g.add_node(Node("@in:x", "Input", "Input", "input", parent=""))
    g.add_node(Node("table", "Embedding", "Embedding", "module", parent=""))  # e.g. rotary cos/sin
    g.add_edge("@in:x", "table")
    prev = "table"
    for k in range(60):
        g.add_node(Node(f"b{k}", "Linear", "Linear", "module", parent=""))
        g.add_edge(prev, f"b{k}")
        if k % 10 == 9:
            g.add_edge("table", f"b{k}")  # read again, far below
        prev = f"b{k}"
    g.add_node(Node("@out", "Output", "Output", "output", parent=""))
    g.add_edge(prev, "@out")
    g.link_children()
    lay = layout(g)
    far = [e for e in lay["edges"] if e.get("far")]
    assert len(far) == 6 and all(len(e["points"]) == 2 for e in far)  # straight: no dummy chain
    xs = {round(v["x"]) for k, v in lay["nodes"].items() if k.startswith("b")}
    assert len(xs) == 1  # the main chain stays one straight column


def test_live_updates_compress_and_big_blocks_skip_their_own_equation(monkeypatch):
    import gzip
    import urllib.request

    from deepvis.viz.html import payload as P

    tf = pytest.importorskip("transformers")
    m = tf.GPT2LMHeadModel(tf.GPT2Config(n_layer=2, n_embd=64, n_head=2, vocab_size=500, bos_token_id=0,
                                         eos_token_id=0)).eval()
    ids = torch.randint(0, 500, (1, 16))
    net = dv.wrap(m, sample_input=ids)
    srv = net.serve(ids, open_browser=False)
    try:
        req = urllib.request.Request(srv.url + "api/state", headers={"Accept-Encoding": "gzip"})
        r = urllib.request.urlopen(req)
        assert r.headers.get("Content-Encoding") == "gzip"
        assert json.loads(gzip.decompress(r.read()))["meta"]["name"]
    finally:
        srv.stop()
    counts = P._compute_counts(net.graph)
    assert counts[""] == len(net.graph.compute_nodes())
    monkeypatch.setattr(P.Budget, "block_eq", 20)
    eqs = P.PayloadBuilder(net).build((ids,))["equations"]["perNode"]
    assert "" not in eqs and "transformer.h.0.attn.c_attn" in eqs  # the whole model: in the Equations tab instead
