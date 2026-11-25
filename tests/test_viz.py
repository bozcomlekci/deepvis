import json
import urllib.request

import numpy as np
import pytest
import torch

import deepvis as dv
from deepvis.anim.texlite import latex_to_pango
from deepvis.viz.html.dashboard import Dashboard
from deepvis.viz.html.payload import PayloadBuilder


def _layout_ok(level):
    nodes = {n["id"]: n["layout"] for n in level["nodes"] if "layout" in n}
    rows = {}
    for nid, l in nodes.items():
        rows.setdefault(l["layer"], []).append(l)
    for row in rows.values():
        row.sort(key=lambda l: l["x"])
        for a, b in zip(row, row[1:]):
            assert a["x"] + a["w"] / 2 <= b["x"] - b["w"] / 2 + 1e-6, "nodes overlap"
    return True


@pytest.mark.parametrize("fixture", ["mlp", "resnet", "transformer", "cnn"])
def test_payload_serializable_and_laid_out(request, fixture):
    model, x = request.getfixturevalue(fixture)
    net = dv.wrap(model, sample_input=x)
    p = PayloadBuilder(net).build((x,))
    s = json.dumps(p)
    assert len(s) < 5_000_000
    assert all(_layout_ok(lv) for lv in p["levels"])
    assert p["acts"] and p["params"] and p["equations"]["modes"]["system"]["lines"]
    html = Dashboard(net, (x,)).html()
    assert '<script id="dv-payload"' in html and "__PAYLOAD__" not in html and "</script" not in s


def test_neuron_view(mlp):
    model, x = mlp
    net = dv.wrap(model, sample_input=x)
    nv = PayloadBuilder(net).build((x,))["neurons"]
    assert [l["size"] for l in nv["layers"]] == [6, 8, 5, 3]
    assert nv["equations"]


def test_live_server_roundtrip(mlp):
    model, x = mlp
    net = dv.wrap(model, sample_input=x)
    srv = net.serve(x, open_browser=False)
    try:
        def post(route, body):
            req = urllib.request.Request(srv.url.rstrip("/") + route, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req).read())

        y0 = net(x).detach().clone()
        r = post("/api/edit", {"op": "scale", "target": "net.0", "args": {"factor": 0}})
        assert "payload" in r and not torch.equal(net(x), y0)
        post("/api/undo", {})
        assert torch.equal(net(x), y0)
        page = urllib.request.urlopen(srv.url).read().decode()
        assert "window.DV_API = \"\"" in page
    finally:
        srv.stop()


def test_texlite():
    assert latex_to_pango(r"W^{(1)}\,x") == "<i>W</i><sup>(1)</sup>\u2009<i>x</i>"
    assert latex_to_pango(r"\mathrm{ReLU}\!\left(z\right)") == "ReLU(<i>z</i>)"
    assert "ℝ<sup>3×4</sup>" in latex_to_pango(r"\in \mathbb{R}^{3\times 4}")


def test_volume_faces():
    from deepvis.viz.html.payload import volume_payload

    v = volume_payload(np.random.randn(6, 5, 4), 0)
    assert v["kind"] == "maps" and v["dims"] == [6, 5, 4]
    shapes = {k: (f["rows"], f["cols"]) for k, f in v["faces"].items()}
    assert shapes == {"front": (5, 4), "side": (5, 6), "top": (4, 6)}
    assert volume_payload(np.random.randn(7, 9), None)["kind"] == "seq"
    assert volume_payload(np.random.randn(10), None)["dims"] == [1, 10, 1]


def test_live_playground(mlp):
    model, x = mlp
    net = dv.wrap(model, sample_input=x)
    srv = net.serve(x, open_browser=False)
    try:
        def post(route, body):
            req = urllib.request.Request(srv.url.rstrip("/") + route, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req).read())

        r = post("/api/input", {"op": "set_vector", "data": [1.0] * 6})
        assert r["payload"]["input"]["kind"] == "vector" and r["payload"]["input"]["values"] == [1.0] * 6
        assert torch.allclose(srv.args[0][0], torch.ones(6))
        post("/api/input", {"op": "reset"})
        assert torch.equal(srv.args[0], x)
        r = post("/api/intervene", {"node": "net.1", "type": "clamp", "min": 0, "max": 0.1})
        ints = r["payload"]["interventions"]
        assert len(ints) == 1 and ints[0]["node"] == "net.1"
        assert r["payload"]["acts"]["net.1"]["stats"]["max"] <= 0.1 + 1e-6
        r = post("/api/remove_intervention", {"id": ints[0]["id"]})
        assert r["payload"]["interventions"] == []
        pp = r["payload"]["params"]["net.0.weight"]
        assert pp["perm"] == [0, 1] and pp["cshape"] == [8, 6]
        post("/api/edit", {"op": "set", "target": "net.0.weight", "index": "[2, 3]", "args": {"value": 4.0}})
        assert model.net[0].weight[2, 3].item() == 4.0
    finally:
        srv.stop()


def test_sample_selection(mlp):
    model, x = mlp
    xb = torch.randn(5, x.shape[-1], generator=torch.Generator().manual_seed(3))
    net = dv.wrap(model, sample_input=xb)
    ref = model(xb).detach().numpy()
    pay = PayloadBuilder(net).build((xb,), sample=3)
    assert pay["meta"]["batch"] == 5 and pay["meta"]["sample"] == 3
    assert np.allclose(pay["output"]["values"], ref[3], atol=1e-6)
    first = net.graph.param_nodes()[0].id
    assert np.isclose(pay["acts"][first]["stats"]["mean"], net.trace(xb)[first][3].mean(), atol=1e-6)
    assert PayloadBuilder(net).build((xb,), sample=99)["meta"]["sample"] == 4  # clamped
    srv = net.serve(xb, open_browser=False)
    try:
        req = urllib.request.Request(srv.url + "api/sample", data=json.dumps({"index": 2}).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        state = json.load(urllib.request.urlopen(req))["payload"]
        assert state["meta"]["sample"] == 2 and np.allclose(state["output"]["values"], ref[2], atol=1e-6)
        assert state["input"]["sample"] == 2
    finally:
        srv.stop()


def test_class_labels_on_outputs(mlp):
    model, x = mlp
    net = dv.wrap(model, sample_input=x)
    n_out = model(x).shape[-1]
    net.labels = [f"class {i}" for i in range(n_out)]
    assert PayloadBuilder(net).build((x,))["output"]["labels"] == net.labels
    net.labels = ["just one"]  # wrong length: ignored rather than mislabeled, with a warning saying so
    with pytest.warns(dv.DeepvisWarning, match="1 names but the output has"):
        assert "labels" not in PayloadBuilder(net).build((x,))["output"]
