"""Regressions from code review: NaN pages, live-server request checks, undo during training, edit groups."""

import json
import os
import urllib.error
import urllib.request

import pytest

import deepvis as dv
from deepvis.manipulate.edits import Edit

torch = pytest.importorskip("torch")
tnn = torch.nn


def _mlp():
    torch.manual_seed(0)
    return tnn.Sequential(tnn.Linear(4, 8), tnn.ReLU(), tnn.Linear(8, 3))


@pytest.mark.torch
def test_nan_activations_still_make_a_page(tmp_path):
    m = _mlp()
    with torch.no_grad():
        m[0].weight.fill_(float("nan"))  # a diverged run
    net = dv.wrap(m)
    x = torch.randn(2, 4)
    assert net.export_html(str(tmp_path / "nan.html"), x)
    srv = net.serve(x, open_browser=False)
    try:
        assert json.load(urllib.request.urlopen(srv.url + "api/state"))
    finally:
        srv.stop()


@pytest.mark.torch
def test_live_server_refuses_other_sites():
    srv = dv.wrap(_mlp()).serve(torch.randn(2, 4), open_browser=False)

    def status(path, data=None, **headers):
        req = urllib.request.Request(srv.url + path, data=data, headers=headers, method="POST" if data else "GET")
        try:
            return urllib.request.urlopen(req).status
        except urllib.error.HTTPError as e:
            return e.code

    try:
        body = b'{"index": 0}'
        assert status("api/sample", body, **{"Content-Type": "text/plain"}) == 403  # CSRF via a no-cors fetch
        assert status("api/sample", body, **{"Content-Type": "application/json", "Sec-Fetch-Site": "cross-site"}) == 403
        assert status("api/sample", body, **{"Content-Type": "application/json", "Sec-Fetch-Site": "same-origin"}) == 200
        assert status("api/state", Host="evil.example") == 403  # DNS rebinding
        assert status("", **{"Sec-Fetch-Site": "cross-site"}) == 200  # the page framed in a notebook / VS Code
    finally:
        srv.stop()


@pytest.mark.torch
def test_undo_keeps_training_progress_outside_the_edit():
    m = _mlp()
    net = dv.wrap(m)
    net.apply_edit(Edit("zero", "0.weight", index="0:2"))
    with torch.no_grad():
        m[0].weight += 1.0  # training between the edit and the undo
    trained = m[0].weight.detach().clone()
    net.undo()
    assert torch.equal(m[0].weight[2:], trained[2:])
    assert not torch.equal(m[0].weight[:2], trained[:2])


@pytest.mark.torch
def test_training_reset_keeps_the_log_and_clears_edits():
    m = tnn.Linear(4, 2)
    net = dv.wrap(m)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)

    def step(i):
        opt.zero_grad()
        loss = m(torch.randn(8, 4)).pow(2).mean()
        loss.backward()
        opt.step()
        return loss.item()

    from deepvis.viz.server import LiveServer

    mine = dv.TrainingLog(net, frames=0, max_snapshots=100)
    srv = LiveServer(net, (torch.randn(2, 4),), train_step=step, training=mine)
    srv.train("step")
    net.apply_edit(Edit("zero", "weight"))
    srv.train("reset")
    assert srv.training is mine and mine.frames == 0 and mine.max_snapshots == 100 and not mine.steps
    assert net.undo() == []


@pytest.mark.torch
def test_logging_a_step_twice_adds_to_it():
    log = dv.TrainingLog(dv.wrap(tnn.Linear(2, 2)))
    log.record(5)
    log.record(5, loss=0.3)  # HuggingFace: on_step_end, then on_log
    log.record(6, loss=0.2)
    assert log.steps == [5, 6] and log.metrics["loss"] == [0.3, 0.2]


@pytest.mark.torch
def test_group_edit_sees_its_earlier_edits():
    m = tnn.Sequential(tnn.Linear(4, 3), tnn.ReLU(), tnn.Linear(3, 2))
    net = dv.wrap(m)
    net.trace(torch.randn(1, 4))
    w0 = m[0].weight.detach().clone()
    net.apply_edit(Edit("group", "0", args={"edits": [Edit("scale", "0", args={"factor": 2.0}),
                                                       Edit("permute_neurons", "0", args={"perm": [2, 0, 1]})]}))
    assert torch.allclose(m[0].weight, 2 * w0[[2, 0, 1]])


@pytest.mark.torch
def test_scaling_an_integer_input():
    from deepvis.viz.server import LiveServer

    class Id(tnn.Module):
        def forward(self, x):
            return x.float()

    srv = LiveServer(dv.wrap(Id()), (torch.full((1, 4), 200, dtype=torch.uint8),))
    srv._set_input({"op": "scale", "factor": 0.5})
    assert srv.args[0].tolist() == [[100, 100, 100, 100]]


@pytest.mark.torch
def test_capture_wins_over_path(tmp_path, monkeypatch):
    from deepvis import CAPTURE_ENV, CAPTURE_STOP_ENV
    from deepvis.errors import Captured

    cap, mine = tmp_path / "cap.html", tmp_path / "mine.html"
    monkeypatch.setenv(CAPTURE_ENV, str(cap))
    monkeypatch.setenv(CAPTURE_STOP_ENV, "1")
    with pytest.raises(Captured):
        dv.visualize(tnn.Linear(3, 2), torch.randn(1, 3), path=str(mine))
    assert os.path.exists(cap) and not os.path.exists(mine)


@pytest.mark.torch
def test_reset_restores_edits_pushed_out_of_the_history():
    m = _mlp()
    net = dv.wrap(m)
    net.history.limit = 3
    w0 = m[0].weight.detach().clone()
    net.apply_edit(Edit("zero", "0.weight", index="0"))
    for _ in range(5):
        net.apply_edit(Edit("scale", "2.weight", args={"factor": 1.5}))
    net.reset()
    assert torch.equal(m[0].weight, w0)


@pytest.mark.torch
def test_restore_layer_keeps_training_progress():
    m = _mlp()
    net = dv.wrap(m)
    net.apply_edit(Edit("zero", "0.weight", index="0:2"))
    with torch.no_grad():
        m[0].weight += 1.0
    trained = m[0].weight.detach().clone()
    vals = net.history.restore_values(["0.weight"])
    assert torch.equal(torch.as_tensor(vals["0.weight"][2:]), trained[2:])


@pytest.mark.torch
def test_training_reset_keeps_edits_made_before_serve():
    from deepvis.viz.server import LiveServer

    m = tnn.Linear(4, 2)
    net = dv.wrap(m)
    w0 = m.weight.detach().clone()
    net.apply_edit(Edit("zero", "weight"))
    srv = LiveServer(net, (torch.randn(2, 4),), train_step=lambda i: 0.0)
    net.apply_edit(Edit("scale", "bias", args={"factor": 2.0}))
    srv.train("reset")
    assert len(net.history) == 1
    net.undo()
    assert torch.equal(m.weight, w0)


@pytest.mark.torch
def test_nan_in_edit_history_is_valid_json():
    srv = dv.wrap(_mlp()).serve(torch.randn(2, 4), open_browser=False)
    try:
        srv.net.apply_edit(Edit("clamp", "0.weight", args={"min": float("-inf"), "max": 0.5}))
        assert json.loads(urllib.request.urlopen(srv.url + "api/history").read().decode())["edits"]
    finally:
        srv.stop()


@pytest.mark.torch
def test_forwarded_host_reported_by_vscode_is_allowed():
    from deepvis.viz.server import _forbidden

    assert _forbidden({"Host": "x-8000.app.github.dev"}, "127.0.0.1", False, True)
    assert not _forbidden({"Host": "x-8000.app.github.dev"}, "127.0.0.1", False, True, {"x-8000.app.github.dev"})


def test_nan_conv_filters_and_infinite_activations():
    import numpy as np

    from deepvis.viz.html.payload import _abs_max, act_payload

    assert _abs_max(np.array([np.nan, -3.0, 2.0])) == 3.0
    top = act_payload(np.array([0.1, np.inf, 0.2, -np.inf, np.nan]), batched=False)["top"]
    assert {i for i, _ in top[:2]} == {1, 3}


def _post(srv, route, body):
    req = urllib.request.Request(srv.url + route.lstrip("/"), data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(req).read())


@pytest.mark.torch
def test_page_edits_run_the_model_twice_in_eval_mode(monkeypatch):
    torch.manual_seed(0)
    m = tnn.Sequential(tnn.Linear(4, 8), tnn.BatchNorm1d(8), tnn.Dropout(0.5), tnn.Linear(8, 3)).train()
    net = dv.wrap(m)
    srv = net.serve(torch.randn(6, 4), open_browser=False)
    try:
        stats = m[1].running_mean.clone()
        calls = []
        real = type(net).trace
        monkeypatch.setattr(type(net), "trace", lambda self, *a, **k: calls.append(1) or real(self, *a, **k))
        r = _post(srv, "/api/edit", {"op": "scale", "target": "3.weight", "args": {"factor": 1.0}})
        assert len(calls) == 2
        assert r["message"].endswith("= 0")  # no dropout noise in the measured change
        assert torch.equal(m[1].running_mean, stats) and m.training  # no BatchNorm updates, mode restored
    finally:
        srv.stop()


@pytest.mark.torch
def test_page_state_is_serialized_once_per_version(monkeypatch):
    import deepvis.viz.server as server

    srv = dv.wrap(_mlp()).serve(torch.randn(2, 4), open_browser=False)
    try:
        calls = []
        real = server.dumps
        monkeypatch.setattr(server, "dumps", lambda o, **k: calls.append(1) or real(o, **k))
        a = urllib.request.urlopen(srv.url + "api/state").read()
        b = urllib.request.urlopen(srv.url + "api/state").read()
        assert a == b and len(calls) == 1
        assert _post(srv, "/api/sample", {"index": 0})["payload"]["meta"]["serverVersion"] == srv.version
    finally:
        srv.stop()


@pytest.mark.torch
def test_replay_frames_read_only_each_layers_main_weight(monkeypatch):
    net = dv.wrap(tnn.Linear(4, 3))
    log = dv.TrainingLog(net)
    read = []
    real = net.backend.get_param
    monkeypatch.setattr(net.backend, "get_param", lambda p: read.append(p) or real(p))
    log._keep_frame(0, None)
    assert read == ["weight"]


def test_cli_calls_factories_but_not_model_instances(tmp_path, monkeypatch):
    nnx = pytest.importorskip("flax.nnx")
    from deepvis.cli import _load_attr

    (tmp_path / "models.py").write_text(
        "import torch\nfrom flax import nnx\n"
        "inst = nnx.Linear(2, 3, rngs=nnx.Rngs(0))\n"
        "def build():\n    return torch.nn.Linear(2, 3)\n"
        "Cls = torch.nn.Identity\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert isinstance(_load_attr(f"{tmp_path}/models.py:inst"), nnx.Linear)
    assert isinstance(_load_attr(f"{tmp_path}/models.py:build"), tnn.Linear)
    assert isinstance(_load_attr(f"{tmp_path}/models.py:Cls"), tnn.Identity)


@pytest.mark.torch
def test_cli_loads_a_windows_checkpoint_path(monkeypatch):
    import deepvis.files as files
    from deepvis.cli import load_model

    seen = []  # the drive's colon must not make it look like a module:attr spec
    monkeypatch.setattr(files, "open_file", lambda path, **k: seen.append(path) or tnn.Linear(2, 2))
    model, _, _ = load_model(r"C:\models\net.pt")
    assert seen == [r"C:\models\net.pt"] and isinstance(model, tnn.Linear)


# ------------------------------------------------------------------------------------------ bug hunt
@pytest.mark.torch
def test_intervening_on_a_module_that_returns_a_model_output():
    tf = pytest.importorskip("transformers")
    torch.manual_seed(0)
    m = tf.GPT2LMHeadModel(tf.GPT2Config(n_layer=1, n_embd=32, n_head=2, vocab_size=100,
                                         bos_token_id=0, eos_token_id=0)).eval()
    x = torch.randint(0, 100, (1, 6))
    net = dv.wrap(m, sample_input=x)
    with torch.no_grad():
        base = m(x).logits
        h = net.intervene("transformer", dv.amplify(1.0))  # its output is a ModelOutput, not a tensor or tuple
        try:
            assert torch.allclose(m(x).logits, base)
        finally:
            h.remove()


@pytest.mark.torch
def test_looking_at_a_training_model_leaves_it_untouched(tmp_path):
    torch.manual_seed(0)
    m = tnn.Sequential(tnn.Linear(4, 8), tnn.BatchNorm1d(8), tnn.Dropout(0.5), tnn.Linear(8, 2)).train()
    x = torch.randn(16, 4) * 3 + 2
    bn = m[1]
    before = {k: v.clone() for k, v in m.state_dict().items()}
    rng = torch.get_rng_state()
    net = dv.wrap(m, sample_input=x)  # builds the graph: runs the model internally
    assert torch.equal(torch.get_rng_state(), rng)  # your dropout stream is where you left it
    net.trace(x)
    net.export_html(str(tmp_path / "p.html"), x)
    assert all(torch.equal(m.state_dict()[k], v) for k, v in before.items())  # no BatchNorm drift
    assert int(bn.num_batches_tracked) == 0 and m.training


@pytest.mark.jax
def test_jax_forward_uses_this_calls_keyword_arrays():
    jnp = pytest.importorskip("jax.numpy")
    f = lambda p, x, mask=None, scale=1.0: (x @ p["w"]) * (1.0 if mask is None else mask) * scale  # noqa: E731
    params = {"w": jnp.ones((3, 2))}
    x = jnp.ones((1, 3))
    net = dv.wrap((f, params), sample_input=x)
    m1, m2 = jnp.array([[1.0, 1.0]]), jnp.array([[0.0, 5.0]])
    assert net(x, mask=m1).tolist() == [[3.0, 3.0]]
    assert net(x, mask=m2).tolist() == [[0.0, 15.0]]  # same shape, new values: not the first call's mask
    assert net(x, mask=m2, scale=2.0).tolist() == [[0.0, 30.0]]


@pytest.mark.torch
@pytest.mark.parametrize("method", ["saliency", "ig", "grad_cam"])
def test_attribution_keeps_each_layers_training_mode(method):
    torch.manual_seed(0)
    m = tnn.Sequential(tnn.Conv2d(3, 4, 3), tnn.BatchNorm2d(4), tnn.ReLU(), tnn.Flatten(), tnn.LazyLinear(5))
    x = torch.randn(2, 3, 8, 8)
    m(x)
    m.train()
    m[1].eval()  # fine-tuning with frozen BatchNorm
    net = dv.wrap(m, sample_input=x)
    net.attribute(x, method=method, steps=4)
    assert m.training and m[0].training and not m[1].training


@pytest.mark.torch
def test_diffusion_sampling_keeps_each_layers_training_mode():
    from deepvis import diffusion as dd

    class Denoiser(tnn.Module):
        def __init__(self):
            super().__init__()
            self.conv, self.bn = tnn.Conv2d(1, 1, 3, padding=1), tnn.BatchNorm2d(1)

        def forward(self, x, t):
            return self.bn(self.conv(x))

    m = Denoiser().train()
    m.bn.eval()  # frozen BatchNorm while fine-tuning
    dd.sample(m, (1, 1, 4, 4), dd.linear_betas(4), frames=2)
    assert m.training and m.conv.training and not m.bn.training


@pytest.mark.torch
def test_a_recurrent_layer_called_twice_is_unrolled_for_both_calls():
    import warnings

    from deepvis.viz.html.payload import PayloadBuilder

    class Siamese(tnn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = tnn.LSTM(3, 4, batch_first=True)

        def forward(self, a, b):
            return self.lstm(a)[0] - self.lstm(b)[0]

    torch.manual_seed(0)
    a, b = torch.randn(1, 5, 3), torch.randn(1, 5, 3)
    net = dv.wrap(Siamese(), sample_input=(a, b))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no "an architecture view was skipped"
        arch = PayloadBuilder(net).arch(net.trace(a, b))
    assert sorted(arch["recurrent"]) == ["lstm", "lstm@2"]
