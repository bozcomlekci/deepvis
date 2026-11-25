"""Watching a network learn: TrainingLog, LiveServer.update, net.load for every framework."""

import json
import urllib.request

import numpy as np
import pytest

import deepvis as dv


# ------------------------------------------------------------------ torch
torch = pytest.importorskip("torch")
tnn = torch.nn


def _torch_setup():
    torch.manual_seed(0)
    m = tnn.Sequential(tnn.Linear(2, 16), tnn.BatchNorm1d(16), tnn.Tanh(), tnn.Dropout(0.2), tnn.Linear(16, 2))
    x = torch.randn(64, 2)
    y = (x[:, 0] * x[:, 1] > 0).long()
    return m, x, y


@pytest.mark.torch
def test_training_log_records_metrics_and_layer_dynamics():
    m, x, y = _torch_setup()
    net = dv.wrap(m, sample_input=x[:8])
    log = dv.TrainingLog(net)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    for step in range(0, 20, 2):  # snapshots two steps apart
        opt.zero_grad()
        loss = tnn.functional.cross_entropy(m(x), y)
        loss.backward()
        opt.step()
        log.record(step, loss=loss.item(), lr=0.1)
    log.record(18)  # a bare refresh of the same step adds nothing
    assert len(log) == 10 and len(log.snap_steps) == 10
    assert log.metrics["lr"] == [0.1] * 10
    rec = log.layers["0"]
    assert set(rec) == {"update", "norm", "grad", "act"}
    assert rec["update"][0] is None and all(v > 0 for v in rec["update"][1:])
    assert np.isclose(rec["norm"][-1], float(torch.linalg.vector_norm(torch.cat([m[0].weight.flatten(), m[0].bias]).detach())),
                      rtol=1e-5)
    assert rec["grad"][-1] > 0  # picked up from .grad
    assert "1" in log.layers and all(p.endswith(("weight", "bias")) for p in log._groups["1"])  # no BN buffers
    d = log.to_dict(max_points=4, max_snapshots=3)
    assert len(d["steps"]) == 4 and len(d["metrics"]["loss"]) == 4 and len(d["snapSteps"]) == 3
    assert d["lastStep"] == 18 and d["count"] == 10
    json.dumps(d, allow_nan=False)


@pytest.mark.torch
def test_update_ratio_is_per_step():
    m, x, _ = _torch_setup()
    net = dv.wrap(m, sample_input=x[:8])
    log = dv.TrainingLog(net)
    log.snapshot(0)
    w0 = m[4].weight.detach().clone()
    with torch.no_grad():
        m[4].weight.add_(0.1 * w0)  # a 10% change over 5 steps
    log.snapshot(5)
    b = m[4].bias.detach()
    full = float(torch.linalg.vector_norm(0.1 * w0) / torch.linalg.vector_norm(torch.cat([w0.flatten(), b])))
    assert np.isclose(log.layers["4"]["update"][-1], full / 5, rtol=1e-4)


@pytest.mark.torch
def test_live_server_follows_training_without_side_effects():
    m, x, y = _torch_setup()
    net = dv.wrap(m, sample_input=x[:8])
    srv = net.serve(x[:8], open_browser=False, min_interval=0.0)
    try:
        opt = torch.optim.Adam(m.parameters(), 1e-2)
        m.train()
        v0 = json.load(urllib.request.urlopen(srv.url + "api/version"))["version"]
        for step in range(5):
            opt.zero_grad()
            loss = tnn.functional.cross_entropy(m(x), y)
            loss.backward()
            opt.step()
            stats = m[1].running_mean.clone()
            assert srv.update(step, loss=loss.item())
            assert torch.equal(stats, m[1].running_mean), "the dashboard's forward pass must not touch BN stats"
            assert m.training and m[3].training, "train/eval flags are restored"
        v = json.load(urllib.request.urlopen(srv.url + "api/version"))
        assert v["version"] == v0 + 5 and v["step"] == 4
        state = json.load(urllib.request.urlopen(srv.url + "api/state"))
        assert state["meta"]["serverVersion"] == v["version"]
        tr = state["training"]
        assert tr["lastStep"] == 4 and len(tr["metrics"]["loss"]) == 5 and len(tr["snapSteps"]) == 5
        srv.min_interval = 3600  # metrics still logged, page state not rebuilt
        assert not srv.update(5, loss=0.1)
        assert json.load(urllib.request.urlopen(srv.url + "api/version"))["version"] == v["version"]
        assert srv.update(5, render=True)
    finally:
        srv.stop()


@pytest.mark.torch
def test_export_html_with_training(tmp_path):
    m, x, y = _torch_setup()
    net = dv.wrap(m, sample_input=x[:8])
    log = dv.TrainingLog(net)
    for step in range(3):
        log.record(step, loss=1.0 / (step + 1))
    path = net.export_html(tmp_path / "run.html", x[:8], training=log, start="monitor")
    html = open(path, encoding="utf-8").read()
    assert '"training":{' in html and '"lastStep":2' in html


@pytest.mark.torch
def test_torch_load_state_dict():
    m, x, _ = _torch_setup()
    net = dv.wrap(m, sample_input=x[:8])
    other = tnn.Sequential(tnn.Linear(2, 16), tnn.BatchNorm1d(16), tnn.Tanh(), tnn.Dropout(0.2), tnn.Linear(16, 2))
    net.load(other.state_dict())
    assert torch.equal(m[0].weight, other[0].weight)
    with pytest.raises(KeyError):
        net.load({"nope": torch.zeros(1)})


# ------------------------------------------------------------------ jax
@pytest.mark.jax
def test_jax_frameworks_load_new_weights():
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    lnn = pytest.importorskip("flax.linen")
    nnx = pytest.importorskip("flax.nnx")
    eqx = pytest.importorskip("equinox")
    x = jax.random.normal(jax.random.PRNGKey(0), (4, 3))

    # linen: new params (without the "params" wrapper) replace the old ones, batch stats are kept
    class M(lnn.Module):
        @lnn.compact
        def __call__(self, x):
            return lnn.Dense(2)(lnn.relu(lnn.Dense(5)(x)))

    m = M()
    v = m.init(jax.random.PRNGKey(1), x)
    net = dv.wrap(m, v, sample_input=x)
    v2 = m.init(jax.random.PRNGKey(2), x)
    net.load(v2["params"])
    assert np.allclose(net(x), m.apply(v2, x), atol=1e-6)
    g = jax.grad(lambda p: jnp.sum(m.apply({"params": p}, x)))(v2["params"])
    log = dv.TrainingLog(net)
    log.snapshot(0, grads=g)
    assert log.layers["Dense_0"]["grad"][0] > 0

    # pure function + params pytree
    def f(p, x):
        return jnp.tanh(x @ p["w"]) @ p["v"]

    p1 = {"w": jnp.ones((3, 4)), "v": jnp.ones((4, 2))}
    pn = dv.wrap((f, p1), sample_input=x)
    p2 = jax.tree_util.tree_map(lambda a: a * 0.5, p1)
    pn.load(p2)
    assert np.allclose(pn(x), f(p2, x), atol=1e-6)
    with pytest.raises(ValueError):
        pn.load({"w": jnp.ones((3, 4))})

    # equinox: a whole new model
    e1 = eqx.nn.MLP(3, 2, 4, 1, key=jax.random.PRNGKey(3))
    e2 = eqx.nn.MLP(3, 2, 4, 1, key=jax.random.PRNGKey(4))
    en = dv.wrap(e1, sample_input=x[0])
    en.load(e2)
    assert np.allclose(en(x[0]), e2(x[0]), atol=1e-6)

    # nnx: modules are trained in place; the live server re-reads them on update()
    class N(nnx.Module):
        def __init__(self, rngs):
            self.fc = nnx.Linear(3, 2, rngs=rngs)

        def __call__(self, x):
            return self.fc(x)

    mod = N(nnx.Rngs(0))
    nn_net = dv.wrap(mod, sample_input=x)
    srv = nn_net.serve(x, open_browser=False, min_interval=0.0)
    try:
        mod.fc.kernel[...] = mod.fc.kernel[...] * 2  # what an optimizer step does
        srv.update(1, loss=0.5)
        assert np.allclose(nn_net(x), mod(x), atol=1e-6)
        assert srv.training.layers["fc"]["update"][-1] is None  # first snapshot
    finally:
        srv.stop()


def _trainable():
    m, x, y = _torch_setup()
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    resets = []

    def step(i):
        m.train()
        loss = tnn.functional.cross_entropy(m(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        return {"loss": loss}

    return m, x, step, resets


@pytest.mark.torch
def test_training_driven_from_the_page():
    import time

    m, x, step, resets = _trainable()
    net = dv.wrap(m, sample_input=x[:8])
    w0 = m[0].weight.detach().clone()
    srv = net.serve(x[:8], open_browser=False, train_step=step, train_steps=30, train_reset=lambda: resets.append(1),
                    min_interval=0.0)
    try:
        assert srv.payload()["meta"]["trainer"] == {"running": False, "step": 0, "total": 30, "error": None}
        srv.train("step")
        assert srv.train_step_count == 1 and not torch.equal(w0, m[0].weight)
        srv.train("start")
        deadline = time.time() + 20
        while srv.trainer_state()["running"] and time.time() < deadline:
            time.sleep(0.05)
        assert srv.train_step_count == 30  # stops at train_steps
        assert srv.training.metrics["loss"][-1] < srv.training.metrics["loss"][0]
        assert "already trained" in srv.train("start")
        req = urllib.request.Request(srv.url + "api/train", data=json.dumps({"action": "reset"}).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        state = json.load(urllib.request.urlopen(req))["payload"]
        assert state["meta"]["trainer"]["step"] == 0 and state["training"] is None
        assert torch.equal(w0, m[0].weight) and resets == [1]  # initial weights back, optimizer rebuilt
    finally:
        srv.stop()


@pytest.mark.torch
def test_a_failing_train_step_is_reported_not_fatal():
    import time

    m, x, _, _ = _trainable()
    net = dv.wrap(m, sample_input=x[:8])

    def bad(i):
        raise RuntimeError("out of data")

    srv = net.serve(x[:8], open_browser=False, train_step=bad)
    try:
        srv.train("start")
        deadline = time.time() + 10
        while srv.trainer_state()["running"] and time.time() < deadline:
            time.sleep(0.05)
        st = srv.trainer_state()
        assert not st["running"] and st["error"] == "RuntimeError in train_step: out of data"
        assert srv.payload()["meta"]["trainer"]["error"]  # the page shows it
    finally:
        srv.stop()
