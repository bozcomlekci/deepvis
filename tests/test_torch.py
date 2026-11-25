import json

import numpy as np
import pytest
import torch
import torch.nn as nn

import deepvis as dv
from conftest import MLP, allclose


def test_graph_modes_and_edges(resnet, transformer, cnn, mlp):
    for model, x in (resnet, transformer, cnn, mlp):
        net = dv.wrap(model, sample_input=x)
        assert net.backend.trace_mode == "dispatch"
        g = net.graph
        ids = set(g.nodes)
        assert all(e.src in ids and e.dst in ids for e in g.edges)
        assert g.outputs() and g.inputs()
        assert allclose(net(x), model(x))
    net = dv.wrap(resnet[0], sample_input=resnet[1])
    adds = [n for n in net.graph.compute_nodes() if n.kind == "add"]
    assert len(adds) == 2, "residual additions must be visible"
    tnet = dv.wrap(transformer[0], sample_input=transformer[1])
    kinds = [n.kind for n in tnet.graph.compute_nodes()]
    assert kinds.count("MultiheadAttention") == 2 and "Parameter" in kinds


def test_fx_without_sample(resnet):
    net = dv.wrap(resnet[0])
    net.graph  # noqa: B018 - built lazily
    assert net.backend.trace_mode == "fx"
    assert any(n.kind == "add" for n in net.graph.compute_nodes())


def test_trace_matches_manual(resnet):
    model, x = resnet
    net = dv.wrap(model, sample_input=x)
    tr = net.trace(x)
    with torch.no_grad():
        h = torch.relu(model.inp(x))
        assert allclose(tr["inp"], model.inp(x))
        b0 = model.blocks[0]
        assert allclose(tr["blocks.0.ln"], b0.ln(b0.fc(h)))
        assert allclose(tr["@add"], h + torch.relu(b0.ln(b0.fc(h))))
    assert tr.inputs["blocks.0.fc"].shape == (4, 8)


def test_param_editing_and_history(mlp):
    model, x = mlp
    net = dv.wrap(model, sample_input=x)
    y0 = net(x).detach().clone()
    w0 = model.net[0].weight.detach().clone()
    net["net.0.weight"][:, 2] = 0.0
    assert torch.all(model.net[0].weight[:, 2] == 0)
    net["net.0.weight"] *= 2.0
    assert allclose(model.net[0].weight[:, 0], 2 * w0[:, 0])
    net["net.0.weight"][1:3] *= 1.0
    assert len(net.history) == 3, "in-place region ops record exactly one edit"
    net.undo()
    net["net.0"].prune(0.5)
    assert (model.net[0].weight == 0).float().mean().item() >= 0.5
    net.undo(3)
    assert torch.equal(model.net[0].weight, w0) and torch.equal(net(x), y0)
    net.redo()
    assert torch.all(model.net[0].weight[:, 2] == 0)
    net.reset()
    assert torch.equal(model.net[0].weight, w0)


def test_edit_ops(mlp):
    model, x = mlp
    net = dv.wrap(model, sample_input=x)
    net["net.2"].low_rank(2)
    assert np.linalg.matrix_rank(net["net.2.weight"].numpy()) == 2
    net["net.0"].quantize(bits=2)
    assert len(np.unique(net["net.0.weight"].numpy())) <= 4
    net.edit("net.4.weight", "zero", where="abs(w) < 0.2")
    assert np.all((np.abs(net["net.4.weight"].numpy()) >= 0.2) | (net["net.4.weight"].numpy() == 0))
    net["net.2"].neurons([1, 3]).zero()
    assert np.all(net["net.2.weight"].numpy()[[1, 3]] == 0) and np.all(net["net.2.bias"].numpy()[[1, 3]] == 0)
    net.edit("net.0.bias", lambda b: b + 1.0)
    with pytest.raises(KeyError):
        net.edit("nope", "scale", factor=2)


def test_temporary_edits_and_snapshots(mlp):
    model, x = mlp
    net = dv.wrap(model, sample_input=x)
    y0 = net(x).detach().clone()
    snap = net.snapshot("start")
    with net.edits():
        net.select(kind="Linear").scale(0.0)
        assert torch.allclose(net(x), model.net[-1].bias.expand(4, 3))
    assert torch.equal(net(x), y0) and len(net.history) == 0
    net["net.0"].noise(0.5, seed=1)
    assert not torch.equal(net(x), y0)
    net.restore(snap)
    assert torch.equal(net(x), y0)
    rows = net.diff(show=False)
    assert rows == [] or all(r["delta_norm"] == 0 for r in rows)


def test_permute_neurons_preserves_function(mlp):
    model, x = mlp
    net = dv.wrap(model, sample_input=x)
    y0 = net(x).detach().clone()
    net.permute_neurons("net.0", np.random.default_rng(0).permutation(8))
    assert not np.array_equal(model.net[0].weight.detach().numpy(), net.history.original["net.0.weight"])
    assert allclose(net(x), y0, atol=1e-6)


def test_replay_edits_on_clone(mlp):
    model, x = mlp
    net = dv.wrap(model, sample_input=x)
    net["net.0"].prune(0.3)
    net["net.2.weight"][0:2] = 0.5
    net["net.4"].neurons(1).scale(-1)
    log = json.dumps(net.history.export())
    other = dv.wrap(MLP().eval(), sample_input=x)
    for p in other.backend.param_paths():  # same starting weights
        other.backend.set_param(p, net.history.original.get(p, net.backend.get_param(p)))
    other.apply_edits(log)
    for p in net.backend.param_paths():
        assert np.allclose(other.backend.get_param(p), net.backend.get_param(p))


def test_interventions(resnet):
    model, x = resnet
    net = dv.wrap(model, sample_input=x)
    y0 = net(x).detach().clone()
    with net.intervene("inp", dv.ablate([0, 1])):
        tr = net.trace(x)
        assert torch.all(tr._acts_native["inp"][:, :2] == 0)
    assert torch.equal(net(x), y0)
    h = net.intervene("blocks.1.ln", dv.clamp(-0.1, 0.1))
    assert net.trace(x)["blocks.1.ln"].max() <= 0.1 + 1e-6
    h.remove()
    tr = net.trace(x, interventions={"@add": dv.ablate()})  # functional op node
    assert np.all(tr["@add"] == 0)
    r = dv.patch(net, x, x * 0.5, at="blocks.0.ln")
    assert np.isfinite(r.recovery())


def test_monitor_and_grads(resnet):
    model, x = resnet
    net = dv.wrap(model, sample_input=x)
    with net.monitor("blocks.*.fc", stats=("mean", "sparsity")) as mon:
        for _ in range(3):
            model(x)  # the raw model: hooks still fire
            mon.step()
    assert mon["blocks.0.fc"].calls == 3 and len(mon["blocks.0.fc"].series("mean")) == 3
    assert "blocks.1.fc" in mon.table()
    tr = net.trace(x, grads=True, loss_fn=lambda y: y[:, 0].sum())
    xr = x.clone()
    model.zero_grad()
    model(xr)[:, 0].sum().backward()
    assert np.allclose(tr.param_grads["inp.weight"], model.inp.weight.grad.numpy(), atol=1e-6)
    assert "blocks.0.fc" in tr.grads


def test_attention_capture(transformer):
    model, ids = transformer
    net = dv.wrap(model, sample_input=ids)
    tr = net.trace(ids)
    att = tr.attention["enc.layers.0.self_attn"]
    assert att.shape == (2, 2, 7, 7) and np.allclose(att.sum(-1), 1, atol=1e-5)


def test_recorded_activations_survive_inplace_ops():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc, self.bn, self.relu = nn.Linear(4, 4), nn.BatchNorm1d(4), nn.ReLU(inplace=True)

        def forward(self, x):
            out = self.bn(self.fc(x))
            out += x  # in-place residual
            return self.relu(out)  # in-place ReLU rewrites both tensors above

    torch.manual_seed(0)
    m, x = Block().eval(), torch.randn(8, 4)
    net = dv.wrap(m, sample_input=x)
    tr = net.trace(x)
    with torch.no_grad():
        bn = m.bn(m.fc(x))
    assert np.allclose(tr["bn"], bn.numpy(), atol=1e-6) and tr["bn"].min() < 0
    add = [n.id for n in net.graph.compute_nodes() if n.op == "function"][0]
    assert np.allclose(tr[add], (bn + x).numpy(), atol=1e-6)


def test_persistent_hooks_on_functional_ops():
    class R(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = nn.Linear(4, 4), nn.Linear(4, 2)

        def forward(self, x):
            return self.b(torch.relu(self.a(x)) + x)

    torch.manual_seed(0)
    m, x = R(), torch.randn(3, 4)
    net = dv.wrap(m, sample_input=x)
    assert net.backend.function_taps
    y0 = m(x).detach()
    with net.monitor("@add") as mon:
        for _ in range(3):
            m(x)  # your own calls, not net(x)
    assert mon.records["@add"].calls == 3
    assert np.isclose(mon.records["@add"].history["mean"][-1], float((torch.relu(m.a(x)) + x).mean().detach()), atol=1e-6)
    with net.intervene("@add", dv.ablate()):
        y = m(x)
        assert torch.allclose(y, m.b.bias.expand(3, 2))
        assert np.allclose(net.trace(x)["@add"], 0)  # traces see the same intervention
        y.sum().backward()  # and it stays differentiable
    assert torch.allclose(m(x), y0)  # removed cleanly
    assert net.backend._fn_replay is None


def test_functional_models_are_traced_into():
    class MaxNet(nn.Module):  # no submodules: the forward itself is the network
        def forward(self, x):
            v, i = torch.max(x, dim=1)
            return v * 2, i

    net = dv.wrap(MaxNet(), sample_input=torch.randn(3, 5))
    assert net.backend.trace_mode == "dispatch"
    assert {"@max", "@max[0]", "@max[1]", "@mul"} <= set(net.graph.nodes)
    assert net.equation("system").text.splitlines()[-1] == "ŷ₁ = 2 · h₁[0]"
    assert dv.wrap(nn.Linear(3, 2), sample_input=torch.randn(1, 3)).backend.trace_mode == "single"
