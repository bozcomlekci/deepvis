import os

import numpy as np
import pytest

import deepvis as dv
from deepvis.anim.data import architecture_data, dense_data, edit_data, equation_data

pytestmark = pytest.mark.anim


def test_data_builders(mlp, transformer):
    model, x = mlp
    net = dv.wrap(model, sample_input=x)
    d = dense_data(net, (x,))
    assert [w.shape for w in d["weights"]] == [(8, 6), (5, 8), (3, 5)]
    e = edit_data(net, (x,), {"op": "zero", "target": "net.2"})
    assert np.all(e["after"]["weights"][1] == 0) and len(net.history) == 0
    assert "system" in equation_data(net)["equations"]
    tnet = dv.wrap(transformer[0], sample_input=transformer[1])
    a = architecture_data(tnet)
    assert max(n["layer"] for n in a["nodes"]) < 14


def test_render_forward(mlp, tmp_path):
    pytest.importorskip("manim")
    model, x = mlp
    net = dv.wrap(model, sample_input=x)
    out = net.animate("forward", x, path=str(tmp_path / "f.mp4"))
    assert os.path.getsize(out) > 10_000


def test_volumes_data_and_render(tmp_path):
    torch = pytest.importorskip("torch")
    from conftest import CNN
    from deepvis.anim.data import volumes_data

    torch.manual_seed(0)
    m, x = CNN().eval(), torch.randn(1, 3, 8, 8)
    net = dv.wrap(m, sample_input=x)
    d = volumes_data(net, (x,))
    kinds = [b["kinds"] for b in d["blocks"]]
    assert kinds[0] == ["input"] and ["Conv2d", "BatchNorm", "ReLU"] in kinds
    assert all(len(b["face"]["colors"]) == b["face"]["rows"] * b["face"]["cols"] for b in d["blocks"])
    pytest.importorskip("manim")
    out = net.animate("volumes", x, path=str(tmp_path / "v.mp4"))
    assert os.path.getsize(out) > 10_000


def test_training_data_and_render(mlp, tmp_path):
    import torch

    from deepvis.anim.data import training_data

    model, x = mlp
    net = dv.wrap(model, sample_input=x)
    log = dv.TrainingLog(net, every_steps=5)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    for step in range(60):
        opt.zero_grad()
        loss = model(x).pow(2).mean()
        loss.backward()
        opt.step()
        log.record(step, loss=loss.item())
    d = training_data(net, log, max_snapshots=8)
    assert d["curve"]["name"] == "loss" and len(d["curve"]["values"]) == 60
    assert d["grid"].shape == (3, 8) and d["colors"].shape == (3, 8)  # 3 Linear layers × 8 snapshot buckets
    assert np.isfinite(d["grid"]).all()  # buckets average over the snapshots they hold
    full = training_data(net, log, max_snapshots=100)["grid"]
    assert full.shape == (3, 12) and np.isnan(full[:, 0]).all() and np.isfinite(full[:, 1:]).all()
    with pytest.raises(ValueError):
        training_data(net, dv.TrainingLog(net))
    pytest.importorskip("manim")
    out = net.animate("training", log=log, path=str(tmp_path / "t.mp4"))
    assert os.path.getsize(out) > 10_000


def test_render_narrated_architecture_and_equation(mlp, tmp_path):
    pytest.importorskip("manim")
    model, x = mlp
    net = dv.wrap(model, sample_input=x)
    for kind in ("architecture", "equation"):
        out = net.animate(kind, x, path=str(tmp_path / f"{kind}.mp4"))
        assert os.path.getsize(out) > 10_000
    out = net.animate("forward", x, path=str(tmp_path / "quiet.mp4"), captions=False)
    assert os.path.getsize(out) > 10_000


def test_render_forward_with_an_image_input(tmp_path):
    pytest.importorskip("manim")
    torch = pytest.importorskip("torch")
    from deepvis.anim.scenes import _pixel_squares  # noqa: F401 - the pixels → input neurons step exists

    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(64, 8), torch.nn.Sigmoid(), torch.nn.Linear(8, 3)).eval()
    x = torch.rand(1, 1, 8, 8)
    out = dv.wrap(m, sample_input=x).animate("forward", x, path=str(tmp_path / "img.mp4"))
    assert os.path.getsize(out) > 10_000
