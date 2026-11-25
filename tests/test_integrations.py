"""deepvis in other tools: PNG / PDF figures, TensorBoard, Weights & Biases, and the notebook widget's Python side."""

import json

import numpy as np
import pytest

import deepvis as dv

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch


def _net():
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3)).eval()
    x = torch.randn(2, 4)
    return dv.wrap(m, sample_input=x), x


def _cairosvg():
    try:
        import cairosvg  # noqa: F401
    except (ImportError, OSError) as exc:  # OSError: the cairo library itself is missing
        pytest.skip(f"cairosvg unusable: {exc}")


def test_png_and_pdf_figures(tmp_path):
    _cairosvg()
    net, x = _net()
    png = net.export_svg(str(tmp_path / "f.png"), x)
    pdf = net.export_svg(str(tmp_path / "f.pdf"), x)
    with open(png, "rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n"
    with open(pdf, "rb") as f:
        assert f.read(5) == b"%PDF-"
    from deepvis.cli import main

    torch.save(net.backend.model.state_dict(), tmp_path / "w.pt")
    main(["export", str(tmp_path / "w.pt"), "-o", str(tmp_path / "w.png")])  # the command line writes them too
    assert (tmp_path / "w.png").stat().st_size > 1000


def test_tensorboard(tmp_path):
    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from torch.utils.tensorboard import SummaryWriter

    net, x = _net()
    log = dv.TrainingLog(net)
    log.record(3, loss=0.25)
    w = SummaryWriter(str(tmp_path))
    dv.integrations.log_tensorboard(w, net, 3, x, training=log)
    w.close()
    ea = EventAccumulator(str(tmp_path), size_guidance={"histograms": 0, "scalars": 0})
    ea.Reload()
    hists, scalars = set(ea.Tags()["histograms"]), set(ea.Tags()["scalars"])
    assert {"weights/0.weight", "weights/2.bias", "activations/1"} <= hists
    assert {"stats/0.weight/norm", "activations/1/mean", "train/loss"} <= scalars
    norm = ea.Scalars("stats/0.weight/norm")[0].value
    assert norm == pytest.approx(float(np.linalg.norm(net.backend.get_param("0.weight"))), rel=1e-5)
    from deepvis.errors import OptionError

    with pytest.raises(OptionError, match="SummaryWriter"):
        dv.integrations.log_tensorboard(object(), net, 0)


def test_weights_and_biases_offline(tmp_path, monkeypatch):
    wandb = pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_SILENT", "true")
    net, x = _net()
    from deepvis.errors import OptionError

    if wandb.run is None:
        with pytest.raises(OptionError, match="wandb.init"):
            dv.integrations.log_wandb(net, x)
    run = wandb.init(mode="offline", dir=str(tmp_path), project="deepvis-test")
    try:
        logged = dv.integrations.log_wandb(net, x, key="model", step=0)
    finally:
        run.finish()
    assert isinstance(logged["model"], wandb.Html)
    assert list(tmp_path.glob("wandb/offline-run-*/files/media/html/*.html"))


def test_widget_answers_like_the_live_server():
    pytest.importorskip("anywidget")
    net, x = _net()
    w = net.widget(x, height=500)
    assert w.height == 500 and "dv-widget:" in w.html and "__dvRequest" in w.html
    status, body = w.answer("GET", "/api/version", None)
    assert status == 200 and json.loads(body)["version"] == 0
    status, body = w.answer("GET", "/api/state", None)
    assert status == 200 and json.loads(body)["meta"]["live"]
    status, body = w.answer("POST", "/api/edit", json.dumps({"op": "scale", "target": "2.weight", "args": {"factor": 0}}))
    assert status == 200 and "payload" in json.loads(body)
    assert not net.backend.get_param("2.weight").any()
    status, body = w.answer("POST", "/api/edit", json.dumps({"op": "nope", "target": "2.weight"}))
    assert status == 400 and "error" in json.loads(body)
    sent = []
    w.send = lambda msg, buffers=None: sent.append(msg)
    w._on_msg(w, {"type": "request", "id": 7, "method": "GET", "path": "/api/history", "body": None}, [])
    assert sent[0]["id"] == 7 and json.loads(sent[0]["body"])["edits"]
