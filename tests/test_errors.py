"""Failures explain themselves: what went wrong and how to fix it — and harmless mismatches just work."""

import os
import socket
import sys
import time
import warnings

import numpy as np
import pytest
import torch
import torch.nn as nn

import deepvis as dv
from deepvis import diffusion as dd
from deepvis.errors import DeepvisWarning, InputError, ModelError, OptionError, PathError, RunError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLP = nn.Sequential(nn.Linear(5, 8), nn.ReLU(), nn.Linear(8, 2))


def page(tmp_path, name="p.html"):
    return str(tmp_path / name)


def fails(exc_type, fn, *phrases):
    with pytest.raises(exc_type) as info:
        fn()
    msg = str(info.value)
    assert "→ fix:" in msg, msg
    for p in phrases:
        assert p in msg, f"{p!r} not in:\n{msg}"
    return info.value


@pytest.mark.parametrize("model,phrase", [(None, "is None"), ("resnet", "the string 'resnet'"),
                                          (nn.Linear, "the class Linear"), (3, "is a int")])
def test_not_a_model(tmp_path, model, phrase):
    fails(ModelError, lambda: dv.visualize(model, torch.randn(1, 5), path=page(tmp_path)), phrase)


def test_input_mistakes(tmp_path):
    p = page(tmp_path)
    fails(InputError, lambda: dv.visualize(MLP, None, path=p), "`inputs` is None")
    fails(InputError, lambda: dv.visualize(MLP, {"input_ids": torch.randn(1, 5)}, path=p), "inputs['input_ids']")
    fails(InputError, lambda: dv.visualize(MLP, torch.randn(0, 5), path=p), "empty batch")


def test_the_model_failing_on_the_inputs_is_explained(tmp_path):
    p = page(tmp_path)
    e = fails(RunError, lambda: dv.visualize(MLP, torch.randn(3, 7), path=p), "expects 5 features", "got 7")
    assert isinstance(e.__cause__, RuntimeError)  # the model's own error stays attached
    fails(RunError, lambda: dv.visualize(MLP, torch.randint(0, 3, (2, 5)), path=p), "x.float()")
    emb = nn.Sequential(nn.Embedding(10, 4), nn.Linear(4, 2))
    fails(RunError, lambda: dv.visualize(emb, torch.randn(2, 3), path=p), "integer token ids")
    fails(RunError, lambda: dv.visualize(emb, torch.tensor([[1, 20]]), path=p), "[0, vocab_size)")
    cnn = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU())
    fails(RunError, lambda: dv.visualize(cnn, torch.randn(1, 1, 8, 8), path=p), "expects 3 channels", "has 1")
    fails(RunError, lambda: dv.visualize(cnn, torch.randn(8, 8), path=p), "x.unsqueeze(0)")

    class Two(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 4)

        def forward(self, x, t):
            return self.a(x) * t

    fails(RunError, lambda: dv.visualize(Two(), torch.randn(2, 4), path=p), "(x, t)")


def test_harmless_mismatches_just_work(tmp_path):
    assert dv.visualize(MLP, np.random.randn(3, 5), path=page(tmp_path, "a.html")).page  # numpy (float64)
    assert dv.visualize(MLP, [0.1, 0.2, 0.3, 0.4, 0.5], path=page(tmp_path, "b.html")).page  # a plain list
    assert dv.visualize(MLP, torch.randn(2, 5), path=str(tmp_path / "new" / "dir" / "c.html")).page  # new folders
    if torch.backends.mps.is_available():  # a CPU input for a model on the GPU
        assert dv.visualize(nn.Linear(5, 2).to("mps"), torch.randn(2, 5), path=page(tmp_path, "d.html")).page


def test_options(tmp_path):
    p = page(tmp_path)
    fails(OptionError, lambda: dv.visualize(MLP, torch.randn(2, 5), path=p, titel="x"), "did you mean 'title'")
    fails(OptionError, lambda: dv.visualize(MLP, torch.randn(2, 5), path=p, views=["graf"]), "did you mean 'graph'")
    fails(OptionError, lambda: dv.visualize(MLP, torch.randn(2, 5), path=p, diffusion="yes"), "deepvis.diffusion.sample")
    with pytest.warns(DeepvisWarning, match="no attention"):
        dv.visualize(MLP, torch.randn(2, 5), path=p, start="attention")
    with pytest.warns(DeepvisWarning, match="3 names but the output has 2"):
        dv.visualize(MLP, torch.randn(2, 5), path=p, labels=["a", "b", "c"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeepvisWarning)  # the normal path is quiet
        dv.visualize(MLP, torch.randn(2, 5), path=p, labels=["cat", "dog"])


@pytest.mark.skipif(sys.platform == "win32", reason="chmod doesn't make a folder read-only on Windows")
def test_unwritable_path(tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        fails(PathError, lambda: dv.visualize(MLP, torch.randn(2, 5), path=str(ro / "p.html")), "can't write")
    finally:
        ro.chmod(0o700)


def test_jax_errors():
    jax = pytest.importorskip("jax")
    fnn = pytest.importorskip("flax.linen")
    fails(ModelError, lambda: dv.visualize(fnn.Dense(3), jax.numpy.ones((1, 4))), "params=variables")
    f = lambda p, x: x @ p["w"]  # noqa: E731
    fails(RunError, lambda: dv.visualize((f, {"w": jax.numpy.ones((4, 3))}), jax.numpy.ones((2, 7))),
          "expects 4 features but the input has 7")


def test_diffusion_errors():
    fails(OptionError, lambda: dd.sample(nn.Identity(), (1, 1, 4, 4), dd.linear_betas(5), sampler="ddmp"),
          "did you mean 'ddpm'")
    fails(RunError, lambda: dd.sample(nn.Conv2d(1, 1, 3, padding=1), (1, 1, 4, 4), dd.linear_betas(5)),
          "model(x, t)", "call=")


def test_logit_lens_needs_a_language_model():
    fails(ModelError, lambda: dv.wrap(MLP, sample_input=torch.randn(2, 5)).logit_lens(torch.randn(2, 5)),
          "transformer-style")


def test_live_server_errors():
    fails(InputError, lambda: dv.wrap(MLP).serve(open_browser=False), "net.serve(x)")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        port = s.getsockname()[1]
        fails(OSError, lambda: dv.wrap(MLP, sample_input=torch.randn(1, 5)).serve(port=port, open_browser=False),
              f"port {port} is already in use", "port=0")
    x = torch.randn(4, 5)
    srv = dv.wrap(MLP, sample_input=x).serve(x, open_browser=False, train_step=lambda i: "oops")
    try:
        srv.train("start")
        for _ in range(100):
            if not srv.trainer_state()["running"]:
                break
            time.sleep(0.05)
        assert "return a dict of numbers" in srv.trainer_state()["error"]
    finally:
        srv.stop()
    srv = dv.wrap(MLP, sample_input=x).serve(x, open_browser=False, train_step=lambda i: 0.5, train_steps=2)
    try:  # a bare loss is fine
        srv.train("step")
        assert srv.training.metrics["loss"] == [0.5]
    finally:
        srv.stop()


def test_cli_errors(capsys):
    from deepvis.cli import main

    pytest.importorskip("torchvision")
    with pytest.raises(SystemExit) as e:
        main(["export", "torchvision:resnet19", "-o", "x.html"])
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "no model called 'resnet19'" in err and "did you mean 'resnet18'" in err and "Traceback" not in err
