"""dv.open: saved model files (.pt2, .onnx, .keras), weights-only files (state_dict, .safetensors, .npz), the
command line for them, torch.export programs, and dict inputs bound to the forward's arguments."""

import numpy as np
import pytest

import deepvis as dv

torch = pytest.importorskip("torch")
nn = torch.nn
pytestmark = pytest.mark.torch


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv2d(4, 4, 3, padding=1), nn.BatchNorm2d(4)
        self.conv2, self.bn2 = nn.Conv2d(4, 4, 3, padding=1), nn.BatchNorm2d(4)

    def forward(self, x):
        return torch.relu(x + self.bn2(self.conv2(torch.relu(self.bn1(self.conv1(x))))))


def _net():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), Block(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                         nn.Linear(4, 5)).eval()


def test_an_exported_program_draws_like_the_module_it_came_from(tmp_path):
    m = _net()
    x = torch.randn(1, 3, 16, 16)
    ep = torch.export.export(m, (x,))
    path = str(tmp_path / "net.pt2")
    torch.export.save(ep, path)
    net = dv.open(path)  # the program carries its own example input
    original = dv.wrap(m, sample_input=x)
    assert {n.id: n.kind for n in net.graph.compute_nodes()} == {n.id: n.kind for n in original.graph.compute_nodes()}
    torch.testing.assert_close(net.trace(x).output, m(x))
    assert net.graph.nodes["1.conv1"].attrs["padding"] == [1, 1]  # read back from the aten call
    with net.intervene("4", lambda v, ctx: v * 0):  # still a live torch module: interventions work
        assert float(net(x).abs().sum()) == 0.0


def test_weights_files_are_drawn_from_their_names(tmp_path):
    from safetensors.torch import save_file

    m = _net()
    sd = m.state_dict()
    torch.save(sd, tmp_path / "w.pt")
    torch.save({"epoch": 2, "state_dict": sd}, tmp_path / "w.ckpt")
    save_file({k: v.contiguous() for k, v in sd.items()}, str(tmp_path / "w.safetensors"))  # keys sorted
    order = None
    for f in ("w.pt", "w.ckpt", "w.safetensors"):
        net = dv.open(tmp_path / f)
        assert net.framework == "weights"
        nodes = [n for n in net.graph.compute_nodes() if n.op == "module"]
        ids = [n.id for n in nodes]
        assert order is None or ids == order, f  # .safetensors' alphabetical order put back in computation order
        order = ids
        kinds = {n.id: n.kind for n in nodes}
        assert kinds["0"] == "Conv2d" and kinds["1.bn1"] == "BatchNorm" and kinds["4"] == "Linear"
        assert net.summary(show=False).startswith("A weights file")
        np.testing.assert_allclose(net.backend.get_param("4.weight"), sd["4.weight"].numpy())
    assert order == ["0", "1.conv1", "1.bn1", "1.conv2", "1.bn2", "4"]
    from deepvis.errors import OptionError

    with pytest.raises(OptionError, match="no computation"):
        net.trace(torch.randn(1, 3, 16, 16))
    net.backend.set_param("4.bias", np.zeros(5, "float32"))  # edits save back
    out = net.backend.save(str(tmp_path / "edited.safetensors"))
    assert not dv.open(out).backend.get_param("4.bias").any()
    assert net.export_html(str(tmp_path / "p.html"))


def test_bfloat16_safetensors_and_npz(tmp_path):
    from safetensors.torch import save_file

    save_file({"model.layers.0.mlp.up_proj.weight": torch.ones(8, 4, dtype=torch.bfloat16),
               "model.layers.0.input_layernorm.weight": torch.ones(4, dtype=torch.bfloat16),
               "model.embed_tokens.weight": torch.ones(10, 4, dtype=torch.bfloat16)}, str(tmp_path / "lm.safetensors"))
    net = dv.open(tmp_path / "lm.safetensors")
    kinds = [(n.id, n.kind) for n in net.graph.compute_nodes() if n.op == "module"]
    assert kinds == [("model.embed_tokens", "Embedding"), ("model.layers.0.input_layernorm", "RMSNorm"),
                     ("model.layers.0.mlp.up_proj", "Linear")]
    assert net.backend.param_info("model.embed_tokens.weight").dtype == "bfloat16"
    assert net.backend.get_param("model.embed_tokens.weight").dtype == np.float32  # shown as float32
    np.savez(tmp_path / "a.npz", **{"fc.weight": np.ones((3, 2), "float32"), "fc.bias": np.zeros(3, "float32")})
    assert dv.open(tmp_path / "a.npz").graph.nodes["fc"].kind == "Linear"


def test_pickled_modules_need_trust(tmp_path):
    from deepvis.errors import ModelError

    m = _net()
    torch.save(m, tmp_path / "whole.pt")
    with pytest.raises(ModelError, match="runs code from the file"):
        dv.open(tmp_path / "whole.pt")
    net = dv.open(tmp_path / "whole.pt", trusted=True)
    assert net.framework == "torch" and "1.conv1" in net.graph.nodes


def test_onnx_and_keras_files_open_too(tmp_path):
    pytest.importorskip("onnxruntime")
    m = _net()
    x = torch.randn(1, 3, 16, 16)
    torch.onnx.export(m, (x,), str(tmp_path / "net.onnx"), dynamo=False)
    net = dv.open(tmp_path / "net.onnx")
    assert net.framework == "onnx" and net.graph.nodes["4"].kind == "Linear"
    page = dv.visualize(str(tmp_path / "net.onnx"), None, path=str(tmp_path / "v.html"), open_browser=False)
    assert page.framework == "onnx"


def test_the_command_line_opens_files(tmp_path, capsys):
    from deepvis.cli import main

    torch.save(_net().state_dict(), tmp_path / "w.pt")
    out = tmp_path / "w.html"
    main(["export", str(tmp_path / "w.pt"), "-o", str(out)])
    assert out.exists() and str(out) in capsys.readouterr().out
    torch.save(_net(), tmp_path / "whole.pt")
    with pytest.raises(SystemExit):
        main(["summary", str(tmp_path / "whole.pt")])
    assert "--trust" in capsys.readouterr().err
    main(["summary", str(tmp_path / "whole.pt"), "--trust"])
    assert "A convolutional network" in capsys.readouterr().out


def test_dict_inputs_bind_to_the_forward_arguments():
    class Two(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(20, 8)

        def forward(self, input_ids, attention_mask=None, scale=1.0):
            h = self.emb(input_ids) * scale
            return h if attention_mask is None else h * attention_mask[..., None]

    m = Two().eval()
    enc = {"input_ids": torch.randint(0, 20, (1, 5)), "attention_mask": torch.tensor([[1, 1, 1, 0, 0]])}
    net = dv.wrap(m, sample_input=enc)
    assert {n.id for n in net.graph.nodes.values() if n.op == "input"} == {"@in:input_ids", "@in:attention_mask"}
    torch.testing.assert_close(net.trace(enc).output, m(**enc))
    torch.testing.assert_close(net(enc), m(**enc))
    from deepvis.errors import InputError

    with pytest.raises(InputError, match="don't match"):
        dv.wrap(m, sample_input={"pixel_values": torch.zeros(1)})
