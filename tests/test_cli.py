"""The ``deepvis`` command line."""

import xml.etree.ElementTree as ET

import pytest

from deepvis.cli import main

torch = pytest.importorskip("torch")

MODELS = '''
import torch, torch.nn as nn

def build():
    return nn.Sequential(nn.Conv2d(1, 4, 3), nn.ReLU(), nn.Flatten(), nn.Linear(4 * 6 * 6, 3)), torch.randn(1, 1, 8, 8)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(12, 8), nn.Linear(8, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))
'''


@pytest.fixture
def models(tmp_path):
    p = tmp_path / "models.py"
    p.write_text(MODELS, encoding="utf-8")
    return p


def test_export_factory_html_and_svg(models, tmp_path, capsys):
    html, svg = tmp_path / "a.html", tmp_path / "a.svg"
    main(["export", f"{models}:build", "-o", str(html), "-o", str(svg), "--title", "Tiny", "--direction", "LR"])
    out = capsys.readouterr().out.split()
    assert out == [str(html), str(svg)]
    assert '"title":"Tiny"' in html.read_text(encoding="utf-8")
    ET.fromstring(svg.read_text(encoding="utf-8"))


def test_export_class_with_shape_and_options(models, tmp_path):
    out = tmp_path / "m.html"
    main(["export", f"{models}:MLP", "--shape", "4,12", "-o", str(out), "--views", "graph,equations",
          "--start", "equations", "--depth", "all", "--style", "compact"])
    text = out.read_text(encoding="utf-8")
    assert '"views":["graph","equations"]' in text and '"start":"equations"' in text


def test_errors_are_explained(models, tmp_path):
    with pytest.raises(SystemExit, match="needs an input shape"):
        main(["export", f"{models}:MLP", "-o", str(tmp_path / "x.html")])
    with pytest.raises(SystemExit, match="unknown output format"):
        main(["export", f"{models}:build", "-o", str(tmp_path / "x.gif")])
    with pytest.raises(SystemExit, match="can't interpret"):
        main(["export", "nonsense", "-o", str(tmp_path / "x.html")])


def test_summary_torchvision(capsys):
    pytest.importorskip("torchvision")
    main(["summary", "torchvision:resnet18"])
    out = capsys.readouterr().out
    assert "ResNet" in out and "11,689,512" in out


def test_export_timm(tmp_path):
    pytest.importorskip("timm")
    out = tmp_path / "t.svg"
    main(["export", "timm:resnet18", "--shape", "1,3,64,64", "-o", str(out)])
    ET.fromstring(out.read_text(encoding="utf-8"))


CNN = '''
import torch.nn as nn

def build():
    return nn.Sequential(nn.Conv2d(3, 4, 3), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 2))
'''


def test_export_with_image_input(tmp_path):
    from PIL import Image

    (tmp_path / "cnn.py").write_text(CNN, encoding="utf-8")
    img = tmp_path / "x.png"
    Image.new("RGB", (40, 30), (200, 30, 30)).save(img)
    out = tmp_path / "c.html"
    main(["export", f"{tmp_path / 'cnn.py'}:build", "--image", str(img), "--shape", "1,3,32,32", "-o", str(out)])
    assert '"input":[[1,3,32,32]]' in out.read_text(encoding="utf-8")
    with pytest.raises(SystemExit, match="needs a tokenizer"):
        main(["export", f"{tmp_path / 'cnn.py'}:build", "--text", "hi", "-o", str(out)])
