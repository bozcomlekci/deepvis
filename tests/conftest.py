import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn


class MLP(nn.Module):
    def __init__(self, sizes=(6, 8, 5, 3), act=nn.Tanh):
        super().__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)

    def forward(self, x):
        return x + torch.relu(self.ln(self.fc(x)))


class ResNet(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.inp = nn.Linear(5, d)
        self.blocks = nn.Sequential(Block(d), Block(d))
        self.out = nn.Linear(d, 3)

    def forward(self, x):
        return torch.softmax(self.out(self.blocks(torch.relu(self.inp(x)))), dim=-1)


class Transformer(nn.Module):
    def __init__(self, vocab=20, d=16):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(12, d) * 0.1)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 2, 32, dropout=0.0, batch_first=True), 2)
        self.head = nn.Linear(d, vocab)

    def forward(self, ids):
        return self.head(self.enc(self.embed(ids) + self.pos[: ids.shape[1]]))


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(torch.flatten(self.pool(torch.relu(self.bn(self.conv(x)))), 1))


@pytest.fixture
def mlp():
    torch.manual_seed(0)
    return MLP().eval(), torch.randn(4, 6)


@pytest.fixture
def resnet():
    torch.manual_seed(0)
    return ResNet().eval(), torch.randn(4, 5)


@pytest.fixture
def transformer():
    torch.manual_seed(0)
    return Transformer().eval(), torch.randint(0, 20, (2, 7))


@pytest.fixture
def cnn():
    torch.manual_seed(0)
    return CNN().eval(), torch.randn(2, 3, 8, 8)


def allclose(a, b, atol=1e-5):
    from deepvis.core.tensor import to_numpy

    return np.allclose(to_numpy(a), to_numpy(b), atol=atol)
