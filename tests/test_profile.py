"""net.profile(data, labels): per-unit statistics checked against NumPy, top inputs, dead units, the 2-D map, linear
CKA's properties, the data sources it reads, and the Dataset tab."""

import numpy as np
import pytest

import deepvis as dv
from deepvis.monitor.profile import _cka_matrix, _Sketch

torch = pytest.importorskip("torch")
nn = torch.nn


def _mlp():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 4)).eval()
    with torch.no_grad():
        m[0].bias[:3] = -100.0  # three units that can never fire
    return m


@pytest.mark.torch
def test_unit_statistics_match_numpy_and_dead_units_are_found():
    m = _mlp()
    X = torch.randn(300, 6)
    net = dv.wrap(m, sample_input=X[:1])
    prof = net.profile(X, batch_size=64)  # 5 batches of 64 and one of 44: the running statistics must agree
    with torch.no_grad():
        h = m[1](m[0](X)).numpy()
        out = m(X).numpy()
    u = prof.units["1"]
    np.testing.assert_allclose(u["mean"], h.mean(0), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(u["std"], h.std(0, ddof=1), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(u["active"], (h > 0).mean(0))
    assert prof.dead_units()["1"] == [0, 1, 2] and u["nonneg"] and prof.n == 300
    best = prof.top_inputs("2", 1)
    assert [i for i, _ in best] == list(np.argsort(-out[:, 1])[:8]) and best[0][1] == pytest.approx(out[:, 1].max(), rel=1e-5)
    assert prof.pred == list(out.argmax(1))


def test_linear_cka_properties_and_the_sketch():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 30))
    q, _ = np.linalg.qr(rng.standard_normal((30, 30)))
    y = rng.standard_normal((200, 30))
    m = _cka_matrix([x, 3.0 * x @ q, y])  # rotation and scale leave CKA at 1; unrelated features don't
    assert m[0, 0] == pytest.approx(1) and m[0, 1] == pytest.approx(1, abs=1e-5) and m[0, 2] < 0.3
    np.testing.assert_allclose(m, m.T)
    wide = rng.standard_normal((50, 2000)).astype(np.float32)
    sk = _Sketch.of(2000)
    a, b = sk(wide), sk(wide.copy())
    np.testing.assert_array_equal(a, b)  # the same size always hashes the same way
    gram, gram_sk = wide @ wide.T, a @ a.T  # inner products survive the sketch, roughly
    assert np.corrcoef(gram.ravel(), gram_sk.ravel())[0, 1] > 0.9


@pytest.mark.torch
def test_convolutions_profile_per_channel_and_flatten_is_the_same_representation():
    torch.manual_seed(1)
    m = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(), nn.Flatten(), nn.Linear(4 * 36, 3)).eval()
    X = torch.randn(64, 1, 6, 6)
    net = dv.wrap(m, sample_input=X[:1])
    prof = net.profile(X, labels=torch.arange(64) % 3, batch_size=16, top_k=4)
    with torch.no_grad():
        r = m[1](m[0](X)).numpy()
    np.testing.assert_allclose(prof.units["1"]["mean"], r.mean((2, 3)).mean(0), rtol=1e-4, atol=1e-6)
    assert len(prof.units["2"]["mean"]) == 144 and prof.labels[:4] == [0, 1, 2, 0]
    i = prof.layers.index("1")
    assert prof.cka[i, i + 1] == pytest.approx(1.0, abs=1e-4)  # the ReLU output and its flattened copy
    assert prof.embed["3"].shape == (64, 2) and len(prof.thumbs) == 64


@pytest.mark.torch
def test_data_loaders_dicts_and_the_dataset_tab(tmp_path):
    import json
    import re

    m = _mlp()
    ds = torch.utils.data.TensorDataset(torch.randn(100, 6), torch.randint(0, 4, (100,)))
    net = dv.wrap(m, sample_input=torch.randn(1, 6))
    prof = net.profile(torch.utils.data.DataLoader(ds, batch_size=32), max_samples=70)
    assert prof.n == 70 and len(prof.labels) == 70  # labels from the (x, y) batches; stopped at max_samples
    gen = ({"input": torch.randn(10, 6), "labels": torch.zeros(10, dtype=torch.long)} for _ in range(3))

    class Named(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = _mlp()

        def forward(self, input):
            return self.m(input)

    prof2 = dv.wrap(Named(), sample_input=torch.randn(1, 6)).profile(gen)
    assert prof2.n == 30 and set(prof2.labels) == {0}
    page = net.export_html(str(tmp_path / "p.html"), torch.randn(1, 6), dataset=prof, start="dataset")
    payload = json.loads(re.search(r'<script id="dv-payload" type="application/json">(.*?)</script>',
                                   open(page, encoding="utf-8").read(), re.S).group(1))
    d = payload["dataset"]
    assert d["n"] == 70 and set(d["layers"]) >= {"0", "1", "2"} and d["cka"]["m"][0][0] == pytest.approx(1)
    assert d["top"]["2"][0]["items"] and d["units"]["1"]["dead"] == 3 and payload["ui"]["start"] == "dataset"
    from deepvis.errors import OptionError

    with pytest.raises(OptionError, match="net.profile"):
        net.export_html(str(tmp_path / "q.html"), torch.randn(1, 6), dataset={"not": "a profile"})


@pytest.mark.jax
def test_a_jax_model_profiles_too():
    jax = pytest.importorskip("jax")
    fnn = pytest.importorskip("flax.linen")

    class MLP(fnn.Module):
        @fnn.compact
        def __call__(self, x):
            return fnn.Dense(3)(fnn.relu(fnn.Dense(8)(x)))

    x = jax.numpy.ones((1, 5))
    mod = MLP()
    net = dv.wrap(mod, mod.init(jax.random.PRNGKey(0), x), sample_input=x)
    X = np.random.default_rng(0).standard_normal((40, 5)).astype(np.float32)
    prof = net.profile(X, batch_size=16)
    assert prof.n == 40 and prof.layers and all(len(u["mean"]) in (8, 3) for u in prof.units.values())


@pytest.mark.torch
def test_a_position_table_is_not_dead_although_every_sample_gets_the_same_one():
    class Pos(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok, self.pos = nn.Embedding(50, 8), nn.Embedding(12, 8)

        def forward(self, ids):
            return self.tok(ids) + self.pos(torch.arange(ids.shape[1]))

    ids = torch.randint(0, 50, (32, 12))
    prof = dv.wrap(Pos().eval(), sample_input=ids[:1]).profile(ids, batch_size=8)
    assert "pos" not in prof.layers and prof.units["tok"]["dead"] == 0  # the same for every sample: not profiled
    assert prof.units["tok"]["mean"].shape == (8,) and prof.n == 32
