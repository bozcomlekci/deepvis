import numpy as np
import pytest

from deepvis import theme
from deepvis.core.graph import ModelGraph, Node
from deepvis.core.selectors import filter_paths, format_index, match, parse_index, split_index
from deepvis.core.tensor import downsample2d, histogram, stats
from deepvis.manipulate.ops import eval_where


def test_glob_and_regex():
    paths = ["enc.layers.0.fc", "enc.layers.1.fc", "enc.layers.1.ln", "head"]
    assert filter_paths("enc.layers.*.fc", paths) == ["enc.layers.0.fc", "enc.layers.1.fc"]
    assert filter_paths("**.ln", paths) == ["enc.layers.1.ln"]
    assert filter_paths("re:enc\\.layers\\.[01]\\.fc", paths) == ["enc.layers.0.fc", "enc.layers.1.fc"]
    assert match("head", "head") and not match("hea", "head")
    assert not match("enc.*", "enc.layers.0.fc")  # * stays within one segment


def test_index_roundtrip():
    assert split_index("fc1.weight[:, 3]") == ("fc1.weight", "[:, 3]")
    assert split_index("fc1.weight") == ("fc1.weight", None)
    idx = parse_index("[0:4, ::2, -1]")
    assert idx == (slice(0, 4, None), slice(None, None, 2), -1)
    assert parse_index(format_index((slice(1, None), [0, 2]))) == (slice(1, None, None), [0, 2])
    with pytest.raises(ValueError):
        parse_index("[__import__('os')]")


def test_where_expressions_are_safe():
    w = np.array([-2.0, -0.001, 0.0005, 3.0])
    assert eval_where("abs(w) < 0.01", w).tolist() == [False, True, True, False]
    assert eval_where("(w > 0) & (w > quantile(w, 0.5))", w).tolist() == [False, False, True, True]
    for bad in ("__import__('os').system('x')", "w.__class__", "open('f')"):
        with pytest.raises(ValueError):
            eval_where(bad, w)


def test_stats_hist_downsample():
    a = np.array([[0.0, 1.0], [2.0, 3.0]])
    s = stats(a)
    assert s["mean"] == 1.5 and s["sparsity"] == 0.25 and s["max"] == 3.0
    h = histogram(a, bins=3)
    assert sum(h["counts"]) == 4
    m, ds = downsample2d(np.ones((200, 50)), 96, 96)
    assert ds and m.shape == (67, 50) and np.allclose(m, 1)


def _block_means(m, fr, fc):
    rows = range(0, m.shape[0], fr)
    cols = range(0, m.shape[1], fc)
    return np.array([[np.nanmean(m[i:i + fr, j:j + fc]) for j in cols] for i in rows])


@pytest.mark.parametrize("shape", [(192, 96), (200, 50), (97, 301), (5, 1000)])
@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64])
def test_downsample2d_matches_block_means(shape, dtype):
    m = (np.random.default_rng(0).normal(size=shape) * 10).astype(dtype)
    out, ds = downsample2d(m, 96, 96)
    fr, fc = -(-shape[0] // 96), -(-shape[1] // 96)
    assert ds and np.allclose(out, _block_means(m.astype(np.float64), fr, fc))


def test_downsample2d_leaves_nan_cells_out():
    m = np.random.default_rng(1).normal(size=(200, 50))
    m[0, 0] = m[150, 7] = np.nan
    out, _ = downsample2d(m, 96, 96)
    assert np.allclose(out, _block_means(m, 3, 1), equal_nan=True)


def test_stats_and_histogram_from_precomputed_finite_values():
    a = np.random.default_rng(2).normal(size=(64, 33))
    a[0, :3] = [np.nan, np.inf, 0.0]
    finite = a[np.isfinite(a)]
    which = ("mean", "std", "min", "max", "norm", "sparsity", "nan", "rms", "abs_mean")
    s, s2 = stats(a, which), stats(a, which, finite=finite)
    assert s == pytest.approx(s2)
    assert s["mean"] == pytest.approx(finite.mean()) and s["norm"] == pytest.approx(np.linalg.norm(finite))
    assert s["sparsity"] == pytest.approx(1 / a.size) and s["nan"] == 2
    assert histogram(a, 40) == histogram(finite, 40, finite=True)


def test_top_k_matches_a_full_sort():
    from deepvis.monitor.lens import top_k

    p = np.random.default_rng(3).random((4, 5000))
    assert np.array_equal(top_k(p, 5), np.argsort(-p, axis=-1)[:, :5])
    assert np.array_equal(top_k(p[:, :3], 5), np.argsort(-p[:, :3], axis=-1))  # k ≥ vocabulary


def test_graph_collapse_and_subgraph():
    g = ModelGraph("m")
    g.add_node(Node("", "M", "M", "container"))
    g.add_node(Node("@in:x", "Input", "Input", "input", parent=""))
    g.add_node(Node("blk", "Block", "Block", "container", parent=""))
    g.add_node(Node("blk.a", "Linear", "Linear", "module", parent="blk"))
    g.add_node(Node("blk.b", "ReLU", "ReLU", "module", parent="blk"))
    g.add_node(Node("@out", "Output", "Output", "output", parent=""))
    g.link_children()
    for s, d in [("@in:x", "blk.a"), ("blk.a", "blk.b"), ("blk.b", "@out")]:
        g.add_edge(s, d)
    c = g.collapse(0)
    assert "blk" in c.nodes and c.nodes["blk"].op == "module" and "blk.a" not in c.nodes
    assert {(e.src, e.dst) for e in c.edges} == {("@in:x", "blk"), ("blk", "@out")}
    sub = g.subgraph("blk")
    assert "@in:blk" in sub.nodes and "@out:blk" in sub.nodes
    assert g.topo_order() == ["@in:x", "blk.a", "blk.b", "@out"]


def test_theme_uses_only_the_manim_palette():
    manim = {v.upper() for k, v in vars(theme).items() if k.isupper() and isinstance(v, str) and v.startswith("#")
             and k.split("_")[0] in ("BLUE", "TEAL", "GREEN", "YELLOW", "GOLD", "RED", "MAROON", "PURPLE", "GREY",
                                     "WHITE", "BLACK", "PINK", "LIGHT", "ORANGE")}
    used = list(theme.FAMILY_COLORS.values()) + theme.DIVERGING_STOPS + theme.SEQUENTIAL_STOPS + [
        theme.BACKGROUND, theme.PANEL, theme.PANEL_2, theme.PANEL_EDGE, theme.GRID, theme.TEXT, theme.TEXT_2,
        theme.MUTED, theme.HIGHLIGHT, theme.POSITIVE, theme.NEGATIVE]
    assert {c.upper() for c in used} <= manim
    lo, mid, hi = theme.colormap([-1, 0, 1], vmin=-1, vmax=1)
    assert mid == theme.GREY_E and lo == theme.RED_A and hi == theme.BLUE_A
