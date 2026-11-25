"""Offline pages: the bundled KaTeX is inlined (no CDN), its fonts as data: URIs; the default follows $DEEPVIS_OFFLINE."""

import pytest

import deepvis as dv
from deepvis.viz.html import dashboard as D


def _page(net, x, tmp_path, **kw):
    return open(net.export_html(str(tmp_path / "p.html"), x, **kw), encoding="utf-8").read()


def test_offline_page_has_no_cdn(tmp_path, mlp, monkeypatch):
    monkeypatch.delenv(D.OFFLINE_ENV, raising=False)
    model, x = mlp
    net = dv.wrap(model, sample_input=x)
    online = _page(net, x, tmp_path)
    assert D.KATEX_CDN in online and "__KATEX__" not in online
    offline = _page(net, x, tmp_path, offline=True)
    assert "cdn.jsdelivr" not in offline and "https://" not in offline.split('id="dv-payload"')[0]
    assert offline.count("data:font/woff2;base64,") == 20 and "url(fonts/" not in offline
    assert "katex.render" in offline or "renderToString" in offline
    assert len(offline) - len(online) < 900_000


@pytest.mark.parametrize("value, inlined", [("1", True), ("true", True), ("0", False), ("", False)])
def test_offline_env(tmp_path, mlp, monkeypatch, value, inlined):
    monkeypatch.setenv(D.OFFLINE_ENV, value)
    model, x = mlp
    page = _page(dv.wrap(model, sample_input=x), x, tmp_path)
    assert (D.KATEX_CDN not in page) == inlined


def test_offline_through_visualize_and_equation_html(tmp_path, mlp, monkeypatch):
    monkeypatch.delenv(D.OFFLINE_ENV, raising=False)
    model, x = mlp
    net = dv.visualize(model, x, path=str(tmp_path / "v.html"), offline=True)
    assert "cdn.jsdelivr" not in open(net.page, encoding="utf-8").read()
    assert "cdn.jsdelivr" not in D.equation_html(net.equation(), offline=True)
    assert D.KATEX_CDN in D.equation_html(net.equation())
