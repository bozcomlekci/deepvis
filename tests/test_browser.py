import pytest

import deepvis as dv

pw = pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.browser


@pytest.mark.parametrize("fixture", ["mlp", "transformer", "cnn"])
def test_dashboard_runs_without_errors(request, tmp_path, fixture):
    model, x = request.getfixturevalue(fixture)
    net = dv.wrap(model, sample_input=x)
    path = net.save_html(str(tmp_path / "d.html"), x)
    errors = []
    with pw.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no browser: {exc}")
        pg = b.new_page(viewport={"width": 1400, "height": 900})
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.on("console", lambda m: errors.append(m.text) if m.type == "error" and "katex" not in m.text else None)
        pg.goto("file://" + path)
        pg.wait_for_timeout(600)
        pg.locator("g.node").nth(1).click()
        for tab in ("Neurons", "Equations", "Monitor", "Graph"):
            pg.get_by_role("tab", name=tab).click()
            pg.wait_for_timeout(200)
        assert pg.locator(".p-head").count() >= 1
        b.close()
    assert errors == []


def test_volumes_tab(request, tmp_path, cnn):
    model, x = cnn
    net = dv.wrap(model, sample_input=x)
    path = net.save_html(str(tmp_path / "v.html"), x)
    errors = []
    with pw.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no browser: {exc}")
        pg = b.new_page(viewport={"width": 1400, "height": 900})
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto("file://" + path)
        pg.wait_for_timeout(500)
        pg.get_by_role("tab", name="Volumes").click()
        pg.wait_for_timeout(500)
        cv = pg.locator("canvas.vol-stage")
        box = cv.bounding_box()
        pg.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        pg.mouse.down(); pg.mouse.move(box["x"] + box["width"] / 2 + 80, box["y"] + box["height"] / 2 + 30); pg.mouse.up()
        pg.get_by_role("button", name="▶ Forward pass").click()
        pg.wait_for_timeout(300)
        # the canvas has painted pixels
        filled = pg.evaluate("() => { const c = document.querySelector('canvas.vol-stage'); const d = c.getContext('2d')"
                             ".getImageData(0, 0, c.width, c.height).data; let n = 0; for (let i = 3; i < d.length; i += 4)"
                             " if (d[i] > 0) n++; return n; }")
        assert filled > 1000
        b.close()
    assert errors == []


def test_expressive_graph_and_math(tmp_path, transformer):
    model, ids = transformer
    net = dv.wrap(model, sample_input=ids)
    path = net.export_html(str(tmp_path / "t.html"), ids)
    errors = []
    with pw.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no browser: {exc}")
        pg = b.new_page(viewport={"width": 1400, "height": 900})
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto("file://" + path)
        pg.wait_for_timeout(800)
        from deepvis.viz.html.layout import _is_noop  # Dropout(p=0) / Identity are not drawn

        assert pg.locator("g.node.x").count() == len([n for n in net.graph.compute_nodes() if not _is_noop(n)])
        assert pg.locator("g.tensor").count() > 5
        pg.locator("g.node.x").nth(3).dispatch_event("click")
        pg.get_by_role("tab", name="Equations").click()
        pg.wait_for_timeout(400)
        if pg.evaluate("() => !!window.katex"):  # KaTeX comes from a CDN; only check when it loaded
            assert pg.locator(".katex-error").count() == 0
        b.close()
    assert errors == []


def test_live_page_follows_training(mlp):
    import torch

    model, x = mlp
    net = dv.wrap(model, sample_input=x)
    srv = net.serve(x, open_browser=False, min_interval=0.0, start="monitor")
    errors = []
    try:
        with pw.sync_playwright() as p:
            try:
                b = p.chromium.launch()
            except Exception as exc:  # noqa: BLE001
                pytest.skip(f"no browser: {exc}")
            pg = b.new_page(viewport={"width": 1400, "height": 900})
            pg.on("pageerror", lambda e: errors.append(str(e)))
            pg.goto(srv.url)
            pg.wait_for_selector("#stage")
            assert pg.locator(".train-badge").count() == 0  # nothing logged yet
            opt = torch.optim.SGD(model.parameters(), lr=0.05)
            for step in range(4):
                opt.zero_grad()
                loss = model(x).pow(2).mean()
                loss.backward()
                opt.step()
                srv.update(step, loss=loss.item())
            pg.wait_for_function("document.querySelector('.train-badge b') && "
                                 "document.querySelector('.train-badge b').textContent === 'step 3'", timeout=5000)
            assert pg.locator(".tcard").count() == 1 and pg.locator("canvas.strip").count() == 1
            pg.locator("canvas.strip").click(position={"x": 5, "y": 2})
            assert pg.locator(".p-sec h4", has_text="Training").count() == 1  # the layer's own dynamics
            pg.get_by_role("button", name="❚❚ Pause updates").click()
            srv.update(4, loss=0.0)
            pg.wait_for_timeout(1500)
            assert pg.locator(".train-badge b").inner_text() == "step 3", "paused pages don't follow"
            b.close()
    finally:
        srv.stop()
    assert errors == []


def test_download_current_view_as_svg(tmp_path, cnn):
    import xml.etree.ElementTree as ET

    model, x = cnn
    net = dv.wrap(model, sample_input=x)
    path = net.save_html(str(tmp_path / "c.html"), x)
    with pw.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no browser: {exc}")
        pg = b.new_page(viewport={"width": 1400, "height": 900}, accept_downloads=True)
        pg.goto("file://" + path)
        pg.wait_for_selector("g.node")
        pg.locator("g.node").nth(2).dispatch_event("click")  # a selection must not leak into the figure
        with pg.expect_download() as dl:
            pg.get_by_role("button", name="⤓ SVG").click()
        out = tmp_path / "view.svg"
        dl.value.save_as(out)
        b.close()
    text = out.read_text(encoding="utf-8")
    root = ET.fromstring(text)  # strict XML
    assert root.tag.endswith("svg") and root.get("viewBox")
    assert "var(--" not in text and "selected" not in text and "dimmed" not in text
    assert text.count("<g") > 10 and "<style>" in text


def test_train_button(mlp):
    import torch

    model, x = mlp
    opt = torch.optim.SGD(model.parameters(), lr=0.05)

    def step(i):
        loss = model(x).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        return {"loss": loss.item()}

    net = dv.wrap(model, sample_input=x)
    srv = net.serve(x, open_browser=False, train_step=step, train_steps=40, train_delay=0.01, min_interval=0.1)
    errors = []
    try:
        with pw.sync_playwright() as p:
            try:
                b = p.chromium.launch()
            except Exception as exc:  # noqa: BLE001
                pytest.skip(f"no browser: {exc}")
            pg = b.new_page(viewport={"width": 1400, "height": 900})
            pg.on("pageerror", lambda e: errors.append(str(e)))
            pg.goto(srv.url)
            pg.wait_for_selector("g.node")
            pg.get_by_role("button", name="▶ Train").click()
            pg.wait_for_function("() => document.querySelector('#dv-actions .btn.primary').textContent.includes('Trained')",
                                 timeout=30000)
            assert pg.locator(".train-prog b").inner_text() == "40 / 40" and pg.locator(".train-badge").count() == 1
            pg.get_by_role("button", name="↺ Reset training").click()
            pg.wait_for_function("() => document.querySelector('.train-prog b').textContent === '0 / 40'")
            pg.get_by_role("button", name="Step").click()
            pg.wait_for_function("() => document.querySelector('.train-prog b').textContent === '1 / 40'")
            b.close()
    finally:
        srv.stop()
    assert errors == []


def test_attention_and_diffusion_tabs(tmp_path):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from deepvis import diffusion as dd

    class Den(nn.Module):
        def __init__(self):
            super().__init__()
            self.t = nn.Embedding(20, 4)
            self.c = nn.Conv2d(1, 4, 3, padding=1)
            self.attn = nn.MultiheadAttention(4, 2, batch_first=True)
            self.o = nn.Conv2d(4, 1, 3, padding=1)

        def forward(self, x, t):
            h = F.silu(self.c(x) + self.t(t)[:, :, None, None])
            s = h.flatten(2).transpose(1, 2)
            a = self.attn(s, s, s)[0].transpose(1, 2).reshape(h.shape)
            return self.o(h + a)

    torch.manual_seed(0)
    m = Den().eval()
    run = dd.sample(m, (2, 1, 6, 6), dd.linear_betas(20), frames=8)
    x, t = run.probe(k=4)
    path = dv.wrap(m, sample_input=(x, t)).export_html(str(tmp_path / "d.html"), x, t, diffusion=run)
    errors = []
    with pw.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no browser: {exc}")
        pg = b.new_page(viewport={"width": 1400, "height": 900})
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto("file://" + path)
        pg.get_by_role("tab", name="Diffusion").click()
        pg.wait_for_selector(".df-grid img")
        pg.locator(".df-slider").fill("7")
        assert pg.locator(".df-t").inner_text() == "t = 0"
        pg.get_by_role("tab", name="Attention").click()
        pg.wait_for_selector(".att-svg")
        hit = pg.locator(".att-svg rect[style*=crosshair]").nth(1)
        box = hit.bounding_box()
        pg.mouse.move(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5)
        assert "attention" in pg.locator(".tip").inner_text()
        assert pg.locator(".att-svg text", has_text="where it looks").count() == 1  # 6×6 pixels = tokens
        b.close()
    assert errors == []


def test_guidance_help_tour_keys_and_links(tmp_path, cnn):
    """The help overlay, the on-demand tour, keyboard navigation, the shareable address, equation ↔ layer links."""
    model, x = cnn
    path = dv.wrap(model, sample_input=x).save_html(str(tmp_path / "g.html"), x)
    errors = []
    with pw.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no browser: {exc}")
        ctx = b.new_context(viewport={"width": 1400, "height": 900})
        pg = ctx.new_page()
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.on("console", lambda m: errors.append(m.text) if m.type == "error" and "katex" not in m.text else None)
        pg.goto("file://" + path)
        pg.wait_for_timeout(600)
        assert pg.locator(".tour-card").count() == 0  # the tour never starts by itself
        assert "Click a layer" in pg.locator(".caption").inner_text()
        pg.keyboard.press("ArrowDown")  # walks the layers; the address follows
        pg.wait_for_timeout(200)
        assert pg.evaluate("location.hash").startswith("#node=")
        assert pg.locator("#panel details.explain .ex-what").inner_text()
        term = pg.locator("#panel .katex [data-n]")
        if term.count():  # a layer with an equation: hovering a symbol lights its layer up
            term.first.hover()
            pg.wait_for_timeout(150)
            assert pg.locator(".node.linked").count() == 1
            pg.mouse.move(5, 400)
        pg.keyboard.press("?")
        pg.wait_for_timeout(200)
        assert pg.locator(".help-card").count() == 1
        pg.locator(".help-card .btn.primary").click()  # ▶ Take the tour
        pg.wait_for_timeout(500)
        seen = 0
        while pg.locator(".tour-card").count() and seen < 12:
            seen += 1
            pg.locator(".tour-card .btn.primary").click()
            pg.wait_for_timeout(450)
        assert 3 <= seen < 12 and pg.locator(".tour-ring").count() == 0
        pg.keyboard.press("2")
        pg.wait_for_timeout(300)
        tab = pg.evaluate("document.querySelector('[role=tab][aria-selected=true]').textContent")
        h = pg.evaluate("location.hash")
        assert tab != "Graph" and "tab=" in h
        pg.locator(".caption .cap-x").click()  # dismissed hints stay dismissed
        pg2 = ctx.new_page()
        pg2.goto("file://" + path + h)
        pg2.wait_for_timeout(500)
        assert pg2.evaluate("document.querySelector('[role=tab][aria-selected=true]').textContent") == tab
        assert pg2.locator(".caption").count() == 0
        b.close()
    assert errors == []


def test_motion_overview_and_empty_monitor(tmp_path, mlp):
    """Forward pass flashes (none under reduced motion), the overview's one-sentence summary, the help dialog's
    focus handling and the Monitor tab's guidance when nothing was recorded."""
    model, x = mlp
    path = dv.wrap(model, sample_input=x).save_html(str(tmp_path / "m.html"), x)
    bare = dv.wrap(model).save_html(str(tmp_path / "bare.html"))
    errors = []
    with pw.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no browser: {exc}")
        pg = b.new_page(viewport={"width": 1400, "height": 900})
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto("file://" + path)
        pg.wait_for_timeout(600)
        assert "network" in pg.locator(".p-note.in-short").inner_text()
        pg.keyboard.press("f")
        pg.wait_for_timeout(200)
        assert pg.locator(".edge-flash").count() > 0
        pg.wait_for_timeout(3500)
        assert pg.locator(".edge-flash").count() == 0  # the flashes clean up after themselves
        pg.locator("#dv-help").focus()
        pg.keyboard.press("Enter")
        pg.wait_for_timeout(150)
        for _ in range(5):
            pg.keyboard.press("Tab")
            assert pg.evaluate("!!document.activeElement.closest('.help-card')")
        pg.keyboard.press("Escape")
        assert pg.evaluate("document.activeElement.id") == "dv-help"
        quiet = b.new_context(viewport={"width": 1400, "height": 900}, reduced_motion="reduce").new_page()
        quiet.goto("file://" + path)
        quiet.wait_for_timeout(500)
        quiet.keyboard.press("f")
        quiet.wait_for_timeout(200)
        assert quiet.locator(".edge-flash").count() == 0
        pb = b.new_page(viewport={"width": 1400, "height": 900})
        pb.goto("file://" + bare)
        pb.wait_for_timeout(500)
        pb.get_by_role("tab", name="Monitor").click()
        pb.wait_for_timeout(200)
        assert "net.serve(x, train_step=fn)" in pb.locator(".empty-help").inner_text()
        b.close()
    assert errors == []


def test_dataset_tab(tmp_path, cnn):
    torch = pytest.importorskip("torch")
    model, x = cnn
    net = dv.wrap(model, sample_input=x)
    data = torch.randn(48, *x.shape[1:])
    prof = net.profile(data, labels=torch.arange(48) % 4, batch_size=16)
    path = net.export_html(str(tmp_path / "ds.html"), x, dataset=prof, start="dataset")
    errors = []
    with pw.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no browser: {exc}")
        pg = b.new_page(viewport={"width": 1400, "height": 1000})
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.on("console", lambda m: errors.append(m.text) if m.type == "error" and "katex" not in m.text else None)
        pg.goto("file://" + path)
        pg.wait_for_timeout(600)
        assert pg.locator(".ds-dot").count() == 48 and pg.locator(".ds-cell").count() == len(prof.layers) ** 2
        pg.locator(".ds-dot").first.hover(force=True)
        assert pg.locator(".tip .ds-thumb").count() == 1  # the sample itself, as a picture
        first = prof.layers[0]
        pg.locator(".ds-cell").first.click(force=True)
        pg.wait_for_timeout(200)
        assert pg.locator(".ds-select").input_value() == first
        pg.get_by_role("tab", name="Graph").click()
        pg.wait_for_timeout(300)
        pg.locator(f"g.node[aria-label$=' {prof.layers[-1]}']").first.click()
        pg.wait_for_timeout(300)
        assert pg.locator("#panel summary", has_text="Inputs that excite").count() == 1
        b.close()
    assert errors == []


def test_notebook_widget_talks_to_python_over_its_own_channel(tmp_path, mlp):
    """The widget's front end (its real render code) with a stand-in model whose send() reaches Python: the framed
    page's requests are answered by the kernel side, and an edit made from the page lands in the model."""
    import json

    pytest.importorskip("anywidget")
    torch = pytest.importorskip("torch")
    from deepvis.viz.widget import ESM

    model, x = mlp
    w = dv.wrap(model, sample_input=x).widget(x, height=600)
    first = next(n.id for n in w.server.net.graph.param_nodes())
    host = tmp_path / "host.html"
    script = ESM.replace("export default { render };", "window.__widget = { render };")
    state = json.dumps({"html": w.html, "height": 600}).replace("</", "<\\/")  # inside a <script>
    host.write_text(f"""<!doctype html><html><body><div id="el"></div><script>{script}
const state = {state};
const listeners = {{}};
const model = {{ get: (k) => state[k], on: (ev, f) => {{ (listeners[ev] = listeners[ev] || []).push(f); }},
  send: async (msg) => {{ const reply = JSON.parse(await window.pySend(JSON.stringify(msg)));
                          (listeners["msg:custom"] || []).forEach((f) => f(reply)); }} }};
window.__widget.render({{ model, el: document.getElementById("el") }});
</script></body></html>""", encoding="utf-8")
    seen = []

    def py_send(raw):
        msg = json.loads(raw)
        seen.append(msg["path"])
        status, body = w.answer(msg["method"], msg["path"], msg["body"])
        return json.dumps({"type": "reply", "id": msg["id"], "status": status, "body": body})

    errors = []
    with pw.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no browser: {exc}")
        pg = b.new_page(viewport={"width": 1300, "height": 800})
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.expose_function("pySend", py_send)
        pg.goto("file://" + str(host))
        pg.wait_for_timeout(2500)  # the page polls /api/version through the widget
        frame = pg.frame_locator("iframe")
        assert frame.locator("g.node").count() > 2
        reply = pg.frames[1].evaluate(f"""async () => {{
            const r = await fetch("dv-widget:/api/edit", {{ method: "POST", headers: {{ "Content-Type": "application/json" }},
              body: JSON.stringify({{ op: "zero", target: "{first}" }}) }});
            return [r.status, (await r.json()).message]; }}""")
        b.close()
    assert "/api/version" in seen and "/api/edit" in seen
    assert reply[0] == 200 and "zero" in reply[1]
    assert torch.count_nonzero(dict(model.named_parameters())[f"{first}.weight"]) == 0  # the edit reached the model
    assert errors == []


def test_offline_page_renders_math_with_no_network(tmp_path, mlp):
    model, x = mlp
    path = dv.wrap(model, sample_input=x).export_html(str(tmp_path / "o.html"), x, offline=True, start="equations")
    requests, errors = [], []
    with pw.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no browser: {exc}")
        pg = b.new_page(viewport={"width": 1400, "height": 900})
        pg.route(lambda url: not url.startswith("file:"), lambda r: (requests.append(r.request.url), r.abort()))
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto("file://" + path)
        pg.wait_for_timeout(800)
        assert pg.evaluate("() => !!window.katex")
        assert pg.locator(".katex").count() > 0 and pg.locator(".katex-error").count() == 0
        # the inlined fonts loaded (data: URIs): KaTeX_Main is available to the page
        assert pg.evaluate("() => document.fonts.check('16px KaTeX_Main')")
        b.close()
    assert requests == [] and errors == []
