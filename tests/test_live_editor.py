"""Live pages in editors: dv.watch, announcing to VS Code, live runs under capture, kernel helpers, replay frames."""

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
pytestmark = pytest.mark.torch

import deepvis as dv  # noqa: E402
from deepvis import capture  # noqa: E402


def mlp():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 3))


@pytest.fixture
def no_browser(monkeypatch):
    import webbrowser

    opened = []
    monkeypatch.setattr(webbrowser, "open", lambda url, *a, **k: opened.append(url))
    return opened


def test_watch_logs_and_keeps_replay_frames(no_browser):
    model, x = mlp(), torch.randn(8, 4)
    y = torch.randint(0, 3, (8,))
    opt = torch.optim.SGD(model.parameters(), 0.5)
    with dv.watch(model, x, every=2, open_browser=False, min_interval=0.0) as w:
        for step in range(12):
            loss = nn.functional.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            w.step(step, loss=loss)
    state = json.loads(urllib.request.urlopen(w.url + "api/state").read())
    w.close()
    T = state["training"]
    assert T["count"] >= 12 and "loss" in T["metrics"]
    frames = T["frames"]
    assert len(frames) >= 5 and frames[0]["step"] == 0 and frames[-1]["step"] >= 10
    assert set(frames[-1]["weights"]) == {"0", "2"} and frames[-1]["output"]["values"]
    assert frames[0]["weights"]["0"]["data"] != frames[-1]["weights"]["0"]["data"]  # the weights moved
    assert not no_browser


def test_frames_are_bounded():
    net = dv.wrap(mlp(), sample_input=torch.randn(2, 4))
    log = dv.TrainingLog(net, every_steps=1, frames=6)
    for s in range(40):
        log.record(s, loss=1.0 / (s + 1))
    f = log.to_dict()["frames"]
    assert len(f) == 6 and f[0]["step"] == 0 and f[-1]["step"] >= 36
    assert dv.TrainingLog(net, frames=0).to_dict()["frames"] == []


class _Listener:
    def __init__(self):
        self.events = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                outer.events.append((self.path, json.loads(self.rfile.read(n))))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/tok"

    def close(self):
        self.httpd.shutdown()


def test_live_page_announces_to_vscode_instead_of_a_browser(monkeypatch, no_browser):
    lis = _Listener()
    monkeypatch.setenv("DEEPVIS_VSCODE", lis.url)
    try:
        srv = dv.visualize(mlp(), torch.randn(2, 4), live=True, title="announced")
        assert not no_browser
        path, ev = lis.events[0]
        assert path == "/tok/announce" and ev["url"] == srv.url and ev["title"] == "announced"
        assert ev["pid"] == os.getpid()
        page = urllib.request.urlopen(srv.url).read().decode()
        assert "__deepvis" in page  # the download / link bridge for the framed page
        srv.stop()
        assert lis.events[-1][1] == {"url": srv.url, "closed": True, "pid": os.getpid()}
    finally:
        lis.close()


def test_no_listener_falls_back_to_the_browser(monkeypatch, no_browser):
    monkeypatch.setenv("DEEPVIS_VSCODE", "http://127.0.0.1:9/nothing")
    srv = dv.visualize(mlp(), torch.randn(2, 4), live=True)
    srv.stop()
    assert no_browser == [srv.url]
    assert "__deepvis" not in dv.viz.html.dashboard.render_page(srv.payload(), api="")


def test_live_only_options_are_checked():
    with pytest.raises(dv.errors.OptionError, match="snapshot_every"):
        dv.visualize(mlp(), torch.randn(2, 4), path="x.html", snapshot_every=5)  # only with live=True


def test_capture_keeps_a_live_run_serving(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "import torch, deepvis as dv\n"
        "model = torch.nn.Linear(4, 2)\n"
        "srv = dv.visualize(model, torch.randn(3, 4), live=True, title='live run')\n"
        "for s in range(3):\n"
        "    srv.update(s, loss=1.0 / (s + 1))\n")
    out = tmp_path / "page.html"
    proc = subprocess.Popen([sys.executable, "-m", "deepvis.capture", str(out), str(script)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            env={**os.environ, "DEEPVIS_VSCODE": "", "PYTHONPATH": os.pathsep.join(sys.path)})
    try:
        res = tmp_path / "page.html.json"
        for _ in range(300):
            if res.exists() and res.stat().st_size:
                break
            time.sleep(0.1)
        r = json.loads(res.read_text(encoding="utf-8"))
        assert r["ok"] and r["live"].startswith("http://127.0.0.1:") and r["page"] is None
        time.sleep(0.3)
        assert proc.poll() is None  # still serving the last state
        st = json.loads(urllib.request.urlopen(r["live"] + "api/version").read())
        assert st["step"] == 2
    finally:
        proc.kill()
        proc.wait(10)


def test_kernel_helpers(tmp_path):
    ns = {"net": mlp(), "x": torch.randn(3, 4), "pair": (torch.randn(2), torch.randn(3)), "dv": dv, "torch": torch,
          "n": 3}
    objs = {o["name"]: o for o in capture.kernel_objects(ns)}
    assert objs["net"]["kind"] == "model" and objs["net"]["params"] == 4 * 16 + 16 + 16 * 3 + 3
    assert objs["x"]["shape"] == [3, 4] and objs["pair"]["type"] == "tuple" and "n" not in objs
    r = capture.visualize_to(str(tmp_path / "a.html"), "net", "x", title="kernel", env=ns)
    assert r["ok"] and "kernel · deepvis" in open(r["page"], encoding="utf-8").read()
    assert json.loads((tmp_path / "a.html.json").read_text(encoding="utf-8"))["ok"]
    r = capture.kernel_call(str(tmp_path / "b.html"), 'dv.visualize(net,\n    x, title="cell")', env=ns)
    assert r["ok"] and "DEEPVIS_CAPTURE" not in os.environ
    r = capture.kernel_call(str(tmp_path / "c.html"), "dv.visualize(net, y)", env=ns)
    assert not r["ok"] and "run the cells above" in r["message"]
    r = capture.visualize_to(str(tmp_path / "d.html"), "net", "torch.randn(3, 5)", env=ns)
    assert not r["ok"] and "expects 4 features" in r["message"]


def _extension_code(*args):
    root = os.path.join(os.path.dirname(__file__), "..", "vscode")
    if not os.path.isdir(root):
        pytest.skip("vscode extension folder not present")
    try:
        out = subprocess.run(["node", os.path.join(root, "test", "kernel_code.js"), *args], capture_output=True,
                             text=True, check=True, timeout=60).stdout
    except FileNotFoundError:
        pytest.skip("node not installed")
    return json.loads(out)


def test_extension_kernel_code_in_a_real_kernel(tmp_path):
    """The exact code the VS Code extension sends to a notebook kernel works in an IPython kernel, uses the live
    objects (trained weights, nothing re-run) and leaves no names behind."""
    jc = pytest.importorskip("jupyter_client")
    pytest.importorskip("ipykernel")
    km = jc.KernelManager(kernel_name="python3")
    km.start_kernel(env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})
    kc = km.client()
    kc.start_channels()
    try:
        kc.wait_for_ready(timeout=120)

        def run(code):
            out = []
            kc.execute_interactive(code, timeout=180, output_hook=lambda m: out.append(
                m["content"].get("text", "") if m["msg_type"] == "stream" else
                ("ERROR " + m["content"]["ename"] + ": " + m["content"]["evalue"]) if m["msg_type"] == "error" else ""))
            return "".join(out)

        assert "ERROR" not in run(
            "import torch, deepvis as dv\n"
            "torch.manual_seed(0)\n"
            "net = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))\n"
            "x = torch.randn(5, 4)\n"
            "with torch.no_grad():\n"
            "    net[0].weight.fill_(0.125)   # 'trained': a state no re-run would reproduce\n")

        def call(*args):
            spec = _extension_code(*args)
            text = run(spec["code"])
            i = text.rfind(spec["mark"])
            assert i >= 0, text
            return json.loads(text[i + len(spec["mark"]):].splitlines()[0])

        objs = {o["name"]: o for o in call("objects")}
        assert objs["net"]["kind"] == "model" and objs["x"]["shape"] == [5, 4]
        out = str(tmp_path / "k.html")
        r = call(json.dumps({"kind": "model", "model": "net", "inputs": "x"}), out)
        assert r["ok"], r
        page = open(out, encoding="utf-8").read()
        assert "0.125" in page  # the live weights
        r = call(json.dumps({"kind": "call", "index": 0}), str(tmp_path / "c.html"), 'dv.visualize(net, x, title="from cell")')
        assert r["ok"] and "from cell · deepvis" in open(r["page"], encoding="utf-8").read()
        r = call(json.dumps({"kind": "model", "model": "nett", "inputs": "x"}), str(tmp_path / "e.html"))
        assert not r["ok"] and "run the\ncell" not in r["message"] and "nett" in r["message"]
        assert run("print(sorted(k for k in dir() if 'deepvis' in k))").strip() == "[]"
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)
