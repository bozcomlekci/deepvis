"""Live dashboard: the page talks to this Python process, which owns the network.

Every edit made in the browser is an ordinary :class:`~deepvis.manipulate.edits.Edit`
applied with ``net.apply_edit`` — so it is undoable from Python too, and the browser's
"Export edits" produces a JSON log you can replay with ``net.apply_edits(...)``.
"""

from __future__ import annotations

import atexit
import json
import os
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING

import numpy as np

from ..manipulate.edits import Edit
from ..manipulate.interventions import ablate, amplify
from .html.dashboard import dumps, in_notebook, render_page
from .html.payload import PayloadBuilder

if TYPE_CHECKING:  # pragma: no cover
    from ..net import Net

VSCODE_ENV = "DEEPVIS_VSCODE"  # set by the VS Code extension: where to announce a live page so it opens in the editor
RUNNING: list["LiveServer"] = []  # live servers started in this process (deepvis.capture keeps them serving)


def vscode_announce(event: dict) -> dict | None:
    """Tell the VS Code extension about a live page (``{"url", "title"}``) or that it closed. Returns its reply
    (``{"host": …}``: the address the panel will load the page from, when VS Code forwards it), None if it didn't listen."""
    target = os.environ.get(VSCODE_ENV)
    if not target:
        return None
    import urllib.request

    body = json.dumps({**event, "pid": os.getpid()}).encode()
    req = urllib.request.Request(target.rstrip("/") + "/announce", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:  # forwarding the port (Codespaces, tunnels) takes a moment
            if r.status != 200:
                return None
            try:
                reply = json.loads(r.read() or b"{}")
            except ValueError:
                reply = {}
            return reply if isinstance(reply, dict) else {}
    except Exception:  # noqa: BLE001 - VS Code closed or the extension reloaded: fall back to the browser
        return None


class LiveServer:
    def __init__(self, net: "Net", args: tuple | None, port: int = 0, host: str = "127.0.0.1",
                 style: str = "expressive", tensor_shapes: bool = True, min_interval: float = 1.0, training=None,
                 snapshot_every: int | None = None, grads: bool = False, loss_fn=None, tokenizer=None, labels=None,
                 train_step=None, train_steps: int | None = None, train_reset=None, train_delay: float = 0.0,
                 attribution: bool = False, heads: bool | None = None, compare=None, compare_names=None,
                 lens_fn=None, dataset=None, offline: bool | None = None, **ui):
        if args is None:
            from ..errors import InputError

            raise InputError("serve() needs an input to run the model on.",
                             "pass one: net.serve(x), or wrap with an example: dv.wrap(model, sample_input=x).")
        self.net = net
        self.host = host
        self.offline = offline
        self.port = port
        self.builder = PayloadBuilder(net, style=style, tensor_shapes=tensor_shapes)
        self.ui = {k: v for k, v in ui.items() if k in ("views", "start", "title", "note", "depth")}
        self.lock = threading.RLock()  # one owner of the model at a time: training step, page action or trace
        self._interventions: list[dict] = []  # {"id", "node", "desc", "handle"}
        self._next_id = 1
        self.args = tuple(args)
        self.base_args = tuple(args)
        self.baseline = net.trace(*args).output
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        # "watch it learn": the training thread pushes snapshots, the page polls /api/version
        from ..monitor.training import TrainingLog

        self.training = training if training is not None else TrainingLog(net, every_steps=snapshot_every)
        self.min_interval = float(min_interval)
        self.version = 0
        self._cache: tuple[int, dict] | None = None
        self._cache_json: tuple[dict, str] | None = None  # the cached payload, serialized once
        self._last_render = -1e18
        self.sample = int(ui.get("sample") or 0)  # which sample of a batched input the page shows
        self.grads, self.loss_fn = bool(grads), loss_fn  # backpropagate on every refresh (gradient flow)
        self.vocab = {"tokenizer": tokenizer, "labels": labels, "attribution": attribution, "heads": heads,
                      "compare": compare, "compare_names": compare_names, "lens_fn": lens_fn, "dataset": dataset}
        # training driven from the page (Train / Pause / Step / Reset buttons)
        self._train_fn, self._train_reset_fn = train_step, train_reset
        self.train_steps = int(train_steps) if train_steps else None
        self.train_delay = float(train_delay)
        self.train_step_count = 0
        self._train_go = threading.Event()
        self._train_quit = threading.Event()
        self._train_thread: threading.Thread | None = None
        self._train_error: str | None = None
        self._init_params = ({p: net.backend.get_param(p) for p in net.backend.param_paths()}
                             if train_step is not None else None)
        self._init_history = net.history.checkpoint() if train_step is not None else None  # edits made before serve()
        self.allowed_hosts: set[str] = set()  # besides loopback: where VS Code forwards the page to

    # ------------------------------------------------------------ payload
    def _trace_interventions(self) -> dict:
        out: dict = {}
        for i in self._interventions:
            for nid, fns in (i.get("trace_only") or {}).items():
                out.setdefault(nid, []).extend(fns)
        return out

    def payload(self) -> dict:
        """The page state for the current version (cached: the page polls, the training loop pushes)."""
        if self._cache is not None and self._cache[0] == self.version:
            return self._cache[1]
        return self._build()

    def state_json(self) -> str:
        """:meth:`payload` as JSON, serialized once per version (polls and replies share it)."""
        with self.lock:
            pay = self.payload()
            if self._cache_json is None or self._cache_json[0] is not pay:
                self._cache_json = (pay, dumps(pay))
            return self._cache_json[1]

    def _output(self):
        """The network's output on the page input, as the page would show it (eval mode, no autograd)."""
        with self.net.backend.inspecting():
            return self.net.trace(*self.args, interventions=self._trace_interventions()).output

    def _trace(self):
        with self.net.backend.inspecting():
            return self.net.trace(*self.args, interventions=self._trace_interventions(), grads=self.grads,
                                  loss_fn=self.loss_fn)

    def _build(self, trace=None) -> dict:
        tr = trace if trace is not None else self._trace()
        ints = [{"id": i["id"], "node": i["node"], "desc": i["desc"]} for i in self._interventions]
        pay = self.builder.build(self.args, trace=tr, live=True, baseline=self.baseline,
                                 monitor=self.net._monitors[-1] if self.net._monitors else None, interventions=ints,
                                 ui=self.ui, training=self.training, sample=self.sample, **self.vocab)
        pay["meta"]["serverVersion"] = self.version
        pay["meta"]["trainer"] = self.trainer_state()
        self._cache = (self.version, pay)
        return pay

    # ------------------------------------------------------------ training
    def update(self, step: int | None = None, *, params=None, grads=None, render: bool | None = None,
               **metrics) -> bool:
        """Push training progress to the open page. Call it from your training loop, e.g. every step::

            srv = net.serve(x)
            for step in range(n):
                loss = train_step(...)
                srv.update(step, loss=loss, lr=sched.get_last_lr()[0])

        ``metrics`` are logged on every call (cheap). At most every ``min_interval`` seconds (or when
        ``render=True``) the server re-runs the network on the dashboard input and publishes a new page
        state; per-layer weight dynamics are snapshotted then, and every ``snapshot_every`` steps if set. JAX: pass the new weights as ``params=`` (whatever
        ``net.export()`` returns); nnx modules updated in place are re-read automatically. ``grads=`` (a tree
        shaped like the params) adds gradient norms; torch ``.grad`` is picked up automatically.
        Returns whether the page state was refreshed.
        """
        with self.lock:
            if params is not None:
                self.net.load(params)
            elif hasattr(self.net.backend, "sync"):
                self.net.backend.sync()
            else:
                self.net._invalidate()
            now = time.monotonic()
            due = (now - self._last_render) >= self.min_interval if render is None else bool(render)
            step_ = step if step is not None else (self.training.last_step or 0) + 1
            snap = due or bool(self.training.every_steps and self.training.due(step_))
            tr = self._trace() if snap else None
            self.training.record(step, grads=grads, trace=tr, snapshot=snap, **metrics)
            if due:
                self.version += 1
                self._build(tr)
                self._last_render = now
            return due

    # ------------------------------------------------------ training from the page
    def trainer_state(self) -> dict | None:
        if self._train_fn is None:
            return None
        return {"running": self._train_go.is_set(), "step": self.train_step_count, "total": self.train_steps,
                "error": self._train_error}

    def _train_once(self, render: bool | None = None) -> None:
        with self.lock:  # the page's actions and traces wait for the step to finish
            step = self.train_step_count
            metrics = self._train_fn(step)
            if metrics is None:
                metrics = {}
            elif isinstance(metrics, (int, float)) or hasattr(metrics, "item"):
                metrics = {"loss": metrics}  # a bare loss is fine
            elif not isinstance(metrics, dict):
                from ..errors import OptionError

                raise OptionError(f"train_step returned a {type(metrics).__name__}.",
                                  "return a dict of numbers, e.g. {'loss': loss.item(), 'accuracy': acc} (or just the loss).")
            self.train_step_count += 1
            params = metrics.pop("params", None)  # JAX: the step's new weights (whatever net.export() returns)
            grads = metrics.pop("grads", None)
            self.update(step, params=params, grads=grads, render=render, **metrics)

    def _train_loop(self) -> None:
        while not self._train_quit.is_set():
            if not self._train_go.wait(0.2):
                continue
            if self.train_steps is not None and self.train_step_count >= self.train_steps:
                self._train_go.clear()
                with self.lock:
                    self.update(max(self.train_step_count - 1, 0), render=True)  # final state, "done"
                continue
            try:
                done = self.train_steps is not None and self.train_step_count + 1 >= self.train_steps
                self._train_once(render=True if done else None)
            except Exception as exc:  # noqa: BLE001 - show it on the page, stop training
                from ..errors import DeepvisError, explain

                first = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
                hint = explain(exc, generic=False)
                self._train_error = (str(exc) if isinstance(exc, DeepvisError)
                                     else f"{type(exc).__name__} in train_step: {first}" + (f" — {hint}" if hint else ""))
                self._train_go.clear()
                with self.lock:
                    self.version += 1
            time.sleep(max(self.train_delay, 0.0005))  # let page requests in between steps

    def train(self, action: str) -> str:
        """``start`` | ``pause`` | ``step`` | ``reset`` — what the page's buttons call (also usable from Python)."""
        if self._train_fn is None:
            raise ValueError("this server has no train_step; pass net.serve(x, train_step=fn)")
        if action == "start":
            self._train_error = None
            if self.train_steps is not None and self.train_step_count >= self.train_steps:
                return f"already trained {self.train_steps} steps — reset to start over"
            if self._train_thread is None or not self._train_thread.is_alive():
                self._train_thread = threading.Thread(target=self._train_loop, daemon=True, name="deepvis-train")
                self._train_thread.start()
            self._train_go.set()
            return "training"
        if action == "pause":
            self._train_go.clear()
            with self.lock:
                if self.train_step_count:
                    self.update(self.train_step_count - 1, render=True)
            return f"paused at step {self.train_step_count}"
        if action == "step":
            self._train_go.clear()
            self._train_once(render=True)
            return f"step {self.train_step_count}"
        if action == "reset":
            self._train_go.clear()
            with self.lock:
                for p_, v in self._init_params.items():
                    self.net.backend.set_param(p_, v)
                self.net._invalidate()
                self.net.history.rollback(self._init_history)  # edits made mid-training are gone with their weights
                self.training.clear()
                self.train_step_count = 0
                self._train_error = None
                if self._train_reset_fn is not None:
                    self._train_reset_fn()
            return "weights restored to their initial values"
        raise ValueError(f"unknown training action {action!r}")

    # --------------------------------------------------------------- input
    def _set_input(self, body: dict) -> str:
        from ..core.tensor import to_numpy

        op = body.get("op")
        if op == "reset":
            self.args = self.base_args
            return "input restored"
        x0 = self.args[0]
        arr = np.array(to_numpy(x0), copy=True)
        batched = arr.ndim >= 2 and not (self.net.framework == "equinox")
        view = arr[min(self.sample, arr.shape[0] - 1)] if batched else arr
        if op == "set_image":
            img = np.asarray(body["data"], dtype=arr.dtype)
            if view.ndim >= 3:
                ch = view.ndim - 1 if self.net.backend.channels_last else 0
                moved = np.moveaxis(view, ch, 0)
                moved[...] = img.reshape(moved.shape[1:])[None]
            else:
                view[...] = img.reshape(view.shape)
            msg = "input image updated"
        elif op == "set_vector":
            view[...] = np.asarray(body["data"], dtype=arr.dtype).reshape(view.shape)
            msg = "input vector updated"
        elif op == "set_tokens":
            ids = np.asarray(body["data"], dtype=arr.dtype).reshape(-1)
            flat = view.reshape(-1)
            flat[: ids.size] = ids[: flat.size]
            view[...] = flat.reshape(view.shape)
            msg = f"tokens set to {ids.tolist()[:12]}"
        elif op == "noise":
            if arr.dtype.kind not in "fc":
                raise ValueError("noise needs a floating-point input")
            rng = np.random.default_rng(int(body.get("seed", 0)))
            view += rng.normal(0, float(body.get("std", 0.1)), view.shape).astype(arr.dtype)
            msg = f"added noise σ={body.get('std', 0.1)} to the input"
        elif op == "scale":
            scaled = view * float(body.get("factor", 1.0))  # in float: an integer image ×0.5 must not become ×0
            if arr.dtype.kind in "iu":
                lim = np.iinfo(arr.dtype)
                scaled = np.clip(np.rint(scaled), lim.min, lim.max)
            view[...] = scaled.astype(arr.dtype)
            msg = f"scaled the input by {body.get('factor')}"
        else:
            raise ValueError(f"unknown input op {op!r}")
        native = self.net.backend.to_native(arr)
        if hasattr(x0, "dtype") and hasattr(native, "to") and hasattr(x0, "device"):
            native = native.to(dtype=x0.dtype, device=x0.device)
        self.args = (native,) + tuple(self.args[1:])
        return msg

    def page(self) -> str:
        with self.lock:
            return render_page(self.payload(), api="", title=self.net.name, bridge=bool(os.environ.get(VSCODE_ENV)),
                               offline=self.offline)

    # ------------------------------------------------------------ actions
    def handle(self, route: str, body: dict) -> tuple[int, dict]:
        net = self.net
        with self.lock:
            msg = ""
            after = None  # a trace of the new state, when one was needed anyway: the page is built from it
            if route == "/api/edit":
                e = Edit(body["op"], body["target"], index=body.get("index"), args=body.get("args") or {},
                         params=body.get("params"), where=body.get("where"))
                before = self._output()
                net.apply_edit(e)
                after = self._trace()
                msg = f"{e.describe()} · {_delta(before, after.output)}"
            elif route == "/api/neurons":
                view = net[body["node"]]
                sel = view.neurons([int(i) for i in body.get("neurons", [])])
                if not sel.neurons:
                    return 400, {"error": "give neuron indices, e.g. 0, 3, 7"}
                op = body.get("op", "zero")
                sel.edit(op, **(body.get("args") or {}))
                msg = f"{op} neurons {sel.neurons} of {body['node']}"
            elif route == "/api/intervene":
                fn = _make_intervention(body)
                self._add_intervention(body["node"], fn, repr(fn))
                msg = f"{fn!r} on {body['node']}"
            elif route == "/api/head":  # switch one attention head off (undoable)
                from ..interpret.attention import head_ablation

                node, hd = body["node"], int(body["head"])
                kind, what = head_ablation(net, node, hd)
                before = self._output()
                if kind == "edits":
                    for e in what:
                        net.apply_edit(e)
                else:
                    self._add_intervention(node, what, f"head {hd} off")
                after = self._trace()
                msg = f"head {hd} of {node} off · {_delta(before, after.output)}"
            elif route == "/api/remove_intervention":
                keep = []
                for i in self._interventions:
                    if i["id"] == body.get("id"):
                        if i["handle"] is not None:
                            i["handle"].remove()
                    else:
                        keep.append(i)
                self._interventions = keep
                msg = "intervention removed"
            elif route == "/api/clear_interventions":
                for i in self._interventions:
                    if i["handle"] is not None:
                        i["handle"].remove()
                n = len(self._interventions)
                self._interventions = []
                msg = f"removed {n} intervention(s)"
            elif route == "/api/train":
                msg = self.train(body.get("action", ""))
            elif route == "/api/sample":
                self.sample = max(0, int(body.get("index", 0)))
                msg = ""
            elif route == "/api/input":
                msg = self._set_input(body)
            elif route == "/api/undo":
                done = net.undo()
                msg = f"undid {done[0].describe()}" if done else "nothing to undo"
            elif route == "/api/redo":
                done = net.redo()
                msg = f"redid {done[0].describe()}" if done else "nothing to redo"
            elif route == "/api/reset":
                net.reset()
                msg = "restored original weights"
            elif route == "/api/restore":
                vals = net.history.restore_values(body.get("paths", []))
                if vals:
                    net.apply_edit(Edit("load", list(vals), args={"values": vals}, label=f"restore {', '.join(vals)}"))
                msg = f"restored {len(vals)} parameter(s)" if vals else "layer already at original values"
            else:
                return 404, {"error": f"unknown route {route}"}
            self.version += 1
            return 200, {"payload": self._build(after) if after is not None else self.payload(), "message": msg}

    def _add_intervention(self, node: str, fn, desc: str) -> None:
        if not self.net.backend.function_taps and self.net.graph.nodes[node].op == "function":
            hook = {"handle": None, "trace_only": {node: [fn]}}  # torch functional ops: applied during tracing only
        else:
            hook = {"handle": self.net.intervene(node, fn)}
        self._interventions.append({"id": self._next_id, "node": node, "desc": desc, **hook})
        self._next_id += 1

    # ------------------------------------------------------------- server
    def start(self, open_browser: bool = True) -> "LiveServer":
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def _send(self, code: int, body: bytes, ctype: str):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                if len(body) > 65536 and "gzip" in (self.headers.get("Accept-Encoding") or ""):
                    import gzip  # a big model's state is megabytes of JSON: ~4x less over a tunnel / Colab proxy

                    body = gzip.compress(body, compresslevel=1)
                    self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _refuse(self) -> bool:
                """Only the page itself may talk to the server: not other websites (CSRF) nor a page whose domain
                was pointed at 127.0.0.1 (DNS rebinding). Answers 403 and returns True otherwise."""
                why = _forbidden(self.headers, server.host, self.command == "POST", self.path.startswith("/api/"),
                                 server.allowed_hosts)
                if why:
                    self._send(403, why.encode(), "text/plain")
                return bool(why)

            def do_GET(self):
                if self._refuse():
                    return
                if self.path in ("/", "/index.html"):
                    return self._send(200, server.page().encode("utf-8"), "text/html; charset=utf-8")
                if self.path == "/api/history":
                    with server.lock:
                        data = {"edits": [e.to_dict() for e in server.net.history.applied]}
                    return self._send(200, dumps(data).encode(), "application/json")
                if self.path == "/api/state":
                    return self._send(200, server.state_json().encode(), "application/json")
                if self.path == "/api/version":
                    data = {"version": server.version, "step": server.training.last_step}
                    return self._send(200, json.dumps(data).encode(), "application/json")
                return self._send(404, b"not found", "text/plain")

            def do_POST(self):
                if self._refuse():
                    return
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(n) or b"{}")
                    code, data = server.handle(self.path, body)
                    if code == 200:  # the page state as /api/state serves it, serialized once
                        reply = '{"payload":' + server.state_json() + ',"message":' + dumps(data["message"]) + "}"
                    else:
                        reply = dumps(data)
                except Exception as exc:  # noqa: BLE001 - report errors to the page
                    reply = dumps({"error": f"{type(exc).__name__}: {exc}"})
                    code = 400
                self._send(code, reply.encode(), "application/json")

        try:
            self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        except OSError as exc:
            import errno

            from ..errors import DeepvisError

            # Windows: WSAEADDRINUSE, or WSAEACCES when another program holds the port exclusively
            if exc.errno in (errno.EADDRINUSE, 48, 98) or getattr(exc, "winerror", None) in (10013, 10048):
                class PortInUse(DeepvisError, OSError):
                    pass

                raise PortInUse(f"port {self.port} is already in use (another dashboard or program).",
                                "pass another one, e.g. net.serve(x, port=8765), or port=0 to pick a free port.") from exc
            raise
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True, name="deepvis-live")
        self._thread.start()
        RUNNING.append(self)
        title = self.ui.get("title") or self.net.name
        reply = vscode_announce({"url": self.url, "title": title})  # VS Code shows it beside the code
        in_editor = reply is not None
        if in_editor and isinstance(reply.get("host"), str):
            self.allowed_hosts.add(reply["host"].strip().lower())
        if in_editor:
            atexit.register(lambda: self._httpd is not None and vscode_announce({"url": self.url, "closed": True}))
        elif open_browser and not in_notebook():
            webbrowser.open(self.url)
        return self

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def stop(self) -> None:
        self._train_go.clear()
        self._train_quit.set()
        if self._train_thread is not None:
            self._train_thread.join(timeout=5)
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
            vscode_announce({"url": self.url, "closed": True})
        if self in RUNNING:
            RUNNING.remove(self)

    def _repr_html_(self) -> str:
        return (f'<iframe src="{self.url}" style="width:100%;height:860px;border:1px solid #26262D;'
                f'border-radius:10px;background:#0E0E11"></iframe>')

    def __repr__(self) -> str:
        state = "running" if self._httpd else "stopped"
        return f"LiveServer({self.url}, {state})"


LOOPBACK = ("127.0.0.1", "localhost", "::1", "[::1]")


def _forbidden(headers, bound_host: str, post: bool, api: bool, allowed_hosts=()) -> str | None:
    """Why a request must be refused, or None. ``headers``: the request's; ``bound_host``: the address we listen on;
    ``api``: an /api/ call (the page itself may be framed cross-site: a notebook, the VS Code panel);
    ``allowed_hosts``: forwarded addresses VS Code reported for this page."""
    # loopback only, unless bound to 0.0.0.0 on purpose — or in a Codespace, whose forwarded ports (another host
    # name) are behind GitHub sign-in already
    if bound_host in LOOPBACK and os.environ.get("CODESPACES") != "true":
        host = (headers.get("Host") or "").strip().lower()
        name = host.rsplit(":", 1)[0] if not host.endswith("]") else host
        if name not in LOOPBACK and host not in allowed_hosts and name not in allowed_hosts:
            return f"forbidden: unexpected Host {host!r}"
    if api and headers.get("Sec-Fetch-Site", "same-origin") not in ("same-origin", "none"):
        return "forbidden: cross-site request"
    if post and (headers.get("Content-Type") or "").split(";")[0].strip().lower() != "application/json":
        return "forbidden: send application/json"  # forms and no-cors fetches can't set it without a CORS preflight
    return None


def _make_intervention(body: dict):
    from ..manipulate.interventions import add_noise, clamp, mean_ablate

    typ = body.get("type", "ablate")
    neurons = body.get("neurons") or None
    if typ == "ablate":
        return ablate(neurons)
    if typ == "amplify":
        return amplify(float(body.get("factor", 2.0)), neurons)
    if typ == "clamp":
        return clamp(body.get("min"), body.get("max"))
    if typ == "noise":
        return add_noise(float(body.get("std", 0.1)), int(body.get("seed", 0)))
    if typ == "mean_ablate":
        return mean_ablate(neurons)
    raise ValueError(f"unknown intervention {typ!r}")


def _delta(a, b) -> str:
    from ..core.tensor import first_array, to_numpy

    x, y = to_numpy(first_array(a)), to_numpy(first_array(b))
    if x.shape != y.shape:
        return "output shape changed"
    return f"Δoutput ‖·‖ = {float(np.linalg.norm(y.astype(np.float64) - x.astype(np.float64))):.4g}"

