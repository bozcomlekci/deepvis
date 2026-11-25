"""``net.widget(x)``: the live page as a notebook widget (anywidget) — edit weights, intervene, play with the input,
watch training — in JupyterLab, Jupyter Notebook, VS Code notebooks and Colab, without a local web server: the page's
requests travel over the widget's own channel to this kernel.

A training loop that runs in the same cell keeps the kernel busy, so the widget follows once the cell is done (run
the loop in a thread, or use ``net.serve`` in a browser, to watch it live).
"""

from __future__ import annotations

import json

ESM = r"""
function render({ model, el }) {
  const frame = document.createElement("iframe");
  frame.style.width = "100%";
  frame.style.height = model.get("height") + "px";
  frame.style.border = "0";
  frame.style.borderRadius = "8px";
  frame.setAttribute("title", "deepvis");
  frame.srcdoc = model.get("html");
  el.appendChild(frame);
  const onMessage = (e) => {
    if (e.source !== frame.contentWindow) return;
    const d = e.data || {};
    if (d.__dvRequest) {
      model.send({ type: "request", id: d.id, method: d.method, path: d.path, body: d.body });
    } else if (d.__deepvis && d.type === "download") {  // the framed page can't save files itself
      const a = document.createElement("a");
      a.href = d.data; a.download = d.name; document.body.appendChild(a); a.click(); a.remove();
    } else if (d.__deepvis && d.type === "open") {
      window.open(d.url, "_blank", "noopener");
    }
  };
  window.addEventListener("message", onMessage);
  model.on("msg:custom", (msg) => {
    if (msg && msg.type === "reply" && frame.contentWindow) {
      frame.contentWindow.postMessage({ __dvReply: 1, id: msg.id, status: msg.status, body: msg.body }, "*");
    }
  });
  model.on("change:html", () => { frame.srcdoc = model.get("html"); });
  model.on("change:height", () => { frame.style.height = model.get("height") + "px"; });
  return () => window.removeEventListener("message", onMessage);
}
export default { render };
"""


def make_widget(server, height: int = 820):
    """A :class:`~deepvis.viz.server.LiveServer` (not started) behind a notebook widget."""
    try:
        import anywidget
        import traitlets
    except ImportError as exc:
        from ..errors import OptionError

        raise OptionError("net.widget needs anywidget.", "pip install anywidget (or deepvis[widget]); in a browser "
                          "without it: net.serve(x).") from exc
    from .html.dashboard import dumps, render_page

    class DeepvisWidget(anywidget.AnyWidget):
        _esm = ESM
        html = traitlets.Unicode("").tag(sync=True)
        height = traitlets.Int(820).tag(sync=True)

        def __init__(self, srv, height):
            with srv.lock:
                page = render_page(srv.payload(), api="dv-widget:", title=srv.net.name, bridge=True, widget=True)
            super().__init__(html=page, height=int(height))
            self.server = srv
            self.on_msg(self._on_msg)

        def _on_msg(self, _widget, content, _buffers):
            if not isinstance(content, dict) or content.get("type") != "request":
                return
            status, body = self.answer(content.get("method", "GET"), content.get("path", ""), content.get("body"))
            self.send({"type": "reply", "id": content.get("id"), "status": status, "body": body})

        def answer(self, method: str, path: str, body: str | None) -> tuple[int, str]:
            """What the live server would answer this request with (status, JSON text)."""
            srv = self.server
            try:
                if method == "GET":
                    if path == "/api/state":
                        return 200, srv.state_json()
                    if path == "/api/version":
                        return 200, json.dumps({"version": srv.version, "step": srv.training.last_step})
                    if path == "/api/history":
                        with srv.lock:
                            return 200, dumps({"edits": [e.to_dict() for e in srv.net.history.applied]})
                    return 404, json.dumps({"error": f"not found: {path}"})
                code, data = srv.handle(path, json.loads(body or "{}"))
                if code == 200:
                    return 200, '{"payload":' + srv.state_json() + ',"message":' + dumps(data["message"]) + "}"
                return code, dumps(data)
            except Exception as exc:  # noqa: BLE001 - shown on the page, like the server's errors
                return 400, json.dumps({"error": f"{type(exc).__name__}: {exc}"})

        def update(self, step: int | None = None, **metrics) -> None:
            """From a training loop: record progress; the widget picks it up on its next poll."""
            self.server.update(step, **metrics)

    return DeepvisWidget(server, height)
