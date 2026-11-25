/* Inside a notebook widget the page has no server to fetch from: its requests to "dv-widget:/api/…" go to the
   widget around it (postMessage), which hands them to the Python kernel and posts the reply back. */
(() => {
  if (window.parent === window) return;
  const pending = new Map();
  let next = 1;
  window.addEventListener("message", (e) => {
    const d = e.data;
    if (!d || !d.__dvReply || e.source !== window.parent) return;
    const done = pending.get(d.id);
    if (!done) return;
    pending.delete(d.id);
    done(new Response(d.body, { status: d.status, headers: { "Content-Type": "application/json" } }));
  });
  const plain = window.fetch.bind(window);
  window.fetch = (url, opts) => {
    if (typeof url !== "string" || !url.startsWith("dv-widget:")) return plain(url, opts);
    const o = opts || {};
    return new Promise((resolve, reject) => {
      const id = next++;
      pending.set(id, resolve);
      setTimeout(() => { if (pending.delete(id)) reject(new Error("the Python kernel didn't answer (busy?)")); }, 120000);
      window.parent.postMessage({ __dvRequest: 1, id, method: o.method || "GET", path: url.slice("dv-widget:".length),
        body: typeof o.body === "string" ? o.body : null }, "*");
    });
  };
})();
