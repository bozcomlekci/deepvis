/* Inside an editor panel (VS Code) the page is framed and can't download files or open links itself: hand those
   to the host window, which saves / opens them. Only included when the page is shown there. */
(() => {
  if (window.parent === window) return;
  const up = (m) => window.parent.postMessage(Object.assign({ __deepvis: 1 }, m), "*");
  const click = HTMLAnchorElement.prototype.click;
  HTMLAnchorElement.prototype.click = function () {
    if (!this.hasAttribute("download")) return click.call(this);
    const name = this.getAttribute("download") || "download";
    fetch(this.href).then((r) => r.blob()).then((b) => {
      const f = new FileReader();
      f.onload = () => up({ type: "download", name, data: f.result });
      f.readAsDataURL(b);
    });
  };
  document.addEventListener("click", (e) => {
    const a = e.target && e.target.closest && e.target.closest("a[href]");
    if (!a) return;
    if (a.hasAttribute("download")) { e.preventDefault(); a.click(); return; }
    const h = a.getAttribute("href") || "";
    if (/^https?:/.test(h)) { e.preventDefault(); up({ type: "open", url: h }); }
  }, true);
})();
