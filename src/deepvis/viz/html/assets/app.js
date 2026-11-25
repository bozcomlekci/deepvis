/* deepvis dashboard — vanilla JS, no build step. Data arrives as one JSON payload. */
(() => {
  "use strict";

  let P = JSON.parse(document.getElementById("dv-payload").textContent);
  const API = typeof window.DV_API === "string" ? window.DV_API : null;  // "" = same origin
  const LIVE = !!(P.meta && P.meta.live && API !== null);

  // ------------------------------------------------------------------ utils
  const SVGNS = "http://www.w3.org/2000/svg";
  function h(tag, attrs, ...kids) {
    const e = document.createElement(tag);
    setAttrs(e, attrs);
    for (const c of kids.flat()) if (c !== null && c !== undefined && c !== false) e.append(c instanceof Node ? c : document.createTextNode(String(c)));
    return e;
  }
  function s(tag, attrs, ...kids) {
    const e = document.createElementNS(SVGNS, tag);
    setAttrs(e, attrs);
    for (const c of kids.flat()) if (c !== null && c !== undefined && c !== false) e.append(c instanceof Node ? c : document.createTextNode(String(c)));
    return e;
  }
  function setAttrs(e, attrs) {
    if (!attrs) return;
    for (const [k, v] of Object.entries(attrs)) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "class") e.setAttribute("class", v);
      else if (k === "text") e.textContent = v;
      else if (k === "style" && typeof v === "object") Object.assign(e.style, v);
      else if (k.startsWith("on") && typeof v === "function") e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v === true ? "" : v);
    }
  }
  const clamp = (v, a = 0, b = 1) => Math.max(a, Math.min(b, v));
  const decoded = new Map();
  function f32(b64) {
    if (decoded.has(b64)) return decoded.get(b64);
    let lo = 0, hi = 0, q = false, body = b64;
    if (b64.startsWith("u8:")) {  // display data quantized to 8 bits over [lo, hi] (see payload.b64q)
      const parts = b64.split(":");
      lo = Number(parts[1]); hi = Number(parts[2]); body = parts.slice(3).join(":"); q = true;
    }
    const bin = atob(body);
    const buf = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
    let out;
    if (q) {
      out = new Float32Array(buf.length);
      const k = (hi - lo) / 255;
      for (let i = 0; i < buf.length; i++) out[i] = lo + buf[i] * k;
    } else out = new Float32Array(buf.buffer);
    if (decoded.size > 400) decoded.clear();
    decoded.set(b64, out);
    return out;
  }
  function hexRgb(hx) {
    const v = hx.replace("#", "");
    return [parseInt(v.slice(0, 2), 16), parseInt(v.slice(2, 4), 16), parseInt(v.slice(4, 6), 16)];
  }
  function ramp(stops) {
    const rgb = stops.map(hexRgb);
    return (t) => {
      t = clamp(isFinite(t) ? t : 0.5);
      const p = t * (rgb.length - 1);
      const i = Math.min(Math.floor(p), rgb.length - 2);
      const f = p - i;
      return [0, 1, 2].map((k) => Math.round(rgb[i][k] * (1 - f) + rgb[i + 1][k] * f));
    };
  }
  let DIV = ramp(P.theme.diverging), SEQ = ramp(P.theme.sequential);
  const rgba = (c, a = 1) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
  function colorOf(v, vmin, vmax, div) {
    const t = vmax === vmin ? 0.5 : (v - vmin) / (vmax - vmin);
    return (div ? DIV : SEQ)(t);
  }
  function fmt(v, d = 3) {
    if (v === null || v === undefined || Number.isNaN(v)) return "—";
    if (v === 0) return "0";
    const a = Math.abs(v);
    if (a >= 1e5 || a < 1e-3) return v.toExponential(Math.max(d - 1, 1));
    const digits = Math.max(0, d - 1 - Math.floor(Math.log10(a)));
    return v.toFixed(Math.min(digits, 6));
  }
  function compact(n) {
    if (n >= 1e9) return (n / 1e9).toFixed(n >= 1e10 ? 0 : 1) + "B";
    if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 0 : 1) + "M";
    if (n >= 1e4) return (n / 1e3).toFixed(n >= 1e5 ? 0 : 1) + "K";
    return n.toLocaleString();
  }
  const SMP = () => (P.meta && P.meta.sample) || 0;
  // which input the page shows, only when there is a choice (a diffusion probe is a set of timesteps)
  const probeT = () => (P.diffusion && P.diffusion.probeTs && P.diffusion.probeTs.length === P.meta.batch ? P.diffusion.probeTs : null);
  const SAMPLE = (sep = " · ") => ((P.meta.batch || 1) > 1 ? sep + (probeT() ? `t = ${probeT()[SMP()]}` : `sample ${SMP()}`) : "");
  // internal node ids (@in:x, @add_3, @param:w) as people read them
  function nodeName(id) {
    if (id === null || id === undefined) return "";
    if (id === "" ) return P.meta.name;
    if (id === "@out") return "output";
    if (id.startsWith("@in:")) return id.slice(4);
    if (id.startsWith("@param:")) return id.slice(7);
    if (id.startsWith("@const:")) return "constant for " + id.slice(7);
    if (/^\d+$/.test(id)) return `[${id}]`;  // an nn.Sequential index
    return id.replace(/^@/, "");
  }
  const detailLabel = (lv) => (lv.depth === null ? "every layer" : lv.depth === 0 ? "top level" : `level ${lv.depth}`);
  const shapeStr = (sh) => (sh && sh.length ? "(" + sh.join(", ") + ")" : "");
  function tex(el, latex, display = true) {
    if (window.katex) {
      try {
        window.katex.render(latex, el, { throwOnError: false, displayMode: display, strict: "ignore", output: "html",
          trust: (ctx) => ctx.command === "\\htmlData" });  // equation terms carry their layer (data-n), nothing else
        return el;
      } catch (e) { /* fall through */ }
    }
    el.textContent = latex;
    el.classList.add("tex-fallback");
    return el;
  }

  // --------------------------------------------------------------- tooltip
  const tip = h("div", { class: "tip", role: "tooltip" });
  document.body.append(tip);
  function showTip(ev, rows) {
    tip.replaceChildren(...rows.map((r) => {
      if (typeof r === "string") return h("div", { class: "tl", text: r });
      if (r instanceof Node) return h("div", { class: "tr tnode" }, r);  // e.g. a sample's thumbnail
      return h("div", { class: "tr" }, h("span", { class: "tv", text: r[0] }), h("span", { class: "tl", text: r[1] || "" }));
    }));
    tip.style.opacity = 1;
    moveTip(ev);
  }
  function moveTip(ev) {
    const pad = 14, w = tip.offsetWidth, hgt = tip.offsetHeight;
    let x = ev.clientX + pad, y = ev.clientY + pad;
    if (x + w > window.innerWidth - 8) x = ev.clientX - w - pad;
    if (y + hgt > window.innerHeight - 8) y = ev.clientY - hgt - pad;
    tip.style.left = x + "px";
    tip.style.top = y + "px";
  }
  const hideTip = () => { tip.style.opacity = 0; };
  const toastEl = h("div", { class: "toast", role: "status" });
  document.body.append(toastEl);
  let toastTimer = null;
  function toast(msg, err = false) {
    toastEl.textContent = msg;
    toastEl.className = "toast show" + (err ? " err" : "");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { toastEl.className = "toast" + (err ? " err" : ""); }, 2600);
  }

  // ----------------------------------------------------------------- state
  const UI = P.ui || {};
  const ALL_VIEWS = [["graph", "Graph"], ["volumes", "Volumes"], ["neurons", "Neurons"], ["attention", "Attention"],
    ["diffusion", "Diffusion"], ["compare", "Compare"], ["dataset", "Dataset"], ["equations", "Equations"], ["monitor", "Monitor"]];
  const HAS = { attention: () => (P.anatomy || []).length > 0, diffusion: () => !!P.diffusion, compare: () => !!P.compare,
    dataset: () => !!P.dataset };
  const VIEWS = ALL_VIEWS.filter(([id]) => (!UI.views || UI.views.includes(id)) && (!HAS[id] || HAS[id]()));
  const S = {
    tab: UI.start && VIEWS.some(([id]) => id === UI.start) ? UI.start : VIEWS[0][0],
    showTensors: true,
    level: P.defaultLevel || 0,
    selected: P.focus || null,
    neuron: null,
    view: null, // {k, x, y}
    eqMode: "system",
    eqDepth: "full",
    monStat: "mean",
    monNode: null,
    search: "",
    volMerge: true,
    volView: null,
  };
  let VS = null;  // the Volumes tab's 3D scene
  const stage = document.getElementById("stage");
  const panel = document.getElementById("panel");
  const level = () => P.levels[Math.min(S.level, P.levels.length - 1)];
  const nodeById = (id) => {
    for (const lv of [level(), ...P.levels]) {
      const n = lv.nodes.find((m) => m.id === id);
      if (n) return n;
    }
    return null;
  };
  const actFor = (n) => n && (P.acts[n.id] || (n.rep ? P.acts[n.rep] : null));

  // ------------------------------------------------------------------ theme
  function applyTheme() {
    const t = P.theme, st = document.documentElement.style, pal = t.palette || {};
    const set = (k, v) => { if (v) st.setProperty(k, v); };
    set("--bg", t.background); set("--panel", t.panel); set("--panel-2", t.panel2); set("--edge", t.grid);
    set("--edge-2", t.panelEdge); set("--grid", t.grid); set("--text", t.text); set("--text-2", t.text2);
    set("--muted", t.muted); set("--faint", t.muted); set("--accent", t.highlight); set("--pos", t.positive);
    set("--neg", t.negative); set("--pos-e", pal.BLUE_E); set("--pos-a", pal.BLUE_A); set("--neg-e", pal.RED_E);
    set("--neg-a", pal.RED_A);
  }

  // ---------------------------------------------------------------- header
  // which sample of a batched input the views show: switchable live, fixed at export time otherwise
  function sampleSwitch() {
    const k = SMP(), B = P.meta.batch, ts = probeT();
    const what = ts ? h("span", null, "t\u00a0=\u00a0", h("b", { text: String(ts[k]) })) : h("span", null, "sample\u00a0", h("b", { text: String(k) }), `\u00a0of\u00a0${B}`);
    if (!LIVE) return h("span", { title: ts ? "the timestep shown in every view" : "the input shown in every view (export_html(..., sample=k) picks another)" }, what);
    const go = (i) => api("/api/sample", { index: (i + B) % B });
    return h("span", { class: "sample-sw", role: "group", "aria-label": "sample in batch" },
      h("button", { class: "btn", text: "‹", title: "previous sample", "aria-label": "previous sample", onclick: () => go(k - 1) }),
      what,
      h("button", { class: "btn", text: "›", title: "next sample", "aria-label": "next sample", onclick: () => go(k + 1) }));
  }
  function renderHeader() {
    document.getElementById("dv-title").textContent = UI.title || P.meta.name;
    document.getElementById("dv-fw").textContent = P.meta.framework + (LIVE ? " · live" : "") +
      (P.meta.shapesOnly ? " · shapes only" : "");
    if (P.meta.shapesOnly) document.getElementById("dv-fw").title = "This model is on the meta device: its architecture and shapes, no numbers";
    const tabs = document.getElementById("dv-tabs");
    tabs.style.display = VIEWS.length > 1 ? "" : "none";
    tabs.replaceChildren(...VIEWS.map(([id, label], i) => h("button", {
      role: "tab", "aria-selected": String(S.tab === id), text: label, title: i < 9 ? `${label}  (${i + 1})` : label,
      onclick: () => { S.tab = id; renderHeader(); renderStage(); },
    })));
    syncHash();
    const st = document.getElementById("dv-stats");
    st.replaceChildren(...[
      h("span", null, h("b", { text: compact(P.meta.params) }), "parameters"),
      h("span", null, h("b", { text: String(P.meta.nodes) }), "nodes"),
      P.meta.edits ? h("span", null, h("b", { text: String(P.meta.edits) }), "edits") : null,
      (P.meta.batch || 1) > 1 ? sampleSwitch() : null,
      P.training ? h("span", { class: "train-badge", title: "training progress" },
        h("b", { text: "step " + (P.training.lastStep ?? "—") }),
        ...Object.entries(P.training.metrics).slice(0, 2).map(([k, v]) => h("span", { text: `${k} ${fmt(lastOf(v), 4)}` }))) : null,
    ].filter(Boolean));
    const act = document.getElementById("dv-actions");
    act.replaceChildren(h("button", { class: "btn help-btn", id: "dv-help", text: "?", "aria-label": "help",
      title: "How to read this page, the keys, and a short tour  (?)", onclick: () => openHelp() }));
    if (LIVE) {
      act.prepend(
        h("button", { class: "btn", title: "Undo last edit", onclick: () => api("/api/undo"), text: "↶ Undo" }),
        h("button", { class: "btn", title: "Redo", onclick: () => api("/api/redo"), text: "↷ Redo" }),
        h("button", { class: "btn danger", title: "Undo every weight edit made in this session", onclick: () => api("/api/reset"), text: "Revert edits" }),
        h("button", { class: "btn", title: "Download the edit log (replay with net.apply_edits)", onclick: exportEdits, text: "Export edits" }),
      );
      const TR = P.meta.trainer;
      if (TR) {  // the page drives training: net.serve(x, train_step=fn)
        const done = TR.total && TR.step >= TR.total;
        act.prepend(
          h("span", { class: "train-prog", title: "optimization steps taken" },
            h("b", { text: TR.total ? `${TR.step} / ${TR.total}` : String(TR.step) }), " steps",
            TR.total ? h("i", { class: "bar" }, h("i", { style: { width: `${Math.min(100, (100 * TR.step) / TR.total)}%` } })) : null),
          h("button", { class: "btn primary", disabled: done ? "" : null, title: TR.running ? "Pause training" : "Run the training loop in the Python process",
            text: TR.running ? "❚❚ Pause" : done ? "✓ Trained" : TR.step ? "▶ Resume" : "▶ Train",
            onclick: () => api("/api/train", { action: TR.running ? "pause" : "start" }) }),
          h("button", { class: "btn", text: "Step", title: "One optimization step", disabled: TR.running || done ? "" : null,
            onclick: () => api("/api/train", { action: "step" }) }),
          h("button", { class: "btn danger", text: "↺ Reset training", title: "Restore the initial weights and clear the training log",
            onclick: () => api("/api/train", { action: "reset" }) }));
        if (TR.error) toast("training stopped: " + TR.error, true);
      } else if (P.training) act.prepend(h("button", { class: "btn", "aria-pressed": String(S.paused),
        title: "Stop following the training loop (the Python side keeps logging)",
        text: S.paused ? "▶ Follow training" : "❚❚ Pause updates", onclick: () => { S.paused = !S.paused; renderHeader(); } }));
    }
  }

  // ----------------------------------------------------------------- stage
  let shownView = null;
  function renderStage() {
    hideTip();
    const view = S.tab + (S.tab === "equations" ? ":" + S.eqMode : "");
    if (view !== shownView && shownView !== null && !reducedMotion() && stage.animate) {
      stage.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 220, easing: "ease-out" });  // 3b1b's FadeIn
    }
    shownView = view;
    stage.replaceChildren();
    if (VS) { VS.dispose(); VS = null; }
    if (S.tab === "graph") renderGraph();
    else if (S.tab === "volumes") renderVolumes();
    else if (S.tab === "neurons") renderNeurons();
    else if (S.tab === "equations") renderEquations();
    else if (S.tab === "attention") renderAttention();
    else if (S.tab === "diffusion") renderDiffusion();
    else if (S.tab === "compare") renderCompare();
    else if (S.tab === "dataset") renderDataset();
    else renderMonitor();
    const cap = captionBar(S.tab);
    if (cap) {
      const doc = stage.querySelector(".eqview, .tableview");  // a text view: the caption opens it, in the flow
      if (doc) { cap.classList.add("inline"); doc.prepend(cap); } else stage.append(cap);
    }
  }

  // ------------------------------------------------------------- pan/zoom
  function makeCanvas(width, height, onBackgroundClick) {
    const svg = s("svg", { class: "canvas", role: "img" });
    const defs = s("defs");
    defs.append(
      s("marker", { id: "arrow", viewBox: "0 0 10 10", refX: "9", refY: "5", markerWidth: "7", markerHeight: "7", orient: "auto-start-reverse" },
        s("path", { d: "M 0 1 L 9 5 L 0 9 z", fill: P.theme.palette.GREY_C })),
      s("marker", { id: "arrow-hl", viewBox: "0 0 10 10", refX: "9", refY: "5", markerWidth: "7", markerHeight: "7", orient: "auto-start-reverse" },
        s("path", { d: "M 0 1 L 9 5 L 0 9 z", fill: P.theme.text })),
      s("filter", { id: "glow", x: "-40%", y: "-40%", width: "180%", height: "180%" },
        s("feGaussianBlur", { stdDeviation: "4", result: "b" }),
        s("feMerge", null, s("feMergeNode", { in: "b" }), s("feMergeNode", { in: "SourceGraphic" }))),
    );
    svg.append(defs);
    const vp = s("g", { class: "viewport" });
    svg.append(vp);
    const view = { k: 1, x: 0, y: 0, width, height };
    const apply = () => vp.setAttribute("transform", `translate(${view.x},${view.y}) scale(${view.k})`);
    const fit = () => {
      const r = svg.getBoundingClientRect();
      if (!r.width || !r.height) return;
      const cap = svg.parentNode && svg.parentNode.querySelector(".caption:not(.inline)");
      const top = 56 + (cap ? cap.offsetHeight + 10 : 0);  // below the toolbar and the view's caption
      const kw = (r.width - 40) / Math.max(width, 1), kh = (r.height - top - 70) / Math.max(height, 1);
      let k = Math.min(kw, kh, 1.4);
      // a very tall network shrunk to fit would be unreadable: keep text legible, pan to see the rest
      if (k < 0.62) k = Math.min(kw, 0.8);
      view.k = Math.max(k, 0.08);
      view.x = (r.width - width * view.k) / 2;
      view.y = top + Math.max(0, (r.height - top - 70 - height * view.k) / 2);
      apply();
    };
    let drag = null;
    svg.addEventListener("pointerdown", (ev) => {
      if (ev.button !== 0) return;
      drag = { x: ev.clientX, y: ev.clientY, vx: view.x, vy: view.y, moved: false, target: ev.target, id: ev.pointerId };
    });
    svg.addEventListener("pointermove", (ev) => {
      if (!drag) return;
      const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
      if (!drag.moved && Math.abs(dx) + Math.abs(dy) > 4) {
        drag.moved = true;
        svg.classList.add("panning");
        try { svg.setPointerCapture(drag.id); } catch (e) { /* pointer already released */ }
      }
      if (drag.moved) { view.x = drag.vx + dx; view.y = drag.vy + dy; apply(); }
    });
    svg.addEventListener("pointerup", (ev) => {
      if (drag && !drag.moved && (drag.target === svg || drag.target.closest(".bg-click")) && onBackgroundClick) onBackgroundClick();
      drag = null;
      svg.classList.remove("panning");
    });
    svg.addEventListener("wheel", (ev) => {
      ev.preventDefault();
      const r = svg.getBoundingClientRect();
      const mx = ev.clientX - r.left, my = ev.clientY - r.top;
      const f = Math.exp(-ev.deltaY * (ev.ctrlKey ? 0.01 : 0.0015));
      const k2 = clamp(view.k * f, 0.05, 6);
      view.x = mx - (mx - view.x) * (k2 / view.k);
      view.y = my - (my - view.y) * (k2 / view.k);
      view.k = k2;
      apply();
    }, { passive: false });
    const zoomBy = (f) => {
      const r = svg.getBoundingClientRect();
      const mx = r.width / 2, my = r.height / 2, k2 = clamp(view.k * f, 0.05, 6);
      view.x = mx - (mx - view.x) * (k2 / view.k);
      view.y = my - (my - view.y) * (k2 / view.k);
      view.k = k2;
      apply();
    };
    return { svg, vp, view, fit, apply, zoomBy };
  }
  function zoomControls(c) {
    return h("div", { class: "zoomctl" },
      h("button", { class: "btn", title: "Zoom in", "aria-label": "Zoom in", onclick: () => c.zoomBy(1.25), text: "+" }),
      h("button", { class: "btn", title: "Zoom out", "aria-label": "Zoom out", onclick: () => c.zoomBy(0.8), text: "−" }),
      h("button", { class: "btn", title: "Fit to view", "aria-label": "Fit to view", onclick: () => c.fit(), text: "⤢" }));
  }
  function smoothPath(pts) {
    let d = `M ${pts[0][0]} ${pts[0][1]}`;
    for (let i = 1; i < pts.length; i++) {
      const [x0, y0] = pts[i - 1], [x1, y1] = pts[i];
      const dy = (y1 - y0) / 2;
      d += ` C ${x0} ${y0 + dy}, ${x1} ${y1 - dy}, ${x1} ${y1}`;
    }
    return d;
  }

  // ------------------------------------------------------------ graph view
  let GV = null, GV_tensors = null;
  function renderGraph() {
    const L = level();
    const lay = L.layout;
    const c = makeCanvas(lay.width, lay.height, () => select(null));
    GV = { c, nodes: new Map(), edges: [] };
    const vp = c.vp;
    // group boxes
    const gLayer = s("g");
    for (const gr of [...lay.groups].sort((a, b) => a.depth - b.depth)) {
      const lab = s("text", { class: "group-label", x: gr.x + 12, y: gr.y + 16 }, gr.type + " ",
        s("tspan", { class: "gid", text: " " + gr.id }));
      const box = s("rect", { class: "group-box", x: gr.x, y: gr.y, width: gr.w, height: gr.h, rx: 14 });
      const g = s("g", { class: "group" }, box, lab);
      g.addEventListener("click", (ev) => { ev.stopPropagation(); select(gr.id); });
      g.style.cursor = "pointer";
      gLayer.append(g);
    }
    vp.append(gLayer);
    // edges
    const eLayer = s("g");
    const shapeOf = new Map(L.edges.map((e) => [e.src + "→" + e.dst, e]));
    for (const e of lay.edges) {
      const info = shapeOf.get(e.src + "→" + e.dst) || {};
      const d = smoothPath(e.points);
      const path = s("path", { class: "edge" + (info.implicit ? " implicit" : "") + (e.far ? " far" : ""), d, "marker-end": "url(#arrow)" });
      const hit = s("path", { class: "edge-hit", d });
      hit.addEventListener("pointermove", (ev) => showTip(ev, [[shapeStr(info.shape) || "tensor", `${e.src || "·"} → ${e.dst}`], info.implicit ? "inferred from execution order" : ""].filter(Boolean)));
      hit.addEventListener("pointerleave", hideTip);
      eLayer.append(path, hit);
      GV.edges.push({ src: e.src, dst: e.dst, path });
    }
    vp.append(eLayer);
    // 3-D tensor shapes flowing along edges
    const tLayer = s("g", { class: "tensors" });
    for (const t of lay.tensors || []) {
      const tg = s("g", { class: "tensor", transform: `translate(${t.x},${t.y})` });
      tg.innerHTML = t.svg != null ? t.svg : (P.tensorSet || [])[t.si] || "";
      const src = L.nodes.find((m) => m.id === t.src);
      tg.addEventListener("pointermove", (ev) => {
        const a = src ? actFor(src) : null;
        const rows = [[src && src.out_shapes && src.out_shapes[0] ? shapeStr(src.out_shapes[0]) : "tensor", `output of ${t.src || P.meta.name}`]];
        if (a && a.stats) rows.push([`${fmt(a.stats.mean)} ± ${fmt(a.stats.std)}`, "mean ± std"]);
        showTip(ev, rows);
      });
      tg.addEventListener("pointerleave", hideTip);
      tLayer.append(tg);
    }
    tLayer.style.display = S.showTensors ? "" : "none";
    vp.append(tLayer);
    GV_tensors = tLayer;
    // nodes
    const nLayer = s("g");
    for (const n of L.nodes) {
      if (!n.layout) continue;
      const g = drawNode(n);
      const cmp = compareOf(n.id);
      if (cmp) g.style.filter = `drop-shadow(0 0 ${2 + 10 * cmp.t}px ${rgba(CMP_COL, 0.35 + 0.6 * cmp.t)})`;  // how much it changed
      nLayer.append(g);
      GV.nodes.set(n.id, g);
    }
    vp.append(nLayer);
    stage.append(c.svg);
    stage.append(graphToolbar(c), legend(), zoomControls(c));
    requestAnimationFrame(() => {
      if (S.view && S.view.level === S.level) { Object.assign(c.view, S.view); c.apply(); } else c.fit();
      applySelection();
      if (MORPH) { morphFrom(MORPH); MORPH = null; }
    });
    c.svg.addEventListener("pointerup", () => { S.view = { ...c.view, level: S.level }; });
    c.svg.addEventListener("wheel", () => { S.view = { ...c.view, level: S.level }; }, { passive: true });
  }

  function paramCount(n) { return (n.params || []).reduce((a, p) => a + p.numel, 0); }
  function isEdited(n) { return (n.params || []).some((p) => P.params[p.path] && P.params[p.path].edited); }

  // each distinct glyph is shipped once (P.glyphSet); P.glyphs maps a node to it (older pages: the SVG itself)
  function glyphOf(id) {
    const v = P.glyphs && P.glyphs[id];
    return typeof v === "number" ? (P.glyphSet || [])[v] : v;
  }

  function drawNode(n) {
    const { x, y, w, h: hh } = n.layout;
    const col = hexRgb(n.color);
    const g = s("g", { class: "node", transform: `translate(${x - w / 2},${y - hh / 2})`, tabindex: "0", role: "button", "aria-label": `${n.kind} ${n.id}` });
    const glyph = glyphOf(n.id);
    if (UI.style === "expressive" && glyph) {
      g.classList.add("x");
      g.innerHTML = glyph;  // SVG built by deepvis.viz.glyphs from this model's numbers
      if (isEdited(n)) g.append(s("circle", { class: "n-dot", cx: w - 6, cy: 6, r: 4 }));
      return nodeEvents(g, n);
    }
    const pill = n.op === "function" || n.op === "input" || n.op === "output" || n.op === "param";
    const rx = pill ? hh / 2 : 10;
    g.append(s("rect", { class: "body", width: w, height: hh, rx, fill: rgba(col, n.op === "input" || n.op === "output" ? 0.06 : 0.13), stroke: rgba(col, 0.75), "stroke-width": 1.25 }));
    if (!pill) {
      g.append(s("rect", { x: 0, y: 0, width: 4, height: hh, rx: 2, fill: n.color }));
      if (n.collapsed) g.append(s("rect", { x: 3.5, y: 3.5, width: w - 7, height: hh - 7, rx: 8, fill: "none", stroke: rgba(col, 0.35), "stroke-width": 1 }));
      const title = n.collapsed ? n.type : n.kind;
      g.append(s("text", { class: "n-kind", x: 14, y: 20 }, title));
      const out = n.out_shapes && n.out_shapes[0] ? shapeStr(n.out_shapes[0]) : "";
      if (out) g.append(s("text", { class: "n-sub", x: w - 10, y: 20, "text-anchor": "end" }, out));
      let sub = n.label || n.id;
      if (sub.length > 22) sub = "…" + sub.slice(-21);
      g.append(s("text", { class: "n-sub", x: 14, y: hh - 13 }, sub));
      const pc = paramCount(n);
      if (pc) g.append(s("text", { class: "n-sub", x: w - 10, y: hh - 13, "text-anchor": "end" }, compact(pc) + " w"));
    } else {
      let label = n.op === "input" ? (n.label || "x") : n.op === "output" ? "output" : n.kind;
      if (n.op === "param") label = "θ " + (n.label || "");
      const shp = (n.op === "input" && n.out_shapes && n.out_shapes[0]) ? " " + shapeStr(n.out_shapes[0]) : "";
      let txt = label + shp;
      if (txt.length > 18) txt = txt.slice(0, 17) + "…";
      g.append(s("text", { class: "n-op", x: w / 2, y: hh / 2 + 4, "text-anchor": "middle" }, txt));
    }
    if (isEdited(n)) g.append(s("circle", { class: "n-dot", cx: w - 6, cy: 6, r: 4 }));
    return nodeEvents(g, n);
  }

  const GLYPH_TITLE = {
    activation: "Transfer function", linear: "Weights", conv: "Kernel", norm: "Normalization", softmax: "Output distribution",
    attention: "Attention pattern", pool: "Pooling", input: "Input tensor", output: "Output", dropout: "Dropout",
    embedding: "Lookup", recurrent: "Recurrence", param: "Parameter", concat: "Concatenation", split: "Split",
    matmul: "Matrix product", mask: "Mask", rotary: "Rotary position", constant: "Constant", expand: "Broadcast",
    graph: "Message passing",
  };
  const GLYPH_CAPTION = {  // one line each: what the drawing encodes
    activation: "Its real curve; bars: the inputs it received, bold: where 90% of them fall.",
    linear: "Inputs (top) to outputs (bottom); lines are real weights, blue + / red −, width |w|.",
    conv: "Depth: number of filters. Face: the first filter's weights.",
    norm: "Grey: before · colour: after. Grid: what is normalized together.",
    softmax: "The probabilities it produced; the winner in blue.",
    attention: "Attention weights, averaged over heads: query rows × key columns.",
    pool: "Many cells to few; lit cells are the ones a max-pool keeps.",
    input: "The input's shape as a block, the image itself on its face.",
    output: "The network's output.",
    dropout: "Hollow: the fraction p dropped while training.",
    embedding: "A lookup: each token id selects one row.",
    recurrent: "A cell feeding its state back into itself, step by step.",
    param: "A weight tensor used directly; its values.",
    concat: "Tensors joined along one axis.",
    split: "One tensor cut into chunks, one per output.",
    matmul: "A row of A meets a column of B in one cell of the product.",
    mask: "Kept cells (blue) and masked ones (hatched).",
    rotary: "Feature pairs rotated by position, at several frequencies.",
    constant: "A tensor made inside forward (a ramp: arange).",
    expand: "One row repeated into several.",
    graph: "Each node gathers its neighbours' features along the edges of the input graph.",
  };

  function nodeEvents(g, n) {
    g.addEventListener("click", (ev) => { ev.stopPropagation(); select(n.id); });
    g.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); select(n.id); } });
    g.addEventListener("pointermove", (ev) => {
      const rows = [[n.collapsed ? n.type : n.kind, n.id || P.meta.name]];
      if (n.in_shapes && n.in_shapes[0]) rows.push([shapeStr(n.in_shapes[0]) + " → " + shapeStr(n.out_shapes && n.out_shapes[0]), "in → out"]);
      const pc = paramCount(n);
      if (pc) rows.push([pc.toLocaleString(), "parameters"]);
      const a = actFor(n);
      if (a && a.stats) rows.push([`${fmt(a.stats.mean)} ± ${fmt(a.stats.std)}`, "activation mean ± std"]);
      if (n.attrs && n.attrs.prims) rows.push("primitives: " + n.attrs.prims.join(", "));
      const ex = (P.explain || {})[n.id];
      if (ex) rows.push(ex.what);
      showTip(ev, rows);
    });
    g.addEventListener("pointerleave", hideTip);
    return g;
  }

  function graphToolbar(c) {
    const lvSel = h("div", { class: "seg", role: "group", "aria-label": "detail level" }, h("span", { class: "seg-lab", text: "detail" }),
      ...P.levels.map((lv, i) => h("button", {
        "aria-pressed": String(i === S.level), title: `${lv.n} nodes`,
        text: detailLabel(lv),
        onclick: () => { if (i === S.level) return; MORPH = captureLayout(); S.level = i; S.view = null; renderStage(); syncHash(); },
      })));
    const search = h("input", { class: "search", type: "search", placeholder: "Search layers…   /", value: S.search, "aria-label": "search layers (/)" });
    search.addEventListener("input", () => { S.search = search.value.trim().toLowerCase(); applySelection(); });
    search.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") {
        const m = level().nodes.find((n) => n.layout && matches(n));
        if (m) select(m.id);
      }
    });
    const hasTensors = level().layout.tensors && level().layout.tensors.length;
    const tbtn = hasTensors ? h("button", { class: "btn", "aria-pressed": String(S.showTensors), text: "3-D tensors",
      title: "Show the shape of the tensor flowing along each edge",
      onclick: () => { S.showTensors = !S.showTensors; tbtn.setAttribute("aria-pressed", String(S.showTensors));
        if (GV_tensors) GV_tensors.style.display = S.showTensors ? "" : "none"; } }) : null;
    return h("div", { class: "toolbar" },
      P.levels.length > 1 ? lvSel : null,
      search, tbtn,
      h("button", { class: "btn", text: "⤓ SVG", title: "Download this view (current depth, tensors on/off) as a standalone SVG figure",
        onclick: () => downloadGraphSvg() }),
      h("div", { class: "spacer" }),
      h("button", { class: "btn primary", onclick: () => animateForward(), text: "▶ Forward pass", title: "Animate data flowing through the network  (F)" }));
  }
  // a standalone SVG of what the graph view shows: the CSS the figure uses is inlined with its variables resolved
  function downloadGraphSvg() {
    if (!GV) return;
    const L = level().layout;
    const NS = "http://www.w3.org/2000/svg";
    const body = GV.c.vp.cloneNode(true);
    body.removeAttribute("transform");
    body.querySelectorAll(".edge-hit").forEach((e) => e.remove());
    body.querySelectorAll("[style*='display: none']").forEach((e) => e.remove());
    const strip = ["dimmed", "selected", "sel", "hl", "flow", "near", "focus", "match", "pulse"];
    [body, ...body.querySelectorAll("[class]")].forEach((e) => {
      const cls = (e.getAttribute("class") || "").split(/\s+/).filter((c) => c && !strip.includes(c));
      if (cls.length) e.setAttribute("class", cls.join(" ")); else e.removeAttribute("class");
    });
    const root = getComputedStyle(document.documentElement);
    const resolve = (txt) => txt.replace(/var\((--[\w-]+)\)/g, (_, v) => root.getPropertyValue(v).trim() || "currentColor");
    const probe = document.createElementNS(NS, "svg");
    probe.append(body);
    const rules = [];
    for (const sheet of document.styleSheets) {
      let list;
      try { list = sheet.cssRules; } catch (e) { continue; }  // cross-origin (KaTeX CDN)
      for (const r of list) {
        if (!r.selectorText || /:hover|:focus|::|panning/.test(r.selectorText)) continue;
        let hit = false;
        try { hit = !!probe.querySelector(r.selectorText); } catch (e) { hit = false; }
        if (hit) rules.push(resolve(r.cssText.replace(/transition:[^;]+;/g, "")));
      }
    }
    const out = document.createElementNS(NS, "svg");
    const W = Math.ceil(L.width), H = Math.ceil(L.height);
    out.setAttribute("xmlns", NS);
    out.setAttribute("viewBox", `0 0 ${W} ${H}`);
    out.setAttribute("width", W); out.setAttribute("height", H);
    out.setAttribute("font-family", "CMU Sans Serif, Inter, system-ui, sans-serif");
    const style = document.createElementNS(NS, "style");
    style.textContent = rules.join("\n");
    const bg = document.createElementNS(NS, "rect");
    bg.setAttribute("width", W); bg.setAttribute("height", H); bg.setAttribute("fill", P.theme.background || "#000000");
    out.append(style, GV.c.svg.querySelector("defs").cloneNode(true), bg, body);
    const xml = '<?xml version="1.0" encoding="UTF-8"?>\n' + new XMLSerializer().serializeToString(out);
    const a = h("a", { href: URL.createObjectURL(new Blob([xml], { type: "image/svg+xml" })),
      download: `${P.meta.name}${level().depth === null ? "" : "-depth" + level().depth}.svg` });
    document.body.append(a); a.click(); a.remove();
    toast("saved the current view as SVG");
  }
  const matches = (n) => S.search && ((n.id || "").toLowerCase().includes(S.search) || (n.kind || "").toLowerCase().includes(S.search) || (n.type || "").toLowerCase().includes(S.search));

  function legend(extra) {
    const fam = P.theme.families, labels = P.theme.familyLabels || {};
    const present = [...new Set(level().nodes.filter((n) => n.layout && !["input", "output", "container"].includes(n.op))
      .map((n) => n.family))].filter((f) => fam[f]);
    const sw = (key) => h("span", { class: "sw" }, h("i", { style: { background: fam[key] } }), labels[key] || key);
    const bar = h("canvas", { class: "bar", width: 120, height: 8 });
    const ctx = bar.getContext("2d");
    for (let i = 0; i < 120; i++) { ctx.fillStyle = rgba(DIV(i / 119)); ctx.fillRect(i, 0, 1, 8); }
    const box = h("div", { class: "legend" + (legendMin() ? " min" : "") });
    const tog = h("button", { class: "lg-tog", title: "show / hide the key", "aria-label": "show or hide the key",
      onclick: () => { const m = box.classList.toggle("min"); try { localStorage.setItem("dv-legend-min", m ? "1" : ""); } catch (e) { /* private mode */ } } },
      h("span", { class: "lg-open", text: "key" }), h("span", { class: "lg-close", text: "×" }));
    return box.appendChild(h("div", { class: "lg-body" }, tog,
      h("div", { class: "row" }, ...present.map(sw)),
      h("div", { class: "row" }, h("span", { class: "tl", text: `${extra || "weights"}:` }), h("span", { text: "negative" }), bar,
        h("span", { text: "positive" }), h("span", { class: "sw" }, h("i", { style: { background: P.theme.highlight } }),
          "selected / edited")))) && box;
  }
  function legendMin() { try { return localStorage.getItem("dv-legend-min") === "1"; } catch (e) { return false; } }

  function applySelection() {
    if (!GV) return;
    const sel = S.selected;
    const near = new Set();
    if (sel !== null && GV.nodes.has(sel)) {
      near.add(sel);
      for (const e of GV.edges) {
        if (e.src === sel) near.add(e.dst);
        if (e.dst === sel) near.add(e.src);
      }
    }
    for (const [id, g] of GV.nodes) {
      g.classList.toggle("selected", id === sel);
      g.classList.toggle("near", near.has(id));
      const n = level().nodes.find((m) => m.id === id);
      g.classList.toggle("match", !!(n && matches(n)));
    }
    for (const e of GV.edges) {
      const on = sel !== null && (e.src === sel || e.dst === sel);
      e.path.classList.toggle("hl", on);
      e.path.setAttribute("marker-end", on ? "url(#arrow-hl)" : "url(#arrow)");
    }
    GV.c.vp.classList.toggle("dimmed", sel !== null && GV.nodes.has(sel));
  }

  function animateForward() {
    if (!GV) return;
    const L = level();
    const layers = new Map();
    for (const n of L.nodes) if (n.layout) {
      const k = n.layout.layer;
      if (!layers.has(k)) layers.set(k, []);
      layers.get(k).push(n.id);
    }
    const keys = [...layers.keys()].sort((a, b) => a - b);
    const step = Math.max(70, Math.min(220, 2400 / Math.max(keys.length, 1)));
    keys.forEach((k, i) => {
      setTimeout(() => {
        for (const id of layers.get(k)) {
          const g = GV.nodes.get(id);
          if (!g) continue;
          g.classList.add("pulse");
          setTimeout(() => g.classList.remove("pulse"), step * 3);
          for (const e of GV.edges) if (e.src === id) flashAlong(e.path, step * 2.4);
        }
      }, i * step);
    });
  }

  // 3b1b's ShowPassingFlash: a short bright segment travels once along a connection, on a copy laid over it
  const reducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
  function flashAlong(el, duration) {
    if (reducedMotion() || !el.animate || !el.getTotalLength) return;
    let len = 0;
    try { len = el.getTotalLength(); } catch (e) { return; }
    if (!len) return;
    const f = el.cloneNode(false), seg = Math.min(26, len * 0.35);
    f.setAttribute("class", "edge-flash");
    f.removeAttribute("marker-end");
    f.removeAttribute("style");
    f.style.strokeDasharray = `${seg} ${len + seg}`;
    el.after(f);
    const a = f.animate([{ strokeDashoffset: seg }, { strokeDashoffset: -len }], { duration, easing: "cubic-bezier(.4, 0, .2, 1)" });
    a.onfinish = a.oncancel = () => f.remove();
  }

  function select(id) {
    S.selected = id;
    S.neuron = null;
    syncHash();
    applySelection();
    if (VS) { for (const b of VS.boxes) b.highlight = b.id === id; VS.draw(); }
    renderPanel();
    if (S.tab === "equations") highlightEqRows();
    if (S.tab === "monitor") renderStage();
  }

  // ---------------------------------------------------------- volumes view
  const texCache = new Map();
  function volTexture(face, vol) {
    if (!face) return null;
    const key = face.data + "|" + vol.vmin + "|" + vol.vmax;
    if (!texCache.has(key)) {
      if (texCache.size > 2000) texCache.clear();
      texCache.set(key, DV3D.texture(face, vol.vmin, vol.vmax, vol.diverging, colorOf, f32));
    }
    return texCache.get(key);
  }
  const volSize = (d) => 0.35 + 0.95 * Math.log2(Math.max(d, 1) + 1);
  const THIN = 0.3;
  function volumeOf(n) {
    const a = actFor(n);
    if (a && a.volume) return a.volume;
    let sh = n.out_shapes && n.out_shapes[0];
    if (!sh || !sh.length) return { kind: "vec", dims: [1, 1, 1] };
    if (sh.length >= 2) sh = sh.slice(1);
    if (sh.length === 1) return { kind: "vec", dims: [1, sh[0], 1] };
    if (sh.length === 2) return { kind: "seq", dims: [1, sh[0], sh[1]] };
    const chFirst = P.meta.channelsLast === undefined ? P.meta.framework === "torch" : !P.meta.channelsLast;
    const C = chFirst ? sh[0] : sh[sh.length - 1];
    const rest = chFirst ? sh.slice(1) : sh.slice(0, -1);
    return { kind: "maps", dims: [C, rest.slice(0, -1).reduce((a, b) => a * b, 1), rest[rest.length - 1]] };
  }
  function boxSize(v, factor) {
    const [C, H, W] = v.dims;
    if (v.kind === "vec") return [THIN, volSize(H), THIN];
    if (v.kind === "seq") return [THIN, volSize(H), volSize(W)];
    return [Math.max(THIN, volSize(C) * factor), volSize(H), volSize(W)];
  }
  const dimStr = (v) => v.kind === "vec" ? `${v.dims[1]}` : v.kind === "seq" ? `${v.dims[1]}×${v.dims[2]}`
    : `${v.dims[0]}×${v.dims[1]}×${v.dims[2]}`;

  function volumeScene(L) {
    const nodes = L.nodes.filter((n) => n.layout && n.op !== "container" && n.op !== "output");
    const ids = new Set(nodes.map((n) => n.id));
    const byId = new Map(nodes.map((n) => [n.id, n]));
    const preds = new Map(nodes.map((n) => [n.id, []])), succs = new Map(nodes.map((n) => [n.id, []]));
    for (const e of L.edges) if (ids.has(e.src) && ids.has(e.dst)) { succs.get(e.src).push(e.dst); preds.get(e.dst).push(e.src); }
    const order = [...nodes].sort((a, b) => a.layout.layer - b.layout.layer || a.layout.x - b.layout.x);
    const light = (n) => ["activation", "norm", "dropout"].includes(n.family) || n.kind === "Identity";
    const same = (a, b) => JSON.stringify((a.out_shapes || [])[0]) === JSON.stringify((b.out_shapes || [])[0]);
    const groups = [], gOf = new Map();
    for (const n of order) {
      let g = null;
      const pr = preds.get(n.id);
      if (S.volMerge && pr.length === 1) {
        const p = byId.get(pr[0]), gp = gOf.get(pr[0]);
        if (gp && gp.members[gp.members.length - 1] === p && succs.get(p.id).length === 1 && p.op !== "input" && light(n) && same(p, n)) g = gp;
      }
      if (!g) { g = { members: [], layer: n.layout.layer, x: n.layout.x }; groups.push(g); }
      g.members.push(n);
      gOf.set(n.id, g);
    }
    const layers = [...new Set(groups.map((g) => g.layer))].sort((a, b) => a - b);
    const boxes = [], labels = [], lines = [];
    // measure each layer (a column of parallel groups), then wrap long networks into rows
    const cols = layers.map((li) => {
      const gs = groups.filter((g) => g.layer === li).sort((a, b) => a.x - b.x);
      for (const g of gs) {
        g.parts = g.members.map((n, i) => { const v = volumeOf(n); return { n, v, s: boxSize(v, i === 0 ? 1 : 0.4) }; });
        g.dx = g.parts.reduce((a, p) => a + p.s[0], 0) + 0.08 * (g.parts.length - 1);
        g.dy = Math.max(...g.parts.map((p) => p.s[1]));
        g.dz = Math.max(...g.parts.map((p) => p.s[2]));
      }
      return { gs, width: Math.max(...gs.map((g) => g.dx)), dy: Math.max(...gs.map((g) => g.dy)) };
    });
    const GAP = 1.7;
    const total = cols.reduce((a, c) => a + c.width + GAP, 0);
    const pitch = Math.max(...cols.map((c) => c.dy), 1) + 3.2;
    const r = stage.getBoundingClientRect();
    const aspect = r.width && r.height ? r.width / Math.max(r.height - 60, 100) : 1.5;
    // choose rows so the wrapped pipeline roughly matches the canvas aspect ratio
    const nRows = Math.max(1, Math.min(8, Math.round(Math.sqrt(total / (pitch * aspect * 1.25)))));
    const target = total / nRows;
    let cursor = 0, rowY = 0, row = 0;
    for (const col of cols) {
      if (cursor > 0 && cursor + col.width > target * 1.04 && row < nRows - 1) {
        cursor = 0; row += 1; rowY -= pitch;
      }
      const totalZ = col.gs.reduce((a, g) => a + g.dz, 0) + 1.6 * (col.gs.length - 1);
      let z = -totalZ / 2;
      for (const g of col.gs) {
        g.z = z + g.dz / 2;
        g.y = rowY;
        z += g.dz + 1.6;
        let x = cursor + (col.width - g.dx) / 2;
        g.x0 = x;
        for (const p of g.parts) {
          const a = actFor(p.n), vol = a && a.volume;
          const box = { id: p.n.id, c: [x + p.s[0] / 2, rowY, g.z], s: p.s, color: p.n.color, node: p.n, vol: p.v,
                        highlight: p.n.id === S.selected, fillAlpha: 0.2 };
          if (vol) box.tex = { front: volTexture(vol.faces.front, vol), side: volTexture(vol.faces.side, vol),
                               top: volTexture(vol.faces.top, vol) };
          boxes.push(box);
          p.box = box;
          x += p.s[0] + 0.08;
        }
        g.x1 = g.x0 + g.dx;
        const kinds = [...new Set(g.members.map((n) => n.collapsed ? n.type : (n.op === "input" ? "input" : n.kind)))];
        const weighty = g.members.some((n) => (n.params || []).length) || g.members[0].op === "input";
        labels.push({ p: [(g.x0 + g.x1) / 2, rowY + g.dy / 2 + 0.45, g.z], text: kinds.join(" · "), sub: dimStr(g.parts[0].v),
                      color: P.theme.text, subColor: P.theme.text2, priority: weighty ? 2 : 1 });
      }
      cursor += col.width + GAP;
    }
    for (const e of L.edges) {
      const a = gOf.get(e.src), b = gOf.get(e.dst);
      if (!a || !b || a === b) continue;
      lines.push({ a: [a.x1, a.y, a.z], b: [b.x0, b.y, b.z], color: P.theme.panelEdge, width: 1.5,
                   alpha: a.y === b.y ? 0.9 : 0.45 });
    }
    return { boxes, labels, lines, groups };
  }

  function renderVolumes() {
    const L = level();
    const cv = h("canvas", { class: "vol-stage", role: "img", "aria-label": "3D view of the network's tensors" });
    stage.append(cv);
    const scene = volumeScene(L);
    const faceName = { front: "mean over channels", side: "mean over width", top: "mean over height" };
    VS = new DV3D.Scene3D(cv, {
      yaw: 0.62, pitch: 0.36, highlight: P.theme.highlight, font: `12px ${getComputedStyle(document.body).fontFamily}`,
      mono: `10.5px ${getComputedStyle(document.body).getPropertyValue("--mono") || "monospace"}`,
      onPick: (hit) => select(hit ? hit.box.id : null),
      onHover: (ev, hit) => {
        const n = hit.box.node, v = hit.box.vol;
        const rows = [[n.collapsed ? n.type : n.kind, n.id || P.meta.name], [dimStr(v), v.kind === "maps" ? "channels × height × width" : v.kind === "seq" ? "positions × features" : "features"]];
        const ex = (P.explain || {})[n.id];
        if (ex) rows.push(ex.what);
        if (hit.value !== undefined) {
          const [C, H, W] = v.dims;
          const ax = { front: [H, W, "h", "w"], side: [H, C, "h", "c"], top: [W, C, "w", "c"] }[hit.key];
          const r = Math.floor((hit.row + 0.5) * ax[0] / hit.rows), c = Math.floor((hit.col + 0.5) * ax[1] / hit.cols);
          const approx = hit.rows < ax[0] || hit.cols < ax[1] ? "≈ " : "";
          rows.unshift([fmt(hit.value, 4), `${faceName[hit.key]} · ${approx}${ax[2]}=${r}, ${ax[3]}=${c}`]);
        }
        showTip(ev, rows);
      },
      onLeave: hideTip,
      onView: (sc) => { S.volView = { yaw: sc.yaw, pitch: sc.pitch, k: sc.k, pan: [...sc.pan], level: S.level }; },
    }).set(scene);
    VS.opts.labelMinK = 0;
    requestAnimationFrame(() => {
      VS.resize();
      if (S.volView && S.volView.level === S.level) Object.assign(VS, S.volView);
      else {
        VS.fit(40, 56);
        // very long networks: keep blocks legible, start at the input; pan / zoom to explore
        const minK = 4;
        if (VS.k < minK && scene.boxes.length) {
          VS.k = minK;
          VS.pan = [0, 0];
          const first = VS.project(scene.boxes[0].c);
          VS.pan = [VS.W * 0.12 - first[0], VS.H * 0.45 - first[1]];
        }
      }
      VS.draw();
    });
    const merge = h("div", { class: "seg", role: "group", "aria-label": "grouping", title: "Draw norms / activations / dropout as slabs of the block they follow, or every op on its own" },
      h("span", { class: "seg-lab", text: "ops" }),
      ...[[true, "merged"], [false, "separate"]].map(([v, t]) => h("button", { "aria-pressed": String(S.volMerge === v), text: t,
        onclick: () => { S.volMerge = v; S.volView = null; renderStage(); } })));
    const lvSel = P.levels.length > 1 ? h("div", { class: "seg", role: "group", "aria-label": "detail level" }, h("span", { class: "seg-lab", text: "detail" }),
      ...P.levels.map((lv, i) => h("button", { "aria-pressed": String(i === S.level), text: detailLabel(lv),
        onclick: () => { S.level = i; S.volView = null; renderStage(); } }))) : null;
    let spinning = false;
    const spin = h("button", { class: "btn", text: "⟲ Turntable", onclick: () => { spinning = !spinning; VS.spin(spinning); spin.setAttribute("aria-pressed", String(spinning)); } });
    stage.append(h("div", { class: "toolbar" }, lvSel, merge, spin,
      h("button", { class: "btn", text: "Reset view", onclick: () => { S.volView = null; renderStage(); } }),
      h("div", { class: "spacer" }),
      h("button", { class: "btn primary", text: "▶ Forward pass", title: "Each tensor lights up as the data reaches it  (F)", onclick: () => volForward(scene) })));
    stage.append(legend("activation"));
  }

  function volForward(scene) {
    if (!VS) return;
    const gs = scene.groups;
    for (const b of VS.boxes) { b.bright = 0.12; b.glow = false; }
    const per = 1 / Math.max(gs.length, 1);
    VS.animate(Math.min(6000, 300 + gs.length * 180), (p) => {
      gs.forEach((g, i) => {
        const t = Math.max(0, Math.min(1, (p - i * per) / (per * 2.2)));
        for (const part of g.parts) {
          part.box.bright = 0.12 + 0.88 * t;
          part.box.glow = t > 0 && t < 1;
        }
      });
      if (p >= 1) for (const b of VS.boxes) { b.bright = 1; b.glow = false; }
    });
  }

  /** Rotatable stack of 2-D slices (channel maps, filters) with a slider to pick one. */
  function stackView(n, rows, cols, data, vmin, vmax, diverging, label) {
    const wrap = h("div", { class: "vol-panel" });
    const cv = h("canvas", { class: "vol-canvas" });
    const count = Math.min(n, 24);
    let sel = 0, all = false;
    const aspect = cols / rows;
    const Hs = 2, Ws = Math.min(4, 2 * aspect), gap = Math.max(0.28, 3.2 / count);
    const boxes = [];
    for (let i = 0; i < count; i++) {
      const tex = DV3D.sliceTexture(data, i * rows * cols, rows, cols, vmin, vmax, diverging, colorOf);
      boxes.push({ id: i, c: [i * gap, 0, 0], s: [0.04, Hs, Ws], color: P.theme.muted, tex: { front: tex }, edge: P.theme.panelEdge });
    }
    const info = h("span", { class: "v" });
    const slider = h("input", { type: "range", min: 0, max: count - 1, step: 1, value: 0, "aria-label": label + " index" });
    const scene = new DV3D.Scene3D(cv, {
      yaw: -1.05, pitch: 0.32, highlight: P.theme.highlight,
      onPick: (hit) => { if (hit) { sel = hit.box.id; all = false; slider.value = sel; paint(); } },
      onHover: (ev, hit) => showTip(ev, [[hit.value === undefined ? "—" : fmt(hit.value, 4), `${label} ${hit.box.id} · (${hit.row}, ${hit.col})`]]),
      onLeave: hideTip,
    }).set({ boxes });
    function paint() {
      boxes.forEach((b, i) => { b.alpha = all || i === sel ? 1 : 0.22; b.highlight = !all && i === sel; });
      info.textContent = all ? `all ${count}` : `${label} ${sel}`;
      scene.draw();
    }
    slider.addEventListener("input", () => { sel = Number(slider.value); all = false; paint(); });
    const allBtn = h("button", { class: "btn", text: "all", onclick: () => { all = !all; paint(); } });
    wrap.append(cv, h("div", { class: "ctl" }, h("label", { text: label }), slider, h("div", { style: { display: "flex", gap: "6px", alignItems: "center" } }, info, allBtn)));
    requestAnimationFrame(() => { scene.resize(); scene.fit(14); paint(); });
    return wrap;
  }

  /** Conv filter bank: each output filter drawn as an (in × kh × kw) volume. */
  function bankView(bank) {
    const cv = h("canvas", { class: "vol-canvas" });
    const [Cin, KH, KW] = bank.dims;
    const v = { kind: "maps", dims: bank.dims, vmin: bank.vmin, vmax: bank.vmax, diverging: true };
    const s = [Math.max(0.35, volSize(Cin) * 0.55), volSize(KH), volSize(KW)];
    const boxes = bank.filters.map((f, i) => ({
      id: i, c: [(i % 6) * (s[0] + 1.1), 0, Math.floor(i / 6) * (s[2] + 1.6)], s, color: P.theme.families.conv,
      tex: f ? { front: volTexture(f.front, v), side: volTexture(f.side, v), top: volTexture(f.top, v) } : null,
    }));
    const scene = new DV3D.Scene3D(cv, {
      yaw: -0.6, pitch: 0.5, highlight: P.theme.highlight,
      onHover: (ev, hit) => showTip(ev, [[hit.value === undefined ? "—" : fmt(hit.value, 4),
        `filter ${hit.box.id} · ${{ front: "mean over input channels", side: "mean over kw", top: "mean over kh" }[hit.key]}`]]),
      onLeave: hideTip,
    }).set({ boxes });
    requestAnimationFrame(() => { scene.resize(); scene.fit(14); scene.draw(); });
    return h("div", { class: "vol-panel" }, cv);
  }

  // ----------------------------------------------------------- neuron view
  function renderNeurons() {
    const NV = P.neurons;
    if (!NV || NV.error || !NV.layers) {
      stage.append(h("div", { class: "empty" }, h("div", null,
        h("h2", { text: "No dense chain to draw" }),
        h("div", { text: NV && NV.error ? NV.error : "The neuron view draws chains of fully-connected layers (Linear/Dense). This model has none — use the Graph tab." }))));
      return;
    }
    const L = NV.layers;
    const R = 11, VGAP = 32, COLW = 230;
    const maxShown = Math.max(...L.map((l) => l.shown.length));
    const colH = (maxShown - 1) * VGAP;
    const imgW = NV.image ? 150 : 0;
    const width = imgW + (L.length - 1) * COLW + 220;
    const height = colH + 150;
    const c = makeCanvas(width, height, () => { S.neuron = null; nvHighlight(null); renderPanel(); });
    const vp = c.vp;
    const X = (i) => imgW + 60 + i * COLW;
    const Y = (li, k) => 80 + (colH - (L[li].shown.length - 1) * VGAP) / 2 + k * VGAP;
    const synLayer = s("g"), neurLayer = s("g"), labLayer = s("g");
    const syn = [];
    NV.links.forEach((lk, i) => {
      const W = f32(lk.w);
      let m = 0;
      for (const v of W) m = Math.max(m, Math.abs(v));
      m = m || 1;
      for (let r = 0; r < lk.rows; r++) for (let q = 0; q < lk.cols; q++) {
        const w = W[r * lk.cols + q], t = Math.abs(w) / m;
        const line = s("line", {
          class: "synapse", x1: X(i) + R, y1: Y(i, q), x2: X(i + 1) - R, y2: Y(i + 1, r),
          stroke: w >= 0 ? P.theme.positive : P.theme.negative, "stroke-width": (0.35 + 2.6 * t).toFixed(2),
          opacity: (0.06 + 0.8 * t).toFixed(3),
        });
        synLayer.append(line);
        syn.push({ el: line, from: [i, q], to: [i + 1, r], w });
      }
    });
    const neurons = [];
    L.forEach((l, li) => {
      const vals = l.values;
      const rng = l.range || [0, 1];
      const signed = rng[0] < 0;
      const vmax = Math.max(Math.abs(rng[0]), Math.abs(rng[1])) || 1;
      l.shown.forEach((idx, k) => {
        const v = vals ? vals[k] : null;
        let fill = P.theme.panel;
        if (v !== null && v !== undefined) {
          fill = signed ? rgba(colorOf(v, -vmax, vmax, true)) : rgba([255, 255, 255], clamp(v / vmax) * 0.95 + 0.02);
        }
        const circ = s("circle", { class: "neuron", cx: X(li), cy: Y(li, k), r: R, fill, tabindex: "0", role: "button", "aria-label": `neuron ${idx} of layer ${li}` });
        circ.dataset.fill = fill;
        circ.addEventListener("pointermove", (ev) => {
          nvHighlight([li, k]);
          const rows = [[v === null ? "—" : fmt(v, 4), `${li === 0 ? "input" : "layer " + li} · neuron ${idx}`]];
          if (li > 0 && l.node) rows.push(l.node);
          showTip(ev, rows);
        });
        circ.addEventListener("pointerleave", () => { hideTip(); nvHighlight(S.neuron); });
        const pick = (ev) => { ev.stopPropagation(); S.neuron = [li, k]; nvHighlight(S.neuron); renderPanel(); };
        circ.addEventListener("click", pick);
        circ.addEventListener("keydown", (ev) => { if (ev.key === "Enter") pick(ev); });
        neurLayer.append(circ);
        neurons.push({ el: circ, li, k, idx });
        if (li === L.length - 1) {
          const lab = P.output && P.output.values && P.output.values.length === l.size ? `${idx}  ${P.output.probabilities ? (P.output.values[idx] * 100).toFixed(1) + "%" : fmt(P.output.values[idx])}` : String(idx);
          labLayer.append(s("text", { class: "nv-muted", x: X(li) + R + 10, y: Y(li, k) + 4 }, lab));
        } else if (l.size > l.shown.length) {
          labLayer.append(s("text", { class: "nv-muted", x: X(li) - R - 6, y: Y(li, k) + 4, "text-anchor": "end" }, String(idx)));
        }
      });
      const top = Y(li, 0) - 42;
      const title = li === 0 ? "input" : nodeName(l.node).split(".").slice(-2).join(".") || `layer ${li}`;
      labLayer.append(s("text", { class: "nv-title", x: X(li), y: top, "text-anchor": "middle" }, s("title", null, nodeName(l.node) || title), title));
      const sub = `${l.size.toLocaleString()} ${li === 0 ? "values" : "neurons"}` + (l.size > l.shown.length ? ` · ${l.shown.length} shown` : "");
      labLayer.append(s("text", { class: "nv-muted", x: X(li), y: top + 16, "text-anchor": "middle" }, sub));
      if (l.size > l.shown.length) labLayer.append(s("text", { class: "nv-label", x: X(li), y: Y(li, l.shown.length - 1) + 30, "text-anchor": "middle" }, "⋮"));
    });
    if (NV.image) {
      const img = NV.image;
      const cv = document.createElement("canvas");
      cv.width = img.side; cv.height = img.side;
      const ctx = cv.getContext("2d");
      const data = f32(img.data);
      let lo = Infinity, hi = -Infinity;
      for (const v of data) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      const id = ctx.createImageData(img.side, img.side);
      for (let i = 0; i < data.length; i++) {
        const t = hi > lo ? (data[i] - lo) / (hi - lo) : 0;
        const g = Math.round(20 + 235 * t);
        id.data.set([g, g, g, 255], i * 4);
      }
      ctx.putImageData(id, 0, 0);
      const size = 112;
      const y0 = 80 + colH / 2 - size / 2;
      vp.append(s("image", { href: crispURI(cv), x: 20, y: y0, width: size, height: size, style: "image-rendering: pixelated" }),
        s("rect", { x: 20, y: y0, width: size, height: size, fill: "none", stroke: P.theme.panelEdge, rx: 3 }),
        s("text", { class: "nv-muted", x: 20 + size / 2, y: y0 + size + 18, "text-anchor": "middle" }, `input ${img.side}×${img.side}`));
    }
    vp.append(synLayer, neurLayer, labLayer);
    stage.append(c.svg);
    const bar = h("div", { class: "toolbar" },
      h("span", { class: "nv-key" }, h("i", { class: "k-pos" }), "positive weight", h("i", { class: "k-neg" }), "negative",
        h("i", { class: "k-on" }), "active neuron"),
      h("div", { class: "spacer" }),
      h("button", { class: "btn primary", text: "▶ Propagate", title: "Activations flow from layer to layer  (F)", onclick: () => propagate() }));
    stage.append(bar, zoomControls(c));
    requestAnimationFrame(() => { c.fit(); nvHighlight(S.neuron); });
    NVstate = { syn, neurons, layers: L };
  }
  let NVstate = null;
  function nvHighlight(sel) {
    if (!NVstate) return;
    const root = NVstate.syn.length ? NVstate.syn[0].el.parentNode.parentNode : null;
    for (const n of NVstate.neurons) n.el.classList.toggle("sel", !!(S.neuron && n.li === S.neuron[0] && n.k === S.neuron[1]));
    if (!root) return;
    root.classList.toggle("nv-dim", !!sel);
    for (const e of NVstate.syn) {
      const on = !!sel && ((e.from[0] === sel[0] && e.from[1] === sel[1]) || (e.to[0] === sel[0] && e.to[1] === sel[1]));
      e.el.classList.toggle("hl", on);
    }
  }
  function propagate() {
    if (!NVstate) return;
    const { syn, neurons, layers } = NVstate;
    for (const n of neurons) if (n.li > 0) n.el.setAttribute("fill", P.theme.panel);
    const step = 650;
    for (let li = 0; li < layers.length - 1; li++) {
      setTimeout(() => {  // the 40 strongest synapses of this layer carry a flash
        syn.filter((e) => e.from[0] === li).sort((a, b) => Math.abs(b.w) - Math.abs(a.w)).slice(0, 40)
          .forEach((e) => flashAlong(e.el, step * 1.3));
      }, li * step);
      setTimeout(() => {
        for (const n of neurons) if (n.li === li + 1) n.el.setAttribute("fill", n.el.dataset.fill);
      }, li * step + step * 0.8);
    }
  }

  // --------------------------------------------------------- equations view
  let eqRows = [];
  function renderEquations() {
    const E = P.equations || {};
    const depths = Object.keys(E.byDepth || {});
    const modes = [["system", "layer by layer"], ["expanded", "fully substituted"], ["compact", "as one function"]];
    const bar = h("div", { class: "toolbar" },
      h("div", { class: "seg", role: "group", "aria-label": "equation form" }, ...modes.map(([m, lab]) => h("button", {
        "aria-pressed": String(S.eqMode === m), text: lab, onclick: () => { S.eqMode = m; renderStage(); },
      }))),
      depths.length && S.eqMode === "system" ? h("div", { class: "seg", role: "group", "aria-label": "unroll depth" }, h("span", { class: "seg-lab", text: "detail" }),
        ...["full", ...depths].map((d) => h("button", {
          "aria-pressed": String(S.eqDepth === d), text: detailLabel({ depth: d === "full" ? null : Number(d) }),
          onclick: () => { S.eqDepth = d; renderStage(); },
        }))) : null);
    let eq = S.eqMode === "system" && S.eqDepth !== "full" ? E.byDepth[S.eqDepth] : (E.modes || {})[S.eqMode];
    const view = h("div", { class: "eqview" });
    if (!eq || eq.error) {
      view.append(h("div", { class: "hint", text: eq && eq.error ? eq.error : "No equations available." }));
    } else {
      const grid = h("div", { class: "eq-grid" });
      eqRows = [];
      for (const ln of eq.lines) {
        const row = h("div", { class: "eq-row" });
        const lhs = tex(h("div", { class: "lhs" }), ln.lhs, false);
        const eqs = tex(h("div"), "=", false);
        const rhs = tex(h("div", { class: "rhs" }), ln.rhs, false);
        if (ln.note) rhs.append(h("span", { class: "note", text: ln.note }));
        row.append(lhs, eqs, rhs);
        const go = () => { if (ln.node !== null && ln.node !== undefined) select(ln.node); };
        lhs.addEventListener("click", go);
        rhs.addEventListener("click", go);
        grid.append(row);
        eqRows.push({ row, node: ln.node });
      }
      view.append(grid);
      if ((eq.definitions && eq.definitions.length) || (eq.legend && eq.legend.length)) {
        const where = h("div", { class: "where" }, h("h4", { text: "where" }));
        for (const d of eq.definitions || []) where.append(tex(h("div", { class: "def" }), d.latex, false));
        for (const d of eq.legend || []) where.append(tex(h("div", { class: "def" }), d.latex, false));
        view.append(where);
      }
      highlightEqRows();
    }
    stage.append(bar, view);
  }
  function highlightEqRows() {
    for (const r of eqRows) r.row.classList.toggle("sel", r.node !== null && r.node === S.selected);
  }

  // ------------------------------------------------------------ monitor view
  function renderMonitor() {
    const view = h("div", { class: "tableview" });
    if (P.training) trainingBlock(view, P.training);
    if (P.lens) lensBlock(view, P.lens);
    if (P.gradflow) gradFlowBlock(view, P.gradflow);
    const M = P.monitor;
    if (M && Object.keys(M).length) {
      const stats = Object.keys(Object.values(M)[0].history || {});
      if (!stats.includes(S.monStat)) S.monStat = stats[0] || "mean";
      view.append(h("h3", { class: "sec", text: "Monitored layers — history across calls" }));
      view.append(h("div", { class: "seg", role: "group", "aria-label": "statistic" }, ...stats.map((st) => h("button", {
        "aria-pressed": String(S.monStat === st), text: st, onclick: () => { S.monStat = st; renderStage(); },
      }))));
      const tbl = h("table", { class: "dt" });
      tbl.append(h("thead", null, h("tr", null, h("th", { text: "node" }), h("th", { text: "calls" }),
        h("th", { class: "l", text: `${S.monStat} over time` }), h("th", { text: "first" }), h("th", { text: "last" }), h("th", { text: "min" }), h("th", { text: "max" }))));
      const tb = h("tbody");
      const ids = Object.keys(M);
      if (!S.monNode || !M[S.monNode]) S.monNode = ids[0];
      for (const id of ids) {
        const series = (M[id].history || {})[S.monStat] || [];
        const tr = h("tr", { class: id === S.monNode ? "sel" : "" },
          h("td", { text: id }), h("td", { text: String(M[id].calls) }),
          h("td", { class: "l" }, sparkline(series, 140, 26)),
          h("td", { text: fmt(series[0]) }), h("td", { text: fmt(series[series.length - 1]) }),
          h("td", { text: fmt(Math.min(...series)) }), h("td", { text: fmt(Math.max(...series)) }));
        tr.addEventListener("click", () => { S.monNode = id; renderStage(); });
        tb.append(tr);
      }
      tbl.append(tb);
      view.append(tbl);
      const series = ((M[S.monNode] || {}).history || {})[S.monStat] || [];
      view.append(h("h3", { class: "sec", text: `${S.monNode} · ${S.monStat} per recorded call` }), lineChart(series, 760, 220, S.monStat));
    }
    if (P.trace && P.trace.table) {
      view.append(h("h3", { class: "sec", text: "Every layer, for this input" }));
      const cols = ["mean", "std", "min", "max", "sparsity"];
      const tbl = h("table", { class: "dt" });
      tbl.append(h("thead", null, h("tr", null, h("th", { text: "layer" }), h("th", { class: "l", text: "kind" }), h("th", { class: "l", text: "shape" }),
        h("th", { class: "l", text: "values" }), ...cols.map((c) => h("th", { text: c === "sparsity" ? "zeros" : c })))));
      const tb = h("tbody");
      for (const r of P.trace.table) {
        const hist = P.acts && P.acts[r.node] && P.acts[r.node].hist;
        const tr = h("tr", { class: r.node === S.selected ? "sel" : "" }, h("td", { text: nodeName(r.node), title: r.node }), h("td", { class: "l", text: r.kind }),
          h("td", { class: "l", text: shapeStr(r.shape) }), h("td", { class: "l" }, hist ? miniHist(hist) : null),
          ...cols.map((c) => h("td", { text: c === "sparsity" ? (r[c] * 100).toFixed(1) + "%" : fmt(r[c]) })));
        tr.addEventListener("click", () => select(r.node));
        tb.append(tr);
      }
      tbl.append(tb);
      view.append(tbl);
    }
    if (!(M && Object.keys(M).length) && !(P.trace && P.trace.table) && !P.training && !P.gradflow && !P.lens) {
      view.append(h("div", { class: "empty" }, h("div", { class: "empty-help" }, h("h2", { text: "Nothing recorded yet" }),
        h("p", { text: "This view shows how things change over time: training curves, per-layer statistics and gradient flow. Fill it with any of:" }),
        h("ul", null,
          h("li", null, h("code", { text: "net.serve(x, train_step=fn)" }), " — train from the page, with ▶ Train / Step buttons"),
          h("li", null, h("code", { text: "dv.watch(model, x)" }), " or a ", h("code", { text: "dv.TrainingLog" }), " in your own training loop"),
          h("li", null, h("code", { text: "net.show(x, monitor=net.monitor(\"fc*\"))" }), " — statistics of the layers you name"),
          h("li", null, h("code", { text: "net.show(x, grads=True)" }), " — how gradients flow back through the layers")))));
    }
    stage.append(view);
  }

  // ------------------------------------------------------------- diffusion
  let DF_TIMER = null;
  function renderDiffusion() {
    const D = P.diffusion;
    if (!D) return;
    if (DF_TIMER) { clearInterval(DF_TIMER); DF_TIMER = null; }
    const F = D.frames, nF = F.length, B = F[0].x.length;
    if (!(S.dfFrame >= 0 && S.dfFrame < nF)) S.dfFrame = 0;
    const view = h("div", { class: "dfview" });
    const tLab = h("b", { class: "df-t" });
    const slider = h("input", { type: "range", min: 0, max: nF - 1, step: 1, value: S.dfFrame, class: "df-slider", "aria-label": "denoising step" });
    const play = h("button", { class: "btn primary", text: "▶" });
    view.append(h("div", { class: "toolbar df-bar" }, play, slider, tLab,
      h("span", { class: "df-meta", text: `${D.sampler.toUpperCase()} · T = ${D.T}` })));
    const rows = [["x", "xₜ"], ["x0", "x̂₀"], ["eps", "ε̂"]];
    const grid = h("div", { class: "df-grid", style: { gridTemplateColumns: `28px repeat(${Math.min(B, 6)}, minmax(0, 1fr))` } });
    const imgs = {};
    for (const [key, name] of rows) {
      grid.append(h("div", { class: "df-rowlab", text: name }));
      imgs[key] = [];
      for (let b = 0; b < Math.min(B, 6); b++) {
        const im = h("img", { class: "df-img" + (key === "x" ? " big" : ""), alt: `${name} of sample ${b}` });
        imgs[key].push(im);
        grid.append(im);
      }
    }
    // schedule: ᾱ_t over t, with the current step marked
    const ab = D.alphasBar, W = 260, Hh = 120;
    const sch = s("svg", { class: "chart df-sched", viewBox: `0 0 ${W} ${Hh}`, width: W, height: Hh, role: "img", "aria-label": "noise schedule" });
    const X = (t) => 8 + (t / (D.T - 1)) * (W - 16), Y = (v) => Hh - 18 - v * (Hh - 30);
    sch.append(s("path", { class: "line", d: ab.map((v, i) => `${i ? "L" : "M"}${X((i / (ab.length - 1)) * (D.T - 1))},${Y(v)}`).join(" ") }),
      s("path", { class: "line snr", d: ab.map((v, i) => `${i ? "L" : "M"}${X((i / (ab.length - 1)) * (D.T - 1))},${Y(Math.sqrt(1 - v))}`).join(" ") }),
      s("text", { class: "tick", x: 8, y: Hh - 4 }, "t = 0"), s("text", { class: "tick", x: W - 8, y: Hh - 4, "text-anchor": "end" }, `t = ${D.T - 1}`),
      s("text", { class: "df-key", x: 12, y: Y(ab[0]) - 6 }, "signal √ᾱₜ"),
      s("text", { class: "df-key noise", x: W - 44, y: Y(Math.sqrt(1 - ab[ab.length - 1])) + 16, "text-anchor": "end" }, "noise √(1−ᾱₜ)"));
    const mark = s("line", { class: "xhair", y1: 8, y2: Hh - 18 });
    sch.append(mark);
    // the whole reverse trajectory of sample 0, and the forward process for comparison
    const strip = h("div", { class: "df-strip" });
    const step = Math.max(1, Math.floor(nF / 12));
    const thumbs = [];
    for (let i = 0; i < nF; i += step) {
      const im = h("img", { class: "df-thumb", src: F[i].x[0], title: `t = ${F[i].t}`, alt: `t ${F[i].t}` });
      im.addEventListener("click", () => show(i));
      thumbs.push([i, im]);
      strip.append(im);
    }
    const last = h("img", { class: "df-thumb final", src: D.final[0], title: "generated", alt: "generated sample" });
    strip.append(h("span", { class: "df-arrow", text: "→" }), last);
    const fwd = D.forward ? h("div", { class: "df-strip fwd" }, ...D.forward.map((f) => h("img", { class: "df-thumb", src: f.x, title: `q(xₜ | x₀), t = ${f.t}`, alt: `forward t ${f.t}` }))) : null;
    const side = h("div", { class: "df-side" }, sch,
      h("div", { class: "df-cap", text: "generated" }), h("div", { class: "df-final" }, ...D.final.slice(0, 6).map((u) => h("img", { class: "df-img", src: u, alt: "sample" }))));
    view.append(h("div", { class: "df-main" }, grid, side),
      h("div", { class: "df-cap", text: "reverse  ·  noise → image" }), strip,
      fwd ? h("div", { class: "df-cap", text: "forward  ·  image → noise" }) : null, fwd);
    function show(i) {
      S.dfFrame = i;
      const f = F[i];
      for (const [key] of rows) imgs[key].forEach((im, b) => { im.src = f[key][b]; });
      tLab.textContent = `t = ${f.t}`;
      slider.value = i;
      mark.setAttribute("x1", X(f.t)); mark.setAttribute("x2", X(f.t)); mark.setAttribute("opacity", 1);
      thumbs.forEach(([j, im]) => im.classList.toggle("on", j <= i && i < j + step));
    }
    slider.addEventListener("input", () => show(Number(slider.value)));
    play.addEventListener("click", () => {
      if (DF_TIMER) { clearInterval(DF_TIMER); DF_TIMER = null; play.textContent = "▶"; return; }
      if (S.dfFrame >= nF - 1) show(0);
      play.textContent = "❚❚";
      DF_TIMER = setInterval(() => {
        if (S.dfFrame >= nF - 1 || S.tab !== "diffusion") { clearInterval(DF_TIMER); DF_TIMER = null; play.textContent = "▶"; return; }
        show(S.dfFrame + 1);
      }, 110);
    });
    show(S.dfFrame);
    stage.append(view);
  }

  // --------------------------------------------------------- inside attention
  // An image of a heat map (rows × cols), optionally transposed; masked cells drawn as the panel colour.
  function heatURI(hm, transpose = false, mask = null, gamma = 1) {
    const data = f32(hm.data), R = hm.rows, C = hm.cols;
    const mk = mask ? f32(mask.data) : null;
    const cv = document.createElement("canvas");
    cv.width = transpose ? R : C; cv.height = transpose ? C : R;
    const ctx = cv.getContext("2d"), img = ctx.createImageData(cv.width, cv.height);
    const bg = hexRgb(P.theme.panel2 || "#222222");
    for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
      const i = r * C + c, o = transpose ? c * R + r : i;
      const val = gamma === 1 ? data[i] : Math.pow(Math.max(0, (data[i] - hm.vmin) / ((hm.vmax - hm.vmin) || 1)), gamma);
      const col = mk && mk.length === data.length && mk[i] > 0.5 ? bg
        : gamma === 1 ? colorOf(val, hm.vmin, hm.vmax, hm.diverging) : colorOf(val, 0, 1, false);
      img.data.set([col[0], col[1], col[2], 255], o * 4);
    }
    ctx.putImageData(img, 0, 0);
    return crispURI(cv);
  }
  // Safari ignores image-rendering on SVG <image>, so small maps are upscaled (nearest-neighbour) before embedding
  function crispURI(cv, target = 256) {
    const k = Math.min(Math.ceil(target / Math.max(cv.width, cv.height, 1)), Math.floor(2048 / Math.max(cv.width, cv.height, 1)));
    if (k <= 1) return cv.toDataURL();
    const big = document.createElement("canvas");
    big.width = cv.width * k; big.height = cv.height * k;
    const bctx = big.getContext("2d");
    bctx.imageSmoothingEnabled = false;
    bctx.drawImage(cv, 0, 0, big.width, big.height);
    return big.toDataURL();
  }
  function renderAttention() {
    const A = P.anatomy || [];
    if (!A.length) return;
    if (!A.some((a) => a.id === S.attLayer)) S.attLayer = A[0].id;
    const L = A.find((a) => a.id === S.attLayer);
    if (!(S.attHead < L.shown)) S.attHead = 0;
    const hd = S.attHead;
    const view = h("div", { class: "attview" });
    view.append(h("div", { class: "toolbar att-bar" },
      h("div", { class: "seg", role: "group", "aria-label": "attention layer" }, ...A.map((a, i) => h("button", {
        "aria-pressed": String(a.id === L.id), text: A.length > 1 ? `layer ${i}` : nodeName(a.name || a.id), title: a.name || nodeName(a.id),
        onclick: () => { S.attLayer = a.id; renderStage(); } }))),
      h("div", { class: "seg", role: "group", "aria-label": "head" }, ...Array.from({ length: L.shown }, (_, i) => h("button", {
        "aria-pressed": String(i === hd), text: `head ${i}`, onclick: () => { S.attHead = i; renderStage(); } }))),
      LIVE ? h("button", { class: "btn danger", title: "an undoable edit: see how the output changes (white ticks = before)",
        text: `Switch head ${hd} off`, onclick: () => api("/api/head", { node: L.id, head: hd }) }) : null));
    const q = L.q[hd], k = L.k[hd], v = L.v[hd], sc = L.scores[hd], pr = L.probs[hd], o = L.out[hd];
    const T = pr.rows, Sk = pr.cols;                         // shown grid (may be block-averaged)
    const c = Math.max(4, Math.min(20, Math.floor(300 / Math.max(T, Sk))));
    const dW = 96, gap = 16, arrowW = 78, top = 34;
    const kH = dW, vH = Sk * c;
    const y0 = top + Math.max(kH, vH) + gap + 8;               // main row
    const xQ = 40, xS = xQ + dW + gap, xP = xS + Sk * c + arrowW, xO = xP + Sk * c + gap;
    const W = xO + dW + 30, H = y0 + T * c + 40;
    const svg = s("svg", { class: "att-svg", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `inside ${L.id}` });
    const img = (uri, x, y, w, hh) => s("image", { href: uri, x, y, width: w, height: hh, preserveAspectRatio: "none", style: "image-rendering:pixelated" });
    const frame = (x, y, w, hh) => s("rect", { x, y, width: w, height: hh, class: "att-frame" });
    const lab = (x, y, t, anchor = "middle") => s("text", { x, y, class: "att-lab", "text-anchor": anchor }, t);
    const masked = L.masked && L.masked.rows === sc.rows ? L.masked : null;
    svg.append(
      img(heatURI(q), xQ, y0, dW, T * c), frame(xQ, y0, dW, T * c), lab(xQ + dW / 2, y0 + T * c + 18, "Q"),
      img(heatURI(k, true), xS, y0 - gap - kH, Sk * c, kH), frame(xS, y0 - gap - kH, Sk * c, kH), lab(xS - 10, y0 - gap - kH / 2 + 5, "Kᵀ", "end"),
      img(heatURI(sc, false, masked), xS, y0, Sk * c, T * c), frame(xS, y0, Sk * c, T * c), lab(xS + Sk * c / 2, y0 + T * c + 18, "QKᵀ / √d"),
      img(heatURI(pr, false, null, 0.5), xP, y0, Sk * c, T * c), frame(xP, y0, Sk * c, T * c), lab(xP + Sk * c / 2, y0 + T * c + 18, "P"),
      img(heatURI(v), xO, y0 - gap - vH, dW, vH), frame(xO, y0 - gap - vH, dW, vH), lab(xO + dW + 10, y0 - gap - vH / 2 + 5, "V", "start"),
      img(heatURI(o), xO, y0, dW, T * c), frame(xO, y0, dW, T * c), lab(xO + dW / 2, y0 + T * c + 18, "P·V"),
    );
    const ax = xS + Sk * c + 10, ay = y0 + (T * c) / 2;       // softmax arrow
    svg.append(s("path", { d: `M${ax},${ay} L${xP - 12},${ay}`, class: "att-arrow", "marker-end": "url(#att-tip)" }),
      lab((ax + xP - 12) / 2, ay - 8, "softmax"));
    svg.prepend(s("defs", null, s("marker", { id: "att-tip", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto" },
      s("path", { d: "M0,1 L9,5 L0,9 z", fill: P.theme.muted }))));
    const toks = L.tokens && T === L.T ? L.tokens : null;
    if (toks && T <= 40) {
      toks.forEach((t, i) => {
        svg.append(s("text", { x: xQ - 6, y: y0 + i * c + c / 2 + 3.5, class: "att-tok", "text-anchor": "end" }, t.replace(/ /g, "·").slice(0, 10)));
        svg.append(s("text", { x: 0, y: 0, class: "att-tok", transform: `translate(${xS + i * c + c / 2 + 3.5},${y0 - gap - kH - 6}) rotate(-60)` }, t.replace(/ /g, "·").slice(0, 10)));
      });
    }
    // hover: one query row lights up everywhere; V's rows dim by how little that query attends to them
    const hl = s("g", { class: "att-hl" });
    svg.append(hl);
    const probs = f32(pr.data);
    let focusHook = null;
    const focus = (i, j) => {
      if (focusHook) focusHook(i);
      hl.replaceChildren();
      const row = (x, w) => s("rect", { x, y: y0 + i * c, width: w, height: c, class: "att-row" });
      hl.append(row(xQ, dW), row(xS, Sk * c), row(xP, Sk * c), row(xO, dW));
      if (j !== null) {
        hl.append(s("rect", { x: xS + j * c, y: y0 - gap - kH, width: c, height: kH + gap + T * c, class: "att-col" }),
          s("rect", { x: xP + j * c, y: y0, width: c, height: T * c, class: "att-col" }));
      }
      const pmax = Math.max(...Array.from({ length: Sk }, (_, jj) => probs[i * Sk + jj]), 1e-9);
      for (let jj = 0; jj < Sk; jj++) {
        const w = probs[i * Sk + jj] / pmax;
        hl.append(s("rect", { x: xO, y: y0 - gap - vH + jj * c, width: dW, height: c, class: "att-dim", opacity: (1 - w) * 0.82 }));
      }
    };
    const pick = (ev, x) => {
      const r = svg.getBoundingClientRect(), k2 = W / r.width;
      const px = (ev.clientX - r.left) * k2, py = (ev.clientY - r.top) * k2;
      const i = Math.floor((py - y0) / c), j = Math.floor((px - x) / c);
      return [i, j];
    };
    for (const x of [xS, xP]) {
      const hit = s("rect", { x, y: y0, width: Sk * c, height: T * c, fill: "transparent", style: "cursor:crosshair" });
      hit.addEventListener("pointermove", (ev) => {
        const [i, j] = pick(ev, x);
        if (i < 0 || j < 0 || i >= T || j >= Sk) return;
        focus(i, j);
        const nm = (t) => (toks ? ` “${toks[t]}”` : "");
        showTip(ev, [[fmt(probs[i * Sk + j], 3), `attention · query ${i}${nm(i)} → key ${j}${nm(j)}`],
          [fmt(f32(sc.data)[i * Sk + j], 3), "score"]]);
      });
      hit.addEventListener("pointerleave", () => { hideTip(); });
      svg.append(hit);
    }
    // image tokens (a square grid, optionally after a class token): fold the hovered query's row onto the grid
    const side = Math.round(Math.sqrt(Sk)), side1 = Math.round(Math.sqrt(Sk - 1));
    const off = side * side === Sk ? 0 : side1 * side1 === Sk - 1 ? 1 : -1;
    if (off >= 0 && !pr.downsampled && !toks && Sk >= 9) {
      const n = off ? side1 : side, box = 150, cs = box / n, gx = xP + (Sk * c - box) / 2, gy = 24;
      const inset = s("g"), cells = [];
      for (let r = 0; r < n; r++) for (let q2 = 0; q2 < n; q2++) {
        const rc = s("rect", { x: gx + q2 * cs, y: gy + r * cs, width: cs - 0.5, height: cs - 0.5 });
        cells.push(rc); inset.append(rc);
      }
      const qmark = s("rect", { class: "att-row", width: cs, height: cs });
      const under = P.inputImage;  // the patches are an image's: draw the attention over it
      if (under) inset.prepend(s("image", { href: under.uri, x: gx, y: gy, width: box, height: box, preserveAspectRatio: "none" }));
      inset.append(qmark, frame(gx, gy, box, box), lab(gx + box / 2, gy + box + 16, "where it looks"));
      svg.append(inset);
      const prevFocus = focus;
      focusHook = (i) => {
        const row = Array.from({ length: n * n }, (_, jj) => probs[i * Sk + off + jj]);
        const m = Math.max(...row, 1e-9);
        row.forEach((p2, jj) => cells[jj].setAttribute("fill", under ? rgba(SEQ(Math.sqrt(p2 / m)), 0.12 + 0.72 * Math.sqrt(p2 / m)) : rgba(SEQ(Math.sqrt(p2 / m)))));
        const qi = i - off;
        if (qi >= 0 && qi < n * n) { qmark.setAttribute("x", gx + (qi % n) * cs); qmark.setAttribute("y", gy + Math.floor(qi / n) * cs); qmark.setAttribute("opacity", 1); }
        else qmark.setAttribute("opacity", 0);
        return prevFocus;
      };
    }
    focus(T - 1, null);                                        // start on the last query
    view.append(h("div", { class: "att-wrap" }, svg));
    acrossLayers(view, A);
    stage.append(view);
  }

  // ------------------------------------------------------ across attention layers: head importance, rollout
  function acrossLayers(view, A) {
    const I = P.interpret;
    if (!I) return;
    const row = h("div", { class: "across" });
    if (I.heads && I.heads.effect.length) {
      const E = I.heads.effect, nL = E.length, nH = I.heads.heads, cs = Math.max(14, Math.min(30, Math.floor(420 / nH)));
      const labW = 70, W = labW + nH * cs + 4, Ht = 18 + nL * cs + 4;
      const vmin = Math.min(...E.flat()), vmax = Math.max(vmin + 1e-12, ...E.flat());  // the grid's own range: differences show
      const svg = s("svg", { class: "heads-grid", viewBox: `0 0 ${W} ${Ht}`, width: W, height: Ht, role: "img", "aria-label": "head importance" });
      for (let hh = 0; hh < nH; hh++) svg.append(s("text", { x: labW + hh * cs + cs / 2, y: 12, class: "att-tok", "text-anchor": "middle" }, String(hh)));
      E.forEach((r, li) => {
        const id = I.heads.layers[li], an = A.find((a) => a.id === id);
        svg.append(s("text", { x: labW - 8, y: 18 + li * cs + cs / 2 + 4, class: "att-tok", "text-anchor": "end" }, `layer ${li}`));
        r.forEach((v, hh) => {
          const sel = id === S.attLayer && hh === S.attHead;
          const cell = s("rect", { x: labW + hh * cs + 1, y: 18 + li * cs + 1, width: cs - 2, height: cs - 2, rx: 3,
            fill: rgba(SEQ((v - vmin) / (vmax - vmin))), class: "head-cell" + (sel ? " sel" : "") });
          cell.addEventListener("pointermove", (ev) => showTip(ev, [[fmt(v, 3), `output change with head ${hh} of layer ${li} off`], nodeName(id)]));
          cell.addEventListener("pointerleave", hideTip);
          cell.addEventListener("click", () => {
            if (!an) return toast("that layer isn't shown in detail (only the first layers are)");
            if (hh >= an.shown) return toast(`only the first ${an.shown} heads are drawn in detail`);
            S.attLayer = id; S.attHead = hh; renderStage();
          });
          svg.append(cell);
        });
      });
      row.append(h("div", { class: "across-card" }, h("h3", { class: "sec", text: "Which heads matter" }),
        h("div", { class: "hint", text: "how much the output changes when one head is switched off · click a head to look inside" }), svg,
        colorbar({ vmin, vmax, diverging: false, orig: [] })));
    }
    if (I.rollout) {
      const R = I.rollout, toks = R.tokens, from = R.from, m = Math.max(...from, 1e-12);
      const who = toks ? `“${(toks[R.row] || "").trim() || R.row}”` : `position ${R.row}`;
      const n1 = Math.round(Math.sqrt(from.length)), n2 = Math.round(Math.sqrt(from.length - 1));
      const grid = !toks && P.inputImage ? (n1 * n1 === from.length ? [n1, 0] : n2 * n2 === from.length - 1 ? [n2, 1] : null) : null;
      const body = grid ? rolloutImage(from, grid[0], grid[1]) : toks
        ? h("div", { class: "tok-chips" }, ...toks.map((t, i) => h("span", { class: "tok", title: `${fmt(from[i], 3)} of position ${i}`,
          text: t.replace(/ /g, "·") || "∅", style: { background: rgba(SEQ(Math.sqrt(from[i] / m)), 0.85), color: from[i] / m > 0.45 ? "#000" : "" } })))
        : heatCanvas({ rows: 1, cols: from.length, data: null, _vals: from }, 380, 26);
      row.append(h("div", { class: "across-card" }, h("h3", { class: "sec", text: "Where the information comes from" }),
        h("div", { class: "hint", text: `attention rollout through all ${R.layers} layers (residual paths included): what ${who} draws on` }),
        body, h("div", { class: "p-sub", text: "every position × every token" }), heatCanvas(R.matrix, 300, 300), colorbar(R.matrix)));
    }
    if (row.children.length) view.append(row);
  }

  function rolloutImage(from, n, off) {  // image patches: fold the rollout row onto the picture
    const box = 240, cs = box / n, I = P.inputImage;
    const vals = from.slice(off), m = Math.max(...vals, 1e-12), lo = Math.min(...vals);  // own range: differences show
    const svg = s("svg", { class: "patch-svg", viewBox: `0 0 ${box} ${box}`, width: box, height: box, role: "img", "aria-label": "rollout on the image" });
    svg.append(s("image", { href: I.uri, x: 0, y: 0, width: box, height: box, preserveAspectRatio: "none" }));
    vals.forEach((v, i) => {
      const t = m - lo > 1e-12 ? (v - lo) / (m - lo) : 0.5, r = s("rect", { x: (i % n) * cs, y: Math.floor(i / n) * cs, width: cs, height: cs, fill: rgba(SEQ(t), 0.1 + 0.75 * t) });
      r.addEventListener("pointermove", (ev) => showTip(ev, [[fmt(v, 3), `patch ${i} (row ${Math.floor(i / n)}, col ${i % n})`]]));
      r.addEventListener("pointerleave", hideTip);
      svg.append(r);
    });
    return svg;
  }
  // ------------------------------------------------------------- architecture views
  function stripRow(label, hm, maxH = 70) {
    return h("div", { class: "arch-row" }, h("div", { class: "p-sub", text: label }), heatCanvas(hm, 380, maxH));
  }
  function recurrentSection(R) {
    const L = R.layers;
    if (!(S.recLayer < L.length)) S.recLayer = 0;
    const cur = L[S.recLayer];
    const kids = [];
    if (L.length > 1) kids.push(h("div", { class: "seg", role: "group", "aria-label": "layer" }, ...L.map((l, i) => h("button", {
      "aria-pressed": String(i === S.recLayer), text: `layer ${l.layer} ${l.direction}`, onclick: () => { S.recLayer = i; renderPanel(); } }))));
    kids.push(stripRow("hₜ — the hidden state", cur.h, 90));
    if (cur.c) kids.push(stripRow("cₜ — the cell (long-term memory)", cur.c, 90));
    for (const g of cur.gates) kids.push(stripRow(g.name + (g.name.includes("gate") ? " — 0 closed · 1 open" : ""), g.heat, 60));
    kids.push(h("div", { class: "hint", text: `time → · one row per unit · every gate recomputed from the layer's weights (reproduces its output to ${fmt(R.match, 2)})` }));
    return section("Unrolled over time", `${R.kind} · ${R.steps} steps · ${R.hidden} units`, ...kids);
  }
  function routingSection(r) {
    const P_ = f32(r.probs.data), T = r.probs.rows, E = r.probs.cols;
    const cw = Math.max(10, Math.min(30, Math.floor(300 / E))), ch = Math.max(8, Math.min(16, Math.floor(420 / T)));
    const labW = r.tokenNames ? 64 : 30, W = labW + E * cw + 4, H = 16 + T * ch + 4;
    const svg = s("svg", { class: "moe-grid", viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": "expert routing" });
    for (let e = 0; e < E; e++) if (E <= 32 || e % 4 === 0) svg.append(s("text", { x: labW + e * cw + cw / 2, y: 11, class: "att-tok", "text-anchor": "middle" }, String(e)));
    for (let t = 0; t < T; t++) {
      const nm = r.tokenNames ? (r.tokenNames[t] || "").replace(/ /g, "·").slice(0, 8) : String(t);
      if (ch >= 9) svg.append(s("text", { x: labW - 6, y: 16 + t * ch + ch / 2 + 3.5, class: "att-tok", "text-anchor": "end" }, nm));
      const chosen = new Set(r.chosen[t] || []);
      for (let e = 0; e < E; e++) {
        const p = P_[t * E + e];
        const cell = s("rect", { x: labW + e * cw + 0.5, y: 16 + t * ch + 0.5, width: cw - 1, height: ch - 1, rx: 2, fill: rgba(SEQ(Math.sqrt(p))) });
        cell.addEventListener("pointermove", (ev) => showTip(ev, [[fmt(p, 3), `token ${t}${r.tokenNames ? ` “${r.tokenNames[t]}”` : ""} → expert ${e}`], chosen.has(e) ? "chosen" : "not chosen"]));
        cell.addEventListener("pointerleave", hideTip);
        svg.append(cell);
        if (chosen.has(e)) svg.append(s("circle", { cx: labW + e * cw + cw / 2, cy: 16 + t * ch + ch / 2, r: Math.min(cw, ch) * 0.22, class: "moe-pick" }));
      }
    }
    const loadMax = Math.max(1, ...r.load);
    const load = h("div", { class: "moe-load" }, ...r.load.map((c, e) => h("div", { class: "moe-bar", title: `expert ${e}: ${c} of ${T * r.k} slots` },
      h("i", { style: { height: `${Math.round(100 * c / loadMax)}%` } }), h("span", { text: String(e) }))));
    return section("Routing — which experts each token uses", `${E} experts · top-${r.k}`, svg,
      h("div", { class: "p-sub", text: "load: tokens sent to each expert" }), load,
      h("div", { class: "hint", text: "shade: the router's probability · dot: an expert the token is actually sent to" }));
  }
  function ssmSection(m) {
    const vals = m.perToken, mx = Math.max(...vals, 1e-12), mn = Math.min(...vals);
    const hgt = (v) => (mx - mn > 1e-9 * mx ? 12 + 88 * (v - mn) / (mx - mn) : 60);  // the strip's own range
    const bars = h("div", { class: "ssm-strip" }, ...vals.map((v, i) => h("div", { class: "ssm-bar",
      title: `${fmt(v, 3)} · token ${i}${m.tokenNames ? ` “${m.tokenNames[i]}”` : ""}` }, h("i", { style: { height: `${Math.round(hgt(v))}%` } }))));
    return section("State updates — Δ per token", "selective scan", bars,
      m.tokenNames ? h("div", { class: "tok-chips" }, ...m.tokenNames.slice(0, 64).map((t) => h("span", { class: "tok", text: t.replace(/ /g, "·") || "∅" }))) : null,
      h("div", { class: "p-sub", text: `mean Δ per token · ${fmt(mn, 3)} – ${fmt(mx, 3)}` }),
      stripRow("Δ per channel (token × channel)", m.delta, 140),
      h("div", { class: "hint", text: "Δ = softplus(dt_proj(x)): large → this token overwrites the state, small → the state carries on" }));
  }
  function graphInputSection(G, layerId) {
    const L = G.layers;
    if (layerId) S.graphLayer = layerId;
    const cur = L.find((l) => l.id === S.graphLayer) || L[L.length - 1];
    const W = 380, H = 300, pad = 12, rN = Math.max(2.5, Math.min(7, 90 / Math.sqrt(G.n)));
    const X = (p) => pad + p[0] * (W - 2 * pad), Y = (p) => pad + p[1] * (H - 2 * pad);
    const svg = s("svg", { class: "gnn-svg", viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": "input graph" });
    const eg = s("g", { class: "gnn-edges" });
    for (const [a, b] of G.edges) eg.append(s("line", { x1: X(G.pos[a]), y1: Y(G.pos[a]), x2: X(G.pos[b]), y2: Y(G.pos[b]) }));
    svg.append(eg);
    const vals = cur.values, lo = Math.min(...vals), hi = Math.max(...vals);
    G.pos.forEach((p, i) => {
      const t = hi > lo ? (vals[i] - lo) / (hi - lo) : 0.5;
      const c = s("circle", { cx: X(p), cy: Y(p), r: rN, fill: rgba(SEQ(t)), class: "gnn-node" });
      c.addEventListener("pointermove", (ev) => showTip(ev, [[fmt(vals[i], 3), `‖h‖ of node ${i} after ${cur.id === "input" ? "the input" : nodeName(cur.id)}`],
        ...(cur.argmax ? [[String(cur.argmax[i]), "predicted class"]] : [])]));
      c.addEventListener("pointerleave", hideTip);
      svg.append(c);
    });
    const chips = h("div", { class: "seg", role: "group", "aria-label": "layer" }, ...L.map((l) => h("button", {
      "aria-pressed": String(l === cur), text: l.id === "input" ? "input" : nodeName(l.id), title: l.kind,
      onclick: () => { S.graphLayer = l.id; renderPanel(); } })));
    return section("The input graph", `${G.n} nodes · ${G.edges.length} edges`, chips, svg,
      colorbar({ vmin: lo, vmax: hi, diverging: false, orig: [] }),
      h("div", { class: "hint", text: "colour: the size ‖h‖ of each node's features after the chosen layer — watch message passing spread them along the edges" }));
  }
  function patchSize(n) {
    if (!n.attrs || !/Conv2d/.test(n.kind || "")) return null;
    const one = (v) => (Array.isArray(v) ? v[0] : typeof v === "string" ? parseInt(v.replace(/[^0-9,]/g, "").split(",")[0], 10) : v);
    const k = one(n.attrs.kernel_size), st = one(n.attrs.stride);
    return k && k === st && k >= 4 ? k : null;
  }
  function patchSection(k) {
    const I = P.inputImage, W = 300, H = Math.round(W * I.h / I.w), nx = Math.floor(I.w / k), ny = Math.floor(I.h / k);
    const svg = s("svg", { class: "patch-svg", viewBox: `0 0 ${I.w} ${I.h}`, width: W, height: H, role: "img", "aria-label": "image patches" });
    svg.append(s("image", { href: I.uri, x: 0, y: 0, width: I.w, height: I.h, preserveAspectRatio: "none", style: "image-rendering:pixelated" }));
    for (let i = 1; i < nx; i++) svg.append(s("line", { x1: i * k, y1: 0, x2: i * k, y2: I.h, class: "patch-line" }));
    for (let j = 1; j < ny; j++) svg.append(s("line", { x1: 0, y1: j * k, x2: I.w, y2: j * k, class: "patch-line" }));
    return section("Patches", `${nx}×${ny} patches of ${k}×${k}`, svg,
      h("div", { class: "hint", text: "this convolution's stride equals its kernel: it cuts the image into patches, one token each" }));
  }

  // ------------------------------------------------------------- compare two versions
  const CMP_COL = hexRgb("#F0AC5F");  // GOLD_C: "changed" — distinct from the yellow selection
  const pct = (v) => (v === null || v === undefined ? "—" : `${fmt(v * 100, 3)} %`);
  let CMP_MAX = null;
  function compareOf(id) {
    if (!P.compare) return null;
    if (CMP_MAX === null) CMP_MAX = Math.max(1e-12, ...P.compare.layers.map((l) => Math.max(l.weight || 0, l.act || 0)));
    const l = P.compare.layers.find((x) => x.id === id);
    if (!l) return null;
    const v = Math.max(l.weight || 0, l.act || 0);
    return v > 1e-9 ? { l, t: Math.sqrt(v / CMP_MAX) } : null;
  }
  function renderCompare() {
    const C = P.compare, [na, nb] = C.names;
    const view = h("div", { class: "tableview cmpview" });
    const moved = C.layers.filter((l) => (l.weight || 0) > 1e-9);
    view.append(h("h3", { class: "sec", text: `${na} → ${nb}` }),
      h("div", { class: "hint", style: { margin: "0 0 12px" }, text: `${moved.length} of ${C.layers.filter((l) => l.weight !== null).length} layers with weights changed · the same input through both · glowing nodes in the Graph changed most` }));
    const rows = C.layers, W = 820, labW = 190, colW = (W - labW - 40) / 2, rowH = 22;
    const wMax = Math.max(1e-12, ...rows.map((l) => l.weight || 0)), aMax = Math.max(1e-12, ...rows.map((l) => l.act || 0));
    const svg = s("svg", { class: "chart cmp-chart", viewBox: `0 0 ${W} ${rows.length * rowH + 30}`, width: W, height: rows.length * rowH + 30, role: "img", "aria-label": "per-layer change" });
    svg.append(s("text", { x: labW, y: 14, class: "cmp-head" }, "weights moved  ‖W′ − W‖ / ‖W‖"),
      s("text", { x: labW + colW + 30, y: 14, class: "cmp-head" }, "output moved  ‖h′ − h‖ / ‖h‖"));
    rows.forEach((l, i) => {
      const y = 24 + i * rowH, sel = l.id === S.selected;
      const g = s("g", { class: "cmp-row" + (sel ? " sel" : ""), style: "cursor:pointer" });
      g.append(s("rect", { x: 0, y: y - 2, width: W, height: rowH, class: "cmp-bg" }),
        s("text", { x: labW - 10, y: y + 13, "text-anchor": "end", class: "cmp-lab" }, `${nodeName(l.id)} · ${l.kind}`.slice(0, 30)));
      const bar = (x0, v, vmax, col) => {
        if (v === null || v === undefined) return s("text", { x: x0, y: y + 13, class: "cmp-val" }, "—");
        const w = Math.max(1.5, (colW - 70) * Math.sqrt(v / vmax));
        return s("g", null, s("rect", { x: x0, y: y + 3, width: w, height: rowH - 8, rx: 3, fill: rgba(col, 0.9) }),
          s("text", { x: x0 + w + 6, y: y + 13, class: "cmp-val" }, pct(v)));
      };
      g.append(bar(labW, l.weight, wMax, CMP_COL), bar(labW + colW + 30, l.act, aMax, hexRgb(P.theme.positive || "#58C4DD")));
      g.addEventListener("click", () => { S.selected = l.id; renderStage(); renderPanel(); });
      g.addEventListener("pointermove", (ev) => showTip(ev, [[pct(l.weight), "weights moved"], [pct(l.act), "output moved"],
        [l.cos === null ? "—" : fmt(l.cos, 4), "cosine of the two outputs"], nodeName(l.id)]));
      g.addEventListener("pointerleave", hideTip);
      svg.append(g);
    });
    view.append(h("div", { class: "cmp-scroll" }, svg));
    const outs = h("div", { class: "cmp-outs" });
    C.outputs.forEach((o, i) => {
      const name = C.names[i];
      const body = !o ? h("span", { class: "hint", text: "no output" }) : o.values ? outputBars(o, 360) : o.topk ? topkSection(o)
        : o.image ? h("img", { class: "out-img", src: o.image, alt: `${name} output` }) : o.heat ? heatCanvas(o.heat, 360, 200) : h("span");
      outs.append(h("div", { class: "cmp-out" }, h("h3", { class: "sec", text: `${name} · output` }), body));
    });
    view.append(outs);
    stage.append(view);
  }


  // ------------------------------------------------------------- dataset
  // net.profile(data, labels): a map of the samples at each layer, layer similarity (CKA), dead units, and the
  // inputs that excite each unit most
  const DS_COLS = ["BLUE_C", "YELLOW_C", "GREEN_C", "RED_C", "PURPLE_C", "TEAL_C", "GOLD_C", "MAROON_C", "GREY_B",
    "BLUE_E", "GREEN_E", "RED_E", "PURPLE_E", "TEAL_E", "GOLD_E", "YELLOW_E"];
  function dsColor(k) {
    const pal = P.theme.palette || {};
    return hexRgb(pal[DS_COLS[k % DS_COLS.length]] || "#58C4DD");
  }
  function dsLayer() {
    const D = P.dataset;
    if (S.dsLayer && D.layers.includes(S.dsLayer)) return S.dsLayer;
    return D.layers.includes(S.selected) ? S.selected : D.layers[D.layers.length - 1];
  }
  function dsClassOf(i) {
    const D = P.dataset, v = D.labels ? D.labels[i] : D.pred ? D.pred[i] : null;
    if (v === null || v === undefined) return null;
    const names = D.classNames;
    return { key: String(v), name: names && typeof v === "number" && names[v] !== undefined ? names[v] : String(v) };
  }
  function dsSample(i) {  // a sample as a picture, a token window, or its index
    const D = P.dataset, uri = D.thumbs[String(i)];
    if (uri) return h("img", { class: "ds-thumb", src: uri, alt: `sample ${i}` });
    return h("span", { class: "ds-idx", text: `#${i}` });
  }
  function renderDataset() {
    const D = P.dataset, L = dsLayer();
    const view = h("div", { class: "tableview dsview" });
    const dead = D.layers.reduce((a, l) => a + D.units[l].dead, 0);
    view.append(h("h3", { class: "sec", text: `${D.n.toLocaleString()} samples through ${D.layers.length} layers` }),
      h("div", { class: "hint", style: { margin: "0 0 12px" },
        text: `${dead.toLocaleString()} dead units · colours: ${D.labels ? "labels" : D.pred ? "the network's predictions" : "one colour (no labels)"} · click a layer anywhere to map it` }));
    const sel = h("select", { class: "ds-select", "aria-label": "layer to map" },
      ...D.layers.map((l) => h("option", { value: l, text: `${nodeName(l)} · ${(nodeById(l) || {}).kind || ""}`, selected: l === L ? "" : null })));
    sel.addEventListener("change", () => { S.dsLayer = sel.value; renderStage(); });
    const row = h("div", { class: "ds-row" });
    row.append(h("div", { class: "ds-card" }, h("div", { class: "ds-head" }, h("b", { text: "The samples at " }), sel), dsScatter(L)),
      h("div", { class: "ds-card" }, h("div", { class: "ds-head" }, h("b", { text: "How alike the layers are" }),
        h("span", { class: "hint", text: ` linear CKA: 1 = the same representation up to rotation and scale · colours span ${D.cka ? fmt(Math.min(...D.cka.m.flat()), 2) : "0"} – 1` })), dsCKA(L)));
    view.append(row, h("h3", { class: "sec", text: "Dead units per layer" }), dsDead(L),
      h("h3", { class: "sec", text: `What excites ${nodeName(L)}'s units most` }), dsTopGrid(L, 12));
    stage.append(view);
  }
  function dsScatter(L) {
    const D = P.dataset, E = D.embed[L], W = 420, H = 380, pad = 16;
    const svg = s("svg", { class: "chart ds-scatter", viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": `map of the samples at ${L}` });
    if (!E) { svg.append(s("text", { x: W / 2, y: H / 2, "text-anchor": "middle", class: "cmp-val" }, "no map for this layer")); return svg; }
    const keys = [];
    E.xy.forEach(([x, y], k) => {
      const i = D.embedIndex[k], c = dsClassOf(i);
      if (c && !keys.includes(c.key)) keys.push(c.key);
      const col = c ? dsColor(keys.indexOf(c.key)) : hexRgb(P.theme.positive || "#58C4DD");
      const dot = s("circle", { cx: pad + (x + 1) / 2 * (W - 2 * pad), cy: pad + (1 - (y + 1) / 2) * (H - 2 * pad), r: 3.2,
        fill: rgba(col, 0.8), class: "ds-dot" });
      dot.addEventListener("pointermove", (ev) => showTip(ev, [dsSample(i), [c ? c.name : "—", D.labels ? "label" : D.pred ? "predicted" : ""], `sample #${i}`]));
      dot.addEventListener("pointerleave", hideTip);
      svg.append(dot);
    });
    if (keys.length && keys.length <= 16) {
      const lg = s("g", { class: "ds-legend" });
      keys.forEach((k, j) => {
        const c = dsClassOf(D.embedIndex[E.xy.findIndex((_, q) => (dsClassOf(D.embedIndex[q]) || {}).key === k)]);
        lg.append(s("circle", { cx: W - 74, cy: 12 + j * 14, r: 4, fill: rgba(dsColor(j), 0.9) }),
          s("text", { x: W - 64, y: 16 + j * 14, class: "cmp-val" }, String(c ? c.name : k).slice(0, 10)));
      });
      svg.append(lg);
    }
    return svg;
  }
  function dsCKA(L) {
    const D = P.dataset, C = D.cka;
    if (!C) return h("span", { class: "hint", text: "not enough samples" });
    const n = C.layers.length, side = Math.max(4, Math.min(22, Math.floor(360 / n))), lab = 110;
    const lo = Math.min(...C.m.flat().filter((v) => v !== null)), span = Math.max(1e-6, 1 - lo);  // the matrix's own range
    const svg = s("svg", { class: "chart ds-cka", viewBox: `0 0 ${lab + n * side + 4} ${n * side + 4}`, width: lab + n * side + 4, height: n * side + 4, role: "img", "aria-label": "layer similarity" });
    C.layers.forEach((a, i) => {
      if (side >= 9) svg.append(s("text", { x: lab - 6, y: i * side + side * 0.75, "text-anchor": "end", class: "cmp-lab" + (a === L ? " sel" : "") }, nodeName(a).slice(-16)));
      C.layers.forEach((b, j) => {
        const v = C.m[i][j];
        const cell = s("rect", { x: lab + j * side, y: i * side, width: side - 0.5, height: side - 0.5, fill: rgba(SEQ((v - lo) / span), 1),
          class: a === L || b === L ? "ds-cell hl" : "ds-cell" });
        cell.addEventListener("pointermove", (ev) => showTip(ev, [[fmt(v, 3), "CKA"], `${nodeName(a)} ↔ ${nodeName(b)}`]));
        cell.addEventListener("pointerleave", hideTip);
        cell.addEventListener("click", () => { S.dsLayer = a; renderStage(); });
        svg.append(cell);
      });
    });
    return svg;
  }
  function dsDead(L) {
    const D = P.dataset, W = 820, rowH = 18, labW = 190;
    const rows = D.layers.map((l) => ({ l, u: D.units[l] }));
    const svg = s("svg", { class: "chart", viewBox: `0 0 ${W} ${rows.length * rowH + 8}`, width: W, height: rows.length * rowH + 8, role: "img", "aria-label": "dead units per layer" });
    rows.forEach(({ l, u }, i) => {
      const y = 4 + i * rowH, frac = u.C ? u.dead / u.C : 0, w = Math.max(frac > 0 ? 2 : 0, (W - labW - 120) * frac);
      const g = s("g", { class: "cmp-row" + (l === L ? " sel" : ""), style: "cursor:pointer" });
      g.append(s("rect", { x: 0, y: y - 2, width: W, height: rowH, class: "cmp-bg" }),
        s("text", { x: labW - 10, y: y + 11, "text-anchor": "end", class: "cmp-lab" }, `${nodeName(l)} · ${(nodeById(l) || {}).kind || ""}`.slice(0, 30)),
        s("rect", { x: labW, y: y + 2, width: w, height: rowH - 7, rx: 3, fill: rgba(hexRgb(P.theme.negative || "#FC6255"), 0.85) }),
        s("text", { x: labW + w + 6, y: y + 11, class: "cmp-val" }, u.dead ? `${u.dead.toLocaleString()} of ${u.C.toLocaleString()} (${pct(frac)})` : `0 of ${u.C.toLocaleString()}`));
      g.addEventListener("click", () => { S.dsLayer = l; renderStage(); });
      g.addEventListener("pointermove", (ev) => showTip(ev, [[u.nonneg ? "never active" : "never changes", "dead means"], nodeName(l)]));
      g.addEventListener("pointerleave", hideTip);
      svg.append(g);
    });
    return h("div", { class: "cmp-scroll" }, svg);
  }
  function dsTopGrid(L, maxUnits) {
    const D = P.dataset, rows = (D.top[L] || []).slice(0, maxUnits);
    const grid = h("div", { class: "ds-top" });
    rows.forEach((r) => {
      const items = r.items.map((it, k) => {
        const cell = h("div", { class: "ds-item" }, r.snippets ? h("code", { class: "ds-snip", text: r.snippets[k] }) : dsSample(it[0]),
          h("div", { class: "ds-val", text: fmt(it[1], 3) }));
        const c = dsClassOf(it[0]);
        cell.addEventListener("pointermove", (ev) => showTip(ev, [[fmt(it[1], 4), "activation"], `sample #${it[0]}` + (c ? ` · ${c.name}` : "")]));
        cell.addEventListener("pointerleave", hideTip);
        return cell;
      });
      grid.append(h("div", { class: "ds-unit" }, h("div", { class: "ds-uname", text: `unit ${r.unit}` }), h("div", { class: "ds-items" }, ...items)));
    });
    if (!rows.length) grid.append(h("span", { class: "hint", text: "no units recorded" }));
    return grid;
  }
  function datasetSection(nid) {
    const box = h("details", { class: "p-sec", open: "" }, h("summary", null, "Inputs that excite it most"),
      h("div", { class: "hint", text: `the ${P.dataset.n.toLocaleString()} profiled samples that drive its most active units hardest` }),
      dsTopGrid(nid, 6));
    return box;
  }

  // ------------------------------------------------------------- logit lens
  function lensBlock(view, L) {
    const P_ = L.positions, toks = L.tokens;
    const name = (t) => (t.label !== null && t.label !== undefined ? t.label : `#${t.id}`);
    view.append(h("h3", { class: "sec", text: "Logit lens — the prediction after each block" }));
    view.append(h("div", { class: "hint", style: { margin: "0 0 10px" },
      text: `each block's output → ${L.norm ? nodeName(L.norm) + " → " : ""}${nodeName(L.head)} · shade: top-1 probability · last row: the real prediction` }));
    const tbl = h("table", { class: "dt lens" });
    tbl.append(h("thead", null, h("tr", null, h("th", { class: "l", text: "after" }),
      ...P_.map((p) => h("th", { class: "l", title: `position ${p}`, text: toks ? (toks[p] || "").replace(/ /g, "·") : String(p) })))));
    const tb = h("tbody");
    L.layers.forEach((layer, li) => {
      const tr = h("tr", { class: (L.nodes || [])[li] === S.selected ? "sel" : "" }, h("td", { class: "l", text: layer }));
      L.top[li].forEach((cell, j) => {
        const t = cell[0], c = SEQ(Math.min(1, Math.sqrt(t.p)));
        const td = h("td", { class: "l lens-cell", text: name(t).replace(/ /g, "·").slice(0, 10),
          style: { background: rgba(c, 0.85), color: t.p > 0.35 ? "#000000" : "" } });
        td.addEventListener("pointermove", (ev) => showTip(ev, cell.map((x) => [`${(x.p * 100).toFixed(1)}%`, `“${name(x)}”`]).concat(
          [`${layer} · position ${P_[j]}${toks ? ` “${toks[P_[j]]}”` : ""}`])));
        td.addEventListener("pointerleave", hideTip);
        tr.append(td);
      });
      const nid = (L.nodes || [])[li];
      if (nid) tr.addEventListener("click", () => select(nid));
      tb.append(tr);
    });
    tbl.append(tb);
    view.append(h("div", { class: "lens-wrap" }, tbl));
  }

  // ----------------------------------------------------------- gradient flow
  function gradViewKids(gv, dl) {
    const kids = [];
    if (gv.saliency) {
      kids.push(h("div", { class: "p-sub", text: "saliency: max |∂L/∂x| over channels — where the input matters" }),
        heatCanvas(gv.saliency, 300, 300), colorbar(gv.saliency));
    }
    if (gv.vector) {
      const v = gv.vector;
      const hm = { rows: 1, cols: Math.min(f32(v.data).length, 4096), data: v.data, vmin: v.vmin, vmax: v.vmax, diverging: v.diverging, downsampled: v.downsampled, orig: [1, v.n], colLabel: "neuron" };
      kids.push(h("div", { class: "p-sub", text: `${dl} per neuron` }), heatCanvas(hm), colorbar(hm));
    }
    if (gv.maps) kids.push(h("div", { class: "p-sub", text: `${dl} per channel (${gv.maps.n} of ${gv.maps.channels})` }), mapsGrid(gv.maps), colorbar({ ...gv.maps, orig: [] }));
    if (gv.matrix) kids.push(h("div", { class: "p-sub", text: `${dl} · ${gv.matrix.rowLabel} × ${gv.matrix.colLabel}` }), heatCanvas(gv.matrix), colorbar(gv.matrix));
    return kids;
  }
  function gradFlowBlock(view, G) {
    const rows = G.rows || [];
    if (!rows.length) return;
    const pal = P.theme.palette || {};
    const cA = pal.BLUE_C || "#58C4DD", cW = pal.GOLD_C || "#F0AC5F";
    view.append(h("h3", { class: "sec", text: `Gradient flow · loss ${fmt(G.loss, 4)} · ${rows.length} layers, input → output` }));
    const vals = rows.flatMap((r) => [r.act, r.actMax, r.param, r.paramMax]).filter((v) => v > 0);
    if (!vals.length) return;
    const lo = Math.floor(Math.log10(Math.min(...vals))), hi = Math.ceil(Math.log10(Math.max(...vals)));
    const withAct = rows.filter((r) => r.act > 0);
    let note = "Mean |gradient| per layer on one shared log axis; ticks mark the largest entry. Click a row to open the layer.";
    if (withAct.length >= 2) {
      const d = Math.log10(withAct[withAct.length - 1].act / withAct[0].act);
      note = `|∂L/∂output| changes ${d >= 0 ? "×" : "÷"}10^${Math.abs(d).toFixed(1)} from the first layer to the last` +
        (d > 3 ? " — gradients vanish on their way back to the input. " : d < -3 ? " — gradients grow on their way back to the input (exploding). " : ". ") + note;
    }
    view.append(h("div", { class: "hint", style: { margin: "0 0 10px" }, text: note }));
    const W = 200, H = 12;
    const X = (v) => ((Math.log10(v) - lo) / Math.max(hi - lo, 1e-9)) * W;
    const bar = (mean, max, col, label, id) => {
      const svg = s("svg", { class: "gbar", width: W, height: H, viewBox: `0 0 ${W} ${H}` });
      svg.append(s("rect", { x: 0, y: 0, width: W, height: H, class: "gbar-bg", rx: 2 }));
      if (mean > 0) svg.append(s("rect", { x: 0, y: 2, width: Math.max(2, X(mean)), height: H - 4, rx: 2, fill: col }));
      if (max > 0) svg.append(s("line", { x1: X(max), x2: X(max), y1: 0, y2: H, stroke: col, "stroke-width": 2 }));
      svg.addEventListener("pointermove", (ev) => showTip(ev, [[fmt(mean, 3), `mean ${label}`], [fmt(max, 3), `max ${label}`], id]));
      svg.addEventListener("pointerleave", hideTip);
      return svg;
    };
    const axis = () => h("span", { class: "gaxis" }, h("span", { text: `1e${lo}` }), h("span", { text: `1e${hi}` }));
    const tbl = h("table", { class: "dt" });
    tbl.append(h("thead", null, h("tr", null, h("th", { text: "layer" }), h("th", { class: "l", text: "kind" }),
      h("th", { class: "l" }, h("span", { class: "gkey", style: { background: cA } }), "|∂L/∂output| ", axis()), h("th", { text: "mean" }),
      h("th", { class: "l" }, h("span", { class: "gkey", style: { background: cW } }), "|∂L/∂θ| ", axis()), h("th", { text: "mean" }))));
    const tb = h("tbody");
    for (const r of rows) {
      const tr = h("tr", { class: r.id === S.selected ? "sel" : "" }, h("td", { text: nodeName(r.id), title: r.id }), h("td", { class: "l", text: r.kind }),
        h("td", { class: "l" }, r.act !== undefined ? bar(r.act, r.actMax, cA, "|∂L/∂output|", r.id) : null), h("td", { text: fmt(r.act) }),
        h("td", { class: "l" }, r.param !== undefined ? bar(r.param, r.paramMax, cW, "|∂L/∂θ|", r.id) : null), h("td", { text: fmt(r.param) }));
      tr.addEventListener("click", () => select(r.id));
      tb.append(tr);
    }
    tbl.append(tb);
    view.append(tbl);
  }

  // ---------------------------------------------------------------- training
  const TSTAT = {
    update: ["relative update", "‖θₜ − θₚᵣₑᵥ‖ / ‖θₚᵣₑᵥ‖ per step, averaged since the previous snapshot — how fast each layer is moving; ~1e-3 is typical (log scale)", "log"],
    grad: ["gradient norm", "‖∇θ‖ at each snapshot (log scale)", "log"],
    norm: ["weight norm", "‖θ‖ relative to the first snapshot — blue grew, red shrank (log₂ ratio)", "ratio"],
    act: ["output spread", "std of the layer's output on the dashboard input, relative to the first snapshot (log₂ ratio)", "ratio"],
  };
  const lastOf = (arr) => { for (let i = arr.length - 1; i >= 0; i--) if (arr[i] !== null && arr[i] !== undefined) return arr[i]; return null; };
  function trainingFor(id) {
    return P.training && P.training.layers ? P.training.layers.find((l) => l.id === id) || null : null;
  }
  function stepChart(xs, ys, width, height, label, log = false) {
    const svg = s("svg", { class: "chart", viewBox: `0 0 ${width} ${height}`, style: { maxWidth: width + "px" }, role: "img", "aria-label": `${label} by training step` });
    const pts = [];
    for (let i = 0; i < ys.length; i++) {
      const v = ys[i];
      if (v !== null && v !== undefined && isFinite(v) && (!log || v > 0)) pts.push([xs[i], v, i]);
    }
    if (!pts.length) return svg;
    const tf = log ? Math.log10 : (v) => v;
    const padL = 54, padB = 20, padT = 8, padR = 12;
    let lo = Math.min(...pts.map((p) => tf(p[1]))), hi = Math.max(...pts.map((p) => tf(p[1])));
    if (hi === lo) { hi += log ? 0.5 : 1; lo -= log ? 0.5 : 1; }
    const x0 = pts[0][0], x1 = pts[pts.length - 1][0];
    const X = (x) => padL + (x1 === x0 ? 0.5 : (x - x0) / (x1 - x0)) * (width - padL - padR);
    const Y = (v) => padT + (1 - (tf(v) - lo) / (hi - lo)) * (height - padT - padB);
    for (let t = 0; t <= 3; t++) {
      const tv = lo + ((hi - lo) * t) / 3, y = padT + (1 - t / 3) * (height - padT - padB);
      svg.append(s("line", { class: "axis", x1: padL, x2: width - padR, y1: y, y2: y, opacity: 0.6 }),
        s("text", { class: "tick", x: padL - 8, y: y + 3, "text-anchor": "end" }, fmt(log ? 10 ** tv : tv)));
    }
    svg.append(s("text", { class: "tick", x: padL, y: height - 4 }, `step ${x0}`),
      s("text", { class: "tick", x: width - padR, y: height - 4, "text-anchor": "end" }, `step ${x1}`));
    const d = pts.map((p, k) => `${k ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(" ");
    svg.append(s("path", { class: "line", d }));
    const xh = s("line", { class: "xhair", y1: padT, y2: height - padB, opacity: 0 });
    const dot = s("circle", { class: "dot", r: 4, opacity: 0 });
    const end = s("circle", { class: "dot", r: 3, cx: X(pts[pts.length - 1][0]), cy: Y(pts[pts.length - 1][1]) });
    svg.append(end, xh, dot);
    const hit = s("rect", { x: padL, y: 0, width: width - padL - padR, height: height - padB, fill: "transparent" });
    hit.addEventListener("pointermove", (ev) => {
      const r = svg.getBoundingClientRect();
      const px = ((ev.clientX - r.left) / r.width) * width;
      let best = pts[0];
      for (const p of pts) if (Math.abs(X(p[0]) - px) < Math.abs(X(best[0]) - px)) best = p;
      xh.setAttribute("x1", X(best[0])); xh.setAttribute("x2", X(best[0])); xh.setAttribute("opacity", 1);
      dot.setAttribute("cx", X(best[0])); dot.setAttribute("cy", Y(best[1])); dot.setAttribute("opacity", 1);
      showTip(ev, [[fmt(best[1], 4), `${label} · step ${best[0]}`]]);
    });
    hit.addEventListener("pointerleave", () => { xh.setAttribute("opacity", 0); dot.setAttribute("opacity", 0); hideTip(); });
    svg.append(hit);
    return svg;
  }
  function trainingBlock(view, T) {
    const names = Object.keys(T.metrics || {});
    view.append(h("h3", { class: "sec", text: `Training · step ${T.lastStep ?? "—"} · ${T.count} logged steps` }));
    if (names.length) {
      const grid = h("div", { class: "train-grid" });
      for (const k of names) {
        const ys = T.metrics[k];
        const pos = ys.every((v) => v === null || v > 0);
        const vals = ys.filter((v) => v !== null && v > 0);
        const spread = vals.length ? Math.max(...vals) / Math.min(...vals) : 1;
        const log = pos && /loss|perplex|err/i.test(k) && spread > 20;
        grid.append(h("div", { class: "tcard" },
          h("div", { class: "tc-head" }, h("span", { text: k + (log ? " (log)" : "") }), h("b", { text: fmt(lastOf(ys), 4) })),
          stepChart(T.steps, ys, 380, 150, k, log)));
      }
      view.append(grid);
    }
    replayBlock(view, T);
    const layers = (T.layers || []).filter((l) => Object.keys(l.stats).length);
    if (!layers.length || !(T.snapSteps || []).length) return;
    const avail = Object.keys(TSTAT).filter((k) => layers.some((l) => l.stats[k]));
    if (!avail.includes(S.trainStat)) S.trainStat = avail[0];
    view.append(h("h3", { class: "sec", text: "Learning dynamics — every layer, every snapshot" }));
    view.append(h("div", { class: "seg", role: "group", "aria-label": "layer statistic" }, ...avail.map((k) => h("button", {
      "aria-pressed": String(S.trainStat === k), text: TSTAT[k][0], onclick: () => { S.trainStat = k; renderStage(); },
    }))));
    view.append(h("div", { class: "hint", style: { margin: "8px 0 10px" }, text: TSTAT[S.trainStat][1] + ". Click a row to follow that layer." }));
    view.append(dynamicsStrip(T, layers, S.trainStat));
    const pick = layers.find((l) => l.id === S.trainNode) || layers[0];
    const ys = pick.stats[S.trainStat] || [];
    view.append(h("h3", { class: "sec", text: `${pick.id} · ${TSTAT[S.trainStat][0]}` }),
      stepChart(T.snapSteps, ys, 760, 190, TSTAT[S.trainStat][0], TSTAT[S.trainStat][2] === "log"));
  }
  function layerLabel(id) {  // "2 · Linear": the path and what it is
    const tl = (P.training.layers || []).find((l) => l.id === id);
    return tl && tl.kind ? `${nodeName(id)} · ${tl.kind}` : nodeName(id);
  }
  // Replay: every layer's weights and the output, frame by frame through training (TrainingLog frames)
  function replayBlock(view, T) {
    const F = T.frames || [];
    if (F.length < 2) return;
    const follow = S.replay == null || S.replayFollow;  // at the newest frame: keep following as training goes on
    S.replay = follow ? F.length - 1 : Math.min(S.replay, F.length - 1);
    const ids = Object.keys(F[F.length - 1].weights).slice(0, 18);
    const scale = {};  // one colour scale per layer across frames, so change reads as change
    for (const id of ids) scale[id] = Math.max(1e-12, ...F.map((f) => (f.weights[id] ? Math.max(Math.abs(f.weights[id].vmin), Math.abs(f.weights[id].vmax)) : 0)));
    const grid = h("div", { class: "replay-grid" });
    const cells = ids.map((id) => {
      const c = h("button", { class: "replay-cell" + (id === S.trainNode ? " sel" : ""), title: `${id} — click to follow this layer`,
        onclick: () => { S.trainNode = id; if (level().nodes.some((m) => m.id === id)) S.selected = id; renderStage(); renderPanel(); } });
      grid.append(c);
      return c;
    });
    const outBox = h("div", { class: "replay-out" });
    const stepLab = h("b", { class: "replay-step" });
    const slider = h("input", { type: "range", min: 0, max: F.length - 1, step: 1, value: S.replay, "aria-label": "training step" });
    const draw = (i) => {
      S.replay = i; S.replayFollow = i === F.length - 1;
      const f = F[i];
      stepLab.textContent = `step ${f.step}`;
      ids.forEach((id, j) => {
        const hm = f.weights[id];
        cells[j].replaceChildren(hm ? heatCanvas({ ...hm, vmin: -scale[id], vmax: scale[id], diverging: true }, 132, 84) : h("span", { class: "hint", text: "—" }),
          h("span", { class: "rc-lab", text: layerLabel(id) }));
      });
      const o = f.output;
      outBox.replaceChildren(h("div", { class: "rc-lab", text: "output" }),
        !o ? h("span", { class: "hint", text: "not recorded" })
          : o.values ? outputBars(o, 280) : o.topk ? topkSection(o)
            : o.image ? h("img", { class: "out-img", src: o.image, alt: `output at step ${f.step}` })
              : o.heat ? heatCanvas(o.heat, 280, 180) : h("span"));
    };
    let timer = null;
    const stop = () => { clearInterval(timer); timer = null; play.textContent = "▶ Play"; };
    const play = h("button", { class: "btn", text: "▶ Play", onclick: () => {
      if (timer) return stop();
      if (S.replay >= F.length - 1) { slider.value = 0; draw(0); }
      play.textContent = "❚❚ Pause";
      timer = setInterval(() => {
        const n = S.replay + 1;
        if (n >= F.length) return stop();
        slider.value = n; draw(n);
      }, Math.max(60, 2400 / F.length));
    } });
    slider.addEventListener("input", () => { stop(); draw(+slider.value); });
    view.append(h("h3", { class: "sec", text: "Replay — watch it learn" }),
      h("div", { class: "replay-bar" }, play, slider, stepLab,
        h("span", { class: "hint", text: `${F.length} frames · steps ${F[0].step}–${F[F.length - 1].step}` })),
      h("div", { class: "replay" }, grid, outBox));
    draw(S.replay);
  }
  function stripValues(layers, stat) {
    const mode = TSTAT[stat][2];
    const rows = layers.map((l) => {
      const ys = l.stats[stat] || [];
      if (mode === "log") return ys.map((v) => (v !== null && v > 0 ? Math.log10(v) : null));
      const base = ys.find((v) => v !== null && v > 0);
      return ys.map((v) => (v !== null && v > 0 && base ? Math.log2(v / base) : null));
    });
    const all = rows.flat().filter((v) => v !== null).sort((a, b) => a - b);
    if (!all.length) return { rows, lo: 0, hi: 1, div: mode !== "log" };
    const q = (p) => all[Math.min(all.length - 1, Math.max(0, Math.round(p * (all.length - 1))))];
    if (mode === "log") return { rows, lo: q(0.02), hi: Math.max(q(0.98), q(0.02) + 1e-6), div: false };
    const m = Math.max(Math.abs(q(0.02)), Math.abs(q(0.98)), 1e-6);
    return { rows, lo: -m, hi: m, div: true };
  }
  function dynamicsStrip(T, layers, stat) {
    const nS = T.snapSteps.length, nL = layers.length;
    const { rows, lo, hi, div } = stripValues(layers, stat);
    const rowH = Math.max(3, Math.min(18, Math.floor(420 / nL)));
    const cv = h("canvas", { class: "heat strip", width: nS, height: nL });
    const ctx = cv.getContext("2d");
    const img = ctx.createImageData(nS, nL);
    const bg = hexRgb(P.theme.palette ? P.theme.palette.GREY_E || "#222222" : "#222222");
    rows.forEach((r, i) => {
      for (let j = 0; j < nS; j++) {
        const v = r[j];
        const c = v === null || v === undefined ? bg : colorOf(v, lo, hi, div);
        img.data.set([c[0], c[1], c[2], 255], (i * nS + j) * 4);
      }
    });
    ctx.putImageData(img, 0, 0);
    cv.style.height = rowH * nL + "px";
    cv.style.width = "100%";
    const labels = h("div", { class: "strip-labels", style: { gridTemplateRows: `repeat(${nL}, ${rowH}px)` } },
      ...layers.map((l) => h("span", { class: l.id === S.trainNode ? "sel" : "", text: rowH >= 10 ? l.id : "" })));
    const pickRow = (ev) => {
      const r = cv.getBoundingClientRect();
      const i = Math.floor(((ev.clientY - r.top) / r.height) * nL), j = Math.floor(((ev.clientX - r.left) / r.width) * nS);
      return [Math.min(nL - 1, Math.max(0, i)), Math.min(nS - 1, Math.max(0, j))];
    };
    cv.addEventListener("pointermove", (ev) => {
      const [i, j] = pickRow(ev);
      const raw = (layers[i].stats[stat] || [])[j];
      showTip(ev, [[fmt(raw, 4), `${TSTAT[stat][0]} · step ${T.snapSteps[j]}`], layers[i].id + " · " + layers[i].kind]);
    });
    cv.addEventListener("pointerleave", hideTip);
    cv.addEventListener("click", (ev) => {
      const [i] = pickRow(ev);
      S.trainNode = layers[i].id;
      if (level().nodes.some((m) => m.id === layers[i].id)) S.selected = layers[i].id;
      renderStage(); renderPanel();
    });
    const lab = div ? ["shrank", "grew"] : ["slow", "fast"];
    const unit = div ? (v) => 2 ** v : (v) => 10 ** v;  // colorbar in real units: ratio ×, or the raw statistic
    const bar = colorbar({ vmin: unit(lo), vmax: unit(hi), diverging: div, orig: [] });
    return h("div", { class: "strip-wrap" }, labels, cv, h("span"),
      h("div", null, h("div", { class: "strip-foot" }, h("span", { text: `step ${T.snapSteps[0]}` }),
        h("span", { text: `${nL} layers × ${nS} snapshots · colour: ${lab[0]} → ${lab[1]}` }),
        h("span", { text: `step ${T.snapSteps[nS - 1]}` })), bar));
  }
  function layerTrainingSection(tl) {
    const T = P.training;
    const rows = Object.keys(TSTAT).filter((k) => tl.stats[k]).map((k) => {
      const ys = tl.stats[k];
      const pts = ys.filter((v) => v !== null && isFinite(v));
      const series = TSTAT[k][2] === "log" ? pts.filter((v) => v > 0).map(Math.log10) : pts;
      return h("div", { class: "trow" }, h("span", { class: "tl", text: TSTAT[k][0] }), sparkline(series, 150, 24), h("b", { text: fmt(lastOf(ys), 3) }));
    });
    return section("Training", `${T.snapSteps.length} snapshots · step ${T.lastStep}`, ...rows,
      h("div", { class: "hint", text: "Per-snapshot dynamics of this layer's parameters. The Monitor tab compares every layer." }));
  }

  // ------------------------------------------------------------------ charts
  function heatCanvas(hm, maxW = 380, maxH = 260, onCell = null) {
    if (hm._vals) {  // plain numbers (not an encoded heat map): one row, 0 … max
      const mx = Math.max(...hm._vals, 1e-12);
      hm = { ...hm, vmin: 0, vmax: mx, diverging: false, orig: [1, hm._vals.length] };
    }
    const data = hm._vals ? Float32Array.from(hm._vals) : f32(hm.data);
    const cv = h("canvas", { class: "heat", width: hm.cols, height: hm.rows });
    const ctx = cv.getContext("2d");
    const img = ctx.createImageData(hm.cols, hm.rows);
    for (let i = 0; i < data.length; i++) {
      const c = colorOf(data[i], hm.vmin, hm.vmax, hm.diverging);
      img.data.set([c[0], c[1], c[2], 255], i * 4);
    }
    ctx.putImageData(img, 0, 0);
    let w = maxW, hh = (hm.rows / hm.cols) * w;
    if (hh > maxH) { hh = maxH; w = (hm.cols / hm.rows) * hh; }
    if (hm.rows === 1) { hh = Math.max(18, Math.min(28, hh)); w = maxW; }
    hh = Math.max(hh, 8);
    cv.style.width = w + "px";
    cv.style.height = hh + "px";
    cv.addEventListener("pointermove", (ev) => {
      const r = cv.getBoundingClientRect();
      const col = Math.floor(((ev.clientX - r.left) / r.width) * hm.cols);
      const row = Math.floor(((ev.clientY - r.top) / r.height) * hm.rows);
      if (col < 0 || row < 0 || col >= hm.cols || row >= hm.rows) return hideTip();
      const v = data[row * hm.cols + col];
      const rl = hm.rowLabel || "row", cl = hm.colLabel || "col";
      let where;
      if (hm.downsampled) {
        const fr = Math.ceil(hm.orig[0] / hm.rows), fc = Math.ceil(hm.orig[1] / hm.cols);
        where = `${rl} ${row * fr}–${Math.min((row + 1) * fr, hm.orig[0]) - 1}, ${cl} ${col * fc}–${Math.min((col + 1) * fc, hm.orig[1]) - 1} (block mean)`;
      } else if (hm.rowNames && hm.colNames) {
        where = `${rl} ${row} “${hm.rowNames[row]}” → ${cl} ${col} “${hm.colNames[col]}”`;
      } else where = hm.rows === 1 ? `${cl} ${col}` : `${rl} ${row}, ${cl} ${col}`;
      showTip(ev, [[fmt(v, 4), where]]);
    });
    cv.addEventListener("pointerleave", hideTip);
    if (onCell) {
      cv.style.cursor = "cell";
      cv.addEventListener("click", (ev) => {
        const r = cv.getBoundingClientRect();
        const col = Math.floor(((ev.clientX - r.left) / r.width) * hm.cols);
        const row = Math.floor(((ev.clientY - r.top) / r.height) * hm.rows);
        if (col >= 0 && row >= 0 && col < hm.cols && row < hm.rows) onCell(row, col, data[row * hm.cols + col], ev);
      });
    }
    return cv;
  }

  /** Popover to set one weight: canonical (out, in, …) heatmap cell → storage index via the param's axis permutation. */
  function cellEditor(ev, path, pp, row, col, value) {
    document.querySelectorAll(".cell-pop").forEach((e) => e.remove());
    const cs = pp.cshape, perm = pp.perm;
    let full;
    if (cs.length <= 1) full = [col];
    else {
      full = [row];
      let rest = col;
      const tail = cs.slice(1);
      const idx = [];
      for (let i = tail.length - 1; i >= 0; i--) { idx.unshift(rest % tail[i]); rest = Math.floor(rest / tail[i]); }
      full = full.concat(idx);
    }
    const storage = new Array(full.length);
    full.forEach((v, i) => { storage[perm[i]] = v; });
    const index = "[" + storage.join(", ") + "]";
    const input = h("input", { type: "number", step: "any", value: String(value) });
    const pop = h("div", { class: "cell-pop" },
      h("div", { class: "tl", text: `${path}${index}` }),
      h("div", { class: "row" }, input,
        h("button", { class: "btn primary", text: "Set", onclick: () => { pop.remove(); api("/api/edit", { op: "set", target: path, index, args: { value: Number(input.value) } }); } }),
        h("button", { class: "btn", text: "0", title: "set to zero", onclick: () => { pop.remove(); api("/api/edit", { op: "set", target: path, index, args: { value: 0 } }); } }),
        h("button", { class: "btn", text: "×", title: "cancel", onclick: () => pop.remove() })));
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { pop.remove(); api("/api/edit", { op: "set", target: path, index, args: { value: Number(input.value) } }); }
      if (e.key === "Escape") pop.remove();
    });
    document.body.append(pop);
    pop.style.left = Math.min(ev.clientX + 8, window.innerWidth - 300) + "px";
    pop.style.top = Math.min(ev.clientY + 8, window.innerHeight - 90) + "px";
    input.focus();
    input.select();
  }
  function colorbar(hm) {
    const cv = h("canvas", { width: 160, height: 8 });
    const ctx = cv.getContext("2d");
    for (let i = 0; i < 160; i++) { ctx.fillStyle = rgba((hm.diverging ? DIV : SEQ)(i / 159)); ctx.fillRect(i, 0, 1, 8); }
    return h("div", { class: "cbar" }, h("span", { text: fmt(hm.vmin) }), cv, h("span", { text: fmt(hm.vmax) }),
      hm.downsampled ? h("span", { text: `· ${hm.orig.join("×")} shown as ${hm.rows}×${hm.cols}` }) : null);
  }
  function histogram(hist, width = 380, height = 86, label = "count") {
    const counts = hist.counts, edges = hist.edges;
    const n = counts.length, maxC = Math.max(...counts, 1);
    const padB = 16, gap = n > 60 ? 0 : 1;
    const svg = s("svg", { class: "chart", viewBox: `0 0 ${width} ${height}`, width, height, role: "img", "aria-label": "histogram" });
    const bw = width / n;
    const x0 = edges[0], x1 = edges[edges.length - 1];
    const xz = x0 < 0 && x1 > 0 ? ((0 - x0) / (x1 - x0)) * width : null;
    counts.forEach((c, i) => {
      const bh = (c / maxC) * (height - padB - 4);
      const rect = s("rect", { class: "bar", x: i * bw + gap / 2, y: height - padB - bh, width: Math.max(bw - gap, 0.5), height: Math.max(bh, c ? 1 : 0), rx: Math.min(2, bw / 3) });
      const hit = s("rect", { x: i * bw, y: 0, width: bw, height: height - padB, fill: "transparent" });
      hit.addEventListener("pointermove", (ev) => { rect.classList.add("hover"); showTip(ev, [[c.toLocaleString(), `${label} in [${fmt(edges[i])}, ${fmt(edges[i + 1])})`]]); });
      hit.addEventListener("pointerleave", () => { rect.classList.remove("hover"); hideTip(); });
      svg.append(rect, hit);
    });
    svg.append(s("line", { class: "axis", x1: 0, x2: width, y1: height - padB + 0.5, y2: height - padB + 0.5 }));
    if (xz !== null) svg.append(s("line", { class: "axis", x1: xz, x2: xz, y1: 0, y2: height - padB }), s("text", { class: "tick", x: xz, y: height - 3, "text-anchor": "middle" }, "0"));
    svg.append(s("text", { class: "tick", x: 0, y: height - 3 }, fmt(x0)), s("text", { class: "tick", x: width, y: height - 3, "text-anchor": "end" }, fmt(x1)));
    return svg;
  }
  function barChart(values, width = 380, height = 70, label = "σ") {
    const n = values.length, m = Math.max(...values, 1e-12);
    const svg = s("svg", { class: "chart", viewBox: `0 0 ${width} ${height}`, width, height, role: "img" });
    const bw = width / n, padB = 14;
    values.forEach((v, i) => {
      const bh = (v / m) * (height - padB - 4);
      const rect = s("rect", { class: "bar", x: i * bw + 0.5, y: height - padB - bh, width: Math.max(Math.min(bw - 1, 24), 0.5), height: bh, rx: Math.min(2, bw / 3) });
      const hit = s("rect", { x: i * bw, y: 0, width: bw, height: height - padB, fill: "transparent" });
      hit.addEventListener("pointermove", (ev) => { rect.classList.add("hover"); showTip(ev, [[fmt(v, 4), `${label}${i + 1}`]]); });
      hit.addEventListener("pointerleave", () => { rect.classList.remove("hover"); hideTip(); });
      svg.append(rect, hit);
    });
    svg.append(s("line", { class: "axis", x1: 0, x2: width, y1: height - padB + 0.5, y2: height - padB + 0.5 }),
      s("text", { class: "tick", x: 0, y: height - 2 }, `${label}₁ = ${fmt(values[0])}`),
      s("text", { class: "tick", x: width, y: height - 2, "text-anchor": "end" }, `${label}${n} = ${fmt(values[n - 1])}`));
    return svg;
  }
  function sparkline(series, width, height) {
    const svg = s("svg", { class: "chart", width, height, viewBox: `0 0 ${width} ${height}` });
    if (!series.length) return svg;
    const lo = Math.min(...series), hi = Math.max(...series);
    const X = (i) => (series.length === 1 ? width / 2 : (i / (series.length - 1)) * (width - 6) + 3);
    const Y = (v) => (hi === lo ? height / 2 : height - 4 - ((v - lo) / (hi - lo)) * (height - 8));
    svg.append(s("path", { class: "line", "stroke-width": 1.5, d: series.map((v, i) => `${i ? "L" : "M"}${X(i)},${Y(v)}`).join(" ") }),
      s("circle", { class: "dot", cx: X(series.length - 1), cy: Y(series[series.length - 1]), r: 3 }));
    return svg;
  }
  function lineChart(series, width, height, label) {
    const svg = s("svg", { class: "chart", viewBox: `0 0 ${width} ${height}`, style: { maxWidth: width + "px" }, role: "img", "aria-label": label + " over time" });
    if (!series.length) return svg;
    const padL = 56, padB = 22, padT = 10, padR = 14;
    let lo = Math.min(...series), hi = Math.max(...series);
    if (hi === lo) { hi += 1; lo -= 1; }
    const X = (i) => padL + (series.length === 1 ? 0.5 : i / (series.length - 1)) * (width - padL - padR);
    const Y = (v) => padT + (1 - (v - lo) / (hi - lo)) * (height - padT - padB);
    for (let t = 0; t <= 4; t++) {
      const v = lo + ((hi - lo) * t) / 4, y = Y(v);
      svg.append(s("line", { class: "axis", x1: padL, x2: width - padR, y1: y, y2: y, opacity: 0.6 }), s("text", { class: "tick", x: padL - 8, y: y + 3, "text-anchor": "end" }, fmt(v)));
    }
    svg.append(s("text", { class: "tick", x: padL, y: height - 4 }, "call 1"), s("text", { class: "tick", x: width - padR, y: height - 4, "text-anchor": "end" }, `call ${series.length}`));
    const d = series.map((v, i) => `${i ? "L" : "M"}${X(i)},${Y(v)}`).join(" ");
    svg.append(s("path", { class: "area", d: `${d} L${X(series.length - 1)},${height - padB} L${X(0)},${height - padB} Z` }), s("path", { class: "line", d }));
    const xh = s("line", { class: "xhair", y1: padT, y2: height - padB, opacity: 0 });
    const dot = s("circle", { class: "dot", r: 4.5, opacity: 0 });
    svg.append(xh, dot);
    const hit = s("rect", { x: padL, y: 0, width: width - padL - padR, height: height - padB, fill: "transparent" });
    hit.addEventListener("pointermove", (ev) => {
      const r = svg.getBoundingClientRect();
      const px = ((ev.clientX - r.left) / r.width) * width;
      const i = Math.round(clamp((px - padL) / (width - padL - padR)) * (series.length - 1));
      xh.setAttribute("x1", X(i)); xh.setAttribute("x2", X(i)); xh.setAttribute("opacity", 1);
      dot.setAttribute("cx", X(i)); dot.setAttribute("cy", Y(series[i])); dot.setAttribute("opacity", 1);
      showTip(ev, [[fmt(series[i], 4), `${label} · call ${i + 1}`]]);
    });
    hit.addEventListener("pointerleave", () => { xh.setAttribute("opacity", 0); dot.setAttribute("opacity", 0); hideTip(); });
    svg.append(hit);
    return svg;
  }
  function mapsGrid(mp) {
    const data = f32(mp.data);
    const grid = h("div", { class: "grid-maps" });
    const per = mp.h * mp.w;
    for (let c = 0; c < mp.n; c++) {
      const cv = h("canvas", { width: mp.w, height: mp.h });
      const ctx = cv.getContext("2d");
      const img = ctx.createImageData(mp.w, mp.h);
      for (let i = 0; i < per; i++) {
        const col = colorOf(data[c * per + i], mp.vmin, mp.vmax, mp.diverging);
        img.data.set([col[0], col[1], col[2], 255], i * 4);
      }
      ctx.putImageData(img, 0, 0);
      cv.addEventListener("pointermove", (ev) => {
        const r = cv.getBoundingClientRect();
        const x = Math.floor(((ev.clientX - r.left) / r.width) * mp.w), y = Math.floor(((ev.clientY - r.top) / r.height) * mp.h);
        showTip(ev, [[fmt(data[c * per + y * mp.w + x], 4), `channel ${c} · (${y}, ${x})`]]);
      });
      cv.addEventListener("pointerleave", hideTip);
      grid.append(h("figure", null, cv, h("figcaption", { text: "ch " + c })));
    }
    return grid;
  }
  function filtersGrid(f) {
    const data = f32(f.data);
    const grid = h("div", { class: "grid-maps" });
    const per = f.kh * f.kw;
    for (let i = 0; i < f.n; i++) {
      const cv = h("canvas", { width: f.kw, height: f.kh });
      const ctx = cv.getContext("2d");
      const img = ctx.createImageData(f.kw, f.kh);
      for (let p = 0; p < per; p++) {
        let col;
        if (f.rgb) {
          col = [0, 1, 2].map((ch) => Math.round(255 * clamp(0.5 + 0.5 * data[i * 3 * per + ch * per + p] / f.vmax)));
        } else col = colorOf(data[i * per + p], -f.vmax, f.vmax, true);
        img.data.set([col[0], col[1], col[2], 255], p * 4);
      }
      ctx.putImageData(img, 0, 0);
      grid.append(h("figure", null, cv, h("figcaption", { text: "#" + i })));
    }
    return grid;
  }
  function miniHist(hist, W = 90, H = 18) {
    const c = hist.counts, m = Math.max(...c, 1), bw = W / c.length;
    const svg = s("svg", { class: "mini-hist", width: W, height: H, viewBox: `0 0 ${W} ${H}` });
    const e = hist.edges, z = e[0] < 0 && e[e.length - 1] > 0 ? ((0 - e[0]) / (e[e.length - 1] - e[0])) * W : null;
    c.forEach((v, i) => { if (v) svg.append(s("rect", { x: i * bw, y: H - (v / m) * H, width: Math.max(bw - 0.4, 0.6), height: (v / m) * H })); });
    if (z !== null) svg.append(s("line", { x1: z, x2: z, y1: 0, y2: H, class: "zero" }));
    return svg;
  }
  function tokenChips(tokens) {
    return h("div", { class: "tok-chips" }, ...tokens.map((t, i) => h("span", { class: "tok", title: `position ${i}`, text: t.replace(/ /g, "·") || "∅" })));
  }
  // ------------------------------------------------------ attribution: why this output
  const ATTR = {
    integrated_gradients: ["Integrated gradients", "blue pushed toward this output, red against — the values add up to how far it is from a blank input"],
    saliency: ["Saliency", "how strongly each input value moves this output"],
    grad_cam: ["Grad-CAM", "where the last convolution's features support this output"],
  };
  function attributionSection(A) {
    if (!A.methods.includes(S.attrMethod)) S.attrMethod = A.methods[0];
    if (!(S.attrTarget < A.targets.length)) S.attrTarget = 0;
    const m = S.attrMethod, ti = S.attrTarget, map = A.maps[m][ti];
    const tname = (t) => (t.label !== null && t.label !== undefined ? String(t.label).trim() || `#${t.index}` : `#${t.index}`);
    const body = h("div", { class: "attr-body" });
    const signed = m === "integrated_gradients";
    if (A.kind === "image") body.append(attrImage(A, map, signed));
    else {
      const vals = map, mx = Math.max(...vals.map(Math.abs), 1e-12);
      const col = (v) => (signed ? (v >= 0 ? hexRgb(P.theme.positive || "#58C4DD") : hexRgb(P.theme.negative || "#FC6255")) : SEQ(Math.sqrt(Math.abs(v) / mx)));
      if (A.kind === "tokens" && A.tokens && A.tokens.length === vals.length) {
        body.append(h("div", { class: "tok-chips" }, ...A.tokens.map((t, i) => {
          const a = Math.abs(vals[i]) / mx;
          return h("span", { class: "tok", title: `${fmt(vals[i], 3)} · position ${i}`, text: t.replace(/ /g, "·") || "∅",
            style: { background: rgba(col(vals[i]), 0.15 + 0.75 * a), color: a > 0.55 ? "#000" : "" } });
        })));
      } else {
        body.append(outputBars({ values: vals, probabilities: false, n: vals.length }, 380));
      }
    }
    return section("Why", `explaining ${tname(A.targets[ti])}${SAMPLE()}`,
      h("div", { class: "seg", role: "group", "aria-label": "method" }, ...A.methods.map((k) => h("button", {
        "aria-pressed": String(k === m), text: (ATTR[k] || [k])[0], onclick: () => { S.attrMethod = k; renderPanel(); } }))),
      h("div", { class: "seg attr-targets", role: "group", "aria-label": "output to explain" }, ...A.targets.map((t, i) => h("button", {
        "aria-pressed": String(i === ti), title: `output ${t.index} = ${fmt(t.score, 4)}`, text: tname(t).slice(0, 18),
        onclick: () => { S.attrTarget = i; renderPanel(); } }))),
      body, h("div", { class: "hint", text: (ATTR[m] || ["", ""])[1] }));
  }
  function attrImage(A, hm, signed) {
    const W = 300, H = Math.round(W * A.shape[0] / A.shape[1]);
    const cv = h("canvas", { class: "attr-canvas", width: W * 2, height: H * 2, style: { width: W + "px", height: H + "px" } });
    const ctx = cv.getContext("2d");
    const base = new Image();
    const data = f32(hm.data), mx = Math.max(...Array.from(data, Math.abs), 1e-12);
    const over = document.createElement("canvas");
    over.width = hm.cols; over.height = hm.rows;
    const octx = over.getContext("2d"), img = octx.createImageData(hm.cols, hm.rows);
    const pos = hexRgb(P.theme.positive || "#58C4DD"), neg = hexRgb(P.theme.negative || "#FC6255");
    for (let i = 0; i < data.length; i++) {
      const r = Math.abs(data[i]) / mx;
      // signed (IG): colour by direction, opacity by strength; unsigned (saliency, Grad-CAM): only the strongest
      // parts light up (a heat ramp over the top 80 %), so the picture underneath stays readable
      const a = signed ? Math.pow(r, 0.7) : Math.max(0, (r - 0.2) / 0.8);
      const c = signed ? (data[i] >= 0 ? pos : neg) : SEQ(a);
      img.data.set([c[0], c[1], c[2], Math.round((signed ? 235 : 215) * Math.pow(a, signed ? 1 : 0.8))], i * 4);
    }
    octx.putImageData(img, 0, 0);
    base.onload = () => {
      ctx.imageSmoothingEnabled = false;
      ctx.globalAlpha = signed ? 0.55 : 0.8;
      ctx.drawImage(base, 0, 0, cv.width, cv.height);
      ctx.globalAlpha = 1;
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(over, 0, 0, cv.width, cv.height);
    };
    base.src = A.base;
    return cv;
  }
  // too many classes for bars (a language model's vocabulary, ImageNet): the k most likely, with names
  function topkSection(out) {
    const rows = out.topk, W = 380, rowH = 20, labW = 120, valW = 64;
    const svg = s("svg", { class: "chart outbars", viewBox: `0 0 ${W} ${rows.length * rowH + 6}`, width: W, height: rows.length * rowH + 6, role: "img", "aria-label": "top predictions" });
    const pmax = Math.max(...rows.map((r) => r.p), 1e-9);
    rows.forEach((r, i) => {
      const y = i * rowH + 3, has = r.label !== null && r.label !== undefined;
      const short = has && r.label.length > 17 ? r.label.slice(0, 16) + "…" : r.label;
      const name = !has ? `#${r.id}` : out.labelKind === "class" ? r.label : `“${r.label}”`;  // quotes show a token's spaces
      const shown = !has ? name : out.labelKind === "class" ? short : `“${short}”`;
      svg.append(s("text", { class: "lbl", x: labW - 8, y: y + 12, "text-anchor": "end" }, shown),
        s("rect", { class: "b" + (i === 0 ? " top" : ""), x: labW, y: y + 2, width: Math.max(1, (r.p / pmax) * (W - labW - valW)), height: 12, rx: 3 }),
        s("text", { class: "val", x: W - valW + 8, y: y + 12 }, (r.p * 100).toFixed(r.p < 0.001 ? 3 : 1) + "%"));
      const hit = s("rect", { x: 0, y, width: W, height: rowH, fill: "transparent" });
      hit.addEventListener("pointermove", (ev) => showTip(ev, [[(r.p * 100).toFixed(3) + "%", `class ${r.id} ${name}`]]));
      hit.addEventListener("pointerleave", hideTip);
      svg.append(hit);
    });
    const after = out.position !== null && out.position !== undefined;
    const what = after && P.tokens && P.tokens[out.position] !== undefined ? `next token after “${P.tokens[out.position]}”` : after ? `at position ${out.position}` : `top ${rows.length} of ${out.n}`;
    return section(after ? "Prediction" : "Output", `${what}${SAMPLE()}`, svg,
      h("div", { class: "hint", text: `top ${rows.length} of ${out.n.toLocaleString()}` }));
  }
  function outputBars(out, width = 380) {
    const vals = out.values, n = vals.length, base = out.baseline, names = out.labels;
    const rowH = 20, labW = names ? Math.min(120, 16 + 6.4 * Math.max(...names.map((t) => Math.min(t.length, 16)))) : 34, valW = 64;
    const height = n * rowH + 6;
    const svg = s("svg", { class: "chart outbars", viewBox: `0 0 ${width} ${height}`, width, height, role: "img", "aria-label": "output values" });
    const lo = Math.min(0, ...vals, ...(base || [])), hi = Math.max(0, ...vals, ...(base || []));
    const span = hi - lo || 1;
    const X = (v) => labW + ((v - lo) / span) * (width - labW - valW);
    let top = 0;
    vals.forEach((v, i) => { if (v > vals[top]) top = i; });
    vals.forEach((v, i) => {
      const y = i * rowH + 3;
      const x0 = X(Math.min(0, v)), x1 = X(Math.max(0, v));
      svg.append(s("text", { class: "lbl", x: labW - 8, y: y + 12, "text-anchor": "end" }, names ? names[i].slice(0, 16) : String(i)));
      svg.append(s("rect", { class: "b" + (i === top ? " top" : ""), x: x0, y: y + 2, width: Math.max(x1 - x0, 1), height: 12, rx: 3 }));
      if (base) {
        const bx = X(base[i]);
        svg.append(s("line", { class: "base", x1: bx, x2: bx, y1: y, y2: y + 16 }));
      }
      const txt = out.probabilities ? (v * 100).toFixed(1) + "%" : fmt(v);
      svg.append(s("text", { class: "val", x: width - valW + 8, y: y + 12 }, txt));
      const hit = s("rect", { x: 0, y, width, height: rowH, fill: "transparent" });
      hit.addEventListener("pointermove", (ev) => showTip(ev, [[fmt(v, 5), `output ${i}` + (names ? ` “${names[i]}”` : "")], base ? [fmt(base[i], 5), "before edits"] : null].filter(Boolean)));
      hit.addEventListener("pointerleave", hideTip);
      svg.append(hit);
    });
    if (lo < 0) svg.append(s("line", { class: "axis", x1: X(0), x2: X(0), y1: 0, y2: height }));
    return svg;
  }
  function statsGrid(st) {
    const items = [["mean", st.mean], ["std", st.std], ["‖·‖₂", st.norm], ["min", st.min], ["max", st.max], ["zeros", st.sparsity === null ? null : st.sparsity * 100]];
    return h("div", { class: "stats" }, ...items.map(([k, v]) => h("div", { class: "stat" }, h("div", { class: "k", text: k }),
      h("div", { class: "v", text: k === "zeros" ? (v === null ? "—" : v.toFixed(1) + "%") : fmt(v) }))));
  }

  // ------------------------------------------------------------------ panel
  function section(title, aside, ...kids) {
    return h("div", { class: "p-sec" }, h("h4", null, title, aside ? h("span", { class: "aside", text: aside }) : null), ...kids);
  }
  function renderPanel() {
    panel.replaceChildren();
    if (S.tab === "neurons" && S.neuron) return renderNeuronPanel();
    const n = S.selected !== null ? nodeById(S.selected) : null;
    if (!n) return renderOverview();
    const fam = n.color;
    const head = h("div", { class: "p-head" },
      h("h2", { class: "p-kind" }, h("i", { style: { background: fam } }), n.op === "container" || n.collapsed ? n.type : n.kind),
      h("div", { class: "p-path code", title: "reach it from Python", text: n.id === "" ? "net" : `net["${n.id}"]` }),
      h("div", { class: "p-meta" },
        n.type && n.type !== n.kind ? h("span", { text: n.type }) : null,
        n.in_shapes && n.in_shapes[0] ? h("span", { text: "in " + shapeStr(n.in_shapes[0]) }) : null,
        n.out_shapes && n.out_shapes[0] ? h("span", { text: "out " + shapeStr(n.out_shapes[0]) }) : null,
        paramCount(n) ? h("span", { text: paramCount(n).toLocaleString() + " params" }) : null,
        ...Object.entries(n.attrs || {}).filter(([k, v]) => !["prims", "collapsed", "roles", "shared"].includes(k) && v !== null && typeof v !== "object" && String(v).length < 24).slice(0, 6).map(([k, v]) => h("span", { text: `${k}=${v}` })),
      ));
    panel.append(head);
    const ex = explainCard(n);
    if (ex) panel.append(ex);
    const AR = P.arch || {};
    if (AR.recurrent && AR.recurrent[n.id]) panel.append(recurrentSection(AR.recurrent[n.id]));
    for (const r of AR.moe || []) if (r.block === n.id || r.router === n.id) panel.append(routingSection(r));
    for (const m of AR.ssm || []) if (m.mixer === n.id) panel.append(ssmSection(m));
    if (AR.graph && AR.graph.layers.some((l) => l.id === n.id)) panel.append(graphInputSection(AR.graph, n.id));
    const patch = patchSize(n);
    if (patch && P.inputImage) panel.append(patchSection(patch));
    if (P.dataset && P.dataset.top && (P.dataset.top[n.id] || []).length) panel.append(datasetSection(n.id));
    const gm = P.glyphMeta && P.glyphMeta[n.id], gsvg = glyphOf(n.id);
    if (UI.style === "expressive" && gm && gsvg && GLYPH_CAPTION[gm.type]) {
      const [bx, by, bw, bh] = gm.box;
      const big = s("svg", { class: "glyph-big", viewBox: `${bx} ${by} ${bw} ${bh}`, role: "img",
        "aria-label": GLYPH_CAPTION[gm.type], style: { height: Math.min(220, 380 * bh / bw) + "px" } });
      big.innerHTML = gsvg;
      panel.append(section(GLYPH_TITLE[gm.type] || "Shape", null, big, h("div", { class: "hint", text: GLYPH_CAPTION[gm.type] })));
    }
    const eq = (P.equations && P.equations.perNode || {})[n.id];
    if (eq && eq.lines && eq.lines.length) {
      const box = h("div", { class: "p-eq" });
      for (const ln of eq.lines) box.append(tex(h("div", { class: "wrap" }), `${ln.lhs} = ${ln.rhs}`, false));
      for (const d of eq.definitions || []) box.append(tex(h("div", { class: "def wrap" }), d.latex, false));
      for (const d of eq.legend || []) box.append(tex(h("div", { class: "def wrap" }), d.latex, false));
      panel.append(section("Equation", null, box));
    }
    const sc = (P.scalars || {})[n.id];
    if (sc && sc.lines && sc.lines.length) {
      const box = h("div", { class: "p-eq" });
      for (const ln of sc.lines) {
        box.append(tex(h("div", { class: "ov" }), `${ln.lhs} = ${ln.rhs}` + (ln.note ? `\\qquad \\text{${ln.note.replace(/[{}\\]/g, "")}}` : ""), false));
      }
      for (const d of sc.legend || []) box.append(h("div", { class: "def", text: d.text }));
      panel.append(section("One output value, with real weights", `top 5 of ${sc.terms} terms`, box,
        h("div", { class: "hint", text: "The most active channel at its strongest position for this input; net.equation('scalar', node=…, neuron=(c, i, j)) unrolls any other." })));
    }
    const tl = trainingFor(n.id);
    if (tl) panel.append(layerTrainingSection(tl));
    if (LIVE && (n.params || []).length) panel.append(liveControls(n));
    if (LIVE && ["module", "function", "input", "param"].includes(n.op) && !n.collapsed) panel.append(interveneSection(n));
    const a = actFor(n);
    if (a) panel.append(activationSection(n, a));
    const params = (n.params || []).filter((p) => P.params[p.path]);
    const cmp = compareOf(n.id);
    if (cmp) {
      const L = cmp.l, nm = P.compare.names;
      panel.append(section(`${nm[0]} → ${nm[1]}`, "this layer",
        h("div", { class: "stats" },
          h("div", { class: "stat" }, h("div", { class: "k", text: "weights moved" }), h("div", { class: "v", text: L.weight === null ? "—" : pct(L.weight) })),
          h("div", { class: "stat" }, h("div", { class: "k", text: "output moved" }), h("div", { class: "v", text: L.act === null ? "—" : pct(L.act) })),
          h("div", { class: "stat" }, h("div", { class: "k", text: "same direction" }), h("div", { class: "v", text: L.cos === null ? "—" : fmt(L.cos, 3) })))));
    }
    for (const p of params) panel.append(paramSection(p, P.params[p.path]));
    if (n.op === "container") {
      const kids = level().nodes.filter((m) => m.parent === n.id);
      panel.append(section("Contains", `${kids.length}`, h("div", { class: "hint", text: kids.map((k) => k.label || k.id).join(" · ") })));
    }
  }
  function paramSection(p, pp) {
    const kids = [];
    kids.push(statsGrid(pp.stats));
    if (pp.bank) {
      kids.push(h("div", { class: "p-sub", text: `filter bank (${pp.bank.n} of ${pp.bank.total}), each ${pp.bank.dims.join("×")} — drag to rotate` }), bankView(pp.bank));
    }
    if (pp.filters) {
      kids.push(h("div", { class: "p-sub", text: `filters (${pp.filters.n} of ${pp.filters.total}${pp.filters.rgb ? ", RGB" : ", mean over input channels"})` }), filtersGrid(pp.filters));
    }
    if (pp.heat) {
      const editable = LIVE && !pp.heat.downsampled;
      const onCell = editable ? (r, c, v, ev) => cellEditor(ev, p.path, pp, r, c, v) : null;
      kids.push(h("div", { class: "p-sub", text: (pp.heat.rows > 1 ? `${pp.heat.rowLabel || "rows"} × ${pp.heat.colLabel || "cols"}` : "values") + (editable ? " · click a cell to set it" : "") }),
        heatCanvas(pp.heat, 380, 260, onCell), colorbar(pp.heat));
    }
    kids.push(h("div", { class: "p-sub", text: "distribution" }), histogram(pp.hist));
    if (pp.svd && pp.svd.length > 1) kids.push(h("div", { class: "p-sub", text: "singular values" }), barChart(pp.svd));
    const title = h("span", null, p.name, pp.edited ? h("span", { class: "p-sub" }, h("span", { class: "edited", text: "edited" })) : null);
    return section(title, shapeStr(pp.shape), ...kids);
  }
  function activationSection(n, a) {
    const kids = [statsGrid(a.stats)];
    if (a.vector) {
      const v = a.vector;
      const hm = { rows: 1, cols: Math.min(v.data ? f32(v.data).length : 0, 4096), data: v.data, vmin: v.vmin, vmax: v.vmax, diverging: v.diverging, downsampled: v.downsampled, orig: [1, v.n], colLabel: "neuron" };
      kids.push(h("div", { class: "p-sub", text: `${v.n} values` + (v.downsampled ? " (block means)" : "") }), heatCanvas(hm), colorbar(hm));
      if (a.top) kids.push(h("div", { class: "toplist" }, ...a.top.slice(0, 4).flatMap(([i, val]) => [h("span", { text: "#" + i }), h("span", { text: fmt(val, 4) })])));
    }
    if (a.maps) {
      kids.push(h("div", { class: "p-sub", text: `volume · ${a.maps.channels} channels, drag to rotate, click a slice` }),
        stackView(a.maps.n, a.maps.h, a.maps.w, f32(a.maps.data), a.maps.vmin, a.maps.vmax, a.maps.diverging, "channel"));
      kids.push(h("div", { class: "p-sub", text: `feature maps (${a.maps.n} of ${a.maps.channels} channels, shared scale)` }), mapsGrid(a.maps), colorbar({ ...a.maps, orig: [] }));
    }
    if (a.matrix) kids.push(h("div", { class: "p-sub", text: `${a.matrix.rowLabel} × ${a.matrix.colLabel}` }), heatCanvas(a.matrix), colorbar(a.matrix));
    if (a.attention) {
      const grid = h("div", { class: "grid-maps" });
      a.attention.heads.forEach((hm, i) => grid.append(h("figure", null, heatCanvas(hm, 90, 90), h("figcaption", { text: "head " + i }))));
      kids.push(h("div", { class: "p-sub", text: `attention patterns (${a.attention.heads.length} of ${a.attention.total} heads)` }), grid);
    }
    kids.push(h("div", { class: "p-sub", text: "distribution" }), histogram(a.hist));
    const dl = n.op === "input" ? "∂L/∂x" : "∂L/∂(output)";
    if (a.grad) kids.push(h("div", { class: "p-sub", text: `gradient ${dl}` }), statsGrid({ ...a.grad }));
    if (a.gradView) kids.push(...gradViewKids(a.gradView, dl));
    return section("Activation", shapeStr(a.shape) + SAMPLE(), ...kids);
  }
  function renderOverview() {
    const tiles = h("div", { class: "tiles" },
      h("div", { class: "tile" }, h("div", { class: "k", text: "parameters" }), h("div", { class: "v", text: compact(P.meta.params) })),
      h("div", { class: "tile" }, h("div", { class: "k", text: "nodes" }), h("div", { class: "v", text: String(P.meta.nodes) })),
      LIVE || P.meta.edits ? h("div", { class: "tile" }, h("div", { class: "k", text: "edits" }), h("div", { class: "v", text: String(P.meta.edits || 0) })) : null);
    panel.append(h("div", { class: "p-head" }, h("h2", { class: "p-kind", text: UI.title || P.meta.name }),
      UI.note ? h("p", { class: "p-note", text: UI.note }) : null,
      h("div", { class: "p-path", text: `${P.meta.framework} · input ${P.meta.input.map(shapeStr).join(", ") || "—"}` })),
    P.meta.describe ? section("In short", null, h("p", { class: "p-note in-short", text: P.meta.describe })) : null,
    section("Start here", null, h("p", { class: "p-note start", text: "Click any layer to see what it does, its equation and its numbers. Hover anything to read its values." }),
      h("div", { class: "hint" }, "Press ", h("kbd", { text: "?" }), " for the keys, how to read the colours, and a short tour.")),
    section("Overview", null, tiles));
    if (LIVE && P.input) panel.append(inputPlayground(P.input));
    if (LIVE && (P.interventions || []).length) {
      panel.append(section("Active interventions", String(P.interventions.length), h("ul", { class: "hist-list" },
        ...P.interventions.map((i) => h("li", null, h("span", { text: `${i.node}: ${i.desc}`, style: { flex: "1" } }),
          h("button", { class: "btn", text: "remove", onclick: () => api("/api/remove_intervention", { id: i.id }) }))))));
    }
    if (P.arch && P.arch.graph) panel.append(graphInputSection(P.arch.graph, null));
    if (P.tokens && P.tokens.length && !(LIVE && P.input)) panel.append(section("Input tokens", `${P.tokens.length} tokens${SAMPLE()}`, tokenChips(P.tokens)));
    if (P.output && P.output.values) {
      panel.append(section("Output", (P.output.probabilities ? "probabilities" : "values") + SAMPLE(), outputBars(P.output),
        P.output.baseline ? h("div", { class: "hint", text: "White ticks mark the output when this session started." }) : null));
    } else if (P.output && P.output.topk) panel.append(topkSection(P.output));
    else if (P.output && P.output.image) panel.append(section("Output", `${P.output.shape.join("×")}${SAMPLE()}`,
      h("img", { class: "out-img", src: P.output.image, alt: "network output" })));
    else if (P.output && P.output.heat) panel.append(section("Output", `${P.output.shape.join("×")}${SAMPLE()}`,
      heatCanvas(P.output.heat, 380, 220), colorbar(P.output.heat)));
    if (P.attribution) panel.append(attributionSection(P.attribution));
    const eq = P.equations && P.equations.modes && P.equations.modes.compact;
    if (eq && eq.lines && eq.lines.length) {
      const box = h("div", { class: "p-eq" });
      const lines = eq.lines.slice(0, 8);
      for (const ln of lines) box.append(tex(h("div", { class: "ov" }), `${ln.lhs} = ${ln.rhs}`, false));
      if (eq.lines.length > lines.length) box.append(h("div", { class: "hint", text: `… ${eq.lines.length - lines.length} more lines in the Equations tab` }));
      panel.append(section("The whole network", null, box));
    }
    if (P.history && P.history.length) {
      panel.append(section("Edit history", String(P.history.length), h("ul", { class: "hist-list" },
        ...P.history.map((e) => h("li", { class: e.applied ? "" : "undone", text: e.label })))));
    }
  }
  function renderNeuronPanel() {
    const [li, k] = S.neuron;
    const NV = P.neurons, L = NV.layers[li];
    const idx = L.shown[k];
    const v = L.values ? L.values[k] : null;
    panel.append(h("div", { class: "p-head" },
      h("h2", { class: "p-kind", text: li === 0 ? `Input ${idx}` : `Neuron ${idx}` }),
      h("div", { class: "p-path", text: li === 0 ? "network input" : `${L.node} · layer ${li}` }),
      h("div", { class: "p-meta" }, h("span", { text: "activation " + fmt(v, 4) }))));
    const eq = NV.equations[`${li}:${idx}`];
    if (eq) {
      const box = h("div", { class: "p-eq" });
      for (const ln of eq.lines) {
        const row = tex(h("div", { class: "wrap" }), `${ln.lhs} = ${ln.rhs}`, false);
        if (ln.note) row.append(h("span", { class: "hint", text: " " + ln.note }));
        box.append(row);
      }
      for (const d of eq.definitions || []) box.append(tex(h("div", { class: "def wrap" }), d.latex, false));
      panel.append(section("This neuron, with real weights", "top-4 terms", box));
    }
    const inc = NV.incoming[`${li}:${idx}`];
    if (inc) panel.append(section("Incoming weights as an image", `${inc.orig.join("×")}`, heatCanvas(inc, 220, 220), colorbar(inc)));
    if (LIVE && li > 0) {
      const node = L.node;
      panel.append(section("Edit this neuron", null, h("div", { class: "ctl-row" },
        h("button", { class: "btn", text: "Zero its weights", onclick: () => api("/api/neurons", { node, neurons: [idx], op: "zero" }) }),
        h("button", { class: "btn", text: "×2 weights", onclick: () => api("/api/neurons", { node, neurons: [idx], op: "scale", args: { factor: 2 } }) }),
        h("button", { class: "btn", text: "Negate", onclick: () => api("/api/neurons", { node, neurons: [idx], op: "negate" }) }),
        h("button", { class: "btn", text: "Ablate activation", onclick: () => api("/api/intervene", { node: L.post || node, type: "ablate", neurons: [idx] }) }),
      )));
    }
    if (P.output && P.output.values) panel.append(section("Output", SAMPLE(""), outputBars(P.output)));
    else if (P.output && P.output.topk) panel.append(topkSection(P.output));
  }

  // ------------------------------------------------------------------- live
  function liveControls(n) {
    const target = n.id;
    const mk = (label, input, btn, show) => h("div", { class: "ctl" }, h("label", { text: label }), input, h("div", null, show || null, btn));
    const range = (min, max, step, val) => h("input", { type: "range", min, max, step, value: val });
    const num = (val, attrs = {}) => h("input", { type: "number", value: val, ...attrs });
    const sc = range(0, 2, 0.05, 1), scV = h("span", { class: "v", text: "×1.00" });
    sc.addEventListener("input", () => { scV.textContent = "×" + Number(sc.value).toFixed(2); });
    const pr = range(0, 0.95, 0.05, 0.5), prV = h("span", { class: "v", text: "50%" });
    pr.addEventListener("input", () => { prV.textContent = Math.round(pr.value * 100) + "%"; });
    const ns = num(0.1, { step: 0.05, min: 0 });
    const rk = num(1, { min: 1, step: 1 });
    const qb = h("select", null, ...[1, 2, 3, 4, 6, 8].map((b) => h("option", { value: b, text: b + "-bit", selected: b === 4 })));
    const nr = h("input", { type: "text", placeholder: "e.g. 0, 3, 7" });
    const edit = (op, args, params) => api("/api/edit", { op, target, args, params });
    const neurons = () => nr.value.split(",").map((x) => parseInt(x.trim(), 10)).filter((x) => Number.isFinite(x));
    const hasOut = Object.values(n.roles || {}).some((r) => r.out_axis !== null && r.out_axis !== undefined);
    return section("Edit weights", "live",
      mk("Scale", sc, h("button", { class: "btn", text: "Apply", onclick: () => edit("scale", { factor: Number(sc.value) }) }), scV),
      mk("Prune", pr, h("button", { class: "btn", text: "Apply", onclick: () => edit("prune", { amount: Number(pr.value) }, ["weight"]) }), prV),
      mk("Noise σ (rel.)", ns, h("button", { class: "btn", text: "Apply", onclick: () => edit("noise", { std: Number(ns.value), relative: true, seed: Math.floor(Math.random() * 1e6) }) })),
      mk("Low rank k", rk, h("button", { class: "btn", text: "Apply", onclick: () => edit("low_rank", { rank: Number(rk.value) }, ["weight"]) })),
      mk("Quantize", qb, h("button", { class: "btn", text: "Apply", onclick: () => edit("quantize", { bits: Number(qb.value) }, ["weight"]) })),
      hasOut ? mk("Neurons", nr, h("div", { style: { display: "flex", gap: "4px" } },
        h("button", { class: "btn", text: "Zero", onclick: () => api("/api/neurons", { node: target, neurons: neurons(), op: "zero" }) }),
        h("button", { class: "btn", text: "Ablate", title: "activation-level knockout (intervention)", onclick: () => api("/api/intervene", { node: target, type: "ablate", neurons: neurons() }) }))) : null,
      h("div", { class: "ctl-row" },
        h("button", { class: "btn", text: "Re-initialize", onclick: () => edit("reinit", { method: "kaiming", seed: 0 }, ["weight"]) }),
        h("button", { class: "btn", text: "Restore this layer", onclick: () => api("/api/restore", { paths: (n.params || []).map((p) => p.path) }) }),
        h("button", { class: "btn", text: "Clear interventions", onclick: () => api("/api/clear_interventions") })));
  }
  function interveneSection(n) {
    const nr = h("input", { type: "text", placeholder: "blank = every neuron" });
    const neurons = () => { const v = nr.value.split(",").map((x) => parseInt(x.trim(), 10)).filter(Number.isFinite); return v.length ? v : null; };
    const lo = h("input", { type: "number", value: "0", step: "any" }), hi = h("input", { type: "number", value: "1", step: "any" });
    const sd = h("input", { type: "number", value: "0.5", step: "any", min: "0" });
    const go = (body) => api("/api/intervene", { node: n.id, neurons: neurons(), ...body });
    const active = (P.interventions || []).filter((i) => i.node === n.id);
    return section("Intervene on activations", "live",
      h("div", { class: "ctl" }, h("label", { text: "Neurons" }), nr, h("span")),
      h("div", { class: "ctl-row" },
        h("button", { class: "btn", text: "Ablate", title: "set to 0", onclick: () => go({ type: "ablate" }) }),
        h("button", { class: "btn", text: "Mean-ablate", onclick: () => go({ type: "mean_ablate" }) }),
        h("button", { class: "btn", text: "×2", onclick: () => go({ type: "amplify", factor: 2 }) }),
        h("button", { class: "btn", text: "×−1", onclick: () => go({ type: "amplify", factor: -1 }) })),
      h("div", { class: "ctl" }, h("label", { text: "Clamp" }), h("div", { style: { display: "flex", gap: "6px" } }, lo, hi),
        h("button", { class: "btn", text: "Apply", onclick: () => go({ type: "clamp", min: Number(lo.value), max: Number(hi.value) }) })),
      h("div", { class: "ctl" }, h("label", { text: "Noise σ" }), sd,
        h("button", { class: "btn", text: "Apply", onclick: () => go({ type: "noise", std: Number(sd.value), seed: Math.floor(Math.random() * 1e6) }) })),
      active.length ? h("ul", { class: "hist-list" }, ...active.map((i) => h("li", null, h("span", { text: i.desc, style: { flex: "1" } }),
        h("button", { class: "btn", text: "remove", onclick: () => api("/api/remove_intervention", { id: i.id }) })))) : null);
  }

  /** Live input playground: draw on images, drag vector entries, type token ids, perturb anything. */
  function inputPlayground(inp) {
    const kids = [];
    const send = (body) => api("/api/input", body);
    if (inp.kind === "image") {
      const rows = inp.rows, cols = inp.cols;
      const data = Float32Array.from(f32(inp.data));
      let lo = Math.min(inp.vmin, 0), hi = Math.max(inp.vmax, 1);
      const scale = Math.max(3, Math.floor(300 / Math.max(rows, cols)));
      const cv = h("canvas", { class: "draw", width: cols, height: rows, style: { width: cols * scale + "px", height: rows * scale + "px" } });
      const ctx = cv.getContext("2d");
      const paint = () => {
        const img = ctx.createImageData(cols, rows);
        for (let i = 0; i < data.length; i++) {
          const g = Math.round(255 * clamp((data[i] - lo) / (hi - lo || 1)));
          img.data.set([g, g, g, 255], i * 4);
        }
        ctx.putImageData(img, 0, 0);
      };
      paint();
      let down = false, erase = false;
      const brush = (ev) => {
        const r = cv.getBoundingClientRect();
        const x = ((ev.clientX - r.left) / r.width) * cols, y = ((ev.clientY - r.top) / r.height) * rows;
        const R = Math.max(1.2, Math.min(rows, cols) / 16);
        for (let i = Math.floor(y - R - 1); i <= y + R + 1; i++) for (let j = Math.floor(x - R - 1); j <= x + R + 1; j++) {
          if (i < 0 || j < 0 || i >= rows || j >= cols) continue;
          const d = Math.hypot(i + 0.5 - y, j + 0.5 - x), w = clamp(1 - d / R);
          if (w <= 0) continue;
          const k = i * cols + j;
          data[k] = erase ? data[k] + (lo - data[k]) * w : data[k] + (hi - data[k]) * w;
        }
        paint();
      };
      cv.addEventListener("contextmenu", (e) => e.preventDefault());
      cv.addEventListener("pointerdown", (ev) => { down = true; erase = ev.button === 2 || ev.shiftKey; cv.setPointerCapture(ev.pointerId); brush(ev); });
      cv.addEventListener("pointermove", (ev) => { if (down) brush(ev); });
      cv.addEventListener("pointerup", () => {
        if (!down) return;
        down = false;
        const arr = [];
        for (let i = 0; i < rows; i++) arr.push(Array.from(data.subarray(i * cols, (i + 1) * cols)));
        send({ op: "set_image", data: arr });
      });
      kids.push(h("div", { class: "hint", text: `Draw with the mouse (right-click or shift to erase).${inp.channels > 1 ? ` Drawing sets all ${inp.channels} channels.` : ""}` }), cv,
        h("div", { class: "ctl-row" },
          h("button", { class: "btn", text: "Clear", onclick: () => { data.fill(lo); paint(); send({ op: "set_image", data: Array.from({ length: rows }, () => new Array(cols).fill(lo)) }); } }),
          h("button", { class: "btn", text: "Invert", onclick: () => { const arr = []; for (let i = 0; i < rows; i++) arr.push(Array.from(data.subarray(i * cols, (i + 1) * cols), (v) => hi + lo - v)); send({ op: "set_image", data: arr }); } }),
          h("button", { class: "btn", text: "Noise", onclick: () => send({ op: "noise", std: 0.15 * (hi - lo), seed: Math.floor(Math.random() * 1e6) }) }),
          h("button", { class: "btn", text: "Reset", onclick: () => send({ op: "reset" }) })));
    } else if (inp.kind === "vector") {
      const vals = [...inp.values];
      const grid = h("div", { class: "vec-grid" });
      vals.forEach((v, i) => {
        const r = h("input", { type: "range", min: Math.min(-3, v - 1).toFixed(2), max: Math.max(3, v + 1).toFixed(2), step: 0.01, value: v });
        const lab = h("span", { class: "v", text: fmt(v) });
        r.addEventListener("input", () => { lab.textContent = fmt(Number(r.value)); });
        r.addEventListener("change", () => { vals[i] = Number(r.value); send({ op: "set_vector", data: vals }); });
        grid.append(h("label", { text: `x${i}` }), r, lab);
      });
      kids.push(grid, h("div", { class: "ctl-row" }, h("button", { class: "btn", text: "Reset", onclick: () => send({ op: "reset" }) })));
    } else if (inp.kind === "tokens") {
      const t = h("input", { type: "text", value: inp.values.join(", ") });
      if (P.tokens && P.tokens.length) kids.push(tokenChips(P.tokens));
      kids.push(h("div", { class: "hint", text: `${inp.n} token ids${SAMPLE()}` }), h("div", { class: "ctl" }, h("label", { text: "ids" }), t,
        h("button", { class: "btn", text: "Apply", onclick: () => send({ op: "set_tokens", data: t.value.split(",").map((x) => parseInt(x.trim(), 10)).filter(Number.isFinite) }) })),
        h("div", { class: "ctl-row" }, h("button", { class: "btn", text: "Reset", onclick: () => send({ op: "reset" }) })));
    }
    if (inp.kind !== "tokens") kids.push(h("div", { class: "ctl-row" },
      h("button", { class: "btn", text: "×0.5", onclick: () => send({ op: "scale", factor: 0.5 }) }),
      h("button", { class: "btn", text: "×2", onclick: () => send({ op: "scale", factor: 2 }) }),
      inp.kind === "other" ? h("button", { class: "btn", text: "Noise", onclick: () => send({ op: "noise", std: 0.1, seed: Math.floor(Math.random() * 1e6) }) }) : null,
      inp.kind === "other" ? h("button", { class: "btn", text: "Reset", onclick: () => send({ op: "reset" }) }) : null));
    return section("Input playground", shapeStr(inp.shape), ...kids);
  }

  async function api(path, body) {
    if (!LIVE) return;
    document.body.classList.add("busy");
    try {
      const r = await fetch(API + path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
      const j = await r.json();
      if (!r.ok || j.error) { toast(j.error || `HTTP ${r.status}`, true); return; }
      applyPayload(j.payload, j.message);
    } catch (e) {
      toast("Lost connection to the Python session: " + e.message, true);
    } finally {
      document.body.classList.remove("busy");
    }
  }
  function applyPayload(np, message) {
    const scrolls = [panel, stage.querySelector(".tableview, .eqview")].map((el) => (el ? el.scrollTop : 0));
    P = np;
    SV = (P.meta && P.meta.serverVersion) || SV;
    texCache.clear();
    DIV = ramp(P.theme.diverging); SEQ = ramp(P.theme.sequential);
    renderHeader();
    const keepView = S.view;
    renderStage();
    S.view = keepView;
    renderPanel();
    const els = [panel, stage.querySelector(".tableview, .eqview")];
    els.forEach((el, i) => { if (el && scrolls[i]) el.scrollTop = scrolls[i]; });
    if (message) toast(message);
  }

  // "watch it learn": a training loop calls srv.update(...); the page follows by polling a version counter
  let SV = (P.meta && P.meta.serverVersion) || 0, pollBusy = false, pointerDown = false;
  S.paused = false;
  async function poll() {
    if (pollBusy || S.paused || document.hidden || pointerDown) return;
    const ae = document.activeElement;
    if (ae && /^(INPUT|TEXTAREA|SELECT)$/.test(ae.tagName) && ae !== document.body) return;  // don't yank a control away
    pollBusy = true;
    try {
      const v = await (await fetch(API + "/api/version", { cache: "no-store" })).json();
      if (v.version !== SV) {
        const np = await (await fetch(API + "/api/state", { cache: "no-store" })).json();
        if (!S.paused) applyPayload(np, null);
      }
    } catch (e) { /* the Python session may have ended; keep the last state on screen */ }
    finally { pollBusy = false; }
  }
  if (LIVE) {
    document.addEventListener("pointerdown", () => { pointerDown = true; }, true);
    const release = () => { pointerDown = false; };  // also when the pointer is let go outside the frame
    document.addEventListener("pointerup", release, true);
    document.addEventListener("pointercancel", release, true);
    window.addEventListener("blur", release);
    document.addEventListener("pointermove", (ev) => { if (!ev.buttons) release(); }, true);
    setInterval(poll, 1000);
  }

  async function exportEdits() {
    try {
      const r = await fetch(API + "/api/history");
      const j = await r.json();
      const blob = new Blob([JSON.stringify(j.edits, null, 2)], { type: "application/json" });
      const a = h("a", { href: URL.createObjectURL(blob), download: `${P.meta.name}-edits.json` });
      document.body.append(a); a.click(); a.remove();
    } catch (e) { toast("export failed: " + e.message, true); }
  }

  // ------------------------------------------------------- guidance: explanations, captions, help, tour
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { if (v) localStorage.setItem(k, v); else localStorage.removeItem(k); } catch (e) { /* private mode */ } },
  };
  const CAPTIONS = {  // one sentence per view: what it shows, and how to explore it
    graph: "Data flows from the top down through every layer. Click a layer to see what it does, its equation and its numbers · drag to pan · scroll to zoom · ↑ ↓ walk the layers.",
    volumes: "Every block is one tensor on its way through the network — height × width, channels along the flow. Drag to orbit · shift-drag to pan · scroll to zoom · hover a face to read its values.",
    neurons: "The classic picture: circles are neurons (brighter = more active), lines are weights (blue +, red −, thicker = larger). Click a neuron to see its weighted sum with the real numbers.",
    attention: "Inside attention: queries meet keys (scores), a softmax turns the scores into weights P, and P mixes the values (P·V). Pick a layer and a head.",
    diffusion: "The reverse process, from noise to a picture: at each step the noisy image, the noise the network predicted, and its guess of the clean image.",
    compare: "Two versions of one network: how far each layer's weights and outputs moved from the first to the second.",
    dataset: "The network over a dataset: where the samples land at each layer (do the classes separate?), how alike the layers' representations are, which units never fire, and the inputs that excite each unit most.",
    equations: "The whole network as mathematics. The colours match the layers in the graph — hover a symbol to find its layer, click it to open it.",
    monitor: "How things change over time: training curves, per-layer statistics, and how gradients flow back through the layers.",
  };
  function captionBar(tab) {
    const text = CAPTIONS[tab];
    if (!text || store.get("dv-cap-" + tab)) return null;
    const bar = h("div", { class: "caption", role: "note" }, h("span", { text }),
      h("button", { class: "cap-x", text: "×", title: "hide this hint (? brings the help back)", "aria-label": "hide this hint",
        onclick: () => { store.set("dv-cap-" + tab, "1"); bar.remove(); } }));
    return bar;
  }
  function explainCard(n) {
    const ex = (P.explain || {})[n.id];
    if (!ex) return null;
    const box = h("details", { class: "p-sec explain", open: store.get("dv-explain-closed") ? null : "" },
      h("summary", null, "What it does"),
      h("p", { class: "ex-what", text: ex.what }),
      ex.why ? h("p", { class: "ex-why" }, h("b", { text: "Why " }), ex.why) : null,
      ex.look ? h("p", { class: "ex-look" }, h("b", { text: "Look at " }), ex.look) : null);
    box.addEventListener("toggle", () => store.set("dv-explain-closed", box.open ? "" : "1"));
    return box;
  }

  let helpEl = null, helpReturn = null;
  function closeHelp() {
    if (!helpEl) return;
    helpEl.remove();
    helpEl = null;
    if (helpReturn && helpReturn.isConnected && helpReturn.focus) helpReturn.focus({ preventScroll: true });
    helpReturn = null;
  }
  function openHelp() {
    const back = document.activeElement;
    closeHelp(); endTour(); hideTip();
    helpReturn = back;
    const fam = P.theme.families, labels = P.theme.familyLabels || {};
    const present = [...new Set(P.levels.flatMap((lv) => lv.nodes).map((n) => n.family))]
      .filter((f) => f && fam[f] && !["io", "container"].includes(f));
    const sw = (style, label, cls = "") => h("div", { class: "hl-row" }, h("i", { class: "hl-sw " + cls, style }), h("span", { text: label }));
    const keys = [["?", "this help"], ["/", "search layers"], [VIEWS.length > 1 ? `1 – ${VIEWS.length}` : "1", "switch views"],
      ["↑  ↓", "previous / next layer"], ["F", "play the forward pass"], ["Esc", "close · deselect"]];
    if (LIVE) keys.push(["Ctrl/⌘ Z", "undo an edit (⇧ to redo)"]);
    const dlg = h("div", { class: "help-card", role: "dialog", "aria-modal": "true", "aria-labelledby": "dv-help-title" },
      h("div", { class: "help-top" }, h("h2", { id: "dv-help-title", text: "How to read this page" }),
        h("button", { class: "btn", text: "×", "aria-label": "close help", onclick: closeHelp })),
      h("div", { class: "help-cols" },
        h("section", null, h("h4", { text: "Colours" }),
          sw({ background: P.theme.positive }, "a positive weight or value"),
          sw({ background: P.theme.negative }, "a negative one"),
          sw({ background: `linear-gradient(90deg, ${P.theme.background}, ${P.theme.positive})` }, "brighter / thicker = larger"),
          sw({}, "what's selected · what to look at", "ring"),
          h("p", { class: "hint", text: "Each kind of layer keeps one colour everywhere — in the graph, in the equations and in the videos:" }),
          h("div", { class: "hl-fams" }, ...present.map((f) => h("span", { class: "sw" }, h("i", { style: { background: fam[f] } }), labels[f] || f)))),
        h("section", null, h("h4", { text: "Moving around" }),
          h("ul", { class: "help-list" },
            h("li", { text: "Click a layer (or a neuron) to open it in the panel on the right; click the background to close it." }),
            h("li", { text: "Drag to pan, scroll to zoom; in Volumes, drag to orbit." }),
            h("li", { text: "Hover anything to read its numbers. Hover a symbol in an equation to find its layer; click it to open it." }),
            h("li", { text: "The page address remembers the view, the layer and the detail level: copy it to share exactly what you see." })),
          h("h4", { text: "Keys" }),
          h("table", { class: "help-keys" }, ...keys.map(([k, d]) => h("tr", null, h("td", null, h("kbd", { text: k })), h("td", { text: d })))))),
      h("div", { class: "help-foot" },
        h("button", { class: "btn primary", text: "▶ Take the tour", onclick: () => startTour() }),
        h("span", { class: "hint", text: "About a minute: what's on the page and how to explore it." })));
    helpEl = h("div", { class: "help-back", onclick: (ev) => { if (ev.target === helpEl) closeHelp(); } }, dlg);
    dlg.addEventListener("keydown", (ev) => {  // a modal dialog: Tab cycles through its own buttons
      if (ev.key !== "Tab") return;
      const f = [...dlg.querySelectorAll("button, [href], input, select, textarea, [tabindex]:not([tabindex='-1'])")];
      if (!f.length) return;
      const i = f.indexOf(document.activeElement);
      const next = ev.shiftKey ? (i <= 0 ? f.length - 1 : i - 1) : (i === f.length - 1 ? 0 : i + 1);
      ev.preventDefault();
      f[next].focus();
    });
    document.body.append(helpEl);
    dlg.querySelector(".btn.primary").focus();
  }

  // the tour: a handful of steps, each pointing (in highlight yellow) at a real part of this page
  let tourEl = null, tourStep = 0, tourSteps = [];
  function tourPlan() {
    const steps = [];
    if (VIEWS.some(([id]) => id === "graph")) {
      const L = P.levels[S.level] || level();
      const firstOf = (pred) => L.nodes.find((n) => n.layout && pred(n));
      const inp = firstOf((n) => n.op === "input");
      const lay = firstOf((n) => n.op === "module" && paramCount(n) > 0) || firstOf((n) => n.op === "module");
      if (inp) steps.push({ tab: "graph", sel: null, node: inp.id, at: () => GV && GV.nodes.get(inp.id),
        text: "This is your network. Data enters here, at the top, and flows down through every layer." });
      if (lay) {
        steps.push({ tab: "graph", sel: lay.id, node: lay.id, at: () => GV && GV.nodes.get(lay.id),
          text: "Each box is a layer, drawn from its own numbers. Click one to open it — here is the first layer with weights." });
        steps.push({ tab: "graph", sel: lay.id, at: () => panel.querySelector(".explain"),
          text: "The panel says what the layer does, in plain words, with its real sizes." });
        steps.push({ tab: "graph", sel: lay.id, at: () => panel.querySelector(".p-eq"),
          text: "… and writes it as maths, in the same colours as the graph. Hover a symbol to find its layer." });
      }
      steps.push({ tab: "graph", sel: null, run: () => animateForward(), at: () => stage.querySelector(".toolbar .btn.primary"),
        text: "▶ Forward pass (or the F key) shows data flowing through the network, layer by layer." });
    }
    if (VIEWS.length > 1) steps.push({ at: () => document.getElementById("dv-tabs"),
      text: "The same network, other ways: " + VIEWS.slice(1).map(([, lab]) => lab).join(", ") + ". The keys 1 – " + VIEWS.length + " switch between them." });
    steps.push({ at: () => document.getElementById("dv-help"),
      text: "Press ? any time for the keys and how to read the colours. Happy exploring!" });
    return steps;
  }
  function startTour() {
    closeHelp(); endTour();
    tourSteps = tourPlan();
    tourStep = 0;
    tourEl = { ring: h("div", { class: "tour-ring" }), card: h("div", { class: "tour-card", role: "dialog", "aria-label": "tour" }) };
    document.body.append(tourEl.ring, tourEl.card);
    showTourStep();
  }
  function endTour() { if (tourEl) { tourEl.ring.remove(); tourEl.card.remove(); tourEl = null; } }
  function tourGo(d) {
    const k = tourStep + d;
    if (k < 0) return;
    if (k >= tourSteps.length) { endTour(); return; }
    tourStep = k;
    showTourStep(d);
  }
  function showTourStep(d = 1) {
    if (!tourEl) return;
    const st = tourSteps[tourStep];
    if (st.tab && S.tab !== st.tab) { S.tab = st.tab; renderHeader(); renderStage(); }
    if ("sel" in st && S.selected !== st.sel) select(st.sel);
    if (st.run) setTimeout(() => tourEl && st.run(), 400);
    if (st.node && nodeById(st.node)) revealNode(nodeById(st.node));  // the canvas may have been panned away
    requestAnimationFrame(() => requestAnimationFrame(() => {
      if (!tourEl) return;
      const el = st.at && st.at();
      if (!el || !el.getBoundingClientRect().width) { tourGo(d >= 0 ? 1 : -1); return; }  // nothing to point at: skip it
      if (el.closest && el.closest("#panel")) el.scrollIntoView({ block: "nearest" });
      placeTour(el, st);
    }));
  }
  function placeTour(el, st) {
    const r = el.getBoundingClientRect(), pad = 6, last = tourStep === tourSteps.length - 1;
    Object.assign(tourEl.ring.style, { left: `${r.left - pad}px`, top: `${r.top - pad}px`, width: `${r.width + 2 * pad}px`, height: `${r.height + 2 * pad}px` });
    tourEl.card.replaceChildren(h("p", { text: st.text }),
      h("div", { class: "tour-nav" }, h("span", { class: "hint", text: `${tourStep + 1} / ${tourSteps.length}` }), h("div", { class: "spacer" }),
        tourStep ? h("button", { class: "btn", text: "Back", onclick: () => tourGo(-1) }) : null,
        h("button", { class: "btn primary", text: last ? "Done" : "Next", onclick: () => tourGo(1) })));
    const cw = Math.min(330, window.innerWidth - 24), ch = tourEl.card.offsetHeight || 120;
    const x = Math.min(Math.max(12, r.left), window.innerWidth - cw - 12);
    let y = r.bottom + 14;
    if (y + ch > window.innerHeight - 12) y = r.top - ch - 14;
    y = clamp(y, 12, Math.max(12, window.innerHeight - ch - 12));  // always on screen, whatever the target
    Object.assign(tourEl.card.style, { left: `${x}px`, top: `${y}px`, width: `${cw}px` });
    tourEl.card.querySelector(".btn.primary").focus({ preventScroll: true });
  }
  window.addEventListener("resize", () => { if (tourEl) showTourStep(0); });

  // ↑ / ↓: walk the layers along the flow, keeping the selected one in view
  function stepLayer(d) {
    const order = level().nodes.filter((n) => n.layout && n.op !== "container")
      .sort((a, b) => a.layout.layer - b.layout.layer || a.layout.x - b.layout.x);
    if (!order.length) return;
    const i = order.findIndex((n) => n.id === S.selected);
    const next = order[i < 0 ? (d > 0 ? 0 : order.length - 1) : clamp(i + d, 0, order.length - 1)];
    select(next.id);
    revealNode(next);
  }
  function revealNode(n) {
    if (!GV || !n.layout) return;
    const c = GV.c, r = c.svg.getBoundingClientRect(), v = c.view, m = 80;
    const sx = v.x + n.layout.x * v.k, sy = v.y + n.layout.y * v.k;
    if (sx < m || sx > r.width - m || sy < m + 60 || sy > r.height - m) {
      v.x = r.width / 2 - n.layout.x * v.k;
      v.y = r.height / 2 - n.layout.y * v.k;
      c.apply();
      S.view = { ...v, level: S.level };
    }
    const g = GV.nodes.get(n.id);
    if (g) g.focus({ preventScroll: true });
  }
  function playForward() {
    if (S.tab === "graph") animateForward();
    else if (S.tab === "neurons") propagate();
    else if (S.tab === "volumes" && VS) volForward(VS);
  }

  // the view in the page address (#tab=…&node=…&level=…): reload or share exactly what you see
  function syncHash() {
    try {
      const q = new URLSearchParams();
      if (S.tab !== VIEWS[0][0]) q.set("tab", S.tab);
      if (S.selected !== null && S.selected !== undefined) q.set("node", S.selected);
      if (S.level !== (P.defaultLevel || 0)) q.set("level", String(S.level));
      const str = q.toString();
      history.replaceState(null, "", str ? "#" + str : location.pathname + location.search);
    } catch (e) { /* a sandboxed frame */ }
  }
  function restoreHash() {
    try {
      const q = new URLSearchParams(location.hash.slice(1));
      const tab = q.get("tab");
      if (tab && VIEWS.some(([id]) => id === tab)) S.tab = tab;
      const lv = Number(q.get("level"));
      if (q.has("level") && Number.isInteger(lv) && lv >= 0 && lv < P.levels.length) S.level = lv;
      const node = q.get("node");
      if (node !== null && nodeById(node)) S.selected = node;
    } catch (e) { /* no usable address */ }
  }

  // an equation term ↔ its layer: hover lights the layer up in the graph, click opens it
  let linked = null;
  function eqTerm(target) {
    const t = target && target.closest && target.closest(".katex [data-n]");
    if (!t) return null;
    const id = (P.eqNodes || [])[Number(t.getAttribute("data-n"))];
    return id === undefined ? null : { t, id };
  }
  function setLinked(id) {
    if (linked === id) return;
    const g0 = GV && linked !== null ? GV.nodes.get(linked) : null;
    if (g0) g0.classList.remove("linked");
    linked = id;
    const g1 = GV && id !== null ? GV.nodes.get(id) : null;
    if (g1) g1.classList.add("linked");
  }
  document.addEventListener("mouseover", (ev) => {
    const e = eqTerm(ev.target);
    if (!e) return;
    setLinked(e.id);
    const n = nodeById(e.id);
    showTip(ev, [[n ? (n.collapsed ? n.type : n.kind) : "", nodeName(e.id)], n ? "click to open this layer" : ""].filter(Boolean));
  });
  document.addEventListener("mouseout", (ev) => {
    const e = eqTerm(ev.target);
    if (e && !(ev.relatedTarget && e.t.contains(ev.relatedTarget))) { setLinked(null); hideTip(); }
  });
  document.addEventListener("click", (ev) => {
    const e = eqTerm(ev.target);
    if (e && nodeById(e.id)) { ev.stopPropagation(); setLinked(null); hideTip(); select(e.id); }
  });

  // changing the detail level morphs the graph (3b1b's Transform): layers glide to their new places, a block
  // opening up grows its layers out of itself
  let MORPH = null;
  function captureLayout() {
    const pos = new Map();
    for (const n of level().nodes) if (n.layout) pos.set(n.id, n.layout);
    for (const gr of level().layout.groups || []) if (!pos.has(gr.id)) pos.set(gr.id, { x: gr.x + gr.w / 2, y: gr.y + gr.h / 2 });
    return pos;
  }
  function morphFrom(prev) {
    if (!GV || !prev || !Element.prototype.animate || matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const dur = 480, ease = "cubic-bezier(.4, 0, .2, 1)";
    for (const n of level().nodes) {
      const g = GV.nodes.get(n.id);
      if (!g || !n.layout) continue;
      let from = prev.get(n.id), born = false;
      if (!from) {  // just appeared: start where the block it was folded into was
        let id = n.id;
        while (id && !prev.has(id)) id = id.includes(".") ? id.slice(0, id.lastIndexOf(".")) : "";
        from = prev.get(id) || null;
        born = true;
      }
      const { x, y, w, h: hh } = n.layout;
      if (!from) { g.animate([{ opacity: 0 }, { opacity: 1 }], { duration: dur, easing: ease }); continue; }
      g.animate([{ transform: `translate(${from.x - w / 2}px, ${from.y - hh / 2}px)`, opacity: born ? 0 : 1 },
        { transform: `translate(${x - w / 2}px, ${y - hh / 2}px)`, opacity: 1 }], { duration: dur, easing: ease });
    }
    GV.c.vp.querySelectorAll(".edge, .tensors, .group").forEach((el) =>
      el.animate([{ opacity: 0 }, { opacity: 0, offset: 0.55 }, { opacity: 1 }], { duration: dur + 120, easing: "ease-out" }));
  }

  // ------------------------------------------------------------------- boot
  window.addEventListener("keydown", (ev) => {
    const t = ev.target;
    const typing = t && (/input|select|textarea/i.test(t.tagName) || t.isContentEditable);
    if (ev.key === "Escape") {
      if (tourEl) endTour();
      else if (helpEl) closeHelp();
      else if (typing) t.blur();
      else { S.neuron = null; select(null); }
      return;
    }
    if (typing) return;
    if (LIVE && (ev.metaKey || ev.ctrlKey) && ev.key.toLowerCase() === "z") { ev.preventDefault(); api(ev.shiftKey ? "/api/redo" : "/api/undo"); return; }
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
    if (tourEl && (ev.key === "ArrowRight" || ev.key === "ArrowLeft")) { ev.preventDefault(); tourGo(ev.key === "ArrowRight" ? 1 : -1); return; }
    if (ev.key === "?") { ev.preventDefault(); if (helpEl) closeHelp(); else openHelp(); return; }
    if (helpEl || tourEl) return;
    if (ev.key === "/") {
      const box = stage.querySelector("input.search");
      if (box) { ev.preventDefault(); box.focus(); box.select(); }
      return;
    }
    if (/^[1-9]$/.test(ev.key)) {
      const v = VIEWS[Number(ev.key) - 1];
      if (v && v[0] !== S.tab) { S.tab = v[0]; renderHeader(); renderStage(); }
      return;
    }
    if (ev.key === "f" || ev.key === "F") { playForward(); return; }
    if ((ev.key === "ArrowDown" || ev.key === "ArrowUp") && S.tab === "graph") { ev.preventDefault(); stepLayer(ev.key === "ArrowDown" ? 1 : -1); }
  });
  applyTheme();
  restoreHash();
  renderHeader();
  renderStage();
  renderPanel();
  window.deepvis = { get payload() { return P; }, select, animateForward, openHelp, startTour };
})();
