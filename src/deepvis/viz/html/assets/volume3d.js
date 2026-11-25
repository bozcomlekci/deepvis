/* deepvis 3D — a tiny orthographic scene renderer on <canvas>.
 *
 * Tensors are boxes whose faces carry real values as textures. With a parallel projection every
 * face rectangle maps to an exact parallelogram, so a texture is placed with one affine transform
 * (crisp voxels, no WebGL). Boxes are drawn back-to-front; each box shows only faces that point at
 * the camera; faces are Lambert-shaded for volume. Hovering a face reads the value of the voxel
 * (well, texel) under the pointer.
 */
(() => {
  "use strict";

  const LIGHT = norm([-0.35, 0.8, 0.55]);
  function norm(v) { const l = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0] / l, v[1] / l, v[2] / l]; }
  function add(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }

  // Box faces: normal, corner P0 (relative to center, in units of half-size), U/V edge directions, texture key.
  // Textures are rows × cols images: U runs along cols, V along rows.
  //   front (H×W) on ±x, side (H×C) on ±z, top (W×C) on ±y — see deepvis.viz.html.payload.volume_payload
  const FACES = [
    { n: [1, 0, 0], p0: [1, 1, -1], u: [0, 0, 2], v: [0, -2, 0], key: "front" },
    { n: [-1, 0, 0], p0: [-1, 1, 1], u: [0, 0, -2], v: [0, -2, 0], key: "front" },
    { n: [0, 0, 1], p0: [-1, 1, 1], u: [2, 0, 0], v: [0, -2, 0], key: "side" },
    { n: [0, 0, -1], p0: [1, 1, -1], u: [-2, 0, 0], v: [0, -2, 0], key: "side" },
    { n: [0, 1, 0], p0: [-1, 1, -1], u: [2, 0, 0], v: [0, 0, 2], key: "top" },
    { n: [0, -1, 0], p0: [-1, -1, 1], u: [2, 0, 0], v: [0, 0, -2], key: "top" },
  ];

  class Scene3D {
    constructor(canvas, opts = {}) {
      this.cv = canvas;
      this.ctx = canvas.getContext("2d");
      this.yaw = opts.yaw ?? -0.55;
      this.pitch = opts.pitch ?? 0.42;
      this.k = 1;
      this.pan = [0, 0];
      this.boxes = [];
      this.lines = [];
      this.labels = [];
      this.faces = [];
      this.center = [0, 0, 0];
      this.opts = opts;
      this.font = opts.font || "12px sans-serif";
      this.mono = opts.mono || "11px monospace";
      this.dpr = window.devicePixelRatio || 1;
      this.spinning = false;
      this._bind();
      this._ro = new ResizeObserver(() => { this.resize(); this.draw(); });
      this._ro.observe(canvas);
    }

    set({ boxes = [], lines = [], labels = [] }) {
      this.boxes = boxes;
      this.lines = lines;
      this.labels = labels;
      let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      for (const b of boxes) for (let i = 0; i < 3; i++) {
        lo[i] = Math.min(lo[i], b.c[i] - b.s[i] / 2);
        hi[i] = Math.max(hi[i], b.c[i] + b.s[i] / 2);
      }
      this.center = boxes.length ? lo.map((l, i) => (l + hi[i]) / 2) : [0, 0, 0];
      return this;
    }

    resize() {
      const r = this.cv.getBoundingClientRect();
      this.W = Math.max(r.width, 10);
      this.H = Math.max(r.height, 10);
      this.cv.width = Math.round(this.W * this.dpr);
      this.cv.height = Math.round(this.H * this.dpr);
    }

    rot(p) {
      const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw), cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
      const x1 = p[0] * cy + p[2] * sy, z1 = -p[0] * sy + p[2] * cy;
      return [x1, p[1] * cp - z1 * sp, p[1] * sp + z1 * cp];
    }

    project(p) {
      const q = this.rot([p[0] - this.center[0], p[1] - this.center[1], p[2] - this.center[2]]);
      return [this.W / 2 + this.pan[0] + q[0] * this.k, this.H / 2 + this.pan[1] - q[1] * this.k, q[2]];
    }

    fit(margin = 36, top = 0) {
      if (!this.W) this.resize();
      this.k = 1;
      this.pan = [0, 0];
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      const pts = [];
      for (const b of this.boxes) for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1])
        pts.push([b.c[0] + sx * b.s[0] / 2, b.c[1] + sy * b.s[1] / 2, b.c[2] + sz * b.s[2] / 2]);
      for (const l of this.labels) pts.push(l.p);
      for (const p of pts) {
        const q = this.project(p);
        x0 = Math.min(x0, q[0]); x1 = Math.max(x1, q[0]); y0 = Math.min(y0, q[1]); y1 = Math.max(y1, q[1]);
      }
      if (!isFinite(x0)) return;
      const k = Math.min((this.W - 2 * margin) / Math.max(x1 - x0, 1e-6), (this.H - 2 * margin - top) / Math.max(y1 - y0, 1e-6));
      this.k = Math.max(k, 1e-3);
      this.pan = [-((x0 + x1) / 2 - this.W / 2) * this.k, -((y0 + y1) / 2 - this.H / 2) * this.k + top / 2];
    }

    draw() {
      const ctx = this.ctx, dpr = this.dpr;
      if (!this.W) this.resize();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, this.W, this.H);
      this.faces = [];
      // lines (dataflow) underneath
      for (const l of this.lines) {
        const a = this.project(l.a), b = this.project(l.b);
        ctx.strokeStyle = l.color || "#444444";
        ctx.globalAlpha = l.alpha ?? 0.9;
        ctx.lineWidth = l.width || 1.5;
        ctx.beginPath();
        ctx.moveTo(a[0], a[1]);
        const mx = (a[0] + b[0]) / 2;
        ctx.bezierCurveTo(mx, a[1], mx, b[1], b[0], b[1]);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
      const order = this.boxes.map((b) => [this.rot([b.c[0] - this.center[0], b.c[1] - this.center[1], b.c[2] - this.center[2]])[2], b])
        .sort((p, q) => p[0] - q[0]).map((p) => p[1]);
      for (const b of order) this._drawBox(b);
      // labels on top, placed greedily (higher priority first) so they never overlap
      const placed = [];
      const cand = this.labels.filter((l) => !l.hidden).map((l) => ({ l, p: this.project(l.p) }))
        .sort((a, b) => (b.l.priority || 0) - (a.l.priority || 0) || a.p[0] - b.p[0]);
      ctx.textAlign = "center";
      for (const { l, p } of cand) {
        ctx.font = this.font;
        const w1 = ctx.measureText(l.text).width;
        ctx.font = this.mono;
        const w2 = l.sub ? ctx.measureText(l.sub).width : 0;
        const w = Math.max(w1, w2) + 6, top = p[1] - (l.sub ? 28 : 16), bottom = p[1] + 4;
        const r = [p[0] - w / 2, top, p[0] + w / 2, bottom];
        if (r[2] < 0 || r[0] > this.W || r[3] < 0 || r[1] > this.H) continue;
        if (placed.some((q) => r[0] < q[2] && r[2] > q[0] && r[1] < q[3] && r[3] > q[1])) continue;
        placed.push(r);
        // a dark halo keeps labels readable over bright faces
        ctx.lineJoin = "round";
        ctx.lineWidth = 3.5;
        ctx.strokeStyle = "rgba(0,0,0,0.85)";
        ctx.fillStyle = l.color || "#DDDDDD";
        ctx.font = this.font;
        ctx.strokeText(l.text, p[0], p[1] - (l.sub ? 14 : 2));
        ctx.fillText(l.text, p[0], p[1] - (l.sub ? 14 : 2));
        if (l.sub) {
          ctx.fillStyle = l.subColor || "#BBBBBB";
          ctx.font = this.mono;
          ctx.strokeText(l.sub, p[0], p[1]);
          ctx.fillText(l.sub, p[0], p[1]);
        }
      }
    }

    _drawBox(b) {
      const ctx = this.ctx, dpr = this.dpr;
      const h = [b.s[0] / 2, b.s[1] / 2, b.s[2] / 2];
      for (const f of FACES) {
        const nr = this.rot(f.n);
        if (nr[2] <= 1e-6) continue;  // back face
        const P0 = [b.c[0] + f.p0[0] * h[0], b.c[1] + f.p0[1] * h[1], b.c[2] + f.p0[2] * h[2]];
        const U = [f.u[0] * h[0], f.u[1] * h[1], f.u[2] * h[2]], V = [f.v[0] * h[0], f.v[1] * h[1], f.v[2] * h[2]];
        const p0 = this.project(P0), pu = this.project(add(P0, U)), pv = this.project(add(P0, V));
        const ux = pu[0] - p0[0], uy = pu[1] - p0[1], vx = pv[0] - p0[0], vy = pv[1] - p0[1];
        if (Math.abs(ux * vy - vx * uy) < 0.5) continue;  // edge-on
        const lambert = Math.max(0, nr[0] * LIGHT[0] + nr[1] * LIGHT[1] + nr[2] * LIGHT[2]);
        const shade = 0.5 * (1 - lambert);
        const tex = b.tex && b.tex[f.key];
        const alpha = (b.alpha ?? 1) * (b.bright ?? 1);
        ctx.save();
        if (tex && tex.img) {
          ctx.globalAlpha = alpha;
          ctx.setTransform(dpr * ux / tex.cols, dpr * uy / tex.cols, dpr * vx / tex.rows, dpr * vy / tex.rows, dpr * p0[0], dpr * p0[1]);
          ctx.imageSmoothingEnabled = false;
          ctx.drawImage(tex.img, 0, 0);
          ctx.fillStyle = `rgba(0,0,0,${shade})`;
          ctx.fillRect(0, 0, tex.cols, tex.rows);
        } else {
          ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
          ctx.globalAlpha = alpha * (b.fillAlpha ?? 0.22);
          ctx.fillStyle = b.color || "#888888";
          this._quad(p0, ux, uy, vx, vy);
          ctx.fill();
          ctx.globalAlpha = alpha * shade;
          ctx.fillStyle = "#000000";
          ctx.fill();
        }
        ctx.restore();
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.globalAlpha = Math.min(1, (b.alpha ?? 1) + 0.15);
        const hl = b.highlight || b.glow;
        ctx.strokeStyle = hl ? (this.opts.highlight || "#FFFF00") : (b.edge || b.color || "#888888");
        ctx.lineWidth = hl ? 2 : (b.edgeWidth || 1);
        this._quad(p0, ux, uy, vx, vy);
        ctx.stroke();
        ctx.globalAlpha = 1;
        this.faces.push({ b, key: f.key, tex, p0, ux, uy, vx, vy });
      }
    }

    _quad(p0, ux, uy, vx, vy) {
      const ctx = this.ctx;
      ctx.beginPath();
      ctx.moveTo(p0[0], p0[1]);
      ctx.lineTo(p0[0] + ux, p0[1] + uy);
      ctx.lineTo(p0[0] + ux + vx, p0[1] + uy + vy);
      ctx.lineTo(p0[0] + vx, p0[1] + vy);
      ctx.closePath();
    }

    pick(mx, my) {
      for (let i = this.faces.length - 1; i >= 0; i--) {
        const f = this.faces[i];
        const det = f.ux * f.vy - f.vx * f.uy;
        const dx = mx - f.p0[0], dy = my - f.p0[1];
        const u = (dx * f.vy - f.vx * dy) / det, v = (f.ux * dy - dx * f.uy) / det;
        if (u >= 0 && u <= 1 && v >= 0 && v <= 1) {
          const hit = { box: f.b, key: f.key, u, v };
          if (f.tex && f.tex.data) {
            hit.col = Math.min(f.tex.cols - 1, Math.floor(u * f.tex.cols));
            hit.row = Math.min(f.tex.rows - 1, Math.floor(v * f.tex.rows));
            hit.value = f.tex.data[hit.row * f.tex.cols + hit.col];
            hit.rows = f.tex.rows;
            hit.cols = f.tex.cols;
          }
          return hit;
        }
      }
      return null;
    }

    _bind() {
      const cv = this.cv;
      let drag = null;
      cv.addEventListener("contextmenu", (ev) => ev.preventDefault());
      cv.addEventListener("pointerdown", (ev) => {
        drag = { x: ev.clientX, y: ev.clientY, yaw: this.yaw, pitch: this.pitch, pan: [...this.pan], moved: false,
                 panMode: ev.shiftKey || ev.button === 2 || ev.button === 1, id: ev.pointerId };
      });
      cv.addEventListener("pointermove", (ev) => {
        const r = cv.getBoundingClientRect();
        if (drag) {
          const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
          if (!drag.moved && Math.abs(dx) + Math.abs(dy) > 3) {
            drag.moved = true;
            try { cv.setPointerCapture(drag.id); } catch (e) { /* released */ }
            cv.style.cursor = drag.panMode ? "grabbing" : "move";
            if (this.opts.onLeave) this.opts.onLeave();
          }
          if (drag.moved) {
            if (drag.panMode) this.pan = [drag.pan[0] + dx, drag.pan[1] + dy];
            else {
              this.yaw = drag.yaw + dx * 0.008;
              this.pitch = Math.max(-1.45, Math.min(1.45, drag.pitch + dy * 0.008));
            }
            this.draw();
            if (this.opts.onView) this.opts.onView(this);
          }
          return;
        }
        const hit = this.pick(ev.clientX - r.left, ev.clientY - r.top);
        cv.style.cursor = hit ? "pointer" : "grab";
        if (hit && this.opts.onHover) this.opts.onHover(ev, hit);
        else if (!hit && this.opts.onLeave) this.opts.onLeave();
      });
      cv.addEventListener("pointerup", (ev) => {
        const r = cv.getBoundingClientRect();
        if (drag && !drag.moved && this.opts.onPick) this.opts.onPick(this.pick(ev.clientX - r.left, ev.clientY - r.top));
        drag = null;
        cv.style.cursor = "grab";
      });
      cv.addEventListener("pointerleave", () => { if (this.opts.onLeave) this.opts.onLeave(); });
      cv.addEventListener("wheel", (ev) => {
        ev.preventDefault();
        const r = cv.getBoundingClientRect();
        const mx = ev.clientX - r.left, my = ev.clientY - r.top;
        const f = Math.exp(-ev.deltaY * (ev.ctrlKey ? 0.01 : 0.0015));
        const qx = (mx - this.W / 2 - this.pan[0]) / this.k, qy = (my - this.H / 2 - this.pan[1]) / this.k;
        this.k = Math.max(1e-3, this.k * f);
        this.pan = [mx - this.W / 2 - qx * this.k, my - this.H / 2 - qy * this.k];
        this.draw();
        if (this.opts.onView) this.opts.onView(this);
      }, { passive: false });
    }

    spin(on) {
      this.spinning = on;
      const step = () => {
        if (!this.spinning || !this.cv.isConnected) return;
        this.yaw += 0.006;
        this.draw();
        requestAnimationFrame(step);
      };
      if (on) requestAnimationFrame(step);
    }

    animate(duration, frame) {
      const t0 = performance.now();
      const tick = (t) => {
        if (!this.cv.isConnected) return;
        const p = Math.min(1, (t - t0) / duration);
        frame(p);
        this.draw();
        if (p < 1) requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    }

    dispose() {
      this.spinning = false;
      this._ro.disconnect();
    }
  }

  /** Make a texture (offscreen canvas + raw values) from a {rows, cols, data(base64 f32)} face. */
  function texture(face, vmin, vmax, diverging, colorOf, decode) {
    if (!face) return null;
    const data = decode(face.data);
    const img = document.createElement("canvas");
    img.width = face.cols;
    img.height = face.rows;
    const ictx = img.getContext("2d");
    const id = ictx.createImageData(face.cols, face.rows);
    for (let i = 0; i < data.length; i++) {
      const c = colorOf(data[i], vmin, vmax, diverging);
      id.data[i * 4] = c[0]; id.data[i * 4 + 1] = c[1]; id.data[i * 4 + 2] = c[2]; id.data[i * 4 + 3] = 255;
    }
    ictx.putImageData(id, 0, 0);
    return { img, data, rows: face.rows, cols: face.cols };
  }

  /** Texture from a slice of a flat Float32Array (for stacks of channel maps / filters). */
  function sliceTexture(data, offset, rows, cols, vmin, vmax, diverging, colorOf) {
    const img = document.createElement("canvas");
    img.width = cols;
    img.height = rows;
    const ictx = img.getContext("2d");
    const id = ictx.createImageData(cols, rows);
    const sub = data.subarray(offset, offset + rows * cols);
    for (let i = 0; i < sub.length; i++) {
      const c = colorOf(sub[i], vmin, vmax, diverging);
      id.data[i * 4] = c[0]; id.data[i * 4 + 1] = c[1]; id.data[i * 4 + 2] = c[2]; id.data[i * 4 + 3] = 255;
    }
    ictx.putImageData(id, 0, 0);
    return { img, data: sub, rows, cols };
  }

  window.DV3D = { Scene3D, texture, sliceTexture };
})();
