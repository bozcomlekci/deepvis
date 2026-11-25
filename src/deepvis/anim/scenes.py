"""Manim scenes. Each takes a plain data dict (see :mod:`deepvis.anim.data`)."""

from __future__ import annotations

import numpy as np
from manim import (
    DOWN, LEFT, RIGHT, UL, UP, AnimationGroup, Arrow, Create, FadeIn, FadeOut,
    GrowArrow, GrowFromCenter, ImageMobject, Indicate, LaggedStart, MovingCameraScene, ReplacementTransform,
    Restore, Scene, ShowPassingFlash, Square, SurroundingRectangle, Transform, TransformMatchingShapes, VGroup, Write,
    config,
    DEGREES, OUT, PI, Dot, Line, Prism, Rectangle, ThreeDScene, ValueTracker, VMobject, always_redraw, linear,
)

from .. import theme
from ..equations.expr import painted
from .mobjects import Block, NetworkDiagram, WeightGrid, bar_chart, caption_text, math, text, weight_style

YELLOW = theme.YELLOW_C


def _title(name: str, subtitle: str) -> VGroup:
    t = text(name, 40)
    s = text(subtitle, 24, theme.GREY_B)
    g = VGroup(t, s).arrange(DOWN, buff=0.12)
    return g


def _corner_title(name: str, subtitle: str) -> VGroup:
    g = VGroup(text(name, 26), text(subtitle, 18, theme.GREY_B)).arrange(RIGHT, buff=0.25, aligned_edge=DOWN)
    return g.to_corner(UL, buff=0.35)


def _image(arr: np.ndarray, height: float = 1.8):
    a = np.asarray(arr, dtype=float)
    lo, hi = a.min(), a.max()
    u8 = (np.clip((a - lo) / (hi - lo if hi > lo else 1), 0, 1) * 235 + 20).astype(np.uint8)
    img = ImageMobject(u8)
    try:
        from manim import RESAMPLING_ALGORITHMS

        img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    except Exception:  # noqa: BLE001
        pass
    img.height = height
    return img


class _Captions:
    """3b1b's caption zone: one plain sentence at a time along the bottom of the frame, cross-faded. ``frame``: a
    moving camera's frame (the caption follows it when shown); ``fixed``: a 3-D scene's fixed-in-frame layer."""

    def __init__(self, scene, on: bool = True, frame=None, fixed: bool = False):
        self.scene, self.on, self.frame, self.fixed, self.cur = scene, on, frame, fixed, None

    def _make(self, s: str):
        t = caption_text(s)
        if self.frame is not None:
            k = self.frame.width / config.frame_width
            t.scale(k).move_to(self.frame.get_bottom() + UP * (0.45 * k + t.height / 2))
        else:
            t.to_edge(DOWN, buff=0.32)
        if self.fixed:
            self.scene.add_fixed_in_frame_mobjects(t)
        return t

    def anims(self, s: str | None) -> list:
        """Animations swapping in caption ``s`` (play them with others); [] when captions are off."""
        if not self.on or not s:
            return []
        new = self._make(s)
        out = [FadeIn(new, shift=UP * 0.12)]
        if self.cur is not None:
            out.append(FadeOut(self.cur, shift=UP * 0.12))
        self.cur = new
        return out

    def say(self, s: str | None, run_time: float = 0.6, hold: float = 0.0):
        a = self.anims(s)
        if a:
            self.scene.play(*a, run_time=run_time)
            if hold:
                self.scene.wait(hold)

    def clear(self, run_time: float = 0.4):
        if self.cur is not None:
            self.scene.play(FadeOut(self.cur), run_time=run_time)
            self.cur = None


def _pixel_squares(img, arr, shown) -> VGroup:
    """Small squares over the image's pixels that are drawn as input neurons (flat index → row, column), each in
    its pixel's grey and outlined in highlight yellow — ready to fly to their neurons."""
    a = np.asarray(arr, dtype=float)
    side = a.shape[0]
    px = img.height / side
    lo, hi = float(a.min()), float(a.max())
    ul = img.get_corner(UL)
    out = VGroup()
    for idx in shown:
        r, c = divmod(int(idx), side)
        if r >= side:
            return VGroup()  # not a flattened image: nothing to carry
        g = int(20 + 235 * (a[r, c] - lo) / ((hi - lo) or 1))  # the same grey _image draws
        out.add(Square(side_length=px * 1.25, stroke_color=YELLOW, stroke_width=1.6, fill_color=f"#{g:02x}{g:02x}{g:02x}",
                       fill_opacity=1).move_to(ul + RIGHT * (c + 0.5) * px + DOWN * (r + 0.5) * px))
    return out


class _Base(Scene):
    def __init__(self, data: dict, **kw):
        self.data = data
        super().__init__(**kw)


class ForwardPassScene(_Base):
    """Activations light up layer by layer; synapses carrying the most signal flash."""

    def construct(self):
        d = self.data
        head = _title(d["name"], "a forward pass")
        self.play(FadeIn(head, shift=UP * 0.2), run_time=0.8)
        self.play(head.animate.scale(0.6).to_corner(UL, buff=0.35), run_time=0.6)
        cap = _Captions(self, d.get("captions", True))
        net = NetworkDiagram(d, width=7.6 if d["image"] is not None else 9.0, height=5.0 if cap.on else 5.2)
        net.shift(RIGHT * (0.9 if d["image"] is not None else 0) + (UP * 0.05 if cap.on else DOWN * 0.2))
        self.play(LaggedStart(*[Create(g) for g in net.links], lag_ratio=0.25), run_time=1.6)
        self.play(LaggedStart(*[GrowFromCenter(c) for col in net.columns for c in col], lag_ratio=0.01),
                  FadeIn(net.dots), run_time=1.0)
        labels = VGroup()
        for li, col in enumerate(net.columns):
            name = "input" if li == 0 else ("output" if li == len(net.columns) - 1 else f"layer {li}")
            size = d["layers"][li]["size"]
            lab = VGroup(text(name, 20), text(f"{size:,}", 14, theme.GREY_C)).arrange(DOWN, buff=0.05)
            labels.add(lab.next_to(col, UP, buff=0.25))
        self.play(FadeIn(labels), *cap.anims(d.get("describe")), run_time=0.6)
        if cap.on and d.get("describe"):
            self.wait(2.0)
        layers = d["layers"]
        if d["image"] is not None:
            img = _image(d["image"]).next_to(net, LEFT, buff=0.7)
            arrow = Arrow(img.get_right(), net.columns[0].get_left(), buff=0.15, color=theme.GREY_C, stroke_width=3)
            side = d["image"].shape[0]
            self.play(FadeIn(img), GrowArrow(arrow), *cap.anims(
                f"The input: a {side} × {side} image — {side * side:,} numbers, one per input neuron."), run_time=0.8)
            pixels = _pixel_squares(img, d["image"], layers[0]["shown"])
            if len(pixels) == len(net.columns[0]):  # each drawn input neuron is one pixel: carry it over
                self.play(LaggedStart(*[FadeIn(q, scale=1.6) for q in pixels], lag_ratio=0.04), *cap.anims(
                    "Each pixel's brightness becomes the value of one input neuron (these are the ones drawn)."),
                    run_time=0.9)
                self.play(LaggedStart(*[q.animate.move_to(c).scale(2.4).set_opacity(0)
                                        for q, c in zip(pixels, net.columns[0])], lag_ratio=0.05),
                          *net.fill_anims(0), run_time=1.5)
                self.remove(pixels)
            else:
                self.play(*net.fill_anims(0), run_time=0.8)
        else:
            cap.say(f"The input: {layers[0]['size']:,} numbers, one per input neuron.")
            self.play(*net.fill_anims(0), run_time=0.8)
        # opacity layering: the layer being computed at full strength, the other synapses as faint context
        for g in net.links:
            g.save_state()
        self.play(*[g.animate.fade(0.75) for g in net.links[1:]], run_time=0.4)
        for i in range(len(net.links)):
            L, prev = layers[i + 1], layers[i]
            who = "The output layer" if i + 1 == len(layers) - 1 else f"Layer {i + 1}"
            say = (f"{who}: {L['size']:,} neurons. Each adds up all {prev['size']:,} inputs, weighted (blue +, red −)"
                   + (f", then applies {L['act']}." if L.get("act") else "."))
            focus = [Restore(net.links[i])] + ([net.links[i - 1].animate.fade(0.75)] if i else [])
            if i:
                self.play(*focus, *cap.anims(say), run_time=0.5)
            else:
                cap.say(say, run_time=0.5)
            top = net.contributions(i, top=36)
            flashes = [ShowPassingFlash(ln.copy().set_stroke(YELLOW, width=3.5, opacity=1), time_width=0.6)
                       for ln in top]
            self.play(AnimationGroup(*flashes, lag_ratio=0.01), run_time=0.9)
            self.play(*net.fill_anims(i + 1), run_time=0.6)
        self.play(*[Restore(g) for g in net.links[:-1]], run_time=0.4)
        out_col = net.columns[-1]
        last = d["layers"][-1]
        out = d["output"]
        vals = VGroup()
        for k, c in enumerate(out_col):
            idx = last["shown"][k]
            v = out[idx] if idx < len(out) else 0.0
            num = text((f"{v * 100:.1f}%" if d["probabilities"] else f"{v:.2f}"), 16, theme.GREY_A)
            vals.add(VGroup(text(str(idx), 16, theme.GREY_C), num).arrange(RIGHT, buff=0.15).next_to(c, RIGHT, 0.2))
        self.play(FadeIn(vals, shift=LEFT * 0.1), run_time=0.6)
        shown_vals = [out[i] if i < len(out) else -np.inf for i in last["shown"]]
        best = int(np.argmax(shown_vals))
        box = SurroundingRectangle(VGroup(out_col[best], vals[best]), color=YELLOW, buff=0.08, corner_radius=0.06)
        idx = last["shown"][best]
        share = f" ({out[idx] * 100:.1f}%)" if d["probabilities"] and idx < len(out) else ""
        self.play(Create(box), Indicate(out_col[best], color=YELLOW), *cap.anims(
            f"The answer: {idx}{share} — the largest of the {len(out):,} outputs."), run_time=0.8)
        self.wait(1.2)


class EquationScene(_Base):
    """From composition, to layer-by-layer equations, to one fully substituted formula — every symbol in the
    colour of the layer it comes from, like the blocks beside it."""

    def construct(self):
        d = self.data
        eqs = d["equations"]
        cap = _Captions(self, d.get("captions", True))
        colors = d.get("colors") or {}
        with painted(color_of=colors.get):  # the LaTeX is coloured as it is written out

            def tex(ln):
                return f"{ln.lhs.latex()} = {ln.rhs.latex()}", f"{ln.lhs.text()} = {ln.rhs.text()}"

            compact = eqs.get("compact")
            system = eqs.get("system")
            expanded = eqs.get("expanded")
            t_compact = tex(compact.lines[0]) if compact is not None and compact.lines else None
            t_system = [(tex(ln), ln.node) for ln in system.lines[:8]] if system is not None else []
            t_full = tex(expanded.lines[0]) if expanded is not None and len(expanded.lines) == 1 else None
        head = _title(d["name"], "unrolled into mathematics")
        self.play(FadeIn(head, shift=UP * 0.2), run_time=0.7)
        self.play(head.animate.scale(0.6).to_corner(UL, buff=0.35), run_time=0.5)
        blocks = VGroup(*[Block(l_["kind"], l_["label"], theme.kind_color(l_["kind"]), width=1.9, height=0.5)
                          for l_ in d["layers"][:9]]).arrange(DOWN, buff=0.32)
        if blocks.height > config.frame_height - 1.9:
            blocks.scale_to_fit_height(config.frame_height - 1.9)
        blocks.to_edge(LEFT, buff=0.5).shift(DOWN * 0.15)
        self.play(LaggedStart(*[FadeIn(b, shift=RIGHT * 0.2) for b in blocks], lag_ratio=0.12), *cap.anims(
            "Each layer of the network, in its own colour."), run_time=1.0)
        right_w = config.frame_width - blocks.width - 1.8
        cur = None
        if t_compact is not None:
            cur = math(*t_compact, 38, max_width=right_w)
            cur.next_to(blocks, RIGHT, buff=0.8).set_y(1.2)
            self.play(Write(cur), *cap.anims("The whole network as one composition of functions."), run_time=1.4)
            self.wait(0.6)
            self.play(cur.animate.set_opacity(0.35).scale(0.8).to_edge(UP, buff=0.4).shift(RIGHT), run_time=0.7)
        lines = VGroup()
        if t_system:
            node_to_block = {l_["id"]: i for i, l_ in enumerate(d["layers"][:9])}
            for (lt, pt), _ in t_system:
                lines.add(math(lt, pt, 32, max_width=right_w))
            lines.arrange(DOWN, aligned_edge=LEFT, buff=0.32)
            if lines.height > config.frame_height - 2.7:
                lines.scale_to_fit_height(config.frame_height - 2.7)
            lines.next_to(blocks, RIGHT, buff=0.8).shift(DOWN * 0.1)
            cap.say("Layer by layer: one line per layer. Each symbol has the colour of the layer it belongs to.",
                    run_time=0.5)
            for m, (_, node) in zip(lines, t_system):
                anims = [Write(m)]
                bi = node_to_block.get(node)
                if bi is None:  # the line is named after an activation: highlight the nearest layer before it
                    ids = [l_["id"] for l_ in d["layers"]]
                    for i, lid in enumerate(ids[:9]):
                        if node and node.startswith(lid.rsplit(".", 1)[0]):
                            bi = i
                if bi is not None:
                    anims.append(Indicate(blocks[bi], color=YELLOW, scale_factor=1.06))
                self.play(*anims, run_time=0.9)
            self.wait(0.8)
        if t_full is not None and len(lines):
            full = math(*t_full, 34, max_width=config.frame_width - 1.0).move_to(UP * 0.1)
            fade = [FadeOut(blocks)] + ([FadeOut(cur)] if cur is not None else [])
            try:  # 3b1b's TransformMatchingShapes: every symbol flies to its place in the substituted formula
                self.play(TransformMatchingShapes(lines, full), *fade, *cap.anims(
                    "Substitute every line into the next: the whole network in one formula."), run_time=2.0)
            except Exception:  # noqa: BLE001 - shapes that can't be matched: a plain morph
                self.play(ReplacementTransform(lines, full), *fade, run_time=1.8)
            box = SurroundingRectangle(full, color=YELLOW, buff=0.18, corner_radius=0.08)
            self.play(Create(box), run_time=0.7)
        self.wait(1.2)


class FocusScene(MovingCameraScene):
    """Zoom into one layer, turn its synapses into the weight matrix, then read off one neuron."""

    def __init__(self, data: dict, **kw):
        self.data = data
        super().__init__(**kw)

    def construct(self):
        d = self.data
        head = _corner_title(d["name"], f"focus on {d['focus_node']}")
        cap = _Captions(self, d.get("captions", True), frame=self.camera.frame)
        net = NetworkDiagram(d, width=9.0, height=5.0 if cap.on else 5.2).shift(UP * 0.05 if cap.on else DOWN * 0.2)
        self.add(head)
        li = d["focus_layer"]
        self.play(FadeIn(net), *cap.anims(f"Zoom into layer {li}: the lines between these two columns are its "
                                          "weights — blue positive, red negative."), run_time=0.8)
        self.wait(0.6)
        cap.clear()
        links = net.links[li - 1]
        cols = VGroup(net.columns[li - 1], net.columns[li])
        self.camera.frame.save_state()
        self.play(self.camera.frame.animate.move_to(cols).set(width=max(cols.width * 2.6, 6.0)),
                  *[c.animate.set_opacity(0.15) for i, c in enumerate(net.columns) if i not in (li - 1, li)],
                  *[g.animate.set_stroke(opacity=0.03) for i, g in enumerate(net.links) if i != li - 1], run_time=1.4)
        self.wait(0.3)
        grid = WeightGrid(d["matrix"], cell=min(0.12, 3.0 / max(d["matrix"].shape))).move_to(cols)
        rows_lab = text("output neuron", 14, theme.GREY_C).rotate(np.pi / 2).next_to(grid, LEFT, buff=0.1)
        cols_lab = text("input neuron", 14, theme.GREY_C).next_to(grid, UP, buff=0.1)
        self.play(ReplacementTransform(links, grid), FadeOut(cols), run_time=1.6)
        self.play(FadeIn(rows_lab), FadeIn(cols_lab), *cap.anims(
            "The same weights as a matrix: one row per output neuron, one column per input."), run_time=0.6)
        self.wait(0.8)
        j = d["neuron"]
        if j < grid.shape_[0]:
            row = grid.row(j)
            hl = SurroundingRectangle(row, color=YELLOW, buff=0.02, stroke_width=3)
            self.play(Create(hl), *cap.anims(f"Row {j} holds the weights of neuron {j}."), run_time=0.6)
            self.wait(0.5)
        cap.clear()
        eq = d["equation"]
        with painted(color_of=(d.get("colors") or {}).get, signs=(theme.POSITIVE, theme.NEGATIVE)):
            texts = [(f"{ln.lhs.latex()} = {ln.rhs.latex()}", f"{ln.lhs.text()} = {ln.rhs.text()}") for ln in eq.lines]
        lines = VGroup(*[math(lt, pt, 30, max_width=self.camera.frame.width * 0.9) for lt, pt in texts])
        lines.arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        lines.scale_to_fit_width(min(lines.width, self.camera.frame.width * 0.9))
        lines.next_to(grid, DOWN, buff=0.35)
        self.play(self.camera.frame.animate.move_to(VGroup(grid, lines)).set(
            height=max(VGroup(grid, lines).height * 1.35, self.camera.frame.height)), run_time=0.8)
        self.play(Write(lines), *cap.anims(f"Neuron {j} adds up its inputs, each times its weight — the same blue "
                                           "and red as the row — plus a bias. The largest terms, with real numbers."),
                  run_time=1.6)
        self.wait(1.6)
        cap.clear()
        self.play(Restore(self.camera.frame), FadeOut(lines), run_time=1.2)
        self.wait(0.3)


class EditScene(_Base):
    """Before/after a weight edit: synapses morph and the output responds."""

    def construct(self):
        d = self.data
        before, after = d["before"], d["after"]
        head = _corner_title(d["name"], "editing weights")
        self.add(head)
        net = NetworkDiagram(before, width=7.4, height=5.0).shift(LEFT * 1.4 + DOWN * 0.2)
        self.play(FadeIn(net), run_time=0.8)
        self.play(*[a for li in range(len(net.columns)) for a in net.fill_anims(li)], run_time=0.6)
        shown = before["layers"][-1]["shown"]
        vb = np.asarray(before["output"])[shown]
        va = np.asarray(after["output"])[shown]
        probs = before["probabilities"] and after["probabilities"]
        scale = 1.0 if probs else max(np.abs(np.concatenate([vb, va])).max(), 1e-9)
        bars = bar_chart(vb / scale, width=2.2, bar_h=0.22, gap=0.14, probs=True).next_to(net, RIGHT, buff=1.2)
        bars.align_to(net.columns[-1], UP)
        cap_out = text("output", 18, theme.GREY_B).next_to(bars, UP, buff=0.25)
        self.play(FadeIn(bars), FadeIn(cap_out), run_time=0.6)
        label = text(d["label"], 22, theme.YELLOW_C).to_edge(DOWN, buff=0.4)
        self.play(Write(label), run_time=0.8)
        anims = []
        for i, (Wb, Wa) in enumerate(zip(before["weights"], after["weights"])):
            wmax = float(np.abs(Wb).max()) or 1.0
            for ln in net.links[i]:
                r, c = ln.rc
                st = weight_style(Wa[r, c], wmax)
                if abs(Wa[r, c] - Wb[r, c]) > 1e-12:
                    anims.append(ln.animate.set_stroke(st["color"], width=st["stroke_width"],
                                                       opacity=st["stroke_opacity"]))
        new_bars = bar_chart(va / scale, width=2.2, bar_h=0.22, gap=0.14, probs=True).move_to(bars, aligned_edge=LEFT)
        if anims:
            self.play(LaggedStart(*anims, lag_ratio=0.002), run_time=1.6)
        fills = []
        for li, l_ in enumerate(after["layers"]):
            fills += net.fill_anims(li, l_["values"], before["layers"][li].get("range"))
        self.play(*fills, Transform(bars, new_bars), run_time=1.2)
        self.wait(1.5)


class ArchitectureScene(_Base):
    """Block diagram of any architecture with data flowing through it (CNNs, transformers, ...)."""

    def construct(self):
        d = self.data
        head = _corner_title(d["name"], "architecture")
        self.add(head)
        cap = _Captions(self, d.get("captions", True))
        nodes = {n["id"]: n for n in d["nodes"]}
        n_layers = max(n["layer"] for n in d["nodes"]) + 1
        by_layer: dict[int, list] = {}
        for n in d["nodes"]:
            by_layer.setdefault(n["layer"], []).append(n)
        for v in by_layer.values():
            v.sort(key=lambda n: n["x"])
        W, H = config.frame_width - 1.2, config.frame_height - (2.9 if cap.on else 2.0)  # room for the captions
        dx = W / max(n_layers, 1)
        bw = min(1.9, dx * 0.82)
        mobs = {}
        for li, ns in by_layer.items():
            k = len(ns)
            dy = min(1.1, H / max(k, 1))
            for i, n in enumerate(ns):
                pill = n["op"] in ("function", "input", "output", "param")
                label = n["label"] if n["op"] == "module" else ""
                kind = n["kind"] if n["op"] not in ("input", "output") else ("input" if n["op"] == "input" else "output")
                b = Block(kind, label, theme.kind_color(n["kind"]), width=bw * (0.8 if pill else 1),
                          height=0.42 if pill else 0.56, pill=pill)
                b.move_to([-W / 2 + dx * (li + 0.5), (i - (k - 1) / 2) * -dy + (0.15 if cap.on else -0.3), 0])
                mobs[n["id"]] = b
        arrows = {}
        for e in d["edges"]:
            a, b = mobs[e["src"]].body, mobs[e["dst"]].body
            arrows[(e["src"], e["dst"])] = Arrow(a.get_right(), b.get_left(), buff=0.05, stroke_width=2.5,
                                                 color=theme.GREY_C, max_tip_length_to_length_ratio=0.12,
                                                 tip_length=0.14)
        order = [n for n in d["order"] if n in mobs]
        self.play(LaggedStart(*[FadeIn(mobs[n], shift=RIGHT * 0.15) for n in order], lag_ratio=0.08), run_time=2.0)
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows.values()], lag_ratio=0.03), *cap.anims(d.get("describe")),
                  run_time=1.0)
        if cap.on and d.get("describe"):
            self.wait(2.2)
        told, narrate = set(), set()  # the first layer of each kind says what it does (a few, not all)
        for n in order:
            nd = nodes[n]
            if nd.get("caption") and nd["op"] in ("module", "function") and nd["kind"] not in told and len(narrate) < 7:
                told.add(nd["kind"])
                narrate.add(n)
        for n in order:
            outs = [arr for (s, _t), arr in arrows.items() if s == n]
            anims = [Indicate(mobs[n].body, color=YELLOW, scale_factor=1.08)]
            anims += [ShowPassingFlash(a.copy().set_color(YELLOW).set_stroke(width=4), time_width=0.7) for a in outs]
            if n in narrate and cap.on:
                self.play(*anims, *cap.anims(f"{nodes[n]['kind']}: {nodes[n]['caption']}"), run_time=0.6)
                self.wait(1.6)
            else:
                self.play(*anims, run_time=0.35)
        cap.clear()
        shapes = VGroup()
        for n in order:
            sh = nodes[n]["shape"]
            if sh and nodes[n]["op"] == "module":
                shapes.add(text("×".join(str(s) for s in sh), 11, theme.GREY_C, serif=False)
                           .next_to(mobs[n].body, UP, buff=0.06))
        if len(shapes):
            self.play(FadeIn(shapes), run_time=0.6)
        self.wait(1.2)


class VolumesScene(ThreeDScene):
    """Tensors as 3-D blocks: channels along the flow, height up, width into depth; the front face carries the
    (channel-mean) activation. A forward pass lights them up in turn while the camera orbits."""

    def __init__(self, data: dict, **kw):
        self.data = data
        super().__init__(**kw)

    def construct(self):
        d = self.data
        blocks = d["blocks"]
        size = lambda n: 0.18 + 0.36 * np.log2(max(n, 1) + 1)  # noqa: E731
        thin = 0.14
        head = _corner_title(d["name"], "tensors in 3D")
        self.add_fixed_in_frame_mobjects(head)
        cap = _Captions(self, d.get("captions", True), fixed=True)
        self.set_camera_orientation(phi=66 * DEGREES, theta=-124 * DEGREES, zoom=0.9)  # input-side faces
        geo, x = [], 0.0
        for b in blocks:
            C, H, W = b["dims"]
            if b["kind"] == "vec":
                dx, dy, dz = thin, thin, size(H)
            elif b["kind"] == "seq":
                dx, dy, dz = thin, size(W), size(H)
            else:
                dx, dy, dz = max(thin, size(C) * 0.8), size(W), size(H)
            geo.append((x + dx / 2, dx, dy, dz))
            x += dx + 0.55
        total = x - 0.55
        scale = min(1.0, 11.5 / max(total, 1e-6))
        prisms, faces, labels = VGroup(), VGroup(), VGroup()
        for b, (cx, dx, dy, dz) in zip(blocks, geo):
            cx, dx, dy, dz = (cx - total / 2) * scale, dx * scale, dy * scale, dz * scale
            pr = Prism(dimensions=[dx, dy, dz], fill_color=b["color"], fill_opacity=0.18, stroke_color=b["color"],
                       stroke_width=1.2).move_to([cx, 0, 0])
            prisms.add(pr)
            rows, cols = b["face"]["rows"], b["face"]["cols"]
            grid = VGroup()
            cw, ch = dy / cols, dz / rows
            for r in range(rows):
                for c in range(cols):
                    sq = Rectangle(width=cw, height=ch, stroke_width=0, fill_color=b["face"]["colors"][r * cols + c],
                                   fill_opacity=0.95)
                    sq.rotate(PI / 2, axis=RIGHT)                       # into the y–z plane…
                    sq.rotate(PI / 2, axis=OUT)                         # …facing −x (towards the input)
                    sq.move_to([cx - dx / 2 - 0.004, -dy / 2 + (c + 0.5) * cw, dz / 2 - (r + 0.5) * ch])
                    grid.add(sq)
            faces.add(grid)
            lab = text(" · ".join(b["kinds"]), 16, theme.GREY_A, serif=False)
            dims = b["dims"]
            sub = "×".join(str(v) for v in (dims[1:2] if b["kind"] == "vec" else dims[1:] if b["kind"] == "seq" else dims))
            above = len(labels) % 2 == 0  # alternate above / below so neighbours never collide
            labels.add(VGroup(lab, text(sub, 12, theme.GREY_C, serif=False)).arrange(DOWN, buff=0.04)
                       .move_to([cx, 0, (dz / 2 + 0.45) if above else -(dz / 2 + 0.45)]))
        for lab in labels:
            self.add_fixed_orientation_mobjects(lab)
        links = VGroup(*[Line(prisms[i].get_right(), prisms[i + 1].get_left(), stroke_color=theme.GREY_D, stroke_width=2)
                         for i in range(len(prisms) - 1)])
        for grid in faces:
            grid.set_fill(opacity=0.12)
        self.play(LaggedStart(*[Create(p) for p in prisms], lag_ratio=0.15), FadeIn(links), *cap.anims(d.get("describe")),
                  run_time=2.0)
        if cap.on and d.get("describe"):
            self.wait(1.6)
        self.play(*[FadeIn(lab) for lab in labels], *[FadeIn(g) for g in faces], *cap.anims(
            "Each block is one tensor on its way through the network: its face is height × width, its depth the "
            "number of channels."), run_time=0.8)
        self.wait(1.2)
        cap.say("A forward pass: each tensor lights up as the data reaches it.", run_time=0.5)
        self.begin_ambient_camera_rotation(rate=0.05)
        for pr, grid in zip(prisms, faces):  # the forward pass: each tensor lights up in turn
            self.play(grid.animate.set_fill(opacity=0.95), pr.animate.set_stroke(YELLOW, width=2.5), run_time=0.45)
            self.play(pr.animate.set_stroke(pr.get_fill_color(), width=1.2), run_time=0.2)
        self.wait(1.5)
        self.stop_ambient_camera_rotation()
        self.move_camera(phi=74 * DEGREES, theta=-112 * DEGREES, run_time=2.0)
        self.wait(0.6)


def _fmt(v: float) -> str:
    from ..equations.expr import sup

    a = abs(v)
    if a and (a < 1e-3 or a >= 1e4):
        m, e = f"{v:.2e}".split("e")
        return f"{m}×10{sup(str(int(e)))}"
    return f"{v:.3g}"


class TrainingScene(_Base):
    """Watch it learn: the loss curve draws itself while a layers × time grid fills in with how fast each
    layer's weights move (relative update per step, BLUE ramp)."""

    def construct(self):
        d = self.data
        self.add(_corner_title(d["name"], "learning"))
        t = ValueTracker(0.0)
        s0 = float(d["snap_steps"][0]) if len(d["snap_steps"]) else 0.0
        steps_all = d["curve"]["steps"] if d["curve"] is not None else d["snap_steps"]
        s_lo, s_hi = float(min(steps_all[0], s0)), float(steps_all[-1])
        step_at = lambda: s_lo + t.get_value() * (s_hi - s_lo)  # noqa: E731
        parts = []
        # ---- left: the metric curve
        W, H = 5.4, 3.4
        origin = np.array([-6.4, -2.2, 0])
        frame = VGroup(Line(origin, origin + [W, 0, 0], color=theme.GREY_D, stroke_width=1.5),
                       Line(origin, origin + [0, H, 0], color=theme.GREY_D, stroke_width=1.5))
        parts.append(frame)
        c = d["curve"]
        if c is not None and len(c["values"]) > 1:
            ys = np.log10(c["values"]) if c["log"] else c["values"]
            y_lo, y_hi = float(ys.min()), float(ys.max())
            y_hi = y_hi if y_hi > y_lo else y_lo + 1
            pts = [origin + [W * (x - s_lo) / max(s_hi - s_lo, 1e-9), H * (y - y_lo) / (y_hi - y_lo), 0]
                   for x, y in zip(c["steps"], ys)]

            def partial():
                k = max(2, int(round(t.get_value() * (len(pts) - 1))) + 1)
                m = VMobject(stroke_color=theme.BLUE_C, stroke_width=3)
                m.set_points_as_corners(pts[:k])
                return m

            curve = always_redraw(partial)
            dot = always_redraw(lambda: Dot(pts[max(0, int(round(t.get_value() * (len(pts) - 1))))],
                                            radius=0.06, color=theme.YELLOW_C))
            name = c["name"] + (" (log scale)" if c["log"] else "")
            lab = text(name, 20, theme.GREY_B).next_to(frame, UP, buff=0.15).align_to(frame, LEFT)
            top = text(_fmt(10 ** y_hi if c["log"] else y_hi), 14, theme.GREY_C, serif=False)
            bot = text(_fmt(10 ** y_lo if c["log"] else y_lo), 14, theme.GREY_C, serif=False)
            top.next_to(origin + [0, H, 0], LEFT, buff=0.1)
            bot.next_to(origin, LEFT, buff=0.1)
            parts += [lab, top, bot]
            readout = always_redraw(lambda: text(
                f"step {int(step_at())}", 18, theme.GREY_A, serif=False).next_to(frame, DOWN, buff=0.2)
                .align_to(frame, LEFT))
        else:
            curve = dot = readout = None
        # ---- right: layers × snapshots
        grid = d["grid"]
        cells = VGroup()
        if grid.size:
            L, S = grid.shape
            gw, gh = 5.2, min(3.4, max(1.5, 0.3 * L))
            cw, ch = gw / S, gh / L
            g0 = np.array([0.9, -2.2 + (3.4 - gh), 0])
            for i in range(L):
                for j in range(S):
                    r = Rectangle(width=cw * 0.92, height=ch * 0.86, stroke_width=0,
                                  fill_color=d["colors"][i, j], fill_opacity=0)
                    r.move_to(g0 + [cw * (j + 0.5), gh - ch * (i + 0.5), 0])
                    r.snap_step = float(d["snap_steps"][j])
                    cells.add(r)
            cells.add_updater(lambda m: [c_.set_fill(opacity=1 if c_.snap_step <= step_at() else 0) for c_ in m])
            title = text("how fast each layer moves", 20, theme.GREY_B).next_to(
                g0 + [0, gh, 0], UP, buff=0.15, aligned_edge=LEFT)
            parts.append(title)
            if L <= 12:
                for i, l_ in enumerate(d["layers"]):
                    lbl = text(f"{l_['id'][-14:]}", 12, theme.GREY_C, serif=False)
                    lbl.next_to(g0 + [0, gh - ch * (i + 0.5), 0], LEFT, buff=0.08)
                    parts.append(lbl)
            if d["range"]:
                bar = VGroup(*[Rectangle(width=0.18, height=0.12, stroke_width=0, fill_opacity=1,
                                         fill_color=col).move_to(g0 + [0.18 * k + 0.09, -0.35, 0])
                               for k, col in enumerate(theme.colormap(np.linspace(0, 1, 16),
                                                                      stops=theme.SEQUENTIAL_STOPS, vmin=0, vmax=1))])
                parts += [bar, text(f"{_fmt(d['range'][0])} per step", 12, theme.GREY_C, serif=False)
                          .next_to(bar, DOWN, buff=0.08, aligned_edge=LEFT),
                          text(_fmt(d["range"][1]), 12, theme.GREY_C, serif=False).next_to(bar, RIGHT, buff=0.1)]
        cap = _Captions(self, d.get("captions", True))
        self.play(LaggedStart(*[FadeIn(p) for p in parts], lag_ratio=0.1), *cap.anims(
            "As training runs, the loss on the left falls; on the right each row is a layer — brighter means its "
            "weights are moving faster."), run_time=1.0)
        live = [m for m in (curve, dot, readout) if m is not None]
        self.add(cells, *live)
        self.play(t.animate.set_value(1.0), run_time=6.0, rate_func=linear)
        for m in [cells, *live]:
            m.clear_updaters()
        cap.clear()
        last = "   ".join(f"{k} {_fmt(v)}" for k, v in (d["last"] or {}).items() if v is not None)
        if last:
            summary = text(last, 20, theme.GREY_A, serif=False).to_edge(DOWN, buff=0.3)
            self.play(FadeIn(summary, shift=UP * 0.1), run_time=0.6)
        self.wait(1.2)


SCENES = {"forward": ForwardPassScene, "equation": EquationScene, "focus": FocusScene, "edit": EditScene,
          "architecture": ArchitectureScene, "volumes": VolumesScene, "training": TrainingScene}
__all__ = ["SCENES", "ForwardPassScene", "EquationScene", "FocusScene", "EditScene", "ArchitectureScene",
           "VolumesScene", "TrainingScene"]
