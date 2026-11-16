"""The showcase film: 1 cycle run twice on the same seed, "Skill says done" then "AVAIL", with overlays from each run's
own log, in the style of the robot demo video style guide (GR00T 2's agentic videos). Evaluator-side.

    uv run python -m avail.analysis.film --rec <AVAIL_DATA>/runs/showcase/film --out <AVAIL_DATA>/runs/showcase

Input: the recordings of ``film_record.py`` (events on the sim clock, the frame log at 30 Hz). Footage: a filming-only
free camera at a front three-quarter angle, which exists only in the renderer and feeds no check, on a white scene,
1920x1080 at 30 fps. The real clock is the sim clock plus the operator's holds (the time to answer and a takeover's
handling), which the sim clock does not hold. Modes and their colors, the same in every panel: CODE (a skill moving,
orange), CHECKING (AVAIL's evidence, blue), OPERATOR (the simulated operator, purple), STOPPED (red). Speed: code moves
2x, checks and contact moments (the insertion's guarded moves and the release) 1x, the operator's holds 4x, and the
speed is always in the mode pill. Each run plays uncut.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

W, H, FPS = 1920, 1080, 30
CODE, CHECKING, OPERATOR, STOPPED = "CODE", "CHECKING", "OPERATOR", "STOPPED"
COLOR = {CODE: (245, 145, 40), CHECKING: (70, 150, 245), OPERATOR: (170, 110, 230), STOPPED: (225, 60, 60)}
YELLOW, GREEN, RED, WHITE, GRAY = (250, 205, 40), (60, 190, 100), (225, 60, 60), (255, 255, 255), (170, 175, 182)
PANEL = (28, 30, 34, 205)
SPEED = {"code": 2.0, "contact": 1.0, CHECKING: 1.0, "takeover": 2.0, "hold": 4.0}
STEP_WORDS = {"pick": "Picking", "carry": "Carrying", "insert": "Inserting", "release": "Releasing"}
PREDICATE = {"pick": "held", "carry": "at_port", "insert": "latched", "release": "released_stable"}


def _fonts():
    from matplotlib import font_manager
    from PIL import ImageFont

    sans = font_manager.findfont("DejaVu Sans")
    bold = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
    mono = font_manager.findfont("DejaVu Sans Mono")
    f = lambda p, s: ImageFont.truetype(p, s)          # noqa: E731
    return {"mono": f(mono, 30), "mono_s": f(mono, 24), "mono_xs": f(mono, 22), "mono_h": f(mono, 32), "sans": f(sans, 32),
            "sans_s": f(sans, 26), "sans_xs": f(sans, 22), "sans_l": f(sans, 40), "bold": f(bold, 44), "bold_l": f(bold, 56),
            "badge": f(bold, 24), "badge_s": f(bold, 20), "sans_xl": f(sans, 44)}


# ----------------------------------------------------------------------------- the story, from the log
@dataclass
class Piece:
    sim0: float
    sim1: float
    hold: float          # real seconds of a hold (the sim clock stands still), 0 for motion
    mode: str
    speed: float


def load(rec: Path, arm: str) -> dict:
    d = json.loads((Path(rec) / arm / "events.json").read_text(encoding="utf-8"))
    ev = d["events"]
    for i, e in enumerate(ev):                      # the recorded order: calls end in it, and nothing nests but substeps
        e["_i"] = i
    return {"arm": arm, "seed": d["seed"], "events": ev,
            "skills": [e for e in ev if e["kind"] == "skill"],
            "subs": [e for e in ev if e["kind"] == "substep"],
            "checks": [e for e in ev if e["kind"] == "check"],
            "ops": [e for e in ev if e["kind"] == "operator"],
            "plan": next(e for e in ev if e["kind"] == "plan")["cables"],
            "end": next(e for e in ev if e["kind"] == "end")}


class Timeline:
    """The real clock: the sim clock plus the time it does not hold, as the driver counts it. A check's stated cost
    falls as it ends. The operator's time to answer falls as they are asked (none for a takeover right after their own
    answer on the same step: they are at the cell already), and a takeover's handling as the robot is handed back. At 1
    sim instant, holds fall in the order the calls ran, so each is keyed by (sim time, event index, phase)."""

    def __init__(self, run: dict):
        self.holds: list[tuple[float, tuple[int, int], float, str]] = []
        for c in run["checks"]:
            if c.get("cost_s", 0.0) > 0:
                self.holds.append((c["t_end"], (c["_i"], 1), float(c["cost_s"]), CHECKING))
        ops = sorted(run["ops"], key=lambda o: o["_i"])
        for j, o in enumerate(ops):
            prev = ops[j - 1] if j else None
            present = (o["question"] == "takeover" and prev is not None and prev["item"] == o["item"]
                       and prev["step"] == o["step"] and abs(o["t"] - prev["t_end"]) < 0.05)
            if not present and o["seconds_to_answer"] > 0:
                self.holds.append((o["t"], (o["_i"], 0), float(o["seconds_to_answer"]), OPERATOR))
            if o["seconds_handling"] > 0:
                self.holds.append((o["t_end"], (o["_i"], 2), float(o["seconds_handling"]), OPERATOR))
        self.holds.sort(key=lambda h: (h[0], h[1]))

    def rt(self, sim: float, key: tuple[int, int] = (10 ** 9, 9), own: bool = False) -> float:
        """The real clock at a point: sim time ``sim``, after every hold that falls before key ``key`` (and the hold at
        ``key`` itself, when ``own``). The default key puts the point after every hold at that sim time."""
        held = sum(s for t, k, s, _ in self.holds if t < sim - 1e-9 or (abs(t - sim) <= 1e-9 and (k < key or (own and k == key))))
        return sim + held


def pieces(run: dict, t_end: float) -> list[Piece]:
    """The run cut into pieces of 1 mode and 1 speed, with the operator's holds where they fall."""
    cuts = {0.0, t_end}
    for e in run["skills"] + run["checks"] + run["ops"]:
        cuts |= {e["t"], e["t_end"]}
    for e in run["subs"]:
        cuts |= {e["t"], e["t_end"]}
    cuts = sorted(c for c in cuts if 0.0 <= c <= t_end)

    def inside(lst, t):
        return next((e for e in lst if e["t"] <= t < e["t_end"]), None)

    out: list[Piece] = []
    holds = list(Timeline(run).holds)

    def emit(at: float) -> None:                    # the holds at this sim instant, in the order the calls ran
        while holds and holds[0][0] <= at + 1e-9:
            t, _, secs, mode = holds.pop(0)
            out.append(Piece(t, t, secs, mode, SPEED[CHECKING] if mode == CHECKING else SPEED["hold"]))

    for a, b in zip(cuts, cuts[1:]):
        emit(a)
        if b - a <= 1e-9:
            continue
        mid = (a + b) / 2
        if inside(run["ops"], mid):
            out.append(Piece(a, b, 0.0, OPERATOR, SPEED["takeover"]))
        elif inside(run["checks"], mid):
            out.append(Piece(a, b, 0.0, CHECKING, SPEED[CHECKING]))
        else:
            sk = inside(run["skills"], mid)
            sub = inside(run["subs"], mid)
            contact = sk is not None and (sk["step"] == "release" or (sk["step"] == "insert" and sub is not None
                                                                      and sub["fn"] in ("move_relative", "wait")))
            out.append(Piece(a, b, 0.0, CODE, SPEED["contact"] if contact else SPEED["code"]))
    emit(t_end + 1.0)
    merged: list[Piece] = []
    for p in out:                                   # neighbors of 1 mode and speed play as 1 piece
        if merged and not p.hold and not merged[-1].hold and merged[-1].mode == p.mode and merged[-1].speed == p.speed:
            merged[-1].sim1 = p.sim1
        else:
            merged.append(p)
    return merged


def schedule(ps: list[Piece]) -> list[tuple[float, float, str, float]]:
    """1 entry per film frame: (sim time, real clock, mode, speed)."""
    out, clock, carry = [], 0.0, 0.0
    for p in ps:
        real = p.hold if p.hold else (p.sim1 - p.sim0)
        frames = (real / p.speed) * FPS + carry
        n = int(frames)
        carry = frames - n
        advance = 0.0 if p.mode == STOPPED else real     # the end card's pause shows the run over: no real time passes
        for k in range(n):
            f = (k + 0.5) / max(n, 1)
            sim = p.sim0 if p.hold else p.sim0 + f * (p.sim1 - p.sim0)
            out.append((sim, clock + f * advance, p.mode, p.speed))
        clock += advance
    return out


def mode_spans(ps: list[Piece]) -> list[tuple[float, float, str]]:
    spans, clock = [], 0.0
    for p in ps:
        if p.mode == STOPPED:
            continue
        real = p.hold if p.hold else (p.sim1 - p.sim0)
        spans.append((clock, clock + real, p.mode))
        clock += real
    return spans


def event_times(run: dict) -> None:
    """Each skill, check and operator call gets its real start and end: ``rt0`` as it is called, ``rt1`` once its result
    is known (after its own hold: a check's cost, the operator's answer, a takeover's handling)."""
    tl = run["_tl"] = Timeline(run)
    for e in run["skills"]:
        e["rt0"], e["rt1"] = tl.rt(e["t"], (e["_i"], -1)), tl.rt(e["t_end"], (e["_i"], -1))
    for c in run["checks"]:
        c["rt0"], c["rt1"] = tl.rt(c["t"], (c["_i"], -1)), tl.rt(c["t_end"], (c["_i"], 1), own=True)
    for o in run["ops"]:
        o["rt0"] = tl.rt(o["t"], (o["_i"], -1))
        o["rt1"] = tl.rt(o["t_end"], (o["_i"], 2), own=True) if o["question"] == "takeover" else tl.rt(o["t_end"], (o["_i"], 0), own=True)


def clock_txt(s: float) -> str:
    return f"{int(s // 60):02d}:{int(s % 60):02d}"


# ----------------------------------------------------------------------------- what the log says
def verdict_line(c: dict) -> str:
    """A check as signal, then conclusion."""
    ev, n = c["evidence"], int(c["step_id"].split(".")[0]) + 1
    d, p = c["decision"], c["predicate"]
    g = lambda s, k, default=None: ev.get(s, {}).get(k, default)          # noqa: E731
    tail = {"accept": "", "reject": ", retry", "abstain": ": unsure, operator called"}[d]
    if p == "held":
        sig = "color matches" if g("vision.cable_tag", "match") else ("jaws closed on nothing" if g("passive.jaw_hold", "closed_empty") else "color does not match")
        con = f"cable {n} held" if d == "accept" else f"cable {n} not held"
    elif p == "at_port":
        lat = g("passive.hover_pose", "lateral_mm")
        sig = f"tool {lat:.1f} mm from its port pose" if lat is not None else "tool pose unread"
        con = f"cable {n} over its port" if d == "accept" else f"cable {n} not confirmed over its port"
    elif p == "released_stable":
        gap = g("vision.gap", "gap_mm_est")
        if g("passive.jaw_hold", "open") == 0.0:
            sig = "jaws did not open"
        elif g("passive.release_record", "lifted_closed") == 1.0:
            sig = "lifted with the jaws shut"
        else:
            sig = "jaws open" + (f", camera gap {gap:.1f} mm" if gap is not None else "")
        con = f"cable {n} released" if d == "accept" else f"cable {n} not released"
    else:
        if g("probe.tug", "disp_mm") is not None:
            sig = f"tug moved {g('probe.tug', 'disp_mm'):.1f} mm"
        elif g("probe.push_seat", "travel_mm") is not None:
            sig = f"push moved {g('probe.push_seat', 'travel_mm'):.1f} mm"
        else:
            axial = g("passive.pose_residual", "axial_mm", 0.0)
            sig = ("click felt" if g("passive.click", "clicked") else "no click") + f", {abs(axial):.1f} mm from home"
        con = f"cable {n} latched" if d == "accept" else f"cable {n} not latched"
    return f"{sig}: {con}{tail}"


def log_lines(run: dict, ps: list[Piece], avail: bool) -> list[dict]:
    """The log as the film shows it: each line with the real clock it appears at (``event_times`` first)."""
    lines = [{"at": 0.0, "seq": -1, "kind": "say",
              "text": "Plan: cables 1-5 into ports " + ", ".join(str(c["port"]) for c in run["plan"])}]
    for e in sorted(run["skills"] + run["checks"] + run["ops"], key=lambda e: (e["rt0"], e["_i"])):
        if e["kind"] == "skill":
            n = e["item"] + 1
            if e["attempt"] == 0:
                lines.append({"at": e["rt0"], "seq": e["_i"], "kind": "say", "text": f"{STEP_WORDS[e['step']]} cable {n}."})
            lines.append({"at": e["rt0"], "seq": e["_i"], "kind": "call", "mode": CODE, "text": "> run_skill",
                          "args": f"step='{e['step']}', item={e['item']}, attempt={e['attempt']}"})
            if not avail:
                word = "done" if e["flag"] else "not done"
                lines.append({"at": e["rt1"], "seq": e["_i"], "kind": "say",
                              "text": f"The skill says {word}: cable {n} {PREDICATE[e['step']]}."})
        elif e["kind"] == "check":
            lines.append({"at": e["rt0"], "seq": e["_i"], "kind": "call", "mode": CHECKING, "text": "> verify",
                          "args": f"predicate='{e['predicate']}', item={e['step_id'].split('.')[0]}"})
            lines.append({"at": e["rt1"], "seq": e["_i"], "kind": "verdict", "decision": e["decision"], "text": verdict_line(e)})
        else:
            lines.append({"at": e["rt0"], "seq": e["_i"], "kind": "call", "mode": OPERATOR, "text": "> ask",
                          "args": f"question='{e['question']}', item={e['item']}, step='{e['step']}'"})
            ans = {"confirm": f"Operator: {e['answer']}.", "choose": f"Operator chooses {e['answer']}.",
                   "takeover": "Operator took over and finished the step."}.get(e["question"], f"Operator: {e['answer']}.")
            lines.append({"at": e["rt1"], "seq": e["_i"], "kind": "say", "text": ans})
    for k, at in enumerate(sorted(cable_done_at(run, ps, avail).values()), 1):
        lines.append({"at": at, "seq": 10 ** 9, "kind": "count", "text": f"{k}/5 cables in"})
    lines.sort(key=lambda l: (l["at"], l["seq"]))
    return lines


def cable_done_at(run: dict, ps: list[Piece], avail: bool) -> dict[int, float]:
    """The real clock at which each cable's badge turns green: AVAIL accepted (or the operator signed off) its release,
    or, on the skill-only run, the skill said its release was done."""
    out = {}
    if avail:
        for e in run["checks"]:
            if e["predicate"] == "released_stable" and e["decision"] == "accept":
                out.setdefault(int(e["step_id"].split(".")[0]), e["rt1"])
        for o in run["ops"]:
            if o["step"] == "release" and ((o["question"] == "confirm" and o["answer"] == "yes")
                                           or (o["question"] == "takeover" and o["truth_label"])):
                out.setdefault(o["item"], o["rt1"])
    else:
        for e in run["skills"]:
            if e["step"] == "release" and e["flag"]:
                out.setdefault(e["item"], e["rt1"])
    return out

# ----------------------------------------------------------------------------- footage
class Stage:
    """The recorded cycle re-rendered: the filming camera on a white scene, and the wrist camera for the insets."""

    def __init__(self, rec: Path, arm: str, seed: int):
        import mujoco
        import pandas as pd

        from ..station.mj_plug import scene as plug_scene
        from ..station.mj_plug.cable_replay import Replay

        folder = Path(rec) / arm
        frames = pd.read_parquet(folder / "frames.parquet")
        self.t = frames["t"].to_numpy(dtype=float)
        self.rep = Replay(seed, "latch", frames, json.loads((folder / "scene.json").read_text(encoding="utf-8")))
        m = self.rep.model
        self.mj = mujoco
        for t in range(m.ntex):                         # a white backdrop, a white floor, a light gray table
            if m.tex_type[t] == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
                a, h, w = m.tex_adr[t], m.tex_height[t], m.tex_width[t]
                m.tex_data[a:a + h * w * m.tex_nchannel[t]] = 255
        for g in range(m.ngeom):
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
                m.geom_matid[g] = -1
                m.geom_rgba[g] = (1, 1, 1, 1)
        for mt in range(m.nmat):
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, mt) or "").startswith("table_mat"):
                m.mat_texid[mt][:] = -1
                m.mat_rgba[mt] = (0.70, 0.70, 0.69, 1)
        m.vis.headlight.ambient[:] = 0.25
        m.vis.headlight.diffuse[:] = 0.30
        m.vis.global_.offwidth, m.vis.global_.offheight = W, H
        self.film = mujoco.Renderer(m, H, W)
        self.wrist = mujoco.Renderer(m, 360, 480)
        st = self.rep.cell.station
        nom = st.nominal()
        board = np.array([nom.board_xy[0], nom.board_xy[1], nom.face_z])
        from ..station.mj_plug import cable_scene

        tray, _ = cable_scene.slot_world(st._cfg, 2)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = (board + np.array([tray[0], tray[1], board[2]])) / 2 + np.array([0.10, 0.0, 0.10])
        self.cam.azimuth, self.cam.elevation, self.cam.distance = 150.0, -30.0, 1.0
        self.ports = {k: plug_scene.port_pose_world(st._cfg, st._lay, c.port)[0] for k, c in enumerate(self.rep.cell.cables)}
        self.fovy = float(m.vis.global_.fovy)
        self._cache: dict = {}

    def index(self, sim: float) -> int:
        return int(np.clip(np.searchsorted(self.t, sim), 0, len(self.t) - 1))

    def _pose(self, idx: int) -> None:
        row = self.rep.frames.iloc[idx]
        self.rep._scene_at(int(row["frame_index"]))
        self.rep.data.qpos[:] = np.asarray(row["qpos"], dtype=float)
        self.mj.mj_forward(self.rep.model, self.rep.data)

    def frame(self, sim: float) -> np.ndarray:
        idx = self.index(sim)
        if ("film", idx) not in self._cache:
            self._cache = {k: v for k, v in self._cache.items() if k[0] != "film"}
            self._pose(idx)
            self.film.update_scene(self.rep.data, camera=self.cam)
            self._cache[("film", idx)] = self.film.render().copy()
        return self._cache[("film", idx)]

    def wrist_frame(self, sim: float) -> np.ndarray:
        idx = self.index(sim)
        if ("wrist", idx) not in self._cache:
            self._cache = {k: v for k, v in self._cache.items() if k[0] != "wrist"}
            self._pose(idx)
            self.wrist.update_scene(self.rep.data, camera="right_camera_d405")
            self._cache[("wrist", idx)] = self.wrist.render().copy()
        return self._cache[("wrist", idx)]

    def project(self, p) -> tuple[float, float] | None:
        """Where a world point lands in the film frame, from the free camera's pose."""
        az, el = math.radians(self.cam.azimuth), math.radians(self.cam.elevation)
        fwd = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        pos = np.asarray(self.cam.lookat) - self.cam.distance * fwd
        right = np.cross(fwd, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        v = np.asarray(p, float) - pos
        z = v @ fwd
        if z <= 1e-6:
            return None
        f = (H / 2) / math.tan(math.radians(self.fovy) / 2)
        return W / 2 + f * (v @ right) / z, H / 2 - f * (v @ up) / z

    def close(self) -> None:
        self.film.close()
        self.wrist.close()
        self.rep.close()


# ----------------------------------------------------------------------------- drawing
def _panel(draw, box, accent=None, bar=False):
    draw.rounded_rectangle(box, radius=14, fill=PANEL, outline=(accent or (90, 92, 98)) + (255,), width=2)
    if bar:
        draw.rounded_rectangle((box[0], box[1], box[0] + 6, box[3]), radius=3, fill=(accent or WHITE) + (255,))


def _badge(draw, xy, n, color, font, r=17, ring=None):
    x, y = xy
    if ring:
        draw.ellipse((x - r - 8, y - r - 8, x + r + 8, y + r + 8), outline=ring + (255,), width=6)
    draw.ellipse((x - r, y - r, x + r, y + r), fill=color + (255,), outline=(20, 20, 20, 255), width=2)
    tw = draw.textlength(str(n), font=font)
    draw.text((x - tw / 2, y - 12), str(n), fill=(20, 20, 20, 255), font=font)


def _wrap(text: str, width: int) -> list[str]:
    """Words onto lines of at most ``width`` characters, the continuation indented."""
    out, cur = [], ""
    for word in text.split(" "):
        if cur and len(cur) + 1 + len(word) > width:
            out.append(cur)
            cur = "  " + word
        else:
            cur = f"{cur} {word}" if cur else word
    return out + [cur]


def _active(lst, sim):
    return next((e for e in lst if e["t"] <= sim < e["t_end"]), None)


def _sub_label(s: dict) -> tuple[str, str]:
    a = s.get("args") or []
    if s["fn"] == "set_gripper" and a:
        return "set_gripper", f"{float(a[0]):.1f}"
    if s["fn"] == "move_relative" and a:
        d = a[0] if isinstance(a[0], list) else []
        return "move_relative", (f"z {1000 * float(d[2]):+.0f} mm" if len(d) == 3 else "")
    if s["fn"] == "wait" and a:
        return "wait", f"{float(a[0]):.1f} s"
    return s["fn"], ""


def overlay(img, run, stage, ps, spans, lines, green_at, sim, clock, mode, speed, fonts, avail, total, final=False):
    """The 7 panels on 1 film frame, sized to be read on a phone held sideways (the log's type at 30 px of 1,080)."""
    from PIL import Image, ImageDraw

    base = Image.fromarray(img).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    # 6: the ports, numbered like the work order, green once done, a red ring on a truly bad cable at the end
    truth = {it["item"]: it["true"] for it in run["end"]["items"]}
    for k, p in stage.ports.items():
        uv = stage.project(p)
        if uv is None:
            continue
        done = k in green_at and clock >= green_at[k]
        ring = RED if (final and not avail and not truth.get(k, True)) else None
        _badge(d, (uv[0], uv[1] - 24), k + 1, GREEN if done else YELLOW, fonts["badge_s"], r=15, ring=ring)
    # 1: the log, newest at the bottom
    LW, ROW, ARGS, TOP, MAXH = 860, 38, 32, 24, 600
    head = ("AVAIL" if avail else "SKILL ONLY") + " · " + clock_txt(clock)
    shown = [l for l in lines if l["at"] <= clock + 1e-6]
    keep, used = [], 0
    for l in reversed(shown):
        if l["kind"] != "call":
            l.setdefault("wrapped", _wrap(l["text"], 38))
        h = ROW + ARGS if l["kind"] == "call" else ROW * len(l["wrapped"])
        if used + h > MAXH - 80:
            break
        keep.append(l)
        used += h
    keep.reverse()
    box = (24, TOP, 24 + LW, TOP + 80 + used + 14)
    _panel(d, box, accent=COLOR[mode], bar=True)
    d.text((48, TOP + 14), head, fill=COLOR[mode] + (255,), font=fonts["mono_h"])
    if avail:                                       # the skill-only run checks nothing
        d.text((48 + d.textlength(head, font=fonts["mono_h"]) + 18, TOP + 22), "witness version tested in E3",
               fill=GRAY + (255,), font=fonts["sans_xs"])
    y = TOP + 74
    for l in keep:
        d.text((48, y + 5), clock_txt(l["at"]), fill=GRAY + (255,), font=fonts["mono_xs"])
        x = 48 + 96
        if l["kind"] == "call":
            d.text((x, y), l["text"], fill=COLOR[l["mode"]] + (255,), font=fonts["mono"])
            d.text((x + 28, y + ROW - 4), l["args"], fill=GRAY + (255,), font=fonts["mono_s"])
            y += ROW + ARGS
        else:
            color = WHITE
            if l["kind"] == "verdict":
                color = {"accept": (150, 230, 170), "reject": (255, 150, 150), "abstain": (215, 185, 250)}[l["decision"]]
            if l["kind"] == "count":
                color = YELLOW
            for part in l["wrapped"]:
                d.text((x, y), part, fill=color + (255,), font=fonts["mono"])
                y += ROW
    # 5: the work order
    tx0 = W - 24 - 610
    _panel(d, (tx0, 24, W - 24, 104))
    d.text((tx0 + 20, 48), "WORK ORDER", fill=WHITE + (255,), font=fonts["badge"])
    for k, c in enumerate(run["plan"]):
        x = tx0 + 236 + k * 72
        done = k in green_at and clock >= green_at[k]
        _badge(d, (x, 64), k + 1, GREEN if done else YELLOW, fonts["badge"], r=19)
        d.text((x + 24, 48), str(c["port"]), fill=WHITE + (255,), font=fonts["sans_s"])
    # 4: the skill now running and its substeps
    sk = _active(run["skills"], sim)
    if sk is not None and mode == CODE:
        subs = [s for s in run["subs"] if sk["t"] <= s["t"] < sk["t_end"] + 1e-9]
        now = _active(subs, sim)
        view = [s for s in subs if s["t"] <= sim][-4:]
        box = (tx0, 120, W - 24, 120 + 64 + 4 * 40)
        _panel(d, box, accent=COLOR[CODE])
        d.text((tx0 + 20, 132), f"run_skill('{sk['step']}')", fill=COLOR[CODE] + (255,), font=fonts["mono"])
        for j, s in enumerate(view):
            name, arg = _sub_label(s)
            active = s is now
            yy = 178 + j * 40
            d.text((tx0 + 20, yy), "▶" if active else " ", fill=WHITE + (255,), font=fonts["mono"])
            d.text((tx0 + 54, yy), name, fill=(WHITE if active else GRAY) + (255,), font=fonts["mono"])
            d.text((tx0 + 54 + d.textlength(name, font=fonts["mono"]) + 14, yy + 4), arg, fill=GRAY + (255,), font=fonts["mono_s"])
    # 7: insets. The wrist camera at insertion and release, a verdict card per check, the operator's card
    ck = _active(run["checks"], sim)
    op = next((o for o in run["ops"] if o["rt0"] <= clock <= o["rt1"] + 1e-6), None)
    contact = (sk is not None and sk["step"] in ("insert", "release")) or (ck is not None and ck["predicate"] in ("latched", "released_stable")) \
        or (op is not None and op["step"] in ("insert", "release"))
    if contact and not final:
        wr = Image.fromarray(stage.wrist_frame(sim)).convert("RGBA").resize((400, 300))
        x0, y0 = W - 24 - 400, 418 if (sk is not None and mode == CODE) else 170
        d.rounded_rectangle((x0 - 8, y0 - 46, x0 + 408, y0 + 308), radius=14, fill=PANEL)
        d.text((x0 + 8, y0 - 40), "wrist camera", fill=WHITE + (255,), font=fonts["sans_s"])
        layer.paste(wr, (x0, y0))
    card = None
    for i, e in enumerate(run["checks"]):
        if e["rt0"] <= clock <= run["_card_until"][i]:
            card = ("AVAIL verdict", verdict_line(e), {"accept": GREEN, "reject": RED, "abstain": COLOR[OPERATOR]}[e["decision"]])
    if op is not None:
        word = {"confirm": f"confirm: {op['answer']}", "choose": f"choose: {op['answer']}",
                "takeover": "takes over the step"}.get(op["question"], op["answer"])
        card = ("Simulated operator (answers from sim truth)", f"cable {op['item'] + 1} {op['step']}: {word}", COLOR[OPERATOR])
    if card and not final:
        title, text, color = card
        tw = max(d.textlength(text, font=fonts["sans"]), d.textlength(title, font=fonts["sans_s"])) + 56
        x0, y0 = max(900, (W - tw) / 2), H - 296
        d.rounded_rectangle((x0, y0, x0 + tw, y0 + 112), radius=16, fill=PANEL, outline=color + (255,), width=4)
        d.text((x0 + 28, y0 + 14), title, fill=color + (255,), font=fonts["sans_s"])
        d.text((x0 + 28, y0 + 54), text, fill=WHITE + (255,), font=fonts["sans"])
    # 2: the mode pill
    d.rounded_rectangle((24, H - 120, 24 + 740, H - 24), radius=46, fill=PANEL)
    d.ellipse((56, H - 88, 88, H - 56), fill=COLOR[mode] + (255,))
    d.text((106, H - 98), mode, fill=WHITE + (255,), font=fonts["bold"])
    d.text((410, H - 92), f"▶▶ {speed:g}×", fill=WHITE + (255,), font=fonts["sans"])
    d.text((580, H - 92), clock_txt(clock), fill=COLOR[mode] + (255,), font=fonts["sans"])
    # 3: the timeline
    bx0, bx1, by = 806, W - 40, H - 66
    d.rounded_rectangle((bx0 - 18, H - 140, W - 24, H - 24), radius=16, fill=PANEL)
    for a, b, m in spans:
        d.rectangle((bx0 + (bx1 - bx0) * a / total, by, bx0 + (bx1 - bx0) * b / total, by + 28), fill=COLOR[m] + (255,))
    px = bx0 + (bx1 - bx0) * min(clock, total) / total
    d.rectangle((px - 3, by - 10, px + 3, by + 38), fill=WHITE + (255,))
    share = {}
    for a, b, m in spans:
        share[m] = share.get(m, 0.0) + (b - a)
    x = bx0
    for m in (CODE, CHECKING, OPERATOR):
        if m in share:
            pct = 100 * share[m] / total
            t = f"{m} {pct:.0f}%" if pct >= 1 else f"{m} {pct:.1f}%"
            d.rectangle((x, H - 126, x + 18, H - 108), fill=COLOR[m] + (255,))
            d.text((x + 26, H - 132), t, fill=WHITE + (255,), font=fonts["sans_s"])
            x += d.textlength(t, font=fonts["sans_s"]) + 56
    t = f"run {clock_txt(total)}"
    d.text((bx1 - d.textlength(t, font=fonts["sans_s"]), H - 132), t, fill=GRAY + (255,), font=fonts["sans_s"])
    return np.asarray(Image.alpha_composite(base, layer).convert("RGB"))


# ----------------------------------------------------------------------------- the film
def title_card(fonts, line1: str, line2: str, color) -> np.ndarray:
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (W, H), (250, 250, 249))
    d = ImageDraw.Draw(im)
    d.ellipse((W / 2 - 520, H / 2 - 74, W / 2 - 492, H / 2 - 46), fill=color)
    d.text((W / 2 - 470, H / 2 - 90), line1, fill=(25, 25, 28), font=fonts["bold_l"])
    d.text((W / 2 - 470, H / 2 - 10), line2, fill=(90, 92, 98), font=fonts["sans_l"])
    d.text((W / 2 - 470, H / 2 + 60), "Simulation. The operator is simulated: it answers from sim truth.", fill=(120, 122, 128),
           font=fonts["sans"])
    return np.asarray(im)


def end_card(fonts, skill_final: np.ndarray, avail_final: np.ndarray, skill_end: dict, avail_end: dict) -> np.ndarray:
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (W, H), (250, 250, 249))
    d = ImageDraw.Draw(im)
    tw, th = 900, 506
    for k, (img, end, title) in enumerate(((skill_final, skill_end, "Skill says done"),
                                           (avail_final, avail_end, "AVAIL + simulated operator"))):
        x = 40 + k * (tw + 40)
        im.paste(Image.fromarray(img).resize((tw, th)), (x, 120))
        claimed = sum(1 for it in end["items"] if it["declared"])
        bad = sum(1 for it in end["items"] if it["declared"] and not it["true"])
        d.text((x, 50), title, fill=(25, 25, 28), font=fonts["bold"])
        line = f"{claimed} claimed, {bad} truly bad" if bad else f"{sum(it['true'] for it in end['items'])} truly in"
        d.text((x, 640), line, fill=(RED if bad else GREEN), font=fonts["sans_xl"])
    d.text((40, 760), "Bad cable shipped: 67.3% of cycles trusting the skill,", fill=(25, 25, 28), font=fonts["sans_xl"])
    d.text((40, 818), "0.62% with AVAIL + simulated operator (5,000 cycles).", fill=(25, 25, 28), font=fonts["sans_xl"])
    d.text((40, 890), "Simulation. E3 on latch, the shift test, read once. Above: dev seed, the cycle picked by the written rule.",
           fill=(110, 112, 118), font=fonts["sans"])
    return np.asarray(im)


def final_view(stage, run: dict, green_at: dict, fonts, avail: bool) -> np.ndarray:
    """The run's last frame without the panels, cropped to the board: each port's badge, and a red ring on every
    cable that is truly not done."""
    from PIL import Image, ImageDraw

    im = Image.fromarray(stage.frame(float(stage.t[-1]))).convert("RGB")
    d = ImageDraw.Draw(im)
    truth = {it["item"]: it["true"] for it in run["end"]["items"]}
    pts = []
    for k, p in stage.ports.items():
        uv = stage.project(p)
        if uv is None:
            continue
        pts.append(uv)
        _badge(d, (uv[0], uv[1] - 24), k + 1, GREEN if k in green_at else YELLOW, fonts["badge_s"], r=15,
               ring=None if truth.get(k, True) else RED)
    cx, cy = np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts])
    w, h = 720, 405
    x0, y0 = int(np.clip(cx - w / 2, 0, W - w)), int(np.clip(cy - h / 2 - 20, 0, H - h))
    return np.asarray(im.crop((x0, y0, x0 + w, y0 + h)))


def run_frames(rec: Path, arm: str, fonts, writer, avail: bool, stills: dict) -> tuple[np.ndarray, dict]:
    run = load(rec, arm)
    stage = Stage(rec, arm, run["seed"])
    try:
        t_end = float(stage.t[-1])
        ps = pieces(run, t_end)
        ps.append(Piece(t_end, t_end, 1.5, STOPPED, 1.0))
        event_times(run)
        total = sum(p.hold if p.hold else p.sim1 - p.sim0 for p in ps if p.mode != STOPPED)
        spans = mode_spans(ps)
        lines = log_lines(run, ps, avail)
        green_at = cable_done_at(run, ps, avail)
        sched = schedule(ps)
        clocks = np.array([s[1] for s in sched])
        run["_card_until"] = []                     # each verdict card stays up 2.5 s of film after its check ends
        for e in run["checks"]:
            j = int(np.searchsorted(clocks, e["rt1"]))
            run["_card_until"].append(float(clocks[min(j + int(2.5 * FPS), len(clocks) - 1)]))
        catch_at = None
        if avail:
            first_reject = next((e for e in sorted(run["checks"], key=lambda e: e["t"]) if e["decision"] == "reject"
                                 and e["predicate"] in ("latched", "released_stable")), None)
            if first_reject is not None:
                catch_at = first_reject["rt1"] + 1.0
        last = None
        for i, (sim, clock, mode, speed) in enumerate(sched):
            final = mode == STOPPED
            img = overlay(stage.frame(sim), run, stage, ps, spans, lines, green_at, sim, clock, mode, speed, fonts, avail,
                          total, final=final)
            writer.append_data(img)
            if avail and i == 0:
                stills["plan"] = img
            if catch_at is not None and "catch" not in stills and clock >= catch_at:
                stills["catch"] = img
            last = img
        return final_view(stage, run, green_at, fonts, avail), run["end"]
    finally:
        stage.close()


def main(argv: list[str] | None = None) -> int:
    import imageio
    from PIL import Image

    ap = argparse.ArgumentParser(prog="python -m avail.analysis.film", description=__doc__.split("\n\n")[0])
    ap.add_argument("--rec", required=True, help="the recordings of film_record.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="avail_showcase.mp4")
    args = ap.parse_args(argv)
    fonts = _fonts()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out / args.name), fps=FPS, codec="libx264", quality=8, macro_block_size=8)
    stills: dict = {}
    try:
        for _ in range(2 * FPS):
            writer.append_data(title_card(fonts, "Run 1: SKILL ONLY", "Each step trusts the skill's own done.", COLOR[CODE]))
        skill_final, skill_end = run_frames(Path(args.rec), "flag", fonts, writer, False, stills)
        for _ in range(2 * FPS):
            writer.append_data(title_card(fonts, "Run 2: AVAIL + simulated operator",
                                          "Same seed, same failures. Every step is checked.", COLOR[CHECKING]))
        avail_final, avail_end = run_frames(Path(args.rec), "avail_simop", fonts, writer, True, stills)
        card = end_card(fonts, skill_final, avail_final, skill_end, avail_end)
        for _ in range(8 * FPS):
            writer.append_data(card)
        stills["end"] = card
    finally:
        writer.close()
    for name in ("plan", "catch", "end"):
        if name in stills:
            Image.fromarray(stills[name]).save(out / f"avail_showcase_{name}.png")
    print("written:", out / args.name, sorted(stills))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
