"""Clips of the plug station for a person to watch. mp4 only. Evaluator-side.

One clip per connector: a seated insert and the pull that holds, then a short insert and the
pull that comes out. The left half is a fixed camera next to the board, the right half is the
robot's own wrist camera. The caption carries the ground truth, which the robot never sees.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...config import load_experiment
from ...seeds import guard
from .inject import ARM, ErrorSpec, scripted_insert
from .truth import truth

FPS = 30
PANEL_W, PANEL_H = 640, 480
CAPTION_H = 64


class _Recorder:
    """Collects frames at a fixed rate from a station while it runs."""

    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []
        self.caption = ""
        self.station = None
        self._next = 0.0
        self._renderer = None

    def attach(self, station) -> None:
        import mujoco

        self.station = station
        self._next = float(station._mj_data.time)
        self._renderer = mujoco.Renderer(station._mj_model, PANEL_H, PANEL_W)

    def detach(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
        self._renderer, self.station = None, None

    def __call__(self, station=None) -> None:
        st = self.station
        if st is None:
            return
        data = st._mj_data
        while float(data.time) >= self._next:
            self._next += 1.0 / FPS
            self.frames.append(self._compose())

    def hold(self, seconds: float) -> None:
        """Repeat the current picture, for a caption a person needs time to read."""
        frame = self._compose()
        self.frames.extend([frame] * int(round(seconds * FPS)))

    def _compose(self) -> np.ndarray:
        import mujoco
        from PIL import Image, ImageDraw, ImageFont

        st = self.station
        mujoco.mj_kinematics(st._mj_model, st._mj_data)
        mujoco.mj_camlight(st._mj_model, st._mj_data)
        panels = []
        for cam in ("video_cam", "right_camera_d405"):
            self._renderer.update_scene(st._mj_data, camera=cam)
            panels.append(self._renderer.render().copy())
        canvas = np.full((PANEL_H + CAPTION_H, 2 * PANEL_W, 3), 255, dtype=np.uint8)
        canvas[:PANEL_H, :PANEL_W] = panels[0]
        canvas[:PANEL_H, PANEL_W:] = panels[1]
        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 26)
        except OSError:
            font = ImageFont.load_default()
        draw.text((16, PANEL_H + 16), self.caption, fill=(20, 20, 20), font=font)
        return np.asarray(img)


def _pull(station, recorder: _Recorder, cap_n: float, travel_m: float, speed: float) -> dict:
    z0 = truth(station)["metrics"]["plug_pos"][2]
    station._tick_hook = recorder
    try:
        result = station.move_relative(ARM, [0.0, 0.0, travel_m], "base", speed, cap_n, travel_m)
    finally:
        station._tick_hook = None
    z1 = truth(station)["metrics"]["plug_pos"][2]
    return {"moved_mm": (z1 - z0) * 1000.0, "stopped_by": result.stopped_by, "force_n": result.force_at_stop_n}


def make_clip(connector: str, seed: int, out_dir: Path) -> dict:
    """Write ``<connector>_seated_vs_short.mp4`` and return what happened in it."""
    import imageio

    guard([seed], purpose=f"S2 clip, {connector}")
    exp = load_experiment()
    probe = exp.probes.tug
    cap = exp.tug_force_cap_n(connector)
    recorder = _Recorder()
    summary: dict = {"connector": connector, "seed": int(seed), "fps": FPS, "speed": "real time"}
    for label, spec in (("seated", ErrorSpec("P0")), ("short", ErrorSpec("N2", shortfall_mm=1.6))):
        recorder.caption = f"{connector}: inserting"

        def on_station(station, rec=recorder) -> None:
            if rec.station is not station:
                rec.detach()
                rec.attach(station)
            rec()

        station, outcome = scripted_insert(seed, connector, spec, recorder=on_station)
        short = outcome.truth["metrics"]["shortfall_mm"]
        state = "seated" if outcome.landed_class == "P0" else f"{short:.1f} mm short"
        recorder.caption = f"{connector}: {state}. The skill reports {'done' if outcome.flag else 'not done'}"
        recorder.hold(1.5)
        recorder.caption = f"{connector}: {state}. Pulling, cap {cap:.1f} N"
        pull = _pull(station, recorder, cap, probe.travel_cap_mm / 1000.0, probe.speed_mm_s / 1000.0)
        recorder.caption = f"{connector}: {state}. The pull moved the plug {max(0.0, pull['moved_mm']):.1f} mm"
        recorder.hold(2.0)
        summary[label] = {"landed": outcome.landed_class, "shortfall_mm": short, "skill_flag": bool(outcome.flag), **pull}
        recorder.detach()
        station.close()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{connector}_seated_vs_short.mp4"
    writer = imageio.get_writer(str(path), fps=FPS, codec="libx264", quality=7, macro_block_size=16)
    try:
        for frame in recorder.frames:
            writer.append_data(frame)
    finally:
        writer.close()
    summary["path"] = str(path)
    summary["frames"] = len(recorder.frames)
    summary["seconds"] = round(len(recorder.frames) / FPS, 1)
    return summary
