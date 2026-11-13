"""Replay a cable-loop cycle from its frame log: pictures for a person to look at. Evaluator-side.

A cycle stores its frames at 20 Hz (the joint angles of the whole model) and the scene events (where
every visual cable and the tray pocket were, and the physical plug's color). This module rebuilds the
same cell for the seed, writes each frame's state into it and renders the wrist camera and the clip
camera beside the board. It moves no physics, so a replay shows exactly what ran.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .cable_cell import CableCell
from .cable_scene import N_CABLES

PANEL_W, PANEL_H = 640, 480


class Replay:
    def __init__(self, seed: int, connector: str, frames, scene_events: list[dict]):
        import mujoco

        self.cell = CableCell(seed, connector)
        self.frames = frames.reset_index(drop=True)
        self.events = sorted(scene_events, key=lambda e: e["frame_index"])
        st = self.cell.station
        self._mj, self.model, self.data = mujoco, st._mj_model, st._mj_data
        self.renderer = mujoco.Renderer(self.model, PANEL_H, PANEL_W)

    def _scene_at(self, index: int) -> None:
        ids, m = self.cell.station._lay.ids, self.model
        event = None
        for e in self.events:
            if e["frame_index"] <= index:
                event = e
        if event is None:
            return
        for k in range(N_CABLES):
            pos, quat = event["vis"][str(k)] if str(k) in event["vis"] else event["vis"][k]
            m.body_pos[ids[f"body:vcable{k}"]] = pos
            m.body_quat[ids[f"body:vcable{k}"]] = quat
        m.body_pos[ids["body:tray"]], m.body_quat[ids["body:tray"]] = event["tray"]
        m.mat_rgba[ids["mat:plug_body"]] = event["plug_rgba"]

    def render(self, index: int, camera: str = "video_cam") -> np.ndarray:
        row = self.frames.iloc[int(index)]
        self._scene_at(int(row["frame_index"]))
        self.data.qpos[:] = np.asarray(row["qpos"], dtype=float)
        self._mj.mj_kinematics(self.model, self.data)
        self._mj.mj_camlight(self.model, self.data)
        self.renderer.update_scene(self.data, camera=camera)
        self.renderer.render()
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def close(self) -> None:
        self.renderer.close()
        self.cell.close()


def load(cycle_parquet: Path):
    import pandas as pd

    frames = pd.read_parquet(cycle_parquet)
    scene = json.loads(Path(str(cycle_parquet).replace(".parquet", ".scene.json")).read_text(encoding="utf-8"))
    return frames, scene
