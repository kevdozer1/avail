"""Teach the cable loop's palette: each cable's color as the wrist camera sees it at the tool point. Dev seeds only.

    uv run python -m avail.station.mj_plug.cable_teach --seeds 23900:23920

A real cell is taught each part's appearance once at setup, as it is taught its poses. Here each
cable is picked cleanly from its slot on a few dev seeds, and the reading ``vision.cable_tag`` takes
(the median color at the projected tool point, ``avail.evidence.cable_tag.tool_color``) is recorded
twice: in the jaws over the rack, where ``held`` is claimed, and over its port, where ``at_port`` is.
The palette is the median of all of them per cable, written to ``cable_palette.json`` beside this
file with the seeds and the spread, so the run that made it can be read back.

Why: on the dev shift dev1 (2026-11-12) the check found no cable color in the jaws often enough that
``held`` abstained on about 1 pick in 5, and a person was called for each. The picks taught here
showed why: a seed's lighting scales the rendered color (the white cable read 0.4 to 0.6 from white
in RGB), while its direction in RGB stays within about 1 degree of its own. The check now compares
directions (``avail/evidence/cable_tag.py``).
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np

PALETTE_FILE = Path(__file__).with_name("cable_palette.json")


def _reading(cell) -> list[float] | None:
    from ...evidence.cable_tag import tool_color
    from .cable_cell import ARM

    st = cell.station
    frame = st.observe([f"{ARM}_camera_d405"]).frames[f"{ARM}_camera_d405"]
    color = tool_color(frame, st.robot_state().arms[ARM].tool_pose.position())
    return None if color is None else [float(c) for c in color]


def teach(seeds: list[int], variant: str = "latch") -> dict:
    from ...seeds import guard
    from . import scene
    from .cable_cell import CLEARANCE_M, N_CABLES, CableCell
    from .cable_scene import CABLE_NAMES

    guard(seeds, purpose="teach the cable loop's palette")
    readings: dict[int, dict[str, list]] = {k: {"rack": [], "port": []} for k in range(N_CABLES)}
    for seed in seeds:
        cell = CableCell(seed, variant)
        try:
            st = cell.station
            for k in range(N_CABLES):
                if not cell._pick_from_slot(k):
                    continue
                at_rack = _reading(cell)
                port_pos, yaw = scene.port_pose_world(st._cfg, st._lay, cell.cables[k].port)
                cell._carry_plug_to(port_pos, yaw, CLEARANCE_M)
                at_port = _reading(cell)
                cell._put_back(k)
                if at_rack is not None:
                    readings[k]["rack"].append(at_rack)
                if at_port is not None:
                    readings[k]["port"].append(at_port)
        finally:
            cell.close()
    colors, spread = [], []
    for k in range(N_CABLES):
        every = np.asarray(readings[k]["rack"] + readings[k]["port"], float)
        center = np.median(every, axis=0)
        colors.append([round(float(c), 4) for c in center])
        spread.append(round(float(np.percentile(np.linalg.norm(every - center, axis=1), 95)), 4))
    own = [np.linalg.norm(np.asarray(readings[k][where], float) - np.asarray(colors[k]), axis=1)
           for k in range(N_CABLES) for where in ("rack", "port") if readings[k][where]]
    nearest_other = min(float(np.linalg.norm(np.asarray(colors[a]) - np.asarray(colors[b])))
                        for a in range(N_CABLES) for b in range(N_CABLES) if a < b)
    out = {"taught_on": f"dev seeds {seeds[0]} to {seeds[-1]}, {variant}", "date": date.today().isoformat(),
           "names": list(CABLE_NAMES), "colors": colors, "spread_p95": spread,
           "n": {CABLE_NAMES[k]: {w: len(readings[k][w]) for w in ("rack", "port")} for k in range(N_CABLES)},
           "worst_own_distance": round(float(max(d.max() for d in own)), 4),
           "nearest_other_cable": round(nearest_other, 4),
           "readings": {CABLE_NAMES[k]: readings[k] for k in range(N_CABLES)}}
    return out


@lru_cache(maxsize=1)
def _taught() -> tuple[tuple[float, ...], ...]:
    if not PALETTE_FILE.is_file():
        raise FileNotFoundError(f"{PALETTE_FILE} is missing. Teach it: uv run python -m avail.station.mj_plug.cable_teach")
    return tuple(tuple(map(float, c)) for c in json.loads(PALETTE_FILE.read_text(encoding="utf-8"))["colors"])


def load_palette() -> list[list[float]]:
    """The taught colors, cable by cable. The cell refuses to run without them, so a check never reads
    material colors by mistake."""
    return [list(c) for c in _taught()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m avail.station.mj_plug.cable_teach")
    parser.add_argument("--seeds", default="23900:23920", help="low:high, high excluded. Shift dev seeds. The committed palette was taught on 23900:23920")
    parser.add_argument("--variant", default="latch")
    args = parser.parse_args(argv)
    lo, hi = (int(x) for x in args.seeds.split(":"))
    out = teach(list(range(lo, hi)), args.variant)
    PALETTE_FILE.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in out.items() if k != "readings"}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
