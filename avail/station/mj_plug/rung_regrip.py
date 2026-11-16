"""The ladder's regrip rung, for ``at_port`` (2026-11-16, docs/07 step 2).

Why a regrip. Of E3b's carries rejected on their last try, 93% of the retries with the operator's marks were rejected
again: the carry moves the cable as it sits in the jaws, so a cable cocked in them stays cocked. The witness reads the
cock as the tool sitting off its taught pose over the port. Re-centering the tool would hide that reading without
straightening anything, so the rung straightens the cable instead.

The motion: when the jaws read something in them, the cable is set back in the slot it came from (the stand-in's own
put-back, whose descent is a guarded move capped at 3 N), taken again from the slot straight (the stand-in's pick
motion), and carried over its port (the stand-in's carry motion, aimed as the stand-in aims). When the jaws read empty,
the cable is taken again where it lies, straight. These are scripted motions: no failure of the stand-in's profile is
drawn for them. The gripper reading decides which branch runs. The witness then checks the new carry like any other.
"""

from __future__ import annotations

import numpy as np

from ..base import Pose
from . import scene
from .inject import ARM, CLEARANCE_M

EMPTY_JAW = 0.05                                    # the pick's own check: the fingers stopped on something
OPEN_JAW = 0.6                                      # as probe.release_tug: over this the jaws read open, holding nothing
FREE_SPEED = 0.25                                   # the stand-in's own free-space speed (cable_cell.FREE_SPEED)


def begin(cell, item: int, step: str):
    from .cable_cell import SkillResult

    st = cell.station
    t0 = float(st._mj_data.time)
    jaw = float(st.robot_state().arms[ARM].gripper_pos)
    if EMPTY_JAW < jaw < OPEN_JAW and cell.physical is not None:      # holding: between empty and open
        cell._put_back(cell.physical)
        ok = bool(cell._pick_from_slot(item))
    else:
        rng = np.random.default_rng([cell.seed, cell._c, item, 4, 7171])
        cell._regrasp(item, rng, bad_p=0.0)
        ok = True
    port_pos, board_yaw = scene.port_pose_world(st._cfg, st._lay, cell.cables[item].port)
    # the stand-in carry's own path: a waypoint halfway to the port first, so the cable has settled in the jaws before
    # the aim is taken. Without it the 1st harm run's regrip left 1 of 268 good carries 1.58 mm off the port's axis
    # (the truth line is 1.5 mm), where the stand-in carry with no failure drawn misses 1 in 4,620 (docs/07)
    tool = st.robot_state().arms[ARM].tool_pose
    mid = Pose(((tool.pos[0] + port_pos[0]) / 2, (tool.pos[1] + port_pos[1]) / 2, max(tool.pos[2], port_pos[2] + 0.09)),
               tool.quat)
    cell._to_pose(mid, FREE_SPEED)
    cell.hover_xy[item] = (float(port_pos[0]), float(port_pos[1]))
    ok = bool(cell._carry_plug_to(np.array([port_pos[0], port_pos[1], port_pos[2]]), board_yaw, CLEARANCE_M)) and ok
    return SkillResult(step=step, item=item, ok=ok, flag=ok, seconds=float(st._mj_data.time) - t0, note="rung:regrip")


def end(cell, item: int, step: str) -> None:
    return None
