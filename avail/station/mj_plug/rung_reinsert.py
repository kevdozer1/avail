"""The ladder's reinsert rung, for ``latched`` (2026-11-16, docs/07 step 2).

Why not a push. The insertion witness already pushes (``probe.push_seat``) and pulls (``probe.tug``) on every claim it
does not settle passively, so a short plug a push can seat is seated before the last reject. What is left after 2 tries
is mostly a plug cocked in the jaws (a lateral start turns it in them, and the stand-in's retry keeps it as it is) and a
blocked latch, which only a person clears. A cocked plug needs to be taken again straight.

The motion: when the jaws read the plug, it is drawn up 20 mm with the stand-in's pull-out force cap (out of the port if
it is in one). When they read empty, the plug is taken again where it is, straight, which draws it out of a port the same
way. Then it is set back in its slot, taken again straight, carried over its port, and inserted by the stand-in's
force-guarded descent under the push force cap. No failure of the stand-in's profile is drawn for these motions. A
blocked latch stays blocked. The witness then checks the new insertion like any other, with its own push and pull.
"""

from __future__ import annotations

import numpy as np

from . import scene
from .inject import ARM, CLEARANCE_M

EMPTY_JAW = 0.05
OPEN_JAW = 0.6                                      # as probe.release_tug: over this the jaws read open, holding nothing
PULL_OUT_M = 0.02


def begin(cell, item: int, step: str):
    from .cable_cell import SkillResult

    st = cell.station
    t0 = float(st._mj_data.time)
    rng = np.random.default_rng([cell.seed, cell._c, item, 4, 7272])
    jaw = float(st.robot_state().arms[ARM].gripper_pos)
    # the jaws hold the plug when they read between empty and open. Open jaws (an insertion that let go early) take the
    # plug again where it is, which closes on it in the port and draws it out under the pull-out cap (review, 2026-11-16)
    if EMPTY_JAW < jaw < OPEN_JAW and cell.physical is not None:
        st.move_relative(ARM, [0.0, 0.0, PULL_OUT_M], "base", 0.01, cell.pull_out_force_n(), PULL_OUT_M + 0.005)
    else:
        cell._regrasp(item, rng, bad_p=0.0)
    if cell.physical is not None:
        cell._put_back(cell.physical)
    ok = bool(cell._pick_from_slot(item))
    port_pos, board_yaw = scene.port_pose_world(st._cfg, st._lay, cell.cables[item].port)
    cell.hover_xy[item] = (float(port_pos[0]), float(port_pos[1]))
    ok = bool(cell._carry_plug_to(np.array([port_pos[0], port_pos[1], port_pos[2]]), board_yaw, CLEARANCE_M)) and ok
    record, _, flag = cell._insert(item, rng, 0, no_failure=True)
    return SkillResult(step=step, item=item, ok=bool(ok and record is not None), flag=bool(flag),
                       seconds=float(st._mj_data.time) - t0, note="rung:reinsert", insert=record)


def end(cell, item: int, step: str) -> None:
    return None
