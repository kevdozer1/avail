"""The ladder's relook rung, for ``held`` (2026-11-16, docs/07 step 2): a 2nd wrist pose, no contact.

The cable in the jaws is lifted ``LIFT_M`` and turned ``TURN_DEG`` about the tool's own axis, so the wrist camera sees it
against another background and in another light. The witness collects its sources there (the jaws' reading and the
cable's color), then ``end`` takes the arm back to where the rung found it. Nothing is touched: the cable stays in the
jaws as it was, and the motion is a position move in free space above the rack.
"""

from __future__ import annotations

import math

import numpy as np

from ..base import Pose
from .inject import ARM, _rot_z

LIFT_M = 0.03
TURN_DEG = 20.0
SPEED = 0.08


def begin(cell, item: int, step: str):
    st = cell.station
    home = st.robot_state().arms[ARM].tool_pose
    cell._rung_home = home
    rot = home.rotation() @ _rot_z(math.radians(TURN_DEG))
    target = Pose.from_matrix(np.asarray(home.pos, float) + np.array([0.0, 0.0, LIFT_M]), rot)
    st.move_to_pose(ARM, target, SPEED)
    st.wait(0.1)
    return None                                     # the step's result is the pick's own


def end(cell, item: int, step: str) -> None:
    home = getattr(cell, "_rung_home", None)
    if home is not None:
        cell.station.move_to_pose(ARM, home, SPEED)
        cell.station.wait(0.1)
    cell._rung_home = None
