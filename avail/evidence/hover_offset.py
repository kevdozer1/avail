"""passive.hover_offset: the hover pose against the port where the robot has found the board to be (S5c fix).

1 check, 1 file. ``passive.hover_pose`` compares the tool with the taught pre-insert pose, which carries the
board's localization error (about 1 mm per axis, the same for every port of a board). The stand-in carry aims at the
port itself, so on a good claim that check measures the localization error, and a cable cocked in the jaws (which
also moves the tool, by about 4.7 mm) overlaps it: ``at_port`` abstained on 24.3% of its good claims (2026-11-13).

The robot measures the error itself: after an insertion its own record calls home, its tool sits over the real port,
and the tool's xy against the taught seated pose is the board's offset there. The cell keeps the median of those over
the cycle (``params['board_offset_mm']``). This check compares the tool with the taught pre-insert pose moved by that
offset. Before the cycle's 1st insertion home there is no estimate, and the check reports the plain offset.

It serves ``at_port``. It moves nothing and costs nothing.

Features: ``lateral_mm`` (tool against the corrected pose, across the board), ``offset_known`` (1 when the board's
offset was measured this cycle), ``offset_mm`` (the size of that offset).
"""

from __future__ import annotations

import numpy as np

from ..schemas import Claim, Cost, Evidence
from .base import make_evidence, register
from .hover_pose import CLEARANCE_M


class HoverOffset:
    source_id = "passive.hover_offset"
    kind = "passive_proprio"
    predicates = {"at_port"}

    def estimate_cost(self, claim: Claim) -> Cost:
        return Cost.zero()

    def collect(self, station, claim: Claim, params: dict) -> Evidence:
        nominal = params["nominal"]
        arm = params.get("arm", nominal.holding_arm)
        port = int(claim.predicate.target["port"])
        tool = station.robot_state().arms[arm].tool_pose.position()
        taught = np.asarray(nominal.pre_insert_tool_pose(port, CLEARANCE_M).position()[:2], float)
        offset = params.get("board_offset_mm")
        known = offset is not None
        shift = np.asarray(offset, float) / 1000.0 if known else np.zeros(2)
        lateral = float(np.linalg.norm(np.asarray(tool[:2]) - (taught + shift))) * 1000.0
        return make_evidence(self, claim, {"lateral_mm": lateral, "offset_known": float(known),
                                           "offset_mm": float(np.linalg.norm(shift)) * 1000.0})


register(HoverOffset())
