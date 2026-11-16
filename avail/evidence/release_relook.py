"""vision.release_relook: after a release, the wrist camera moves to an inspection pose by the port and looks (S5c).

1 check, 1 file. It touches nothing. E3's released_stable rule let 31 bad cables through: the plug came up a
fraction of a millimeter to a few millimeters as the jaws opened and the lift caught it, and the camera read at the
end of the lift found the plug in only 58% of such drags. This source moves the holding arm, after the release
skill's own lift, to a fixed pose relative to the target port, chosen on dev seeds so the wrist camera faces the
plug's side from close by, takes 1 frame there and measures it with the same method as ``vision.gap``: the gap
between the plug's shoulder and the socket face, and the slant of the shoulder against the socket's top edge. Then it
moves back to where it was. The release skill's motion is unchanged: this runs only as a stage of the witness.

The inspection pose: the taught seated tool pose of the target port, raised ``RELOOK_UP_MM`` and moved
``RELOOK_BACK_MM`` along the board's y axis, the tool's orientation kept. Both were set on dev seeds 21,400 to
21,463 before any table seed ran (docs/07, 2026-11-13): of 38 poses, 60 mm up and 30 mm toward the robot found the
plug on 88% and 89% of good releases in 2 samples, reading their gap within 0.55 mm (95th percentile).

Features: those of ``vision.gap`` (``gap_mm_est``, ``slant_deg``, ``plug_found``, ``confidence`` and the rest), plus
``reached`` (1 when the arm reached the pose). Score: as ``vision.gap``'s.
"""

from __future__ import annotations

import numpy as np

from ..schemas import Claim, Cost, Evidence
from ..station.base import Pose
from .base import make_evidence, register
from .vision import MM, measure_gap

RELOOK_UP_MM = 60.0
RELOOK_BACK_MM = -30.0
SPEED_M_S = 0.08
TYPICAL_S = 2.0


def inspection_pose(nominal, port: int, up_mm: float = RELOOK_UP_MM, back_mm: float = RELOOK_BACK_MM) -> Pose:
    seated = nominal.seated_tool_pose(port)
    c, s = np.cos(nominal.board_yaw), np.sin(nominal.board_yaw)
    pos = np.asarray(seated.pos, float) + np.array([-s * back_mm * MM, c * back_mm * MM, up_mm * MM])
    return Pose(tuple(float(v) for v in pos), seated.quat)


class ReleaseRelook:
    source_id = "vision.release_relook"
    kind = "passive_vision"
    predicates = {"released_stable"}

    def estimate_cost(self, claim: Claim) -> Cost:
        return Cost(seconds=TYPICAL_S, usd=0.0, moved_robot=True)

    def collect(self, station, claim: Claim, params: dict) -> Evidence:
        nominal, port = params.get("nominal"), claim.predicate.target.get("port")
        if nominal is None or not isinstance(port, int):
            return make_evidence(self, claim, {})
        arm = params.get("arm", nominal.holding_arm)
        state = station.robot_state()
        t0, home = float(state.ts), state.arms[arm].tool_pose
        res = station.move_to_pose(arm, inspection_pose(nominal, port), SPEED_M_S)
        camera = f"{arm}_camera_d405"
        frame = station.observe([camera]).frames[camera]
        features = dict(measure_gap(frame, nominal, port))
        features["reached"] = 1.0 if res.ok else 0.0
        station.move_to_pose(arm, home, SPEED_M_S)
        if features.get("plug_found") and "gap_mm_est" in features:
            score = -features["gap_mm_est"]
        else:
            score = -nominal.full_depth_m / MM
        return make_evidence(self, claim, features, score=score, seconds=float(station.robot_state().ts) - t0,
                             moved_robot=True)


register(ReleaseRelook())
