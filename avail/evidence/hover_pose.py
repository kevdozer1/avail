"""passive.hover_pose: is the tool where the taught geometry says it should be, above the named port?

1 check, 1 file. Before an insertion the arm holds the part a little above its port. This check reads
the tool pose from the robot state and compares it with the taught pre-insert pose of the port the
claim names (``NominalGeometry.pre_insert_tool_pose``), and finds which taught port the tool is
nearest. It serves ``at_port``. It moves nothing and costs nothing.

Features: ``lateral_mm`` (tool against the taught pose, across the board), ``height_mm`` (along the
vertical), ``nearest_port``, ``port_match``, ``in_band``. Score: ``-lateral_mm`` when the nearest port
is the named one, else ``-(lateral_mm + 100)``.
"""

from __future__ import annotations

import numpy as np

from ..schemas import Claim, Cost, Evidence
from .base import make_evidence, register

CLEARANCE_M = 0.010        # the taught pre-insert height: the part's tip this far above the socket face
BAND_MM = 3.0              # within this across and along, the tool is at the taught pose


class HoverPose:
    source_id = "passive.hover_pose"
    kind = "passive_proprio"
    predicates = {"at_port"}

    def estimate_cost(self, claim: Claim) -> Cost:
        return Cost.zero()

    def collect(self, station, claim: Claim, params: dict) -> Evidence:
        nominal = params["nominal"]
        arm = params.get("arm", nominal.holding_arm)
        port = int(claim.predicate.target["port"])
        tool = station.robot_state().arms[arm].tool_pose.position()
        taught = nominal.pre_insert_tool_pose(port, CLEARANCE_M).position()
        lateral = float(np.linalg.norm(tool[:2] - taught[:2])) * 1000.0
        height = float(tool[2] - taught[2]) * 1000.0
        nearest, _ = nominal.nearest_port(tool[:2])
        match = int(nearest) == port
        features = {"lateral_mm": lateral, "height_mm": height, "nearest_port": float(nearest), "port_match": float(match),
                    "in_band": float(match and lateral <= BAND_MM and abs(height) <= BAND_MM)}
        score = -lateral if match else -(lateral + 100.0)
        return make_evidence(self, claim, features, score=score)


register(HoverPose())
