"""passive.jaw_hold: what the gripper's own reading says about a part in its jaws.

1 check, 1 file. It reads the finger opening from the robot state and compares it with the opening
the taught part width gives (``NominalGeometry.expected_jaw_pos``). Fingers that stopped on a body of
the part's width hold the part. Fingers that closed well past it hold nothing. Fingers far wider than
the part are open. It serves ``held`` and ``at_port`` (the part is in the jaws) and ``released_stable``
(the jaws let go). It moves nothing and costs nothing.

Features: ``jaw_mm`` (opening of 1 finger from closed), ``expected_mm``, ``error_mm``, ``in_band``,
``closed_empty``, ``open``. Score: ``-abs(error_mm)`` for a claim that the part is held, and
``jaw_mm - expected_mm`` for a claim that it was let go.
"""

from __future__ import annotations

from ..schemas import Claim, Cost, Evidence
from .base import make_evidence, register

IN_BAND_MM = 1.0           # within this of the taught opening, the fingers stopped on the part
EMPTY_MM = 2.0             # closed this far past it, there is nothing between them
OPEN_MM = 5.0              # this far wider than the part, the jaws are open


class JawHold:
    source_id = "passive.jaw_hold"
    kind = "passive_proprio"
    predicates = {"held", "at_port", "released_stable"}

    def estimate_cost(self, claim: Claim) -> Cost:
        return Cost.zero()

    def collect(self, station, claim: Claim, params: dict) -> Evidence:
        nominal = params["nominal"]
        arm = params.get("arm", nominal.holding_arm)
        state = station.robot_state().arms[arm]
        travel_mm = nominal.finger_travel_m * 1000.0
        jaw = float(state.gripper_pos) * travel_mm
        expected = float(nominal.expected_jaw_pos()) * travel_mm
        error = jaw - expected
        features = {"jaw_mm": jaw, "expected_mm": expected, "error_mm": error,
                    "in_band": float(abs(error) <= IN_BAND_MM), "closed_empty": float(error < -EMPTY_MM),
                    "open": float(error > OPEN_MM)}
        score = error if claim.predicate.name == "released_stable" else -abs(error)
        return make_evidence(self, claim, features, score=score)


register(JawHold())
