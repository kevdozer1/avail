"""passive.release_record: what the release skill's own record says about how it let go.

1 check, 1 file. For each of its attempts on a cable, the release skill records the gripper's own
reading just before it lifted away, and how far it lifted. Fingers that had not opened before the
lift took the plug up with them, out of its port or against its latch, whatever the jaws read later.
The dev shift of 2026-11-12 found that case: a stuck release lifted the plug out, the retry opened
the jaws above the port, and a rule that read only the jaws accepted the release.

It serves ``released_stable``. It moves nothing and costs nothing. A claim without a release record
gets no features.

Features: ``attempts`` (the release attempts so far on this cable), ``lifted_closed`` (1 when any of
them lifted with the jaws not open), ``jaw_before_lift_mm`` (the opening of 1 finger from closed
before the last lift). Score: ``-lifted_closed``.
"""

from __future__ import annotations

from ..schemas import Claim, Cost, Evidence
from .base import make_evidence, register
from .jaw_hold import OPEN_MM


class ReleaseRecord:
    source_id = "passive.release_record"
    kind = "passive_proprio"
    predicates = {"released_stable"}

    def estimate_cost(self, claim: Claim) -> Cost:
        return Cost.zero()

    def collect(self, station, claim: Claim, params: dict) -> Evidence:
        attempts = list((params.get("release") or {}).get("attempts") or [])
        if not attempts:
            return make_evidence(self, claim, {})
        nominal = params["nominal"]
        travel_mm = nominal.finger_travel_m * 1000.0
        expected = float(nominal.expected_jaw_pos()) * travel_mm
        lifted_closed = any(float(a.get("lift_mm", 0.0)) > 0.0 and float(a["jaw_before_lift"]) * travel_mm - expected <= OPEN_MM
                            for a in attempts)
        features = {"attempts": float(len(attempts)), "lifted_closed": float(lifted_closed),
                    "jaw_before_lift_mm": float(attempts[-1]["jaw_before_lift"]) * travel_mm}
        return make_evidence(self, claim, features, score=-float(lifted_closed))


register(ReleaseRecord())
