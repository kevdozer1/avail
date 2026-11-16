"""passive.release_lift: how the release skill's own lift and opening went, from its record (S5c fix).

1 check, 1 file. The release skill opens the jaws, then lifts 2 cm away. Its record now also holds
the wrist's force along the port axis on every control tick of that lift, and the change in the wrist's
force as the jaws opened. The motion is unchanged. E3 let 31 bad cables through at the release
(2026-11-13): the jaws read open, no try lifted with them shut, and yet the plug came up. On dev seeds
the opening tipped the plug a few degrees and the lift then caught it and dragged it up, by 0.4 mm to
6 mm. A plug that rides up with the fingers pulls on the wrist during the lift.

It serves ``released_stable``. It moves nothing and costs nothing. A claim without a release record,
or one written before the record had these fields, gets no features.

Features, from the last attempt on this cable:

- ``lift_peak_n``, ``lift_min_n``, ``lift_early_mean_n``  the wrist's force along z during the lift against its
  value before the jaws opened: its largest and smallest value, and its mean over the 1st quarter
- ``open_above_seat_mm``                                 where the tool was when the jaws opened, above its
  taught pose with the plug seated in the target port
- ``open_force_dz_n``, ``open_force_dxy_n``              the change in the wrist's force as the jaws opened,
  along the port axis and across it

No score of its own: the thresholds are a calibrated witness's, fit and calibrated on the cable loop's
release claims (``avail/station/mj_plug/check_zoo.py``).
"""

from __future__ import annotations

from ..schemas import Claim, Cost, Evidence
from .base import make_evidence, register

FIELDS = ("lift_peak_n", "lift_min_n", "lift_early_mean_n", "open_above_seat_mm", "open_force_dz_n", "open_force_dxy_n")


class ReleaseLift:
    source_id = "passive.release_lift"
    kind = "passive_proprio"
    predicates = {"released_stable"}

    def estimate_cost(self, claim: Claim) -> Cost:
        return Cost.zero()

    def collect(self, station, claim: Claim, params: dict) -> Evidence:
        attempts = list((params.get("release") or {}).get("attempts") or [])
        if not attempts:
            return make_evidence(self, claim, {})
        last = attempts[-1]
        features = {name: float(last[name]) for name in FIELDS if name in last}
        return make_evidence(self, claim, features)


register(ReleaseLift())
