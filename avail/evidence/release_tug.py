"""probe.release_tug: after a release, re-close the jaws on the plug and pull it lightly, below the latch's hold (S5c).

1 check, 1 file. The release witness on passive signals and 2 camera reads could not separate a plug that came off
its latch as the jaws opened from a seated one (2026-11-14): in the calibration table 72 bad releases read as seated
to a camera that found the plug, and 111 more were found by neither camera. A plug off its latch sits in its port and
looks seated. Only a pull tells: a latched plug holds it, a plug off its latch slides out.

Motion: from where the release skill's lift left the tool, straight down to the taught seated grasp height of the
target port, keeping the tool's xy (the release opened the jaws there), close the jaws, then ``probe.tug``'s own pull
with its config caps (``probes.tug``: 0.6 of the latch's seated retention, so a latched plug holds), which pushes a plug
that slipped back home and reports whether it clicked. Then the jaws open and the tool lifts as the release skill
lifts (2 cm at 30 mm/s). That last let-go can itself drag a plug, as the release can, and the tables' truth after this
probe counts it. When the gripper did not read open before the probe (the jaws stuck shut), it does not move and
reports ``jaws_open = 0``.

Features: ``probe.tug``'s (``disp_mm``, ``slipped``, ``stopped_by_force``, ``reseat_clicked`` and the rest), plus
``jaws_open``, ``regrip_holding`` (1 when the jaws closed at the plug's width) and ``descend_mm`` (how far the tool came
down to the grasp height).

Added 2026-11-16 (docs/07, the night's step 1): ``force_frac``, the force reading at the end of the pull over the pull's
force cap, and ``held_at_cap``, 1 when the pull stopped on force with that reading at ``HELD_AT_CAP_FRAC`` of the cap or
more. A guarded move also stops "on force" when the commanded force sits at the cap and the tool does not move, and
then the reading need not have risen at all. Both of E3b's uncaught cables passed this check that way (readings of 0.36
and -12.1 N at the stop, against 14.6 to 15.1 N for 98% of pulls), and the check tables hold 26 more such stops, 19 of
them bad. A stall with a low reading is not a latch holding the pull. Both features are functions of what the pull
already reports, so ``derive`` also adds them to a stored table's columns.
"""

from __future__ import annotations

import numpy as np

from ..config import load_experiment
from ..schemas import Claim, Cost, Evidence
from ..station.base import Pose
from .base import make_evidence, register
from .probes import TUG, _arm, _clock, _connector, _holding

OPEN_JAW = 0.6
LIFT_M, LIFT_SPEED = 0.02, 0.03          # the release skill's own lift
TYPICAL_S = 4.0
HELD_AT_CAP_FRAC = 0.8                   # in the gap between a held pull (14.6 N or more, 97% of the cap) and a stall (7.3 N or less)


def derived(stopped_by_force, force_at_stop_n, cap_n: float) -> dict[str, float]:
    """``force_frac`` and ``held_at_cap`` from what the pull reports."""
    frac = float(force_at_stop_n) / float(cap_n)
    return {"force_frac": frac, "held_at_cap": 1.0 if float(stopped_by_force) == 1.0 and frac >= HELD_AT_CAP_FRAC else 0.0}


def derive(table, connector: str, prefix: str = "B/probe.release_tug/"):
    """The 2 derived features as columns of a stored features table, where the pull ran. A row with no pull keeps NaN."""
    cap = load_experiment().tug_force_cap_n(connector)
    bf, f = f"{prefix}stopped_by_force", f"{prefix}force_at_stop_n"
    if bf not in table.columns or f not in table.columns or f"{prefix}held_at_cap" in table.columns:
        return table
    table = table.copy()
    ran = table[bf].notna() & table[f].notna()
    frac = np.where(ran, table[f].astype(float) / cap, np.nan)
    held = np.where(ran, ((table[bf].astype(float) == 1.0) & (frac >= HELD_AT_CAP_FRAC)).astype(float), np.nan)
    table[f"{prefix}force_frac"] = frac
    table[f"{prefix}held_at_cap"] = held
    return table


class ReleaseTug:
    source_id = "probe.release_tug"
    kind = "active_probe"
    predicates = {"released_stable"}

    def estimate_cost(self, claim: Claim) -> Cost:
        return Cost(seconds=TYPICAL_S, usd=0.0, moved_robot=True)

    def collect(self, station, claim: Claim, params: dict) -> Evidence:
        nominal, port = params.get("nominal"), claim.predicate.target.get("port")
        if nominal is None or not isinstance(port, int):
            return make_evidence(self, claim, {})
        arm, connector, cfg = _arm(params), _connector(params), load_experiment()
        state = station.robot_state()
        t0 = float(state.ts)
        if float(state.arms[arm].gripper_pos) < OPEN_JAW:
            return make_evidence(self, claim, {"jaws_open": 0.0})
        tool = state.arms[arm].tool_pose
        seated = nominal.seated_tool_pose(port)
        grasp = Pose((float(tool.pos[0]), float(tool.pos[1]), float(seated.pos[2])), tool.quat)
        station.move_to_pose(arm, grasp, 0.03)
        station.set_gripper(arm, 0.0)
        now = station.robot_state()
        features = {"jaws_open": 1.0, "descend_mm": (float(tool.pos[2]) - float(seated.pos[2])) * 1000.0}
        holding = _holding(now.arms[arm], params)
        features["regrip_holding"] = 1.0 if holding else 0.0
        score, side = None, "none"
        if holding:
            score, side = TUG._probe(station, arm, connector, cfg, now, features)
            if "stopped_by_force" in features and "force_at_stop_n" in features:
                features.update(derived(features["stopped_by_force"], features["force_at_stop_n"],
                                        cfg.tug_force_cap_n(connector)))
        station.set_gripper(arm, 1.0)
        pose = station.robot_state().arms[arm].tool_pose
        station.move_to_pose(arm, Pose((pose.pos[0], pose.pos[1], pose.pos[2] + LIFT_M), pose.quat), LIFT_SPEED)
        return make_evidence(self, claim, features, score=score, seconds=_clock(station) - t0, moved_robot=True,
                             side_effect=side)


register(ReleaseTug())
