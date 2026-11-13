"""passive.release_record and the released_stable rule (S5, after the dev shift dev1).

On dev1 a stuck release lifted the plug out of its port with the jaws shut, the retry opened them
above the port, and the rule, which read only the jaws, accepted. The release skill now records its
gripper reading before each lift, and a release with a lift on shut jaws is a reject, whatever the
jaws read afterwards. No simulator here: the check reads only its params.
"""

from __future__ import annotations

from types import SimpleNamespace

from avail.core.task import load_task
from avail.evidence import base
from avail.schemas import Claim, ClaimContext, Predicate
from avail.witness.rules import RuleWitness

NOMINAL = SimpleNamespace(finger_travel_m=0.0375, expected_jaw_pos=lambda: 0.1)    # 3.75 mm expected, 37.5 mm open
OPENED = {"attempt": 1, "jaw_before_lift": 1.0, "lift_mm": 30.0}
SHUT = {"attempt": 0, "jaw_before_lift": 0.1, "lift_mm": 30.0}


def _claim() -> Claim:
    return Claim(claim_id="claim_1", run_id="run_x", cycle_id="cycle_1", step_id="0.release",
                 predicate=Predicate(name="released_stable", target={"item": 0}), raised_by="skill",
                 context=ClaimContext(task="cable_loop", stage="release", attempt=1))


def _features(attempts) -> dict:
    params = {"nominal": NOMINAL} if attempts is None else {"nominal": NOMINAL, "release": {"attempts": attempts}}
    return dict(base.get("passive.release_record").collect(None, _claim(), params).features)


def test_a_lift_on_shut_jaws_is_remembered_across_tries():
    assert _features([OPENED]) == {"attempts": 1.0, "lifted_closed": 0.0, "jaw_before_lift_mm": 37.5}
    assert _features([SHUT])["lifted_closed"] == 1.0
    after = _features([SHUT, OPENED])                         # the retry opened the jaws, the plug had gone up already
    assert after["lifted_closed"] == 1.0 and after["attempts"] == 2.0 and after["jaw_before_lift_mm"] == 37.5
    assert _features(None) == {} and _features([]) == {}      # no release record, no features


def test_the_released_stable_rule_rejects_a_release_that_lifted_the_plug():
    spec = load_task("cable_loop").predicates["released_stable"]
    assert "passive.release_record" in spec.checks
    rule = RuleWitness.from_predicate(spec, version="test")
    jaws_open = {"passive.jaw_hold/open": 1.0}
    assert rule.judge({**jaws_open, "passive.release_record/lifted_closed": 0.0}).decision == "accept"
    assert rule.judge({**jaws_open, "passive.release_record/lifted_closed": 1.0}).decision == "reject"
    assert rule.judge({"passive.jaw_hold/open": 0.0, "passive.release_record/lifted_closed": 0.0}).decision == "reject"
