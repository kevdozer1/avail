"""Light tests for the visuals' counting (avail/analysis/visuals.py): the drawings themselves are looked at."""

from __future__ import annotations

from avail.analysis.film import CHECKING, CODE, OPERATOR, STOPPED
from avail.analysis.visuals import cycle_class, fates, replay_hour


def _end(claims, steps, items, cycle="c1"):
    return {"cycle_id": cycle, "claims": claims, "steps": steps, "items": items}


def test_a_cycle_is_bad_undone_or_good():
    assert cycle_class({"uncaught": 1, "items": [{"declared": True}]}) == "bad"
    assert cycle_class({"uncaught": 0, "items": [{"declared": True}, {"declared": False}]}) == "undone"
    assert cycle_class({"uncaught": 0, "items": [{"declared": True}]}) == "good"


def test_each_failed_step_gets_1_fate():
    claims = [
        {"item": 0, "step": "pick", "attempt": 0, "rung": None, "true": False, "decision": "reject"},
        {"item": 0, "step": "pick", "attempt": 1, "rung": None, "true": True, "decision": "accept"},
        {"item": 0, "step": "carry", "attempt": 0, "rung": None, "true": False, "decision": "reject"},
        {"item": 0, "step": "insert", "attempt": 0, "rung": None, "true": False, "decision": "reject"},
        {"item": 0, "step": "release", "attempt": 0, "rung": None, "true": True, "decision": "accept"},
        {"item": 1, "step": "release", "attempt": 0, "rung": None, "true": False, "decision": "reject"},
        {"item": 2, "step": "release", "attempt": 0, "rung": None, "true": False, "decision": "accept"},
    ]
    steps = [
        {"item": 0, "step": "pick", "declared": True, "true": True, "by": "verdict"},
        {"item": 0, "step": "carry", "declared": True, "true": True, "by": "verdict"},
        {"item": 0, "step": "insert", "declared": True, "true": True, "by": "rung:reinsert"},
        {"item": 0, "step": "release", "declared": True, "true": True, "by": "verdict"},
        {"item": 1, "step": "release", "declared": False, "true": False, "by": None},
        {"item": 2, "step": "release", "declared": True, "true": False, "by": "verdict"},
    ]
    items = [{"item": 0, "uncaught": False}, {"item": 1, "uncaught": False}, {"item": 2, "uncaught": True}]
    ivs = [{"cycle_id": "c1", "item": "0", "step": "carry"}]
    out = fates([_end(claims, steps, items)], ivs)
    assert out["total"] == 5
    assert out["fates"] == {"retry": 1, "operator": 1, "rung": 1, "undone": 1, "escaped": 1}


def test_1_operator_serves_calls_in_turn_and_a_robot_waits_its_turn():
    robots = [[{"segments": [(CODE, 10.0, 0), (OPERATOR, 30.0, 1), (CHECKING, 5.0, 0)]}],
              [{"segments": [(CODE, 20.0, 0), (OPERATOR, 30.0, 1)]}]]
    hour = replay_hour(robots, horizon=100.0)
    assert hour["operator"] == [(10.0, 40.0, 0), (40.0, 70.0, 1)]
    assert (20.0, 40.0, STOPPED) in hour["lanes"][1]
    assert hour["busy_s"] == 60.0 and hour["robot_wait_s"] == 20.0 and hour["calls"] == 2
