"""The night of 2026-11-16 (docs/07, steps 2 and 3): the crossing rule, and the escalation ladder before a call.

No simulator. A fake cell and fake witnesses stand in, so what is tested is the rule and the driver's flow:
a rung never decides, only the witness accepts, and the operator is asked once the rungs are spent."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from avail.stats import decide, decide_many


# ----------------------------------------------------------------------------- the crossing rule
def test_crossed_thresholds_abstain_by_default_and_accept_above_tau_accept_when_asked():
    # tau_reject 9.0 above tau_accept -1.0: crossed
    assert decide(5.0, 9.0, -1.0) == "abstain"                      # the rule as E3 and E3b ran it
    assert decide(5.0, 9.0, -1.0, "accept") == "accept"             # above tau_accept: the false-accept bound holds
    assert decide(-2.0, 9.0, -1.0, "accept") == "reject"            # below it, and so below tau_reject too
    assert decide(-1.0, 9.0, -1.0, "accept") == "abstain"           # a tie abstains
    assert decide(9.5, 9.0, -1.0, "accept") == "accept"
    assert decide(float("nan"), 9.0, -1.0, "accept") == "abstain"


def test_thresholds_that_do_not_cross_decide_the_same_under_both_rules():
    for s in (-3.0, 0.0, 2.0, 4.5, 7.0):
        assert decide(s, 1.0, 4.0) == decide(s, 1.0, 4.0, "accept")
    s = np.array([-3.0, 0.0, 2.0, 4.5, 7.0])
    assert list(decide_many(s, 1.0, 4.0)) == list(decide_many(s, 1.0, 4.0, "accept"))
    assert list(decide_many(s, 3.0, 0.5, "accept")) == ["reject", "reject", "accept", "accept", "accept"]
    with pytest.raises(ValueError):
        decide(1.0, 0.0, 2.0, "maybe")


def test_a_cascade_keeps_its_crossing_rule_through_a_stored_file():
    from avail.evidence.fused import fit_scorer
    from avail.witness.cascade import Cascade, FittedStage, StageDef

    import pandas as pd

    frame = pd.DataFrame({"a/x": [0.0, 1.0, 0.0, 1.0]})
    scorer = fit_scorer("s", frame, ["a/x"], np.array([False, True, False, True]))
    stage = FittedStage("passive", ("a",), scorer, 2.0, -1.0, 0.01, 0.05, 100, 100)
    c = Cascade("held", "latch", 0.01, 0.05, [stage], station_id="mj_plug",
                stage_defs=(StageDef("passive", ("a",), "A", "y0"),), crossing="accept")
    back = Cascade.from_dict(c.to_dict())
    assert back.crossing == "accept"
    old = c.to_dict()
    old.pop("crossing")
    assert Cascade.from_dict(old).crossing == "abstain"            # a file written before the rule existed


# ----------------------------------------------------------------------------- the ladder in the driver
class _Verdict(SimpleNamespace):
    pass


class _Witness:
    """Abstains until the cell says a rung has run, then accepts."""

    def __init__(self, cell):
        self.cell = cell

    def verify(self, view, claim, params):
        from avail.evidence.base import make_evidence
        from avail.schemas import Cost, Verdict, new_id

        ev = make_evidence(SimpleNamespace(source_id="passive.jaw_hold", kind="passive_proprio"), claim, {"in_band": 1.0})
        decision = "accept" if self.cell.fixed else "abstain"
        return Verdict(verdict_id=new_id("verdict"), claim_id=claim.claim_id, decision=decision, score=1.0,
                       stage_reached=1, witness_version="t", evidence_ids=[ev.evidence_id],
                       cost_total=Cost(seconds=0.0, usd=0.0, moved_robot=False)), [ev]


class _Cell:
    variant, n_items = "latch", 1

    def __init__(self, rung_fixes: bool):
        self.rung_fixes, self.fixed, self.t, self.rungs = rung_fixes, False, 0.0, []
        self.view = SimpleNamespace(observe=lambda: SimpleNamespace(frames={}))

    def target_of(self, item):
        return {"port": 3}

    def run_skill(self, step, item, attempt=0, marks=None):
        self.t += 1.0
        return SimpleNamespace(ok=True, flag=True, seconds=1.0, failure=None)

    def claim_params(self, item, step, res=None):
        return {}

    def now(self):
        return self.t

    def begin_assist(self):
        return 0

    def end_assist(self):
        return 1

    def rung_begin(self, name, item, step):
        self.rungs.append(name)
        self.t += 5.0
        if self.rung_fixes:
            self.fixed = True
        return None

    def rung_end(self, name, item, step):
        pass


class _Human:
    assistant_id = "test"

    def __init__(self):
        self.asked = []

    def ask(self, request):
        self.asked.append(request.question)
        return SimpleNamespace(answer="yes", truth_label=True, seconds_to_answer=10.0, seconds_handling=0.0,
                               correction_kind=None, correction=None, handoff=None)


def _task():
    from avail.core.task import TaskSpec

    return TaskSpec.model_validate({
        "task": "t", "title": "a test task", "items": {"kind": "thing", "count": 1},
        "steps": [{"name": "pick", "claim": "held"}],
        "predicates": {"held": {"done_means": "The gripper holds the named thing in its jaws.", "checks": ["x.y"],
                                "decide": {"kind": "witness", "witness": "{variant}"}}},
        "fork": {"max_attempts": 2, "ask": {"abstain": "confirm"}, "ladder": {"held": ["relook"]}}})


def _run(rung_fixes: bool):
    from avail.shift.driver import CycleDriver

    cell, human, events = _Cell(rung_fixes), _Human(), []

    class Ledger:
        def append(self, kind, payload):
            events.append((kind, payload))

    d = CycleDriver(_task(), cell, "avail_simop", witnesses={"held": _Witness(cell)}, human=human, ledger=Ledger(),
                    evaluator=lambda item, predicate: True)
    log = d.run()
    return d, log, cell, human, events


def test_a_rung_whose_new_claim_the_witness_accepts_needs_no_call():
    d, log, cell, human, events = _run(rung_fixes=True)
    assert cell.rungs == ["relook"] and human.asked == [] and log.assists == 0
    assert log.items[0].declared_done
    assert [a["decision"] for a in d.claim_audit] == ["abstain", "accept"] and d.claim_audit[1]["rung"] == "relook"
    assert d.rung_audit[0]["after"] == "abstain" and d.rung_audit[0]["decision"] == "accept"
    assert any(k == "tool_call" and p.get("tool") == "rung:relook" for k, p in events)
    assert log.motion_s >= 5.0                                     # the rung's own time is the robot's


def test_when_the_rungs_are_spent_the_operator_is_asked_once():
    d, log, cell, human, events = _run(rung_fixes=False)
    assert cell.rungs == ["relook"] and human.asked == ["confirm"] and log.assists == 1
    assert log.items[0].declared_done                              # the operator signed it off


class _ScriptedWitness:
    """Decides from a list, 1 decision per verify, then accepts."""

    def __init__(self, decisions):
        self.decisions = list(decisions)

    def verify(self, view, claim, params):
        from avail.evidence.base import make_evidence
        from avail.schemas import Cost, Verdict, new_id

        ev = make_evidence(SimpleNamespace(source_id="passive.jaw_hold", kind="passive_proprio"), claim, {"in_band": 1.0})
        decision = self.decisions.pop(0) if self.decisions else "accept"
        return Verdict(verdict_id=new_id("verdict"), claim_id=claim.claim_id, decision=decision, score=1.0,
                       stage_reached=1, witness_version="t", evidence_ids=[ev.evidence_id],
                       cost_total=Cost(seconds=0.0, usd=0.0, moved_robot=False)), [ev]


def _run_scripted(decisions, fork):
    from avail.core.task import TaskSpec
    from avail.shift.driver import CycleDriver

    spec = _task().model_dump()
    spec["fork"] = fork
    task = TaskSpec.model_validate(spec)
    cell, human = _Cell(rung_fixes=False), _Human()
    d = CycleDriver(task, cell, "avail_simop", witnesses={"held": _ScriptedWitness(decisions)}, human=human,
                    ledger=None, evaluator=lambda item, predicate: True)
    return d, d.run(), cell, human


def test_an_abstain_whose_rung_claim_is_rejected_gets_its_retry():
    # reviewer's case (2026-11-16): try 1 abstains, the rung's claim is rejected with a try left, so the step is tried
    # again (and accepted), with no call and no cable left unresolved
    d, log, cell, human = _run_scripted(["abstain", "reject"], {"max_attempts": 2, "ladder": {"held": ["relook"]}})
    assert cell.rungs == ["relook"] and human.asked == [] and log.assists == 0
    assert log.items[0].declared_done and log.items[0].unresolved is None
    assert [a["attempt"] for a in d.claim_audit] == [0, 0, 1]


def test_ladder_after_keeps_a_rung_off_a_last_reject():
    fork = {"max_attempts": 2, "ask": {"reject_last": "choose"}, "ladder": {"held": ["relook"]},
            "ladder_after": {"held": ["abstain"]}}
    d, log, cell, human = _run_scripted(["reject", "reject"], fork)
    assert cell.rungs == [] and human.asked[0] == "choose"


def test_a_task_file_ladder_must_name_known_predicates_and_rung_lists():
    from avail.core.task import TaskSpec

    good = _task().model_dump()
    bad = dict(good, fork={"max_attempts": 2, "ladder": {"nope": ["relook"]}})
    with pytest.raises(Exception):
        TaskSpec.model_validate(bad)
    with pytest.raises(Exception):
        TaskSpec.model_validate(dict(good, fork={"max_attempts": 2, "ladder": {"held": "relook"}}))
