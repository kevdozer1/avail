"""The cycle driver's records of a person (S5, after the 2nd review): 1 label per claim, and the options offered.

A toy task of 1 step on a fake cell, a witness that answers from a script and a scripted person, so
nothing here needs the simulator. 2 paths end in a takeover right after the person's own answer:

- a confirm answered no on the last try: the confirm labels the claim, and the takeover adds no 2nd label
- a choose answered takeover: the choose is no label, and the takeover's label is what the person found
"""

from __future__ import annotations

from types import SimpleNamespace

from avail.core.task import TaskSpec
from avail.fork.human import HumanAnswer, ScriptedHuman
from avail.schemas import Cost, Verdict, new_id
from avail.shift.driver import CycleDriver

TASK = TaskSpec.model_validate({
    "task": "toy", "title": "Place 1 widget in its slot", "items": {"kind": "widget", "count": 1},
    "steps": [{"name": "place", "claim": "placed"}],
    "predicates": {"placed": {"done_means": "The widget sits in its slot and stays there.",
                              "checks": ["passive.jaw_hold"],
                              "decide": {"kind": "rule", "accept_all": [{"feature": "passive.jaw_hold/in_band", "above": 0.5}]}}},
    "fork": {"max_attempts": 2, "ask": {"abstain": "confirm", "reject_last": "choose"}},
})


class Cell:
    variant, n_items = "v1", 1

    def __init__(self):
        self.view = SimpleNamespace(observe=lambda: (_ for _ in ()).throw(RuntimeError("no camera")))

    def target_of(self, item):
        return {}

    def run_skill(self, step, item, attempt, marks=None):
        return SimpleNamespace(ok=True, flag=True, seconds=1.0, failure=None)

    def claim_params(self, item, step, result):
        return {}

    def now(self):
        return 0.0

    def begin_assist(self):
        return 10

    def end_assist(self):
        return 20


class Witness:
    """Answers each try from a list of decisions."""

    def __init__(self, decisions):
        self.decisions = list(decisions)

    def verify(self, view, claim, params):
        verdict = Verdict(verdict_id=new_id("verdict"), claim_id=claim.claim_id, decision=self.decisions.pop(0),
                          stage_reached=1, witness_version="toy-1", cost_total=Cost.zero())
        return verdict, []


def person(confirm=None, choose=None):
    def script(request):
        if request.question == "confirm":
            return HumanAnswer(answer=confirm, truth_label={"yes": True, "no": False}[confirm], seconds_to_answer=20.0)
        if request.question == "choose":
            return HumanAnswer(answer=choose, seconds_to_answer=20.0)
        return HumanAnswer(answer="taken over", truth_label=True, correction_kind="takeover", correction={"ok": True},
                           seconds_to_answer=20.0, seconds_handling=45.0,
                           handoff={"ok": True, "robot_seconds": 3.0, "before": False})
    return ScriptedHuman(script, assistant_id="toy_person")


def run(decisions, human):
    events = []
    ledger = SimpleNamespace(append=lambda kind, payload: events.append((kind, payload)))
    driver = CycleDriver(TASK, Cell(), "avail_simop", witnesses={"placed": Witness(decisions)}, human=human,
                         ledger=ledger)
    log = driver.run()
    return log, [p for k, p in events if k == "intervention"], [p for k, p in events if k == "assist"]


def test_a_confirm_no_then_a_takeover_labels_the_claim_once():
    log, ivs, assists = run(["reject", "abstain"], person(confirm="no"))
    assert [(iv.question, iv.truth_label) for iv in ivs] == [("confirm", False), ("takeover", None)]
    assert ivs[0].claim_id == ivs[1].claim_id                         # the same claim: 1 label
    assert (log.assists, log.takeovers) == (1, 1) and ivs[1].seconds_waiting == 0.0
    assert log.items[0].declared_done                                 # the done gate took the sign-off after the handback
    assert assists[0].options == ["yes", "no", "cannot_tell"] and assists[1].options == []


def test_a_chosen_takeover_carries_the_only_label():
    log, ivs, assists = run(["reject", "reject"], person(choose="takeover"))
    assert [(iv.question, iv.truth_label) for iv in ivs] == [("choose", None), ("takeover", False)]
    assert (log.assists, log.takeovers) == (1, 1)
    assert assists[0].options == ["retry", "takeover", "abort"] and assists[0].answer in assists[0].options
