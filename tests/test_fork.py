"""The fork (avail/fork): the default policy, what a person is shown and answers, and takeover bookkeeping.

The policy rules of docs/05: a reject with tries left is retried, the last reject and an abstain go to
a person when one is there, and with nobody there the step is aborted and stays unresolved. Nothing
here ever offers or picks accept. tests/test_done_gate.py holds the done gate and the static rule.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from avail.fork import (
    ASSISTED, AUTONOMOUS, DEFAULT_QUESTIONS, ForkPlan, Handback, HandoffLog, Human, HumanAnswer, HumanRequest,
    ScriptedHuman, Takeover, TeleopSource, answer_for, check_answer, choose, decide, fork_record, to_assist,
)
from avail.paths import REPO_ROOT
from avail.schemas import Assist, Claim, ClaimContext, Fork, Intervention, Predicate, kinds

CABLE_ASK = {"abstain": "confirm", "reject_last": "choose", "at_port_reject": "guide_marks"}


def claim(name: str = "seated") -> Claim:
    return Claim(claim_id="claim_1", run_id="run_t", cycle_id="cycle_1", step_id="insert",
                 predicate=Predicate(name=name, target={"item": 2, "port": 5}), raised_by="skill",
                 context=ClaimContext(task="cable_loop", stage="insert", attempt=0))


# ------------------------------------------------------------------ the policy
def test_a_reject_with_tries_left_is_retried_by_the_policy():
    plan = decide("reject", 0, 2, True, CABLE_ASK)
    assert (plan.chosen, plan.question, plan.options) == ("retry", None, ["retry", "ask_human", "abort"])
    assert plan.reason == "reject on try 1 of 2, 1 try left: retry"
    alone = decide("reject", 0, 2, False, CABLE_ASK)
    assert (alone.chosen, alone.options) == ("retry", ["retry", "abort"])      # nobody to ask is not on the menu


def test_the_policy_retries_then_asks_a_person():
    picks = [decide("reject", attempt, 3, True, CABLE_ASK) for attempt in range(3)]
    assert [plan.chosen for plan in picks] == ["retry", "retry", "ask_human"]
    assert picks[1].reason == "reject on try 2 of 3, 1 try left: retry"
    assert (picks[2].question, picks[2].options) == ("choose", ["ask_human", "abort"])
    assert picks[2].reason == "reject on the last of 3 tries: ask a person (choose)"
    assert decide("reject", 7, 3, True).chosen == "ask_human"                 # past the limit counts as used up
    assert decide("reject", 0, 1, True).chosen == "ask_human"                 # 1 try in all: no retry


def test_the_last_reject_asks_the_task_file_s_question():
    assert decide("reject", 1, 2, True).question == "choose"                  # the default
    assert decide("reject", 1, 2, True, {}).question == "choose"
    assert decide("reject", 1, 2, True, {"reject_last": "guide_text"}).question == "guide_text"
    # a predicate may have its own question for its last reject
    assert decide("reject", 1, 2, True, CABLE_ASK, predicate="at_port").question == "guide_marks"
    assert decide("reject", 1, 2, True, CABLE_ASK, predicate="seated").question == "choose"
    assert decide("reject", 1, 2, True, CABLE_ASK).question == "choose"       # not told the predicate
    assert decide("reject", 0, 2, True, CABLE_ASK, predicate="at_port").chosen == "retry"   # tries left: retry first


def test_an_abstain_asks_a_person_to_confirm():
    for attempt in (0, 1, 4):
        plan = decide("abstain", attempt, 2, True, CABLE_ASK)
        assert (plan.chosen, plan.question, plan.options) == ("ask_human", "confirm", ["ask_human", "abort"])
        assert plan.reason == "abstain: ask a person (confirm)"
    assert decide("abstain", 0, 2, True).question == DEFAULT_QUESTIONS["abstain"] == "confirm"
    assert decide("abstain", 0, 2, True, {"abstain": "guide_text"}).question == "guide_text"


def test_with_nobody_to_ask_the_step_is_aborted_and_stays_unresolved():
    for plan in (decide("abstain", 0, 2, False, CABLE_ASK), decide("reject", 1, 2, False, CABLE_ASK)):
        assert (plan.chosen, plan.options, plan.question) == ("abort", ["abort"], None)
        assert "nobody to ask: abort, the step stays unresolved" in plan.reason


def test_the_policy_never_offers_or_picks_accept():
    for decision, attempt, most, human, ask in itertools.product(
            ("reject", "abstain"), range(5), range(1, 5), (True, False), (None, {}, CABLE_ASK)):
        plan = decide(decision, attempt, most, human, ask, predicate="at_port")
        assert "accept" not in plan.options and plan.chosen != "accept"
        assert plan.chosen in plan.options
        assert fork_record("verdict_1", plan).chosen != "accept"
    with pytest.raises(ValueError, match="an accept opens no fork"):
        decide("accept", 0, 2, True)


@pytest.mark.parametrize("args, words", [
    (("Accept", 0, 2, True), "is not a verdict decision"),
    (("The plug looks seated, so accept.", 0, 2, True), "is not a verdict decision"),
    (("", 0, 2, True), "is not a verdict decision"),
    ((None, 0, 2, True), "is not a verdict decision"),
    (("reject", -1, 2, True), "attempt is -1"),
    (("reject", 1.0, 2, True), "attempt is 1.0"),
    (("reject", True, 2, True), "attempt is True"),
    (("reject", 0, 0, True), "max_attempts is 0"),
    (("reject", 0, 2, "yes"), "human is 'yes'"),
    (("reject", 0, 2, 1), "human is 1"),
    (("abstain", 0, 2, True, {"abstain": "ask"}), "gives abstain the question 'ask'"),
    (("abstain", 0, 2, True, {"Abstain": "confirm"}), "has the key 'Abstain'"),
    (("abstain", 0, 2, True, ["confirm"]), "Fix: give the task file's fork.ask block"),
])
def test_decide_refuses_what_is_not_a_decision_or_a_count(args, words):
    with pytest.raises(ValueError, match="Fix: ") as caught:
        decide(*args)
    assert words in str(caught.value)
    with pytest.raises(ValueError, match="give the claim's predicate name"):
        decide("reject", 1, 2, True, predicate=" ")


def test_the_task_files_fork_blocks_work_with_the_policy():
    for path in sorted((REPO_ROOT / "tasks").glob("*.yaml")):
        task = yaml.safe_load(path.read_text(encoding="utf-8"))
        block = task.get("fork", {})
        for name in task["predicates"]:
            for decision in ("reject", "abstain"):
                for attempt in range(block.get("max_attempts", 2)):
                    decide(decision, attempt, block.get("max_attempts", 2), True, block.get("ask"), predicate=name)


def test_the_task_specs_of_avail_core_drive_the_policy():
    task_module = pytest.importorskip("avail.core.task")
    task = task_module.load_task("cable_loop")
    plan = decide("reject", task.fork["max_attempts"] - 1, task.fork["max_attempts"], True, task.fork.get("ask"),
                  predicate="at_port")
    assert (plan.chosen, plan.question) == ("ask_human", "guide_marks")


# ------------------------------------------------------------------ the plan and the fork record
def test_a_fork_plan_checks_itself():
    with pytest.raises(ValueError, match="no options leaves nothing to choose"):
        ForkPlan([], None, None, "")
    with pytest.raises(ValueError, match="more than once"):
        ForkPlan(["retry", "retry"], "retry", None, "")
    with pytest.raises(ValueError, match="is not on the menu"):
        ForkPlan(["retry"], "abort", None, "")
    with pytest.raises(ValueError, match="is not a question type"):
        ForkPlan(["ask_human"], "ask_human", "ask", "")
    with pytest.raises(ValueError, match="must offer ask_human"):
        ForkPlan(["retry"], "retry", "confirm", "")
    with pytest.raises(ValueError, match="must say what to ask"):
        ForkPlan(["ask_human"], "ask_human", None, "")
    with pytest.raises(ValueError, match="is not a fork option"):
        ForkPlan(["retry", "skip"], "retry", None, "")
    with pytest.raises(TypeError, match="give a list of fork options"):
        ForkPlan("retry", "retry", None, "")
    with pytest.raises(TypeError, match="say why in plain words"):
        ForkPlan(["retry"], "retry", None, None)
    menu = ["retry", "abort"]
    plan = ForkPlan(menu, None, None, "an agent picks")
    menu.append("ask_human")
    assert plan.options == ["retry", "abort"]                    # the plan keeps its own copy
    with pytest.raises(AttributeError):
        plan.chosen = "retry"                                    # frozen


def test_fork_record_carries_the_plan():
    plan = decide("reject", 0, 2, True)
    fork = fork_record("verdict_9", plan)
    assert isinstance(fork, Fork) and fork.fork_id.startswith("fork_")
    assert (fork.verdict_id, fork.options, fork.chosen, fork.chosen_by, fork.reason) == (
        "verdict_9", ["retry", "ask_human", "abort"], "retry", "policy", plan.reason)
    assert fork_record("verdict_9", plan, fork_id="fork_7").fork_id == "fork_7"
    assert fork_record("verdict_9", plan, chosen_by="agent").chosen_by == "agent"
    open_fork = fork_record("verdict_9", ForkPlan(["retry", "abort"], None, None, "for the agent"))
    assert open_fork.chosen is None and open_fork.chosen_by is None
    with pytest.raises(ValueError, match="chosen_by is 'model'"):
        fork_record("verdict_9", plan, chosen_by="model")
    with pytest.raises(TypeError, match="pass the ForkPlan that decide returned"):
        fork_record("verdict_9", {"options": ["retry"], "chosen": "retry"})


def test_choose_closes_an_open_fork_from_its_menu_only():
    open_fork = fork_record("verdict_9", ForkPlan(["retry", "abort"], None, None, "for the agent"))
    closed = choose(open_fork, "abort", "agent", reason="the port is blocked")
    assert (closed.chosen, closed.chosen_by, closed.reason) == ("abort", "agent", "the port is blocked")
    assert open_fork.chosen is None
    for off_menu in ("ask_human", "accept", "repair:anything"):
        with pytest.raises(ValidationError):
            choose(open_fork, off_menu, "agent")


# ------------------------------------------------------------------ what a person is shown and answers
def test_a_confirm_request_offers_yes_no_and_cannot_tell():
    request = HumanRequest("fork_1", claim(), "confirm")
    assert request.options == ["yes", "no", "cannot_tell"] == list(kinds.CONFIRM_ANSWERS)
    assert request.frames is None and request.context == {}
    frames = {"wrist": np.zeros((4, 4, 3), dtype=np.uint8)}
    assert HumanRequest("fork_1", claim(), "guide_marks", frames=frames).frames["wrist"].shape == (4, 4, 3)


@pytest.mark.parametrize("kwargs, error, words", [
    ({"question": "ask"}, ValueError, "is not a question type"),
    ({"question": "choose"}, ValueError, "offers nothing to choose"),
    ({"question": "choose", "options": ["retry", "accept"]}, ValueError, "asks a person to accept"),
    ({"question": "confirm", "options": ["Accept it", "no"]}, ValueError, "asks a person to accept"),
    ({"question": "choose", "options": "retry"}, TypeError, "give a list of the answers offered"),
    ({"question": "choose", "options": ["retry", ""]}, ValueError, "an empty or non-text answer"),
    ({"question": "confirm", "fork_id": " "}, ValueError, "give the id of the fork that asks"),
    ({"question": "confirm", "claim": {"claim_id": "claim_1"}}, TypeError, "pass the Claim record"),
    ({"question": "confirm", "frames": [1, 2]}, TypeError, "camera name to image"),
])
def test_a_request_refuses_what_a_person_cannot_be_shown(kwargs, error, words):
    fields = {"fork_id": "fork_1", "claim": claim(), **kwargs}
    with pytest.raises(error, match="Fix: ") as caught:
        HumanRequest(**fields)
    assert words in str(caught.value)


def test_an_answer_checks_its_fields():
    answer = HumanAnswer("yes", truth_label=np.bool_(True), seconds_to_answer=np.float64(12.5), seconds_handling=3)
    assert answer.truth_label is True and answer.seconds_to_answer == 12.5 and answer.seconds_handling == 3.0
    marks = HumanAnswer("marks", correction_kind="marks", correction={"points": [[10, 20]]})
    assert marks.correction == {"points": [[10, 20]]}
    for bad, words in ((dict(truth_label="yes"), "Text such as 'yes'"), (dict(truth_label=1), "give True, False"),
                       (dict(seconds_to_answer=-1.0), "seconds_to_answer is -1.0"),
                       (dict(seconds_handling=float("nan")), "seconds_handling is nan"),
                       (dict(seconds_handling=True), "seconds_handling is True"),
                       (dict(correction_kind="voice"), "is not a correction kind"),
                       (dict(correction={"text": "go left"}), "a correction is given with no kind")):
        with pytest.raises(ValueError, match="Fix: ") as caught:
            HumanAnswer("yes", **bad)
        assert words in str(caught.value)
    with pytest.raises(ValueError, match="give the person's answer as text"):
        HumanAnswer(" ")
    with pytest.raises(TypeError, match="give a dict"):
        HumanAnswer("taken over", handoff=[1, 2])


def test_a_confirm_answer_carries_the_label_its_word_carries():
    request = HumanRequest("fork_1", claim(), "confirm")
    assert answer_for(request, "yes").truth_label is True
    assert answer_for(request, "no").truth_label is False
    assert answer_for(request, "cannot tell").truth_label is None
    assert answer_for(request, "yes", truth_label=False).truth_label is True    # the word gives the label
    choose_request = HumanRequest("fork_1", claim(), "choose", options=["retry", "abort"])
    assert answer_for(choose_request, "retry").truth_label is None
    assert answer_for(choose_request, "retry", seconds_to_answer=4.0).seconds_to_answer == 4.0


def test_check_answer_refuses_an_answer_that_does_not_fit_its_question():
    confirm = HumanRequest("fork_1", claim(), "confirm")
    with pytest.raises(ValueError, match="is not an answer to a confirm question"):
        check_answer(confirm, HumanAnswer("maybe"))
    with pytest.raises(ValueError, match="carries the truth label True, and the answer says False"):
        check_answer(confirm, HumanAnswer("yes", truth_label=False))
    with pytest.raises(ValueError, match="carries the truth label None"):
        check_answer(confirm, HumanAnswer("cannot_tell", truth_label=True))
    menu = HumanRequest("fork_1", claim(), "choose", options=["retry", "abort"])
    with pytest.raises(ValueError, match="is not one of the options"):
        check_answer(menu, HumanAnswer("accept"))
    assert check_answer(menu, HumanAnswer("abort")).answer == "abort"
    marks = HumanRequest("fork_1", claim("at_port"), "guide_marks")
    assert check_answer(marks, HumanAnswer("marks", correction_kind="marks", correction={"points": []}))
    with pytest.raises(TypeError, match="return a HumanAnswer"):
        check_answer(confirm, "yes")


def test_scripted_human_answers_from_its_script_and_keeps_each_request():
    def script(request):
        if request.question == "confirm":
            return "yes"
        if request.question == "choose":
            return {"answer": request.options[-1], "seconds_to_answer": 9.0}
        return HumanAnswer("marks", correction_kind="marks", correction={"points": [[1, 2]]})

    person = ScriptedHuman(script, assistant_id="test")
    assert isinstance(person, Human)
    answers = [person.ask(HumanRequest("fork_1", claim(), "confirm")),
               person.ask(HumanRequest("fork_2", claim(), "choose", options=["retry", "abort"])),
               person.ask(HumanRequest("fork_3", claim("at_port"), "guide_marks"))]
    assert [(a.answer, a.truth_label) for a in answers] == [("yes", True), ("abort", None), ("marks", None)]
    assert answers[1].seconds_to_answer == 9.0 and len(person.asked) == 3
    assert [request.fork_id for request in person.asked] == ["fork_1", "fork_2", "fork_3"]
    liar = ScriptedHuman(lambda request: HumanAnswer("yes", truth_label=False))
    with pytest.raises(ValueError, match="carries the truth label"):
        liar.ask(HumanRequest("fork_1", claim(), "confirm"))
    with pytest.raises(TypeError, match="pass a HumanRequest"):
        person.ask({"question": "confirm"})
    with pytest.raises(TypeError, match="not callable"):
        ScriptedHuman("yes")


def test_to_assist_makes_the_assist_record_of_an_answer():
    request = HumanRequest("fork_3", claim("at_port"), "guide_marks")
    answer = HumanAnswer("marks", correction_kind="marks", correction={"points": [[212, 148]], "arm": "right"},
                         seconds_to_answer=18.5, seconds_handling=6.0)
    assist = to_assist(request, answer, "oracle", assist_id="assist_9")
    assert isinstance(assist, Assist) and assist.assist_id == "assist_9" and assist.fork_id == "fork_3"
    assert (assist.question, assist.answer, assist.truth_label, assist.assistant_id) == (
        "guide_marks", "marks", None, "oracle")
    assert assist.correction.kind == "marks" and assist.correction.payload["arm"] == "right"
    assert (assist.seconds_to_answer, assist.seconds_handling) == (18.5, 6.0)
    confirm = HumanRequest("fork_4", claim(), "confirm")
    plain_yes = to_assist(confirm, answer_for(confirm, "yes"), "oracle")
    assert plain_yes.correction is None and plain_yes.truth_label is True and plain_yes.assist_id.startswith("assist_")
    assert plain_yes.options == ["yes", "no", "cannot_tell"]


# ------------------------------------------------------------------ takeover
def test_the_handoff_log_tags_each_frame_autonomous_or_assisted():
    log = HandoffLog()
    assert not log.active and log.last() is None and log.tag(0) == AUTONOMOUS == "autonomous"
    log.begin(5, 10.0)
    assert log.active and log.tag(4) == AUTONOMOUS and log.tag(5) == ASSISTED == "assisted" and log.tag(900) == ASSISTED
    span = log.end(9, 12.5)
    assert isinstance(span, Takeover) and not span.is_open and not log.active
    assert [log.tag(i) for i in range(3, 12)] == ["autonomous"] * 2 + ["assisted"] * 5 + ["autonomous"] * 2
    log.begin(20, 15.0)
    assert log.last().is_open and [s.handoff_frame for s in log.takeovers] == [5, 20]
    assert log.end(20, 15.0).handback_frame == 20                     # 1 frame under the person is a takeover too
    assert span.fields() == {"handoff_ts": 10.0, "handback_ts": 12.5, "handoff_frame": 5, "handback_frame": 9}
    record = Intervention(intervention_id="iv_1", run_id="run_t", cycle_id="cycle_1", item=2, step="insert",
                          question="takeover", trigger="reject after 2 attempts", answer="taken over",
                          truth_label=True, asked_ts=9.0, answered_ts=10.0, seconds_waiting=1.0,
                          seconds_handling=2.5, outcome="taken_over_done", **span.fields())
    assert (record.handoff_frame, record.handback_frame) == (5, 9)
    log.reset()
    assert log.takeovers == [] and log.tag(7) == AUTONOMOUS


def test_the_handoff_log_refuses_a_takeover_out_of_order():
    log = HandoffLog()
    with pytest.raises(RuntimeError, match="no takeover is open"):
        log.end(3, 1.0)
    log.begin(5, 10.0)
    with pytest.raises(RuntimeError, match="already open since frame 5"):
        log.begin(6, 11.0)
    with pytest.raises(RuntimeError, match="Fix: end it first"):
        log.reset()
    with pytest.raises(ValueError, match="the handback frame 4 comes before the handoff frame 5"):
        log.end(4, 11.0)
    with pytest.raises(ValueError, match="the handback time 9.000 comes before the handoff time 10.000"):
        log.end(8, 9.0)
    log.end(8, 11.0)
    with pytest.raises(ValueError, match="is not after the last handback, frame 8"):
        log.begin(8, 12.0)
    for bad in (-1, True, 2.5, None):
        with pytest.raises(ValueError, match="frame_index is"):
            log.tag(bad)
    with pytest.raises(ValueError, match="ts is nan"):
        log.begin(10, float("nan"))


def test_a_handback_says_how_the_takeover_ended():
    back = Handback(True, 31)
    assert back.ok is True and back.seconds == 31.0 and back.note == "" and back.before is None
    found = Handback(True, 31, before=False)                 # done after the fix, not done when the person took the arm
    assert (found.ok, found.before) == (True, False)
    for bad in (dict(ok="yes", seconds=1.0), dict(ok=1, seconds=1.0), dict(ok=True, seconds=-1.0),
                dict(ok=True, seconds=float("inf")), dict(ok=True, seconds=1.0, before="no"),
                dict(ok=True, seconds=1.0, before=0)):
        with pytest.raises(ValueError, match="Fix: "):
            Handback(**bad)

    class Hands:
        def take_over(self, cell, request):
            return Handback(True, 4.0, "seated it by hand")

    assert isinstance(Hands(), TeleopSource)
    assert Hands().take_over(None, HumanRequest("fork_1", claim(), "takeover")).note == "seated it by hand"
