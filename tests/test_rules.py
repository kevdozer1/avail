"""The rule witness (avail/witness/rules.py): a task file's thresholds on named features.

Every evidence source here is a fake registered for the test. The built-in sources are never
imported, so nothing reaches a station or a simulator.
"""

from __future__ import annotations

import pytest
import yaml
from pydantic import BaseModel

from avail.evidence import base
from avail.evidence.base import make_evidence
from avail.paths import REPO_ROOT
from avail.schemas import Claim, ClaimContext, Cost, Predicate, Verdict
from avail.witness.rules import RuleCondition, RuleJudgement, RuleWitness


class FakeSource:
    """An evidence source that reports fixed features and remembers what it was asked."""

    kind = "passive_proprio"

    def __init__(self, source_id: str, features: dict, seconds: float = 0.25):
        self.source_id, self.features, self.seconds = source_id, dict(features), seconds
        self.predicates = {"held", "at_port", "released_stable"}
        self.calls: list[dict] = []

    def estimate_cost(self, claim):
        return Cost(seconds=self.seconds, usd=0.0, moved_robot=False)

    def collect(self, station, claim, params):
        self.calls.append(dict(params))
        return make_evidence(self, claim, self.features, seconds=self.seconds)


@pytest.fixture
def source(monkeypatch):
    """``source(id, features)`` registers a fake for this test. The built-in modules are never loaded."""
    monkeypatch.setattr(base, "load_builtin", lambda: base.REGISTRY)

    def add(source_id: str, features: dict, seconds: float = 0.25) -> FakeSource:
        fake = FakeSource(source_id, features, seconds)
        monkeypatch.setitem(base.REGISTRY, source_id, fake)
        return fake

    return add


def claim(name: str = "held") -> Claim:
    return Claim(claim_id="claim_1", run_id="run_t", cycle_id="cycle_1", step_id="pick",
                 predicate=Predicate(name=name, target={"item": 3, "port": 2}), raised_by="skill",
                 context=ClaimContext(task="cable_loop", stage="pick", attempt=0))


HELD = {
    "kind": "rule",
    "accept_all": [{"feature": "fake.jaw/in_band", "above": 0.5}, {"feature": "fake.tag/match", "above": 0.5}],
    "reject_any": [{"feature": "fake.jaw/closed_empty", "above": 0.5}, {"feature": "fake.tag/mismatch", "above": 0.5}],
}


# ------------------------------------------------------------------ the 3 answers
def test_accept_when_every_accept_condition_holds(source):
    jaw = source("fake.jaw", {"in_band": 1.0, "closed_empty": 0.0}, seconds=0.25)
    tag = source("fake.tag", {"match": 1.0, "mismatch": 0.0}, seconds=0.5)
    witness = RuleWitness("held", HELD, "rule-v1")
    assert witness.checks == ("fake.jaw", "fake.tag")              # in the order the conditions name them
    verdict, evidence = witness.verify(object(), claim(), {"arm": "right"})
    assert isinstance(verdict, Verdict) and verdict.decision == "accept"
    assert [item.source_id for item in evidence] == ["fake.jaw", "fake.tag"]
    assert verdict.evidence_ids == [item.evidence_id for item in evidence]      # an accept lists its evidence
    assert verdict.cost_total.seconds == pytest.approx(0.75) and verdict.witness_version == "rule-v1"
    assert (verdict.stage_reached, verdict.score, verdict.claim_id) == (1, None, "claim_1")
    assert jaw.calls == [{"arm": "right"}] and tag.calls == [{"arm": "right"}]
    assert witness.judge(witness.features(evidence)).reason == "accept: all 2 accept conditions hold"


def test_a_reject_condition_rejects_even_when_every_accept_condition_holds(source):
    source("fake.jaw", {"in_band": 1.0, "closed_empty": 1.0})
    source("fake.tag", {"match": 1.0, "mismatch": 0.0})
    witness = RuleWitness("held", HELD, "rule-v1")
    verdict, evidence = witness.verify(object(), claim(), {})
    assert verdict.decision == "reject" and len(verdict.evidence_ids) == 2   # a sign of failure blocks the accept
    judgement = witness.judge(witness.features(evidence))
    assert judgement.reason == "reject: fake.jaw/closed_empty is 1, above 0.5"
    assert [condition.feature for condition in judgement.fired] == ["fake.jaw/closed_empty"]


def test_an_accept_condition_that_fails_abstains(source):
    source("fake.jaw", {"in_band": 0.0, "closed_empty": 0.0})
    source("fake.tag", {"match": 1.0, "mismatch": 0.0})
    witness = RuleWitness("held", HELD, "rule-v1")
    verdict, evidence = witness.verify(object(), claim(), {})
    assert verdict.decision == "abstain" and len(evidence) == 2
    assert witness.judge(witness.features(evidence)).reason == (
        "abstain: fake.jaw/in_band is 0 and needs to be above 0.5")


def test_missing_evidence_never_accepts_and_never_rejects(source):
    source("fake.jaw", {"in_band": 1.0, "closed_empty": 0.0})
    source("fake.tag", {})                                         # the camera found nothing to read
    witness = RuleWitness("held", HELD, "rule-v1")
    verdict, evidence = witness.verify(object(), claim(), {})
    assert verdict.decision == "abstain"
    judgement = witness.judge(witness.features(evidence))
    assert judgement.missing == ("fake.tag/match", "fake.tag/mismatch")
    assert judgement.reason == "abstain: fake.tag/match was not measured"
    # a reject feature that was not measured does not fire either
    assert witness.judge({"fake.jaw/in_band": 1.0, "fake.tag/match": 1.0}).decision == "accept"
    assert witness.judge({}).decision == "abstain"
    assert witness.judge({"fake.jaw/in_band": float("nan"), "fake.tag/match": 1.0}).decision == "abstain"


def test_thresholds_are_strict_and_a_band_needs_both_ends():
    above = RuleCondition("a/x", above=0.5)
    assert not above.holds({"a/x": 0.5}) and above.holds({"a/x": 0.5000001}) and not above.holds({})
    below = RuleCondition("a/x", below=0.8)
    assert not below.holds({"a/x": 0.8}) and below.holds({"a/x": 0.79})
    band = RuleCondition("a/x", above=0.2, below=0.8)
    assert band.holds({"a/x": 0.5}) and not band.holds({"a/x": 0.1}) and not band.holds({"a/x": 0.9})
    assert band.bounds() == "above 0.2 and below 0.8" and band.check == "a"
    assert band.says({"a/x": 0.9}) == "a/x is 0.9 and needs to be above 0.2 and below 0.8"
    assert band.says({"a/x": 0.5}) == "a/x is 0.5, above 0.2 and below 0.8" and band.says({}) == "a/x was not measured"


def test_a_rule_with_only_reject_conditions_never_accepts():
    witness = RuleWitness("held", {"kind": "rule", "reject_any": [{"feature": "fake.jaw/closed_empty", "above": 0.5}]},
                          "rule-v1")
    assert witness.judge({"fake.jaw/closed_empty": 0.0}).decision == "abstain"
    assert witness.judge({"fake.jaw/closed_empty": 1.0}).decision == "reject"
    assert "no accept condition" in witness.judge({"fake.jaw/closed_empty": 0.0}).reason


def test_a_check_with_a_view_runs_its_source_with_that_view(source):
    cam = source("fake.cam", {"plug_found": 1.0, "gap_mm_est": 0.3})
    jaw = source("fake.jaw", {"open": 1.0})
    rule = {"kind": "rule",
            "accept_all": [{"feature": "fake.jaw/open", "above": 0.5}, {"feature": "fake.cam@now/plug_found", "above": 0.5},
                           {"feature": "fake.cam@now/gap_mm_est", "below": 0.8}],
            "reject_any": [{"feature": "fake.cam@now/gap_mm_est", "above": 2.0}]}
    witness = RuleWitness("released_stable", rule, "rule-v1", checks=["fake.jaw", "fake.cam@now"])
    params = {"arm": "right"}
    verdict, evidence = witness.verify(object(), claim("released_stable"), params)
    assert verdict.decision == "accept"
    assert cam.calls == [{"arm": "right", "view": "now"}] and jaw.calls == [{"arm": "right"}]
    assert params == {"arm": "right"}                                   # the caller's params are not changed
    assert witness.features(evidence) == {"fake.jaw/open": 1.0, "fake.cam@now/plug_found": 1.0,
                                          "fake.cam@now/gap_mm_est": 0.3}


def test_checks_run_in_the_order_given_even_when_a_condition_names_them_otherwise(source):
    first, second = source("fake.tag", {"match": 1.0}), source("fake.jaw", {"in_band": 1.0})
    witness = RuleWitness("held", HELD, "rule-v1", checks=["fake.tag", "fake.extra", "fake.jaw"])
    extra = source("fake.extra", {"noise": 0.5})
    verdict, evidence = witness.verify(object(), claim(), None)
    assert [item.source_id for item in evidence] == ["fake.tag", "fake.extra", "fake.jaw"]
    assert verdict.decision == "accept" and len(first.calls) == len(second.calls) == len(extra.calls) == 1


# ------------------------------------------------------------------ the task files' rules
def _task(name: str) -> dict:
    return yaml.safe_load((REPO_ROOT / "tasks" / f"{name}.yaml").read_text(encoding="utf-8"))


def _meeting(conditions: list[RuleCondition]) -> float:
    """A value that meets every condition given, all on 1 feature."""
    low = max((c.above for c in conditions if c.above is not None), default=None)
    high = min((c.below for c in conditions if c.below is not None), default=None)
    if low is not None and high is not None:
        return (low + high) / 2
    return low + 1.0 if low is not None else (high - 1.0 if high is not None else 0.0)


def _rules_in_the_task_files() -> list[tuple[str, str, dict]]:
    found = []
    for path in sorted((REPO_ROOT / "tasks").glob("*.yaml")):
        for name, spec in yaml.safe_load(path.read_text(encoding="utf-8"))["predicates"].items():
            if spec["decide"]["kind"] == "rule":
                found.append((path.stem, name, spec))
    return found


def test_the_task_files_hold_rules():
    assert len(_rules_in_the_task_files()) >= 1


@pytest.mark.parametrize("task, name, spec", _rules_in_the_task_files(), ids=lambda v: v if isinstance(v, str) else "")
def test_each_task_file_rule_accepts_rejects_and_abstains_as_written(source, task, name, spec):
    """Whatever thresholds the task file holds: a state that meets every accept condition is accepted, each reject
    condition alone turns it into a reject, and an accept feature that was not measured gives an abstain."""
    witness = RuleWitness(name, spec["decide"], "rule-v1", checks=spec["checks"])
    by_feature: dict[str, list[RuleCondition]] = {}
    for condition in witness.accept_all:
        by_feature.setdefault(condition.feature, []).append(condition)
    good = {feature: _meeting(conditions) for feature, conditions in by_feature.items()}
    assert witness.judge(good).decision == "accept", f"{task}.{name}: no reject condition may fire on a good state"
    for condition in witness.reject_any:
        assert witness.judge({**good, condition.feature: _meeting([condition])}).decision == "reject", condition
    for condition in witness.accept_all:
        less = {feature: value for feature, value in good.items() if feature != condition.feature}
        assert witness.judge(less).decision == "abstain", condition
    # and live, with each check a fake that reports its share of the good state
    for check in witness.checks:
        prefix = check + "/"
        source(check.partition("@")[0], {f[len(prefix):]: v for f, v in good.items() if f.startswith(prefix)})
    verdict, evidence = witness.verify(object(), claim(name), {})
    assert verdict.decision == "accept" and len(evidence) == len(witness.checks)


def test_every_rule_in_the_cable_task_file_builds():
    predicates = _task("cable_loop")["predicates"]
    rules = {name: spec for name, spec in predicates.items() if spec["decide"]["kind"] == "rule"}
    assert rules
    for name, spec in rules.items():
        witness = RuleWitness(name, spec["decide"], "rule-v1", checks=spec["checks"])
        assert witness.checks == tuple(spec["checks"]) and witness.accept_all and witness.reject_any
        assert RuleWitness.from_predicate({"name": name, **spec}, "rule-v1").checks == witness.checks
        assert RuleWitness(name, spec, "rule-v1").checks == witness.checks          # the whole entry works too
        again = RuleWitness.from_dict(witness.to_dict())
        assert again.to_dict() == witness.to_dict()
    for name, spec in predicates.items():
        if spec["decide"]["kind"] == "witness":
            with pytest.raises(ValueError, match="A cascade comes from the witness store"):
                RuleWitness(name, spec["decide"], "rule-v1")


def test_a_pydantic_spec_reads_like_a_dict():
    class Condition(BaseModel):
        feature: str
        above: float | None = None
        below: float | None = None

    class Decide(BaseModel):
        kind: str = "rule"
        accept_all: list[Condition] = []
        reject_any: list[Condition] = []
        witness: str | None = None

    spec = Decide(accept_all=[Condition(feature="fake.jaw/in_band", above=0.5)],
                  reject_any=[Condition(feature="fake.jaw/closed_empty", above=0.5)])
    witness = RuleWitness("held", spec, "rule-v1")
    assert witness.checks == ("fake.jaw",) and witness.judge({"fake.jaw/in_band": 1.0}).decision == "accept"


def test_the_task_specs_of_avail_core_build_rule_witnesses():
    task_module = pytest.importorskip("avail.core.task")
    task = task_module.load_task("cable_loop")
    built = 0
    for name, predicate in task.predicates.items():
        if predicate.decide.kind == "rule":
            witness = RuleWitness.from_predicate(predicate, "rule-v1")
            assert witness.predicate == name and witness.checks == tuple(predicate.checks)
            built += 1
    assert built >= 3


# ------------------------------------------------------------------ a bad rule fails when it is built
@pytest.mark.parametrize("spec, words", [
    ({"kind": "witness", "witness": "{variant}"}, "A cascade comes from the witness store"),
    ({"kind": "rule"}, "has no condition"),
    ({"kind": "rule", "accept_all": [], "reject_any": []}, "has no condition"),
    ({"kind": "rule", "accept_all": [{"feature": "a/x", "above": 0.5}], "thresholds": 1}, "which a rule does not take"),
    ({"kind": "rule", "accept_all": [{"feature": "a/x"}]}, "has no threshold"),
    ({"kind": "rule", "accept_all": [{"feature": "a/x", "above": 0.5, "abuve": 1}]}, "which a condition does not take"),
    ({"kind": "rule", "accept_all": [{"feature": "in_band", "above": 0.5}]}, "name it <check>/<feature>"),
    ({"kind": "rule", "accept_all": [{"feature": "a/", "above": 0.5}]}, "name it <check>/<feature>"),
    ({"kind": "rule", "accept_all": [{"feature": "a/x", "above": 0.8, "below": 0.2}]}, "which no value meets"),
    ({"kind": "rule", "accept_all": [{"feature": "a/x", "above": "0.5"}]}, "which is not a number"),
    ({"kind": "rule", "accept_all": [{"feature": "a/x", "above": True}]}, "which is not a number"),
    ({"kind": "rule", "accept_all": [{"feature": "a/x", "below": float("inf")}]}, "give a finite number"),
    ({"kind": "rule", "accept_all": ["a/x above 0.5"]}, "which is not a condition"),
    ({"kind": "rule", "accept_all": [{"feature": "a/x", "above": 0.5}], "witness": "latch"}, "takes no witness name"),
    ("accept when it looks right", "which is not a decide block"),
])
def test_a_bad_rule_is_refused_when_the_witness_is_built(spec, words):
    with pytest.raises(ValueError, match="Fix: ") as caught:
        RuleWitness("held", spec, "rule-v1")
    assert words in str(caught.value)


def test_the_checks_must_cover_every_condition_once():
    with pytest.raises(ValueError, match="which is not among the checks it collects"):
        RuleWitness("held", HELD, "rule-v1", checks=["fake.jaw"])
    with pytest.raises(ValueError, match="list each check once"):
        RuleWitness("held", HELD, "rule-v1", checks=["fake.jaw", "fake.tag", "fake.jaw"])
    with pytest.raises(ValueError, match="give the predicate's name"):
        RuleWitness("", HELD, "rule-v1")
    with pytest.raises(ValueError, match="give the rule a version label"):
        RuleWitness("held", HELD, " ")


def test_a_claim_about_another_predicate_is_refused(source):
    source("fake.jaw", {"in_band": 1.0})
    source("fake.tag", {"match": 1.0})
    with pytest.raises(ValueError, match="this rule judges held, and the claim claim_1 is about at_port"):
        RuleWitness("held", HELD, "rule-v1").verify(object(), claim("at_port"), {})


def test_an_unknown_source_is_a_key_error(source):
    source("fake.jaw", {"in_band": 1.0})
    with pytest.raises(KeyError, match="unknown evidence source 'fake.tag'"):
        RuleWitness("held", HELD, "rule-v1").verify(object(), claim(), {})


def test_features_wants_the_evidence_verify_returned(source):
    source("fake.jaw", {"in_band": 1.0})
    source("fake.tag", {"match": 1.0})
    witness = RuleWitness("held", HELD, "rule-v1")
    _verdict, evidence = witness.verify(object(), claim(), {})
    with pytest.raises(ValueError, match="got 1 evidence items for 2 checks"):
        witness.features(evidence[:1])
    with pytest.raises(ValueError, match="the evidence from fake.tag sits where fake.jaw belongs"):
        witness.features(evidence[::-1])
    assert isinstance(witness.judge(witness.features(evidence)), RuleJudgement)
