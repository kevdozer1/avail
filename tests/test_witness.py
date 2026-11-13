"""The witness: thresholds, the budget split, the cascade, the fused scorer and the store.

Everything here runs on synthetic evidence with known answers. No simulator.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from avail.evidence.fused import Scorer, fit_scorer
from avail.stats import ABSTAIN, ACCEPT, REJECT, conformal_accept_threshold, conformal_reject_threshold
from avail.witness import store
from avail.witness.calibrate import accept_rates, budget_split, stage_thresholds, threshold_at_true_accept
from avail.witness.cascade import Cascade, StageDef, calibrate_cascade, evaluate, fit_scorers, stage_frame

TWO_STAGES = (StageDef("first", ("src.a",), "A", "y0"), StageDef("second", ("src.b",), "B", "B/src.b/_y"))
STATION = "test_cell"               # since S5 a caller names the station a cascade is calibrated on


def repair_table(n: int, seed: int, repair: float = 0.8, neg0: float = 0.6) -> pd.DataFrame:
    """2 stages. Stage 1 is weak. Stage 2 acts first (it repairs most bad states), then measures well."""
    r = np.random.default_rng(seed)
    y0 = r.random(n) >= neg0
    y1 = y0 | (r.random(n) < repair)
    return pd.DataFrame({
        "cls": np.where(y0, "P0", "N2"), "y0": y0,
        "A/src.a/x": np.where(y0, 1.0, 0.0) + r.normal(0, 1, n), "A/src.a/_seconds": 0.0,
        "B/src.b/x": np.where(y1, 4.0, 0.0) + r.normal(0, 1, n), "B/src.b/_y": y1, "B/src.b/_seconds": 1.0,
    })


# ------------------------------------------------------------------ thresholds
def test_stage_thresholds_are_the_conformal_thresholds():
    r = np.random.default_rng(0)
    scores = np.concatenate([r.normal(0, 1, 300), r.normal(3, 1, 200)])
    labels = np.array([False] * 300 + [True] * 200)
    thr = stage_thresholds(scores, labels, 0.02, 0.05)
    assert thr["tau_accept"] == conformal_accept_threshold(scores[:300], 0.02)
    assert thr["tau_reject"] == conformal_reject_threshold(scores[300:], 0.05)
    assert (thr["n_neg"], thr["n_pos"]) == (300, 200)
    assert thr["can_accept"] and thr["can_reject"]


def test_a_stage_with_too_few_failures_never_accepts():
    """docs/04 section 2: at alpha 0.02 a stage needs 49 false states before it may accept at all."""
    scores, labels = np.arange(60.0), np.array([False] * 48 + [True] * 12)
    assert stage_thresholds(scores, labels, 0.02, 0.5)["tau_accept"] == math.inf
    labels = np.array([False] * 49 + [True] * 11)
    assert math.isfinite(stage_thresholds(scores, labels, 0.02, 0.5)["tau_accept"])


def test_scores_that_are_nan_stand_on_nobody():
    thr = stage_thresholds([np.nan, 1.0, 2.0], [False, False, True], 0.5, 0.5)
    assert (thr["n_neg"], thr["n_pos"]) == (1, 1)


def test_threshold_at_true_accept_reaches_the_asked_share():
    r = np.random.default_rng(1)
    s = r.normal(0, 1, 1000)
    y = np.ones(1000, dtype=bool)
    tau = threshold_at_true_accept(s, y, 0.95)
    accepted = (s > tau).mean()
    assert 0.95 <= accepted < 0.952                        # the largest threshold that still accepts 95%


def test_threshold_at_true_accept_with_a_flag_accepts_the_flagged():
    s = np.array([1.0] * 90 + [0.0] * 10 + [1.0] * 40 + [0.0] * 60)
    y = np.array([True] * 100 + [False] * 100)
    # accepting only flag 1 keeps 90% of the true cases, short of 95%, so everything must be accepted
    assert threshold_at_true_accept(s, y, 0.95) == -math.inf
    # at 90% the flag itself is the rule
    tau = threshold_at_true_accept(s, y, 0.90)
    assert tau == 0.0
    rates = accept_rates(s, y, tau)
    assert rates["accepted_true"] == 90 and rates["accepted_false"] == 40


def test_threshold_at_true_accept_counts_unscored_true_cases_as_not_accepted():
    s = np.array([np.nan] * 10 + list(np.linspace(1, 2, 90)))
    y = np.ones(100, dtype=bool)
    assert threshold_at_true_accept(s, y, 0.95) == -math.inf     # 10 unscored already breaks 95%
    assert threshold_at_true_accept(s, y, 0.90) == -math.inf     # exactly 90 scored, none to spare
    assert math.isfinite(threshold_at_true_accept(s, y, 0.85))


# ------------------------------------------------------------------ the budget split
def test_budget_is_shared_equally_when_no_stage_repairs():
    a, b = budget_split([0.6, 0.6, 0.6], [0.4, 0.4, 0.4], 0.03, 0.06)
    assert a == pytest.approx([0.01, 0.01, 0.01])
    assert b == pytest.approx([0.02, 0.02, 0.02])


def test_budget_weights_a_stage_by_how_many_bad_states_it_still_faces():
    a, b = budget_split([0.6, 0.12], [0.4, 0.88], 0.02, 0.05)
    assert a == pytest.approx([0.01 * 0.12 / 0.6, 0.01])    # the early stage faces 5 times the bad states
    assert b == pytest.approx([0.025, 0.025 * 0.4 / 0.88])
    assert sum(a) <= 0.02 + 1e-12 and sum(b) <= 0.05 + 1e-12


def test_budget_of_a_stage_that_can_never_act_goes_to_the_others():
    # 3 stages at alpha 0.02: each would get 1/150, which needs 149 false states. Stage 0 has 60.
    a, _ = budget_split([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], 0.02, 0.05, n_neg=[60, 500, 500], n_pos=[500, 500, 500])
    assert a[0] == 0.0
    assert a[1] == pytest.approx(0.01) and a[2] == pytest.approx(0.01)


def test_budget_is_zero_everywhere_when_no_stage_has_enough_failures():
    a, _ = budget_split([0.5, 0.5], [0.5, 0.5], 0.02, 0.05, n_neg=[10, 10], n_pos=[500, 500])
    assert a == [0.0, 0.0]


# ------------------------------------------------------------------ the fused scorer
def test_scorer_separates_and_round_trips():
    r = np.random.default_rng(2)
    y = r.random(800) < 0.5
    frame = pd.DataFrame({"a": np.where(y, 2.0, 0.0) + r.normal(0, 1, 800), "b": r.normal(0, 1, 800)})
    scorer = fit_scorer("t", frame, ["a", "b"], y)
    s = scorer.score(frame)
    assert s[y].mean() > s[~y].mean() + 1.0
    again = Scorer.from_dict(scorer.to_dict())
    assert np.array_equal(again.score(frame), s)
    assert scorer.score_one({"a": 3.0, "b": 0.0}) > scorer.score_one({"a": -1.0, "b": 0.0})


def test_scorer_handles_missing_features_and_flags_them():
    r = np.random.default_rng(3)
    y = r.random(600) < 0.5
    slip = np.where(y, np.nan, 1.0 + r.normal(0, 0.1, 600))          # a slip force exists only when the plug slipped
    frame = pd.DataFrame({"slip_n": slip, "disp": np.where(y, 0.05, 3.0) + r.normal(0, 0.02, 600)})
    scorer = fit_scorer("t", frame, ["slip_n", "disp"], y)
    assert scorer.flag_missing[0] is True and scorer.flag_missing[1] is False
    assert np.isfinite(scorer.score(frame)).all()
    assert np.isfinite(scorer.score_one({"disp": 0.05}))             # a feature left out entirely
    assert scorer.score_one({"disp": 0.05}) > scorer.score_one({"disp": 3.0, "slip_n": 1.0})


def test_scorer_from_one_class_says_so():
    frame = pd.DataFrame({"a": np.arange(10.0)})
    scorer = fit_scorer("t", frame, ["a"], np.ones(10, dtype=bool))
    assert scorer.constant is True
    assert len(set(scorer.score(frame))) == 1


# ------------------------------------------------------------------ the cascade
@pytest.fixture(scope="module")
def fitted():
    fit = repair_table(6000, 11)
    return fit_scorers(fit, "latch", "latched", TWO_STAGES)


def test_stage_frame_names_columns_without_the_pass_prefix():
    table = repair_table(10, 5)
    assert list(stage_frame(table, TWO_STAGES, 0).columns) == ["src.a/x"]
    assert list(stage_frame(table, TWO_STAGES, 1).columns) == ["src.a/x", "src.b/x"]


def test_cascade_decisions_follow_the_thresholds(fitted):
    scorers, neg, pos = fitted
    cal, test = repair_table(3000, 12), repair_table(2000, 13)
    cascade = calibrate_cascade(scorers, neg, pos, cal, "latch", "latched", 0.02, 0.05, TWO_STAGES, station_id=STATION)
    dec = cascade.decide_table(test)
    s1 = cascade.stages[0].scorer.score(stage_frame(test, TWO_STAGES, 0))
    first = cascade.stages[0]
    lo, hi = min(first.tau_reject, first.tau_accept), max(first.tau_reject, first.tau_accept)
    at_first = dec["stage_reached"] == 1
    assert ((s1 > hi) | (s1 < lo))[at_first].all()                  # decided at stage 1 only outside the band
    assert (dec["decision"][at_first & (s1 > hi)] == ACCEPT).all()
    assert (dec["decision"][at_first & (s1 < lo)] == REJECT).all()
    # the truth a decision is judged against is the truth after the last stage that ran
    assert (dec["y_dec"][at_first] == test["y0"][at_first]).all()
    assert (dec["y_dec"][~at_first] == test["B/src.b/_y"][~at_first]).all()
    # time: stage 2 costs 1 s and only those who reach it pay
    assert (dec["seconds"][at_first] == 0.0).all() and (dec["seconds"][~at_first] == 1.0).all()
    assert set(dec["decision"]) <= {ACCEPT, REJECT, ABSTAIN}


def _mean_far(scorers, neg, pos, budget: str, draws: int = 120) -> tuple[float, float]:
    test = repair_table(40000, 999)
    fars = []
    for d in range(draws):
        cal = repair_table(3000, 5000 + d)
        cascade = calibrate_cascade(scorers, neg, pos, cal, "latch", "latched", 0.02, 0.05, TWO_STAGES,
                                    station_id=STATION, budget=budget)
        fars.append(evaluate(cascade, test)["far"]["rate"])
    fars = np.array(fars)
    return float(fars.mean()), float(fars.std(ddof=1) / math.sqrt(draws))


@pytest.mark.slow
def test_weighted_budget_keeps_the_false_accept_rate_under_alpha_when_a_stage_repairs(fitted):
    """The promise, in simulation: mean false-accept over calibration draws is at most alpha.

    'Within sampling error' means at most alpha plus 3 standard errors of the mean.
    """
    scorers, neg, pos = fitted
    mean, se = _mean_far(scorers, neg, pos, "weighted")
    assert mean <= 0.02 + 3 * se, (mean, se)


@pytest.mark.slow
def test_equal_budget_breaks_the_promise_when_a_stage_repairs(fitted):
    """Why the split is weighted. With docs/04's equal split, a weak early stage is calibrated
    against 5 times the bad states the decision is judged on, and the rate at the decision
    runs well past alpha. This test fails if someone makes the weighting a no-op."""
    scorers, neg, pos = fitted
    mean, se = _mean_far(scorers, neg, pos, "equal", draws=40)
    assert mean > 0.02 + 5 * se, (mean, se)
    assert mean > 0.03


def test_evaluate_counts_add_up(fitted):
    scorers, neg, pos = fitted
    cascade = calibrate_cascade(scorers, neg, pos, repair_table(3000, 21), "latch", "latched", 0.02, 0.05, TWO_STAGES,
                                station_id=STATION)
    test = repair_table(2000, 22)
    res = evaluate(cascade, test)
    dec = cascade.decide_table(test)
    assert res["n"] == 2000
    assert res["far"]["n"] + res["frr"]["n"] == 2000
    assert res["far"]["k"] == int(((dec["decision"] == ACCEPT) & ~dec["y_dec"]).sum())
    assert res["abstain"]["k"] == int((dec["decision"] == ABSTAIN).sum())
    assert res["far_ucb"] >= res["far"]["rate"]
    assert sum(res["stage_reached"].values()) == 2000
    assert sum(v["n"] for v in res["per_class"].values()) == 2000


def test_cascade_round_trips_and_exports_a_valid_witness_record(fitted):
    scorers, neg, pos = fitted
    cascade = calibrate_cascade(scorers, neg, pos, repair_table(3000, 31), "latch", "latched", 0.02, 0.05, TWO_STAGES,
                                station_id=STATION)
    again = Cascade.from_dict(cascade.to_dict())
    test = repair_table(500, 32)
    assert cascade.decide_table(test).equals(again.decide_table(test))
    witness = cascade.to_witness()
    assert witness.predicate_name == "latched" and len(witness.stages) == 2
    assert sum(s.alpha_k for s in witness.stages) <= witness.alpha + 1e-9
    assert sum(s.beta_k for s in witness.stages) <= witness.beta + 1e-9


# ------------------------------------------------------------------ live verification and the done gate
class _FakeSource:
    kind = "passive_proprio"
    predicates = {"latched"}

    def __init__(self, source_id: str, value: float):
        self.source_id, self.value, self.calls = source_id, value, 0

    def estimate_cost(self, claim):
        from avail.schemas import Cost

        return Cost(seconds=0.5, usd=0.0, moved_robot=False)

    def collect(self, station, claim, params):
        from avail.evidence.base import make_evidence

        self.calls += 1
        return make_evidence(self, claim, {"x": self.value}, score=self.value, seconds=0.5)


def _claim():
    from avail.schemas import Claim, ClaimContext, Predicate

    return Claim(claim_id="c1", run_id="r", cycle_id="cy", step_id="insert",
                 predicate=Predicate(name="latched", target={"plug": "p1", "port": 3}), raised_by="skill",
                 context=ClaimContext(task="plug_cycle", stage="insert", attempt=0))


def test_live_verify_stops_at_the_first_decision_and_lists_its_evidence(fitted, monkeypatch):
    from avail.evidence import base

    scorers, neg, pos = fitted
    cascade = calibrate_cascade(scorers, neg, pos, repair_table(3000, 41), "latch", "latched", 0.02, 0.05, TWO_STAGES,
                                station_id=STATION)
    a, b = _FakeSource("src.a", 0.3), _FakeSource("src.b", 9.0)
    monkeypatch.setitem(base.REGISTRY, "src.a", a)
    monkeypatch.setitem(base.REGISTRY, "src.b", b)
    verdict, evidence = cascade.verify(object(), _claim(), {})
    assert verdict.decision == ACCEPT and verdict.stage_reached == 2          # weak first stage, strong second
    assert verdict.evidence_ids == [e.evidence_id for e in evidence] and len(evidence) == 2
    assert verdict.cost_total.seconds == pytest.approx(1.0)
    assert verdict.witness_version == cascade.version
    # a first stage that is sure ends it there, and the second source is never asked
    a.value, a.calls, b.calls = 50.0, 0, 0
    verdict, evidence = cascade.verify(object(), _claim(), {})
    assert verdict.decision == ACCEPT and verdict.stage_reached == 1
    assert b.calls == 0 and len(evidence) == 1


def test_live_verify_abstains_between_the_thresholds(fitted, monkeypatch):
    from avail.evidence import base

    scorers, neg, pos = fitted
    cascade = calibrate_cascade(scorers, neg, pos, repair_table(3000, 42), "latch", "latched", 0.02, 0.05, TWO_STAGES,
                                station_id=STATION)
    monkeypatch.setitem(base.REGISTRY, "src.a", _FakeSource("src.a", 0.5))
    monkeypatch.setitem(base.REGISTRY, "src.b", _FakeSource("src.b", 2.0))        # between a bad and a good second stage
    verdict, evidence = cascade.verify(object(), _claim(), {})
    assert verdict.decision == ABSTAIN
    assert verdict.stage_reached == 2 and len(evidence) == 2


def test_an_accepting_verdict_cannot_exist_without_evidence():
    """CLAUDE.md rule 11. The cascade is the only place a verdict is built, and the record refuses this."""
    from avail.schemas import Cost, Verdict

    with pytest.raises(Exception):
        Verdict(verdict_id="v", claim_id="c", decision="accept", score=1.0, stage_reached=1, witness_version="v1",
                evidence_ids=[], cost_total=Cost(seconds=0.0, usd=0.0, moved_robot=False))


# ------------------------------------------------------------------ the store
def test_store_publishes_versions_and_never_rewrites(fitted, tmp_path):
    scorers, neg, pos = fitted
    cascade = calibrate_cascade(scorers, neg, pos, repair_table(3000, 51), "latch", "latched", 0.02, 0.05, TWO_STAGES,
                                station_id=STATION)
    path1, c1 = store.publish(cascade, checks={"x": 1}, provenance={"split": "test"}, root=tmp_path)
    assert path1.name == "v1.json" and c1.version == "v1"
    path2, c2 = store.publish(cascade, root=tmp_path)
    assert path2.name == "v2.json" and c2.version == "v2"
    assert store.versions("latch", "latched", tmp_path) == [1, 2]
    loaded, payload = store.load("latch", "latched", 1, tmp_path)
    assert payload["sim_or_real"] == "sim" and payload["checks"] == {"x": 1}
    assert payload["witness"]["version"] == "v1"
    test = repair_table(300, 52)
    assert loaded.decide_table(test).equals(cascade.decide_table(test))
    with pytest.raises(FileNotFoundError):
        store.load("fric", "seated", root=tmp_path)


def test_a_dropped_late_stage_still_sets_the_floor():
    """Regression, found by the statistics review of 2026-11-11.

    With few calibration states the later stage cannot act at its share and is dropped. Its
    budget went to the early stage at full weight, but states that reach the dropped stage are
    still repaired there and judged on the truth after it, so the early stage answered for 5 times
    the bad states the rate is judged on. The floor of the weights must cover every stage.
    """
    a, _ = budget_split([0.6, 0.12], [0.4, 0.88], 0.02, 0.05, n_neg=[420, 80], n_pos=[280, 620])
    assert a[1] == 0.0                                       # 80 false states cannot carry 0.01
    assert a[0] == pytest.approx(0.02 * 0.12 / 0.6)           # and the early stage stays weighted


@pytest.mark.slow
def test_weighted_budget_holds_with_few_calibration_states():
    """The same 2-stage world as above with 700 calibration states, where the drop happens."""
    fit = repair_table(6000, 61)
    scorers, neg, pos = fit_scorers(fit, "latch", "latched", TWO_STAGES)
    test = repair_table(40000, 62)
    fars = []
    for d in range(100):
        cascade = calibrate_cascade(scorers, neg, pos, repair_table(700, 7000 + d), "latch", "latched", 0.02, 0.05,
                                    TWO_STAGES, station_id=STATION)
        fars.append(evaluate(cascade, test)["far"]["rate"])
    fars = np.array(fars)
    assert fars.mean() <= 0.02 + 3 * fars.std(ddof=1) / math.sqrt(len(fars)), fars.mean()
