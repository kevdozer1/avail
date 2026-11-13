"""E1 and E2 on a synthetic zoo table with the real column layout. No simulator.

The synthetic world: a push repairs most short plugs, a tug catches an unlatched one, the
camera is noisy. The tests pin the accounting (who is judged against which truth) and the
way H1 and H2 are marked.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from avail.analysis import e1, e2
from avail.witness.cascade import evaluate


def fake_zoo(n: int, seed: int, vision_noise: float = 0.8) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    cls = r.choice(["P0", "N1", "N2", "N3", "N5", "N6"], size=n, p=[0.4, 0.12, 0.12, 0.12, 0.12, 0.12])
    y0 = cls == "P0"
    short = np.select([cls == "N1", cls == "N2", cls == "N3"], [0.75, 1.5, 3.0], 0.0)
    t = pd.DataFrame({"seed": np.arange(n), "cls": cls, "claim_predicate": "latched", "y0": y0})
    t["A/passive.skill_flag/flag"] = 1.0
    t["A/passive.pose_residual/axial_mm"] = short + r.normal(0, 1.1, n)
    t["A/passive.pose_residual/port_match"] = (cls != "N6").astype(float)
    t["A/passive.click/snap_mm"] = np.where(y0 | (cls == "N6"), 0.4, 0.0) + r.normal(0, 0.08, n)
    t["A/passive.jaw_state/jaw_pos"] = 0.16 + r.normal(0, 0.002, n)
    t["A/vision.gap@claim/gap_mm_est"] = short + r.normal(0, vision_noise, n)
    for key in ("passive.skill_flag", "passive.pose_residual", "passive.click", "passive.jaw_state", "vision.gap@claim"):
        t[f"A/{key}/_y"] = y0
        t[f"A/{key}/_seconds"] = 0.0
    short_cls = np.isin(cls, ["N1", "N2", "N3"])
    repaired = short_cls & (r.random(n) < 0.9)
    y_push = y0 | repaired
    for p in ("A", "B"):
        t[f"{p}/probe.push_seat/travel_mm"] = np.where(short_cls, short, 0.05) + r.normal(0, 0.05, n)
        t[f"{p}/probe.push_seat/clicked"] = repaired.astype(float)
        t[f"{p}/probe.push_seat/_y"] = y_push
        t[f"{p}/probe.push_seat/_seconds"] = 1.0
    slip_b = (cls == "N5") | (~y_push & (cls != "N6"))
    t["B/probe.tug/disp_mm"] = np.where(slip_b, 3.0, 0.05) + r.normal(0, 0.03, n)
    t["B/probe.tug/reseat_clicked"] = (slip_b & (cls != "N5")).astype(float)
    t["B/probe.tug/_y"] = y_push | (slip_b & (cls != "N5"))
    t["B/probe.tug/_seconds"] = 2.0
    slip_a = ~y0 & (cls != "N6")
    t["A/probe.tug/disp_mm"] = np.where(slip_a, 3.0, 0.05) + r.normal(0, 0.03, n)
    t["A/probe.tug/reseat_clicked"] = short_cls.astype(float)
    t["A/probe.tug/_y"] = y0 | short_cls
    t["A/probe.tug/_seconds"] = 2.0
    # the last stage looks again with the same camera after the push and the tug, and moves nothing
    t["B/vision.gap@now/gap_mm_est"] = np.where(t["B/probe.tug/_y"], 0.0, short) + r.normal(0, vision_noise, n)
    t["B/vision.gap@now/_y"] = t["B/probe.tug/_y"]
    t["B/vision.gap@now/_seconds"] = 0.1
    return t


@pytest.fixture(scope="module")
def world():
    fit, cal, test = fake_zoo(4000, 1), fake_zoo(2000, 2), fake_zoo(2000, 3)
    scorers = e1.fit_checks(fit, "latch", "latched")
    thresholds = e1.calibrate_checks(scorers, cal, 0.95)
    table = e1.evaluate_checks(scorers, thresholds, test)
    return {"fit": fit, "cal": cal, "test": test, "scorers": scorers, "thresholds": thresholds, "table": table}


def test_every_check_with_features_gets_a_row(world):
    table = world["table"]
    for check in ("passive.skill_flag", "passive.pose_residual", "passive.click", "vision.gap@claim",
                  "probe.push_seat", "probe.tug", "fused.passive", "fused.all"):
        assert check in table, check
    assert "probe.wiggle" not in table                      # no wiggle columns in this table, so no row and no crash


def test_a_flag_that_is_always_1_accepts_every_bad_plug(world):
    row = world["table"]["passive.skill_flag"]
    assert row["far"]["rate"] == 1.0
    assert row["true_accept"]["rate"] == 1.0
    assert row["balanced_accuracy"] == pytest.approx(0.5)   # the do-nothing baseline scores at the base rate


def test_true_accept_is_matched_near_95_percent(world):
    for check in ("passive.pose_residual", "vision.gap@claim", "probe.tug"):
        assert 0.92 <= world["table"][check]["true_accept"]["rate"] <= 0.98, check


def test_a_probe_is_judged_against_the_truth_after_it_acted(world):
    test, table = world["test"], world["table"]
    bad_after_push = int((~test["A/probe.push_seat/_y"]).sum())
    bad_at_claim = int((~test["y0"]).sum())
    assert table["probe.push_seat"]["far"]["n"] == bad_after_push < bad_at_claim
    assert table["vision.gap@claim"]["far"]["n"] == bad_at_claim
    # the repair shows up as bad states turning good
    assert table["probe.push_seat"]["repair"]["k"] == bad_at_claim - bad_after_push
    assert table["probe.push_seat"]["harm"]["k"] == 0


def test_leaves_bad_uses_every_state_of_the_class_as_denominator(world):
    test, table = world["test"], world["table"]
    n1 = int((test["cls"] == "N1").sum())
    for check in ("probe.push_seat", "vision.gap@claim"):
        assert table[check]["leaves_bad_by_class"]["N1"]["n"] == n1
    assert table["probe.push_seat"]["by_class"]["N1"]["n"] < n1          # most N1 were repaired


def test_h1_is_held_when_vision_is_far_worse_than_the_probes(world):
    h1 = e1.h1(world["table"], has_n5=True)
    assert h1["status"] == "held"
    assert h1["best_probe"] == "probe.tug"
    assert h1["classes"] == ["N1", "N2", "N5"]


def _h1_table(vision_k: int, probe_k: int, n: int = 400, probe_left: int | None = None) -> dict:
    """Near-miss cells for 1 camera check and 2 probes. ``probe_left`` is how many bad near misses a probe left."""
    left = n if probe_left is None else probe_left
    table = {"vision.gap@claim": {"far_near_miss": e1._rate(vision_k, n), "leaves_bad_near_miss": e1._rate(vision_k, n)}}
    for probe in ("probe.push_seat", "probe.tug"):
        table[probe] = {"far_near_miss": e1._rate(probe_k, left), "leaves_bad_near_miss": e1._rate(probe_k, n)}
    return table


def test_h1_is_marked_on_intervals_of_leaves_bad():
    assert e1.h1(_h1_table(4, 4), True)["status"] == "failed"           # 1% against 1%
    assert e1.h1(_h1_table(28, 4), True)["status"] == "between"         # 7 times on the point, not clear of the intervals
    held = e1.h1(_h1_table(120, 4), True)                                # 30% against 1%
    assert held["status"] == "held" and held["ratio"] == pytest.approx(30.0)
    assert held["leaves_bad"]["vision"]["lo"] >= 5 * held["leaves_bad"]["probe"]["hi"]


def test_h1_with_a_probe_that_accepts_no_bad_plug():
    h1 = e1.h1(_h1_table(120, 0), True)
    assert h1["status"] == "held" and h1["ratio"] == float("inf")
    assert h1["ratio_conservative"] == pytest.approx(e1._rate(120, 400)["lo"] / e1._rate(0, 400)["hi"])
    # 2 camera false accepts against none: an infinite ratio under the first rule, no gap under this one
    h1 = e1.h1(_h1_table(2, 0), True)
    assert h1["status"] == "failed" and h1["ratio"] == float("inf")
    h1 = e1.h1(_h1_table(0, 0), True)
    assert h1["status"] == "failed" and np.isnan(h1["ratio"])           # 0 against 0 is no ratio


def test_h1_is_measured_when_the_probes_repair_every_near_miss():
    h1 = e1.h1(_h1_table(1, 0, probe_left=0), False)                     # nothing left for the probes to accept
    assert h1["status"] == "failed" and h1["classes"] == ["N1", "N2"]
    assert "far_ratio" not in h1                                         # the false-accept ratio is 0 of 0 and is not given
    assert e1.h1(_h1_table(1, 0, probe_left=3), False)["far_ratio"]["ratio"] == float("inf")


def test_h1_is_not_measured_without_both_sides():
    assert e1.h1({"probe.tug": {"far_near_miss": {"k": 0, "n": 10, "rate": 0.0, "lo": 0.0, "hi": 0.3},
                                "leaves_bad_near_miss": {"k": 0, "n": 10, "rate": 0.0, "lo": 0.0, "hi": 0.3}}}, True)["status"] == "not measured"


def test_e2_cascade_and_h2(world):
    cascade, scorers, neg, pos = e2.build(world["fit"], world["cal"], "latch", "latched", 0.02, 0.05)
    assert [s.name for s in cascade.stages] == ["passive", "push", "tug", "look"]
    assert sum(s.alpha_k for s in cascade.stages) <= 0.02 + 1e-12
    assert neg[0] > neg[1] >= neg[2]                         # the push repairs, so fewer bad states face later stages
    assert cascade.stages[0].alpha_k < cascade.stages[2].alpha_k        # so the early stage gets the smaller share
    result = evaluate(cascade, world["test"])
    resplits = e2.resplit(scorers, neg, pos, world["cal"], world["test"], "latch", "latched", 0.02, 0.05, n=30)
    h2 = e2.h2(result, resplits, 0.02)
    assert h2["status"] == "held"
    assert resplits["far"]["mean"] <= 0.02
    assert len(resplits["far_values"]) == 30
    sweep = e2.sweep(scorers, neg, pos, world["cal"], world["test"], "latch", "latched", [0.005, 0.02, 0.05], 0.05)
    assert [row["alpha"] for row in sweep] == [0.005, 0.02, 0.05]
    # a tighter promise does not buy fewer calls, up to the wobble of crossed thresholds on a few rows
    assert sweep[0]["abstain"]["rate"] >= sweep[-1]["abstain"]["rate"] - 0.01


def test_h2_fails_on_any_of_its_3_parts():
    ok = {"far_ucb": 0.025, "far_ucb_m": 500, "abstain": {"rate": 0.10}, "far": {"rate": 0.012, "k": 6, "n": 500}}
    rs = {"far": {"mean": 0.015, "se": 0.001}}
    assert e2.h2(ok, rs, 0.02)["status"] == "held"
    assert e2.h2(dict(ok, far_ucb=0.031), rs, 0.02)["status"] == "failed"
    assert e2.h2(dict(ok, abstain={"rate": 0.16}), rs, 0.02)["status"] == "failed"
    assert e2.h2(ok, {"far": {"mean": 0.023, "se": 0.001}}, 0.02)["status"] == "failed"
    within = e2.h2(ok, {"far": {"mean": 0.0215, "se": 0.001}}, 0.02)
    assert within["status"] == "held"                        # inside 2 standard errors
    assert within["parts"]["mean_far_at_most_alpha_outright"] is False          # and the report says it is not under alpha outright


def test_resplits_are_reproducible(world):
    _, scorers, neg, pos = e2.build(world["fit"], world["cal"], "latch", "latched", 0.02, 0.05)
    a = e2.resplit(scorers, neg, pos, world["cal"], world["test"], "latch", "latched", 0.02, 0.05, n=8)
    b = e2.resplit(scorers, neg, pos, world["cal"], world["test"], "latch", "latched", 0.02, 0.05, n=8)
    assert a["far_values"] == b["far_values"]


def test_the_exploratory_cascade_without_a_camera_uses_no_camera_column(world):
    from types import SimpleNamespace

    from avail.analysis import run

    res = run._no_camera(world["fit"], world["cal"], world["test"], "latch", "latched", SimpleNamespace(alpha=0.02, beta=0.05))
    assert res["stages"] == ["passive", "push", "tug"]
    assert sum(res["alpha_k"]) <= 0.02 + 1e-12
    assert res["far"]["n"] > 0 and 0.0 <= res["abstain"]["rate"] <= 1.0
    assert not any("vision" in source for stage in run.NO_CAMERA_STAGES for source in stage.sources)


# --------------------------------------------------------------------------- S4b: the hard-mode breakdowns
def _with_jams(t: pd.DataFrame, seed: int) -> pd.DataFrame:
    """The synthetic zoo with hard mode's columns: what made each short plug short, and the landed class."""
    r = np.random.default_rng(seed)
    t = t.copy()
    short = t["cls"].isin(["N1", "N2", "N3"]).to_numpy()
    kinds = np.where(short, r.choice(["burr", "wedge", "none"], size=len(t)), "none")
    t["spec_jam_kind"] = pd.Series(np.where(kinds == "none", None, kinds), dtype=object)
    landed = t["cls"].to_numpy().copy()
    landed[(t["cls"] == "N1").to_numpy() & (kinds == "burr")] = "N2"     # a burr N1 that sprang back to N2 depth
    t["landed_cls"] = landed
    return t


def test_jam_breakdowns_count_each_short_plug_once_by_its_landed_class(world):
    test = _with_jams(world["test"], 7)
    by = e1.by_jam(world["scorers"], world["thresholds"], test)
    short = test["landed_cls"].isin(["N1", "N2", "N3", "N4"])
    total = sum(by["passive.click"][k]["leaves_bad"]["n"] for k in e1.JAM_KINDS)
    assert total == int(short.sum())
    probes = e1.probe_by_jam(test)
    assert set(probes) == {"probe.push_seat", "probe.tug"}
    assert sum(probes["probe.push_seat"][k]["n"] for k in e1.JAM_KINDS) == int((short & ~test["y0"].astype(bool)).sum())


def test_the_hard_table_also_gives_every_rate_by_landed_class(world):
    test = _with_jams(world["test"], 8)
    table = e1.evaluate_checks(world["scorers"], world["thresholds"], test)
    row = table["vision.gap@claim"]
    assert "by_landed_class" in row and "leaves_bad_by_landed_class" in row
    landed_n = sum(c["n"] for c in row["leaves_bad_by_landed_class"].values())
    assert landed_n == len(test)
    assert row["leaves_bad_by_landed_class"]["N2"]["n"] == int((test["landed_cls"] == "N2").sum())


def test_the_cascade_by_jam_splits_the_short_plugs(world):
    from avail.analysis.run import _cascade_by_jam

    test = _with_jams(world["test"], 9)
    cascade, _, _, _ = e2.build(world["fit"], world["cal"], "latch", "latched", 0.02, 0.05)
    out = _cascade_by_jam(cascade, test)
    short = test["landed_cls"].isin(["N1", "N2", "N3", "N4"])
    assert sum(v["n"] for v in out.values()) == int(short.sum())
