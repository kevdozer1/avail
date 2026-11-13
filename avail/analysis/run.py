"""Run E1 and E2 for a connector, from zoo tables on disk.

2 modes, same code:

- ``rehearsal``: the calibration split stands in for the test split. Thresholds come from
  calibration seeds 14,000 to 14,999 and the numbers are read on 15,000 to 15,999. This is
  where bugs get fixed (sessions/S4.md step 5).
- ``final``: thresholds from the whole calibration split, numbers from the test split. The
  test features must already be on disk. Making them is the single read of the test seeds,
  done by ``avail zoo features --split test --final-run E1E2``. This module never simulates.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..config import config_hash, load_experiment
from ..witness.cascade import StageDef, calibrate_cascade, evaluate, fit_scorers
from ..zoo import store
from ..zoo.stages import PLUG_STAGES, SKIP_FEATURES, STATION_ID
from . import e1, e2

MODES = ("rehearsal", "final")

# Exploratory, not in docs/07: the E2 cascade with the camera taken out of every stage. The
# simulated camera sees clean renders, so this asks what the probes buy where a camera cannot
# be trusted. It marks no hypothesis.
NO_CAMERA_STAGES = (
    StageDef("passive", ("passive.skill_flag", "passive.pose_residual", "passive.jaw_state", "passive.click"), "A", "y0"),
    StageDef("push", ("probe.push_seat",), "B", "B/probe.push_seat/_y"),
    StageDef("tug", ("probe.tug",), "B", "B/probe.tug/_y"),
)


def primary_predicate(connector: str) -> str:
    return "latched" if connector == "latch" else "seated"


PROFILE_SUFFIX = {"easy": "", "hard": "_hard",      # S4b: hard mode's splits are fit_hard, calibration_hard, test_hard
                  "hard2": "_hard2"}                 # S4c: hard mode v2's are fit_hard2, calibration_hard2, test_hard2
RESULT_FOLDER = {"easy": "e1e2", "hard": "e1e2b", "hard2": "e1e2c"}     # runs/<folder>_<mode>/
PROFILE_LABEL = {"easy": "", "hard": ", hard mode", "hard2": ", hard mode v2"}


def load_split(connector: str, split: str, root=None):
    """Features joined with the states table for one split."""
    feats = store.load(connector, split, "features", root)
    states = store.load(connector, split, "states", root)
    store.check_seeds_in_split(feats["seed"], split)       # a stray dev state never enters an analysis table
    keep = ["seed", "true_shortfall_mm", "skill_flag", "scene_plug_color", "scene_board_mat", "true_tilt_deg",
            "spec_jam_kind", "hard_lot_factor"]
    return feats.merge(states[[c for c in keep if c in states.columns]], on="seed", how="left")


def tables_for(connector: str, mode: str, root=None, profile: str = "easy") -> dict:
    """The fit, calibration and evaluation tables of a mode, and what each one is."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if profile not in PROFILE_SUFFIX:
        raise ValueError(f"profile must be one of {tuple(PROFILE_SUFFIX)}, got {profile!r}")
    sfx = PROFILE_SUFFIX[profile]
    fit = load_split(connector, "fit" + sfx, root)
    cal = load_split(connector, "calibration" + sfx, root)
    if mode == "rehearsal":
        seeds = np.asarray(cal["seed"])
        mid = (int(seeds.min()) + int(seeds.max()) + 1) // 2
        return {"fit": fit, "cal": cal[seeds < mid].reset_index(drop=True), "eval": cal[seeds >= mid].reset_index(drop=True),
                "eval_name": f"calibration seeds {mid} and up, standing in for test",
                "cal_name": f"calibration seeds below {mid}", "split_id": f"zoo_calibration{sfx}_first_half"}
    test = load_split(connector, "test" + sfx, root)
    return {"fit": fit, "cal": cal, "eval": test, "eval_name": "test seeds", "cal_name": "calibration seeds",
            "split_id": f"zoo_calibration{sfx}"}


def _counts(table) -> dict:
    return {"rows": int(len(table)), "true": int(np.asarray(table["y0"]).astype(bool).sum()),
            "classes": {k: int(v) for k, v in table["cls"].value_counts().sort_index().items()}}


def run_connector(connector: str, mode: str, root=None, profile: str = "easy") -> dict:
    """E1 and E2 for one connector. Everything a report needs, as plain data.

    ``profile`` "hard" runs E1b and E2b (S4b) on the hard-mode splits with the same code.
    """
    exp = load_experiment()
    bounds = exp.bounds
    predicate = primary_predicate(connector)
    t0 = time.perf_counter()
    tabs = tables_for(connector, mode, root, profile)
    fit, cal, ev = (e1.main_slice(tabs[k], predicate) for k in ("fit", "cal", "eval"))
    out: dict = {
        "connector": connector, "predicate": predicate, "mode": mode, "profile": profile, "sim_or_real": "sim",
        "evaluated_on": tabs["eval_name"], "calibrated_on": tabs["cal_name"], "config_hash": config_hash(),
        "counts": {"fit": _counts(fit), "calibration": _counts(cal), "evaluation": _counts(ev)},
        "alpha": bounds.alpha, "beta": bounds.beta, "true_accept_for_e1": bounds.true_accept_for_e1,
    }
    for name, table in (("fit", tabs["fit"]), ("calibration", tabs["cal"]), ("evaluation", tabs["eval"])):
        errors = {c: int(table[c].notna().sum()) for c in table.columns if c.endswith("/_error") and table[c].notna().any()}
        out["counts"][name]["source_errors"] = errors
        if "rebuild_matches" in table.columns:
            out["counts"][name]["rebuild_matches"] = float(table["rebuild_matches"].mean())

    # E1
    scorers = e1.fit_checks(fit, connector, predicate)
    thresholds = e1.calibrate_checks(scorers, cal, bounds.true_accept_for_e1)
    table = e1.evaluate_checks(scorers, thresholds, ev)
    out["e1"] = {"table": table, "thresholds": thresholds, "h1": e1.h1(table, connector == "latch"),
                 "scorers": {k: {"features": v.features, "constant": v.constant, "top_terms": v.top_terms(4)} for k, v in scorers.items()}}

    # E2
    cascade, stage_scorers, neg_share, pos_share = e2.build(fit, cal, connector, predicate, bounds.alpha, bounds.beta,
                                                            split_id=tabs["split_id"])
    result = evaluate(cascade, ev)
    resplits = e2.resplit(stage_scorers, neg_share, pos_share, cal, ev, connector, predicate, bounds.alpha, bounds.beta)
    sweep = e2.sweep(stage_scorers, neg_share, pos_share, cal, ev, connector, predicate, list(bounds.alpha_sweep), bounds.beta)
    equal = calibrate_cascade(stage_scorers, neg_share, pos_share, cal, connector, predicate, bounds.alpha, bounds.beta,
                              PLUG_STAGES, station_id=STATION_ID, skip_features=SKIP_FEATURES, budget="equal")
    equal_result = evaluate(equal, ev)
    equal_resplits = e2.resplit(stage_scorers, neg_share, pos_share, cal, ev, connector, predicate, bounds.alpha, bounds.beta,
                                n=60, budget="equal")
    out["e2"] = {
        "result": result, "resplits": {k: v for k, v in resplits.items()}, "sweep": sweep,
        "h2": e2.h2(result, resplits, bounds.alpha), "stages": e2.stage_report(cascade),
        "neg_share_fit": neg_share, "pos_share_fit": pos_share,
        "equal_budget": {"far": equal_result["far"], "abstain": equal_result["abstain"], "far_ucb": equal_result["far_ucb"],
                         "alpha_k": [s.alpha_k for s in equal.stages],
                         "resplit_far": {k: equal_resplits["far"][k] for k in ("mean", "se", "sd")},
                         "resplit_abstain_mean": equal_resplits["abstain"]["mean"], "n_resplits": equal_resplits["n_resplits"]},
        "weighted_alpha_k": [s.alpha_k for s in cascade.stages],
    }
    cascade.calibration.update({"far_point": result["far"]["rate"], "far_ucb95": result["far_ucb"],
                                "frr_point": result["frr"]["rate"]})
    out["e2"]["no_camera"] = _no_camera(fit, cal, ev, connector, predicate, bounds)
    out["e2"]["click_settles"] = click_settles(cascade, fit, cal, ev, connector, predicate, bounds)

    # the release slice: did the fingers let go?
    out["release"] = _release_slice(tabs, exp.bounds.true_accept_for_e1)
    if "spec_jam_kind" in ev.columns:                     # S4b: each check on jammed short plugs and on stuck-joint ones
        out["e1"]["by_jam"] = e1.by_jam(scorers, thresholds, ev)
        out["e1"]["probe_by_jam"] = e1.probe_by_jam(ev)
        out["e2"]["result_by_jam"] = _cascade_by_jam(cascade, ev)
    out["landed"] = {"evaluation": _landed(ev)}
    out["seconds"] = round(time.perf_counter() - t0, 1)
    out["_cascade"] = cascade
    out["_checks"] = {k: {"scorer": v.to_dict(), "tau": thresholds[k]["tau"]} for k, v in scorers.items()}
    return out


def _landed(table) -> dict:
    """How many states of each injected class landed in each class by truth."""
    out: dict = {}
    if "landed_cls" not in table.columns:
        return out
    for cls, part in table.groupby("cls"):
        out[str(cls)] = {str(k): int(v) for k, v in part["landed_cls"].value_counts().sort_index().items()}
    return out


def _cascade_by_jam(cascade, ev) -> dict:
    """The cascade's decisions on the short classes, split by what made the plug short (hard mode)."""
    from ..stats import ACCEPT, ABSTAIN

    dec = cascade.decide_table(ev)
    kind = ev["spec_jam_kind"].fillna("stuck joint").to_numpy()
    cls = (ev["landed_cls"] if "landed_cls" in ev.columns else ev["cls"]).to_numpy()
    y = dec["y_dec"].to_numpy().astype(bool)
    acc = (dec["decision"] == ACCEPT).to_numpy()
    abst = (dec["decision"] == ABSTAIN).to_numpy()
    out = {}
    for k in ("stuck joint", "burr", "wedge"):
        sel = np.isin(cls, ["N1", "N2", "N3", "N4"]) & (kind == k)
        out[k] = {"n": int(sel.sum()), "accepted_bad": int((acc & ~y & sel).sum()), "abstain": int((abst & sel).sum()),
                  "bad_at_decision": int((~y & sel).sum())}
    return out


def click_settles(cascade, fit, cal, ev, connector: str, predicate: str, bounds) -> dict:
    """S4c: the share of claims the click settles. Written down before the hard-mode v2 test read.

    A claim is settled by the click when the cascade decides it at its passive stage and the same
    cascade with the click taken out of that stage (refit on the fit split, recalibrated on the
    calibration split, same alpha and beta) does not decide it the same way at its passive stage:
    it then goes on to a probe, ends with a person, or gets the other decision. Denominator: every
    claim of the main slice. Also given: the share the passive stage decides, and among those the
    share where the click fired (the insertion trace showed a jump past the commanded motion at the
    end of a hard stop) and the decision was accept, or did not fire and the decision was reject.
    """
    from ..stats import ABSTAIN, ACCEPT, wilson

    no_click = tuple(StageDef(d.name, tuple(s for s in d.sources if s != "passive.click"), d.zoo_pass, d.label)
                     for d in PLUG_STAGES)
    scorers, neg_share, pos_share = fit_scorers(fit, connector, predicate, no_click, SKIP_FEATURES)
    other = calibrate_cascade(scorers, neg_share, pos_share, cal, connector, predicate, bounds.alpha, bounds.beta,
                              stage_defs=no_click, station_id=STATION_ID, skip_features=SKIP_FEATURES)
    full, alt = cascade.decide_table(ev), other.decide_table(ev)
    n = int(len(ev))
    at_passive = (full["stage_reached"] == 1) & (full["decision"] != ABSTAIN)
    same = (alt["stage_reached"] == 1) & (alt["decision"] == full["decision"])
    settled = at_passive & ~same
    alt_passive = (alt["stage_reached"] == 1) & (alt["decision"] != ABSTAIN)
    left_open = at_passive & ~alt_passive                  # without the click the passive stage leaves it to a probe
    flipped = at_passive & alt_passive & (alt["decision"] != full["decision"])
    reverse = ~at_passive & alt_passive                    # decided at passive only once the click is gone
    y = full["y_dec"].to_numpy().astype(bool)
    acc_full = (full["decision"] == ACCEPT).to_numpy()
    s = settled.to_numpy()
    names = [st.name for st in other.stages]
    ends = {}
    for k, name in enumerate(names, start=1):
        sel = left_open.to_numpy() & (alt["stage_reached"].to_numpy() == k) & (alt["decision"] != ABSTAIN).to_numpy()
        ends[name] = int(sel.sum())
    ends["person"] = int((left_open & (alt["decision"] == ABSTAIN)).sum())
    fired = ev.get("A/passive.click/clicked")
    agree = None
    if fired is not None:
        fired = np.asarray(fired.fillna(0.0), dtype=float) > 0.5
        acc = (full["decision"] == ACCEPT).to_numpy()
        agree_mask = at_passive.to_numpy() & ((fired & acc) | (~fired & ~acc))
        agree = _cell(int(agree_mask.sum()), int(at_passive.sum()), wilson)
    no_click_result = evaluate(other, ev)
    return {
        "definition": "decided at the passive stage, and not decided the same way there once the click is taken out",
        "settled": _cell(int(settled.sum()), n, wilson),
        "decided_at_passive": _cell(int(at_passive.sum()), n, wilson),
        "click_agrees_with_passive_decision": agree,
        "settled_split": {"left_open": int(left_open.sum()), "flipped": int(flipped.sum()), "reverse": int(reverse.sum()),
                          "settled_accepts": int((s & acc_full).sum()), "settled_rejects": int((s & ~acc_full).sum()),
                          "settled_accepts_wrong": int((s & acc_full & ~y).sum()),
                          "settled_rejects_wrong": int((s & ~acc_full & y).sum()),
                          "left_open_end_without_click": ends},
        "no_click_cascade": {"far": no_click_result["far"], "far_ucb": no_click_result["far_ucb"],
                             "abstain": no_click_result["abstain"], "frr": no_click_result["frr"],
                             "decided_by_stage": no_click_result["decided_by_stage"]},
    }


def _cell(k: int, n: int, wilson) -> dict:
    lo, hi = wilson(k, n) if n else (float("nan"), float("nan"))
    return {"k": k, "n": n, "rate": (k / n) if n else float("nan"), "lo": lo, "hi": hi}


def _no_camera(fit, cal, ev, connector: str, predicate: str, bounds) -> dict:
    """The E2 cascade with no camera stage and no camera feature. Exploratory."""
    scorers, neg_share, pos_share = fit_scorers(fit, connector, predicate, NO_CAMERA_STAGES, SKIP_FEATURES)
    cascade = calibrate_cascade(scorers, neg_share, pos_share, cal, connector, predicate, bounds.alpha, bounds.beta,
                                stage_defs=NO_CAMERA_STAGES, station_id=STATION_ID, skip_features=SKIP_FEATURES)
    res = evaluate(cascade, ev)
    return {"stages": [d.name for d in NO_CAMERA_STAGES], "alpha_k": [s.alpha_k for s in cascade.stages],
            "far": res["far"], "far_ucb": res["far_ucb"], "far_ucb_m": res["far_ucb_m"], "frr": res["frr"],
            "abstain": res["abstain"], "mean_seconds": res["mean_seconds"], "decided_by_stage": res["decided_by_stage"],
            "per_class": res.get("per_class", {})}


def _release_slice(tabs: dict, true_accept: float) -> dict:
    from ..evidence.fused import fit_scorer
    from ..witness.calibrate import threshold_at_true_accept
    from ..witness.cascade import source_columns

    fit, cal, ev = (t[t["claim_predicate"] == "released_stable"].reset_index(drop=True) for t in (tabs["fit"], tabs["cal"], tabs["eval"]))
    out = {"n_fit": int(len(fit)), "n_cal": int(len(cal)), "n_eval": int(len(ev))}
    if len(fit) < 20 or len(cal) < 20 or len(ev) < 10:
        out["status"] = "too few states"
        return out
    cols = source_columns(fit, "A", "passive.jaw_state", SKIP_FEATURES)
    if not cols:
        out["status"] = "no jaw_state features"
        return out
    import pandas as pd

    def frame(t):
        return pd.DataFrame({name: np.asarray(t[col], dtype=float) for name, col in cols.items()})

    scorer = fit_scorer("check.release.jaw_state", frame(fit), list(cols), np.asarray(fit["y0"]).astype(bool))
    tau = threshold_at_true_accept(scorer.score(frame(cal)), np.asarray(cal["y0"]).astype(bool), true_accept)
    s, y = scorer.score(frame(ev)), np.asarray(ev["y0"]).astype(bool)
    acc = s > tau
    out.update({"status": "measured", "check": "passive.jaw_state",
                "far": e1._rate(int((acc & ~y).sum()), int((~y).sum())),
                "true_accept": e1._rate(int((acc & y).sum()), int(y.sum()))})
    return out


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None if value != value else ("inf" if value > 0 else "-inf")
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def save_result(result: dict, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"e1e2_{result['connector']}.json"
    path.write_text(json.dumps(_jsonable(result), indent=1), encoding="utf-8", newline="\n")
    return path
