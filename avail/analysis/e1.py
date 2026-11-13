"""E1, the evidence table: how often does each check accept a bad plug?

Protocol (docs/07_experiments.md, fixed before any test seed was read):

- A check is 1 evidence source run by itself on the untouched claim-time state (pass ``A`` of
  the zoo features), or a fused scorer.
- Its score is a logistic model over its own features, fit on the fit split.
- Its accept threshold is set on the calibration split so that it accepts 95% of the states
  where the claim is true.
- Its false-accept rate is then counted on the evaluation split: accepts among the states
  where the claim is false. A check that moves the robot is judged against the truth after
  it acted, so a push that seats a short plug and then accepts has accepted a good plug.
- Also reported: balanced accuracy at the threshold that maximizes it on calibration, the
  share of bad plugs of each injected class that leave accepted, cost, and for probes the
  harm rate, the repair rate and the worst real force.

Dated note, 2026-11-11, written before the test read. docs/07 defines the per-class rate
against truth after the check acts. For a probe that repairs most of a class, few bad states
of that class remain and the rate stands on a small n. The table therefore also carries
``leaves_bad``: the share of ALL states of a class that end up accepted and still bad. It has
the same denominator for every check.
"""

from __future__ import annotations

import math

import numpy as np

from ..evidence.fused import fit_scorer
from ..stats import cp_upper, wilson
from ..witness.calibrate import threshold_at_true_accept
from ..witness.cascade import fit_scorers, source_columns, stage_frame
from ..zoo.stages import PLUG_STAGES, SKIP_FEATURES

# check id -> (zoo pass, source key, what it is, moves the robot)
CHECKS: dict[str, tuple[str, str, str, bool]] = {
    "passive.skill_flag": ("A", "passive.skill_flag", "the skill's own flag", False),
    "passive.pose_residual": ("A", "passive.pose_residual", "tool pose against the taught pose", False),
    "passive.jaw_state": ("A", "passive.jaw_state", "gripper position and torque", False),
    "passive.click": ("A", "passive.click", "click in the insertion trace", False),
    "vision.gap@claim": ("A", "vision.gap@claim", "gap from the wrist camera", False),
    "probe.wiggle": ("A", "probe.wiggle", "lateral wiggle", True),
    "probe.push_seat": ("A", "probe.push_seat", "capped push", True),
    "probe.tug": ("A", "probe.tug", "capped tug", True),
    "instr.link": ("A", "instr.link", "link signal (instrumented)", False),
}
VISION_CHECKS = ("vision.gap@claim",)
PROBE_CHECKS = ("probe.push_seat", "probe.tug")
NEAR_MISS = ("N1", "N2", "N5")


def _rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n) if n else (float("nan"), float("nan"))
    return {"k": int(k), "n": int(n), "rate": (k / n) if n else float("nan"), "lo": lo, "hi": hi}


def _auc(scores: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(scores)
    s, y = scores[ok], y[ok]
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):                                  # midranks for ties
        tie = s == v
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _best_ba_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(scores)
    s, yy = scores[ok], y[ok]
    if yy.all() or not yy.any():
        return -math.inf
    best, best_tau = -1.0, -math.inf
    for tau in np.concatenate([[-math.inf], np.unique(s)]):
        acc = s > tau
        ba = 0.5 * ((acc & yy).sum() / yy.sum() + (~acc & ~yy).sum() / (~yy).sum())
        if ba > best + 1e-12:
            best, best_tau = ba, float(tau)
    return best_tau


def main_slice(table, predicate: str):
    """The rows whose claim is the connector's main predicate."""
    return table[table["claim_predicate"] == predicate].reset_index(drop=True)


def check_frame(table, check: str):
    zoo_pass, key, _, _ = CHECKS[check]
    import pandas as pd

    cols = source_columns(table, zoo_pass, key, SKIP_FEATURES)
    frame = pd.DataFrame({name: np.asarray(table[col], dtype=float) for name, col in cols.items()}, index=table.index)
    label_col = f"{zoo_pass}/{key}/_y"
    y = table[label_col] if label_col in table.columns else table["y0"]
    return frame, y


def fit_checks(fit, connector: str, predicate: str) -> dict:
    """Fit every check's scorer on the fit split."""
    scorers = {}
    for check in CHECKS:
        frame, y = check_frame(fit, check)
        if frame.shape[1] == 0:
            continue
        ok = y.notna()
        scorers[check] = fit_scorer(f"check.{connector}.{predicate}.{check}", frame[ok], list(frame.columns),
                                    np.asarray(y[ok]).astype(bool))
    stage_scorers, _, _ = fit_scorers(fit, connector, predicate, PLUG_STAGES, SKIP_FEATURES)
    scorers["fused.passive"] = stage_scorers[0]
    scorers["fused.all"] = stage_scorers[-1]
    return scorers


def _check_scores(table, check: str, scorer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scores, truth after the check acted, and truth at claim time."""
    y0 = np.asarray(table["y0"]).astype(bool)
    if check == "fused.passive":
        frame = stage_frame(table, PLUG_STAGES, 0, SKIP_FEATURES)
        return scorer.score(frame), y0, y0
    if check == "fused.all":
        frame = stage_frame(table, PLUG_STAGES, len(PLUG_STAGES) - 1, SKIP_FEATURES)
        y = np.asarray(table[PLUG_STAGES[-1].label].fillna(False)).astype(bool)
        return scorer.score(frame), y, y0
    frame, y = check_frame(table, check)
    return scorer.score(frame), np.asarray(y.fillna(False)).astype(bool), y0


def calibrate_checks(scorers: dict, cal, true_accept: float) -> dict:
    """Each check's accept threshold at matched true-accept, and its best balanced-accuracy threshold."""
    out = {}
    for check, scorer in scorers.items():
        s, y, _ = _check_scores(cal, check, scorer)
        out[check] = {"tau": threshold_at_true_accept(s, y, true_accept), "tau_ba": _best_ba_threshold(s, y),
                      "n_cal_true": int(y.sum()), "n_cal_false": int((~y).sum())}
    return out


def evaluate_checks(scorers: dict, thresholds: dict, table) -> dict:
    """The evidence table on an evaluation split."""
    cls = np.asarray(table["cls"])
    classes = sorted(set(cls))
    out = {}
    for check, scorer in scorers.items():
        s, y, y0 = _check_scores(table, check, scorer)
        tau = thresholds[check]["tau"]
        acc = np.isfinite(s) & (s > tau)
        acc_ba = np.isfinite(s) & (s > thresholds[check]["tau_ba"])
        n_true, n_false = int(y.sum()), int((~y).sum())
        row = {
            "what": CHECKS[check][2] if check in CHECKS else ("fused over passive signals" if check == "fused.passive"
                                                             else "fused over the whole cascade"),
            "tau": tau, "far": _rate(int((acc & ~y).sum()), n_false), "true_accept": _rate(int((acc & y).sum()), n_true),
            "far_ucb": cp_upper(int((acc & ~y).sum()), n_false) if n_false else float("nan"),
            "balanced_accuracy": 0.5 * (((acc_ba & y).sum() / n_true if n_true else float("nan"))
                                        + ((~acc_ba & ~y).sum() / n_false if n_false else float("nan"))),
            "auc": _auc(s, y),
            "leaves_bad": _rate(int((acc & ~y).sum()), len(y)),
            "by_class": {}, "leaves_bad_by_class": {},
        }
        for c in classes:
            sel = cls == c
            row["by_class"][c] = _rate(int((acc & ~y & sel).sum()), int((~y & sel).sum()))
            row["leaves_bad_by_class"][c] = _rate(int((acc & ~y & sel).sum()), int(sel.sum()))
        if "landed_cls" in table.columns:                    # S4b: the class each state is in by truth at claim time
            landed = np.asarray(table["landed_cls"])
            row["by_landed_class"], row["leaves_bad_by_landed_class"] = {}, {}
            for c in sorted(set(landed)):
                sel = landed == c
                row["by_landed_class"][c] = _rate(int((acc & ~y & sel).sum()), int((~y & sel).sum()))
                row["leaves_bad_by_landed_class"][c] = _rate(int((acc & ~y & sel).sum()), int(sel.sum()))
        near = np.isin(cls, NEAR_MISS)
        row["far_near_miss"] = _rate(int((acc & ~y & near).sum()), int((~y & near).sum()))
        row["leaves_bad_near_miss"] = _rate(int((acc & ~y & near).sum()), int(near.sum()))
        if check in CHECKS:
            zoo_pass, key, _, moves = CHECKS[check]
            sec = f"{zoo_pass}/{key}/_seconds"
            row["seconds_median"] = float(np.nanmedian(table[sec])) if sec in table.columns else float("nan")
            row["usd"] = 0.0
            if moves:
                row["harm"] = _rate(int((y0 & ~y).sum()), int(y0.sum()))
                row["repair"] = _rate(int((~y0 & y).sum()), int((~y0).sum()))
                for name in ("_true_peak_push_n", "_true_peak_pull_n", "_true_peak_raw_n", "_true_overrun_mm"):
                    col = f"{zoo_pass}/{key}/{name}"
                    if col in table.columns:
                        row["worst" + name] = float(np.nanmax(table[col]))
        out[check] = row
    return out


SHORT_CLASSES = ("N1", "N2", "N3", "N4")
JAM_KINDS = ("stuck joint", "burr", "wedge")


def _jam_kind(table) -> np.ndarray:
    return table["spec_jam_kind"].fillna("stuck joint").to_numpy() if "spec_jam_kind" in table.columns \
        else np.full(len(table), "stuck joint", dtype=object)


def _short_class(table) -> np.ndarray:
    """The class that decides whether a state is a short plug: landed by truth when the table has it.
    A burr-jammed plug can spring back to a deeper shortfall than injected, so the landed class is the fair one."""
    return np.asarray(table["landed_cls"] if "landed_cls" in table.columns else table["cls"])


def by_jam(scorers: dict, thresholds: dict, table) -> dict:
    """S4b: each check on the short plugs (landed N1 to N4), split by what made them short.

    Per kind: ``leaves_bad`` (accepted and still bad, over all states of the kind) and ``far``
    (accepted among the bad ones after the check acted).
    """
    cls = _short_class(table)
    kind = _jam_kind(table)
    short = np.isin(cls, SHORT_CLASSES)
    out = {}
    for check, scorer in scorers.items():
        s, y, _ = _check_scores(table, check, scorer)
        acc = np.isfinite(s) & (s > thresholds[check]["tau"])
        row = {}
        for k in JAM_KINDS:
            sel = short & (kind == k)
            row[k] = {"leaves_bad": _rate(int((acc & ~y & sel).sum()), int(sel.sum())),
                      "far": _rate(int((acc & ~y & sel).sum()), int((~y & sel).sum()))}
        out[check] = row
    return out


def probe_by_jam(table) -> dict:
    """S4b: each probe's repair rate on the short plugs (landed N1 to N4), by what made them short, from pass A."""
    cls = _short_class(table)
    kind = _jam_kind(table)
    y0 = np.asarray(table["y0"]).astype(bool)
    short = np.isin(cls, SHORT_CLASSES)
    out = {}
    for probe in ("probe.push_seat", "probe.tug", "probe.wiggle"):
        col = f"A/{probe}/_y"
        if col not in table.columns:
            continue
        y = np.asarray(table[col].fillna(False)).astype(bool)
        out[probe] = {k: _rate(int((~y0 & y & short & (kind == k)).sum()), int((~y0 & short & (kind == k)).sum()))
                      for k in JAM_KINDS}
    return out


def _ratio(a: float, b: float) -> float:
    """a over b. 0 against 0 is no ratio at all."""
    if b > 0:
        return a / b
    return math.inf if a > 0 else math.nan


def h1(table: dict, has_n5: bool) -> dict:
    """H1: on near misses the best vision check lets bad plugs through at least 5 times as often as push or tug.

    Marked as the dated note of docs/07 written after the dry run on fit seeds says: on
    ``leaves_bad``, the share of all near-miss states that end up accepted and still bad, which
    has the same denominator for every check. Held when the low end of the camera's 95% interval
    is at least 5 times the high end of the better probe's. Failed when the camera's rate is at
    most 2 times that high end. Between otherwise. A probe that repairs a near miss and then
    accepts it has accepted a good plug, so a probe that repairs every near miss scores 0 here
    where its false-accept rate would be 0 of 0.
    """
    out = {"classes": list(NEAR_MISS if has_n5 else NEAR_MISS[:2]),
           "vision": {c: table[c]["far_near_miss"] for c in VISION_CHECKS if c in table},
           "probes": {c: table[c]["far_near_miss"] for c in PROBE_CHECKS if c in table},
           "vision_leaves_bad": {c: table[c]["leaves_bad_near_miss"] for c in VISION_CHECKS if c in table},
           "probes_leaves_bad": {c: table[c]["leaves_bad_near_miss"] for c in PROBE_CHECKS if c in table}}
    vision = {c: r for c, r in out["vision_leaves_bad"].items() if r["n"] > 0}
    probes = {c: r for c, r in out["probes_leaves_bad"].items() if r["n"] > 0}
    if not vision or not probes:
        out["status"] = "not measured"
        return out
    best_vision = min(vision, key=lambda c: vision[c]["rate"])
    best_probe = min(probes, key=lambda c: probes[c]["rate"])
    v, p = vision[best_vision], probes[best_probe]
    out.update({"best_vision": best_vision, "best_probe": best_probe, "leaves_bad": {"vision": v, "probe": p},
                "ratio": _ratio(v["rate"], p["rate"]), "ratio_conservative": _ratio(v["lo"], p["hi"])})
    if v["lo"] >= 5 * p["hi"]:
        out["status"] = "held"
    elif v["rate"] <= 2 * p["hi"]:
        out["status"] = "failed"
    else:
        out["status"] = "between"
    # note 7's reading on the false-accept rates, each probe on the bad plugs it left
    far_v = out["vision"].get(best_vision, {})
    far_p = {c: r for c, r in out["probes"].items() if r["n"] > 0}
    if far_v.get("n") and far_p:
        best = min(far_p, key=lambda c: far_p[c]["rate"])
        out["far_ratio"] = {"probe": best, "ratio": _ratio(far_v["rate"], far_p[best]["rate"])}
    return out
