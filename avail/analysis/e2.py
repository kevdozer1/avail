"""E2, a bound that holds: can a cascade promise a false-accept rate and keep it, and at what price?

Protocol (docs/07_experiments.md, fixed before any test seed was read):

- Fused scorers are fit on the fit split. Thresholds come from the calibration split at
  alpha = 0.02 and beta = 0.05, shared between stages as ``avail.witness.calibrate`` explains.
- On the evaluation split: false-accept rate, false-reject rate, abstain rate, mean
  verification time, and the one-sided 95% Clopper-Pearson bound on the false-accept rate.
- Then 200 random re-splits of calibration and evaluation rows together. Thresholds are
  recomputed on one half each time and the false-accept rate is read on the other half.
  Scorers stay as fit.
- The same at alpha 0.5%, 1% and 5%: abstain rate against alpha, the price list.

H2 holds when all 3 hold: the mean false-accept rate across re-splits is at most alpha
within 2 standard errors, the certified bound on the untouched evaluation split is at most
3%, and the abstain rate on it is at most 15%.

Dated note, 2026-11-11, written before the test read. "False-accept rate" here is accepts
among the states where the claim is false at the moment of the decision, after whatever the
stages that ran did to the plug. "Standard error" is the standard deviation of the re-split
means' estimate, sd / sqrt(200). The report also says whether the mean is at most alpha
outright, which is the stricter reading.
"""

from __future__ import annotations

import numpy as np

from ..witness.cascade import Cascade, calibrate_cascade, evaluate, fit_scorers
from ..zoo.stages import PLUG_STAGES, SKIP_FEATURES, STATION_ID

RESPLITS = 200
RESPLIT_SEED = 20261111


def build(fit, cal, connector: str, predicate: str, alpha: float, beta: float, split_id: str = "zoo_calibration"):
    """Fit the scorers and calibrate the cascade. Returns (cascade, scorers, neg_share, pos_share)."""
    scorers, neg_share, pos_share = fit_scorers(fit, connector, predicate, PLUG_STAGES, SKIP_FEATURES)
    cascade = calibrate_cascade(scorers, neg_share, pos_share, cal, connector, predicate, alpha, beta, PLUG_STAGES,
                                station_id=STATION_ID, skip_features=SKIP_FEATURES, split_id=split_id)
    return cascade, scorers, neg_share, pos_share


def resplit(scorers, neg_share, pos_share, cal, evaluation, connector: str, predicate: str, alpha: float, beta: float,
            n: int = RESPLITS, seed: int = RESPLIT_SEED, budget: str = "weighted") -> dict:
    """Recalibrate on random halves of calibration plus evaluation rows and read the rates on the other halves."""
    import pandas as pd

    pool = pd.concat([cal, evaluation], ignore_index=True)
    rng = np.random.default_rng(seed)
    far, frr, abstain, m_neg = [], [], [], []
    for _ in range(n):
        perm = rng.permutation(len(pool))
        half = len(pool) // 2
        a, b = pool.iloc[perm[:half]], pool.iloc[perm[half:]]
        cascade = calibrate_cascade(scorers, neg_share, pos_share, a, connector, predicate, alpha, beta, PLUG_STAGES,
                                    station_id=STATION_ID, skip_features=SKIP_FEATURES, budget=budget)
        res = evaluate(cascade, b)
        far.append(res["far"]["rate"])
        frr.append(res["frr"]["rate"])
        abstain.append(res["abstain"]["rate"])
        m_neg.append(res["far"]["n"])
    far, frr, abstain = np.array(far), np.array(frr), np.array(abstain)

    def summary(x: np.ndarray) -> dict:
        return {"mean": float(np.nanmean(x)), "sd": float(np.nanstd(x, ddof=1)), "se": float(np.nanstd(x, ddof=1) / np.sqrt(len(x))),
                "p05": float(np.nanquantile(x, 0.05)), "p50": float(np.nanquantile(x, 0.5)), "p95": float(np.nanquantile(x, 0.95)),
                "max": float(np.nanmax(x))}

    return {"n_resplits": n, "alpha": alpha, "beta": beta, "far": summary(far), "frr": summary(frr),
            "abstain": summary(abstain), "median_bad_states_per_half": float(np.median(m_neg)),
            "far_values": [float(v) for v in far]}


def sweep(scorers, neg_share, pos_share, cal, evaluation, connector: str, predicate: str, alphas: list[float],
          beta: float) -> list[dict]:
    """Abstain rate, false-accept rate and time against alpha: the price list."""
    out = []
    for alpha in alphas:
        cascade = calibrate_cascade(scorers, neg_share, pos_share, cal, connector, predicate, alpha, beta, PLUG_STAGES,
                                    station_id=STATION_ID, skip_features=SKIP_FEATURES)
        res = evaluate(cascade, evaluation)
        out.append({"alpha": alpha, "abstain": res["abstain"], "far": res["far"], "far_ucb": res["far_ucb"],
                    "frr": res["frr"], "mean_seconds": res["mean_seconds"],
                    "stages_that_can_accept": [s.name for s in cascade.stages if np.isfinite(s.tau_accept)]})
    return out


def h2(result: dict, resplits: dict, alpha: float, ucb_gate: float = 0.03, abstain_gate: float = 0.15) -> dict:
    """Mark H2 from the evaluation-split result and the re-split distribution."""
    mean, se = resplits["far"]["mean"], resplits["far"]["se"]
    parts = {
        "mean_far_within_2_se": bool(mean <= alpha + 2 * se),
        "mean_far_at_most_alpha_outright": bool(mean <= alpha),
        "certified_bound_at_most_3pct": bool(result["far_ucb"] <= ucb_gate),
        "abstain_at_most_15pct": bool(result["abstain"]["rate"] <= abstain_gate),
    }
    held = parts["mean_far_within_2_se"] and parts["certified_bound_at_most_3pct"] and parts["abstain_at_most_15pct"]
    return {"status": "held" if held else "failed", "parts": parts, "mean_far": mean, "se": se,
            "far_ucb": result["far_ucb"], "far_ucb_m": result["far_ucb_m"], "abstain": result["abstain"]["rate"],
            "far_point": result["far"]["rate"], "far_k": result["far"]["k"], "far_n": result["far"]["n"]}


def stage_report(cascade: Cascade) -> list[dict]:
    """Per stage: thresholds, budgets and what the scorer leans on."""
    out = []
    for stage in cascade.stages:
        out.append({"stage": stage.name, "sources": list(stage.sources), "alpha_k": stage.alpha_k, "beta_k": stage.beta_k,
                    "tau_accept": None if not np.isfinite(stage.tau_accept) else float(stage.tau_accept),
                    "tau_reject": None if not np.isfinite(stage.tau_reject) else float(stage.tau_reject),
                    "n_neg": stage.n_neg, "n_pos": stage.n_pos, "features": len(stage.scorer.features),
                    "top_terms": stage.scorer.top_terms(5)})
    return out
