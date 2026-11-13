"""Thresholds from labeled evidence (docs/04_calibration_math.md, sections 2 to 4).

3 things live here:

- ``stage_thresholds``: the conformal accept and reject thresholds of one stage, from the
  calibration scores of the states where the predicate is false and where it is true.
- ``budget_split``: how the cascade's error budgets alpha and beta are shared between stages.
- ``threshold_at_true_accept``: the E1 rule, a threshold that accepts a set share of the
  true cases, so different checks can be compared at matched true-accept.

On the budget split. docs/04 section 4 bounds the cascade's false-accept rate by the sum of
the stages' alpha_k. That argument assumes a state that is bad stays bad. Here a push can
seat a short plug, so the pool of bad states shrinks from stage to stage, and a stage that
runs before the repair is calibrated against a larger pool than the one the cascade's rate
is measured on. Written out: P(false accept) <= sum_k alpha_k * P(bad at stage k), and the
false-accept rate divides that by P(bad at the decision). Sharing the budget in proportion
to min_k P(bad at stage k) / P(bad at stage k) makes the sum come out at alpha again. It is
exact when no probe turns a good plug bad, and the measured harm rate says how close that is.
The shares are computed on the fit split, as section 4 allows.
"""

from __future__ import annotations

import math

import numpy as np

from ..stats import conformal_accept_threshold, conformal_reject_threshold, min_negatives_to_accept, min_positives_to_reject


def stage_thresholds(scores, labels, alpha_k: float, beta_k: float) -> dict:
    """Conformal thresholds for one stage. ``labels`` is the truth after the stage's action.

    Returns tau_accept (accept when score > tau_accept, +inf when the stage can never
    accept), tau_reject (reject when score < tau_reject, -inf when it can never reject) and
    the counts they stand on.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    ok = np.isfinite(s)
    neg, pos = s[ok & ~y], s[ok & y]
    tau_accept = conformal_accept_threshold(neg, alpha_k) if (alpha_k > 0 and len(neg)) else math.inf
    tau_reject = conformal_reject_threshold(pos, beta_k) if (beta_k > 0 and len(pos)) else -math.inf
    return {
        "tau_accept": float(tau_accept), "tau_reject": float(tau_reject),
        "n_neg": int(len(neg)), "n_pos": int(len(pos)),
        "can_accept": bool(math.isfinite(tau_accept)), "can_reject": bool(math.isfinite(tau_reject)),
    }


def budget_split(neg_share: list[float], pos_share: list[float], alpha: float, beta: float,
                 n_neg: list[int] | None = None, n_pos: list[int] | None = None) -> tuple[list[float], list[float]]:
    """Share alpha and beta between stages so the union bound holds when a stage can repair.

    ``neg_share[k]`` is the share of states where the predicate is false after stage k acted,
    ``pos_share[k]`` the share where it is true. A stage with too few calibration states to
    ever accept (or reject) at its share gets none, and its share goes to the others.
    """
    def split(shares: list[float], total: float, counts: list[int] | None, fewest) -> list[float]:
        k = len(shares)
        live = [True] * k
        # The floor is taken over every stage, including one that gets no budget: a state can be
        # decided at any stage, and the rate is judged on the truth at the stage that decided it.
        # A stage dropped for lack of data still repairs, so the pool it leaves behind still counts.
        positive = [v for v in shares if v > 0]
        floor = min(positive) if positive else 0.0
        for _ in range(k + 1):
            m = sum(live)
            if m == 0:
                return [0.0] * k
            out = [(total / m) * (floor / shares[i]) if live[i] and shares[i] > 0 else 0.0 for i in range(k)]
            if counts is None:
                return out
            # A stage with too few states to ever act at its share is dropped, the worst one first,
            # and the split is redone so the others get its share.
            short = [(fewest(out[i]) / max(counts[i], 1), i) for i in range(k)
                     if live[i] and out[i] > 0 and counts[i] < fewest(out[i])]
            if not short:
                return out
            live[max(short)[1]] = False
        return [0.0] * k

    return (split(list(neg_share), alpha, n_neg, min_negatives_to_accept),
            split(list(pos_share), beta, n_pos, min_positives_to_reject))


def threshold_at_true_accept(scores, labels, true_accept: float = 0.95) -> float:
    """The largest threshold that still accepts at least ``true_accept`` of the true cases.

    The rule is accept when score > threshold. Candidate thresholds are the scores seen in the
    calibration data, true and false cases alike. With a score that takes few values (a flag),
    the share accepted can end up above ``true_accept``. The share actually reached is what a
    report must state. A true case with no score (NaN) counts as not accepted.
    """
    if not 0.0 < true_accept <= 1.0:
        raise ValueError(f"true_accept must be in (0, 1], got {true_accept}")
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    n_pos = int(y.sum())
    if n_pos == 0:
        return math.inf
    pos = np.sort(s[y & np.isfinite(s)])
    unscored = n_pos - len(pos)
    # reject at most floor(n * (1 - true_accept)) of the true cases
    allowed = int(math.floor(n_pos * (1.0 - true_accept) + 1e-9)) - unscored
    if allowed < 0:
        return -math.inf
    best = -math.inf
    for tau in np.unique(s[np.isfinite(s)]):
        rejected = int(np.searchsorted(pos, tau, side="right"))        # true cases with score <= tau are not accepted
        if rejected <= allowed:
            best = float(tau)
        else:
            break
    return best


def accept_rates(scores, labels, tau: float) -> dict:
    """Accepts among the true and among the false cases at a threshold."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    acc = np.isfinite(s) & (s > tau)
    return {
        "accepted_true": int((acc & y).sum()), "n_true": int(y.sum()),
        "accepted_false": int((acc & ~y).sum()), "n_false": int((~y).sum()),
    }
