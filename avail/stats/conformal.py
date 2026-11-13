"""Thresholds with a stated error bound, and the 3-way decision they drive.

Spec: ``docs/04_calibration_math.md`` section 2.

The accept threshold comes from the scores of calibration negatives, the states
where the predicate is false. With ``n`` negatives and a level ``alpha`` it is
the k-th smallest score, ``k = ceil((n + 1) * (1 - alpha))``, and the rule is
accept when ``score > tau_accept``. For a new negative drawn like the
calibration negatives, the chance of a false accept is at most ``alpha``. This
is class-conditional split conformal calibration.

The reject threshold mirrors it on positives. With ``m`` positives and a level
``beta`` it is the j-th smallest score, ``j = floor((m + 1) * beta)``, and the
rule is reject when ``score < tau_reject``.

What the bound means. It holds on average over calibration sets. It assumes new
states look like the calibration states. It says nothing after a camera is
bumped, which is what the control charts in ``spc.py`` watch for.

A higher score always means the predicate is more likely true.
"""

from __future__ import annotations

import math

import numpy as np

from . import _validate as check

__all__ = [
    "ACCEPT",
    "REJECT",
    "ABSTAIN",
    "conformal_accept_index",
    "conformal_accept_threshold",
    "conformal_reject_index",
    "conformal_reject_threshold",
    "decide",
    "decide_many",
    "min_negatives_to_accept",
    "min_positives_to_reject",
]

ACCEPT = "accept"
REJECT = "reject"
ABSTAIN = "abstain"

# (n + 1) * (1 - alpha) is often a whole number on paper and a hair off in floating point.
# 1000 * (1 - 0.059) evaluates to 941.0000000000001, and a plain ceil turns that into rank 942.
# 3000 * 0.009 evaluates to 26.999999999999996, and a plain floor turns that into rank 26.
# A product within this relative distance of a whole number is taken to be that whole number.
_SNAP = 1e-12


def _whole_if_close(x: float) -> int | None:
    nearest = round(x)
    if abs(x - nearest) <= _SNAP * max(1.0, abs(x)):
        return int(nearest)
    return None


def _ceil(x: float) -> int:
    whole = _whole_if_close(x)
    return math.ceil(x) if whole is None else whole


def _floor(x: float) -> int:
    whole = _whole_if_close(x)
    return math.floor(x) if whole is None else whole


def _accept_index(n: int, alpha: float) -> int | None:
    k = max(1, _ceil((n + 1) * (1.0 - alpha)))
    return k if k <= n else None


def _reject_index(m: int, beta: float) -> int | None:
    j = min(m, _floor((m + 1) * beta))
    return j if j >= 1 else None


def conformal_accept_index(n: int, alpha: float) -> int | None:
    """The rank of the calibration negative whose score becomes the accept threshold.

    ``k = ceil((n + 1) * (1 - alpha))``, counting from 1 at the smallest score.
    None means ``k > n``: the stage never accepts, because ``n`` negatives are
    too few for this ``alpha``. A stage can accept once ``n >= 1 / alpha - 1``.

    ``alpha = 0`` is allowed and always gives None. ``n = 0`` gives None too.

    Example: ``conformal_accept_index(99, 0.02)`` is 98, and
    ``conformal_accept_index(40, 0.02)`` is None.
    """
    n = check.count("conformal_accept_index", "n", n, 0, "Pass the number of calibration negatives, 0 or more.")
    alpha = check.fraction("conformal_accept_index", "alpha", alpha, low_ok=True, example="0.02")
    return _accept_index(n, alpha)


def conformal_reject_index(m: int, beta: float) -> int | None:
    """The rank of the calibration positive whose score becomes the reject threshold.

    ``j = floor((m + 1) * beta)``, counting from 1 at the smallest score. None
    means ``j = 0``: the stage never rejects, because ``m`` positives are too
    few for this ``beta``. A stage can reject once ``m >= 1 / beta - 1``.

    ``beta = 0`` is allowed and always gives None. ``m = 0`` gives None too.

    Example: ``conformal_reject_index(99, 0.05)`` is 5, and
    ``conformal_reject_index(18, 0.05)`` is None.
    """
    m = check.count("conformal_reject_index", "m", m, 0, "Pass the number of calibration positives, 0 or more.")
    beta = check.fraction("conformal_reject_index", "beta", beta, low_ok=True, example="0.05")
    return _reject_index(m, beta)


def _score_array(func: str, name: str, values) -> np.ndarray:
    """``values`` as a 1-D float array. None becomes NaN. Text and flags are refused."""
    try:
        raw = np.asarray(values)
    except Exception:
        raw = None
    if raw is None or raw.ndim != 1:
        shape = "an unreadable value" if raw is None else f"shape {raw.shape}"
        raise ValueError(
            f"{func}: {name} must be a flat list or a 1-D array of scores, got {shape}. "
            "Pass the scores of 1 stage, 1 number per state."
        )
    if raw.dtype.kind in "iuf":
        return raw.astype(float)
    if raw.dtype.kind == "O":
        out = np.empty(raw.shape, dtype=float)
        for i, value in enumerate(raw):
            if value is None:
                out[i] = math.nan
            elif check.is_real(value):
                out[i] = float(value)
            else:
                raise ValueError(
                    f"{func}: {name}[{i}] must be a number, got {value!r}. "
                    "Pass numeric scores. Use NaN for a missing score."
                )
        return out
    raise ValueError(
        f"{func}: {name} must hold numbers, got dtype {raw.dtype}. "
        "Pass numeric scores. Use NaN for a missing score."
    )


def _calibration_scores(func: str, name: str, values) -> np.ndarray:
    scores = _score_array(func, name, values)
    return scores[~np.isnan(scores)]


def conformal_accept_threshold(neg_scores, alpha: float) -> float:
    """The accept threshold ``tau_accept`` from the scores of calibration negatives.

    It is the k-th smallest score with ``k = conformal_accept_index(n, alpha)``.
    The rule is accept when ``score > tau_accept``. Then a new negative drawn
    like these is falsely accepted with probability at most ``alpha``, on
    average over calibration sets.

    Returns ``inf`` when the stage never accepts: too few negatives for this
    ``alpha``, or none at all.

    NaN scores are left out. A NaN score abstains at decision time, so it can
    never be a false accept and the bound still holds on the scores that remain.
    """
    alpha = check.fraction("conformal_accept_threshold", "alpha", alpha, low_ok=True, example="0.02")
    scores = _calibration_scores("conformal_accept_threshold", "neg_scores", neg_scores)
    k = _accept_index(scores.size, alpha)
    if k is None:
        return math.inf
    return float(np.partition(scores, k - 1)[k - 1])


def conformal_reject_threshold(pos_scores, beta: float) -> float:
    """The reject threshold ``tau_reject`` from the scores of calibration positives.

    It is the j-th smallest score with ``j = conformal_reject_index(m, beta)``.
    The rule is reject when ``score < tau_reject``. Then a new positive drawn
    like these is falsely rejected with probability at most ``beta``, on
    average over calibration sets.

    Returns ``-inf`` when the stage never rejects: too few positives for this
    ``beta``, or none at all.

    NaN scores are left out, for the same reason as in ``conformal_accept_threshold``.
    """
    beta = check.fraction("conformal_reject_threshold", "beta", beta, low_ok=True, example="0.05")
    scores = _calibration_scores("conformal_reject_threshold", "pos_scores", pos_scores)
    j = _reject_index(scores.size, beta)
    if j is None:
        return -math.inf
    return float(np.partition(scores, j - 1)[j - 1])


def _threshold(func: str, name: str, value) -> float:
    number = float(value) if check.is_real(value) else math.nan
    if math.isnan(number):
        raise ValueError(
            f"{func}: {name} must be a number, got {value!r}. Pass the value from "
            "conformal_accept_threshold or conformal_reject_threshold. Use inf for a stage that "
            "never accepts and -inf for a stage that never rejects."
        )
    return number


def decide(score, tau_reject: float, tau_accept: float) -> str:
    """The 3-way decision of 1 stage: ``"accept"``, ``"reject"`` or ``"abstain"``.

    - accept when ``score > tau_accept``
    - reject when ``score < tau_reject``
    - abstain otherwise

    Ties abstain. A score equal to ``tau_accept`` is not accepted and a score
    equal to ``tau_reject`` is not rejected.

    If the 2 thresholds cross (``tau_reject > tau_accept``), the stage accepts
    only above the larger one and rejects only below the smaller one. The
    region between them abstains, and both error bounds still hold.

    A score of NaN or None abstains: no evidence, no verdict. ``tau_accept = inf``
    is a stage that never accepts. ``tau_reject = -inf`` is a stage that never
    rejects. The score must be a number. Text and flags are refused, so a
    model's words can never be read as a score.
    """
    low = _threshold("decide", "tau_reject", tau_reject)
    high = _threshold("decide", "tau_accept", tau_accept)
    if low > high:
        low, high = high, low
    if score is None:
        return ABSTAIN
    if not check.is_real(score):
        raise ValueError(
            f"decide: score must be a number, got {score!r}. Pass the stage score as a float. "
            "Use NaN or None for a missing score, which abstains."
        )
    value = float(score)
    if value > high:                # False for NaN, so a NaN score falls through to abstain
        return ACCEPT
    if value < low:
        return REJECT
    return ABSTAIN


def decide_many(scores, tau_reject: float, tau_accept: float) -> np.ndarray:
    """``decide`` applied to a 1-D array of scores.

    Returns an array of the same length holding ``"accept"``, ``"reject"`` or
    ``"abstain"``. The rule, the ties and the crossing case are those of ``decide``.
    """
    low = _threshold("decide_many", "tau_reject", tau_reject)
    high = _threshold("decide_many", "tau_accept", tau_accept)
    if low > high:
        low, high = high, low
    values = _score_array("decide_many", "scores", scores)
    out = np.full(values.shape, ABSTAIN, dtype="<U7")
    out[values > high] = ACCEPT     # comparisons with NaN are False, so NaN stays abstain
    out[values < low] = REJECT
    return out


def _fewest(func: str, name: str, level: float, works) -> int:
    """The smallest count for which ``works`` is True, starting from ``ceil(1 / level - 1)``."""
    estimate = 1.0 / level - 1.0
    if not math.isfinite(estimate):
        raise ValueError(
            f"{func}: {name} is too close to 0 for a count to exist, got {level!r}. "
            "Pass a level such as 0.02."
        )
    n = max(1, _ceil(estimate))
    # Rounding can move the closed form by 1. Step to the count the index function agrees with.
    for _ in range(2):
        if n > 1 and works(n - 1):
            n -= 1
    for _ in range(2):
        if not works(n):
            n += 1
    return n


def min_negatives_to_accept(alpha: float) -> int:
    """The fewest calibration negatives with which a stage can accept: ``ceil(1 / alpha - 1)``.

    49 at ``alpha = 0.02`` and 199 at ``alpha = 0.005``. With fewer negatives
    ``conformal_accept_index`` returns None. A verifier cannot be calibrated
    without failure examples.
    """
    alpha = check.fraction("min_negatives_to_accept", "alpha", alpha, example="0.02")
    return _fewest("min_negatives_to_accept", "alpha", alpha, lambda n: _accept_index(n, alpha) is not None)


def min_positives_to_reject(beta: float) -> int:
    """The fewest calibration positives with which a stage can reject: ``ceil(1 / beta - 1)``.

    19 at ``beta = 0.05``. With fewer positives ``conformal_reject_index`` returns None.
    """
    beta = check.fraction("min_positives_to_reject", "beta", beta, example="0.05")
    return _fewest("min_positives_to_reject", "beta", beta, lambda m: _reject_index(m, beta) is not None)
