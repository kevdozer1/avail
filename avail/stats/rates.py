"""Rates with their n and their interval, the way every report states them.

Spec: ``docs/04_calibration_math.md`` section 1.

- False-accept rate (FAR): accepts among claims whose truth is False.
- False-reject rate (FRR): rejects among claims whose truth is True.
- Abstain rate: abstains among all claims.

These helpers build table cells, and a table can have an empty cell. A rate
over 0 trials has no value, so its point is NaN and its interval is the whole
range 0 to 1. That keeps a report running and keeps every "is the upper bound
under the bar" check failing closed. The interval functions in ``intervals.py``
are strict and raise at ``n = 0``.
"""

from __future__ import annotations

import math

import numpy as np

from . import _validate as check
from .conformal import ABSTAIN, ACCEPT, REJECT
from .intervals import cp_upper, wilson

__all__ = ["rate_with_interval", "balanced_accuracy", "decision_rates"]

_DECISIONS = (ACCEPT, REJECT, ABSTAIN)


def _cell(func: str, k, n, conf) -> dict:
    conf = check.fraction(func, "conf", conf, example="0.95")
    n = check.count(func, "n", n, 0, "Pass the number of trials, 0 or more.")
    k = check.count(func, "k", k, 0, "Pass the number of events seen, 0 or more.")
    if k > n:
        raise ValueError(
            f"{func}: k must be between 0 and n, got k={k} with n={n}. "
            "Pass the event count first and the number of trials second."
        )
    if n == 0:
        return {"k": 0, "n": 0, "point": math.nan, "lo": 0.0, "hi": 1.0}
    lo, hi = wilson(k, n, conf)
    return {"k": k, "n": n, "point": k / n, "lo": lo, "hi": hi}


def rate_with_interval(k: int, n: int, conf: float = 0.95) -> dict:
    """A rate as a report states it: ``{"k", "n", "point", "lo", "hi"}``.

    ``point`` is ``k / n`` and ``lo`` to ``hi`` is the two-sided Wilson interval
    at confidence ``conf``.

    An empty cell (``n = 0``) has ``point`` NaN, ``lo`` 0.0 and ``hi`` 1.0: no
    estimate, and an interval that says nothing is known.

    Example: ``rate_with_interval(8, 400)`` has ``point`` 0.02 and an interval
    of about 0.010 to 0.039.
    """
    return _cell("rate_with_interval", k, n, conf)


def balanced_accuracy(tp: int, fn: int, tn: int, fp: int) -> float:
    """Balanced accuracy of a 2-way detector: the mean of its true-positive and true-negative rates.

    ``(tp / (tp + fn) + tn / (tn + fp)) / 2``. A detector that always gives the
    same answer scores 0.5 whatever the class mix.

    Returns NaN when either class is empty, because the rate for that class is undefined.
    """
    counts = []
    for name, value in (("tp", tp), ("fn", fn), ("tn", tn), ("fp", fp)):
        counts.append(check.count("balanced_accuracy", name, value, 0,
                                  f"Pass the {name} cell of the confusion table, 0 or more."))
    tp, fn, tn, fp = counts
    if tp + fn == 0 or tn + fp == 0:
        return math.nan
    return (tp / (tp + fn) + tn / (tn + fp)) / 2.0


def _labels(values) -> list[bool]:
    out = []
    for i, value in enumerate(values):
        if isinstance(value, (bool, np.bool_)):
            out.append(bool(value))
        elif check.is_real(value) and value in (0, 1):
            out.append(bool(value))
        else:
            raise ValueError(
                f"decision_rates: labels[{i}] must be True or False, got {value!r}. "
                "Pass the truth of every claim. Drop the claims that have no truth label before calling."
            )
    return out


def decision_rates(decisions, labels, conf: float = 0.95) -> dict:
    """Counts and rates of accept, reject and abstain against truth labels.

    ``decisions`` holds ``"accept"``, ``"reject"`` or ``"abstain"`` per claim.
    ``labels`` holds the truth per claim: True when the predicate held.

    Returns a dict with:

    - ``n``, ``n_pos``, ``n_neg``: claims in all, with label True, with label False
    - ``counts``: ``{"pos": {...}, "neg": {...}, "all": {...}}``, each mapping
      the 3 decisions to a count
    - ``far``: accepts among label False, as a ``rate_with_interval`` dict plus
      ``ucb``, the one-sided Clopper-Pearson upper bound at ``conf``
    - ``frr``: rejects among label True
    - ``abstain``: abstains among all claims
    - ``tar``: accepts among label True (the true-accept rate)
    - ``abstain_pos`` and ``abstain_neg``: abstains among label True and among label False
    - ``conf``

    Every rate carries its ``k``, its ``n`` and a Wilson interval. A rate over
    0 claims is an empty cell: ``point`` NaN, interval 0 to 1, and for ``far``
    a ``ucb`` of 1.0.

    Every claim needs a label. Labels are often collected on a biased sample,
    so dropping the unlabeled claims is a choice the caller has to make.
    """
    func = "decision_rates"
    conf = check.fraction(func, "conf", conf, example="0.95")
    try:
        decisions = list(decisions)
        labels = list(labels)
    except TypeError:
        raise ValueError(
            f"{func}: decisions and labels must be sequences. Pass 2 lists of the same length, 1 entry per claim."
        ) from None
    if len(decisions) != len(labels):
        raise ValueError(
            f"{func}: decisions and labels must have the same length, got {len(decisions)} and {len(labels)}. "
            "Pass 1 decision and 1 truth label per claim."
        )
    truth = _labels(labels)
    counts = {group: dict.fromkeys(_DECISIONS, 0) for group in ("pos", "neg", "all")}
    for i, (decision, positive) in enumerate(zip(decisions, truth)):
        if not isinstance(decision, str) or decision not in _DECISIONS:
            raise ValueError(
                f"{func}: decisions[{i}] must be 'accept', 'reject' or 'abstain', got {decision!r}. "
                "Pass the decision string of each verdict."
            )
        counts["pos" if positive else "neg"][decision] += 1
        counts["all"][decision] += 1
    n_pos = sum(counts["pos"].values())
    n_neg = sum(counts["neg"].values())
    n = n_pos + n_neg

    far = _cell(func, counts["neg"][ACCEPT], n_neg, conf)
    far["ucb"] = cp_upper(far["k"], n_neg, conf) if n_neg else 1.0
    return {
        "n": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "counts": counts,
        "far": far,
        "frr": _cell(func, counts["pos"][REJECT], n_pos, conf),
        "abstain": _cell(func, counts["all"][ABSTAIN], n, conf),
        "tar": _cell(func, counts["pos"][ACCEPT], n_pos, conf),
        "abstain_pos": _cell(func, counts["pos"][ABSTAIN], n_pos, conf),
        "abstain_neg": _cell(func, counts["neg"][ABSTAIN], n_neg, conf),
        "conf": conf,
    }
