"""Confidence intervals for a rate, and the trial counts a certificate needs.

Spec: ``docs/04_calibration_math.md`` sections 3, 6 and 7.

2 kinds of interval live here. Wilson is the two-sided interval every report
prints next to a rate. Clopper-Pearson is the exact interval behind a
certificate: its one-sided upper bound is the number a false-accept rate or an
uncaught failure rate is certified under.

Counts are whole numbers. Rates and confidence levels are fractions between 0
and 1. Every function needs at least 1 trial and raises ValueError otherwise.
"""

from __future__ import annotations

import math

from scipy import special

from . import _validate as check

__all__ = [
    "wilson",
    "clopper_pearson",
    "cp_upper",
    "cp_lower",
    "zero_failure_ucb",
    "n_to_certify",
    "max_failures_to_certify",
    "certify_power",
]


_MAX_TRIALS = 2**53          # above this a float no longer holds every whole number


def _conf(func: str, conf) -> float:
    return check.fraction(func, "conf", conf, example="0.95")


def wilson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for ``k`` events in ``n`` trials, as ``(lo, hi)``.

    Two-sided at confidence ``conf``. The ends are exact at the edges: ``lo`` is
    0.0 when ``k`` is 0 and ``hi`` is 1.0 when ``k`` equals ``n``.

    ``n`` must be at least 1. For a table cell that can be empty, call
    ``rate_with_interval``.

    Example: ``wilson(29, 30)`` is about ``(0.833, 0.994)``.
    """
    k, n = check.events_in_trials("wilson", "k", k, n)
    conf = _conf("wilson", conf)
    z = float(special.ndtri(0.5 + conf / 2.0))
    z2 = z * z
    p = k / n
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    # center - half is 0 at k = 0 and center + half is 1 at k = n, up to rounding. Return them exactly,
    # so a perfect score never yields an upper end of 0.9999999999999998.
    lo = 0.0 if k == 0 else max(0.0, center - half)
    hi = 1.0 if k == n else min(1.0, center + half)
    return lo, hi


def _cp_upper(x: int, n: int, conf: float) -> float:
    if x == n:
        return 1.0
    if x == 0:
        return -math.expm1(math.log1p(-conf) / n)          # 1 - (1 - conf) ** (1 / n), without cancellation
    return float(special.betaincinv(x + 1, n - x, conf))


def _cp_lower(x: int, n: int, conf: float) -> float:
    if x == 0:
        return 0.0
    if x == n:
        return math.exp(math.log1p(-conf) / n)             # (1 - conf) ** (1 / n)
    return float(special.betaincinv(x, n - x + 1, 1.0 - conf))


def cp_upper(x: int, n: int, conf: float = 0.95) -> float:
    """One-sided Clopper-Pearson upper bound on a rate after ``x`` events in ``n`` trials.

    The bound is the rate at which seeing ``x`` or fewer events in ``n`` trials
    has probability ``1 - conf``. Any larger rate would have made a count this
    low even less likely. This is the certificate of section 3: count false
    accepts on a split the witness never saw and report this bound.

    With ``x = 0`` it equals ``zero_failure_ucb(n, conf)``. With ``x = n`` it is 1.0.

    Example: ``cp_upper(16, 5000)`` is about 0.0049 and ``cp_upper(17, 5000)``
    is about 0.0051, so 5,000 cycles certify a 0.5% bar with up to 16 failures.
    """
    x, n = check.events_in_trials("cp_upper", "x", x, n)
    return _cp_upper(x, n, _conf("cp_upper", conf))


def cp_lower(x: int, n: int, conf: float = 0.95) -> float:
    """One-sided Clopper-Pearson lower bound on a rate after ``x`` events in ``n`` trials.

    The bound is the rate at which seeing ``x`` or more events in ``n`` trials
    has probability ``1 - conf``. With ``x = 0`` it is 0.0.

    It mirrors the upper bound: ``cp_lower(x, n) = 1 - cp_upper(n - x, n)``.
    """
    x, n = check.events_in_trials("cp_lower", "x", x, n)
    return _cp_lower(x, n, _conf("cp_lower", conf))


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Exact (Clopper-Pearson) two-sided interval for ``k`` events in ``n`` trials, as ``(lo, hi)``.

    Each tail gets ``(1 - conf) / 2``, so the 2 ends are the one-sided bounds at
    confidence ``(1 + conf) / 2``. The interval covers the true rate with
    probability at least ``conf`` whatever that rate is.

    ``lo`` is 0.0 when ``k`` is 0 and ``hi`` is 1.0 when ``k`` equals ``n``.
    """
    k, n = check.events_in_trials("clopper_pearson", "k", k, n)
    one_sided = 0.5 + _conf("clopper_pearson", conf) / 2.0
    return _cp_lower(k, n, one_sided), _cp_upper(k, n, one_sided)


def zero_failure_ucb(n: int, conf: float = 0.95) -> float:
    """Upper bound on a failure rate after ``n`` trials with 0 failures: ``1 - (1 - conf) ** (1 / n)``.

    At 95% confidence this is roughly ``3 / n``. 300 clean trials bound the rate
    by about 0.00994.
    """
    n = check.count(
        "zero_failure_ucb", "n", n, 1,
        "The bound needs at least 1 clean trial. Pass the number of trials that ran without a failure.",
    )
    return _cp_upper(0, n, _conf("zero_failure_ucb", conf))


def _p_max(func: str, p_max) -> float:
    return check.fraction(func, "p_max", p_max, example="0.005 for a 0.5% bar")


def n_to_certify(p_max: float, allowed_failures: int = 0, conf: float = 0.95) -> int:
    """The smallest number of trials that can certify a failure rate of at most ``p_max``.

    It is the smallest ``n`` with ``cp_upper(allowed_failures, n, conf) <= p_max``.
    A run of that many trials certifies the bar when it shows ``allowed_failures``
    failures or fewer. A run with more failures needs more trials.

    Examples at 95% confidence: 598 clean trials certify 0.5%. With 1 failure
    it takes 947, and with 16 failures 4,857.
    """
    p_max = _p_max("n_to_certify", p_max)
    failures = check.count(
        "n_to_certify", "allowed_failures", allowed_failures, 0,
        "Pass the number of failures the certificate may contain, 0 or more.",
    )
    conf = _conf("n_to_certify", conf)
    # The bound falls as n grows. Bracket the answer by doubling, then bisect.
    low, high = failures, failures + 1          # cp_upper(x, x) is 1.0, which is above every p_max
    while _cp_upper(failures, high, conf) > p_max:
        low, high = high, 2 * high
        if high > _MAX_TRIALS:
            raise ValueError(
                f"n_to_certify: p_max={p_max!r} would take more than 9e15 trials, which is more than a "
                "float can count. Pass a larger p_max, for example 0.005 for a 0.5% bar."
            )
    while high - low > 1:
        mid = (low + high) // 2
        if _cp_upper(failures, mid, conf) > p_max:
            low = mid
        else:
            high = mid
    return high


def _max_failures(n: int, p_max: float, conf: float) -> int | None:
    if _cp_upper(0, n, conf) > p_max:
        return None
    low, high = 0, n                            # x = 0 certifies and x = n never does
    while high - low > 1:
        mid = (low + high) // 2
        if _cp_upper(mid, n, conf) <= p_max:
            low = mid
        else:
            high = mid
    return low


def max_failures_to_certify(n: int, p_max: float, conf: float = 0.95) -> int | None:
    """The largest failure count with which ``n`` trials still certify a rate of at most ``p_max``.

    It is the largest ``x`` with ``cp_upper(x, n, conf) <= p_max``. None means
    ``n`` is too small to certify the bar even with 0 failures.

    Example: ``max_failures_to_certify(5000, 0.005)`` is 16.
    """
    n = check.count("max_failures_to_certify", "n", n, 1, "Pass the number of trials planned, 1 or more.")
    return _max_failures(n, _p_max("max_failures_to_certify", p_max), _conf("max_failures_to_certify", conf))


def _binom_cdf(x: int, n: int, p: float) -> float:
    """P(at most ``x`` events in ``n`` trials at rate ``p``)."""
    if x >= n:
        return 1.0
    return float(special.betainc(n - x, x + 1, 1.0 - p))


def certify_power(n: int, p_max: float, p_true: float, conf: float = 0.95) -> float:
    """The chance that ``n`` trials certify a rate of at most ``p_max`` when the true rate is ``p_true``.

    The run certifies the bar when it shows ``max_failures_to_certify(n, p_max, conf)``
    failures or fewer, so this is an exact binomial probability. It is 0.0
    when ``n`` is too small to certify the bar at all.

    Example: ``certify_power(5000, 0.005, p_true=0.002)`` is about 0.97. With a
    true rate of 0.2%, 5,000 cycles certify a 0.5% bar about 97 times in 100.
    At 0.3% it is about 66 in 100.
    """
    n = check.count("certify_power", "n", n, 1, "Pass the number of trials planned, 1 or more.")
    p_true = check.fraction("certify_power", "p_true", p_true, low_ok=True, high_ok=True, example="0.002")
    most = _max_failures(n, _p_max("certify_power", p_max), _conf("certify_power", conf))
    if most is None:
        return 0.0
    return _binom_cdf(most, n, p_true)
