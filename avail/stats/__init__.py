"""The statistics the rest of AVAIL quotes.

Spec: ``docs/04_calibration_math.md``. Known-answer tests: ``tests/test_stats.py``.

Pure functions on numpy and scipy. Nothing here reads a file, a ledger or the
station.

- ``intervals``: Wilson and Clopper-Pearson intervals, the zero-failure bound,
  the trial counts a certificate needs
- ``planning``: trials to tell 2 success rates apart, Wald's sequential test, McNemar
- ``conformal``: accept and reject thresholds with a stated error bound, the 3-way decision
- ``spc``: p-chart limits and CUSUM, for noticing drift
- ``economics``: uncaught failures per claim, robots per assistant, MTBI, autonomy, cost per cycle
- ``rates``: a rate with its n and its interval, the way every report states it
- ``stubs``: prediction-powered inference and Rogan-Gladen, built in S6
- ``cli``: the ``avail trials`` command

Conventions: counts are whole numbers, rates and confidence levels are
fractions between 0 and 1, and a higher score means the predicate is more
likely true. A bad argument raises ValueError with a message that says how to
fix the call.
"""

from .conformal import (
    ABSTAIN,
    ACCEPT,
    REJECT,
    conformal_accept_index,
    conformal_accept_threshold,
    conformal_reject_index,
    conformal_reject_threshold,
    decide,
    decide_many,
    min_negatives_to_accept,
    min_positives_to_reject,
)
from .economics import (
    autonomy,
    busy_share,
    cost_per_cycle,
    defect_rate_among_accepted,
    mtbi,
    robots_per_assistant,
    uncaught_per_claim,
)
from .intervals import (
    certify_power,
    clopper_pearson,
    cp_lower,
    cp_upper,
    max_failures_to_certify,
    n_to_certify,
    wilson,
    zero_failure_ucb,
)
from .planning import WaldSPRT, mcnemar, n_two_proportions, wald_boundaries
from .rates import balanced_accuracy, decision_rates, rate_with_interval
from .spc import Cusum, p_chart_limits
from .stubs import prediction_powered_mean, rogan_gladen

__all__ = [
    # intervals
    "wilson",
    "clopper_pearson",
    "cp_upper",
    "cp_lower",
    "zero_failure_ucb",
    "n_to_certify",
    "max_failures_to_certify",
    "certify_power",
    # planning
    "n_two_proportions",
    "wald_boundaries",
    "WaldSPRT",
    "mcnemar",
    # conformal
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
    # spc
    "p_chart_limits",
    "Cusum",
    # economics
    "busy_share",
    "robots_per_assistant",
    "mtbi",
    "autonomy",
    "cost_per_cycle",
    "uncaught_per_claim",
    "defect_rate_among_accepted",
    # rates
    "rate_with_interval",
    "balanced_accuracy",
    "decision_rates",
    # stubs
    "prediction_powered_mean",
    "rogan_gladen",
]
