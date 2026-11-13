"""Audit statistics that session S6 builds. Stubs with the formulas, so callers can plan around them.

Spec: ``docs/04_calibration_math.md`` section 7. Both functions raise
NotImplementedError until S6 (``sessions/S6.md`` step 9) fills them in, with
tests on synthetic data where the true value is known.
"""

from __future__ import annotations

__all__ = ["prediction_powered_mean", "rogan_gladen"]


def prediction_powered_mean(g_all, g_audited, y_audited, conf: float = 0.95) -> dict:
    """Audit-efficient estimate of the defect rate among accepted cycles. Built in S6.

    Each accepted cycle has a calibrated defect probability ``g`` from its
    evidence. A random sample of the accepted cycles is audited by a person,
    who records the true label ``y`` (1 for a defect, 0 for a good cycle).
    With ``N`` accepted cycles and ``n`` audited ones:

        D_hat    = mean(g over the N accepted) + mean(y - g over the n audited)
        variance = var(g) / N + var(y - g) / n

    This is prediction-powered inference. The first term uses every cycle. The
    second term corrects it with the audits. The estimate beats the plain audit
    estimate when ``g`` tracks the truth, and it stays unbiased when ``g`` is poor.

    Arguments:

    - ``g_all``: ``g`` for all ``N`` accepted cycles
    - ``g_audited``: ``g`` for the ``n`` audited cycles
    - ``y_audited``: the audit labels for those ``n`` cycles, in the same order
    - ``conf``: confidence for the interval

    Planned return: a dict with ``estimate``, ``variance``, ``lo``, ``hi``,
    ``n_all`` and ``n_audited``, the interval being
    ``estimate`` plus and minus ``z * sqrt(variance)``.

    The audited cycles must be a random sample of the accepted ones. Rows
    sampled with unequal probabilities need inverse-probability weights, which
    S6 adds (``docs/06_flywheel.md``).
    """
    raise NotImplementedError(
        "prediction_powered_mean is built in session S6 (sessions/S6.md step 9). "
        "Until then use the plain audit estimate: clopper_pearson(defects, audits)."
    )


def rogan_gladen(q: float, tar: float, far: float) -> float:
    """Success rate corrected for an imperfect judge (Rogan and Gladen, 1978). Built in S6.

    An automatic judge accepts a share ``q`` of episodes. Its true-accept rate
    ``TAR`` (accepts among true successes) and its false-accept rate ``FAR``
    (accepts among true failures) are known from a labeled set. The observed
    share mixes the 2: ``q = s * TAR + (1 - s) * FAR`` for a true success rate
    ``s``. Solving for ``s``:

        corrected = (q - FAR) / (TAR - FAR)

    Arguments:

    - ``q``: the share of episodes the judge accepted
    - ``tar``: the judge's true-accept rate
    - ``far``: the judge's false-accept rate

    The correction needs ``TAR > FAR``, a judge that is better than chance. The
    raw value can fall outside 0 to 1 when ``q`` is noisy, and S6 decides how
    to report that. Intervals come from a bootstrap over both the episodes and
    the labeled set.
    """
    raise NotImplementedError(
        "rogan_gladen is built in session S6 (sessions/S6.md step 9). "
        "Until then report the judge's raw accept share next to its measured error rates."
    )
