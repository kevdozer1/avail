"""Control charts that notice when new claims stop looking like the calibration set.

Spec: ``docs/04_calibration_math.md`` section 9.

The conformal bound assumes tomorrow looks like the calibration set. These
charts watch a series for the day it does not: the abstain rate, the reject
rate, the audit disagreement rate, or the running mean of a key feature.

- ``p_chart_limits``: 3-sigma limits for a rate measured on windows of claims.
- ``Cusum``: a running sum that catches a small shift that lasts.
"""

from __future__ import annotations

import math

import numpy as np

from . import _validate as check

__all__ = ["p_chart_limits", "Cusum"]


def p_chart_limits(pbar: float, w: int, sigmas: float = 3.0) -> tuple[float, float, float]:
    """Limits of a p-chart on windows of ``w`` claims, as ``(lcl, center, ucl)``.

    The center is the baseline rate ``pbar``. The limits sit ``sigmas`` standard
    errors on each side. With the default of 3 they are
    ``pbar - 3 * sqrt(pbar * (1 - pbar) / w)`` and
    ``pbar + 3 * sqrt(pbar * (1 - pbar) / w)``, clipped to the range 0 to 1.
    A window whose rate falls outside the limits is out of control.

    A baseline of exactly 0 or 1 gives limits of zero width, so every window
    with a different rate is out of control. Use a longer baseline in that case.

    Example: ``p_chart_limits(0.5, 100)`` is ``(0.35, 0.5, 0.65)``.
    """
    pbar = check.fraction("p_chart_limits", "pbar", pbar, low_ok=True, high_ok=True, example="0.1")
    w = check.count("p_chart_limits", "w", w, 1, "Pass the number of claims in each window, for example 50.")
    sigmas = check.real(
        "p_chart_limits", "sigmas", sigmas, "Pass the limit width in standard errors, for example 3.",
        minimum=0.0, strict=True,
    )
    half = sigmas * math.sqrt(pbar * (1.0 - pbar) / w)
    return max(0.0, pbar - half), pbar, min(1.0, pbar + half)


class Cusum:
    """One-sided CUSUM chart for a small upward shift that lasts.

    The statistic starts at 0 and follows ``S = max(0, S + x - (p0 + k))`` with
    ``k = (p1 - p0) / 2``. The chart is in alarm when ``S > h``.

    ``p0`` is the baseline level of the series and ``p1`` is the shifted level
    worth catching. ``x`` is 1 observation: True or False for an event such as
    an abstain, a window rate, or a feature value. Observations below the
    midpoint ``p0 + k`` pull ``S`` down toward 0 and observations above it push
    ``S`` up, so only a run of high values reaches ``h``.

    To watch for a drop, negate the series and both levels.

    ``k_fraction`` sets ``k`` as a share of the shift, ``k = k_fraction * (p1 - p0)``.
    The default 0.5 is the formula above.

    Attributes: ``s`` (the statistic), ``n`` (observations used), ``skipped``
    (missing observations left out), ``first_alarm_at`` (the value of ``n`` at
    the first alarm, or None), ``alarm`` (True while ``S > h``).

    Example::

        chart = Cusum(p0=0.10, p1=0.30, h=4.0)
        for abstained in stream:
            if chart.update(abstained):
                switch_to_conservative_mode()
                break
    """

    def __init__(self, p0: float, p1: float, h: float, *, k_fraction: float = 0.5) -> None:
        self.p0 = check.real("Cusum", "p0", p0, "Pass the baseline level of the watched series, for example 0.1.")
        self.p1 = check.real("Cusum", "p1", p1, "Pass the shifted level worth catching, for example 0.3.")
        if self.p1 <= self.p0:
            raise ValueError(
                f"Cusum: p1 must be larger than p0, got p0={p0!r} and p1={p1!r}. The chart watches for a rise. "
                "Pass the baseline level as p0 and the higher level as p1. To watch for a drop, negate the "
                "series and both levels."
            )
        self.h = check.real("Cusum", "h", h, "Pass a positive alarm height, for example 4.0.",
                            minimum=0.0, strict=True)
        self.k_fraction = check.fraction("Cusum", "k_fraction", k_fraction, example="0.5")
        self.k = self.k_fraction * (self.p1 - self.p0)
        self.reference = self.p0 + self.k
        self.s = 0.0
        self.n = 0
        self.skipped = 0
        self.first_alarm_at: int | None = None

    @property
    def alarm(self) -> bool:
        """True while the statistic is above ``h``."""
        return self.s > self.h

    def update(self, x) -> bool:
        """Add 1 observation and return whether the chart is in alarm.

        A missing observation (None or NaN) is left out: the statistic does not
        move and ``skipped`` goes up by 1.
        """
        if x is None:
            value = math.nan
        elif isinstance(x, (bool, np.bool_)):
            value = 1.0 if x else 0.0
        elif check.is_real(x):
            value = float(x)
        else:
            raise ValueError(
                f"Cusum.update: x must be a number or True or False, got {x!r}. "
                "Pass 1 observation of the watched series. Use NaN for a missing one."
            )
        if math.isnan(value):
            self.skipped += 1
            return self.alarm
        if math.isinf(value):
            raise ValueError(
                f"Cusum.update: x must be finite, got {x!r}. "
                "Fix the series upstream, or pass NaN to leave this observation out."
            )
        self.n += 1
        self.s = max(0.0, self.s + value - self.reference)
        if self.s > self.h and self.first_alarm_at is None:
            self.first_alarm_at = self.n
        return self.alarm

    def reset(self) -> None:
        """Back to the starting state. Call it after the cause of an alarm is dealt with."""
        self.s = 0.0
        self.n = 0
        self.skipped = 0
        self.first_alarm_at = None

    def __repr__(self) -> str:
        return (f"Cusum(p0={self.p0}, p1={self.p1}, h={self.h}, k={self.k:.6g}, "
                f"s={self.s:.6g}, n={self.n}, alarm={self.alarm})")
