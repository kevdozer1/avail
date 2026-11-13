"""Trial planning: how many trials, when to stop early, and paired comparisons.

Spec: ``docs/04_calibration_math.md`` section 8.

- ``n_two_proportions``: trials per arm to tell 2 success rates apart.
- ``WaldSPRT``: Wald's sequential test, which stops as soon as the data decide.
- ``mcnemar``: 2 policies run on the same seeds.

The zero-failure certificate count is ``n_to_certify`` in ``intervals.py``.
"""

from __future__ import annotations

import math

from scipy import special

from . import _validate as check

__all__ = ["n_two_proportions", "wald_boundaries", "WaldSPRT", "mcnemar"]


def _two_proportion_terms(func: str, p1, p2, alpha, power,
                          names: tuple[str, str, str, str] = ("p1", "p2", "alpha", "power")) -> dict:
    """Every term of the section 8 formula, for the function and for the command line.

    ``names`` are the argument names used in error messages, so the command line
    can report a problem under the flag the person typed.
    """
    name_1, name_2, name_alpha, name_power = names
    p1 = check.fraction(func, name_1, p1, low_ok=True, high_ok=True, example="0.95")
    p2 = check.fraction(func, name_2, p2, low_ok=True, high_ok=True, example="0.99")
    alpha = check.fraction(func, name_alpha, alpha, example="0.05")
    power = check.fraction(func, name_power, power, example="0.8")
    gap2 = (p1 - p2) ** 2
    if gap2 == 0.0:
        raise ValueError(
            f"{func}: {name_1} and {name_2} are equal ({p1!r}), so no number of trials can tell them apart. "
            "Pass 2 different rates, for example 0.95 and 0.99."
        )
    if power <= alpha:
        raise ValueError(
            f"{func}: {name_power} must be larger than {name_alpha}, got {power!r} and {alpha!r}. "
            "Pass the chance of detecting the difference as the power, for example 0.8 with an alpha of 0.05."
        )
    z_a = float(special.ndtri(1.0 - alpha / 2.0))
    z_b = float(special.ndtri(power))
    pbar = (p1 + p2) / 2.0
    root = z_a * math.sqrt(2.0 * pbar * (1.0 - pbar)) + z_b * math.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))
    n_exact = root * root / gap2
    return {
        "p1": p1, "p2": p2, "alpha": alpha, "power": power,
        "pbar": pbar, "z_a": z_a, "z_b": z_b, "n_exact": n_exact, "n": max(1, math.ceil(n_exact)),
    }


def n_two_proportions(p1: float, p2: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Trials per arm to tell success rate ``p1`` from ``p2`` with a two-sided two-proportion z-test.

    ``n = (z_a * sqrt(2 * pbar * (1 - pbar)) + z_b * sqrt(p1 * (1 - p1) + p2 * (1 - p2)))^2 / (p1 - p2)^2``
    with ``pbar = (p1 + p2) / 2``, ``z_a`` the normal quantile at ``1 - alpha / 2``
    and ``z_b`` the normal quantile at ``power``. The result is rounded up.
    ``power`` must be larger than ``alpha``.

    At alpha 0.05 and power 0.8, telling 95% from 99% takes 285 trials per arm
    and telling 99% from 99.9% takes 1,059.
    """
    return _two_proportion_terms("n_two_proportions", p1, p2, alpha, power)["n"]


def _error_rates(func: str, alpha, beta) -> tuple[float, float]:
    alpha = check.fraction(func, "alpha", alpha, example="0.05")
    beta = check.fraction(func, "beta", beta, example="0.2")
    if alpha + beta >= 1.0:
        raise ValueError(
            f"{func}: alpha + beta must be below 1, got alpha={alpha!r} and beta={beta!r}. "
            "Pass 2 small error rates, for example alpha=0.05 and beta=0.2."
        )
    return alpha, beta


def wald_boundaries(alpha: float = 0.05, beta: float = 0.2) -> tuple[float, float]:
    """The 2 stopping boundaries of Wald's sequential test, as ``(lower, upper)``.

    ``lower = ln(beta / (1 - alpha))`` and ``upper = ln((1 - beta) / alpha)``.
    The test stops for ``p0`` when the log-likelihood ratio falls to ``lower``
    and for ``p1`` when it rises to ``upper``.

    ``alpha`` is the chance of stopping for ``p1`` when the rate is ``p0``.
    ``beta`` is the chance of stopping for ``p0`` when the rate is ``p1``.
    """
    alpha, beta = _error_rates("wald_boundaries", alpha, beta)
    return math.log(beta / (1.0 - alpha)), math.log((1.0 - beta) / alpha)


class WaldSPRT:
    """Wald's sequential test of a success rate: ``p <= p0`` against ``p >= p1``.

    Each success adds ``ln(p1 / p0)`` to the log-likelihood ratio and each
    failure adds ``ln((1 - p1) / (1 - p0))``. The test stops for ``p1`` at
    ``upper = ln((1 - beta) / alpha)`` and for ``p0`` at ``lower = ln(beta / (1 - alpha))``.

    ``decision`` is ``"continue"`` until a boundary is reached, then ``"p1"``
    (the data support ``p >= p1``) or ``"p0"`` (the data support ``p <= p0``).
    The decision is final. Later calls to ``update`` change nothing, because the
    error rates only hold for a test that stops at the first crossing. It is
    read-only: the test sets it, and nothing else can.

    This is the outcome of a statistical test. It is unrelated to the decision
    of a Verdict, which only the witness sets.

    Example::

        test = WaldSPRT(p0=0.95, p1=0.99)
        for ok in outcomes:
            if test.update(ok) != "continue":
                break
        print(test.decision, test.n)

    68 successes in a row stop this test for ``p1``. A failure on the first
    trial stops it for ``p0``.
    """

    CONTINUE = "continue"
    P0 = "p0"
    P1 = "p1"

    def __init__(self, p0: float, p1: float, alpha: float = 0.05, beta: float = 0.2) -> None:
        self.p0 = check.fraction("WaldSPRT", "p0", p0, example="0.95")
        self.p1 = check.fraction("WaldSPRT", "p1", p1, example="0.99")
        if self.p0 >= self.p1:
            raise ValueError(
                f"WaldSPRT: p0 must be below p1, got p0={p0!r} and p1={p1!r}. "
                "Pass the lower success rate as p0 and the higher one as p1."
            )
        self.alpha, self.beta = _error_rates("WaldSPRT", alpha, beta)
        self.lower, self.upper = wald_boundaries(self.alpha, self.beta)
        self.step_success = math.log(self.p1 / self.p0)
        self.step_failure = math.log1p(-self.p1) - math.log1p(-self.p0)
        self.n = 0
        self.successes = 0
        self.llr = 0.0
        self._stopped_for = self.CONTINUE       # "continue", or the hypothesis the test stopped for

    @property
    def decision(self) -> str:
        """``"continue"``, ``"p0"`` or ``"p1"``. Read-only. Only ``update`` moves it, and only once."""
        return self._stopped_for

    @property
    def failures(self) -> int:
        return self.n - self.successes

    @property
    def boundaries(self) -> tuple[float, float]:
        """``(lower, upper)``."""
        return self.lower, self.upper

    def update(self, success) -> str:
        """Add the outcome of 1 trial and return the decision."""
        if success is None or isinstance(success, str) or success not in (True, False):
            raise ValueError(
                f"WaldSPRT.update: success must be True or False, got {success!r}. "
                "Pass the outcome of 1 trial. Leave out trials that have no outcome."
            )
        if self._stopped_for != self.CONTINUE:
            return self._stopped_for
        self.n += 1
        self.successes += 1 if success else 0
        # Recompute from the counts, so rounding error does not build up over a long run.
        self.llr = self.successes * self.step_success + self.failures * self.step_failure
        if self.llr >= self.upper:
            self._stopped_for = self.P1
        elif self.llr <= self.lower:
            self._stopped_for = self.P0
        return self._stopped_for

    def __repr__(self) -> str:
        return (f"WaldSPRT(p0={self.p0}, p1={self.p1}, alpha={self.alpha}, beta={self.beta}, "
                f"n={self.n}, successes={self.successes}, llr={self.llr:.4f}, decision={self.decision!r})")


def mcnemar(b: int, c: int) -> dict:
    """McNemar's test for 2 policies run on the same seeds.

    ``b`` counts the pairs where the first policy succeeded and the second
    failed. ``c`` counts the opposite pairs. Pairs where both agree carry no
    information about the difference and are left out.

    Returns a dict with:

    - ``b``, ``c`` and ``n_discordant = b + c``
    - ``p_exact``: the two-sided exact binomial p-value, ``min(1, 2 * P(X <= min(b, c)))``
      for ``X`` binomial with ``b + c`` trials at rate 0.5. Quote this one.
    - ``chi2``: the continuity-corrected chi-square, ``(|b - c| - 1)^2 / (b + c)``
    - ``p_chi2``: its p-value on 1 degree of freedom

    When ``b`` equals ``c`` there is no difference to correct, so ``chi2`` is
    0.0 and both p-values are 1.0. That includes the case of no discordant pairs.
    """
    b = check.count("mcnemar", "b", b, 0, "Pass the number of pairs where only the first policy succeeded.")
    c = check.count("mcnemar", "c", c, 0, "Pass the number of pairs where only the second policy succeeded.")
    n = b + c
    low = min(b, c)
    if b == c:
        p_exact, chi2 = 1.0, 0.0
    else:
        tail = float(special.betainc(n - low, low + 1, 0.5))      # P(X <= low) for X ~ Binomial(n, 0.5)
        p_exact = min(1.0, 2.0 * tail)
        chi2 = (abs(b - c) - 1.0) ** 2 / n
    p_chi2 = math.erfc(math.sqrt(chi2 / 2.0))                     # chi-square survival function, 1 df
    return {"b": b, "c": c, "n_discordant": n, "p_exact": p_exact, "chi2": chi2, "p_chi2": p_chi2}
