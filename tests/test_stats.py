"""Known-answer tests for ``avail/stats``. Spec: ``docs/04_calibration_math.md``.

The rest of the project quotes these numbers. Every expected value below is
taken from the doc as printed and compared at the precision the doc prints.
Section 11 of the doc lists the rows that must pass before anything is built on
the module. Each row is its own test, named ``test_s11_...``.

The simulation tests draw synthetic scores from fixed numpy seeds. They never
touch a station seed. What "within sampling error" means is defined once, above
the simulation tests.
"""

from __future__ import annotations

import ast
import inspect
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as sps

import avail.stats as S
from avail import cli as avail_cli
from avail.stats import (
    ABSTAIN,
    ACCEPT,
    REJECT,
    Cusum,
    WaldSPRT,
    autonomy,
    balanced_accuracy,
    busy_share,
    certify_power,
    clopper_pearson,
    conformal_accept_index,
    conformal_accept_threshold,
    conformal_reject_index,
    conformal_reject_threshold,
    cost_per_cycle,
    cp_lower,
    cp_upper,
    decide,
    decide_many,
    decision_rates,
    defect_rate_among_accepted,
    max_failures_to_certify,
    mcnemar,
    min_negatives_to_accept,
    min_positives_to_reject,
    mtbi,
    n_to_certify,
    n_two_proportions,
    p_chart_limits,
    prediction_powered_mean,
    rate_with_interval,
    robots_per_assistant,
    rogan_gladen,
    uncaught_per_claim,
    wald_boundaries,
    wilson,
    zero_failure_ucb,
)
from avail.stats import cli as trials_cli

EM_DASH, EN_DASH = chr(0x2014), chr(0x2013)      # spelled as code points so this file stays plain ASCII


def shown(value: float, decimals: int) -> str:
    """``value`` the way the doc prints it, so a comparison happens at the doc's precision."""
    return f"{value:.{decimals}f}"


# =============================================================================================
# Section 11: the known-answer table. One test per row, expected values as the doc prints them.
# =============================================================================================

def test_s11_n_two_proportions_95_against_99_is_285():
    assert n_two_proportions(0.95, 0.99, alpha=0.05, power=0.8) == 285


def test_s11_n_two_proportions_99_against_999_is_1059():
    assert n_two_proportions(0.99, 0.999, alpha=0.05, power=0.8) == 1059


def test_s11_wilson_60_of_60_is_0940_to_1000():
    lo, hi = wilson(60, 60)
    assert f"{shown(lo, 3)} to {shown(hi, 3)}" == "0.940 to 1.000"


def test_s11_wilson_29_of_30_is_0833_to_0994():
    lo, hi = wilson(29, 30)
    assert f"{shown(lo, 3)} to {shown(hi, 3)}" == "0.833 to 0.994"


def test_s11_wilson_292_of_300_is_0948_to_0986():
    lo, hi = wilson(292, 300)
    assert f"{shown(lo, 3)} to {shown(hi, 3)}" == "0.948 to 0.986"


def test_s11_zero_failure_ucb_300_is_000994():
    assert shown(zero_failure_ucb(300), 5) == "0.00994"


def test_s11_n_to_certify_1_percent_is_299():
    assert n_to_certify(0.01) == 299


def test_s11_n_to_certify_half_a_percent_is_598():
    assert n_to_certify(0.005) == 598


def test_s11_n_to_certify_half_a_percent_with_1_failure_is_947():
    assert n_to_certify(0.005, allowed_failures=1) == 947


def test_s11_n_to_certify_half_a_percent_with_8_failures_is_2884():
    assert n_to_certify(0.005, allowed_failures=8) == 2884


def test_s11_n_to_certify_half_a_percent_with_16_failures_is_4857():
    assert n_to_certify(0.005, allowed_failures=16) == 4857


def test_s11_cp_upper_16_and_17_of_5000_are_on_either_side_of_the_bar():
    with_16, with_17 = cp_upper(16, 5000), cp_upper(17, 5000)
    assert shown(with_16, 4) == "0.0049"
    assert shown(with_17, 4) == "0.0051"
    assert with_16 < 0.005 < with_17


def test_s11_conformal_accept_index_99_is_rank_98_of_99():
    assert conformal_accept_index(n=99, alpha=0.02) == 98


def test_s11_conformal_accept_index_40_never_accepts():
    assert conformal_accept_index(n=40, alpha=0.02) is None


def test_s11_robots_per_assistant_at_70_percent_is_14():
    value = robots_per_assistant(assists_per_hour=6, handle_s=30, utilization=0.7)
    assert shown(value, 1) == "14.0"
    assert value == pytest.approx(14.0, abs=1e-9)


# =============================================================================================
# The tables and worked numbers in sections 2, 3, 6, 7, 8 and 10, and the power line of docs/07.
# =============================================================================================

@pytest.mark.parametrize("p_max, clean_negatives", [(0.05, 59), (0.02, 149), (0.01, 299), (0.005, 598), (0.001, 2995)])
def test_s3_clean_negatives_needed_to_certify_a_false_accept_rate(p_max, clean_negatives):
    assert n_to_certify(p_max) == clean_negatives
    # the count is the smallest one: 1 fewer clean negative leaves the bound above the bar
    assert zero_failure_ucb(clean_negatives) <= p_max < zero_failure_ucb(clean_negatives - 1)


@pytest.mark.parametrize("failures, cycles", [(0, 598), (1, 947), (2, 1258), (4, 1829), (8, 2884), (16, 4857)])
def test_s6_cycles_needed_to_certify_half_a_percent(failures, cycles):
    assert n_to_certify(0.005, allowed_failures=failures) == cycles
    assert cp_upper(failures, cycles) <= 0.005 < cp_upper(failures, cycles - 1)


def test_s3_zero_failure_bound_is_the_formula_and_roughly_3_over_m():
    for m in (59, 300, 598, 2995):
        assert zero_failure_ucb(m) == pytest.approx(1 - 0.05 ** (1 / m), rel=1e-12)
        assert zero_failure_ucb(m) == cp_upper(0, m)
    for m in (300, 598, 2995):
        assert 0.99 < zero_failure_ucb(m) * m / 3 < 1.0


def test_s6_5000_cycles_certify_the_bar_with_up_to_16_failures_and_fail_it_at_17():
    assert max_failures_to_certify(5000, 0.005) == 16
    assert cp_upper(16, 5000) <= 0.005 < cp_upper(17, 5000)
    assert n_to_certify(0.005, allowed_failures=16) <= 5000 < n_to_certify(0.005, allowed_failures=17)


def _binomial_cdf_exact(most: int, n: int, p: Fraction) -> float:
    """P(at most ``most`` events in ``n`` trials), summed term by term in exact rational arithmetic."""
    return float(sum(math.comb(n, x) * p**x * (1 - p) ** (n - x) for x in range(most + 1)))


@pytest.mark.parametrize("true_rate, runs_in_100", [(Fraction(2, 1000), 97), (Fraction(3, 1000), 66)])
def test_s6_chance_that_5000_cycles_certify_the_bar(true_rate, runs_in_100):
    """Section 6: at a true rate of 0.2% the bar is certified about 97 times in 100, at 0.3% about 66."""
    exact = _binomial_cdf_exact(16, 5000, true_rate)          # certified when 16 or fewer cycles fail
    assert abs(100 * exact - runs_in_100) <= 1.0              # the doc says "about": within 1 point
    assert certify_power(5000, 0.005, p_true=float(true_rate)) == pytest.approx(exact, abs=1e-12)


def test_certify_power_is_zero_when_the_run_is_too_short_to_certify_at_all():
    assert max_failures_to_certify(597, 0.005) is None
    assert certify_power(597, 0.005, p_true=0.0) == 0.0
    assert max_failures_to_certify(598, 0.005) == 0
    assert certify_power(598, 0.005, p_true=0.0) == 1.0
    assert certify_power(598, 0.005, p_true=0.001) == pytest.approx(0.999**598, rel=1e-12)   # no failure in 598
    assert certify_power(5000, 0.005, p_true=0.002, conf=0.99) < certify_power(5000, 0.005, p_true=0.002)


def test_s6_uncaught_failures_per_100_claims():
    """Section 6: with p = 0.85, a check with FAR 30% lets 4.5 through per 100 claims, FAR 2% lets 0.3."""
    assert 100 * uncaught_per_claim(0.85, 0.30) == pytest.approx(4.5, abs=1e-9)
    assert 100 * uncaught_per_claim(0.85, 0.02) == pytest.approx(0.3, abs=1e-9)
    assert shown(100 * uncaught_per_claim(0.85, 0.30), 1) == "4.5"
    assert shown(100 * uncaught_per_claim(0.85, 0.02), 1) == "0.3"
    # a base failure rate of 15% and a cascade at alpha = 0.01 expect about 0.15% uncaught
    assert f"{uncaught_per_claim(0.85, 0.01):.2%}" == "0.15%"


def test_s6_defect_rate_among_accepted_claims():
    p, far, frr, abstain_pos = 0.85, 0.02, 0.05, 0.10
    bad = 0.15 * 0.02                                  # accepted and in truth a failure
    good = 0.85 * (1 - 0.05 - 0.10)                    # accepted and in truth fine
    assert defect_rate_among_accepted(p, far, frr, abstain_pos) == pytest.approx(bad / (good + bad), rel=1e-12)
    assert defect_rate_among_accepted(p, far, frr, abstain_pos) == pytest.approx(0.004135, abs=5e-7)
    assert defect_rate_among_accepted(p, 0.0, frr, abstain_pos) == 0.0      # a check that never errs
    assert defect_rate_among_accepted(0.0, 0.3, 0.0, 0.0) == 1.0            # every claim is bad
    assert math.isnan(defect_rate_among_accepted(1.0, 0.3, 0.6, 0.4))       # nothing is accepted


def test_s2_negatives_a_stage_needs_before_it_can_accept():
    """Section 2: at alpha = 0.02 that is 49 negatives, at alpha = 0.005 it is 199."""
    assert min_negatives_to_accept(0.02) == 49
    assert min_negatives_to_accept(0.005) == 199
    assert conformal_accept_index(48, 0.02) is None and conformal_accept_index(49, 0.02) == 49
    assert conformal_accept_index(198, 0.005) is None and conformal_accept_index(199, 0.005) == 199
    assert min_positives_to_reject(0.05) == 19
    assert conformal_reject_index(18, 0.05) is None and conformal_reject_index(19, 0.05) == 1


def test_s7_598_clean_audits_certify_half_a_percent():
    lo, hi = clopper_pearson(0, 598, conf=0.90)        # two-sided 90% has the one-sided 95% upper end
    assert lo == 0.0 and hi == pytest.approx(cp_upper(0, 598), rel=1e-12)
    assert cp_upper(0, 598) <= 0.005 < cp_upper(0, 597)
    assert 0.99 < cp_upper(0, 598) * 598 / 3 < 1.0     # "about 3 / n"


def test_s10_robots_per_assistant_examples():
    """Section 10: 6 an hour at 30 s gives 20 and 14. 2 an hour at 45 s gives 40 and 28."""
    assert robots_per_assistant(6, 30) == pytest.approx(20.0, abs=1e-9)
    assert robots_per_assistant(6, 30, utilization=0.7) == pytest.approx(14.0, abs=1e-9)
    assert robots_per_assistant(2, 45) == pytest.approx(40.0, abs=1e-9)
    assert robots_per_assistant(2, 45, utilization=0.7) == pytest.approx(28.0, abs=1e-9)
    assert busy_share(6, 30) == pytest.approx(0.05, abs=1e-12)
    assert robots_per_assistant(6, 30, 0.7) == pytest.approx(0.7 / busy_share(6, 30), rel=1e-12)


def test_docs07_wilson_intervals_quoted_for_the_e1_power_statement():
    """docs/07 E1: 2% of 400 is about 1.0% to 3.9%, 30% of 400 about 26% to 35%, 30% of 160 about 23% to 38%."""
    lo, hi = wilson(8, 400)
    assert (f"{lo:.1%}", f"{hi:.1%}") == ("1.0%", "3.9%")
    lo, hi = wilson(120, 400)
    assert (f"{lo:.0%}", f"{hi:.0%}") == ("26%", "35%")
    lo, hi = wilson(48, 160)
    assert (f"{lo:.0%}", f"{hi:.0%}") == ("23%", "38%")


# =============================================================================================
# Intervals: edges, independent cross-checks, defining equations, exact coverage.
# =============================================================================================

GRID = [(0, 1), (1, 1), (0, 7), (3, 7), (7, 7), (0, 30), (1, 30), (29, 30), (30, 30),
        (16, 5000), (2500, 5000), (4999, 5000)]
CONFS = [0.8, 0.9, 0.95, 0.99]


def test_wilson_is_exact_at_the_edges():
    """Carried over from loopkit, whose report once crashed at a perfect score (docs/10_reuse_map.md).

    Without the exact edge, 60 of 60 gives an upper end of 0.9999999999999998, which is below the point estimate.
    """
    for n in (1, 20, 60, 61, 997):
        assert wilson(n, n)[1] == 1.0
        assert wilson(0, n)[0] == 0.0
    lo, hi = wilson(0, 20)
    assert lo == 0.0 and hi == pytest.approx(0.1611, abs=1e-4)
    lo, hi = wilson(20, 20)
    assert lo == pytest.approx(0.8389, abs=1e-4) and hi == 1.0
    assert wilson(30, 60) == pytest.approx((0.3773, 0.6227), abs=1e-4)


def test_clopper_pearson_is_exact_at_the_edges():
    for n in (1, 30, 997):
        assert clopper_pearson(0, n)[0] == 0.0 and clopper_pearson(n, n)[1] == 1.0
        assert cp_lower(0, n) == 0.0 and cp_upper(n, n) == 1.0
        # the closed forms at the other 2 edges
        assert cp_upper(0, n) == pytest.approx(1 - 0.05 ** (1 / n), rel=1e-12)
        assert cp_lower(n, n) == pytest.approx(0.05 ** (1 / n), rel=1e-12)
        assert clopper_pearson(0, n)[1] == pytest.approx(1 - 0.025 ** (1 / n), rel=1e-12)
        assert clopper_pearson(n, n)[0] == pytest.approx(0.025 ** (1 / n), rel=1e-12)


@pytest.mark.parametrize("conf", CONFS)
def test_wilson_matches_scipy(conf):
    for k, n in GRID:
        ref = sps.binomtest(k, n).proportion_ci(confidence_level=conf, method="wilson")
        assert wilson(k, n, conf) == pytest.approx((ref.low, ref.high), abs=1e-12), (k, n)


@pytest.mark.parametrize("conf", CONFS)
def test_clopper_pearson_matches_scipy(conf):
    for k, n in GRID:
        ref = sps.binomtest(k, n).proportion_ci(confidence_level=conf, method="exact")
        assert clopper_pearson(k, n, conf) == pytest.approx((ref.low, ref.high), rel=1e-9, abs=1e-15), (k, n)


@pytest.mark.parametrize("conf", CONFS)
def test_one_sided_bounds_match_scipy(conf):
    for x, n in GRID:
        upper = sps.binomtest(x, n, alternative="less").proportion_ci(confidence_level=conf, method="exact").high
        lower = sps.binomtest(x, n, alternative="greater").proportion_ci(confidence_level=conf, method="exact").low
        assert cp_upper(x, n, conf) == pytest.approx(upper, rel=1e-9), (x, n)
        assert cp_lower(x, n, conf) == pytest.approx(lower, rel=1e-9, abs=1e-15), (x, n)


@pytest.mark.parametrize("x, n", [(0, 30), (1, 30), (16, 5000), (17, 5000), (299, 300)])
def test_cp_upper_solves_its_defining_equation(x, n):
    """At the bound, seeing x or fewer events has probability 1 - conf."""
    for conf in (0.9, 0.95, 0.99):
        assert sps.binom.cdf(x, n, cp_upper(x, n, conf)) == pytest.approx(1 - conf, rel=1e-9)


@pytest.mark.parametrize("x, n", [(1, 30), (30, 30), (16, 5000), (4999, 5000)])
def test_cp_lower_solves_its_defining_equation(x, n):
    """At the bound, seeing x or more events has probability 1 - conf."""
    for conf in (0.9, 0.95, 0.99):
        assert sps.binom.sf(x - 1, n, cp_lower(x, n, conf)) == pytest.approx(1 - conf, rel=1e-9)


def test_intervals_mirror_when_events_and_non_events_swap():
    for k, n in GRID:
        lo, hi = wilson(k, n)
        mirrored = wilson(n - k, n)
        assert (lo, hi) == pytest.approx((1 - mirrored[1], 1 - mirrored[0]), abs=1e-12)
        assert cp_lower(k, n) == pytest.approx(1 - cp_upper(n - k, n), abs=1e-12)
        # a two-sided interval at conf is the pair of one-sided bounds at (1 + conf) / 2
        assert clopper_pearson(k, n, 0.90) == pytest.approx((cp_lower(k, n, 0.95), cp_upper(k, n, 0.95)), abs=1e-12)


def test_intervals_contain_the_point_estimate_and_widen_with_confidence():
    for k, n in GRID:
        for interval in (wilson, clopper_pearson):
            lo, hi = interval(k, n)
            assert 0.0 <= lo <= k / n <= hi <= 1.0
            lo99, hi99 = interval(k, n, 0.99)
            assert lo99 <= lo and hi <= hi99
        assert cp_lower(k, n) <= k / n <= cp_upper(k, n)


def test_cp_upper_covers_every_true_rate_at_least_95_times_in_100():
    """Exact coverage, summed over the binomial. A bound built at the wrong confidence fails this."""
    n = 40
    events = np.arange(n + 1)
    upper = np.array([cp_upper(int(x), n) for x in events])
    lower_2, upper_2 = np.array([clopper_pearson(int(x), n) for x in events]).T
    worst_one_sided, worst_two_sided = 1.0, 1.0
    for p in np.linspace(0.002, 0.7, 350):
        pmf = sps.binom.pmf(events, n, p)
        worst_one_sided = min(worst_one_sided, pmf[upper >= p].sum())
        worst_two_sided = min(worst_two_sided, pmf[(lower_2 <= p) & (p <= upper_2)].sum())
    assert worst_one_sided >= 0.95 - 1e-12
    assert worst_two_sided >= 0.95 - 1e-12
    # and the one-sided bound is not padded: somewhere on the grid its coverage comes close to 95%
    assert worst_one_sided < 0.96


@pytest.mark.parametrize("p_max", [0.2, 0.05, 0.01, 0.005, 0.001])
@pytest.mark.parametrize("failures", [0, 1, 3, 10])
def test_n_to_certify_is_the_smallest_count_that_certifies(p_max, failures):
    n = n_to_certify(p_max, failures)
    assert cp_upper(failures, n) <= p_max < cp_upper(failures, n - 1)
    # the same count straight from the binomial: the smallest n at which seeing `failures` or
    # fewer events has probability at most 1 - conf when the true rate sits exactly at the bar
    assert sps.binom.cdf(failures, n, p_max) <= 0.05 < sps.binom.cdf(failures, n - 1, p_max)


def test_n_to_certify_follows_confidence_and_allowed_failures():
    assert n_to_certify(0.005, conf=0.99) == math.ceil(math.log(0.01) / math.log(0.995)) == 919
    assert n_to_certify(0.005, conf=0.90) == math.ceil(math.log(0.10) / math.log(0.995)) == 460
    counts = [n_to_certify(0.005, k) for k in range(0, 20)]
    assert counts == sorted(counts) and len(set(counts)) == len(counts)       # each failure costs more trials
    assert n_to_certify(0.95) == 1                                            # 1 clean trial bounds the rate by 0.95
    assert n_to_certify(0.9499) == 2


def test_max_failures_to_certify_is_the_inverse_of_n_to_certify():
    for failures in (0, 1, 5, 16):
        n = n_to_certify(0.005, failures)
        assert max_failures_to_certify(n, 0.005) == failures
        assert max_failures_to_certify(n - 1, 0.005) == (failures - 1 if failures else None)


# =============================================================================================
# Trial planning: two proportions, Wald's sequential test, McNemar.
# =============================================================================================

@pytest.mark.parametrize("p1, p2, alpha, power, per_arm", [
    (0.9, 0.8, 0.01, 0.9, 377),
    (0.2, 0.3, 0.05, 0.8, 294),
    (0.5, 0.6, 0.05, 0.8, 388),
    (0.6, 0.7, 0.05, 0.9, 477),
])
def test_n_two_proportions_is_the_section_8_formula(p1, p2, alpha, power, per_arm):
    """4 more pairs of rates, each worked a second way with the normal quantiles from scipy.stats."""
    z_a, z_b = sps.norm.ppf(1 - alpha / 2), sps.norm.ppf(power)
    pbar = (p1 + p2) / 2
    n = (z_a * math.sqrt(2 * pbar * (1 - pbar)) + z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / (p1 - p2) ** 2
    assert math.ceil(n) == per_arm
    assert n_two_proportions(p1, p2, alpha=alpha, power=power) == per_arm


def test_n_two_proportions_is_symmetric_and_grows_with_power_and_closeness():
    assert n_two_proportions(0.99, 0.95) == n_two_proportions(0.95, 0.99) == 285
    assert n_two_proportions(0.95, 0.99, power=0.9) > 285
    assert n_two_proportions(0.95, 0.99, alpha=0.01) > 285
    assert n_two_proportions(0.95, 0.97) > n_two_proportions(0.95, 0.99)
    assert n_two_proportions(0.0, 1.0) >= 1


def test_wald_boundaries_and_steps_are_the_section_8_formulas():
    lower, upper = wald_boundaries(alpha=0.05, beta=0.2)
    assert upper == pytest.approx(math.log(0.8 / 0.05), rel=1e-12)
    assert upper == pytest.approx(math.log(16), rel=1e-12)
    assert lower == pytest.approx(math.log(0.2 / 0.95), rel=1e-12)
    assert (shown(lower, 3), shown(upper, 3)) == ("-1.558", "2.773")
    test = WaldSPRT(p0=0.95, p1=0.99, alpha=0.05, beta=0.2)
    assert test.boundaries == (lower, upper) == (test.lower, test.upper)
    assert test.step_success == pytest.approx(math.log(0.99 / 0.95), rel=1e-12)
    assert test.step_failure == pytest.approx(math.log(0.01 / 0.05), rel=1e-12)
    assert (test.decision, test.n, test.llr) == ("continue", 0, 0.0)


def test_wald_stops_for_p1_after_68_successes_in_a_row():
    test = WaldSPRT(p0=0.95, p1=0.99)
    decisions = [test.update(True) for _ in range(68)]
    assert decisions[:67] == ["continue"] * 67 and decisions[67] == "p1"
    assert (test.n, test.successes, test.failures) == (68, 68, 0)
    assert 67 * math.log(0.99 / 0.95) < test.upper <= 68 * math.log(0.99 / 0.95)      # why it is 68
    assert test.llr == pytest.approx(68 * math.log(0.99 / 0.95), rel=1e-12)


def test_wald_stops_for_p0_on_an_early_failure():
    first = WaldSPRT(p0=0.95, p1=0.99)
    assert first.update(False) == "p0" and first.n == 1            # ln(0.2) = -1.609 is below ln(0.2 / 0.95) = -1.558
    second = WaldSPRT(p0=0.95, p1=0.99)
    assert [second.update(x) for x in (True, False)] == ["continue", "p0"]
    assert second.llr == pytest.approx(math.log(0.99 / 0.95) + math.log(0.2), rel=1e-12)
    third = WaldSPRT(p0=0.95, p1=0.99)
    assert [third.update(x) for x in (True, True, False)] == ["continue"] * 3       # 2 successes buy 1 failure
    assert third.lower < third.llr < third.upper


def test_wald_decision_is_final():
    test = WaldSPRT(p0=0.95, p1=0.99)
    test.update(False)
    snapshot = (test.decision, test.n, test.successes, test.llr)
    for _ in range(200):
        assert test.update(True) == "p0"
    assert (test.decision, test.n, test.successes, test.llr) == snapshot
    with pytest.raises(AttributeError):
        test.decision = "p1"                  # read-only: only the test itself moves it
    assert test.decision == "p0"


STEP_CAP = 10_000      # a sequential run that has not stopped by now is broken, and the test fails instead of hanging


def _sprt_error_rate(seed: int, p_true: float, wrong: str, runs: int = 2000) -> float:
    rng = np.random.default_rng(seed)
    errors = 0
    for _ in range(runs):
        test = WaldSPRT(p0=0.80, p1=0.95, alpha=0.05, beta=0.2)
        for _ in range(STEP_CAP):
            if test.update(bool(rng.random() < p_true)) != "continue":
                break
        else:
            pytest.fail(f"WaldSPRT reached no decision in {STEP_CAP} trials at a true rate of {p_true}")
        errors += test.decision == wrong
    return errors / runs


def test_wald_error_rates_stay_within_walds_bounds():
    """Wald's inequalities: the chance of a wrong stop is at most alpha / (1 - beta) and beta / (1 - alpha).

    2,000 seeded runs at each of the 2 rates. The allowance is 3 binomial standard errors.
    """
    alpha, beta, runs = 0.05, 0.2, 2000
    bound_a, bound_b = alpha / (1 - beta), beta / (1 - alpha)
    false_p1 = _sprt_error_rate(4105, p_true=0.80, wrong="p1", runs=runs)
    false_p0 = _sprt_error_rate(4106, p_true=0.95, wrong="p0", runs=runs)
    assert false_p1 <= bound_a + 3 * math.sqrt(bound_a * (1 - bound_a) / runs)
    assert false_p0 <= bound_b + 3 * math.sqrt(bound_b * (1 - bound_b) / runs)
    assert false_p1 > 0 and false_p0 > 0          # the test does stop both ways, so the bounds were exercised


def test_mcnemar_exact_p_values_worked_by_hand():
    assert mcnemar(0, 5)["p_exact"] == pytest.approx(2 * 1 / 32, rel=1e-12)             # 2 * P(X = 0), n = 5
    assert mcnemar(1, 5)["p_exact"] == pytest.approx(2 * (1 + 6) / 64, rel=1e-12)       # 2 * P(X <= 1), n = 6
    assert mcnemar(2, 10)["p_exact"] == pytest.approx(2 * (1 + 12 + 66) / 4096, rel=1e-12)
    assert mcnemar(5, 1)["p_exact"] == mcnemar(1, 5)["p_exact"]                         # the order does not matter
    assert mcnemar(1, 5) == {"b": 1, "c": 5, "n_discordant": 6, "p_exact": pytest.approx(0.21875),
                             "chi2": pytest.approx(1.5), "p_chi2": pytest.approx(sps.chi2.sf(1.5, 1), rel=1e-9)}


def test_mcnemar_continuity_corrected_chi_square():
    result = mcnemar(121, 59)                                    # by hand: (|121 - 59| - 1)^2 / 180 = 3721 / 180
    assert result["chi2"] == pytest.approx(61**2 / 180, rel=1e-12)
    assert shown(result["chi2"], 3) == "20.672"
    assert result["p_chi2"] == pytest.approx(sps.chi2.sf(61**2 / 180, 1), rel=1e-9)
    assert result["p_exact"] == pytest.approx(sps.binomtest(59, 180, 0.5).pvalue, rel=1e-9)


def test_mcnemar_matches_scipy_binomial_test_on_a_grid():
    for b in range(0, 12):
        for c in range(0, 12):
            if b != c:
                assert mcnemar(b, c)["p_exact"] == pytest.approx(sps.binomtest(b, b + c, 0.5).pvalue, rel=1e-9), (b, c)


def test_mcnemar_with_no_difference_gives_no_evidence():
    for b in (0, 1, 5, 40):
        result = mcnemar(b, b)
        assert (result["p_exact"], result["chi2"], result["p_chi2"]) == (1.0, 0.0, 1.0)
    assert mcnemar(0, 0)["n_discordant"] == 0


# =============================================================================================
# Conformal thresholds: ranks, thresholds, the floating point guard.
# =============================================================================================

def test_accept_threshold_is_the_kth_smallest_negative_score():
    scores = np.random.default_rng(4110).permutation(np.arange(1.0, 100.0))       # 1..99 in random order
    before = scores.copy()
    assert conformal_accept_threshold(scores, 0.02) == 98.0                       # rank 98 of 99
    assert conformal_accept_threshold(scores, 0.05) == 95.0                       # ceil(100 * 0.95)
    assert conformal_accept_threshold(scores, 0.5) == 50.0
    assert conformal_accept_threshold(list(scores), 0.02) == 98.0                 # a plain list works too
    assert np.array_equal(scores, before)                                         # the input is left alone


def test_reject_threshold_is_the_jth_smallest_positive_score():
    scores = np.random.default_rng(4111).permutation(np.arange(1.0, 100.0))
    assert conformal_reject_index(99, 0.05) == 5                                  # floor(100 * 0.05)
    assert conformal_reject_threshold(scores, 0.05) == 5.0
    assert conformal_reject_threshold(scores, 0.02) == 2.0
    assert conformal_reject_threshold(scores, 0.5) == 50.0


def test_a_stage_with_too_few_negatives_never_accepts():
    scores = np.linspace(0.0, 1.0, 40)
    tau = conformal_accept_threshold(scores, 0.02)
    assert tau == math.inf
    for score in (0.5, 1.0, 1e300, math.inf):
        assert decide(score, -math.inf, tau) == ABSTAIN
    assert conformal_accept_threshold([], 0.02) == math.inf                       # no negatives at all
    assert conformal_accept_index(0, 0.02) is None
    assert conformal_accept_threshold(np.linspace(0.0, 1.0, 500), 0.0) == math.inf    # a stage with no accept budget
    assert conformal_accept_index(10**6, 0.0) is None


def test_a_stage_with_too_few_positives_never_rejects():
    scores = np.linspace(0.0, 1.0, 18)
    tau = conformal_reject_threshold(scores, 0.05)
    assert tau == -math.inf
    for score in (0.5, 0.0, -1e300, -math.inf):
        assert decide(score, tau, math.inf) == ABSTAIN
    assert conformal_reject_threshold([], 0.05) == -math.inf
    assert conformal_reject_index(0, 0.05) is None
    assert conformal_reject_threshold(np.linspace(0.0, 1.0, 500), 0.0) == -math.inf
    assert conformal_reject_index(10**6, 0.0) is None


def test_nan_scores_are_left_out_of_calibration():
    rng = np.random.default_rng(4112)
    finite = rng.normal(size=99)
    with_gaps = rng.permutation(np.concatenate([finite, np.full(20, np.nan)]))
    assert conformal_accept_threshold(with_gaps, 0.02) == conformal_accept_threshold(finite, 0.02)
    assert conformal_reject_threshold(with_gaps, 0.05) == conformal_reject_threshold(finite, 0.05)
    assert conformal_accept_threshold(list(finite) + [None], 0.02) == conformal_accept_threshold(finite, 0.02)
    # 40 usable scores are too few at alpha = 0.02, however many NaN rows come with them
    assert conformal_accept_threshold(np.concatenate([finite[:40], np.full(500, np.nan)]), 0.02) == math.inf


def test_infinite_scores_stay_in_calibration():
    """A negative that scored +inf is a real false accept waiting to happen. Dropping it would loosen the bound."""
    finite = np.linspace(0.0, 1.0, 97)
    negatives = np.concatenate([finite, [math.inf, math.inf]])                # 99 negatives, rank 98 is inf
    assert conformal_accept_threshold(negatives, 0.02) == math.inf
    assert conformal_accept_threshold(np.concatenate([finite, [-math.inf, -math.inf]]), 0.02) == finite[-2]
    positives = np.concatenate([np.linspace(0.0, 1.0, 94), [-math.inf] * 5])  # 99 positives, rank 5 is -inf
    assert conformal_reject_threshold(positives, 0.05) == -math.inf
    positives = np.concatenate([np.linspace(0.0, 1.0, 94), [math.inf] * 5])   # now rank 5 is the 5th finite score
    assert conformal_reject_threshold(positives, 0.05) == pytest.approx(4 / 93)


def test_ranks_stay_inside_1_to_n_at_extreme_levels():
    nearly_1 = 1 - 1e-14
    assert conformal_accept_index(10, nearly_1) == 1          # the lowest rank there is, never rank 0
    assert conformal_accept_threshold([3.0, 1.0, 2.0], nearly_1) == 1.0
    assert conformal_reject_index(10, nearly_1) == 10         # the highest rank there is, never rank 11
    assert conformal_reject_threshold([3.0, 1.0, 2.0], nearly_1) == 3.0


def test_rank_arithmetic_is_guarded_against_floating_point_error():
    # the spec's example: (99 + 1) * (1 - 0.02) must give exactly 98
    assert conformal_accept_index(99, 0.02) == 98
    # 1000 * (1 - 0.059) evaluates to 941.0000000000001, so a plain ceil gives 942
    assert math.ceil((999 + 1) * (1 - 0.059)) == 942
    assert conformal_accept_index(999, 0.059) == 941
    # 3000 * 0.009 evaluates to 26.999999999999996, so a plain floor gives 26
    assert math.floor((2999 + 1) * 0.009) == 26
    assert conformal_reject_index(2999, 0.009) == 27
    assert math.floor((749 + 1) * 0.036) == 26
    assert conformal_reject_index(749, 0.036) == 27


LEVELS = [Fraction(i, 1000) for i in (1, 2, 5, 9, 10, 18, 20, 25, 36, 50, 59, 72, 100, 172, 200, 500, 900)]
LEVELS += [Fraction(2, 100) / 3, Fraction(2, 100) / 7, Fraction(1, 100) / 3]      # budget splits


def test_ranks_equal_exact_rational_arithmetic_on_a_grid():
    for level in LEVELS:
        as_float = float(level)
        for n in range(0, 1201):
            k = math.ceil((n + 1) * (1 - level))
            j = math.floor((n + 1) * level)
            assert conformal_accept_index(n, as_float) == (k if k <= n else None), (n, as_float)
            assert conformal_reject_index(n, as_float) == (j if j >= 1 else None), (n, as_float)


def test_minimum_counts_agree_with_the_rank_functions():
    for level in LEVELS:
        as_float = float(level)
        fewest = max(1, math.ceil(1 / level - 1))
        assert min_negatives_to_accept(as_float) == fewest
        assert conformal_accept_index(fewest, as_float) is not None
        assert fewest == 1 or conformal_accept_index(fewest - 1, as_float) is None
        assert min_positives_to_reject(as_float) == fewest
        assert conformal_reject_index(fewest, as_float) is not None
        assert fewest == 1 or conformal_reject_index(fewest - 1, as_float) is None


@pytest.mark.parametrize("n, alpha", [(99, 0.02), (49, 0.02), (40, 0.02), (199, 0.005), (120, 0.05), (300, 0.013),
                                      (999, 0.059)])
def test_leave_one_out_false_accepts_equal_the_conformal_level(n, alpha):
    """The exchangeability argument, made exact.

    Take n + 1 distinct scores. Hold each one out in turn, calibrate on the other n, and ask if the
    held-out score is accepted. Each score is equally likely to be the new one, so the share accepted
    is the false-accept rate. It must be at most alpha, and within 1 rank of alpha.
    """
    scores = np.random.default_rng(n).permutation(n + 1).astype(float)
    accepted = 0
    for i in range(n + 1):
        tau_accept = conformal_accept_threshold(np.delete(scores, i), alpha)
        accepted += decide(scores[i], -math.inf, tau_accept) == ACCEPT
    k = conformal_accept_index(n, alpha)
    assert accepted == (0 if k is None else n + 1 - k)
    assert accepted / (n + 1) <= alpha + 1e-12
    assert accepted / (n + 1) > alpha - 1 / (n + 1)


@pytest.mark.parametrize("m, beta", [(99, 0.05), (19, 0.05), (18, 0.05), (199, 0.02), (300, 0.013), (2999, 0.009)])
def test_leave_one_out_false_rejects_equal_the_conformal_level(m, beta):
    scores = np.random.default_rng(m).permutation(m + 1).astype(float)
    rejected = 0
    for i in range(m + 1):
        tau_reject = conformal_reject_threshold(np.delete(scores, i), beta)
        rejected += decide(scores[i], tau_reject, math.inf) == REJECT
    j = conformal_reject_index(m, beta)
    assert rejected == (0 if j is None else j)
    assert rejected / (m + 1) <= beta + 1e-12
    assert rejected / (m + 1) > beta - 1 / (m + 1)


def test_leave_one_out_bound_holds_when_some_scores_are_missing():
    """A missing score abstains and is left out of calibration. The false-accept share stays within alpha."""
    alpha = 0.02
    scores = np.random.default_rng(4116).permutation(np.concatenate([np.arange(100.0), np.full(20, np.nan)]))
    accepted = 0
    for i in range(scores.size):
        tau_accept = conformal_accept_threshold(np.delete(scores, i), alpha)
        accepted += decide(scores[i], -math.inf, tau_accept) == ACCEPT
    assert accepted == 2                      # the top 2 of the 100 real scores, as if the 20 gaps were not there
    assert accepted / scores.size <= alpha


def test_thresholds_move_the_right_way_with_the_level():
    scores = np.random.default_rng(4117).normal(size=500)
    levels = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    accept = [conformal_accept_threshold(scores, level) for level in levels]
    reject = [conformal_reject_threshold(scores, level) for level in levels]
    assert accept == sorted(accept, reverse=True) and len(set(accept)) == len(levels)     # a larger alpha lowers the bar
    assert reject == sorted(reject) and len(set(reject)) == len(levels)                   # a larger beta raises it
    assert max(reject) < min(accept)


# =============================================================================================
# The 3-way decision: ties, crossing thresholds, missing scores.
# =============================================================================================

def test_decide_accepts_above_rejects_below_and_abstains_between():
    assert decide(0.9, tau_reject=0.2, tau_accept=0.7) == ACCEPT == "accept"
    assert decide(0.1, tau_reject=0.2, tau_accept=0.7) == REJECT == "reject"
    assert decide(0.5, tau_reject=0.2, tau_accept=0.7) == ABSTAIN == "abstain"


def test_a_score_equal_to_tau_accept_is_not_accepted():
    assert decide(0.7, 0.2, 0.7) == ABSTAIN
    assert decide(math.nextafter(0.7, 1.0), 0.2, 0.7) == ACCEPT             # the next float up is accepted
    assert decide(math.inf, -math.inf, math.inf) == ABSTAIN                 # even at infinity a tie is a tie


def test_a_score_equal_to_tau_reject_is_not_rejected():
    assert decide(0.2, 0.2, 0.7) == ABSTAIN
    assert decide(math.nextafter(0.2, 0.0), 0.2, 0.7) == REJECT             # the next float down is rejected
    assert decide(-math.inf, -math.inf, math.inf) == ABSTAIN


def test_a_calibration_score_that_became_the_threshold_is_itself_not_accepted():
    negatives = [0.1] * 60 + [0.4] * 30 + [0.9] * 9                         # 99 negatives with heavy ties
    tau_accept = conformal_accept_threshold(negatives, 0.02)                # rank 98 lands in the 0.9 block
    assert tau_accept == 0.9
    assert decide(0.9, -math.inf, tau_accept) == ABSTAIN
    positives = [0.3] * 10 + [0.8] * 89                                     # 99 positives, rank 5 is 0.3
    tau_reject = conformal_reject_threshold(positives, 0.05)
    assert tau_reject == 0.3
    assert decide(0.3, tau_reject, math.inf) == ABSTAIN


def test_crossed_thresholds_accept_only_above_the_larger_and_reject_only_below_the_smaller():
    tau_reject, tau_accept = 0.8, 0.3                                       # crossed: reject sits above accept
    assert decide(0.9, tau_reject, tau_accept) == ACCEPT
    assert decide(0.8, tau_reject, tau_accept) == ABSTAIN                   # a tie at the larger one
    assert decide(0.5, tau_reject, tau_accept) == ABSTAIN                   # between them both rules would fire
    assert decide(0.3, tau_reject, tau_accept) == ABSTAIN                   # a tie at the smaller one
    assert decide(0.2, tau_reject, tau_accept) == REJECT


def test_every_decision_respects_both_thresholds_whether_or_not_they_cross():
    """Accept implies the score is above both thresholds. Reject implies it is below both."""
    rng = np.random.default_rng(4113)
    seen = {ACCEPT: 0, REJECT: 0, ABSTAIN: 0}
    crossed = 0
    for score, tau_reject, tau_accept in rng.normal(size=(3000, 3)):
        decision = decide(score, tau_reject, tau_accept)
        seen[decision] += 1
        crossed += tau_reject > tau_accept
        assert (decision == ACCEPT) == (score > max(tau_reject, tau_accept))
        assert (decision == REJECT) == (score < min(tau_reject, tau_accept))
    assert min(seen.values()) > 300 and 1000 < crossed < 2000               # all 3 outcomes and both orders occur


def test_a_missing_score_abstains():
    assert decide(math.nan, 0.2, 0.7) == ABSTAIN
    assert decide(None, 0.2, 0.7) == ABSTAIN
    assert decide(np.float64("nan"), 0.2, 0.7) == ABSTAIN
    assert decide(math.nan, -math.inf, -math.inf) == ABSTAIN                # even when every real score is accepted
    assert decide(-1e300, -math.inf, -math.inf) == ACCEPT
    assert decide(math.nan, math.inf, math.inf) == ABSTAIN                  # even when every real score is rejected
    assert decide(1e300, math.inf, math.inf) == REJECT


def test_decide_many_is_decide_applied_to_each_score():
    scores = [0.9, 0.7, 0.5, 0.2, 0.1, math.nan, None, math.inf, -math.inf, 3]
    for tau_reject, tau_accept in [(0.2, 0.7), (0.7, 0.2), (-math.inf, math.inf), (0.5, 0.5)]:
        many = decide_many(scores, tau_reject, tau_accept)
        assert many.shape == (len(scores),)
        assert list(many) == [decide(score, tau_reject, tau_accept) for score in scores]
    assert list(decide_many([], 0.2, 0.7)) == []
    assert list(decide_many(np.array([0.9, 0.1]), 0.2, 0.7)) == [ACCEPT, REJECT]


def test_decision_strings_are_the_schema_vocabulary():
    """The strings ``decide`` returns are the ones a Verdict accepts (avail/schemas/kinds.py)."""
    # Since S5 kinds.py reads the predicate texts from the task files, so it is imported as part of its package.
    from avail.schemas import kinds

    assert set(kinds.DECISIONS) == {ACCEPT, REJECT, ABSTAIN}


# =============================================================================================
# Simulation tests (section 11).
#
# What "within sampling error" means here. Each simulation repeats N_DRAWS = 2,000 calibration
# draws. Draw i calibrates on fresh synthetic scores and then measures 1 error rate, rate_i, on
# fresh scores that the calibration never saw. The bound speaks about the mean of rate_i. Its
# standard error is se = sd(rate_i) / sqrt(2,000). "At most alpha within sampling error" is
#
#     mean(rate_i) <= alpha + 3 * se
#
# With n = 99 negatives and alpha = 0.02 the exact expected false-accept rate is 1 - 98/100 =
# 0.02 and se is about 0.00034, so the check allows a mean of up to about 0.0210. A threshold
# taken 1 rank too low (97 in place of 98) has an expected rate of 3/100 = 0.03, which is about
# 24 standard errors above alpha, so the same check fails. The wrong-rank tests prove that.
#
# The seeds are numpy seeds for synthetic scores. They are unrelated to the station seed budget
# in config/experiment.yaml, and they sit in its dev range anyway.
# =============================================================================================

N_DRAWS = 2000
SIGMAS = 3.0


def _mean_and_se(per_draw: np.ndarray) -> tuple[float, float]:
    return float(per_draw.mean()), float(per_draw.std(ddof=1) / math.sqrt(per_draw.size))


def _negatives(rng: np.random.Generator, size: int) -> np.ndarray:
    """Synthetic scores of states where the predicate is false: 70% clear failures, 30% near misses."""
    near_miss = rng.random(size) < 0.3
    return np.where(near_miss, rng.normal(2.0, 0.5, size), rng.normal(0.0, 1.0, size))


def _positives(rng: np.random.Generator, size: int) -> np.ndarray:
    return rng.normal(3.5, 0.8, size)


def _false_accept_per_draw(seed: int, threshold_from, n_cal: int = 99, n_test: int = 500, draw=_negatives) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rates = np.empty(N_DRAWS)
    for i in range(N_DRAWS):
        tau_accept = threshold_from(draw(rng, n_cal))
        rates[i] = np.mean(decide_many(draw(rng, n_test), -math.inf, tau_accept) == ACCEPT)
    return rates


def _false_reject_per_draw(seed: int, threshold_from, m_cal: int = 99, n_test: int = 500) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rates = np.empty(N_DRAWS)
    for i in range(N_DRAWS):
        tau_reject = threshold_from(_positives(rng, m_cal))
        rates[i] = np.mean(decide_many(_positives(rng, n_test), tau_reject, math.inf) == REJECT)
    return rates


def test_sim_mean_false_accept_over_2000_calibration_draws_is_at_most_alpha():
    alpha, n = 0.02, 99
    rates = _false_accept_per_draw(4101, lambda negatives: conformal_accept_threshold(negatives, alpha), n_cal=n)
    mean, se = _mean_and_se(rates)
    assert mean <= alpha + SIGMAS * se, f"mean false-accept {mean:.5f} exceeds alpha {alpha} by more than 3 se ({se:.5f})"
    # The level is also used in full: the exact expected rate at n = 99 is 1 - 98/100, which is alpha.
    exact = 1 - conformal_accept_index(n, alpha) / (n + 1)
    assert exact == pytest.approx(alpha, abs=1e-12)
    assert abs(mean - exact) <= SIGMAS * se, f"mean false-accept {mean:.5f} is not the exact level {exact} within 3 se"


def test_sim_a_threshold_1_rank_too_low_breaks_the_bound():
    """The simulation has teeth: rank k - 1 in place of k is caught by the same check."""
    alpha, n = 0.02, 99
    k = conformal_accept_index(n, alpha)
    wrong = lambda negatives: float(np.sort(negatives)[(k - 1) - 1])        # the (k - 1)-th smallest score
    rates = _false_accept_per_draw(4101, wrong, n_cal=n)
    mean, se = _mean_and_se(rates)
    assert not mean <= alpha + SIGMAS * se
    assert mean == pytest.approx(3 / 100, abs=SIGMAS * se)                  # the exact expected rate at rank 97


def test_sim_mean_false_reject_over_2000_calibration_draws_is_at_most_beta():
    beta, m = 0.05, 99
    rates = _false_reject_per_draw(4102, lambda positives: conformal_reject_threshold(positives, beta), m_cal=m)
    mean, se = _mean_and_se(rates)
    assert mean <= beta + SIGMAS * se
    exact = conformal_reject_index(m, beta) / (m + 1)                       # 5/100, which is beta
    assert exact == pytest.approx(beta, abs=1e-12)
    assert abs(mean - exact) <= SIGMAS * se


def test_sim_a_reject_threshold_1_rank_too_high_breaks_the_bound():
    beta, m = 0.05, 99
    j = conformal_reject_index(m, beta)
    wrong = lambda positives: float(np.sort(positives)[(j + 1) - 1])        # the (j + 1)-th smallest score
    rates = _false_reject_per_draw(4102, wrong, m_cal=m)
    mean, se = _mean_and_se(rates)
    assert not mean <= beta + SIGMAS * se
    assert mean == pytest.approx(6 / 100, abs=SIGMAS * se)


def _coarse_negatives(rng: np.random.Generator, size: int) -> np.ndarray:
    """A score with only 10 possible values, so ties with the threshold are common."""
    return rng.binomial(9, 0.3, size).astype(float)


def test_sim_the_tie_rule_keeps_the_bound_on_coarse_scores():
    """With tied scores the strict rule (accept only above tau) holds the bound. Accepting ties breaks it."""
    alpha, n = 0.02, 99
    threshold = lambda negatives: conformal_accept_threshold(negatives, alpha)
    rates = _false_accept_per_draw(4108, threshold, n_cal=n, draw=_coarse_negatives)
    mean, se = _mean_and_se(rates)
    assert 0 < mean <= alpha + SIGMAS * se

    rng = np.random.default_rng(4108)
    ties_accepted = np.empty(N_DRAWS)
    for i in range(N_DRAWS):
        tau_accept = threshold(_coarse_negatives(rng, n))
        ties_accepted[i] = np.mean(_coarse_negatives(rng, 500) >= tau_accept)       # the wrong rule
    mean_wrong, se_wrong = _mean_and_se(ties_accepted)
    assert mean_wrong > alpha + SIGMAS * se_wrong


CASCADE_ALPHAS = (0.01, 0.005, 0.005)       # sum 0.02
CASCADE_BETAS = (0.02, 0.02, 0.01)          # sum 0.05
CASCADE_N = 199                             # ranks 198, 199, 199 of 199: each stage uses its budget in full


def _cascade(stage_scores: np.ndarray, thresholds: list[tuple[float, float]]) -> np.ndarray:
    """The flow of docs/01: stages run in order, the first accept or reject is final, an abstain moves on."""
    out = np.full(len(stage_scores), ABSTAIN, dtype="<U7")
    undecided = np.ones(len(stage_scores), dtype=bool)
    for stage, (tau_reject, tau_accept) in enumerate(thresholds):
        decisions = decide_many(stage_scores[:, stage], tau_reject, tau_accept)
        settled = undecided & (decisions != ABSTAIN)
        out[settled] = decisions[settled]
        undecided &= ~settled
    return out


def _stage_scores(rng: np.random.Generator, size: int, rho: float, means) -> np.ndarray:
    """3 stage scores per state. ``rho`` is the share of variance the stages have in common."""
    shared = rng.normal(0.0, 1.0, (size, 1))
    own = rng.normal(0.0, 1.0, (size, 3))
    return math.sqrt(rho) * shared + math.sqrt(1.0 - rho) * own + np.asarray(means)


def _cascade_per_draw(seed: int, *, rho: float, pos_means, accept_rank_shift: int = 0, use_accept: bool = True,
                      use_reject: bool = True, n_test: int = 500) -> dict[str, np.ndarray]:
    """Per-draw rates of a 3-stage cascade. Every stage is calibrated on all calibration states."""
    rng = np.random.default_rng(seed)
    out = {name: np.empty(N_DRAWS) for name in ("far", "frr", "tar", "crossed")}
    for i in range(N_DRAWS):
        cal_neg = _stage_scores(rng, CASCADE_N, rho, (0.0, 0.0, 0.0))
        cal_pos = _stage_scores(rng, CASCADE_N, rho, pos_means)
        thresholds = []
        for stage in range(3):
            tau_accept, tau_reject = math.inf, -math.inf
            if use_accept and accept_rank_shift == 0:
                tau_accept = conformal_accept_threshold(cal_neg[:, stage], CASCADE_ALPHAS[stage])
            elif use_accept:
                k = conformal_accept_index(CASCADE_N, CASCADE_ALPHAS[stage])
                tau_accept = float(np.sort(cal_neg[:, stage])[(k + accept_rank_shift) - 1])
            if use_reject:
                tau_reject = conformal_reject_threshold(cal_pos[:, stage], CASCADE_BETAS[stage])
            thresholds.append((tau_reject, tau_accept))
        on_negatives = _cascade(_stage_scores(rng, n_test, rho, (0.0, 0.0, 0.0)), thresholds)
        on_positives = _cascade(_stage_scores(rng, n_test, rho, pos_means), thresholds)
        out["far"][i] = np.mean(on_negatives == ACCEPT)
        out["frr"][i] = np.mean(on_positives == REJECT)
        out["tar"][i] = np.mean(on_positives == ACCEPT)
        out["crossed"][i] = any(tau_reject > tau_accept for tau_reject, tau_accept in thresholds)
    return out


def test_sim_a_cascade_of_3_stages_stays_within_the_sum_of_its_stage_budgets():
    """Independent stages and no rejects: the hardest case for the union bound, which is then nearly tight."""
    budget = sum(CASCADE_ALPHAS)
    rates = _cascade_per_draw(4103, rho=0.0, pos_means=(2.0, 3.0, 4.0), use_reject=False)
    mean, se = _mean_and_se(rates["far"])
    assert mean <= budget + SIGMAS * se, f"cascade false-accept {mean:.5f} exceeds {budget} by more than 3 se ({se:.5f})"
    # The exact expected rate for independent stages is 1 - (198/200) * (199/200) * (199/200) = 0.019875.
    exact = 1 - math.prod(conformal_accept_index(CASCADE_N, a) / (CASCADE_N + 1) for a in CASCADE_ALPHAS)
    assert exact == pytest.approx(0.019875, abs=1e-6)
    assert abs(mean - exact) <= SIGMAS * se


def test_sim_a_cascade_with_every_stage_1_rank_too_low_breaks_its_budget():
    budget = sum(CASCADE_ALPHAS)
    rates = _cascade_per_draw(4103, rho=0.0, pos_means=(2.0, 3.0, 4.0), use_reject=False, accept_rank_shift=-1)
    mean, se = _mean_and_se(rates["far"])
    assert not mean <= budget + SIGMAS * se
    assert mean == pytest.approx(1 - (197 / 200) * (198 / 200) * (198 / 200), abs=SIGMAS * se)       # 0.0346


def test_sim_a_cascade_of_3_stages_stays_within_the_sum_of_its_reject_budgets():
    """The same holds for false rejects and beta (section 4). Accepts are switched off to make it tight."""
    budget = sum(CASCADE_BETAS)
    rates = _cascade_per_draw(4109, rho=0.0, pos_means=(2.0, 3.0, 4.0), use_accept=False)
    mean, se = _mean_and_se(rates["frr"])
    assert mean <= budget + SIGMAS * se
    exact = 1 - math.prod(1 - conformal_reject_index(CASCADE_N, b) / (CASCADE_N + 1) for b in CASCADE_BETAS)
    assert exact == pytest.approx(0.049204, abs=1e-6)                     # 1 - (196/200) * (196/200) * (198/200)
    assert abs(mean - exact) <= SIGMAS * se
    assert rates["far"].max() == 0.0                                      # with accepts off nothing is accepted


def test_sim_a_full_cascade_holds_both_bounds_with_correlated_stages_and_crossed_thresholds():
    """Accept and reject thresholds together, stages that share half their variance, and a well
    separated last stage whose 2 thresholds cross in most draws. Both bounds still hold."""
    rates = _cascade_per_draw(4104, rho=0.5, pos_means=(2.0, 3.0, 6.0))
    far, far_se = _mean_and_se(rates["far"])
    frr, frr_se = _mean_and_se(rates["frr"])
    assert far <= sum(CASCADE_ALPHAS) + SIGMAS * far_se
    assert frr <= sum(CASCADE_BETAS) + SIGMAS * frr_se
    assert rates["tar"].mean() > 0.9          # the cascade is useful: it accepts most good states
    assert far > 0 and frr > 0                # and it does err both ways, so both bounds were exercised
    assert rates["crossed"].mean() > 0.5      # the crossing rule was in play in most draws


# =============================================================================================
# Control charts.
# =============================================================================================

def test_p_chart_limits_are_3_standard_errors_from_the_center():
    assert p_chart_limits(0.5, 100) == pytest.approx((0.35, 0.5, 0.65), abs=1e-12)
    lcl, center, ucl = p_chart_limits(0.2, 64)
    assert (lcl, center, ucl) == pytest.approx((0.2 - 3 * 0.05, 0.2, 0.2 + 3 * 0.05), abs=1e-12)     # sqrt(0.16 / 64)
    narrow, wide = p_chart_limits(0.2, 400), p_chart_limits(0.2, 100)
    assert narrow[2] - narrow[1] == pytest.approx((wide[2] - wide[1]) / 2, rel=1e-12)                # 4 times the window
    assert p_chart_limits(0.5, 100, sigmas=2)[2] == pytest.approx(0.6, abs=1e-12)


def test_p_chart_limits_are_clipped_to_0_and_1():
    lcl, center, ucl = p_chart_limits(0.1, 50)
    assert lcl == 0.0 and center == 0.1
    assert ucl == pytest.approx(0.1 + 3 * math.sqrt(0.1 * 0.9 / 50), rel=1e-12)
    lcl, center, ucl = p_chart_limits(0.95, 20)
    assert ucl == 1.0 and center == 0.95
    assert lcl == pytest.approx(0.95 - 3 * math.sqrt(0.95 * 0.05 / 20), rel=1e-12)
    assert p_chart_limits(0.0, 50) == (0.0, 0.0, 0.0)
    assert p_chart_limits(1.0, 50) == (1.0, 1.0, 1.0)


def test_cusum_follows_the_section_9_recursion():
    chart = Cusum(p0=0.10, p1=0.30, h=4.0)
    assert chart.k == pytest.approx(0.10, abs=1e-12) and chart.reference == pytest.approx(0.20, abs=1e-12)
    path, alarms = [], []
    for x in (1, 1, 0, 1, 1, 1, 1):
        alarms.append(chart.update(x))
        path.append(chart.s)
    assert path == pytest.approx([0.8, 1.6, 1.4, 2.2, 3.0, 3.8, 4.6], abs=1e-9)        # +0.8 per event, -0.2 otherwise
    assert alarms == [False] * 6 + [True]
    assert chart.alarm and chart.first_alarm_at == 7 and chart.n == 7


def test_cusum_never_goes_below_zero_and_alarms_only_strictly_above_h():
    chart = Cusum(p0=0.25, p1=0.75, h=2.0)          # reference 0.5, so every step is exactly +0.5 or -0.5
    for _ in range(10):
        assert chart.update(False) is False
    assert chart.s == 0.0
    for expected in (0.5, 1.0, 1.5, 2.0):
        assert chart.update(True) is False
        assert chart.s == expected
    assert chart.first_alarm_at is None             # S equal to h is not an alarm
    assert chart.update(True) is True and chart.s == 2.5
    assert chart.first_alarm_at == 15
    assert chart.update(False) is False and chart.s == 2.0          # back at h: the alarm clears
    assert chart.first_alarm_at == 15                               # the first alarm stays on record
    chart.reset()
    assert (chart.s, chart.n, chart.skipped, chart.first_alarm_at, chart.alarm) == (0.0, 0, 0, None, False)


def test_cusum_takes_rates_and_feature_values_and_skips_missing_ones():
    chart = Cusum(p0=1.0, p1=2.0, h=3.0)            # a feature mean, in its own units
    assert chart.reference == 1.5
    for x, expected in ((1.75, 0.25), (3.5, 2.25), (1.0, 1.75), (math.nan, 1.75), (None, 1.75), (np.float64(3.0), 3.25)):
        chart.update(x)
        assert chart.s == pytest.approx(expected, abs=1e-12)
    assert chart.n == 4 and chart.skipped == 2 and chart.alarm and chart.first_alarm_at == 4
    shifted = Cusum(p0=0.10, p1=0.30, h=4.0, k_fraction=0.25)
    assert shifted.k == pytest.approx(0.05, abs=1e-12) and shifted.reference == pytest.approx(0.15, abs=1e-12)
    assert Cusum(0.1, 0.3, 4.0).update(np.True_) is False and Cusum(0.1, 0.3, 0.5).update(True) is True


def _cusum_mean_run_length(seed: int, event_rate: float, runs: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    lengths = np.empty(runs)
    for i in range(runs):
        chart = Cusum(p0=0.25, p1=0.75, h=2.0)
        for _ in range(STEP_CAP):
            if chart.update(bool(rng.random() < event_rate)):
                break
        else:
            pytest.fail(f"Cusum raised no alarm in {STEP_CAP} steps at an event rate of {event_rate}")
        lengths[i] = chart.first_alarm_at
    return float(lengths.mean()), float(lengths.std(ddof=1) / math.sqrt(runs))


def _cusum_exact_run_length(event_rate: float) -> float:
    """Mean steps to the alarm for the chart above, from its Markov chain.

    With reference 0.5 the statistic moves on the grid 0, 0.5, ..., 2.0 and alarms at 2.5. State i is
    S = i / 2. Each step goes up 1 state with probability ``event_rate`` and down 1 otherwise, held at 0.
    The mean time to the alarm E_i solves E_i = 1 + q * E_(i+1) + (1 - q) * E_max(i-1, 0), with E_5 = 0.
    """
    q = event_rate
    a = np.zeros((5, 5))
    for i in range(5):
        a[i, i] += 1.0
        if i + 1 < 5:
            a[i, i + 1] -= q
        a[i, max(i - 1, 0)] -= 1.0 - q
    return float(np.linalg.solve(a, np.ones(5))[0])


@pytest.mark.parametrize("seed, event_rate", [(4107, 0.5), (4114, 0.75)])
def test_cusum_mean_run_length_matches_its_markov_chain(seed, event_rate):
    exact = _cusum_exact_run_length(event_rate)
    if event_rate == 0.5:
        assert exact == pytest.approx(30.0, abs=1e-9)          # 5 * 6 for a fair walk held at 0
    mean, se = _cusum_mean_run_length(seed, event_rate)
    assert abs(mean - exact) <= SIGMAS * se, f"simulated mean run length {mean:.2f}, exact {exact:.2f}, se {se:.2f}"


# =============================================================================================
# Economics.
# =============================================================================================

def test_robots_per_assistant_is_unbounded_when_robots_never_ask():
    assert robots_per_assistant(0, 30) == math.inf
    assert robots_per_assistant(6, 0) == math.inf
    assert busy_share(0, 30) == 0.0
    assert busy_share(120, 60) == pytest.approx(2.0, abs=1e-12)      # a robot that needs 2 people
    assert robots_per_assistant(120, 60) == pytest.approx(0.5, abs=1e-12)


def test_mtbi_is_operating_seconds_per_assist():
    assert mtbi(3600, 12) == 300.0
    assert mtbi(1701.5, 5) == pytest.approx(340.3, abs=1e-9)
    assert mtbi(3600, 0) == math.inf              # a shift with no assist must not crash a report
    assert mtbi(3600.0, np.int64(4)) == 900.0


def test_autonomy_is_the_share_of_operating_time_with_no_person_involved():
    assert autonomy(assist_s=90, operating_s=3600) == pytest.approx(0.975, abs=1e-12)
    assert autonomy(0, 3600) == 1.0
    assert autonomy(3600, 3600) == 0.0
    assert shown(100 * autonomy(64.8, 3600), 1) == "98.2"


def test_cost_per_cycle_adds_robot_time_assistant_time_and_defects():
    cost = cost_per_cycle(cycle_s=60, assists_per_cycle=0.05, handle_s=45, uncaught_per_cycle=0.0015,
                          robot_rate_usd_per_hour=10, human_rate_usd_per_hour=30, defect_cost_usd=50)
    robot = 10 / 3600 * 60                  # 0.1667
    assistant = 30 / 3600 * 0.05 * 45       # 0.0188
    defects = 50 * 0.0015                   # 0.0750
    assert cost == pytest.approx(robot + assistant + defects, rel=1e-12)
    assert shown(cost, 4) == "0.2604"
    parts = dict(cycle_s=0, assists_per_cycle=0, handle_s=45, uncaught_per_cycle=0,
                 robot_rate_usd_per_hour=10, human_rate_usd_per_hour=30, defect_cost_usd=50)
    assert cost_per_cycle(**parts) == 0.0
    assert cost_per_cycle(**{**parts, "cycle_s": 3600}) == pytest.approx(10.0, rel=1e-12)             # 1 robot-hour
    assert cost_per_cycle(**{**parts, "assists_per_cycle": 2, "handle_s": 1800}) == pytest.approx(30.0, rel=1e-12)
    assert cost_per_cycle(**{**parts, "uncaught_per_cycle": 1}) == 50.0
    with pytest.raises(TypeError):
        cost_per_cycle(60, 0.05, 45, 0.0015, 10, 30, 50)        # 7 bare numbers are easy to mix up: names only


# =============================================================================================
# Rates for reports.
# =============================================================================================

def test_rate_with_interval_states_k_n_point_and_the_wilson_interval():
    cell = rate_with_interval(29, 30)
    assert set(cell) == {"k", "n", "point", "lo", "hi"}
    assert (cell["k"], cell["n"]) == (29, 30)
    assert cell["point"] == pytest.approx(29 / 30, rel=1e-12)
    assert (cell["lo"], cell["hi"]) == wilson(29, 30)
    assert (rate_with_interval(29, 30, conf=0.99)["lo"], rate_with_interval(29, 30, conf=0.99)["hi"]) == wilson(29, 30, 0.99)
    perfect = rate_with_interval(60, 60)
    assert perfect["point"] == 1.0 and perfect["hi"] == 1.0
    assert rate_with_interval(np.int64(8), np.int64(400))["point"] == 0.02


def test_rate_with_interval_returns_an_empty_cell_at_n_0():
    cell = rate_with_interval(0, 0)
    assert (cell["k"], cell["n"]) == (0, 0)
    assert math.isnan(cell["point"])
    assert (cell["lo"], cell["hi"]) == (0.0, 1.0)          # nothing is known, and "hi <= bar" fails closed


def test_balanced_accuracy_is_the_mean_of_the_2_class_rates():
    assert balanced_accuracy(tp=8, fn=2, tn=6, fp=4) == pytest.approx(0.7, abs=1e-12)
    assert balanced_accuracy(tp=10, fn=0, tn=10, fp=0) == 1.0
    assert balanced_accuracy(tp=90, fn=0, tn=0, fp=10) == 0.5       # always says yes: 0.5 whatever the mix
    assert balanced_accuracy(tp=0, fn=90, tn=10, fp=0) == 0.5       # always says no
    assert math.isnan(balanced_accuracy(tp=0, fn=0, tn=5, fp=5))    # no positives: undefined
    assert math.isnan(balanced_accuracy(tp=5, fn=5, tn=0, fp=0))


def _labeled_decisions() -> tuple[list[str], list[bool]]:
    """10 true claims (7 accept, 1 reject, 2 abstain) and 8 false claims (1 accept, 5 reject, 2 abstain)."""
    rows = ([(ACCEPT, True)] * 7 + [(REJECT, True)] + [(ABSTAIN, True)] * 2
            + [(ACCEPT, False)] + [(REJECT, False)] * 5 + [(ABSTAIN, False)] * 2)
    order = np.random.default_rng(4115).permutation(len(rows))
    return [rows[i][0] for i in order], [rows[i][1] for i in order]


def test_decision_rates_counts_and_rates_against_labels():
    decisions, labels = _labeled_decisions()
    r = decision_rates(decisions, labels)
    assert (r["n"], r["n_pos"], r["n_neg"], r["conf"]) == (18, 10, 8, 0.95)
    assert r["counts"] == {
        "pos": {ACCEPT: 7, REJECT: 1, ABSTAIN: 2},
        "neg": {ACCEPT: 1, REJECT: 5, ABSTAIN: 2},
        "all": {ACCEPT: 8, REJECT: 6, ABSTAIN: 4},
    }
    expected = {"far": (1, 8), "frr": (1, 10), "abstain": (4, 18), "tar": (7, 10), "abstain_pos": (2, 10),
                "abstain_neg": (2, 8)}
    for name, (k, n) in expected.items():
        cell = r[name]
        assert (cell["k"], cell["n"]) == (k, n), name
        assert cell["point"] == pytest.approx(k / n, rel=1e-12), name
        assert (cell["lo"], cell["hi"]) == wilson(k, n), name
    assert r["far"]["ucb"] == cp_upper(1, 8) == pytest.approx(sps.beta.ppf(0.95, 2, 7), rel=1e-9)
    assert "ucb" not in r["frr"]


def test_decision_rates_accepts_arrays_and_follows_conf():
    decisions, labels = _labeled_decisions()
    r = decision_rates(np.array(decisions), np.array(labels), conf=0.99)
    assert r["far"]["ucb"] == cp_upper(1, 8, 0.99)
    assert (r["frr"]["lo"], r["frr"]["hi"]) == wilson(1, 10, 0.99)
    as_ints = decision_rates(decisions, [int(x) for x in labels])
    assert as_ints["counts"] == decision_rates(decisions, labels)["counts"]


def test_decision_rates_certifies_598_clean_negatives_end_to_end():
    decisions = [REJECT] * 500 + [ABSTAIN] * 98 + [ACCEPT] * 40
    labels = [False] * 598 + [True] * 40
    r = decision_rates(decisions, labels)
    assert (r["far"]["k"], r["far"]["n"], r["far"]["point"]) == (0, 598, 0.0)
    assert r["far"]["ucb"] == zero_failure_ucb(598) <= 0.005
    assert decision_rates(decisions[1:], labels[1:])["far"]["ucb"] > 0.005        # 597 are 1 too few


def test_decision_rates_leaves_an_empty_class_as_an_empty_cell():
    only_positives = decision_rates([ACCEPT, ABSTAIN, ACCEPT], [True, True, True])
    far = only_positives["far"]
    assert (far["k"], far["n"], far["lo"], far["hi"], far["ucb"]) == (0, 0, 0.0, 1.0, 1.0)
    assert math.isnan(far["point"])
    assert only_positives["tar"]["point"] == pytest.approx(2 / 3)
    nothing = decision_rates([], [])
    assert nothing["n"] == 0 and math.isnan(nothing["abstain"]["point"]) and nothing["far"]["ucb"] == 1.0


# =============================================================================================
# Stubs for S6.
# =============================================================================================

def test_prediction_powered_mean_is_a_documented_stub_until_s6():
    with pytest.raises(NotImplementedError, match="S6"):
        prediction_powered_mean([0.1, 0.2, 0.3], [0.1], [0])
    doc = inspect.getdoc(prediction_powered_mean)
    assert "mean(g over the N accepted) + mean(y - g over the n audited)" in doc
    assert "var(g) / N + var(y - g) / n" in doc


def test_rogan_gladen_is_a_documented_stub_until_s6():
    with pytest.raises(NotImplementedError, match="S6"):
        rogan_gladen(0.9, 0.95, 0.1)
    assert "(q - FAR) / (TAR - FAR)" in inspect.getdoc(rogan_gladen)


# =============================================================================================
# Bad arguments: ValueError with a message that says how to fix the call.
# =============================================================================================

FIX_WORDS = ("Pass ", "Use ", "Drop ", "Divide ", "call ", "Fix ")
_COST = dict(cycle_s=60, assists_per_cycle=0.05, handle_s=45, uncaught_per_cycle=0.0015,
             robot_rate_usd_per_hour=10, human_rate_usd_per_hour=30, defect_cost_usd=50)

BAD_CALLS = [
    # n = 0: an interval needs at least 1 trial
    ("wilson n=0", lambda: wilson(0, 0), ["wilson", "n must be at least 1", "rate_with_interval"]),
    ("clopper_pearson n=0", lambda: clopper_pearson(0, 0), ["clopper_pearson", "n must be at least 1"]),
    ("cp_upper n=0", lambda: cp_upper(0, 0), ["cp_upper", "n must be at least 1"]),
    ("cp_lower n=0", lambda: cp_lower(0, 0), ["cp_lower", "n must be at least 1"]),
    ("zero_failure_ucb n=0", lambda: zero_failure_ucb(0), ["zero_failure_ucb", "n must be at least 1"]),
    ("max_failures_to_certify n=0", lambda: max_failures_to_certify(0, 0.005), ["max_failures_to_certify", "n must be"]),
    ("certify_power n=0", lambda: certify_power(0, 0.005, 0.002), ["certify_power", "n must be at least 1"]),
    # counts out of range
    ("wilson k>n", lambda: wilson(61, 60), ["wilson", "k must be between 0 and n", "k=61", "n=60"]),
    ("wilson k<0", lambda: wilson(-1, 60), ["wilson", "k must be at least 0"]),
    ("wilson n<0", lambda: wilson(5, -3), ["wilson", "n must be at least 1"]),
    ("clopper_pearson k>n", lambda: clopper_pearson(31, 30), ["clopper_pearson", "k must be between 0 and n"]),
    ("cp_upper x>n", lambda: cp_upper(11, 10), ["cp_upper", "x must be between 0 and n"]),
    ("cp_lower x<0", lambda: cp_lower(-2, 10), ["cp_lower", "x must be at least 0"]),
    ("zero_failure_ucb n<0", lambda: zero_failure_ucb(-5), ["zero_failure_ucb", "n must be at least 1"]),
    # counts that are not whole numbers
    ("wilson k float", lambda: wilson(2.5, 10), ["wilson", "k must be a whole number", "int"]),
    ("wilson k integral float", lambda: wilson(3.0, 10), ["wilson", "k must be a whole number", "int(value)"]),
    ("wilson n float", lambda: wilson(2, 10.0), ["wilson", "n must be a whole number"]),
    ("wilson k bool", lambda: wilson(True, 10), ["wilson", "k must be a whole number"]),
    ("cp_upper x string", lambda: cp_upper("3", 10), ["cp_upper", "x must be a whole number"]),
    ("n_to_certify failures float", lambda: n_to_certify(0.005, 1.5), ["n_to_certify", "allowed_failures"]),
    ("n_to_certify failures<0", lambda: n_to_certify(0.005, -1), ["n_to_certify", "allowed_failures must be at least 0"]),
    # confidence and rates outside the unit interval
    ("wilson conf percent", lambda: wilson(5, 10, conf=95), ["wilson", "conf", "Divide a percentage by 100"]),
    ("wilson conf=1", lambda: wilson(5, 10, conf=1.0), ["wilson", "conf must be above 0 and below 1"]),
    ("wilson conf=0", lambda: wilson(5, 10, conf=0.0), ["wilson", "conf must be above 0 and below 1"]),
    ("wilson conf nan", lambda: wilson(5, 10, conf=math.nan), ["wilson", "conf"]),
    ("wilson conf none", lambda: wilson(5, 10, conf=None), ["wilson", "conf"]),
    ("clopper_pearson conf", lambda: clopper_pearson(5, 10, conf=1.5), ["clopper_pearson", "conf"]),
    ("cp_upper conf", lambda: cp_upper(1, 10, conf=-0.1), ["cp_upper", "conf"]),
    ("cp_lower conf", lambda: cp_lower(1, 10, conf=2), ["cp_lower", "conf"]),
    ("zero_failure_ucb conf", lambda: zero_failure_ucb(300, conf=95), ["zero_failure_ucb", "conf"]),
    ("n_to_certify p_max=0", lambda: n_to_certify(0), ["n_to_certify", "p_max must be above 0 and below 1"]),
    ("n_to_certify p_max=1", lambda: n_to_certify(1.0), ["n_to_certify", "p_max"]),
    ("n_to_certify p_max percent", lambda: n_to_certify(0.5 * 100), ["n_to_certify", "p_max", "Divide a percentage"]),
    ("n_to_certify conf", lambda: n_to_certify(0.005, conf=95), ["n_to_certify", "conf"]),
    ("n_to_certify p_max tiny", lambda: n_to_certify(1e-300), ["n_to_certify", "p_max=1e-300", "9e15 trials"]),
    ("max_failures_to_certify p_max", lambda: max_failures_to_certify(5000, 0.5 * 100),
     ["max_failures_to_certify", "p_max"]),
    ("certify_power p_true", lambda: certify_power(5000, 0.005, 1.5), ["certify_power", "p_true"]),
    ("certify_power p_max", lambda: certify_power(5000, 0, 0.002), ["certify_power", "p_max"]),
    # planning
    ("n_two_proportions equal", lambda: n_two_proportions(0.95, 0.95), ["n_two_proportions", "equal", "2 different rates"]),
    ("n_two_proportions percent", lambda: n_two_proportions(95, 99), ["n_two_proportions", "p1", "Divide a percentage"]),
    ("n_two_proportions p2", lambda: n_two_proportions(0.95, -0.1), ["n_two_proportions", "p2"]),
    ("n_two_proportions alpha=0", lambda: n_two_proportions(0.95, 0.99, alpha=0), ["n_two_proportions", "alpha"]),
    ("n_two_proportions power=1", lambda: n_two_proportions(0.95, 0.99, power=1), ["n_two_proportions", "power"]),
    ("n_two_proportions power<alpha", lambda: n_two_proportions(0.95, 0.99, alpha=0.05, power=0.01),
     ["n_two_proportions", "power must be larger than alpha"]),
    ("wald_boundaries sum", lambda: wald_boundaries(0.6, 0.5), ["wald_boundaries", "alpha + beta must be below 1"]),
    ("wald_boundaries alpha", lambda: wald_boundaries(0, 0.2), ["wald_boundaries", "alpha"]),
    ("WaldSPRT order", lambda: WaldSPRT(0.99, 0.95), ["WaldSPRT", "p0 must be below p1"]),
    ("WaldSPRT equal", lambda: WaldSPRT(0.95, 0.95), ["WaldSPRT", "p0 must be below p1"]),
    ("WaldSPRT p1=1", lambda: WaldSPRT(0.95, 1.0), ["WaldSPRT", "p1 must be above 0 and below 1"]),
    ("WaldSPRT p0=0", lambda: WaldSPRT(0.0, 0.5), ["WaldSPRT", "p0"]),
    ("WaldSPRT beta", lambda: WaldSPRT(0.95, 0.99, beta=1.2), ["WaldSPRT", "beta"]),
    ("WaldSPRT update text", lambda: WaldSPRT(0.95, 0.99).update("yes"), ["WaldSPRT.update", "True or False"]),
    ("WaldSPRT update none", lambda: WaldSPRT(0.95, 0.99).update(None), ["WaldSPRT.update", "True or False"]),
    ("WaldSPRT update 2", lambda: WaldSPRT(0.95, 0.99).update(2), ["WaldSPRT.update", "True or False"]),
    ("mcnemar b<0", lambda: mcnemar(-1, 3), ["mcnemar", "b must be at least 0"]),
    ("mcnemar c float", lambda: mcnemar(1, 3.5), ["mcnemar", "c must be a whole number"]),
    # conformal
    ("accept_index n<0", lambda: conformal_accept_index(-1, 0.02), ["conformal_accept_index", "n must be at least 0"]),
    ("accept_index n float", lambda: conformal_accept_index(10.5, 0.02), ["conformal_accept_index", "whole number"]),
    ("accept_index alpha=1", lambda: conformal_accept_index(10, 1.0),
     ["conformal_accept_index", "alpha must be at least 0 and below 1"]),
    ("accept_index alpha<0", lambda: conformal_accept_index(10, -0.1), ["conformal_accept_index", "alpha"]),
    ("accept_index alpha percent", lambda: conformal_accept_index(10, 2),
     ["conformal_accept_index", "alpha", "Divide a percentage"]),
    ("reject_index m<0", lambda: conformal_reject_index(-1, 0.05), ["conformal_reject_index", "m must be at least 0"]),
    ("reject_index beta=1", lambda: conformal_reject_index(10, 1.0), ["conformal_reject_index", "beta"]),
    ("accept_threshold 2-D", lambda: conformal_accept_threshold([[1.0, 2.0], [3.0, 4.0]], 0.02),
     ["conformal_accept_threshold", "neg_scores", "1-D"]),
    ("accept_threshold scalar", lambda: conformal_accept_threshold(5.0, 0.02), ["conformal_accept_threshold", "neg_scores"]),
    ("accept_threshold text", lambda: conformal_accept_threshold(["0.1", "0.2"], 0.02),
     ["conformal_accept_threshold", "neg_scores", "numbers"]),
    ("accept_threshold flags", lambda: conformal_accept_threshold([True, False], 0.02),
     ["conformal_accept_threshold", "neg_scores", "numbers"]),
    ("accept_threshold mixed", lambda: conformal_accept_threshold([0.1, "high"], 0.02),
     ["conformal_accept_threshold", "neg_scores"]),
    ("accept_threshold alpha", lambda: conformal_accept_threshold([0.1, 0.2], 1.5), ["conformal_accept_threshold", "alpha"]),
    ("reject_threshold beta", lambda: conformal_reject_threshold([0.1, 0.2], 1.5), ["conformal_reject_threshold", "beta"]),
    ("reject_threshold 2-D", lambda: conformal_reject_threshold(np.zeros((3, 2)), 0.05),
     ["conformal_reject_threshold", "pos_scores", "1-D"]),
    ("decide text score", lambda: decide("0.9", 0.2, 0.7), ["decide", "score must be a number"]),
    ("decide flag score", lambda: decide(True, 0.2, 0.7), ["decide", "score must be a number"]),
    ("decide nan tau_reject", lambda: decide(0.5, math.nan, 0.7), ["decide", "tau_reject", "-inf"]),
    ("decide none tau_accept", lambda: decide(0.5, 0.2, None), ["decide", "tau_accept", "inf"]),
    ("decide_many text", lambda: decide_many(["accept", "reject"], 0.2, 0.7), ["decide_many", "scores"]),
    ("decide_many nan tau", lambda: decide_many([0.5], 0.2, math.nan), ["decide_many", "tau_accept"]),
    ("min_negatives alpha=0", lambda: min_negatives_to_accept(0), ["min_negatives_to_accept", "alpha must be above 0"]),
    ("min_negatives alpha=1", lambda: min_negatives_to_accept(1), ["min_negatives_to_accept", "alpha"]),
    ("min_negatives alpha tiny", lambda: min_negatives_to_accept(5e-324), ["min_negatives_to_accept", "alpha"]),
    ("min_positives beta", lambda: min_positives_to_reject(-0.05), ["min_positives_to_reject", "beta"]),
    # control charts
    ("p_chart pbar", lambda: p_chart_limits(1.5, 50), ["p_chart_limits", "pbar must be between 0 and 1"]),
    ("p_chart w=0", lambda: p_chart_limits(0.1, 0), ["p_chart_limits", "w must be at least 1"]),
    ("p_chart w float", lambda: p_chart_limits(0.1, 50.0), ["p_chart_limits", "w must be a whole number"]),
    ("p_chart sigmas", lambda: p_chart_limits(0.1, 50, sigmas=0), ["p_chart_limits", "sigmas"]),
    ("Cusum order", lambda: Cusum(0.3, 0.1, 4.0), ["Cusum", "p1 must be larger than p0", "negate"]),
    ("Cusum h=0", lambda: Cusum(0.1, 0.3, 0), ["Cusum", "h must be larger than 0"]),
    ("Cusum p0 nan", lambda: Cusum(math.nan, 0.3, 4.0), ["Cusum", "p0 must be a finite number"]),
    ("Cusum k_fraction", lambda: Cusum(0.1, 0.3, 4.0, k_fraction=0), ["Cusum", "k_fraction"]),
    ("Cusum update text", lambda: Cusum(0.1, 0.3, 4.0).update("x"), ["Cusum.update", "x must be a number"]),
    ("Cusum update inf", lambda: Cusum(0.1, 0.3, 4.0).update(math.inf), ["Cusum.update", "x must be finite"]),
    # economics
    ("busy_share rate<0", lambda: busy_share(-1, 30), ["busy_share", "assists_per_hour must be at least 0"]),
    ("busy_share handle nan", lambda: busy_share(6, math.nan), ["busy_share", "handle_s must be a finite number"]),
    ("robots utilization=0", lambda: robots_per_assistant(6, 30, utilization=0), ["robots_per_assistant", "utilization"]),
    ("robots utilization percent", lambda: robots_per_assistant(6, 30, utilization=70),
     ["robots_per_assistant", "utilization", "Divide a percentage"]),
    ("robots handle<0", lambda: robots_per_assistant(6, -30), ["robots_per_assistant", "handle_s"]),
    ("mtbi operating=0", lambda: mtbi(0, 3), ["mtbi", "operating_s must be larger than 0"]),
    ("mtbi assists<0", lambda: mtbi(100, -1), ["mtbi", "n_assists must be at least 0"]),
    ("mtbi assists float", lambda: mtbi(100, 2.5), ["mtbi", "n_assists must be a whole number"]),
    ("autonomy more assist than time", lambda: autonomy(200, 100), ["autonomy", "assist_s must be at most operating_s"]),
    ("autonomy operating=0", lambda: autonomy(0, 0), ["autonomy", "operating_s must be larger than 0"]),
    ("autonomy assist<0", lambda: autonomy(-1, 100), ["autonomy", "assist_s must be at least 0"]),
    ("cost cycle<0", lambda: cost_per_cycle(**{**_COST, "cycle_s": -1}), ["cost_per_cycle", "cycle_s must be at least 0"]),
    ("cost defect nan", lambda: cost_per_cycle(**{**_COST, "defect_cost_usd": math.nan}),
     ["cost_per_cycle", "defect_cost_usd must be a finite number"]),
    ("uncaught p percent", lambda: uncaught_per_claim(85, 0.02), ["uncaught_per_claim", "p must be between 0 and 1"]),
    ("uncaught far", lambda: uncaught_per_claim(0.85, 1.2), ["uncaught_per_claim", "far must be between 0 and 1"]),
    ("defect rate shares", lambda: defect_rate_among_accepted(0.85, 0.02, 0.7, 0.6),
     ["defect_rate_among_accepted", "frr + abstain_pos must be at most 1"]),
    ("defect rate p", lambda: defect_rate_among_accepted(-0.1, 0.02, 0.05, 0.1),
     ["defect_rate_among_accepted", "p must be"]),
    # rates
    ("rate k>n", lambda: rate_with_interval(5, 3), ["rate_with_interval", "k must be between 0 and n"]),
    ("rate k>0 at n=0", lambda: rate_with_interval(1, 0), ["rate_with_interval", "k must be between 0 and n"]),
    ("rate k<0", lambda: rate_with_interval(-1, 3), ["rate_with_interval", "k must be at least 0"]),
    ("rate n<0", lambda: rate_with_interval(0, -1), ["rate_with_interval", "n must be at least 0"]),
    ("rate float", lambda: rate_with_interval(0.5, 3), ["rate_with_interval", "k must be a whole number"]),
    ("rate conf", lambda: rate_with_interval(1, 3, conf=95), ["rate_with_interval", "conf"]),
    ("balanced_accuracy tp<0", lambda: balanced_accuracy(-1, 2, 3, 4), ["balanced_accuracy", "tp must be at least 0"]),
    ("balanced_accuracy fp float", lambda: balanced_accuracy(1, 2, 3, 4.5),
     ["balanced_accuracy", "fp must be a whole number"]),
    ("decision_rates lengths", lambda: decision_rates([ACCEPT], [True, False]),
     ["decision_rates", "same length", "1 and 2"]),
    ("decision_rates word", lambda: decision_rates(["maybe"], [True]), ["decision_rates", "decisions[0]", "'maybe'"]),
    ("decision_rates not text", lambda: decision_rates([1], [True]), ["decision_rates", "decisions[0]"]),
    ("decision_rates label none", lambda: decision_rates([ACCEPT, REJECT], [True, None]),
     ["decision_rates", "labels[1]", "Drop the claims that have no truth label"]),
    ("decision_rates label nan", lambda: decision_rates([ACCEPT], [math.nan]), ["decision_rates", "labels[0]"]),
    ("decision_rates label text", lambda: decision_rates([ACCEPT], ["True"]), ["decision_rates", "labels[0]"]),
    ("decision_rates conf", lambda: decision_rates([ACCEPT], [True], conf=0), ["decision_rates", "conf"]),
]


@pytest.mark.parametrize("call, fragments", [pytest.param(call, fragments, id=name) for name, call, fragments in BAD_CALLS])
def test_bad_arguments_raise_value_error_that_says_how_to_fix_the_call(call, fragments):
    with pytest.raises(ValueError) as caught:
        call()
    message = str(caught.value)
    for fragment in fragments:
        assert fragment in message, f"{fragment!r} is missing from: {message}"
    assert any(word in message for word in FIX_WORDS), f"the message does not say how to fix the call: {message}"
    assert ";" not in message and EM_DASH not in message


def test_counts_may_be_numpy_integers():
    assert wilson(np.int64(29), np.int32(30)) == wilson(29, 30)
    assert cp_upper(np.int64(16), np.int64(5000)) == cp_upper(16, 5000)
    assert n_to_certify(np.float64(0.005), np.int64(1)) == 947
    assert conformal_accept_index(np.int64(99), np.float64(0.02)) == 98
    assert isinstance(n_to_certify(0.005), int) and isinstance(n_two_proportions(0.95, 0.99), int)
    assert all(type(x) is float for x in wilson(29, 30) + clopper_pearson(29, 30) + (cp_upper(1, 30), cp_lower(1, 30)))


# =============================================================================================
# The ``avail trials`` command.
# =============================================================================================

def _trials(capsys, *argv: str) -> tuple[int, list[str], str]:
    code = trials_cli.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out.splitlines(), captured.err


def _bare_numbers(lines: list[str]) -> list[str]:
    """The lines a script would read: those that hold a whole number and nothing else."""
    return [line for line in lines if line.strip().isdigit()]


def test_cli_prints_285_alone_on_its_own_line(capsys):
    code, lines, err = _trials(capsys, "--p0", "0.95", "--p1", "0.99")
    assert code == 0 and err == ""
    assert "285" in lines
    assert _bare_numbers(lines) == ["285"]


def test_cli_prints_the_formula_with_the_numbers_plugged_in(capsys):
    _, lines, _ = _trials(capsys, "--p0", "0.95", "--p1", "0.99")
    text = "\n".join(lines)
    assert "n = (z_a * sqrt(2 * pbar * (1 - pbar)) + z_b * sqrt(p0 * (1 - p0) + p1 * (1 - p1)))^2 / (p0 - p1)^2" in text
    assert "pbar = (0.95 + 0.99) / 2 = 0.97" in text
    assert "z_a = z(0.975) = 1.9600" in text and "z_b = z(0.8) = 0.8416" in text
    assert "(1.9600 * sqrt(2 * 0.97 * 0.03) + 0.8416 * sqrt(0.95 * 0.05 + 0.99 * 0.01))^2 / (0.95 - 0.99)^2" in text
    assert "= 284.32, rounded up to 285 per arm" in text
    assert "alpha = 0.05, power = 0.8" in text


def test_cli_compare_follows_alpha_and_power(capsys):
    code, lines, _ = _trials(capsys, "--p0", "0.99", "--p1", "0.999")
    assert code == 0 and _bare_numbers(lines) == ["1059"]
    code, lines, _ = _trials(capsys, "--p0", "0.95", "--p1", "0.99", "--alpha", "0.01", "--power", "0.9")
    assert code == 0
    assert _bare_numbers(lines) == [str(n_two_proportions(0.95, 0.99, alpha=0.01, power=0.9))]
    assert "alpha = 0.01, power = 0.9" in "\n".join(lines)
    assert int(_bare_numbers(lines)[0]) > 285


def test_cli_certify_prints_598_and_its_formula(capsys):
    code, lines, err = _trials(capsys, "--certify", "0.005")
    text = "\n".join(lines)
    assert code == 0 and err == ""
    assert _bare_numbers(lines) == ["598"]
    assert "1 - (1 - conf)^(1 / n)" in text
    assert "n = ln(0.05) / ln(0.995) = 597.65, rounded up to 598" in text
    assert "upper bound after 0 in 598: 0.00499706, at most 0.005" in text
    assert "upper bound after 0 in 597: 0.00500541, above 0.005" in text


def test_cli_certify_follows_allowed_failures_and_conf(capsys):
    code, lines, _ = _trials(capsys, "--certify", "0.005", "--allowed-failures", "16")
    text = "\n".join(lines)
    assert code == 0 and _bare_numbers(lines) == ["4857"]
    assert "with 16 failures allowed" in text
    assert "cp_upper(k, n, conf) <= p_max" in text
    assert "upper bound after 16 in 4857" in text and "upper bound after 16 in 4856" in text
    code, lines, _ = _trials(capsys, "--certify", "0.005", "--allowed-failures", "1")
    assert _bare_numbers(lines) == ["947"] and "with 1 failure allowed" in "\n".join(lines)
    code, lines, _ = _trials(capsys, "--certify", "0.005", "--conf", "0.99")
    assert _bare_numbers(lines) == ["919"] and "conf = 0.99" in "\n".join(lines)


def test_cli_answers_both_questions_in_one_call(capsys):
    code, lines, _ = _trials(capsys, "--p0", "0.95", "--p1", "0.99", "--certify", "0.005")
    assert code == 0 and _bare_numbers(lines) == ["285", "598"]


def test_cli_is_reachable_as_avail_trials(capsys):
    assert avail_cli.main(["trials", "--p0", "0.95", "--p1", "0.99"]) == 0
    assert _bare_numbers(capsys.readouterr().out.splitlines()) == ["285"]


@pytest.mark.parametrize("argv, fragment", [
    ([], "--p0 and --p1"),
    (["--p0", "0.95"], "--p0 and --p1 go together"),
    (["--p0", "95", "--p1", "0.99"], "--p0 must be between 0 and 1"),
    (["--p0", "0.95", "--p1", "0.95"], "--p0 and --p1 are equal"),
    (["--p0", "0.95", "--p1", "0.99", "--power", "0.01"], "--power must be larger than --alpha"),
    (["--certify", "0.5", "--alpha", "0.01"], "--alpha and --power apply to a comparison"),
    (["--p0", "0.95", "--p1", "0.99", "--conf", "0.99"], "--allowed-failures and --conf apply to a certificate"),
    (["--certify", "5"], "--certify must be above 0 and below 1"),
    (["--certify", "0.005", "--allowed-failures", "-1"], "--allowed-failures must be at least 0"),
    (["--certify", "0.005", "--conf", "95"], "--conf must be above 0 and below 1"),
    (["--bogus"], "unrecognized arguments"),
    (["--p0", "high", "--p1", "0.99"], "invalid float value"),
])
def test_cli_bad_command_lines_return_2_and_explain(capsys, argv, fragment):
    code, lines, err = _trials(capsys, *argv)
    assert code == 2
    assert lines == []                       # nothing on stdout, so a script never reads a wrong count
    assert fragment in err
    assert err.startswith("avail trials: ") or "avail trials: error:" in err


def test_cli_help_lists_every_flag(capsys):
    code, lines, _ = _trials(capsys, "--help")
    text = "\n".join(lines)
    assert code == 0
    for flag in ("--p0", "--p1", "--alpha", "--power", "--certify", "--allowed-failures", "--conf"):
        assert flag in text


# =============================================================================================
# The package itself: public names, purity, plain prose.
# =============================================================================================

SPEC_NAMES = [
    "wilson", "clopper_pearson", "cp_upper", "cp_lower", "zero_failure_ucb", "n_to_certify",
    "n_two_proportions", "WaldSPRT", "wald_boundaries", "mcnemar",
    "conformal_accept_index", "conformal_accept_threshold", "conformal_reject_index", "conformal_reject_threshold",
    "decide", "min_negatives_to_accept",
    "p_chart_limits", "Cusum",
    "busy_share", "robots_per_assistant", "mtbi", "autonomy", "cost_per_cycle", "uncaught_per_claim",
    "defect_rate_among_accepted",
    "rate_with_interval", "balanced_accuracy", "decision_rates",
    "prediction_powered_mean", "rogan_gladen",
]


def test_the_package_re_exports_every_public_name():
    for name in SPEC_NAMES:
        assert name in S.__all__, name
    for name in S.__all__:
        assert hasattr(S, name), name
    assert len(set(S.__all__)) == len(S.__all__)
    for module in (S.intervals, S.planning, S.conformal, S.spc, S.economics, S.rates, S.stubs):
        for name in module.__all__:
            assert getattr(S, name) is getattr(module, name), name


def test_the_package_imports_only_numpy_scipy_and_the_standard_library():
    """Pure functions: nothing here may pull in the station, the ledger, pandas or another avail package."""
    package = Path(S.__file__).parent
    files = sorted(package.glob("*.py"))
    assert {"intervals.py", "planning.py", "conformal.py", "spc.py", "economics.py", "rates.py", "stubs.py",
            "cli.py", "__init__.py"} <= {path.name for path in files}
    for path in files:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                assert node.level <= 1, f"{path.name} reaches outside avail/stats"
                roots = [] if node.level == 1 else [node.module.split(".")[0]]
            else:
                continue
            for root in roots:
                assert root in {"numpy", "scipy"} or root in sys.stdlib_module_names, f"{path.name} imports {root}"


def test_public_docstrings_are_present_and_plain():
    """Every public name explains itself, with no em dash and no semicolon."""
    targets = [getattr(S, name) for name in S.__all__ if callable(getattr(S, name))]
    targets += [WaldSPRT.update, Cusum.update, Cusum.reset, trials_cli.main]
    targets += [S, S.intervals, S.planning, S.conformal, S.spc, S.economics, S.rates, S.stubs, trials_cli]
    assert len(targets) > 40
    for target in targets:
        doc = inspect.getdoc(target)
        label = getattr(target, "__qualname__", getattr(target, "__name__", repr(target)))
        assert doc and len(doc) > 30, f"{label} has no real docstring"
        for mark in (EM_DASH, EN_DASH, ";"):
            assert mark not in doc, f"{label} docstring contains {mark!r}"


# ------------------------------------------------------------------ gaps found by the review's mutation run, 2026-11-11
# The review's own mutants left 23 of 179 alive. These pin the ones that can change an answer.

def test_zero_failure_bound_follows_the_confidence_it_is_given():
    assert S.zero_failure_ucb(300, conf=0.99) == pytest.approx(1.0 - 0.01 ** (1.0 / 300.0), rel=1e-12)
    assert S.zero_failure_ucb(300, conf=0.99) > S.zero_failure_ucb(300, conf=0.95)


def test_max_failures_to_certify_follows_the_confidence_it_is_given():
    loose = S.max_failures_to_certify(5000, 0.005, conf=0.80)
    strict = S.max_failures_to_certify(5000, 0.005, conf=0.99)
    assert loose is not None and strict is not None and loose > strict
    assert S.max_failures_to_certify(5000, 0.005) == 16      # the docs/04 bar at 0.95


def test_decision_rates_report_the_confidence_they_used():
    out = S.decision_rates(["accept", "reject", "abstain"], [True, True, False], conf=0.9)
    assert out["conf"] == 0.9


def test_cusum_keeps_the_index_of_its_first_alarm():
    c = S.Cusum(p0=0.1, p1=0.3, h=0.5)
    seen = [c.update(1.0) for _ in range(8)]
    assert any(seen) and c.first_alarm_at == seen.index(True) + 1


def test_trials_cli_states_the_total_and_honors_conf(capsys):
    from avail.stats.cli import main

    assert main(["--p0", "0.95", "--p1", "0.99"]) == 0
    out = capsys.readouterr().out
    assert "285" in out and "570 in total" in out
    assert main(["--certify", "0.005", "--conf", "0.99"]) == 0
    out = capsys.readouterr().out
    assert str(S.n_to_certify(0.005, conf=0.99)) in out.split()
