"""E4's plan and its pre-registered statistics (docs/07, 2026-11-16). No seed is read and nothing is simulated."""

from __future__ import annotations

import numpy as np
import pytest

from avail.stats import bootstrap_mean_upper, cp_upper


def test_the_bootstrap_bound_is_fixed_by_its_seed_and_sits_above_the_mean():
    rng = np.random.default_rng(1)
    x = rng.poisson(1.0, size=500).astype(float)
    a = bootstrap_mean_upper(x, resamples=2000, seed=40000)
    assert a == bootstrap_mean_upper(x, resamples=2000, seed=40000)       # the same values give the same bound
    assert a > x.mean() and a < x.mean() + 4 * x.std() / np.sqrt(len(x))
    assert bootstrap_mean_upper(np.full(50, 2.0), resamples=200) == 2.0
    for bad in ([], [1.0, float("nan")]):
        with pytest.raises(ValueError):
            bootstrap_mean_upper(bad)


def test_h4a_allows_16_bad_cycles_in_5000_and_not_17():
    assert cp_upper(16, 5000) <= 0.005 < cp_upper(17, 5000)


def test_the_e4_plan_reads_e4s_own_range_from_its_start():
    from avail.shift.cli import e4_plan

    plan = e4_plan()
    assert plan["split"] == "shift_test_e4" and plan["variant"] == "latch"
    assert plan["arms"]["avail_simop"] == {"driver_arm": "avail_simop", "cycles": 5000, "seeds": [40000, 44999]}
    assert plan["arms"]["skill_done"] == {"driver_arm": "flag", "cycles": 1000, "seeds": [40000, 40999]}
    assert plan["h4a"]["max_bad_cycles"] == 16 and plan["h4b"]["max_calls_per_100"] == 112
    assert plan["h4b"]["rng_seed"] == 40000 and plan["h4b"]["resamples"] == 10000


def test_shift_run_refuses_the_e4_split():
    from avail.shift.cli import main

    with pytest.raises(SystemExit):
        main(["run", "--variant", "latch", "--arm", "avail_simop", "--seeds", "40000:40001", "--shift", "x",
              "--final-run", "E4"])
