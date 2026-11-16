"""probe.release_tug's derived features (2026-11-16, docs/07 step 1): a pull that stalled with a low force reading is
not a hold. No simulator: the features are functions of what the pull reports."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from avail.evidence.release_tug import HELD_AT_CAP_FRAC, derive, derived


def test_a_stall_with_a_low_reading_is_not_held_at_the_cap():
    assert derived(1.0, 15.05, 15.0) == {"force_frac": 15.05 / 15.0, "held_at_cap": 1.0}
    assert derived(1.0, 0.358, 15.0)["held_at_cap"] == 0.0            # E3b seed 82,607: a stall at 0.36 N
    assert derived(1.0, -12.14, 15.0)["held_at_cap"] == 0.0           # E3b seed 84,047: a stall at -12.1 N
    assert derived(0.0, 15.0, 15.0)["held_at_cap"] == 0.0             # stopped by travel: it slipped
    assert derived(1.0, HELD_AT_CAP_FRAC * 15.0, 15.0)["held_at_cap"] == 1.0


def test_a_stored_table_gets_the_same_features_as_columns():
    p = "B/probe.release_tug/"
    t = pd.DataFrame({f"{p}stopped_by_force": [1.0, 1.0, 0.0, np.nan], f"{p}force_at_stop_n": [15.0, 0.4, 9.0, np.nan]})
    out = derive(t, "latch")
    assert list(out[f"{p}held_at_cap"].iloc[:3]) == [1.0, 0.0, 0.0]
    assert math.isnan(out[f"{p}held_at_cap"].iloc[3]) and math.isnan(out[f"{p}force_frac"].iloc[3])   # no pull, no feature
    assert derive(out, "latch") is out                                # already there: left as it is
