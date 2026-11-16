"""A one-sided upper bound on a mean from a bootstrap over units (E4's H4b, docs/07 2026-11-16).

The units (cycles) are resampled with replacement ``resamples`` times by ``numpy.random.default_rng(seed)``, 1 draw of
``n`` indices per resample in order, and the bound is the ``confidence`` quantile of the resampled means (numpy's
default linear interpolation). The seed, the count and the quantile are part of the pre-registered test, so the number
is the same every time the same values come in.
"""

from __future__ import annotations

import numpy as np

__all__ = ["bootstrap_mean_upper"]


def bootstrap_mean_upper(values, resamples: int = 10000, seed: int = 40000, confidence: float = 0.95) -> float:
    """The percentile bootstrap's one-sided upper bound on the mean of ``values``."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) == 0:
        raise ValueError(f"bootstrap_mean_upper: values must be a non-empty 1-D sequence, got shape {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("bootstrap_mean_upper: values must all be finite numbers")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"bootstrap_mean_upper: confidence must be in (0, 1), got {confidence}")
    if int(resamples) < 1:
        raise ValueError(f"bootstrap_mean_upper: resamples must be 1 or more, got {resamples}")
    rng = np.random.default_rng(int(seed))
    n = len(x)
    means = np.empty(int(resamples))
    for b in range(int(resamples)):
        means[b] = x[rng.integers(0, n, size=n)].mean()
    return float(np.quantile(means, confidence))
