"""Argument checks shared by the statistics functions.

Every check raises ValueError. The message names the function and the argument,
shows the value it got and says how to fix the call.
"""

from __future__ import annotations

import math
import numbers

import numpy as np

_BOOLS = (bool, np.bool_)


def is_real(value) -> bool:
    """True for an int or a float (numpy scalars included). A bool is a flag, so it does not count."""
    return isinstance(value, numbers.Real) and not isinstance(value, _BOOLS)


def count(func: str, name: str, value, minimum: int, hint: str) -> int:
    """``value`` as an int of at least ``minimum``. ``hint`` says what to pass."""
    if isinstance(value, _BOOLS) or not isinstance(value, numbers.Integral):
        raise ValueError(
            f"{func}: {name} must be a whole number, got {value!r}. "
            "Pass a count as an int. If the count is stored as a float, pass int(value)."
        )
    value = int(value)
    if value < minimum:
        raise ValueError(f"{func}: {name} must be at least {minimum}, got {value}. {hint}")
    return value


def events_in_trials(func: str, events_name: str, events, n) -> tuple[int, int]:
    """An event count and a trial count with ``0 <= events <= n`` and ``n >= 1``."""
    n = count(
        func, "n", n, 1,
        "An interval needs at least 1 trial. For a table cell that can be empty, call "
        "rate_with_interval, which returns an empty cell at n = 0.",
    )
    events = count(func, events_name, events, 0, "Pass the number of events seen, 0 or more.")
    if events > n:
        raise ValueError(
            f"{func}: {events_name} must be between 0 and n, got {events_name}={events} with n={n}. "
            "Pass the event count first and the number of trials second."
        )
    return events, n


def _span(low_ok: bool, high_ok: bool) -> str:
    if low_ok and high_ok:
        return "between 0 and 1"
    if low_ok:
        return "at least 0 and below 1"
    if high_ok:
        return "above 0 and at most 1"
    return "above 0 and below 1"


def fraction(func: str, name: str, value, *, low_ok: bool = False, high_ok: bool = False,
             example: str = "0.05") -> float:
    """``value`` as a float in the unit interval. The ends are excluded unless allowed."""
    good = is_real(value)
    number = float(value) if good else math.nan
    if good:
        above = number > 0.0 or (low_ok and number == 0.0)
        below = number < 1.0 or (high_ok and number == 1.0)
        good = above and below          # NaN fails both comparisons
    if not good:
        raise ValueError(
            f"{func}: {name} must be {_span(low_ok, high_ok)}, got {value!r}. "
            f"Pass a fraction such as {example}. Divide a percentage by 100."
        )
    return number


def real(func: str, name: str, value, hint: str, *, minimum: float | None = None,
         strict: bool = False) -> float:
    """``value`` as a finite float, optionally bounded below. ``hint`` says what to pass."""
    number = float(value) if is_real(value) else math.nan
    if not math.isfinite(number):
        raise ValueError(f"{func}: {name} must be a finite number, got {value!r}. {hint}")
    if minimum is not None and (number <= minimum if strict else number < minimum):
        bound = "larger than" if strict else "at least"
        raise ValueError(f"{func}: {name} must be {bound} {minimum:g}, got {value!r}. {hint}")
    return number
