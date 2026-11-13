"""From per-claim rates to a shift: uncaught failures, robots per assistant, cost per cycle.

Spec: ``docs/04_calibration_math.md`` sections 6 and 10.

These are closed-form planning numbers. With retries, repairs and a person in
the loop the real numbers depend on the fork policy, so the closed-loop
experiments measure them directly. Use these for intuition and for planning.

Definitions that every report states:

- MTBI (mean time between interventions): operating seconds divided by the
  number of assists.
- Autonomy: the share of operating time with no person involved,
  ``1 - assist_seconds / operating_seconds``. This definition is AVAIL's own.
"""

from __future__ import annotations

import math

from . import _validate as check

__all__ = [
    "busy_share",
    "robots_per_assistant",
    "mtbi",
    "autonomy",
    "cost_per_cycle",
    "uncaught_per_claim",
    "defect_rate_among_accepted",
]


def _at_least_zero(func: str, name: str, value, what: str) -> float:
    return check.real(func, name, value, f"Pass {what}, 0 or more.", minimum=0.0)


def busy_share(assists_per_hour: float, handle_s: float) -> float:
    """The share of an assistant's time that 1 robot takes: ``assists_per_hour * handle_s / 3600``.

    6 assists an hour at 30 s each is a busy share of 0.05.
    """
    rate = _at_least_zero("busy_share", "assists_per_hour", assists_per_hour, "the assists per robot-hour")
    handle = _at_least_zero("busy_share", "handle_s", handle_s, "the seconds a person spends on 1 assist")
    return rate * handle / 3600.0


def robots_per_assistant(assists_per_hour: float, handle_s: float, utilization: float = 1.0) -> float:
    """How many robots 1 assistant can cover: ``utilization / busy_share``.

    ``utilization`` is the share of the assistant's time that goes to assists.
    6 assists an hour at 30 s each gives 20 robots at full utilization and 14
    at 70%. 2 an hour at 45 s gives 40 and 28.

    Returns ``inf`` when the robots never ask.
    """
    rate = _at_least_zero("robots_per_assistant", "assists_per_hour", assists_per_hour,
                          "the assists per robot-hour")
    handle = _at_least_zero("robots_per_assistant", "handle_s", handle_s,
                            "the seconds a person spends on 1 assist")
    utilization = check.fraction("robots_per_assistant", "utilization", utilization, high_ok=True, example="0.7")
    busy_seconds_per_hour = rate * handle
    if busy_seconds_per_hour == 0.0:
        return math.inf
    return utilization * 3600.0 / busy_seconds_per_hour


def _operating(func: str, operating_s) -> float:
    return check.real(
        func, "operating_s", operating_s,
        "Pass the seconds the station was operating, more than 0.",
        minimum=0.0, strict=True,
    )


def mtbi(operating_s: float, n_assists: int) -> float:
    """Mean time between interventions in seconds: ``operating_s / n_assists``.

    Returns ``inf`` when there were 0 assists. Report that case as "no
    intervention in ``operating_s`` seconds".
    """
    operating = _operating("mtbi", operating_s)
    assists = check.count("mtbi", "n_assists", n_assists, 0, "Pass the number of assists in that time, 0 or more.")
    if assists == 0:
        return math.inf
    return operating / assists


def autonomy(assist_s: float, operating_s: float) -> float:
    """The share of operating time with no person involved: ``1 - assist_s / operating_s``.

    ``assist_s`` is the total time people spent on assists. It must cover the
    same robots and the same span as ``operating_s``.
    """
    operating = _operating("autonomy", operating_s)
    assist = _at_least_zero("autonomy", "assist_s", assist_s, "the seconds people spent on assists")
    if assist > operating:
        raise ValueError(
            f"autonomy: assist_s must be at most operating_s, got assist_s={assist_s!r} with "
            f"operating_s={operating_s!r}. Pass 2 totals that cover the same robots and the same span of time."
        )
    return 1.0 - assist / operating


def cost_per_cycle(
    *,
    cycle_s: float,
    assists_per_cycle: float,
    handle_s: float,
    uncaught_per_cycle: float,
    robot_rate_usd_per_hour: float,
    human_rate_usd_per_hour: float,
    defect_cost_usd: float,
) -> float:
    """Dollars per cycle: robot time, plus assistant time, plus the cost of uncaught failures.

    ``robot_rate * cycle_s + human_rate * assists_per_cycle * handle_s + defect_cost_usd * uncaught_per_cycle``

    The 2 rates are per hour, as in ``config/experiment.yaml``, and are
    converted to per second here. ``uncaught_per_cycle`` is the uncaught failure
    rate per cycle. Every argument is passed by name.

    Example: a 60 s cycle with 0.05 assists of 45 s each and 0.0015 uncaught
    failures, at 10 and 30 dollars an hour and 50 dollars a defect, costs about
    0.26 dollars: 0.167 for the robot, 0.019 for the assistant and 0.075 for defects.
    """
    func = "cost_per_cycle"
    cycle = _at_least_zero(func, "cycle_s", cycle_s, "the seconds 1 cycle takes")
    assists = _at_least_zero(func, "assists_per_cycle", assists_per_cycle, "the mean number of assists in 1 cycle")
    handle = _at_least_zero(func, "handle_s", handle_s, "the seconds a person spends on 1 assist")
    uncaught = _at_least_zero(func, "uncaught_per_cycle", uncaught_per_cycle, "the uncaught failures per cycle")
    robot = _at_least_zero(func, "robot_rate_usd_per_hour", robot_rate_usd_per_hour, "the robot cost per hour")
    human = _at_least_zero(func, "human_rate_usd_per_hour", human_rate_usd_per_hour, "the assistant cost per hour")
    defect = _at_least_zero(func, "defect_cost_usd", defect_cost_usd, "the cost of 1 uncaught failure")
    return robot / 3600.0 * cycle + human / 3600.0 * assists * handle + defect * uncaught


def _rate(func: str, name: str, value, example: str) -> float:
    return check.fraction(func, name, value, low_ok=True, high_ok=True, example=example)


def uncaught_per_claim(p: float, far: float) -> float:
    """Uncaught failures per claim: ``(1 - p) * far``.

    ``p`` is the chance the skill leaves the predicate true at the moment of
    the decision, and ``far`` is the false-accept rate of the check.

    With ``p = 0.85``, a check with a false-accept rate of 30% lets 4.5 bad
    plugs through per 100 claims. A check at 2% lets 0.3 through.
    """
    p = _rate("uncaught_per_claim", "p", p, "0.85")
    far = _rate("uncaught_per_claim", "far", far, "0.02")
    return (1.0 - p) * far


def defect_rate_among_accepted(p: float, far: float, frr: float, abstain_pos: float) -> float:
    """The share of accepted claims that are in truth failures.

    ``(1 - p) * far / (p * (1 - frr - abstain_pos) + (1 - p) * far)``

    ``frr`` is the false-reject rate and ``abstain_pos`` the abstain rate on
    true positives, so ``1 - frr - abstain_pos`` is the share of good claims
    that are accepted.

    Returns NaN when nothing is accepted at all, because the share is then undefined.
    """
    func = "defect_rate_among_accepted"
    p = _rate(func, "p", p, "0.85")
    far = _rate(func, "far", far, "0.02")
    frr = _rate(func, "frr", frr, "0.05")
    abstain_pos = _rate(func, "abstain_pos", abstain_pos, "0.1")
    true_accept = 1.0 - frr - abstain_pos
    if true_accept < -1e-12:
        raise ValueError(
            f"{func}: frr + abstain_pos must be at most 1, got frr={frr!r} and abstain_pos={abstain_pos!r}. "
            "Pass the reject share and the abstain share of the same set of true positives."
        )
    bad = (1.0 - p) * far
    accepted = p * max(0.0, true_accept) + bad
    if accepted == 0.0:
        return math.nan
    return bad / accepted
