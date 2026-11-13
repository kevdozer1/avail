"""Contact probes: push_seat, tug and wiggle. Caps come from config and never change at run time.

A probe is a guarded move through the Station protocol. It gets a force cap and a travel cap
from ``config/experiment.yaml``, it stops at whichever comes first, and it reports which one
stopped it (CLAUDE.md rule 15). Every guarded move here goes through ``_guarded``, which passes
the config force cap unchanged and clips the travel to the config travel cap, so no code path
can ask the station for more.

What each probe can tell, in 1 line:

- ``probe.push_seat``  presses along the insertion axis. A seated plug gives almost nothing. A
                       short plug travels the rest of the way and clicks home, which repairs it.
- ``probe.tug``        pulls back along the axis. A seated plug holds the pull. A short plug or
                       a blocked latch comes out, and the probe pushes it back.
- ``probe.wiggle``     small moves sideways. A plug whose shoulder is pressed on the socket face
                       gives less than one that floats in the bore.

Travel comes from the joint encoders and is the signal these probes lean on. The force estimate
comes from motor torques and carries about 0.8 N of noise, so it is reported and kept out of
every rule except the station's own force stop.

All 3 probes assume the arm still holds the plug. Each one reads the gripper opening before
it moves. When the fingers are not at the plug's width the probe does not move and returns
``holding = 0`` with no score, because pressing an empty gripper onto the board measures
nothing about the plug. A claim about a predicate other than ``seated`` or ``latched`` gets an
empty Evidence and no motion.

Every probe ends on a finished guarded move, which the station relaxes to a position hold with
zero commanded force. ``cost.seconds`` is station time, read from ``robot_state().ts``.

The thresholds below were tuned on dev seeds 3000 to 3999 of the simulated station, whose
travel trace has no encoder noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..config import ExperimentConfig, load_experiment
from ..schemas import Claim, Cost, Evidence
from ..station.base import MotionResult, NominalGeometry, Station
from .base import make_evidence, register

MM = 1e-3
SEATING_PREDICATES = frozenset({"seated", "latched"})

# typical station time per call, from the table in docs/03_evidence_sources.md
PUSH_TYPICAL_S = 1.0
TUG_TYPICAL_S = 1.5
WIGGLE_TYPICAL_S = 1.5

# the grasp check
JAW_BAND = 0.08               # of the jaw's travel. The fingers count as holding the plug within this of the expected opening

# the click: a jump in travel that outruns the commanded speed and ends at a stop
SNAP_WINDOW_S = 0.020         # a click moves the tool 0.3 to 0.6 mm inside about 20 ms
SNAP_AVERAGE_S = 0.008        # travel is averaged over this long before a gain is taken
SNAP_MIN_MM = 0.20            # a jump at least this far past the commanded motion counts as a click
SNAP_REST_MM = 0.20           # and the tool must travel less than this after the jump

# the seat score
GIVE_MM = 0.20                # travel that no click explains halves the score at this value
QUIET_BASE = 0.9              # the most a push with no click can score

# the tug
TUG_PULL_SLIPS = 2.0          # the pull ends once travel reaches this many slip distances, or the travel cap if that is less
STIFFNESS_MIN_TRAVEL_MM = 0.05   # the position signal resolves about this much

# the wiggle
WIGGLE_SPEED_MM_S = 2.0       # config has no wiggle speed
WIGGLE_AXES = (("xp", (1.0, 0.0, 0.0)), ("xn", (-1.0, 0.0, 0.0)), ("yp", (0.0, 1.0, 0.0)), ("yn", (0.0, -1.0, 0.0)))
RETURN_TOL_MM = 0.03          # a tool this close to its starting position counts as back
RETURN_TRIES = 3
WIGGLE_PATTERN = "zigzag"
PLAY_MM = 0.25                # mean lateral travel that halves the wiggle score


# ----------------------------------------------------------------------------- shared helpers
def _connector(params: dict) -> str:
    connector = params.get("connector")
    if isinstance(connector, str) and connector:
        return connector
    nominal = params.get("nominal")
    if isinstance(nominal, NominalGeometry) and nominal.connector:
        return nominal.connector
    raise ValueError("a probe needs the connector type to look up its caps. Put 'connector' or 'nominal' in params")


def _arm(params: dict) -> str:
    arm = params.get("arm")
    if isinstance(arm, str) and arm:
        return arm
    nominal = params.get("nominal")
    if isinstance(nominal, NominalGeometry):
        return nominal.holding_arm
    raise ValueError("a probe needs to know which arm holds the plug. Put 'arm' or 'nominal' in params")


def _holding(arm_state, params: dict) -> bool | None:
    """True when the gripper is at the opening expected while it holds the plug. None when that cannot be known."""
    nominal = params.get("nominal")
    if not isinstance(nominal, NominalGeometry):
        return None
    expected = float(nominal.expected_jaw_pos())
    jaw = float(arm_state.gripper_pos)
    if not (math.isfinite(expected) and math.isfinite(jaw)) or expected <= 0.0:
        return None
    return abs(jaw - expected) <= JAW_BAND


def _guarded(station: Station, arm: str, direction, frame: str, travel_mm: float, speed_mm_s: float,
             force_cap_n: float, travel_cap_mm: float) -> MotionResult:
    """One guarded move of at most ``travel_cap_mm`` along ``direction``, with the force cap unchanged."""
    travel_m = min(float(travel_mm), float(travel_cap_mm)) * MM
    unit = np.asarray(direction, dtype=float)
    unit = unit / float(np.linalg.norm(unit))
    return station.move_relative(arm, tuple(float(v) * travel_m for v in unit), frame, float(speed_mm_s) * MM,
                                 float(force_cap_n), travel_m)


def _stop_flags(result: MotionResult) -> dict[str, float]:
    """Which cap stopped a guarded move. Both are 0 when it ended for another reason."""
    return {"stopped_by_force": 1.0 if result.stopped_by == "force" else 0.0,
            "stopped_by_travel": 1.0 if result.stopped_by == "travel" else 0.0}


def _trace(result: MotionResult) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """A guarded move's trace as 3 equal-length finite arrays (time, travel, force), or None."""
    trace = result.trace if isinstance(result.trace, dict) else {}
    try:
        t, s, f = (np.asarray(trace.get(key, ()), dtype=float) for key in ("t_s", "travel_mm", "force_n"))
    except (TypeError, ValueError):
        return None
    if t.ndim != 1 or len(t) < 2 or t.shape != s.shape or t.shape != f.shape:
        return None
    if not (np.isfinite(t).all() and np.isfinite(s).all() and np.isfinite(f).all()):
        return None
    return t, s, f


def snap_of(result: MotionResult, speed_mm_s: float) -> tuple[float, float] | None:
    """The largest travel jump that ended at a stop: (snap_mm, travel after it in mm).

    ``snap_mm`` is the travel gained across 20 ms minus what the commanded speed explains in
    20 ms. Only jumps after which the tool travelled less than 0.20 mm are considered, so the
    lurch of an arm that breaks away and keeps going does not count. None when the trace is
    too short to tell.
    """
    arrays = _trace(result)
    if arrays is None:
        return None
    t, s, _ = arrays
    dt = float(np.median(np.diff(t)))
    if not dt > 0.0:
        return None
    n_win = max(1, int(round(SNAP_WINDOW_S / dt)))
    n_avg = max(1, int(round(SNAP_AVERAGE_S / dt)))
    smooth = np.convolve(s, np.full(n_avg, 1.0 / n_avg), mode="valid")
    if len(smooth) <= n_win:
        return None
    excess = smooth[n_win:] - smooth[:-n_win] - float(speed_mm_s) * n_win * dt
    after = smooth[-1] - smooth[n_win:]
    settled = np.flatnonzero(after <= SNAP_REST_MM)       # never empty: the last window has nothing after it
    best = int(settled[np.argmax(excess[settled])])
    return float(excess[best]), float(after[best])


def seat_score(stopped_by_force: bool, clicked: bool, unexplained_mm: float) -> float:
    """The hand-made score of a push. The PushSeat docstring states the formula."""
    if not stopped_by_force:
        return 0.0
    base = 1.0 if clicked else QUIET_BASE
    return base / (1.0 + (max(0.0, unexplained_mm) / GIVE_MM) ** 2)


@dataclass(frozen=True)
class _Push:
    """One capped push along the insertion axis, measured."""

    result: MotionResult
    travel_mm: float
    clicked: bool
    snap_mm: float | None
    unexplained_mm: float

    @property
    def by_force(self) -> bool:
        return self.result.stopped_by == "force"

    @property
    def repaired(self) -> bool:
        return self.clicked and self.by_force

    @property
    def score(self) -> float:
        return seat_score(self.by_force, self.clicked, self.unexplained_mm)


def _push(station: Station, arm: str, connector: str, cfg: ExperimentConfig) -> _Push:
    """Press along tool +z with the push caps from config and measure what happened."""
    caps = cfg.probes.push_seat
    result = _guarded(station, arm, (0.0, 0.0, 1.0), "tool", caps.travel_cap_mm, caps.speed_mm_s,
                      cfg.push_force_cap_n(connector), caps.travel_cap_mm)
    travel_mm = float(result.travel_m) * 1000.0
    snap = snap_of(result, caps.speed_mm_s)
    clicked = snap is not None and snap[0] >= SNAP_MIN_MM
    unexplained = max(0.0, snap[1]) if clicked else max(0.0, travel_mm)
    return _Push(result, travel_mm, clicked, snap[0] if snap is not None else None, unexplained)


def _clock(station: Station) -> float:
    return float(station.robot_state().ts)


class _Probe:
    """What the 3 probes share: the predicates, the cost estimate and the checks before any motion."""

    source_id = ""
    kind = "active_probe"
    predicates = {"seated", "latched"}
    typical_s = 0.0

    def estimate_cost(self, claim: Claim) -> Cost:
        return Cost(seconds=self.typical_s, usd=0.0, moved_robot=True)

    def collect(self, station: Station, claim: Claim, params: dict) -> Evidence:
        if claim.predicate.name not in self.predicates:
            return make_evidence(self, claim, {})
        arm = _arm(params)
        state = station.robot_state()
        if arm not in state.arms:
            raise ValueError(f"the station has no arm named {arm!r}. Known: {', '.join(sorted(state.arms))}")
        holding = _holding(state.arms[arm], params)
        if holding is False:
            return make_evidence(self, claim, {"holding": 0.0})
        features: dict[str, float] = {} if holding is None else {"holding": 1.0}
        score, side_effect = self._probe(station, arm, _connector(params), load_experiment(), state, features)
        return make_evidence(self, claim, features, score=score, seconds=_clock(station) - float(state.ts),
                             moved_robot=True, side_effect=side_effect)

    def _probe(self, station: Station, arm: str, connector: str, cfg: ExperimentConfig, state,
               features: dict[str, float]) -> tuple[float | None, str]:
        raise NotImplementedError


# ----------------------------------------------------------------------------- probe.push_seat
class PushSeat(_Probe):
    """Presses along the insertion axis up to the push force cap and measures travel to the stop.

    One guarded move along tool +z at ``probes.push_seat.speed_mm_s``, force cap
    ``push_force_cap_n(connector)``, travel cap ``probes.push_seat.travel_cap_mm``.

    Features:

    - ``travel_mm``          measured travel when the move ended
    - ``force_at_stop_n``    the force estimate when it ended
    - ``peak_n``             the largest force estimate during the move
    - ``stopped_by_force``   1.0 when the force cap ended the move
    - ``stopped_by_travel``  1.0 when the travel cap ended it
    - ``snap_mm``            the largest travel jump that ended at a stop: the gain across 20 ms
                             minus what the commanded speed explains in 20 ms
    - ``clicked``            1.0 when ``snap_mm`` is at least 0.20 mm
    - ``unexplained_mm``     travel that no click explains: the travel after the click, or all of
                             the travel when there was no click
    - ``holding``            1.0 when the gripper was at the plug's width before the push

    Side effect: ``repaired`` when the push clicked and then stopped on force. ``disturbed`` when
    it moved the tool more than ``probes.tug.slip_mm`` without that. ``none`` otherwise.

    Score, in 0 to 1::

        score = 0                                           the push did not stop on force
        score = base / (1 + (unexplained_mm / 0.2) ** 2)    it did, with base 1.0 after a click and 0.9 without

    A plug that was seated travels almost nothing and stops on force, so it scores near 0.9. A
    plug that clicked home and then stopped on force is seated now and scores near 1.0. It ranks
    above the quiet stop because a click is direct evidence that the detent engaged, and a quiet
    stop is also what a plug in the wrong port looks like. A push that ran to the travel cap
    scores 0. A push that travelled and then stopped on force with no click scores low: 0.18 at
    0.4 mm, 0.03 at 1 mm.

    What it cannot do. It cannot tell which port the plug is in. It cannot tell a blocked latch
    from an engaged one: the plug is at full depth either way and stops the push the same way.
    A blocked latch gives about 0.1 mm more than a latched plug, and that overlaps. A tilted jam
    that stops the push within 0.1 mm looks seated.
    """

    source_id = "probe.push_seat"
    typical_s = PUSH_TYPICAL_S

    def _probe(self, station, arm, connector, cfg, state, features):
        push = _push(station, arm, connector, cfg)
        features.update(_stop_flags(push.result))
        features.update({
            "travel_mm": push.travel_mm, "force_at_stop_n": push.result.force_at_stop_n, "peak_n": push.result.peak_force_n,
            "snap_mm": push.snap_mm, "clicked": 1.0 if push.clicked else 0.0, "unexplained_mm": push.unexplained_mm,
        })
        if push.repaired:
            side_effect = "repaired"
        else:
            side_effect = "disturbed" if push.travel_mm > cfg.probes.tug.slip_mm else "none"
        return push.score, side_effect


# ----------------------------------------------------------------------------- probe.tug
class Tug(_Probe):
    """Pulls back along the insertion axis up to the tug force cap, and pushes the plug back if it moved.

    The pull is one guarded move along tool -z at ``probes.tug.speed_mm_s`` with force cap
    ``tug_force_cap_n(connector)``. Its travel limit is 2 slip distances (2 x
    ``probes.tug.slip_mm``), or ``probes.tug.travel_cap_mm`` if that is less. A plug that has
    moved 2 slip distances under less than the force cap has slipped, and pulling it farther
    only takes it farther from home. If travel passed ``slip_mm`` the probe then pushes the plug
    back with the push caps, exactly as ``probe.push_seat`` does. If the plug held, the station
    releases the pull at the end of the guarded move and nothing else is done.

    Features:

    - ``disp_mm``             travel when the pull ended
    - ``slipped``             1.0 when ``disp_mm`` passed ``slip_mm``
    - ``stopped_by_force``    1.0 when the force cap ended the pull: the plug held it
    - ``stopped_by_travel``   1.0 when the travel limit ended it
    - ``force_at_stop_n``     the force estimate when the pull ended
    - ``peak_n``              the largest force estimate during the pull
    - ``force_at_slip_n``     the force estimate when travel first passed ``slip_mm``. Left out
                              when it never did
    - ``stiffness_n_per_mm``  force gained per millimeter of travel, from the start of the pull to
                              the slip point, or to the stop when it never slipped. Travel is
                              floored at 0.05 mm, which is what the position signal resolves
    - ``reseat_travel_mm``, ``reseat_stopped_by_force``, ``reseat_snap_mm``, ``reseat_clicked``,
      ``reseat_unexplained_mm``   the push back, measured as in ``probe.push_seat``. Left out
                              when the plug held
    - ``holding``             1.0 when the gripper was at the plug's width before the pull

    Side effect: ``disturbed`` when travel passed ``slip_mm``, whatever the push back did.

    Score, in 0 to 1::

        hold   = 1 / (1 + (disp_mm / slip_mm) ** 2)    when the force cap ended the pull, else 0
        reseat = the push_seat score of the push back    when there was one, else 0
        score  = max(hold, reseat)

    A plug that held the pull with little stretch scores near 1. A plug that slipped and went
    back in with a click is seated now and scores near 1. A plug that slipped and went back in
    with no click, which is what a blocked latch does, scores under 0.05.

    What it cannot do. It cannot tell which port the plug is in. A tilted jam can hold the pull
    as well as a seated plug does. The pull stretches a held latch plug by 0.2 to 0.5 mm at the
    tool, close to ``slip_mm``, so a held plug can pass the slip line. It is then pushed back and
    scored by its stretch.
    """

    source_id = "probe.tug"
    typical_s = TUG_TYPICAL_S

    def _probe(self, station, arm, connector, cfg, state, features):
        caps = cfg.probes.tug
        limit_mm = min(caps.travel_cap_mm, TUG_PULL_SLIPS * caps.slip_mm)
        pull = _guarded(station, arm, (0.0, 0.0, -1.0), "tool", limit_mm, caps.speed_mm_s,
                        cfg.tug_force_cap_n(connector), caps.travel_cap_mm)
        disp_mm = float(pull.travel_m) * 1000.0
        slipped = disp_mm > caps.slip_mm
        held = pull.stopped_by == "force"
        features.update(_stop_flags(pull))
        features.update({"disp_mm": disp_mm, "slipped": 1.0 if slipped else 0.0,
                         "force_at_stop_n": pull.force_at_stop_n, "peak_n": pull.peak_force_n})
        features.update(_pull_shape(pull, caps.slip_mm))
        hold = 1.0 / (1.0 + (max(0.0, disp_mm) / caps.slip_mm) ** 2) if held else 0.0
        if not slipped:
            return hold, "none"
        push = _push(station, arm, connector, cfg)
        features.update({
            "reseat_travel_mm": push.travel_mm, "reseat_stopped_by_force": 1.0 if push.by_force else 0.0,
            "reseat_snap_mm": push.snap_mm, "reseat_clicked": 1.0 if push.clicked else 0.0,
            "reseat_unexplained_mm": push.unexplained_mm,
        })
        return max(hold, push.score), "disturbed"


def _pull_shape(pull: MotionResult, slip_mm: float) -> dict[str, float]:
    """Force at the slip point and the stiffness before it, from the trace of a pull."""
    arrays = _trace(pull)
    if arrays is None:
        return {}
    _, s, f = arrays
    out: dict[str, float] = {}
    past = np.flatnonzero(s > slip_mm)
    end = int(past[0]) if len(past) else len(s) - 1
    if len(past):
        out["force_at_slip_n"] = float(f[end])
    out["stiffness_n_per_mm"] = float(f[end] - f[0]) / max(float(s[end] - s[0]), STIFFNESS_MIN_TRAVEL_MM)
    return out


# ----------------------------------------------------------------------------- probe.wiggle
class Wiggle(_Probe):
    """Small lateral moves each way along tool x and tool y, each with the wiggle force cap.

    ``probes.wiggle.cycles`` times over, the tool moves out along +x, -x, +y and -y. Each move
    out is a guarded move from the starting position with force cap ``probes.wiggle.force_cap_n``
    and travel cap ``probes.wiggle.travel_cap_mm``, at 2 mm/s. After each one the tool is brought
    back to the starting position with guarded moves under the same caps, steered by the measured
    tool position. The probe ends at the starting position.

    Features:

    - ``travel_xp_mm``, ``travel_xn_mm``, ``travel_yp_mm``, ``travel_yn_mm``   travel reached each
                                       way, the mean over the cycles
    - ``lateral_compliance_mm_per_n``  the sum of those travels over the sum of the force
                                       estimates at their stops. Each force is floored at the
                                       noise of the estimate (``station.noise.force_estimate_sigma_n``)
    - ``force_stop_share``             the share of the moves out that the force cap ended
    - ``return_error_mm``              how far from the starting position the tool ended
    - ``holding``                      1.0 when the gripper was at the plug's width before the moves

    Side effect: ``disturbed`` when the tool ended more than ``probes.tug.slip_mm`` from where it
    started. ``none`` otherwise.

    Score, in 0 to 1: ``1 / (1 + (mean_travel_mm / 0.25) ** 2)``, with the mean over the 4
    directions. A plug whose shoulder is pressed on the socket face moves little under 2 N, so it
    scores high. A plug that floats in the bore, or a tool that meets nothing, scores low.

    What the 2 N cap can tell. The force estimate carries 0.8 N of noise, so a move that the
    force cap ended stopped at a real force anywhere from about 0.5 N to 2 N, and the force
    features say little. The travel is the signal. Friction holds a seated plug sideways with
    about 1 to 2 N, the same size as the cap, so whether a seated plug slides inside its
    clearance is partly chance.
    """

    source_id = "probe.wiggle"
    typical_s = WIGGLE_TYPICAL_S

    def _probe(self, station, arm, connector, cfg, state, features):
        caps = cfg.probes.wiggle
        floor_n = max(float(cfg.station.noise.force_estimate_sigma_n), 1e-6)
        home = state.arms[arm].tool_pose.position()
        travel: dict[str, list[float]] = {name: [] for name, _ in WIGGLE_AXES}
        force_sum, force_stops, moves = 0.0, 0, 0
        if WIGGLE_PATTERN == "home":
            order = [(pair, True) for _ in range(caps.cycles) for pair in WIGGLE_AXES]
        else:                                               # zigzag: all of x, back, then all of y, back
            order = []
            for axis in (WIGGLE_AXES[:2], WIGGLE_AXES[2:]):
                steps = [pair for _ in range(caps.cycles) for pair in axis]
                order += [(pair, k == len(steps) - 1) for k, pair in enumerate(steps)]
        for (name, direction), back in order:
            out = _guarded(station, arm, direction, "tool", caps.travel_cap_mm, WIGGLE_SPEED_MM_S,
                           caps.force_cap_n, caps.travel_cap_mm)
            travel[name].append(max(0.0, float(out.travel_m) * 1000.0))
            at_stop = out.force_at_stop_n if out.force_at_stop_n is not None else 0.0
            force_sum += max(float(at_stop), floor_n)
            force_stops += out.stopped_by == "force"
            moves += 1
            if back:
                _go_back(station, arm, home, caps.force_cap_n, caps.travel_cap_mm)
        error_mm = _go_back(station, arm, home, caps.force_cap_n, caps.travel_cap_mm)
        means = {name: float(np.mean(values)) for name, values in travel.items()}
        total = sum(sum(values) for values in travel.values())
        features.update({f"travel_{name}_mm": value for name, value in means.items()})
        features.update({"lateral_compliance_mm_per_n": total / force_sum, "force_stop_share": force_stops / moves,
                         "return_error_mm": error_mm})
        mean_travel = float(np.mean(list(means.values())))
        score = 1.0 / (1.0 + (mean_travel / PLAY_MM) ** 2)
        return score, "disturbed" if error_mm > cfg.probes.tug.slip_mm else "none"


def _go_back(station: Station, arm: str, home: np.ndarray, force_cap_n: float, travel_cap_mm: float) -> float:
    """Bring the tool back to ``home`` with guarded moves under the same caps. Returns the distance left, in mm."""
    left_mm = float("inf")
    for attempt in range(RETURN_TRIES + 1):
        delta = home - station.robot_state().arms[arm].tool_pose.position()
        left_mm = float(np.linalg.norm(delta)) * 1000.0
        if left_mm <= RETURN_TOL_MM or attempt == RETURN_TRIES:
            break
        _guarded(station, arm, delta, "base", left_mm, WIGGLE_SPEED_MM_S, force_cap_n, travel_cap_mm)
    return left_mm


PUSH_SEAT = register(PushSeat())
TUG = register(Tug())
WIGGLE = register(Wiggle())
