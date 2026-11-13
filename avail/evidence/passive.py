"""Passive sources: skill_flag, pose_residual, jaw_state and click. They cost nothing and move nothing.

Each one reads what the station already has when a claim is raised: the insertion skill's own
record, the joint encoders through forward kinematics, the gripper reading, and the taught
geometry. None of them commands a motion, so ``cost.seconds`` is 0 and ``moved_robot`` is false.

What each can tell, in 1 line:

- ``passive.skill_flag``     the skill's own word. It is the baseline everybody trusts today.
- ``passive.pose_residual``  where the tool stopped against the taught seated pose. The taught pose
                             is off by about 1 mm and the plug sits up to 3 mm high or low in the
                             fingers, so it is weak for a shortfall under 2 mm.
- ``passive.jaw_state``      whether the fingers still hold something of the plug's width. It says
                             nothing about how deep the plug is.
- ``passive.click``          whether the insertion ended with the snap of a plug going home.

A source gives a score only for a predicate in its ``predicates`` set. For any other claim it
still returns the features it could measure, with score None. A feature that could not be
measured is left out.
"""

from __future__ import annotations

import math

import numpy as np

from ..schemas import Claim, Cost, Evidence
from ..station.base import NominalGeometry, Pose, Station
from .base import make_evidence, register

SEATING_PREDICATES = frozenset({"seated", "latched"})


# ----------------------------------------------------------------------------- small helpers
def _number(value) -> float | None:
    """A finite float, or None for anything else. True and False are refused."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _insert_record(params: dict) -> dict:
    """What the insertion skill recorded, or an empty dict when the caller has none."""
    record = params.get("insert")
    return record if isinstance(record, dict) else {}


def _nominal(params: dict) -> NominalGeometry | None:
    nominal = params.get("nominal")
    return nominal if isinstance(nominal, NominalGeometry) else None


def _holding_arm(params: dict) -> str | None:
    """The arm that holds the plug: ``params['arm']``, else the taught geometry's holding arm."""
    arm = params.get("arm")
    if isinstance(arm, str) and arm:
        return arm
    nominal = _nominal(params)
    return nominal.holding_arm if nominal is not None else None


def _claimed_port(claim: Claim, nominal: NominalGeometry) -> int | None:
    """The port named in the claim, when it is one of the board's ports."""
    port = claim.predicate.target.get("port")
    if isinstance(port, bool) or not isinstance(port, int):
        return None
    return port if 1 <= port <= len(nominal.ports_board) else None


# ----------------------------------------------------------------------------- passive.skill_flag
class SkillFlag:
    """The insertion skill's own success flag, read from ``params['insert']['flag']``.

    Feature: ``flag``, 1.0 when the skill reported success and 0.0 when it did not.

    Score: ``flag``. For a release claim it is still the insertion's flag, because the release
    step records none.

    In the simulated station the skill sets its flag from what it can see: it stopped on force
    within 5 mm of the taught travel. A joint that sticks 1 mm early trips the same force
    threshold, so the flag is 1 on a short insertion. It answers nothing about the port or the latch.

    With no ``insert`` record, or no flag in it, the evidence has no features and no score.
    """

    source_id = "passive.skill_flag"
    kind = "passive_proprio"
    predicates = {"seated", "latched", "at_port", "released_stable"}

    def estimate_cost(self, claim: Claim) -> Cost:
        return Cost.zero()

    def collect(self, station: Station, claim: Claim, params: dict) -> Evidence:
        flag = _insert_record(params).get("flag")
        if isinstance(flag, (bool, np.bool_)):
            value = 1.0 if flag else 0.0
        else:
            number = _number(flag)                      # 1 and 0 are accepted. Text is refused
            value = number if number in (0.0, 1.0) else None
        if value is None:
            return make_evidence(self, claim, {})
        score = value if claim.predicate.name in self.predicates else None
        return make_evidence(self, claim, {"flag": value}, score=score)


# ----------------------------------------------------------------------------- passive.pose_residual
DEEP_BAND_MM = 2.0            # a low grasp and the localization error explain a tool this far past the taught depth
DEEP_WEIGHT = 4.0             # each millimeter past that band counts as 4 mm of shortfall
LATERAL_BAND_MM = 3.0         # the localization error and the grasp explain a tool this far off the port axis
TILT_BAND_DEG = 2.0           # the tilt tolerance of the seated predicate
TILT_WEIGHT_MM_PER_DEG = 2.0  # each degree past the band counts as 2 mm of shortfall
WRONG_PORT_MM = 25.0          # a tool nearest to another port counts as 25 mm of shortfall


def pose_residual_features(tool: Pose, nominal: NominalGeometry, port: int | None) -> dict[str, float]:
    """The measured tool pose against the taught seated pose. Pure geometry, no station call.

    ``axial_mm`` is how far the tool is short of the taught seated depth along the insertion
    axis. Positive means short. ``lateral_mm`` is the distance across that axis. Both need the
    claimed port. ``port_index_est`` is the port whose taught seated pose is nearest to the tool
    across the axis, and ``port_match`` is 1.0 when that is the claimed port. ``tilt_deg`` is the
    angle between the tool axis and the taught axis.
    """
    pos = tool.position()
    axis = nominal.tool_rotation()[:, 2]                # the taught insertion axis: tool +z, down at the board in v0
    features: dict[str, float] = {}
    cos = float(np.clip(tool.rotation()[:, 2] @ axis, -1.0, 1.0))
    features["tilt_deg"] = math.degrees(math.acos(cos))

    def residual(index: int) -> tuple[float, float]:
        delta = pos - nominal.seated_tool_pose(index).position()
        along = float(axis @ delta)
        return -along * 1000.0, float(np.linalg.norm(delta - along * axis)) * 1000.0

    ports = range(1, len(nominal.ports_board) + 1)
    if len(ports):
        nearest = min(ports, key=lambda index: residual(index)[1])
        features["port_index_est"] = float(nearest)
        if port is not None:
            features["axial_mm"], features["lateral_mm"] = residual(port)
            features["port_match"] = 1.0 if nearest == port else 0.0
    return features


def pose_residual_score(features: dict[str, float], predicate: str, full_depth_mm: float) -> float | None:
    """The hand-made score of ``passive.pose_residual``. The class docstring states the formula."""
    needed = ("axial_mm", "lateral_mm", "tilt_deg", "port_match")
    if any(key not in features for key in needed):
        return None
    axial = features["axial_mm"]
    over = max(0.0, -axial - DEEP_BAND_MM)
    lateral_excess = max(0.0, features["lateral_mm"] - LATERAL_BAND_MM)
    tilt_excess = max(0.0, features["tilt_deg"] - TILT_BAND_DEG)
    wrong_port = WRONG_PORT_MM * (1.0 - features["port_match"])
    if predicate in SEATING_PREDICATES:
        return -(axial + DEEP_WEIGHT * over + lateral_excess + TILT_WEIGHT_MM_PER_DEG * tilt_excess + wrong_port)
    if predicate == "at_port":
        above_mouth = max(0.0, axial - full_depth_mm)
        return -(above_mouth + DEEP_WEIGHT * over + lateral_excess + wrong_port)
    return None


class PoseResidual:
    """The measured tool pose against the taught seated pose of the claimed port.

    The tool pose comes from the joint encoders through forward kinematics
    (``robot_state().arms[arm].tool_pose``). The taught pose is
    ``params['nominal'].seated_tool_pose(port)`` with ``port`` from the claim's target.

    Features:

    - ``axial_mm``        how far the tool is short of the taught seated depth along the insertion
                          axis. Positive means short
    - ``lateral_mm``      distance from the taught pose across that axis
    - ``port_index_est``  the port whose taught pose is nearest to the tool across the axis
    - ``port_match``      1.0 when that is the claimed port
    - ``tilt_deg``        angle between the tool axis and the taught axis

    Score for ``seated`` and ``latched``, in millimeters of apparent shortfall with the sign flipped::

        over_mm        = max(0, -axial_mm - 2)        the tool went deeper than a low grasp explains
        lateral_excess = max(0, lateral_mm - 3)       farther off the axis than localization explains
        tilt_excess    = max(0, tilt_deg - 2)         more tilt than the seated predicate allows
        score = -(axial_mm + 4 * over_mm + lateral_excess + 2 * tilt_excess + 25 * (1 - port_match))

    A tool that stopped high lowers the score 1 for 1. A tool that stopped a little low raises it,
    because a low reading is what a low grasp of a seated plug looks like. Past 2 mm low the
    score falls 3 for 1: nothing stopped the tool where a seated plug would have.

    Score for ``at_port``: the same without the shortfall term. A plug in the mouth of the right
    port counts, so only a tool higher than the whole socket depth is charged::

        score = -(max(0, axial_mm - full_depth_mm) + 4 * over_mm + lateral_excess + 25 * (1 - port_match))

    What it cannot do. ``axial_mm`` is the true shortfall plus the grasp offset along the plug
    (up to 3 mm either way) minus the height error of the taught port (about 0.5 mm), so a
    shortfall under 2 mm hides in it. It reads the tool and never the plug, so a plug that
    tilted or slid in the fingers looks seated, and both scores speak for the plug only while
    the fingers hold it. A release claim gets the features and no score: after a release the
    arm has backed away and the residual says nothing about the plug.
    """

    source_id = "passive.pose_residual"
    kind = "passive_proprio"
    predicates = {"seated", "latched", "at_port"}

    def estimate_cost(self, claim: Claim) -> Cost:
        return Cost.zero()

    def collect(self, station: Station, claim: Claim, params: dict) -> Evidence:
        state = station.robot_state()
        nominal, arm = _nominal(params), _holding_arm(params)
        features: dict[str, float] = {}
        score = None
        if nominal is not None and arm in state.arms:
            features = pose_residual_features(state.arms[arm].tool_pose, nominal, _claimed_port(claim, nominal))
            score = pose_residual_score(features, claim.predicate.name, nominal.full_depth_m * 1000.0)
        seconds = station.robot_state().ts - state.ts
        return make_evidence(self, claim, features, score=score, seconds=seconds)


# ----------------------------------------------------------------------------- passive.jaw_state
JAW_BAND = 0.02               # of the jaw's travel. Pad squeeze and part tolerance explain an opening this far off


class JawState:
    """Gripper position and holding torque, against the reading expected while holding the plug.

    Features:

    - ``jaw_pos``         gripper position, 0 closed to 1 open
    - ``jaw_torque``      the gripper motor's effort, in the backend's unit. It is high while the
                          fingers squeeze something and near 0 when they are open or closed on air
    - ``jaw_open_error``  ``jaw_pos`` minus ``params['nominal'].expected_jaw_pos()``, the reading
                          expected while the fingers hold the plug. Near 0 means the fingers are
                          at the plug's width

    Score for ``seated``, ``latched`` and ``held``: ``-max(0, abs(jaw_open_error) - 0.02)``. It
    answers whether the fingers still hold something of the plug's width. It is 0 for any
    opening within 0.02 of the expected one, and falls as the fingers open wider or close on air.

    Score for ``released_stable``: ``max(0, jaw_open_error - 0.02)``. It answers whether the
    fingers opened. It is 0 while the fingers are at the plug's width or tighter.

    The 0.02 band is 0.75 mm of finger travel on this gripper. Inside it every reading scores the
    same on purpose. The simulated gripper reads in steps of 0.05 mm of finger travel with 0.02 mm
    of noise, and its force with 0.3 N of noise (``station.noise`` in the config), so 2 reads of
    the same state can differ. Before the station read that coarsely, the opening differed by 2 to
    15 micrometers between a plug pressed onto its stop and one that stopped short, and a model
    fitted on the raw features could pick that up. A real gripper does not read that finely.

    What it cannot do. It says nothing about depth, port or latch: every plug that is still in
    the fingers scores the same, seated or 4 mm short. It cannot tell whether a released plug
    stayed where it was. The torque is left out of the score because its unit changes with the
    backend. It stays in the features for the fused scorer.
    """

    source_id = "passive.jaw_state"
    kind = "passive_proprio"
    predicates = {"seated", "latched", "held", "released_stable"}

    def estimate_cost(self, claim: Claim) -> Cost:
        return Cost.zero()

    def collect(self, station: Station, claim: Claim, params: dict) -> Evidence:
        state = station.robot_state()
        nominal, arm = _nominal(params), _holding_arm(params)
        features: dict[str, float] = {}
        score = None
        if arm in state.arms:
            jaw = state.arms[arm]
            features["jaw_pos"] = jaw.gripper_pos
            features["jaw_torque"] = jaw.gripper_torque
            expected = _number(nominal.expected_jaw_pos()) if nominal is not None else None
            position = _number(jaw.gripper_pos)
            if expected is not None and position is not None:
                error = position - expected
                features["jaw_open_error"] = error
                if claim.predicate.name == "released_stable":
                    score = max(0.0, error - JAW_BAND)
                elif claim.predicate.name in self.predicates:
                    score = -max(0.0, abs(error) - JAW_BAND)
        seconds = station.robot_state().ts - state.ts
        return make_evidence(self, claim, features, score=score, seconds=seconds)


# ----------------------------------------------------------------------------- passive.click
SNAP_WINDOW_S = 0.024             # a click moves the tool 0.3 to 0.5 mm inside about 20 ms
SNAP_AVERAGE_S = 0.012            # travel is averaged over this long before a gain is taken
TAIL_MM = 3.0                     # the click is looked for in the last 3 mm of travel
SNAP_MIN_MM = 0.20                # a jump at least this far past the commanded motion counts as a click
SNAP_FULL_MM = 0.30               # the score is full at this jump
PEAK_LOOKBACK_MM = 1.0            # the force barrier of a click sits inside the last millimeter before the jump
HARD_STOPS = ("force", "stall")   # the stop reasons that mean something stopped the tool
MIN_TRACE_SAMPLES = 12


def _trace_arrays(record: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """The insertion trace as 3 equal-length finite arrays (time, travel, force), or None."""
    trace = record.get("trace")
    if not isinstance(trace, dict):
        return None
    try:
        t, s, f = (np.asarray(trace.get(key), dtype=float) for key in ("t_s", "travel_mm", "force_n"))
    except (TypeError, ValueError):
        return None
    if t.ndim != 1 or t.shape != s.shape or t.shape != f.shape or len(t) < MIN_TRACE_SAMPLES:
        return None
    if not (np.isfinite(t).all() and np.isfinite(s).all() and np.isfinite(f).all()):
        return None
    return t, s, f


def _stopped_hard(record: dict) -> bool:
    """True when the insertion ended against something. A record with no stop reason counts as true."""
    reason = record.get("stopped_by")
    return reason in HARD_STOPS if isinstance(reason, str) else True


def click_features(record: dict) -> dict[str, float]:
    """Features of the click in an insertion record. Pure function of ``params['insert']``.

    The trace is the guarded move's own log: time, measured travel and estimated force along
    the motion. The largest travel jump in the last 3 mm is the candidate click, and the jump
    and force features all describe that 1 event.
    """
    features: dict[str, float] = {}
    travel, expected = _number(record.get("travel_mm")), _number(record.get("expected_travel_mm"))
    if travel is not None and expected is not None:
        features["travel_short_mm"] = expected - travel
    arrays = _trace_arrays(record)
    if arrays is None:
        final_force = _number(record.get("force_at_stop_n"))
        if final_force is not None:
            features["final_force_n"] = final_force
        return features
    t, s, f = arrays
    features["final_force_n"] = float(f[-1])
    dt = float(np.median(np.diff(t)))
    if not dt > 0.0:
        return features
    n_win = max(1, int(round(SNAP_WINDOW_S / dt)))
    n_avg = max(1, int(round(SNAP_AVERAGE_S / dt)))
    smooth = np.convolve(s, np.full(n_avg, 1.0 / n_avg), mode="valid")     # smooth[k] averages s[k] to s[k + n_avg - 1]
    if len(smooth) <= n_win:
        return features
    window_s = n_win * dt

    # the tail: from the last time the tool was more than 3 mm from its final travel, to the end
    outside = np.flatnonzero(smooth < smooth[-1] - TAIL_MM)
    k0 = int(outside[-1]) + 1 if len(outside) else 0

    # the travel jump: the largest gain across any window in the tail, past what the commanded speed explains
    gains = smooth[n_win:] - smooth[:-n_win]            # gains[k] is the travel gained from k to k + n_win
    tail_from = min(k0, len(gains) - 1)
    speed = _number(record.get("speed_mm_s"))           # the commanded speed, when the skill recorded it
    if speed is None or speed <= 0.0:                   # else the speed the tool held on the way in
        approach = gains[:k0] if k0 >= 4 * n_win else gains
        speed = float(np.median(approach)) / window_s
    best = tail_from + int(np.argmax(gains[tail_from:]))
    snap = float(gains[best]) - speed * window_s
    features["snap_mm"] = snap
    burst = slice(best, best + n_win + n_avg)           # the raw samples that the jump spans
    if len(s[burst]) > n_avg:
        features["snap_speed_mm_s"] = float(np.max(s[burst][n_avg:] - s[burst][:-n_avg])) / (n_avg * dt)

    # the force side of the same event: the peak in the last millimeter before the jump, and the fall across the jump
    far = np.flatnonzero(s[:best + 1] < s[best] - PEAK_LOOKBACK_MM)
    lead = int(far[-1]) + 1 if len(far) else 0
    high = lead + int(np.argmax(f[lead:best + 1]))
    features["peak_n"] = float(f[high])
    features["drop_n"] = float(f[high] - np.min(f[burst]))
    features["peak_depth_mm"] = float(s[-1] - s[high])

    features["clicked"] = 1.0 if snap >= SNAP_MIN_MM and _stopped_hard(record) else 0.0
    return features


class Click:
    """Looks back at the insertion's own trace for a click: a force peak, a drop, and a travel jump.

    A plug that goes home crosses a force barrier just before full depth and then snaps the last
    fraction of a millimeter. In the trace of the guarded insertion that shows as a jump in
    travel inside about 20 ms and a dip in the force estimate. An insertion that stopped on a
    stuck joint shows a force rise at the end with no jump before it.

    Features, all from ``params['insert']``. The largest travel jump in the last 3 mm of travel
    is the candidate click, and the 5 features at the top of the list describe that 1 event:

    - ``snap_mm``          the largest travel gain across any 24 ms, minus what the commanded
                           speed explains in 24 ms. Each end of the 24 ms is a 12 ms average, so
                           encoder noise of 0.05 mm per sample does not pass for a jump. The
                           commanded speed is ``insert['speed_mm_s']`` when the skill recorded
                           it, else the median speed of the approach
    - ``snap_speed_mm_s``  the fastest 12 ms inside that jump, as a speed
    - ``peak_n``           the largest force estimate in the 1 mm of travel before the jump
    - ``drop_n``           how far the force fell from that peak by the end of the jump
    - ``peak_depth_mm``    how far before the end of travel that peak sat
    - ``final_force_n``    the force estimate at the end of the trace
    - ``travel_short_mm``  ``expected_travel_mm`` minus ``travel_mm``
    - ``clicked``          1.0 or 0.0 by the rule below

    The click rule: ``snap_mm`` is at least 0.20 mm and the insertion then stopped on force
    (``insert['stopped_by']`` is ``force`` or ``stall``). A jump in an insertion that ran to its
    travel cap is a slip, since nothing stopped the tool. A record with no stop reason is judged
    on the jump alone. The force estimate carries about 0.8 N of noise, which is as large as the
    dip, so ``peak_n`` and ``drop_n`` are reported and kept out of the rule.

    Score: 0 when the insertion did not stop on force. Otherwise
    ``min(max(snap_mm, 0), 0.3) / 0.3``, which is 0 with no jump and 1 from a 0.3 mm jump up.
    Where there is a score, ``clicked`` is 1 exactly when the score is at least 2/3.

    What it cannot do. A click says the plug went home in some socket. It does not say which
    port. A connector with no barrier, such as a latch that is blocked, goes in without a click,
    so a missing click counts against ``latched`` more surely than against ``seated``.

    With no ``insert`` record the evidence has no features and no score. With a record and no
    usable trace it has ``travel_short_mm`` and ``final_force_n`` when those were recorded.
    """

    source_id = "passive.click"
    kind = "passive_proprio"
    predicates = {"seated", "latched"}

    def estimate_cost(self, claim: Claim) -> Cost:
        return Cost.zero()

    def collect(self, station: Station, claim: Claim, params: dict) -> Evidence:
        record = _insert_record(params)
        features = click_features(record)
        score = None
        if "snap_mm" in features and claim.predicate.name in self.predicates:
            jump = min(max(features["snap_mm"], 0.0), SNAP_FULL_MM) / SNAP_FULL_MM
            score = jump if _stopped_hard(record) else 0.0
        return make_evidence(self, claim, features, score=score)


SKILL_FLAG = register(SkillFlag())
POSE_RESIDUAL = register(PoseResidual())
JAW_STATE = register(JawState())
CLICK = register(Click())
