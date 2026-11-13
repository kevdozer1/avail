"""Failure injection: an oracle insertion with a controlled error. Privileged.

``scripted_insert(seed, connector, error_spec)`` starts with the plug in the gripper above the
port, runs a force-guarded descent, and makes the requested near miss happen the way it happens
on a real arm: a joint sticks before the plug is home and the skill's force threshold trips on
the stuck joint instead of on the socket floor. The plug ends up short, the skill reports
success, and the robot is left holding the plug in a physically consistent state.

This module reads ground truth to decide when the joint sticks and to label the result. It
is evaluator-side code. Evidence sources never import it.

The insertion's own success flag is computed from what the skill can see: it stopped on force
and its tool travelled about as far as the taught geometry said it would.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from ...config import HARD_PROFILES, load_experiment
from ..base import MotionResult, Pose
from . import scene
from .station import TOP_DOWN_SEED, MjPlugStation
from .truth import truth

ARM = "right"
CLEARANCE_M = 0.010                 # tip height above the socket face when the descent starts
INSERT_SPEED = 0.008                # m/s
EXTRA_TRAVEL_M = 0.006              # the descent may run this far past the taught depth
FLAG_TOL_M = 0.005                  # the skill calls itself done if it stopped on force within this of the taught depth
STICTION_NM = (4.0, 4.0, 4.0, 1.5, 1.5, 1.5)
STICTION_LEAD_MM = 0.09             # the tool coasts about this far after the joints stick
RELEASE_LIFT_M = 0.020
CLASSES = ("P0", "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9")
SHORTFALL_RANGE_MM = {"N1": (0.5, 1.0), "N2": (1.0, 2.0), "N3": (2.0, 4.0)}


@dataclass(frozen=True)
class ErrorSpec:
    """One injected state. ``cls`` is the intended class, the rest says how it is made."""

    cls: str
    shortfall_mm: float | None = None            # N1 to N4: how far from home the joint sticks
    wrong_port: int | None = None                # N6: the port the plug goes into
    lateral_mm: tuple[float, float] | None = None   # N7: offset at entry, board frame
    tilt_deg: tuple[float, float] | None = None     # N7: the plug cocked in the fingers, about the tool's x axis
    drop_mm: float | None = None                 # N8: how far into the descent the fingers open
    release: bool = False                        # P0 only: let go and back away (the positive twin of N9)
    latch_blocked: bool = False                  # N5
    jam_kind: str | None = None                  # hard mode, N1 to N4: "burr" or "wedge" stops the plug at shortfall_mm
    wedge_hold_n: float | None = None            # hard mode, a wedge: the pull it holds at its stop

    def to_dict(self) -> dict:
        return asdict(self)


def class_mix(connector: str) -> dict[str, float]:
    """Probability of each class in the zoo. Near misses (N1, N2, N3, N5) are half of all negatives."""
    cfg = load_experiment().zoo
    pos, neg = cfg.positive_share, 1.0 - cfg.positive_share
    near = cfg.near_miss_share_of_negatives * neg
    near_classes = ["N1", "N2", "N3"] + (["N5"] if connector == "latch" else [])
    mix = {"P0": pos}
    for c in near_classes:
        mix[c] = near / len(near_classes)
    rest = neg - near
    for c, share in (("N4", 0.27), ("N6", 0.27), ("N7", 0.23), ("N8", 0.13), ("N9", 0.10)):
        mix[c] = rest * share
    return mix


def sample_error(seed: int, connector: str, cls: str | None = None, profile: str = "easy") -> ErrorSpec:
    """The injected error for a seed. A pure function of (seed, connector, cls, profile).

    Hard mode draws the same class and the same shortfall as easy mode, then decides from its
    own random stream whether a short plug (N1 to N4) is made by a jam instead of a sticking joint.
    """
    spec = _sample_error_easy(seed, connector, cls)
    if profile == "easy" or spec.shortfall_mm is None:
        return spec
    if profile not in HARD_PROFILES:
        raise ValueError(f"profile must be 'easy' or one of {HARD_PROFILES}, got {profile!r}")
    from dataclasses import replace

    from .hard import jam_of, sample_hard

    kind, hold = jam_of(sample_hard(seed, connector), spec.cls, connector)
    return replace(spec, jam_kind=kind, wedge_hold_n=hold) if kind else spec


def _sample_error_easy(seed: int, connector: str, cls: str | None = None) -> ErrorSpec:
    """The injected error of easy mode. A pure function of (seed, connector, cls)."""
    exp = load_experiment()
    names = sorted(exp.connectors)
    rng = np.random.default_rng([int(seed), names.index(connector), 4242])
    mix = class_mix(connector)
    drawn = str(rng.choice(list(mix), p=np.array(list(mix.values())) / sum(mix.values())))
    released_twin = bool(rng.random() < mix["N9"] / mix["P0"])
    if cls is None:
        cls = drawn
    if cls not in CLASSES:
        raise ValueError(f"unknown class {cls!r}. Known: {', '.join(CLASSES)}")
    if cls == "N5" and connector != "latch":
        raise ValueError("class N5 (unlatched) exists only for the latch connector")
    depth = exp.connector(connector).full_depth_mm
    if cls == "P0":
        return ErrorSpec("P0", release=released_twin)
    if cls in SHORTFALL_RANGE_MM:
        lo, hi = SHORTFALL_RANGE_MM[cls]
        margin = 0.12 * (hi - lo)
        return ErrorSpec(cls, shortfall_mm=round(float(rng.uniform(lo + margin, hi - margin)), 3))
    if cls == "N4":
        return ErrorSpec("N4", shortfall_mm=round(float(depth - rng.uniform(1.0, 3.0)), 3))
    if cls == "N5":
        return ErrorSpec("N5", latch_blocked=True)
    if cls == "N6":
        target = scene.sample_scene(seed, connector).target_port
        cfg = exp.station.board
        row, col = divmod(target - 1, cfg.cols)
        neighbors = [r * cfg.cols + c + 1 for r, c in ((row, col - 1), (row, col + 1), (row - 1, col), (row + 1, col))
                     if 0 <= r < cfg.rows and 0 <= c < cfg.cols]
        return ErrorSpec("N6", wrong_port=int(rng.choice(neighbors)))
    if cls == "N7":
        angle = rng.uniform(0, 2 * math.pi)
        off = rng.uniform(0.7, 1.3)
        tilt = rng.uniform(4.0, 8.0) * rng.choice([-1.0, 1.0])
        rng.random()                                        # keeps the draws after this one where they were
        return ErrorSpec("N7", lateral_mm=(round(off * math.cos(angle), 3), round(off * math.sin(angle), 3)),
                         tilt_deg=(round(tilt, 2), 0.0))
    if cls == "N8":
        return ErrorSpec("N8", drop_mm=round(float(rng.uniform(0.0, 14.0)), 2))
    return ErrorSpec("N9")


@dataclass
class InsertOutcome:
    """What the injection produced. ``flag`` and ``insert`` are what the skill itself saw."""

    seed: int
    connector: str
    spec: ErrorSpec
    target_port: int                  # the port named in the claim
    claim_predicate: str
    flag: bool
    insert: MotionResult
    expected_travel_m: float
    truth: dict
    landed_class: str
    used_fallback: bool = False
    release_ref: list[float] | None = None
    released: bool = False
    seconds: float = 0.0


def classify(tr: dict, connector: str, released_cmd: bool = False) -> str:
    """The class a state belongs to by ground truth."""
    m, p = tr["metrics"], tr["predicates"]
    in_target = m["in_bore"] and m["nearest_port"] == tr["target_port"]
    if m["finger_contacts"] < 2 and not (m["in_bore"] and p["seated"]) and not released_cmd:
        return "N8"
    if m["in_bore"] and not in_target:
        return "N6"
    if not m["in_bore"]:
        return "N7" if m["finger_contacts"] >= 2 else "N8"
    if m["tilt_deg"] > load_experiment().predicates["seated"]["tilt_tol_deg"]:
        return "N7"
    short = m["shortfall_mm"]
    if p["seated"]:
        if connector == "latch" and not p["latched"]:
            return "N5"
        if released_cmd and not p["released_stable"]:
            return "N9"
        return "P0"
    if short < 0.5:
        return "N7"                  # deep enough yet not straight
    if short < 1.0:
        return "N1"
    if short < 2.0:
        return "N2"
    if short <= 4.0:
        return "N3"
    return "N4"


def _rot_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


PLUG_TO_TOOL = np.diag([1.0, -1.0, -1.0])      # tool x across the fingers, tool z down the plug axis


def _grasp_transform(cfg: scene.SceneConfig, extra_tilt: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Where the tool point sits in the plug frame for this episode's grasp: (rotation, offset).

    The plug can turn in a parallel grip about the line between the pads, which is the tool's
    x axis. ``extra_tilt`` (rad) adds to the episode's small grasp tilt about that axis.
    """
    axial, across, along, _ = cfg.grasp_offset
    # The episode's small grasp tilt is not used. A plug held even 1 degree off in this stiff grip
    # wedges in a bore with 0.25 mm of clearance, and when the fingers open the stored contact
    # energy throws it out of the socket. A real grip gives way and the bore straightens the plug.
    # Only the tilted jam (N7) is cocked in the fingers, on purpose.
    return _rot_x(extra_tilt) @ PLUG_TO_TOOL, np.array([across, along, scene.GRASP_HEIGHT + axial])


def start_pose(station: MjPlugStation, port: int, lateral_m=(0.0, 0.0), extra_tilt: float = 0.0,
               clearance_m: float = CLEARANCE_M) -> tuple[Pose, np.ndarray, np.ndarray]:
    """The tool pose an insertion starts from, and the grasp it holds the plug with. Privileged.

    Height: the taught pre-insert height of the port, from nominal geometry, so the start carries
    this episode's localization error the way a real cell's does. Across the board: the plug's tip
    straight over the true port center, plus ``lateral_m``. This stands in for a robot that found
    the port with its camera before it went down. Orientation: the tool straight down at the true
    board yaw.
    """
    cfg, lay = station._cfg, station._lay
    port_pos, yaw = scene.port_pose_world(cfg, lay, port)
    rot_g, off_g = _grasp_transform(cfg, extra_tilt)
    tool_rot = _rot_z(yaw) @ PLUG_TO_TOOL
    plug_rot = tool_rot @ rot_g.T
    lever = plug_rot @ (off_g + np.array([0.0, 0.0, lay.geom.full_depth]))      # from the plug's tip to the tool point
    taught_z = float(station.nominal().pre_insert_tool_pose(port, clearance_m).pos[2])
    tip_xy = port_pos[:2] + np.asarray(lateral_m, dtype=float)
    tool_pos = np.array([tip_xy[0] + lever[0], tip_xy[1] + lever[1], taught_z])
    return Pose.from_matrix(tool_pos, tool_rot), rot_g, off_g


def hold_plug_at(station: MjPlugStation, tool_pose: Pose, rot_g: np.ndarray, off_g: np.ndarray) -> None:
    """Put the right arm at ``tool_pose``, then put the plug in its fingers with the given grasp.

    Privileged: the arm and the plug are placed by writing simulator state. Used to start every
    injected state with the plug already picked.
    """
    mujoco, data, model, lay = station._mj, station._mj_data, station._mj_model, station._lay
    q, pos_err, rot_err = station.solve_ik(ARM, tool_pose, seed_q=np.array(TOP_DOWN_SEED))
    if pos_err > 2e-4 or rot_err > 2e-3:
        raise RuntimeError(f"the arm cannot reach the plug pose: position error {pos_err * 1000:.2f} mm")
    st = station._arms[ARM]
    ids = st.ids
    data.qpos[ids["qadr"]] = q
    data.qvel[ids["dadr"]] = 0.0
    st.q_plan, st.qd_plan = q.copy(), np.zeros(6)
    st.tau_ff, st.force_cmd = np.zeros(6), np.zeros(3)
    half = lay.geom.grip_width / 2 + 0.0015
    data.qpos[ids["finger_qadr"]], data.qpos[ids["finger2_qadr"]] = half, -half
    st.grip_target = half
    qa, va = lay.ids["plug_qadr"], lay.ids["plug_dadr"]
    data.qpos[qa:qa + 3] = (0.9, -0.55, scene.TABLE_TOP_Z + 0.2)       # out of the way until the fingers are around it
    data.qvel[va:va + 6] = 0.0
    station._hold_offsets(ARM)                              # the arm holds its pose under its own weight from here
    station._set_gaps()                                     # hard mode v2: each servo's gears touching on the side that holds it
    station._command_all()
    # place the plug in the fingers, relative to where the tool really is, and close on it
    tool_pos, tool_rot = station._fk(ARM, station._q(ARM))
    rot_p = tool_rot @ rot_g.T
    sh = tool_pos - rot_p @ off_g
    quat = Pose.from_matrix(sh, rot_p).quat
    st.grip_target = 0.0
    for _ in range(100):                                  # pin the plug while the fingers close on it
        data.qpos[qa:qa + 3] = sh
        data.qpos[qa + 3:qa + 7] = quat
        data.qvel[va:va + 6] = 0.0
        station._hold(0.004)
    station.wait(0.3)


def _insertion_flag(result: MotionResult, expected_travel_m: float) -> bool:
    """The skill's own success flag. Uses only what the skill can see."""
    return bool(result.stopped_by == "force" and abs(result.travel_m - expected_travel_m) <= FLAG_TOL_M)


def scripted_insert(seed: int, connector: str, error_spec: ErrorSpec | None = None, *, noise: bool = True,
                    recorder=None, profile: str = "easy") -> tuple[MjPlugStation, InsertOutcome]:
    """Run an oracle insertion with a controlled error. Returns the station, still holding the plug, and the outcome.

    ``recorder(station)`` is called on every control tick of the descent. It is an evaluator hook
    for measuring the real forces. Nothing the skill reports depends on it.

    Hard mode: a short plug with a ``jam_kind`` is stopped by the jam stop in the target bore at
    its shortfall. Once its tip is at the stop, the arm holds where it is, as for a stuck joint, so
    the plug stays at the stop when the descent lets go. A capped push still cannot pass the stop.
    """
    spec = error_spec or sample_error(seed, connector, profile=profile)
    exp = load_experiment()
    if spec.jam_kind is not None and spec.shortfall_mm is None:
        raise ValueError("a jam needs the shortfall at which it stops the plug")
    jam = (spec.jam_kind, float(spec.shortfall_mm), spec.wedge_hold_n) if spec.jam_kind is not None else None
    station = MjPlugStation(seed, connector, latch_blocked=spec.latch_blocked, noise=noise, profile=profile, jam=jam)
    cfg, lay = station._cfg, station._lay
    target = cfg.target_port
    motion_port = spec.wrong_port if spec.wrong_port is not None else target
    port_pos, yaw = scene.port_pose_world(cfg, lay, motion_port)
    depth = lay.geom.full_depth

    # the plug starts in the fingers above the port it is going into, at the taught height
    lateral, extra_tilt = (0.0, 0.0), 0.0
    if spec.cls == "N7":
        lateral = tuple(_rot_z(yaw)[:2, :2] @ np.array(spec.lateral_mm) * scene.MM)
        extra_tilt = math.radians(spec.tilt_deg[0])
    tool_pose, rot_g, off_g = start_pose(station, motion_port, lateral, extra_tilt)
    hold_plug_at(station, tool_pose, rot_g, off_g)
    station.zero_force_estimate(ARM)

    # the taught geometry says how far the tool should travel to seat the plug in the claimed port
    nominal = station.nominal()
    tool_z = float(station.robot_state().arms[ARM].tool_pose.pos[2])
    expected_travel = tool_z - float(nominal.seated_tool_pose(target).pos[2])
    max_travel = CLEARANCE_M + depth + EXTRA_TRAVEL_M
    force_cap = exp.push_force_cap_n(connector)

    state = {"stuck": False, "dropped": False}
    ret = station._retention
    tool_start = float(station._fk(ARM, station._q(ARM))[0][2])     # privileged, like the drop depth it is compared with

    # A jammed plug (hard mode) is held at its stop by the arm, the way a stuck-joint plug is held in easy
    # mode. Without it the soft servos let the detent push a plug that stops in the push-out zone back up.
    jam_band = load_experiment().hard(profile).jams.wedge_seat_mm if spec.jam_kind is not None else 0.0
    lead = jam_band if spec.jam_kind is not None else STICTION_LEAD_MM

    def hook() -> None:
        if recorder is not None:
            recorder(station)
        if spec.shortfall_mm is not None and not state["stuck"] and ret.captured \
                and ret.shortfall_mm <= spec.shortfall_mm + lead:
            station._set_stiction(ARM, STICTION_NM)
            state["stuck"] = True
        if spec.drop_mm is not None and not state["dropped"]:
            fallen = tool_start - float(station._fk(ARM, station._q(ARM))[0][2])
            if fallen * 1000.0 >= spec.drop_mm:
                station._arms[ARM].grip_target = 0.0375
                state["dropped"] = True

    t0 = float(station._mj_data.time)
    station._tick_hook = hook
    try:
        result = station.move_relative(ARM, [0.0, 0.0, -max_travel], "base", INSERT_SPEED, force_cap, max_travel)
    finally:
        station._tick_hook = None
    station.wait(0.3)                                     # a stuck joint stays stuck until the arm is next told to move
    flag = _insertion_flag(result, expected_travel)

    released, release_ref = False, None
    if spec.cls == "P0" and spec.release:
        release_ref = truth(station, target)["metrics"]["plug_pos"]
        station.set_gripper(ARM, 1.0)
        pose = station.robot_state().arms[ARM].tool_pose
        up = Pose((pose.pos[0], pose.pos[1], pose.pos[2] + RELEASE_LIFT_M), pose.quat)
        station.move_to_pose(ARM, up, 0.03)
        station.wait(exp.predicates["released_stable"]["settle_s"])
        released = True
    elif spec.cls == "N9":
        release_ref = truth(station, target)["metrics"]["plug_pos"]
        station.wait(exp.predicates["released_stable"]["settle_s"])      # the release was commanded and nothing opened

    release_claim = spec.cls == "N9" or (spec.cls == "P0" and spec.release)
    if release_claim:
        predicate = "released_stable"
    else:
        predicate = "latched" if connector == "latch" else "seated"
    tr = truth(station, target, release_ref=release_ref)
    landed = classify(tr, connector, released_cmd=release_claim)
    outcome = InsertOutcome(
        seed=int(seed), connector=connector, spec=spec, target_port=target, claim_predicate=predicate, flag=flag,
        insert=result, expected_travel_m=float(expected_travel), truth=tr, landed_class=landed,
        release_ref=release_ref, released=released, seconds=float(station._mj_data.time) - t0,
    )
    return station, outcome
