"""The cable loop cell: 5 cables in a rack, each picked, carried, inserted into its assigned port and released.

Privileged, simulator side. This is the world and the stand-in System 1 that works in it. The stand-in
is a scripted skill set with the failure profile of ``config/cable_cell.yaml``: it is never described
as a learned policy. Evidence sources never import this module. They see the cell through
``view`` (a StationView) and ``claim_params``.

How the cycle is simulated. The station has 1 physical plug. It stands for the cable the arm is
handling, colored as that cable. When the arm turns to another cable, the one it leaves becomes a
visual body where it lies (``cable_scene``). Every step runs on the physics: the pick closes the
fingers on the cable in its slot, the carry moves it through the air, the insertion is the zoo's
force-guarded descent, the release opens the fingers. 3 things are scripted placements, because the
simulator cannot do them or a person does them: a regrasp of a dropped cable, which the skill finds
with its camera and which lands in the jaws upright or at a bad angle drawn from config, a slip in
the jaws, which turns the cable by a drawn angle, and the hands of a person who takes over.

Failure draws are a pure function of (seed, connector, item, step, attempt), so every arm of a shift
meets the same failures in the same places.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from ...config import load_experiment
from ...paths import REPO_ROOT
from ..base import Pose
from . import cable_scene, scene
from .cable_scene import CABLE_NAMES, CABLE_RGBA, N_CABLES, build_cable_model
from .inject import (ARM, CLEARANCE_M, EXTRA_TRAVEL_M, FLAG_TOL_M, INSERT_SPEED, PLUG_TO_TOOL, RELEASE_LIFT_M,
                     STICTION_LEAD_MM, STICTION_NM, _rot_x, _rot_z)
from .station import CONTROL_DT, TOP_DOWN_SEED, MjPlugStation
from .truth import plug_metrics, truth

CELL_YAML = REPO_ROOT / "config" / "cable_cell.yaml"
STEPS = ("pick", "carry", "insert", "release")
STEP_INDEX = {s: i for i, s in enumerate(STEPS)}
APPROACH_M = 0.045
NEAR_SPEED = 0.04
FREE_SPEED = 0.25
FRAME_DT = 0.05                       # the frame log's rate: 20 Hz
HELD_TILT_DEG = 4.0                   # truth of held: the plug's axis within this of the tool's
AT_PORT_LATERAL_MM = 1.5              # truth of at_port: the tip within this of the port's axis (chamfer plus clearance)


def _lift_record(force_z: list[float], before_z: float) -> dict:
    """What the release skill keeps of its lift: the wrist's force along z on each control tick, against its value
    before the jaws opened. Its peak, its low and its mean over the 1st quarter of the lift."""
    force = np.asarray(force_z or [before_z], dtype=float) - float(before_z)
    force = force[np.isfinite(force)] if np.isfinite(force).any() else np.zeros(1)
    early = force[: max(1, len(force) // 4)]
    return {"lift_peak_n": float(force.max()), "lift_min_n": float(force.min()), "lift_early_mean_n": float(early.mean())}


@lru_cache(maxsize=2)
def cell_config(path: str | None = None) -> dict:
    return yaml.safe_load(Path(path or CELL_YAML).read_text(encoding="utf-8"))


def _rung(name: str):
    """The module of a ladder rung (``rung_<name>.py`` beside this file)."""
    import importlib

    if name not in CableCell.RUNGS:
        raise KeyError(f"no rung {name!r}. Known: {', '.join(CableCell.RUNGS)}")
    return importlib.import_module(f"{__package__}.rung_{name}")


@dataclass
class SkillResult:
    """What a stand-in skill reports about 1 step. ``flag`` is the skill's own success flag."""

    step: str
    item: int
    ok: bool
    flag: bool
    seconds: float
    note: str = ""
    insert: dict | None = None        # the insertion record, as the zoo's (flag, stopped_by, travel_mm, trace, ...)
    failure: str | None = None        # privileged: the failure injected on this attempt, for the evaluator only
    release: dict | None = None       # the release record: per attempt on this cable, the gripper reading before the lift


@dataclass
class CableState:
    index: int
    port: int
    where: str = "rack"               # rack, physical, placed (a visual where it came to rest)
    pose: tuple | None = None         # (pos, quat) of a placed cable
    final: dict = field(default_factory=dict)      # privileged: truth when it was last let go


class CableCell:
    """5 cables, 5 assigned ports, 1 arm. The world of the cable loop, and its stand-in skills."""

    probes = ("probe.push_seat", "probe.tug")          # what ``avail_probe`` may run here

    def __init__(self, seed: int, connector: str, *, profile: str = "easy", noise: bool = True):
        self.seed, self.connector, self.profile = int(seed), connector, profile
        self.station = MjPlugStation(seed, connector, profile=profile, noise=noise, model_builder=build_cable_model)
        st = self.station
        self.cfg = cell_config()
        exp = load_experiment()
        self._exp = exp
        self._c = sorted(exp.connectors).index(connector)
        rng = np.random.default_rng([self.seed, self._c, 5556])
        ports = rng.permutation(np.arange(1, scene.N_PORTS + 1))[:N_CABLES]
        self.cables = [CableState(k, int(ports[k])) for k in range(N_CABLES)]
        self.physical: int | None = None
        self.release_ref: dict[int, list[float]] = {}
        self.latch_blocked_for: set[int] = set()
        self.hover_xy: dict[int, tuple[float, float]] = {}      # where the last carry took each cable, board x and y
        self.release_log: dict[int, list[dict]] = {}            # the release skill's own record, per cable and attempt
        # S5c enrichment only (docs/07, 2026-11-13): (low, high) degrees of tip given to the plug as the jaws open on a
        # 1st release try. None everywhere else, so no shift and no E3b cycle ever has it
        self.release_tip: tuple[float, float] | None = None
        self._seat_offsets: list[list[float]] = []      # S5c fix: tool xy minus taught seated xy, each insertion home by its flag
        # the frame log and what a replay needs
        self.frames: list[dict] = []
        self.scene_events: list[dict] = []
        self.tag = "autonomous"
        self.item_now, self.step_now = -1, ""
        self._next_frame_t = 0.0
        self._extra_hook = None
        self._force_log: list[float] | None = None
        st._tick_hook = self._on_tick
        self._place_rack()
        self._hide_physical()
        self._record_scene()

    # ------------------------------------------------------------------ what the cycle driver sees
    @property
    def view(self):
        from ...evidence.base import StationView

        return StationView(self.station)

    @property
    def variant(self) -> str:
        return self.connector

    @property
    def n_items(self) -> int:
        return N_CABLES

    def port_of(self, item: int) -> int:
        return self.cables[item].port

    def pull_out_force_n(self) -> float:
        """The force a pull that takes a plug back out of a bore may reach: config ``pull_out``, not a probe cap."""
        return float(self.cfg["pull_out"]["force_x_tug_cap"]) * self._exp.tug_force_cap_n(self.connector)

    def target_of(self, item: int) -> dict:
        """What a claim about this cable names besides the item: its assigned port."""
        return {"port": int(self.cables[item].port)}

    def now(self) -> float:
        return float(self.station._mj_data.time)

    def begin_assist(self) -> int:
        """A person takes over: frames are tagged assisted from here. Returns the handoff frame index."""
        self._log_frame(self.now())
        self.tag = "assisted"
        return len(self.frames) - 1

    def end_assist(self) -> int:
        """The person hands back. Returns the handback frame index."""
        self._log_frame(self.now())
        self.tag = "autonomous"
        return len(self.frames) - 1

    def marks_to_port_xy(self, marks: dict | None):
        """A person's tap on the wrist image, turned into a point on the board's face plane with the camera's own
        calibration and the taught face height. What a robot would do with the marks. Nothing privileged."""
        if not marks or marks.get("u") is None:
            return None
        st = self.station
        cam = marks.get("camera", f"{ARM}_camera_d405")
        frame = st.observe([cam]).frames[cam]
        k_inv = np.linalg.inv(np.asarray(frame.intrinsics, float))
        ray = k_inv @ np.array([float(marks["u"]), float(marks["v"]), 1.0])
        t = np.asarray(frame.extrinsics, float)
        d, o = t[:3, :3] @ ray, t[:3, 3]
        face_z = st.nominal().face_z
        if abs(d[2]) < 1e-9:
            return None
        s = (face_z - o[2]) / d[2]
        p = o + s * d
        return float(p[0]), float(p[1])

    # ------------------------------------------------------------------ the world

    def _wrist_force(self) -> np.ndarray:
        """The wrist's force estimate as the arm reads it, without drawing the read stream's noise, so sampling it
        changes nothing later in the cycle."""
        st = self.station
        _, _, jac = st._jac(ARM, st._q_read(ARM))
        return np.asarray(st._tool_force(ARM, jac), dtype=float)

    def _on_tick(self) -> None:
        if self._extra_hook is not None:
            self._extra_hook()
        if self._force_log is not None:                 # the release skill's lift is running (S5c fix)
            self._force_log.append(float(self._wrist_force()[2]))
        t = float(self.station._mj_data.time)
        if t + 1e-9 >= self._next_frame_t:
            # on a fixed 20 Hz grid. Stepping from the tick's own time drifted to 1 frame per 13 ticks of 4 ms, 19.2 Hz
            self._next_frame_t += FRAME_DT
            if self._next_frame_t <= t:
                self._next_frame_t = t + FRAME_DT
            self._log_frame(t)

    def _log_frame(self, t: float) -> None:
        st = self.station
        q = st._q_read(ARM)
        grip = float(st._mj_data.qpos[st._arms[ARM].ids["finger_qadr"]]) / 0.0375
        act = st._ctrl_cmd[ARM]
        self.frames.append({
            "frame_index": len(self.frames), "t": t, "item": self.item_now, "step": self.step_now, "tag": self.tag,
            **{f"state_j{i}": float(q[i]) for i in range(6)}, "state_grip": grip,
            **{f"action_j{i}": float(act[i]) for i in range(6)},
            "action_grip": float(st._arms[ARM].grip_target) / 0.0375,
            "qpos": st._mj_data.qpos.astype(np.float32).tolist(),
        })

    def _record_scene(self) -> None:
        """What a replay needs besides qpos: where every visual cable and the tray pocket are, and the plug's color."""
        m, ids = self.station._mj_model, self.station._lay.ids
        self.scene_events.append({
            "frame_index": len(self.frames),
            "vis": {k: (m.body_pos[ids[f"body:vcable{k}"]].tolist(), m.body_quat[ids[f"body:vcable{k}"]].tolist())
                    for k in range(N_CABLES)},
            "tray": (m.body_pos[ids["body:tray"]].tolist(), m.body_quat[ids["body:tray"]].tolist()),
            "plug_rgba": m.mat_rgba[ids["mat:plug_body"]].tolist(),
        })

    def _place_rack(self) -> None:
        m, ids, cfg = self.station._mj_model, self.station._lay.ids, self.station._cfg
        center, yaw = cable_scene.rack_pose(cfg)
        m.body_pos[ids["body:cable_rack"]] = center
        m.body_quat[ids["body:cable_rack"]] = scene._quat_z_rad(yaw)
        for k in range(N_CABLES):
            self._vis_to_slot(k)

    def _slot_jitter(self, k: int) -> tuple[float, float, float]:
        rng = np.random.default_rng([self.seed, self._c, k, 5557])
        return (float(rng.uniform(-0.0006, 0.0006)), float(rng.uniform(-0.0006, 0.0006)), math.radians(rng.uniform(-3, 3)))

    def _slot_pose(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Where cable k's shoulder sits in its slot, and its rotation. Privileged, true."""
        st = self.station
        center, yaw = cable_scene.slot_world(st._cfg, k)
        dx, dy, dyaw = self._slot_jitter(k)
        c, s = math.cos(yaw), math.sin(yaw)
        pos = np.array([center[0] + c * dx - s * dy, center[1] + s * dx + c * dy, cable_scene.shoulder_z_in_slot(st._lay)])
        return pos, _rot_z(yaw + dyaw)

    def _vis_to_slot(self, k: int) -> None:
        m, ids = self.station._mj_model, self.station._lay.ids
        pos, rot = self._slot_pose(k)
        m.body_pos[ids[f"body:vcable{k}"]] = pos
        m.body_quat[ids[f"body:vcable{k}"]] = Pose.from_matrix(pos, rot).quat
        self.cables[k].where = "rack"

    def _vis_hide(self, k: int) -> None:
        m, ids = self.station._mj_model, self.station._lay.ids
        m.body_pos[ids[f"body:vcable{k}"]] = cable_scene.HIDDEN

    def _plug_pose(self) -> tuple[np.ndarray, np.ndarray]:
        st = self.station
        st._mj.mj_kinematics(st._mj_model, st._mj_data)
        site = st._lay.ids["site:plug_shoulder"]
        return st._mj_data.site_xpos[site].copy(), st._mj_data.site_xmat[site].reshape(3, 3).copy()

    def _tip_plug(self, item: int, attempt: int) -> str:
        """Enrichment only (docs/07, 2026-11-13): tip the plug, just let go, by an angle drawn from ``release_tip``
        about a horizontal axis of drawn direction, pivoting on its shoulder at the socket face. That is what the
        opening did on dev seed 21,188 (0.8 to 3.4 degrees) before the lift dragged the plug up. The lift that follows
        is the skill's own and unchanged, so any drag, and its force, is the simulator's. A draw of its own, so no other
        draw of the cycle changes. Returns the failure's name for the evaluator's record."""
        st = self.station
        rng = np.random.default_rng([self.seed, self._c, item, attempt, 9191])
        lo, hi = self.release_tip
        theta = math.radians(float(rng.uniform(lo, hi)))
        phi = float(rng.uniform(0.0, 2.0 * math.pi))
        axis = np.array([math.cos(phi), math.sin(phi), 0.0])
        k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
        turn = np.eye(3) + math.sin(theta) * k + (1.0 - math.cos(theta)) * (k @ k)      # Rodrigues
        data, ids = st._mj_data, st._lay.ids
        qa, va = ids["plug_qadr"], ids["plug_dadr"]
        pos = data.qpos[qa:qa + 3].copy()
        rot = Pose((0, 0, 0), tuple(data.qpos[qa + 3:qa + 7])).rotation()
        shoulder, _ = self._plug_pose()
        new_rot = turn @ rot
        new_pos = shoulder + turn @ (pos - shoulder)
        data.qpos[qa:qa + 3] = new_pos
        data.qpos[qa + 3:qa + 7] = Pose.from_matrix(new_pos, new_rot).quat
        data.qvel[va:va + 6] = 0.0
        st._mj.mj_forward(st._mj_model, data)
        return f"release_tip_{math.degrees(theta):.1f}deg"

    def _hide_physical(self) -> None:
        st = self.station
        qa, va = st._lay.ids["plug_qadr"], st._lay.ids["plug_dadr"]
        st._mj_data.qpos[qa:qa + 3] = (0.9, -0.55, scene.TABLE_TOP_Z + 0.2)
        st._mj_data.qpos[qa + 3:qa + 7] = (1.0, 0.0, 0.0, 0.0)
        st._mj_data.qvel[va:va + 6] = 0.0
        self.physical = None

    def _freeze_physical(self) -> None:
        """The physical plug becomes a visual cable where it lies, so the plug body can stand for another."""
        k = self.physical
        if k is None:
            return
        self.cables[k].final = {p: self.truth_of(k, p) for p in ("seated", "latched", "released_stable")}
        pos, rot = self._plug_pose()
        m, ids = self.station._mj_model, self.station._lay.ids
        m.body_pos[ids[f"body:vcable{k}"]] = pos
        m.body_quat[ids[f"body:vcable{k}"]] = Pose.from_matrix(pos, rot).quat
        self.cables[k].where = "placed"
        self.cables[k].pose = (pos.tolist(), Pose.from_matrix(pos, rot).quat)
        self._hide_physical()

    def _make_physical(self, k: int, where: str = "slot") -> None:
        """Cable k becomes the physical plug: in its slot, with the tray pocket under it, in its own color."""
        if self.physical == k:
            return
        self._freeze_physical()
        st = self.station
        m, ids, data = st._mj_model, st._lay.ids, st._mj_data
        m.mat_rgba[ids["mat:plug_body"]] = CABLE_RGBA[k]
        if where == "slot":
            center, yaw = cable_scene.slot_world(st._cfg, k)
            m.body_pos[ids["body:tray"]] = (center[0], center[1], scene.TABLE_TOP_Z)
            m.body_quat[ids["body:tray"]] = scene._quat_z_rad(yaw)
            pos, rot = self._slot_pose(k)
        else:                                           # a placed cable: the plug body goes where its visual was
            pos = np.array(self.cables[k].pose[0])
            rot = Pose((0, 0, 0), tuple(self.cables[k].pose[1])).rotation()
        qa, va = ids["plug_qadr"], ids["plug_dadr"]
        data.qpos[qa:qa + 3] = pos
        data.qpos[qa + 3:qa + 7] = Pose.from_matrix(pos, rot).quat
        data.qvel[va:va + 6] = 0.0
        self._vis_hide(k)
        self.cables[k].where = "physical"
        self.physical = k
        st._retention.latch_blocked = k in self.latch_blocked_for
        st._mj.mj_forward(m, data)
        self._record_scene()

    # ------------------------------------------------------------------ privileged reads (evaluator, OracleHuman)
    def holding(self) -> bool:
        """The fingers hold the physical plug. Privileged."""
        return plug_metrics(self.station)["finger_contacts"] >= 2 if self.physical is not None else False

    def in_hand_tilt_deg(self) -> float:
        """Angle between the plug's axis and the tool's, degrees. Privileged."""
        st = self.station
        _, plug_rot = self._plug_pose()
        _, tool_rot = st._fk(ARM, st._q(ARM))
        cosang = float(np.clip(plug_rot[:, 2] @ (-tool_rot[:, 2]), -1.0, 1.0))
        return math.degrees(math.acos(cosang))

    def truth_of(self, item: int, predicate: str) -> bool:
        """Whether a predicate really holds for a cable now. Privileged: the evaluator and OracleHuman only."""
        cable = self.cables[item]
        if self.physical != item:
            if predicate in ("seated", "latched", "released_stable"):
                return bool(cable.final.get(predicate, False))
            return False
        tr = truth(self.station, cable.port, release_ref=self.release_ref.get(item))
        p, m = tr["predicates"], tr["metrics"]
        if predicate == "held":
            return bool(self.holding())
        if predicate == "at_port":
            port_pos, _ = scene.port_pose_world(self.station._cfg, self.station._lay, cable.port)
            tip = self._tip()
            lateral = float(np.hypot(*(tip[:2] - port_pos[:2]))) * 1000.0
            return bool(self.holding() and self.in_hand_tilt_deg() <= HELD_TILT_DEG and lateral <= AT_PORT_LATERAL_MM
                        and tip[2] > port_pos[2])
        if predicate == "seated":
            return bool(p["seated"])
        if predicate == "latched":
            return bool(p["latched"])
        if predicate == "released_stable":
            # the task file's definition: the gripper let go and the plug is still seated in its assigned port (and
            # latched, on latch). truth.py's released_stable also asks that the plug moved under 0.5 mm as it was let
            # go, which the definition does not ask: on dev1 that alone failed 1 cable that stayed seated and latched
            let_go = bool(m["finger_contacts"] < 2 and m["gripper_open"][ARM])
            ok = let_go and bool(p["seated"])
            return ok and (bool(p["latched"]) if self.connector == "latch" else True)
        raise KeyError(f"unknown predicate {predicate!r}")

    def _tip(self) -> np.ndarray:
        st = self.station
        st._mj.mj_kinematics(st._mj_model, st._mj_data)
        return st._mj_data.site_xpos[st._lay.ids["site:plug_tip"]].copy()

    def item_success(self, item: int) -> bool:
        """The cable ended where the task wants it: seated (and latched) in its port, let go, still. Privileged."""
        key = "released_stable"
        if self.physical == item:
            return self.truth_of(item, key)
        return bool(self.cables[item].final.get(key, False))

    def settle_item(self, item: int) -> None:
        """Record the final truth of the cable in play once a cable's steps are over. Privileged.

        That is the physical cable, which is the named one unless the skill picked the wrong one."""
        k = self.physical
        if k is not None:
            self.cables[k].final = {p: self.truth_of(k, p) for p in ("seated", "latched", "released_stable")}

    # ------------------------------------------------------------------ failure draws
    def _draw(self, item: int, step: str, attempt: int) -> np.random.Generator:
        return np.random.default_rng([self.seed, self._c, item, STEP_INDEX[step], attempt, 5151])

    # ------------------------------------------------------------------ motions
    def _to_pose(self, pose: Pose, speed: float) -> bool:
        st = self.station
        res = st.move_to_pose(ARM, pose, speed)
        if not res.ok and res.stopped_by == "error":
            q, err, _ = st.solve_ik(ARM, pose, seed_q=np.array(TOP_DOWN_SEED))
            if err < 2e-3:
                from .oracle import _joint_move

                _joint_move(st, q)
                return True
        return bool(res.ok)

    def _hover_over(self, pos: np.ndarray, yaw: float, height: float) -> bool:
        tool_rot = _rot_z(yaw) @ PLUG_TO_TOOL
        target = Pose.from_matrix(np.array([pos[0], pos[1], pos[2] + height]), tool_rot)
        st = self.station
        q, err, _ = st.solve_ik(ARM, target, seed_q=np.array(TOP_DOWN_SEED))
        if err > 2e-3:
            return False
        from .oracle import _joint_move

        return _joint_move(st, q, seconds=1.2)

    def _grasp_tool_pose(self, shoulder: np.ndarray, plug_rot: np.ndarray) -> Pose:
        yaw = float(np.arctan2(plug_rot[1, 0], plug_rot[0, 0]))
        return Pose.from_matrix(shoulder + _rot_z(yaw) @ np.array([0.0, 0.0, scene.GRASP_HEIGHT]), _rot_z(yaw) @ PLUG_TO_TOOL)

    def _pick_from_slot(self, k: int) -> bool:
        """Physically pick cable k from its slot: hover, open, down, close, up."""
        st = self.station
        self._make_physical(k, "slot")
        shoulder, plug_rot = self._plug_pose()
        grasp = self._grasp_tool_pose(shoulder, plug_rot)
        if not self._hover_over(grasp.position() - np.array([0, 0, scene.GRASP_HEIGHT]), float(np.arctan2(plug_rot[1, 0], plug_rot[0, 0])),
                                scene.GRASP_HEIGHT + APPROACH_M):
            return False
        opening = min(1.0, (st._lay.geom.grip_width / 2 + 0.008) / 0.0375)
        st.set_gripper(ARM, opening)
        ok = st.move_to_pose(ARM, grasp, NEAR_SPEED).ok
        st.set_gripper(ARM, 0.0)
        st.wait(0.2)
        up = Pose((grasp.pos[0], grasp.pos[1], grasp.pos[2] + APPROACH_M), grasp.quat)
        ok = st.move_to_pose(ARM, up, NEAR_SPEED).ok and ok
        return ok

    def _put_back(self, k: int) -> None:
        """Return the cable in the jaws to its own slot and let go of it there (a retry of a wrong pick)."""
        st = self.station
        center, yaw = cable_scene.slot_world(st._cfg, k)
        m, ids = st._mj_model, st._lay.ids
        m.body_pos[ids["body:tray"]] = (center[0], center[1], scene.TABLE_TOP_Z)
        m.body_quat[ids["body:tray"]] = scene._quat_z_rad(yaw)
        st._mj.mj_forward(m, st._mj_data)
        self._carry_plug_to(np.array([center[0], center[1], cable_scene.shoulder_z_in_slot(st._lay) - st._lay.geom.full_depth]),
                            yaw, CLEARANCE_M + 0.002)
        st.move_relative(ARM, [0.0, 0.0, -0.03], "base", 0.02, 3.0, CLEARANCE_M + 0.004)
        st.set_gripper(ARM, 1.0)
        pose = st.robot_state().arms[ARM].tool_pose
        st.move_to_pose(ARM, Pose((pose.pos[0], pose.pos[1], pose.pos[2] + APPROACH_M), pose.quat), NEAR_SPEED)
        st.wait(0.3)
        self._freeze_physical()
        self._vis_to_slot(k)                            # it stands in its slot again
        self._record_scene()

    def _carry_plug_to(self, tip_target: np.ndarray, yaw: float, clearance: float, lateral_m=(0.0, 0.0)) -> bool:
        """Move the plug in the jaws so its tip is ``clearance`` above ``tip_target`` (plus ``lateral_m``),
        keeping the plug as it sits in the jaws. Uses where the plug really is: the stand-in found the port
        with its camera, as in every zoo state."""
        st = self.station
        shoulder, plug_rot = self._plug_pose()
        tool_pos, tool_rot = st._fk(ARM, st._q(ARM))
        rot_g = plug_rot.T @ tool_rot
        off_g = plug_rot.T @ (tool_pos - shoulder)
        tool_rot_new = _rot_z(yaw) @ PLUG_TO_TOOL
        plug_rot_new = tool_rot_new @ rot_g.T
        tip_in_plug = np.array([0.0, 0.0, -st._lay.geom.full_depth])
        tip_goal = tip_target + np.array([lateral_m[0], lateral_m[1], clearance])
        shoulder_goal = tip_goal - plug_rot_new @ tip_in_plug
        tool_goal = shoulder_goal + plug_rot_new @ off_g
        above = Pose.from_matrix(tool_goal + np.array([0.0, 0.0, 0.03]), tool_rot_new)
        ok = self._to_pose(above, FREE_SPEED)
        ok = st.move_to_pose(ARM, Pose.from_matrix(tool_goal, tool_rot_new), NEAR_SPEED).ok and ok
        return ok

    def _attach_in_hand(self, tilt_deg: float = 0.0) -> None:
        """Put the physical plug in the jaws at the tool, upright or cocked by ``tilt_deg`` about the line between
        the pads, and close on it. A scripted placement: a regrasp the skill makes with its camera, a slip, or a
        person's hands."""
        st = self.station
        data, lay = st._mj_data, st._lay
        stt = st._arms[ARM]
        tool_pos, tool_rot = st._fk(ARM, st._q(ARM))
        rot_g = _rot_x(math.radians(tilt_deg)) @ PLUG_TO_TOOL
        off_g = np.array([0.0, 0.0, scene.GRASP_HEIGHT])
        rot_p = tool_rot @ rot_g.T
        sh = tool_pos - rot_p @ off_g
        quat = Pose.from_matrix(sh, rot_p).quat
        qa, va = lay.ids["plug_qadr"], lay.ids["plug_dadr"]
        half = lay.geom.grip_width / 2 + 0.0015
        data.qpos[stt.ids["finger_qadr"]], data.qpos[stt.ids["finger2_qadr"]] = half, -half
        stt.grip_target = 0.0
        for _ in range(60):
            data.qpos[qa:qa + 3] = sh
            data.qpos[qa + 3:qa + 7] = quat
            data.qvel[va:va + 6] = 0.0
            st._hold(CONTROL_DT)
        st.wait(0.2)

    def _loose_target(self, item: int) -> int:
        """The cable a regrasp goes after: the one the skill was handling and lost, else the named one."""
        if self.physical is not None and not self.holding():
            return self.physical
        return item

    def _regrasp(self, item: int, rng: np.random.Generator, bad_p: float | None = None) -> str:
        """The cable is out of the jaws: go to it and take it again. Returns '' or 'regrasp_bad_angle'.

        The fingers hold nothing when this runs, so the physical plug can stand for the cable taken."""
        st = self.station
        if self.physical != item:
            self._make_physical(item, "placed" if self.cables[item].where == "placed" else "slot")
        shoulder, plug_rot = self._plug_pose()
        if plug_metrics(st)["in_bore"] or self.cables[item].where == "rack":
            # a plug in a port or a slot stands upright: the fingers close on it there and draw it straight out
            grasp = self._grasp_tool_pose(shoulder, plug_rot)
            self._hover_over(shoulder, float(np.arctan2(plug_rot[1, 0], plug_rot[0, 0])), scene.GRASP_HEIGHT + APPROACH_M)
            st.set_gripper(ARM, min(1.0, (st._lay.geom.grip_width / 2 + 0.008) / 0.0375))
            st.move_to_pose(ARM, grasp, NEAR_SPEED)
            st.set_gripper(ARM, 0.0)
            st.wait(0.2)
            st.move_relative(ARM, [0.0, 0.0, 0.03], "base", 0.01, self.pull_out_force_n(), 0.03)
            pose = st.robot_state().arms[ARM].tool_pose
            st.move_to_pose(ARM, Pose((pose.pos[0], pose.pos[1], pose.pos[2] + APPROACH_M), pose.quat), NEAR_SPEED)
            self.cables[item].where = "physical"
            return ""
        yaw = st.nominal().board_yaw
        self._hover_over(shoulder, yaw, scene.GRASP_HEIGHT + APPROACH_M)
        tool = Pose.from_matrix(shoulder + np.array([0.0, 0.0, scene.GRASP_HEIGHT + 0.004]), _rot_z(yaw) @ PLUG_TO_TOOL)
        st.set_gripper(ARM, 1.0)
        st.move_to_pose(ARM, tool, NEAR_SPEED)
        bad_p = self.cfg["failures"]["regrasp_bad_angle"] if bad_p is None else bad_p
        tilt, failure = 0.0, ""
        if rng.random() < bad_p:
            tilt = float(rng.uniform(*self.cfg["failures"]["regrasp_tilt_deg"])) * float(rng.choice([-1.0, 1.0]))
            failure = "regrasp_bad_angle"
        self._attach_in_hand(tilt)
        pose = st.robot_state().arms[ARM].tool_pose
        st.move_to_pose(ARM, Pose((pose.pos[0], pose.pos[1], pose.pos[2] + APPROACH_M), pose.quat), NEAR_SPEED)
        return failure

    # ------------------------------------------------------------------ the stand-in skills
    def run_skill(self, step: str, item: int, attempt: int = 0, *, marks: dict | None = None,
                  marks_port_xy: tuple[float, float] | None = None) -> SkillResult:
        """Run 1 step of the stand-in System 1 for 1 cable. ``attempt`` 0 is the first try. ``marks`` are a
        person's taps on the wrist image (a carry then goes where they marked)."""
        st = self.station
        if marks is not None and marks_port_xy is None:
            marks_port_xy = self.marks_to_port_xy(marks)
        self.item_now, self.step_now = item, step
        t0 = float(st._mj_data.time)
        rng = self._draw(item, step, attempt)
        fail = self.cfg["failures"]
        failure, insert, note, ok, flag = None, None, "", True, True
        cable = self.cables[item]
        port_pos, board_yaw = scene.port_pose_world(st._cfg, st._lay, cable.port)
        if step == "pick":
            # a retry first puts back a cable that is not the one asked for
            if self.physical is not None and self.physical != item and self.holding():
                self._put_back(self.physical)
            if self.physical == item and self.holding() and self.in_hand_tilt_deg() > HELD_TILT_DEG:
                self._put_back(item)
            target = item
            in_rack = [k for k in range(N_CABLES) if self.cables[k].where == "rack" and k != item]
            if rng.random() < fail["wrong_cable"] and in_rack:
                target = int(in_rack[int(rng.integers(0, len(in_rack)))])
                failure = "wrong_cable"
            if self.physical == item and self.holding():
                pass                                    # already in the jaws, straight
            elif self.cables[target].where == "rack":
                ok = self._pick_from_slot(target)
            else:
                failure = (self._regrasp(target, rng) or failure)
            jaw = st.robot_state().arms[ARM].gripper_pos
            flag = bool(jaw > 0.05)                     # the skill's own check: the fingers stopped on something
        elif step == "carry":
            # the skill carries whatever is in its jaws. With nothing there, it first takes again the cable it lost
            if not self.holding():
                failure = self._regrasp(self._loose_target(item), rng) or None
            # a drop or a slip happens on the way, between the rack and the board
            mid_tool = st.robot_state().arms[ARM].tool_pose
            mid = Pose(((mid_tool.pos[0] + port_pos[0]) / 2, (mid_tool.pos[1] + port_pos[1]) / 2,
                        max(mid_tool.pos[2], port_pos[2] + 0.09)), mid_tool.quat)
            self._to_pose(mid, FREE_SPEED)
            u = rng.random()
            if u < fail["drop"]:
                st.set_gripper(ARM, 1.0)
                st.wait(0.4)
                st.set_gripper(ARM, 0.0)
                failure = "drop"
            elif u < fail["drop"] + fail["slip"]:
                tilt = float(rng.uniform(*fail["slip_tilt_deg"])) * float(rng.choice([-1.0, 1.0]))
                self._attach_in_hand(tilt)
                failure = "slip"
            # a person's marks give the port's position on the board. Else the stand-in found it with its camera,
            # unless it takes the cable over the wrong port
            target_xy = port_pos[:2]
            if marks_port_xy is not None:
                target_xy = np.asarray(marks_port_xy, float)
            elif rng.random() < fail["wrong_port"]:
                free = self._free_neighbors(cable.port)
                if free:
                    wrong_pos, _ = scene.port_pose_world(st._cfg, st._lay, int(free[int(rng.integers(0, len(free)))]))
                    target_xy = wrong_pos[:2]
                    failure = failure or "wrong_port"
            self.hover_xy[item] = (float(target_xy[0]), float(target_xy[1]))
            if self.holding():
                ok = self._carry_plug_to(np.array([target_xy[0], target_xy[1], port_pos[2]]), board_yaw, CLEARANCE_M)
            else:                                       # nothing in the jaws: it goes on to the port anyway
                ok = self._to_pose(Pose((port_pos[0], port_pos[1], port_pos[2] + 0.08), mid_tool.quat), FREE_SPEED)
            flag = bool(ok)
        elif step == "insert":
            insert, failure, flag = self._insert(item, rng, attempt)
            ok = insert is not None
        elif step == "release":
            held = self.holding()
            force_before = self._wrist_force()
            if held and rng.random() < fail["stuck_release"]:
                failure = "stuck_release"               # the fingers do not open
                st.wait(0.3)
            else:
                st.set_gripper(ARM, 1.0)
                if self.release_tip is not None and held and attempt == 0 and self.physical == item:
                    failure = self._tip_plug(item, attempt)    # enrichment tables only: the opening tips the plug
            # the wrist's force as the jaws opened (S5c fix): a plug that was pressed or pulled as it was let go shows here
            open_force_change = self._wrist_force() - force_before
            if held and self.physical is not None:
                self.release_ref[self.physical] = plug_metrics(st)["plug_pos"]
            state = st.robot_state().arms[ARM]          # the skill's own record: its gripper reading before it lifts
            pose = state.tool_pose
            seat_z = float(st.nominal().seated_tool_pose(cable.port).pos[2])
            entry = {"attempt": int(attempt), "jaw_before_lift": float(state.gripper_pos),
                     "lift_mm": RELEASE_LIFT_M * 1000.0, "open_above_seat_mm": (float(state.tool_pose.pos[2]) - seat_z) * 1000.0,
                     "open_force_dz_n": float(open_force_change[2]),
                     "open_force_dxy_n": float(np.hypot(open_force_change[0], open_force_change[1]))}
            self.release_log.setdefault(item, []).append(entry)
            # the lift is the same position move as before. The wrist's force is sampled on every control tick while it
            # runs (S5c fix), so the record shows a plug that rode up with the fingers, and the motion does not change
            self._force_log = []
            st.move_to_pose(ARM, Pose((pose.pos[0], pose.pos[1], pose.pos[2] + RELEASE_LIFT_M), pose.quat), 0.03)
            entry.update(_lift_record(self._force_log, float(force_before[2])))
            self._force_log = None
            st.wait(self._exp.predicates["released_stable"]["settle_s"])
            flag = True
        else:
            raise KeyError(f"unknown step {step!r}. Known: {', '.join(STEPS)}")
        release = {"attempts": [dict(a) for a in self.release_log.get(item, [])]} if step == "release" else None
        return SkillResult(step=step, item=item, ok=bool(ok), flag=bool(flag), seconds=float(st._mj_data.time) - t0,
                           note=note, insert=insert, failure=failure, release=release)

    def _insert(self, item: int, rng: np.random.Generator, attempt: int, *, no_failure: bool = False):
        """The insertion: from the pre-insert pose, the zoo's force-guarded descent, with a failure drawn from the mix.
        ``no_failure``: the descent alone, with no failure drawn (the ladder's reinsert rung, docs/07 2026-11-16)."""
        st = self.station
        exp, fail, mix_cfg = self._exp, self.cfg["failures"], self.cfg["insert_failures"]
        cable = self.cables[item]
        port = cable.port
        failure = None
        # a retry first takes the plug back out of whatever it is in, or takes again the cable it lost
        if self.holding():
            m = plug_metrics(st)
            if m["in_bore"]:
                st.move_relative(ARM, [0.0, 0.0, 0.02], "base", 0.01, self.pull_out_force_n(), 0.025)
        else:
            self._regrasp(self._loose_target(item), rng, bad_p=0.0)
        if not no_failure and rng.random() < fail["insert"]:
            mix = {k: v for k, v in exp.standin_s1["failure_mix"].items() if k != "wrong_port"}   # that one is the carry's
            kinds = sorted(mix)
            p = np.array([mix[k] for k in kinds], float)
            failure = str(rng.choice(kinds, p=p / p.sum()))
            if failure == "unlatched" and self.connector != "latch":
                failure = "under_push"
        port_pos, board_yaw = scene.port_pose_world(st._cfg, st._lay, port)
        lateral = (0.0, 0.0)
        stick_mm = None
        if failure == "lateral_offset":
            angle = float(rng.uniform(0, 2 * math.pi))
            off = float(rng.uniform(*mix_cfg["lateral_offset_mm"])) * 1e-3
            lateral = (off * math.cos(angle), off * math.sin(angle))
            tilt = float(rng.uniform(*mix_cfg["lateral_tilt_deg"])) * float(rng.choice([-1.0, 1.0]))
            self._attach_in_hand(tilt)
        if failure == "under_push":
            stick_mm = float(rng.uniform(*mix_cfg["under_push_shortfall_mm"]))
        if failure == "early_release":
            stick_mm = float(rng.uniform(*mix_cfg["early_release_shortfall_mm"]))
        if failure == "unlatched":
            self.latch_blocked_for.add(self.physical)
            st._retention.latch_blocked = True
        # it goes down where the carry left it: over its port, or over the wrong one
        hx, hy = self.hover_xy.get(item, (float(port_pos[0]), float(port_pos[1])))
        self._carry_plug_to(np.array([hx, hy, port_pos[2]]), board_yaw, CLEARANCE_M, lateral_m=lateral)
        st.zero_force_estimate(ARM)
        nominal = st.nominal()
        tool_z = float(st.robot_state().arms[ARM].tool_pose.pos[2])
        expected = tool_z - float(nominal.seated_tool_pose(port).pos[2])
        depth = st._lay.geom.full_depth
        max_travel = CLEARANCE_M + depth + EXTRA_TRAVEL_M
        ret = st._retention
        state = {"stuck": False}

        def hook() -> None:
            if stick_mm is not None and not state["stuck"] and ret.captured and ret.shortfall_mm <= stick_mm + STICTION_LEAD_MM:
                st._set_stiction(ARM, STICTION_NM)
                state["stuck"] = True

        self._extra_hook = hook
        try:
            res = st.move_relative(ARM, [0.0, 0.0, -max_travel], "base", INSERT_SPEED, exp.push_force_cap_n(self.connector),
                                   max_travel)
        finally:
            self._extra_hook = None
        st.wait(0.3)
        flag = bool(res.stopped_by == "force" and abs(res.travel_m - expected) <= FLAG_TOL_M)
        if flag:
            # the robot's own estimate of where the board really is (S5c fix): its tool, home in a port by its own
            # record, against the taught seated pose. Read from the joint readings without the read stream's noise
            pos, _ = st._fk(ARM, st._q_read(ARM))
            self._seat_offsets.append(((np.asarray(pos[:2], float) - np.asarray(nominal.seated_tool_pose(port).pos[:2], float))
                                       * 1000.0).tolist())
        if failure == "early_release":
            st.set_gripper(ARM, 1.0)                    # it lets go short of home and calls it done
            self.release_ref[self.physical] = plug_metrics(st)["plug_pos"]
            pose = st.robot_state().arms[ARM].tool_pose
            st.move_to_pose(ARM, Pose((pose.pos[0], pose.pos[1], pose.pos[2] + RELEASE_LIFT_M), pose.quat), 0.03)
        record = {"flag": flag, "stopped_by": res.stopped_by, "travel_mm": res.travel_m * 1000.0,
                  "expected_travel_mm": expected * 1000.0, "force_at_stop_n": res.force_at_stop_n,
                  "peak_force_n": res.peak_force_n, "seconds": res.duration_s, "trace": res.trace}
        return record, failure, flag

    # ------------------------------------------------------------------ the escalation ladder (docs/07, 2026-11-16)
    RUNGS = ("relook", "regrip", "reinsert")            # reseat was dropped on 2026-11-16 (archive/s5d_rung_reseat)

    def rung_begin(self, name: str, item: int, step: str):
        """A rung of the ladder: the cell's recovery action, 1 file each (``rung_<name>.py``). Returns the step's new
        result when the rung made one (a new carry or insertion), else None."""
        self.item_now, self.step_now = item, step
        return _rung(name).begin(self, item, step)

    def rung_end(self, name: str, item: int, step: str) -> None:
        _rung(name).end(self, item, step)

    def _free_neighbors(self, port: int) -> list[int]:
        cfg = self._exp.station.board
        row, col = divmod(port - 1, cfg.cols)
        taken = {c.port for c in self.cables}
        out = []
        for r, c in ((row, col - 1), (row, col + 1), (row - 1, col), (row + 1, col)):
            if 0 <= r < cfg.rows and 0 <= c < cfg.cols:
                p = r * cfg.cols + c + 1
                if p not in taken:
                    out.append(p)
        return out

    # ------------------------------------------------------------------ what a check gets
    def claim_params(self, item: int, step: str, result: SkillResult | None = None) -> dict:
        """What the evidence sources get for a claim: the station's taught geometry, the cable's identity and port,
        the insertion record from the step's skill result (when the skill made one) and the claim-time frames.
        Nothing privileged."""
        from ...evidence.base import FrameStore

        from .cable_teach import load_palette

        st = self.station
        frames = FrameStore()
        frames.put_observation("claim", st.observe([f"{ARM}_camera_d405", "top_camera_d405"]))
        palette = load_palette()                        # each cable's color as the wrist camera saw it, taught on dev seeds
        params = {"nominal": st.nominal(), "connector": self.connector, "arm": ARM, "frames": frames,
                  "cable": {"index": item, "name": CABLE_NAMES[item], "rgba": list(palette[item]), "palette": palette},
                  "port": self.cables[item].port, "step": step}
        insert = getattr(result, "insert", None)
        if insert is not None:
            params["insert"] = insert
        release = getattr(result, "release", None)
        if release is not None:
            params["release"] = release
        if self._seat_offsets:                           # the board's offset as the robot measured it this cycle
            params["board_offset_mm"] = [float(v) for v in np.median(np.asarray(self._seat_offsets), axis=0)]
        return params

    def close(self) -> None:
        self.station._tick_hook = None
        self.station.close()
