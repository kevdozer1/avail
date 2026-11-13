"""MjPlugStation: the simulated plug station behind the Station protocol.

What a caller can reach through the protocol is what a real cell offers: camera images,
joint positions and velocities, a joint torque estimate, a tool force estimate built from
those torques, a gripper reading, and guarded motion. The simulator objects live in
attributes whose names start with ``_mj``. Evidence sources never touch them, and a test
(tests/test_truth_wall.py) fails if they try.

Control. The YAM joint servos in ASPIRE's model are soft position servos:
``torque = kp * (target - q) - kd * qdot`` with kp 40, 20 or 10 Nm/rad. A tool force is
therefore commanded as a joint target offset, ``J^T f / kp``, and the tool force estimate is
read back from the commanded motor torques through the Jacobian, as on a real arm without a
force sensor. Free moves add velocity and inertia feedforward from the arm's own model and a
slow integral per joint. A guarded move regulates tool speed along one line with a force
that never exceeds the cap it was given, then relaxes to a position hold.

The tool point is the point between the fingertip pads (``scene.TOOL_SITE``).

Hard mode (S4b, ``profile="hard"``, config ``profiles.hard``) changes 4 things and keeps the rest:
the wrist cameras become 1080p webcams with a fisheye-style lens (``webcam.py``), a short plug
can be held by a jam stop that no push under the cap passes, the force estimate drifts, walks and
lags by a play, and the detent force follows the connector's lot. A hard-mode seed is only ever
built in hard mode (``avail.seeds.assert_profile``).

Hard mode v2 (S4c, ``profile="hard2"``, config ``profiles.hard2``) is hard mode plus an SO-101-class
servo at every arm joint. Its encoder reads the output shaft, so a joint reading is the joint's
angle on a 12-bit count grid. Its gears have lash: when a joint's torque turns, the servo passes no
torque until its motor side has crossed the lash (``_apply_lash``). Each torque reading carries the
servo's current counts and their noise. The guarded move's own speed loop and travel stop still run
on the joint angles, as the servo's internal loop would. What the robot reports, and so every
check, sees the readings: the joint positions, the tool pose and travel from forward kinematics of
those positions, the camera extrinsics, the torque estimate and the tool force estimate built from it.
"""

from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ...config import HARD_PROFILES, PROFILES, load_experiment
from ..base import ArmState, CameraFrame, MotionResult, NominalGeometry, Observation, Pose, RobotState
from . import aspire, scene
from .retention import Retention

CONTROL_DT = 0.004                 # 250 Hz
FORCE_RATE = 15.0                  # N/s, how fast a guarded move may build force
FORCE_KP = 1.0                     # N per unit of normalized speed error
STALL_S = 0.15                     # commanded force at the cap this long with no motion counts as a force stop
FORCE_FILTER_S = 0.02
SPEED_FILTER_S = 0.012
NOISE_CORR_S = 0.05                # correlation time of the force estimate noise
INTEGRAL_GAIN = 5.0                # 1/s, joint integral in free moves and holds
INTEGRAL_DEADBAND = 2e-4           # rad
INTEGRAL_CLAMP = 0.25              # rad
SETTLE_S = 0.5
JOINT_FRICTION = 0.1               # Nm, the model's joint friction loss, used as feedforward
FRICTION_SPEED = 0.003             # rad/s, joint speed above which a joint counts as moving
TRUE_FORCE_FILTER_S = 0.02
LONG_MOVE_M = 0.05                 # beyond this a move is planned in joint space
LONG_MOVE_RAD = 0.35
MAX_JOINT_SPEED = 1.2              # rad/s in joint-space moves
STUCK_SOLIMP = (0.999, 0.999, 0.001, 0.5, 2.0)
LINK_SHORTFALL_MM = 0.5            # the link signal comes up within this of full depth
LINK_DELAY_S = 0.2
TOP_DOWN_SEED = (0.0, 1.5, 1.05, -1.12, 0.0, -1.571)   # a right-arm posture with the tool pointing down
FINGER_TRAVEL_MM = aspire.FINGER_OPEN_M * 1000.0
SUPERSAMPLE = 2                    # each pixel is the mean of 2 x 2 rendered samples
RETENTION_STATE = ("latch_blocked", "captured", "port", "depth_mm", "shortfall_mm", "detent_force", "friction_level",
                   "wedged", "wedged_at")
CAMERA_HISTORY_S = 0.1             # hard mode keeps the wrist cameras' poses this far back, for rolling shutter and motion blur


@dataclass
class _Arm:
    name: str
    ids: dict
    q_plan: np.ndarray = field(default_factory=lambda: np.zeros(6))
    qd_plan: np.ndarray = field(default_factory=lambda: np.zeros(6))
    offset: np.ndarray = field(default_factory=lambda: np.zeros(6))
    tau_ff: np.ndarray = field(default_factory=lambda: np.zeros(6))
    force_cmd: np.ndarray = field(default_factory=lambda: np.zeros(3))     # commanded tool force on the world, N
    tau_tare: np.ndarray = field(default_factory=lambda: np.zeros(6))
    grip_target: float = 0.0                                               # finger joint target, m
    grip_limit: float | None = None


class MjPlugStation:
    """The MuJoCo plug station. One live station per connector type per process."""

    station_id = "mj_plug"

    def __init__(self, seed: int, connector: str, *, latch_blocked: bool = False, noise: bool = True,
                 width: int | None = None, height: int | None = None, profile: str = "easy",
                 jam: tuple[str, float, float | None] | None = None, model_builder: Callable | None = None):
        import mujoco

        from ...seeds import assert_profile, assert_simulable

        assert_simulable(seed)                             # CLAUDE.md rule 12: no test seed before its final run
        if profile not in PROFILES:
            raise ValueError(f"profile must be one of {PROFILES}, got {profile!r}")
        assert_profile(seed, profile)                      # a hard-mode seed is never built easy, nor the other way
        if jam is not None and profile not in HARD_PROFILES:
            raise ValueError("a jam exists in hard mode only")
        self._mj = mujoco
        cfg = load_experiment()
        self.connector = connector
        self.seed = int(seed)
        self.profile = profile
        self._exp = cfg
        self._spec = cfg.connector(connector)
        self._hard_on = profile in HARD_PROFILES
        # hard mode v2 builds the hard-mode scene: it changes readings only. A cell with parts of its own
        # (the cable loop) passes ``model_builder(connector, scene_profile) -> (model, layout)``.
        build = model_builder or scene.build_model
        base_model, self._lay = build(connector, "hard" if self._hard_on else profile)
        self._mj_model = copy.deepcopy(base_model)          # stations never share a model they edit
        self._cfg = scene.sample_scene(seed, connector)
        self._hard = None                                  # hard mode's per-seed draws. Privileged
        self._jam = jam                                    # (kind, shortfall at the stop in mm, wedge hold in N). Privileged
        if self._hard_on:
            from .hard import sample_hard

            self._hard = sample_hard(seed, connector)
        scene.apply_scene(self._mj_model, self._lay, self._cfg, self._hard, None if jam is None else jam[1])
        self._mj_data = mujoco.MjData(self._mj_model)
        self._kin = mujoco.MjData(self._mj_model)          # the robot's own kinematic model
        wedge = None
        if jam is not None and jam[0] == "wedge":
            js = cfg.hard(profile).jams
            wedge = (self._cfg.target_port, float(jam[1]), float(jam[2]), js.wedge_seat_mm, js.wedge_ramp_mm, js.wedge_span_mm)
        self._retention = Retention(self._mj_model, self._lay, self._cfg, self._spec, latch_blocked=latch_blocked,
                                    detent_scale=self._hard.lot_factor if self._hard_on else 1.0, wedge=wedge)
        self._arms = {arm: _Arm(arm, self._lay.arm[arm]) for arm in aspire.ARMS}
        self._arm_roots = {int(self._mj_model.body(f"{arm}_arm").id) for arm in aspire.ARMS}
        self._substeps = max(1, int(round(CONTROL_DT / self._mj_model.opt.timestep)))
        self._noise_on = bool(noise)
        self._sigma = cfg.station.noise.force_estimate_sigma_n if noise else 0.0
        self._drift = cfg.station.noise.force_estimate_bias_drift_n_per_min / 60.0 if noise else 0.0
        force = cfg.hard(profile).force if self._hard_on else None
        if force is not None and noise:
            self._drift = force.bias_drift_n_per_min / 60.0
        self._walk = force.bias_walk_n_per_sqrt_s if (force is not None and noise) else 0.0
        self._hyst = force.hysteresis_n if (force is not None and noise) else 0.0
        self._tare_noisy = bool(force is not None and noise and force.tare_with_noise)
        self._play = {arm: np.zeros(3) for arm in aspire.ARMS}         # hard mode: the play operator's output, N
        self._hyst_off = {arm: np.zeros(3) for arm in aspire.ARMS}     # hard mode: play output minus its input, N
        self._tare_noise = {arm: np.zeros(3) for arm in aspire.ARMS}   # hard mode: what the last tare kept of the noise, N
        self._rng = np.random.default_rng([self.seed, sorted(cfg.connectors).index(connector), 77])
        self._noise = {arm: np.zeros(3) for arm in aspire.ARMS}
        self._bias = {arm: (np.array(self._cfg.force_bias) if noise else np.zeros(3)) for arm in aspire.ARMS}
        self._drift_dir = {arm: self._rng.normal(0.0, 1.0, size=3) * self._drift for arm in aspire.ARMS}
        noise_cfg = cfg.station.noise
        self._grip_res = float(getattr(noise_cfg, "gripper_position_resolution_mm", 0.0)) / FINGER_TRAVEL_MM
        self._grip_sigma = (float(getattr(noise_cfg, "gripper_position_sigma_mm", 0.0)) / FINGER_TRAVEL_MM) if noise else 0.0
        self._grip_force_sigma = float(getattr(noise_cfg, "gripper_force_sigma_n", 0.0)) if noise else 0.0
        self._qd_sigma = float(getattr(noise_cfg, "joint_velocity_sigma_rad_s", 0.0)) if noise else 0.0
        self._read_rng = np.random.default_rng([self.seed, sorted(cfg.connectors).index(connector), 78])
        # hard mode v2: the servo's encoder, gear lash and current readings. Their own streams, so hard mode's draws stay put
        joints = cfg.hard(profile).joint_readings if self._hard_on else None
        lash = cfg.hard(profile).gear_lash if self._hard_on else None
        current = cfg.hard(profile).current_readings if self._hard_on else None
        self._sense_on = joints is not None
        self._lash_on = lash is not None
        self._cur_on = current is not None
        draw = np.random.default_rng([self.seed, sorted(cfg.connectors).index(connector), 6600])
        if joints is not None:
            self._enc_step = 2.0 * math.pi / joints.counts_per_turn            # rad per count
            self._enc_phase = {arm: draw.uniform(*joints.zero_phase_counts, size=6) for arm in aspire.ARMS}
        if lash is not None:
            self._lash = {arm: np.radians(draw.uniform(*lash.lash_deg, size=6)) for arm in aspire.ARMS}
            self._lash_speed = float(lash.motor_speed_rad_s)
            self._lash_breakaway = float(lash.breakaway_nm)
        # Where each servo's motor side sits in its lash, rad, from -lash/2 to +lash/2. At an end the gears
        # touch and the servo's torque reaches the joint. In between the joint gets none.
        self._gap = {arm: np.zeros(6) for arm in aspire.ARMS}
        self._ctrl_cmd = {arm: np.zeros(6) for arm in aspire.ARMS}         # the servo target the controller set
        if current is not None:
            self._cur_count = current.count_ma / 1000.0 / current.stall_current_a   # 1 count as a share of peak torque
            self._cur_sigma = float(current.noise_counts_sd) if noise else 0.0
            self._tau_peak = {arm: np.abs(self._mj_model.actuator_forcerange[self._lay.arm[arm]["act"], 1]).copy()
                              for arm in aspire.ARMS}
        self._cur_rng = np.random.default_rng([self.seed, sorted(cfg.connectors).index(connector), 6601])
        self._cur_noise = {arm: np.zeros(6) for arm in aspire.ARMS}          # counts, drawn once per control tick
        self._width = int(width or cfg.station.render.width)
        self._height = int(height or cfg.station.render.height)
        self._renderer = None
        self._contact_buf = np.zeros(6)
        self._jacp = np.zeros((3, self._mj_model.nv))
        self._jacr = np.zeros((3, self._mj_model.nv))
        self._mass = np.zeros((self._mj_model.nv, self._mj_model.nv))
        # privileged, for the evaluator: the real vertical force the fingers put on the plug
        self._true_fz = 0.0                                # low-passed, N, positive lifts the plug
        self._true_peak_push = 0.0                         # largest low-passed downward force, N
        self._true_peak_pull = 0.0                         # largest low-passed upward force, N
        self._true_peak_raw = 0.0                          # largest instantaneous magnitude, includes impacts
        self._true_overrun_mm = -math.inf                  # the most a guarded move ran past its own travel limit, mm
        self._stop_requested = False
        self._tick_hook: Callable[[], None] | None = None  # privileged per-tick hook, used by failure injection
        self._stuck = {arm: False for arm in aspire.ARMS}  # privileged: joints held by injected stiction
        self._link_since: float | None = None
        # hard mode's wrist cameras
        self._webcams: dict = {}
        self._frame_count = 0
        self._frame_info: dict[str, dict] = {}            # evaluator-side: what the camera did for the last frame of each
        self._keep_stages: dict | None = None              # set to {} to keep each webcam's intermediate pictures, for figures
        self._cam_hist: deque = deque(maxlen=int(math.ceil(CAMERA_HISTORY_S / CONTROL_DT)) + 2)
        if self._hard_on:
            from .hard import WRIST_CAMERAS
            from .webcam import Webcam

            spec = cfg.hard(profile).wrist_camera
            for name in WRIST_CAMERAS:
                cam = self._lay.ids[f"cam:{name}"]
                self._webcams[name] = Webcam(spec, self._hard.lenses[name])
                plan = self._webcams[name].plan
                self._mj_model.cam_sensorsize[cam] = 0.0      # a plain pinhole with square pixels and this fovy
                self._mj_model.cam_fovy[cam] = plan["fovy_deg"]
            plan = next(iter(self._webcams.values())).plan
            vis = self._mj_model.vis.global_
            vis.offwidth = max(int(vis.offwidth), plan["width"], SUPERSAMPLE * self._width)
            vis.offheight = max(int(vis.offheight), plan["height"], SUPERSAMPLE * self._height)
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self) -> None:
        """Arms at their ready joints, fingers open, plug standing in the tray pocket."""
        mujoco, model, data = self._mj, self._mj_model, self._mj_data
        mujoco.mj_resetData(model, data)
        ready = aspire.ready_joints()
        for arm, st in self._arms.items():
            ids = st.ids
            q = np.array(ready[arm])
            data.qpos[ids["qadr"]] = q
            st.q_plan, st.qd_plan = q.copy(), np.zeros(6)
            st.offset, st.tau_ff, st.force_cmd, st.tau_tare = np.zeros(6), np.zeros(6), np.zeros(3), np.zeros(6)
            st.grip_target, st.grip_limit = aspire.FINGER_OPEN_M, None
            data.qpos[ids["finger_qadr"]] = aspire.FINGER_OPEN_M
            data.qpos[ids["finger2_qadr"]] = -aspire.FINGER_OPEN_M
        self._place_plug_in_tray()
        mujoco.mj_forward(model, data)
        for arm in self._arms:
            self._hold_offsets(arm)
        self._set_gaps(first=True)
        self._command_all()

    def _set_gaps(self, first: bool = False) -> None:
        """Hard mode v2: each servo's gears touching on the side its motor pushes to hold the arm up.

        Called where the arm is placed by writing simulator state. A servo whose load is under the
        breakaway torque keeps where it was in its lash, as it does on every substep, so a joint
        with almost no load does not jump on a rounding error. ``first`` (from ``reset``) puts every
        servo on a side. Nothing happens in other modes, so their runs stay byte for byte as they were."""
        if not self._lash_on:
            return
        data = self._mj_data
        self._mj.mj_forward(self._mj_model, data)
        for arm, st in self._arms.items():
            dadr = st.ids["dadr"]
            # the whole load a real motor carries: gravity included, which this model compensates outside the motors
            load = (data.qfrc_bias - data.qfrc_passive + data.qfrc_gravcomp)[dadr]
            half = 0.5 * self._lash[arm]
            gap = self._gap[arm]
            if first:
                gap[:] = np.where(load >= 0.0, half, -half)
            else:
                gap[load > self._lash_breakaway] = half[load > self._lash_breakaway]
                gap[load < -self._lash_breakaway] = -half[load < -self._lash_breakaway]

    def _apply_lash(self) -> None:
        """Hard mode v2, before every physics substep: a servo whose gears are not touching passes no torque.

        The servo's torque is the actuator's on the target the controller set, plus the arm's weight,
        which ASPIRE's model holds up from outside the motors and a real servo carries. When that torque
        pushes toward the side of the lash its motor is not on, the motor side crosses the lash at the
        servo's no-load speed, and until it gets there the actuator's target is set where it makes no
        torque. Past the breakaway torque only: below it friction holds the motor where it is.
        """
        data, dt = self._mj_data, self._mj_model.opt.timestep
        for arm, st in self._arms.items():
            ids = st.ids
            act, dadr, qadr = ids["act"], ids["dadr"], ids["qadr"]
            q, qd = data.qpos[qadr], data.qvel[dadr]
            target = self._ctrl_cmd[arm]
            torque = ids["kp"] * (target - q) - ids["kd"] * qd + data.qfrc_gravcomp[dadr]
            half = 0.5 * self._lash[arm]
            gap = self._gap[arm]
            push = np.where(torque > self._lash_breakaway, 1.0, np.where(torque < -self._lash_breakaway, -1.0, 0.0))
            goal = push * half
            moving = (push != 0.0) & (np.abs(gap - goal) > 1e-12)
            gap[moving] = np.clip(gap[moving] + push[moving] * self._lash_speed * dt, -half[moving], half[moving])
            touching = (push == 0.0) | (np.abs(gap - goal) <= 1e-12)
            free = q + ids["kd"] * qd / ids["kp"]               # the target at which the actuator makes no torque
            data.ctrl[act] = np.where(touching, target, np.clip(free, ids["lower"], ids["upper"]))

    def _hold_offsets(self, arm: str) -> None:
        """Set an arm's joint integral to what holds it still where it is: its gravity load over the servo gain.

        Without it the arm sags and recovers over the first second, and any reading taken later
        carries the time since reset. With it an arm is at rest from the start.
        """
        st, data = self._arms[arm], self._mj_data
        self._mj.mj_forward(self._mj_model, data)
        load = (data.qfrc_bias - data.qfrc_passive)[st.ids["dadr"]]
        st.offset = np.clip(load / st.ids["kp"], -INTEGRAL_CLAMP, INTEGRAL_CLAMP)

    def _place_plug_in_tray(self) -> None:
        cfg, lay, data = self._cfg, self._lay, self._mj_data
        tx = scene.TRAY_CENTER_XY[0] + cfg.tray_xy_yaw[0]
        ty = scene.TRAY_CENTER_XY[1] + cfg.tray_xy_yaw[1]
        yaw = cfg.tray_xy_yaw[2]
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy, dyaw = cfg.plug_in_tray
        z = scene.TABLE_TOP_Z + 2 * scene.TRAY_BLOCK_HALF[2] - scene.TRAY_POCKET_DEPTH + lay.geom.full_depth + 0.0005
        qa = lay.ids["plug_qadr"]
        data.qpos[qa:qa + 3] = (tx + c * dx - s * dy, ty + s * dx + c * dy, z)
        data.qpos[qa + 3:qa + 7] = (math.cos((yaw + dyaw) / 2), 0.0, 0.0, math.sin((yaw + dyaw) / 2))
        va = lay.ids["plug_dadr"]
        data.qvel[va:va + 6] = 0.0

    def nominal(self) -> NominalGeometry:
        """The station's taught geometry, with this episode's localization error in it."""
        cfg, lay = self._cfg, self._lay
        ex, ey, ez = cfg.localization_error
        return NominalGeometry(
            connector=self.connector, full_depth_m=lay.geom.full_depth, grasp_height_m=scene.GRASP_HEIGHT,
            board_xy=(scene.BOARD_CENTER_XY[0] + cfg.board_xy[0] + ex, scene.BOARD_CENTER_XY[1] + cfg.board_xy[1] + ey),
            board_yaw=cfg.board_yaw + cfg.yaw_error, face_z=scene.TABLE_TOP_Z + lay.geom.face_z + ez,
            ports_board=lay.ports_board, pitch_m=self._exp.station.board.pitch_mm * scene.MM,
            plug_body_m=lay.geom.body, plug_tongue_m=lay.geom.tongue, housing_half_m=lay.geom.housing_half,
            housing_height_m=lay.geom.housing_height, finger_travel_m=aspire.FINGER_OPEN_M, pad_thickness_m=0.001,
            holding_arm="right", free_arm="left",
        )

    # ------------------------------------------------------------------ the robot's own kinematics
    def _kin_at(self, arm: str, q: np.ndarray) -> None:
        kin, ids = self._kin, self._arms[arm].ids
        kin.qpos[:] = self._mj_model.qpos0
        kin.qpos[ids["qadr"]] = q
        self._mj.mj_kinematics(self._mj_model, kin)

    def _fk(self, arm: str, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._kin_at(arm, q)
        site = self._arms[arm].ids["site"]
        return self._kin.site_xpos[site].copy(), self._kin.site_xmat[site].reshape(3, 3).copy()

    def _jac(self, arm: str, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tool position, rotation and the 6 x 6 Jacobian (linear on top) at joint angles ``q``."""
        mujoco, kin, ids = self._mj, self._kin, self._arms[arm].ids
        self._kin_at(arm, q)
        mujoco.mj_comPos(self._mj_model, kin)
        mujoco.mj_jacSite(self._mj_model, kin, self._jacp, self._jacr, ids["site"])
        jac = np.vstack([self._jacp[:, ids["dadr"]], self._jacr[:, ids["dadr"]]])
        return kin.site_xpos[ids["site"]].copy(), kin.site_xmat[ids["site"]].reshape(3, 3).copy(), jac

    def _arm_inertia(self, arm: str, q: np.ndarray) -> np.ndarray:
        mujoco, kin, ids = self._mj, self._kin, self._arms[arm].ids
        self._kin_at(arm, q)
        mujoco.mj_comPos(self._mj_model, kin)
        mujoco.mj_crb(self._mj_model, kin)
        mujoco.mj_fullM(self._mj_model, kin, self._mass)   # MuJoCo 3.13: (model, data, dst)
        return self._mass[np.ix_(ids["dadr"], ids["dadr"])].copy()

    def _q(self, arm: str) -> np.ndarray:
        return self._mj_data.qpos[self._arms[arm].ids["qadr"]].copy()

    def _qd(self, arm: str) -> np.ndarray:
        return self._mj_data.qvel[self._arms[arm].ids["dadr"]].copy()

    def _motor_torque(self, arm: str) -> np.ndarray:
        return self._mj_data.actuator_force[self._arms[arm].ids["act"]].copy()

    def _q_read(self, arm: str) -> np.ndarray:
        """The joint angles the robot reads. Exact, except in hard mode v2.

        Hard mode v2: the servo's encoder reads the output shaft, so a reading is the joint's own
        angle on the encoder's count grid, at this joint's zero phase. The gear lash does not show
        here. It shows in how the joint moves (``_apply_lash``).
        """
        q = self._q(arm)
        if not self._sense_on:
            return q
        phase = self._enc_phase[arm]
        return (np.round(q / self._enc_step + phase) - phase) * self._enc_step

    def _motor_torque_read(self, arm: str) -> np.ndarray:
        """The joint torques the robot reads. Exact, except in hard mode v2.

        Hard mode v2: a torque is read as the servo's current, in counts of ``count_ma`` with
        ``noise_counts_sd`` of noise, 1 count being the same share of the joint's peak torque as
        6.5 mA is of the servo's stall current.
        """
        tau = self._motor_torque(arm)
        if not self._cur_on:
            return tau
        # The counts are of the servo's whole current, the arm's weight included. The robot then takes
        # its gravity model off, here exact, so what is left is the actuator's torque at the servo's resolution.
        weight = self._mj_data.qfrc_gravcomp[self._arms[arm].ids["dadr"]]
        unit = self._tau_peak[arm] * self._cur_count                        # Nm per count
        return np.round((tau + weight) / unit + self._cur_noise[arm]) * unit - weight

    def _update_readings(self) -> None:
        """Hard mode v2, once per control tick: the current sensing draws its noise."""
        if self._cur_on and self._cur_sigma > 0:
            for arm in self._arms:
                self._cur_noise[arm] = self._cur_rng.normal(0.0, self._cur_sigma, size=6)

    @staticmethod
    def _rot_err(r_des: np.ndarray, r: np.ndarray) -> np.ndarray:
        return 0.5 * (np.cross(r[:, 0], r_des[:, 0]) + np.cross(r[:, 1], r_des[:, 1]) + np.cross(r[:, 2], r_des[:, 2]))

    def _ik_step(self, arm: str, q: np.ndarray, p_des: np.ndarray, r_des: np.ndarray, iters: int = 2,
                 max_step: float = 0.2) -> np.ndarray:
        """Damped least squares from joints ``q`` toward a tool pose."""
        ids = self._arms[arm].ids
        q = q.copy()
        for _ in range(iters):
            p, r, jac = self._jac(arm, q)
            err = np.concatenate([p_des - p, self._rot_err(r_des, r)])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-6 * np.eye(6), err)
            q = np.clip(q + np.clip(dq, -max_step, max_step), ids["lower"], ids["upper"])
        return q

    def solve_ik(self, arm: str, pose: Pose, seed_q=None, iters: int = 200) -> tuple[np.ndarray, float, float]:
        """Joint angles that put the tool at ``pose``. Returns (q, position error m, rotation error rad).

        Uses mink when it imports, otherwise damped least squares.
        """
        st = self._arms[arm]
        q0 = st.q_plan.copy() if seed_q is None else np.asarray(seed_q, float).copy()
        p_des, r_des = pose.position(), pose.rotation()
        try:
            import mink
        except Exception:
            mink = None
        if mink is not None:
            model = self._mj_model
            full = model.qpos0.copy()
            full[st.ids["qadr"]] = q0
            conf = mink.Configuration(model)
            conf.update(q=full)
            task = mink.FrameTask(scene.TOOL_SITE[arm], "site", position_cost=1.0, orientation_cost=1.0, lm_damping=1e-4)
            task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(r_des), p_des))
            posture = mink.PostureTask(model, cost=1e-4)
            posture.set_target(full)
            own = set(st.ids["dadr"].tolist())
            freeze = mink.DofFreezingTask(model, dof_indices=[d for d in range(model.nv) if d not in own])
            limits = [mink.ConfigurationLimit(model)]
            for _ in range(iters):
                vel = mink.solve_ik(conf, [task, posture], 0.02, "daqp", damping=1e-8, limits=limits, constraints=[freeze])
                conf.integrate_inplace(vel, 0.02)
                if np.linalg.norm(vel) < 1e-5:
                    break
            q = conf.q[st.ids["qadr"]].copy()
        else:
            q = q0
            for _ in range(iters):
                q = self._ik_step(arm, q, p_des, r_des, iters=1)
        p, r = self._fk(arm, q)
        return q, float(np.linalg.norm(p_des - p)), float(np.linalg.norm(self._rot_err(r_des, r)))

    # ------------------------------------------------------------------ the control tick
    def _command(self, arm: str, qdd_plan: np.ndarray | None = None) -> None:
        st, ids = self._arms[arm], self._arms[arm].ids
        kp, kd = ids["kp"], ids["kd"]
        torque = kd * st.qd_plan + st.tau_ff
        if qdd_plan is not None:
            torque = torque + self._arm_inertia(arm, st.q_plan) @ qdd_plan
        target = st.q_plan + torque / kp + st.offset
        self._mj_data.ctrl[ids["act"]] = np.clip(target, ids["lower"], ids["upper"])
        self._ctrl_cmd[arm] = self._mj_data.ctrl[ids["act"]].copy()   # hard mode v2's lash reads the target from here
        grip = st.grip_target
        if st.grip_limit is not None:                       # limit the squeeze: do not lead the finger by more than F / kp
            lead = st.grip_limit / scene.FINGER_KP
            grip = max(grip, float(self._mj_data.qpos[ids["finger_qadr"]]) - lead)
        self._mj_data.ctrl[ids["grip_act"]] = min(max(grip, 0.0), aspire.FINGER_OPEN_M)

    def _command_all(self) -> None:
        for arm in self._arms:
            self._command(arm)

    def _integrate_hold(self, arm: str) -> None:
        st = self._arms[arm]
        err = st.q_plan - self._q(arm)
        err = np.where(np.abs(err) < INTEGRAL_DEADBAND, 0.0, err)
        st.offset = np.clip(st.offset + INTEGRAL_GAIN * CONTROL_DT * err, -INTEGRAL_CLAMP, INTEGRAL_CLAMP)

    def _tick(self) -> None:
        mujoco, model, data = self._mj, self._mj_model, self._mj_data
        for _ in range(self._substeps):
            self._retention.update(model, data)
            if self._lash_on:
                self._apply_lash()
            mujoco.mj_step(model, data)
        if self._cur_on:
            self._update_readings()
        if self._noise_on:
            rho = math.exp(-CONTROL_DT / NOISE_CORR_S)
            scale = self._sigma * math.sqrt(1.0 - rho * rho)
            for arm in self._arms:
                self._noise[arm] = rho * self._noise[arm] + scale * self._rng.normal(0.0, 1.0, size=3)
                self._bias[arm] = self._bias[arm] + self._drift_dir[arm] * CONTROL_DT
                if self._walk > 0:                          # hard mode: the bias also walks
                    self._bias[arm] = self._bias[arm] + self._walk * math.sqrt(CONTROL_DT) * self._rng.normal(0.0, 1.0, size=3)
        if self._hard_on:
            self._record_cameras()
        raw = self._true_grip_force_z()
        a = TRUE_FORCE_FILTER_S / (TRUE_FORCE_FILTER_S + CONTROL_DT)
        self._true_fz = a * self._true_fz + (1 - a) * raw
        self._true_peak_push = max(self._true_peak_push, -self._true_fz)
        self._true_peak_pull = max(self._true_peak_pull, self._true_fz)
        self._true_peak_raw = max(self._true_peak_raw, abs(raw))
        ret = self._retention
        if ret.captured and ret.port == self._cfg.target_port and ret.shortfall_mm <= LINK_SHORTFALL_MM:
            if self._link_since is None:
                self._link_since = float(data.time)
        else:
            self._link_since = None
        if self._tick_hook is not None:
            self._tick_hook()

    def _reset_true_peaks(self) -> None:
        self._true_peak_push = self._true_peak_pull = self._true_peak_raw = 0.0
        self._true_overrun_mm = -math.inf

    def _hold(self, seconds: float, integrate: bool = True) -> None:
        for _ in range(max(1, int(round(seconds / CONTROL_DT)))):
            for arm, st in self._arms.items():
                st.qd_plan = np.zeros(6)
                if integrate and not np.any(st.force_cmd):
                    self._integrate_hold(arm)
            self._command_all()
            self._tick()

    # ------------------------------------------------------------------ force estimate
    def _tool_force(self, arm: str, jac: np.ndarray | None = None, with_noise: bool = True) -> np.ndarray:
        """The force the tool applies to the world, base frame, from the commanded motor torques."""
        st = self._arms[arm]
        if jac is None:
            _, _, jac = self._jac(arm, self._q_read(arm))
        # A moving joint loses its friction torque on the way to the tool, so that is taken off.
        # A stalled joint keeps an unknown share of it, so nothing is taken off and the estimate reads high.
        tau = self._motor_torque_read(arm) - st.tau_tare - JOINT_FRICTION * np.tanh(self._qd(arm) / FRICTION_SPEED)
        wrench = np.linalg.solve(jac.T + 1e-9 * np.eye(6), tau)
        force = wrench[:3]
        if with_noise and self._noise_on:
            if self._hyst > 0:
                # Hard mode: gearbox friction. The motor current that drives a rising load also overcomes
                # friction, so the estimate reads high by up to hysteresis_n while the load rises and low
                # while it falls, with a play of 2 x hysteresis_n between the 2 when the load turns.
                play = play_operator(force, self._play[arm], self._hyst)
                self._play[arm] = play
                self._hyst_off[arm] = force - play
                force = force + self._hyst_off[arm]
            force = force + self._noise[arm] + self._bias[arm] - self._tare_noise[arm]
        return force

    def zero_force_estimate(self, arm: str, seconds: float = 0.1) -> None:
        """Tare the force estimate. Call it with the arm still and touching nothing.

        Easy mode tares exactly. Hard mode averages the noisy estimate over the same 0.1 s, so the
        tare keeps that average of the noise as an offset, and the play starts again from 0.
        """
        st = self._arms[arm]
        total, n = np.zeros(6), 0
        noise_sum = np.zeros(3)
        for _ in range(max(1, int(round(seconds / CONTROL_DT)))):
            self._hold(CONTROL_DT)
            total += self._motor_torque_read(arm)
            noise_sum += self._noise[arm]
            n += 1
        st.tau_tare = total / n
        if self._noise_on:                                  # a tare removes the bias seen at that moment
            self._bias[arm] = np.zeros(3)
        if self._tare_noisy:
            self._tare_noise[arm] = noise_sum / n
            self._play[arm] = np.zeros(3)
            self._hyst_off[arm] = np.zeros(3)

    # ------------------------------------------------------------------ Station protocol
    def observe(self, cameras: list[str] | None = None) -> Observation:
        mujoco, model, data = self._mj, self._mj_model, self._mj_data
        names = list(cameras) if cameras is not None else list(self._exp.station.cameras)
        unknown = [n for n in names if n not in aspire.CAMERAS]
        if unknown:
            raise KeyError(f"unknown camera {unknown}. Known: {', '.join(aspire.CAMERAS)}")
        if self._renderer is None:
            # The GPU's multisample resolve differs by 1 gray level in a few pixels in about 1 frame
            # in 8. Rendering at twice the size with multisampling off and averaging each 2 x 2 block
            # here is the same antialiasing, done in integer arithmetic, so a frame is exact.
            model.vis.quality.offsamples = 0
            if self._hard_on:                               # 1 renderer for every view. See _render_view
                plan = next(iter(self._webcams.values())).plan
                self._renderer = mujoco.Renderer(model, max(SUPERSAMPLE * self._height, plan["height"]),
                                                 max(SUPERSAMPLE * self._width, plan["width"]))
            else:
                self._renderer = mujoco.Renderer(model, SUPERSAMPLE * self._height, SUPERSAMPLE * self._width)
        mujoco.mj_kinematics(model, data)                  # poses only. The solver state is left alone
        mujoco.mj_camlight(model, data)
        frames = {}
        for name in names:
            if name in self._webcams:                       # hard mode's wrist cameras
                frames[name] = self._webcam_frame(name)
                continue
            # A frame lags a camera switch: the first render after switching cameras differs from
            # the next one by up to 7 gray levels over the whole image. Rendering twice and keeping
            # the second makes a frame a function of the camera and the state alone, byte for byte.
            if self._hard_on:
                rgb = _downsample(self._render_view(name, SUPERSAMPLE * self._width, SUPERSAMPLE * self._height), SUPERSAMPLE)
            else:
                self._renderer.update_scene(data, camera=name)
                self._renderer.render()
                self._renderer.update_scene(data, camera=name)
                rgb = _downsample(self._renderer.render(), SUPERSAMPLE)
            frames[name] = CameraFrame(name=name, rgb=rgb, intrinsics=self._intrinsics(name),
                                       extrinsics=self._nominal_extrinsics(name), ts=float(data.time))
        return Observation(frames=frames, ts=float(data.time))

    def _render_view(self, name: str, width: int, height: int) -> np.ndarray:
        """Hard mode: render 1 camera at a size within the renderer's buffer, twice, and keep the second.

        mujoco's Renderer draws into a rectangle of its own size. Hard mode needs 2 sizes (the
        overhead camera's and the webcams' wide render), and a second renderer would cost a second
        GL context, about 2 GB of commit on this machine, so the rectangle is set per view.
        """
        r = self._renderer
        r._width, r._height = int(width), int(height)       # mujoco 3.13 Renderer: render() reads these 3
        r._rect = self._mj.MjrRect(0, 0, int(width), int(height))
        data = self._mj_data
        r.update_scene(data, camera=name)
        r.render()
        r.update_scene(data, camera=name)
        return r.render()

    def _webcam_frame(self, name: str) -> CameraFrame:
        """Hard mode: 1 wrist frame through the webcam model, with the calibration's intrinsics."""
        from .webcam import Motion

        cam = self._webcams[name]
        plan = cam.plan
        pinhole = self._render_view(name, plan["width"], plan["height"])
        data, model = self._mj_data, self._mj_model
        cid = self._lay.ids[f"cam:{name}"]
        port, _ = scene.port_pose_world(self._cfg, self._lay, self._cfg.target_port)
        pos = data.cam_xpos[cid].copy()
        r_cv = data.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
        now_point = r_cv.T @ (port - pos)
        times, points = [], []
        t_now = float(data.time)
        for t, poses in self._cam_hist:
            if name in poses:
                p, r = poses[name]
                times.append(t - t_now)
                points.append(r.T @ (port - p))
        times.append(0.0)
        points.append(now_point)
        motion = Motion(times=np.array(times), points_cam=np.array(points)) if len(times) >= 2 else None
        index = sorted(self._webcams).index(name)
        rng = np.random.default_rng([self.seed, sorted(self._exp.connectors).index(self.connector), index,
                                     self._frame_count, 5150])
        self._frame_count += 1
        keep = {} if self._keep_stages is not None else None
        rgb, k, info = cam.process(pinhole, rng, distance_m=float(max(now_point[2], 1e-3)), motion=motion, keep=keep)
        if keep is not None:
            self._keep_stages[name] = {**keep, "pinhole": pinhole}
        record = info.as_dict()
        record["port_raw_px_per_mm"] = cam.raw_px_per_mm(now_point)
        record["port_distance_m"] = float(np.linalg.norm(now_point))
        self._frame_info[name] = record
        return CameraFrame(name=name, rgb=rgb, intrinsics=k, extrinsics=self._nominal_extrinsics(name), ts=t_now)

    def _record_cameras(self) -> None:
        """Hard mode: keep each wrist camera's pose (OpenCV axes) for the rolling shutter and the motion blur."""
        data = self._mj_data
        flip = np.diag([1.0, -1.0, -1.0])
        poses = {}
        for name in self._webcams:
            cid = self._lay.ids[f"cam:{name}"]
            poses[name] = (data.cam_xpos[cid].copy(), data.cam_xmat[cid].reshape(3, 3) @ flip)
        self._cam_hist.append((float(data.time), poses))

    def _intrinsics(self, name: str) -> np.ndarray:
        model, cam = self._mj_model, self._lay.ids[f"cam:{name}"]
        fovy = math.radians(float(model.cam_fovy[cam]))
        f = 0.5 * self._height / math.tan(fovy / 2)
        return np.array([[f, 0.0, self._width / 2], [0.0, f, self._height / 2], [0.0, 0.0, 1.0]])

    def _nominal_extrinsics(self, name: str) -> np.ndarray:
        """Camera to base from the calibration: the model's camera mount without this seed's jitter."""
        mujoco, model, kin, lay = self._mj, self._mj_model, self._kin, self._lay
        cam = lay.ids[f"cam:{name}"]
        kin.qpos[:] = model.qpos0
        for arm, st in self._arms.items():
            kin.qpos[st.ids["qadr"]] = self._q_read(arm)   # the robot knows its joints through its readings
        mujoco.mj_kinematics(model, kin)
        body = model.cam_bodyid[cam]
        r_body = kin.xmat[body].reshape(3, 3)
        pos = kin.xpos[body] + r_body @ lay.base["cam_pos"][cam]
        r_cam = r_body @ Pose((0.0, 0.0, 0.0), tuple(lay.base["cam_quat"][cam])).rotation()
        out = np.eye(4)
        out[:3, :3] = r_cam @ np.diag([1.0, -1.0, -1.0])   # MuJoCo camera axes to OpenCV axes
        out[:3, 3] = pos
        return out

    def robot_state(self) -> RobotState:
        data = self._mj_data
        arms = {}
        for arm, st in self._arms.items():
            ids = st.ids
            q, qd = self._q_read(arm), self._qd(arm)
            pos, rot, jac = self._jac(arm, q)
            force = self._tool_force(arm, jac)
            tau = self._motor_torque_read(arm)
            if self._noise_on:
                tau = tau + jac[:3].T @ (self._noise[arm] + self._bias[arm] - self._tare_noise[arm] + self._hyst_off[arm])
            grip = float(data.qpos[ids["finger_qadr"]] / aspire.FINGER_OPEN_M)
            grip_force = float(-data.actuator_force[ids["grip_act"]])
            if self._noise_on:
                if self._grip_sigma > 0:
                    grip += float(self._read_rng.normal(0.0, self._grip_sigma))
                if self._grip_force_sigma > 0:
                    grip_force += float(self._read_rng.normal(0.0, self._grip_force_sigma))
                if self._qd_sigma > 0:
                    qd = qd + self._read_rng.normal(0.0, self._qd_sigma, size=6)
            if self._grip_res > 0:                         # the gripper encoder's resolution
                grip = round(grip / self._grip_res) * self._grip_res
            arms[arm] = ArmState(
                joint_pos=q, joint_vel=qd, joint_torque_est=tau,
                gripper_pos=float(np.clip(grip, 0.0, 1.0)), gripper_torque=grip_force,
                tool_pose=Pose.from_matrix(pos, rot), tool_force_est=force,
            )
        return RobotState(arms=arms, ts=float(data.time))

    def stop(self) -> None:
        self._stop_requested = True
        for st in self._arms.values():
            st.qd_plan = np.zeros(6)
        self._command_all()

    def set_gripper(self, arm: str, position: float, max_torque: float | None = None) -> MotionResult:
        st = self._arms[arm]
        st.grip_target = float(np.clip(position, 0.0, 1.0)) * aspire.FINGER_OPEN_M
        st.grip_limit = None if max_torque is None else float(max_torque)
        data, ids = self._mj_data, st.ids
        t0, still = data.time, 0
        while data.time - t0 < 1.2:
            self._hold(CONTROL_DT)
            if abs(float(data.qvel[ids["finger_dadr"]])) < 2e-4 and data.time - t0 > 0.15:
                still += 1
                if still >= 5:
                    break
            else:
                still = 0
        reached = abs(float(data.qpos[ids["finger_qadr"]]) - st.grip_target) < 5e-4
        return MotionResult(ok=True, stopped_by="target" if reached else "force", duration_s=float(data.time - t0),
                            force_at_stop_n=float(-data.actuator_force[ids["grip_act"]]))

    def move_to_pose(self, arm: str, pose: Pose, speed: float, max_force: float | None = None) -> MotionResult:
        st = self._arms[arm]
        self._relax(arm)
        self._break_away(arm)
        p0, r0 = self._fk(arm, st.q_plan)
        p0_report = self._fk(arm, self._q_read(arm))[0] if self._sense_on else p0   # hard mode v2 reports from readings
        p1, r1 = pose.position(), pose.rotation()
        dist = float(np.linalg.norm(p1 - p0))
        rot_vec = _log_rot(r1 @ r0.T)
        angle = float(np.linalg.norm(rot_vec))
        data = self._mj_data
        t0 = data.time
        # A short move follows a straight line in space. A long one goes to an IK solution in joint space.
        q_start, q_goal = st.q_plan.copy(), None
        if dist > LONG_MOVE_M or angle > LONG_MOVE_RAD:
            q_goal, pos_err, rot_err = self.solve_ik(arm, pose, seed_q=st.q_plan)
            if pos_err > 1e-3 or rot_err > 1e-2:
                return MotionResult(ok=False, stopped_by="error", note=f"pose not reachable: IK left {pos_err * 1000:.1f} mm")
            seconds = max(dist / max(speed, 1e-4), float(np.abs(q_goal - q_start).max()) / MAX_JOINT_SPEED, 0.3)
        else:
            seconds = max(dist / max(speed, 1e-4), angle / 1.0, 0.25)
        ticks = max(2, int(round(seconds / CONTROL_DT)))
        peak, stopped = 0.0, "target"
        qd_prev = np.zeros(6)
        self._stop_requested = False
        for k in range(1, ticks + 1):
            u = k / ticks
            s = 10 * u**3 - 15 * u**4 + 6 * u**5           # minimum jerk
            if q_goal is not None:
                q_new = q_start + s * (q_goal - q_start)
            else:
                q_new = self._ik_step(arm, st.q_plan, p0 + s * (p1 - p0), _exp_rot(rot_vec * s) @ r0)
            st.qd_plan = (q_new - st.q_plan) / CONTROL_DT
            qdd = (st.qd_plan - qd_prev) / CONTROL_DT
            qd_prev = st.qd_plan
            st.q_plan = q_new
            self._integrate_hold(arm)
            for other, ost in self._arms.items():
                if other != arm:
                    ost.qd_plan = np.zeros(6)
                    self._integrate_hold(other)
            self._command_all()
            self._command(arm, qdd_plan=qdd)
            self._tick()
            if max_force is not None:
                f = float(np.linalg.norm(self._tool_force(arm)))
                peak = max(peak, f)
                if f > max_force:
                    stopped = "force"
                    break
            if self._stop_requested:
                stopped = "stopped"
                break
        st.qd_plan = np.zeros(6)
        if stopped == "target":
            self._hold(SETTLE_S)
        else:
            st.q_plan = self._q(arm)
            self._hold(0.1)
        p_end, _ = self._fk(arm, self._q_read(arm))        # reported: from the joint readings
        return MotionResult(ok=stopped == "target", stopped_by=stopped, travel_m=float(np.linalg.norm(p_end - p0_report)),
                            duration_s=float(data.time - t0), peak_force_n=peak if max_force is not None else None,
                            note=f"end error {np.linalg.norm(p_end - p1) * 1000:.2f} mm")

    def move_relative(self, arm: str, delta_m, frame: str, speed: float, max_force: float, max_travel: float) -> MotionResult:
        st = self._arms[arm]
        delta = np.asarray(delta_m, dtype=float)
        length = float(np.linalg.norm(delta))
        if length < 1e-9:
            raise ValueError("delta_m has zero length. Give the direction and distance of the guarded move")
        if max_force <= 0 or max_travel <= 0 or speed <= 0:
            raise ValueError("speed, max_force and max_travel must all be positive")
        self._relax(arm)
        self._break_away(arm)
        q0 = self._q(arm)
        p0, r0, jac = self._jac(arm, q0)
        if frame == "tool":
            axis = r0 @ (delta / length)
        elif frame == "base":
            axis = delta / length
        else:
            raise ValueError(f"frame must be 'base' or 'tool', got {frame!r}")
        limit = min(length, float(max_travel))
        data = self._mj_data
        st.q_plan = q0.copy()                              # the reference rides on the measured tool
        st.qd_plan = np.zeros(6)
        f, e_prev, s_prev, v_filt, at_cap = 0.0, 0.0, 0.0, 0.0, 0.0
        peak_s = 0.0                                       # evaluator-side: the farthest the tool went along the axis
        # What the move reports comes from the joint readings. They are the joint angles, except in
        # hard mode v2, where the speed loop and the travel stop keep the angles and the report does not.
        sensed = self._sense_on
        p0_read, _, jac_read = self._jac(arm, self._q_read(arm)) if sensed else (p0, r0, jac)
        s_read = 0.0
        force_filt = float(axis @ self._tool_force(arm, jac_read))
        trace_t, trace_s, trace_f = [], [], []
        stopped, peak = "timeout", force_filt
        timeout = limit / speed * 4.0 + max_force / FORCE_RATE * 3.0 + 2.0
        a_speed = SPEED_FILTER_S / (SPEED_FILTER_S + CONTROL_DT)
        a_force = FORCE_FILTER_S / (FORCE_FILTER_S + CONTROL_DT)
        t0 = data.time
        self._stop_requested = False
        while data.time - t0 < timeout:
            # speed regulator: more force when the tool is slow, less when it runs ahead. Never above the cap
            e = float(np.clip((speed - v_filt) / speed, -4.0, 1.0))
            f += FORCE_RATE * CONTROL_DT * max(e, -1.0) + FORCE_KP * (e - e_prev)
            e_prev = e
            f = float(np.clip(f, -max_force, max_force))
            st.force_cmd = f * axis
            # reference: on the line, at the measured progress, at the starting orientation
            st.q_plan = self._ik_step(arm, st.q_plan, p0 + s_prev * axis, r0, iters=1, max_step=0.05)
            qd_des = np.linalg.solve(jac + 1e-9 * np.eye(6), np.concatenate([axis * speed, np.zeros(3)]))
            st.tau_ff = jac[:3].T @ st.force_cmd + JOINT_FRICTION * np.tanh(qd_des / 0.01)
            for other, ost in self._arms.items():
                if other != arm:
                    ost.qd_plan = np.zeros(6)
                    self._integrate_hold(other)
            self._command_all()
            self._tick()
            # measure
            p, _, jac = self._jac(arm, self._q(arm))
            s = float(axis @ (p - p0))
            v_filt = a_speed * v_filt + (1 - a_speed) * (s - s_prev) / CONTROL_DT
            s_prev = s
            if sensed:
                p_read, _, jac_read = self._jac(arm, self._q_read(arm))
                s_read = float(axis @ (p_read - p0_read))
            else:
                jac_read, s_read = jac, s
            force_filt = a_force * force_filt + (1 - a_force) * float(axis @ self._tool_force(arm, jac_read))
            peak = max(peak, force_filt)
            trace_t.append(float(data.time - t0))
            trace_s.append(s_read * 1000.0)
            trace_f.append(force_filt)
            peak_s = max(peak_s, s)
            if s >= limit:
                stopped = "travel"
                break
            if force_filt >= max_force:
                stopped = "force"
                break
            if f >= max_force - 1e-9 and abs(v_filt) < 0.2 * speed:
                at_cap += CONTROL_DT
                if at_cap >= STALL_S:
                    stopped = "force"
                    break
            else:
                at_cap = 0.0
            if self._stop_requested:
                stopped = "stopped"
                break
        travel, travel_true = s_read, s_prev               # reported, and the joint angles' own (evaluator-side)
        if stopped == "travel":                            # nothing holds the tool back: brake at once
            st.force_cmd, st.tau_ff = np.zeros(3), np.zeros(6)
            st.q_plan = self._q(arm)
            self._hold(0.04, integrate=False)
            p_end, _ = self._fk(arm, self._q(arm))
            travel_true = float(axis @ (p_end - p0))
            travel = float(axis @ (self._fk(arm, self._q_read(arm))[0] - p0_read)) if sensed else travel_true
        relax_peak = self._relax(arm, track=(p0, axis))    # stop pressing and hold where the tool is
        peak_s = max(peak_s, relax_peak, float(travel_true))
        self._true_overrun_mm = max(self._true_overrun_mm, (peak_s - limit) * 1000.0)
        if self._hard_on:                                  # hard mode: where the tool ended once the force was gone.
            p_end, _ = self._fk(arm, self._q_read(arm))    # A wedge can let go during the relax and carry the tool on
            travel = max(float(travel), float(axis @ (p_end - p0_read)))
        return MotionResult(
            ok=stopped in ("travel", "force"), stopped_by=stopped, travel_m=float(travel),
            duration_s=float(data.time - t0), force_at_stop_n=force_filt, peak_force_n=peak,
            trace={"t_s": trace_t, "travel_mm": trace_s, "force_n": trace_f},
        )

    def _relax(self, arm: str, seconds: float = 0.12, track: tuple | None = None) -> float:
        """Ramp any commanded tool force to zero and hold where the tool is.

        ``track`` is (start position, axis) of a guarded move. The return value is then the farthest
        the tool went along that axis during the relax, in meters (evaluator-side). Else 0.
        """
        st = self._arms[arm]
        peak = 0.0
        if not np.any(st.force_cmd) and not np.any(st.tau_ff):
            return peak
        ticks = max(1, int(round(seconds / CONTROL_DT)))
        f0, tau0 = st.force_cmd.copy(), st.tau_ff.copy()
        for k in range(1, ticks + 1):
            w = 1.0 - k / ticks
            st.force_cmd, st.tau_ff = f0 * w, tau0 * w
            st.q_plan = self._q(arm)
            st.qd_plan = np.zeros(6)
            for other, ost in self._arms.items():
                if other != arm:
                    ost.qd_plan = np.zeros(6)
                    self._integrate_hold(other)
            self._command_all()
            self._tick()
            if track is not None:
                p, _ = self._fk(arm, self._q(arm))
                peak = max(peak, float(track[1] @ (p - track[0])))
        st.force_cmd, st.tau_ff = np.zeros(3), np.zeros(6)
        st.q_plan = self._q(arm)
        return peak

    def wait(self, seconds: float) -> None:
        """Hold still for a while."""
        self._hold(seconds)

    def read_signal(self, name: str) -> dict:
        """A signal the station itself reports, the way a switch reports a link. Instrumented stations only.

        ``link``: up once the plug has sat at full depth in the target port for 0.2 s. It says the
        contacts mate. It says nothing about a latch.
        """
        if name != "link":
            raise KeyError(f"this station has no signal named {name!r}. Known: link")
        ret, now = self._retention, float(self._mj_data.time)
        mated = bool(ret.captured and ret.port == self._cfg.target_port and ret.shortfall_mm <= LINK_SHORTFALL_MM)
        if not mated:
            self._link_since = None
        elif self._link_since is None:
            self._link_since = now
        up = self._link_since is not None and now - self._link_since >= LINK_DELAY_S
        return {"link_up": bool(up), "ts": now}

    # ------------------------------------------------------------------ privileged helpers (evaluator side)
    def _true_grip_force_z(self) -> float:
        """Net vertical force the robot puts on the plug right now, from simulator contacts. Privileged."""
        mujoco, model, data = self._mj, self._mj_model, self._mj_data
        plug = self._lay.ids["body:plug"]
        total = 0.0
        buf = self._contact_buf
        for i in range(data.ncon):
            con = data.contact[i]
            b1, b2 = model.geom_bodyid[con.geom1], model.geom_bodyid[con.geom2]
            if b1 == plug:
                other, sign = b2, -1.0
            elif b2 == plug:
                other, sign = b1, 1.0
            else:
                continue
            if model.body_rootid[other] not in self._arm_roots:
                continue
            mujoco.mj_contactForce(model, data, i, buf)
            frame = con.frame
            # the contact frame's rows are its axes. z of the force in the base frame:
            total += sign * (frame[2] * buf[0] + frame[5] * buf[1] + frame[8] * buf[2])
        return float(total)

    def _set_stiction(self, arm: str, extra_nm) -> None:
        """Add friction loss to an arm's joints. Privileged: this is how a stuck joint is injected.

        The joints stay stuck while the arm is still and break away when the arm is next told to move.
        """
        ids = self._arms[arm].ids
        base = self._lay.base["dof_frictionloss"][ids["dadr"]]
        self._mj_model.dof_frictionloss[ids["dadr"]] = base + np.asarray(extra_nm, dtype=float)
        self._stuck[arm] = bool(np.any(np.asarray(extra_nm) > 0))
        # MuJoCo's friction loss creeps under load at its default impedance. A stuck joint does not.
        self._mj_model.dof_solimp[ids["dadr"]] = STUCK_SOLIMP if self._stuck[arm] else self._lay.base["dof_solimp"][ids["dadr"]]

    def _break_away(self, arm: str) -> None:
        if self._stuck.get(arm):
            self._set_stiction(arm, np.zeros(6))

    def _snapshot(self) -> dict:
        """Everything needed to come back to this exact moment, derived quantities included. Privileged."""
        model = self._mj_model
        ret = self._retention
        return {
            "data": copy.copy(self._mj_data),              # a full MjData copy, so a replay matches byte for byte
            "model": {name: getattr(model, name).copy() for name in ("dof_frictionloss", "dof_damping", "dof_solimp")},
            "retention": {name: getattr(ret, name) for name in RETENTION_STATE},
            "arms": {a: (s.q_plan.copy(), s.qd_plan.copy(), s.offset.copy(), s.tau_ff.copy(), s.force_cmd.copy(),
                         s.tau_tare.copy(), s.grip_target, s.grip_limit) for a, s in self._arms.items()},
            "rng": self._rng.bit_generator.state,
            "read_rng": self._read_rng.bit_generator.state,
            "noise": {a: v.copy() for a, v in self._noise.items()},
            "bias": {a: v.copy() for a, v in self._bias.items()},
            "true": (self._true_fz, self._true_peak_push, self._true_peak_pull, self._true_peak_raw, self._true_overrun_mm),
            "stuck": dict(self._stuck),
            "link_since": self._link_since,
            "hard": {
                "play": {a: v.copy() for a, v in self._play.items()},
                "hyst_off": {a: v.copy() for a, v in self._hyst_off.items()},
                "tare_noise": {a: v.copy() for a, v in self._tare_noise.items()},
                "cam_hist": [(t, {n: (p.copy(), r.copy()) for n, (p, r) in poses.items()}) for t, poses in self._cam_hist],
                "frame_count": self._frame_count,
            },
            "readings": {                                  # hard mode v2
                "gap": {a: v.copy() for a, v in self._gap.items()},
                "ctrl_cmd": {a: v.copy() for a, v in self._ctrl_cmd.items()},
                "cur_noise": {a: v.copy() for a, v in self._cur_noise.items()},
                "cur_rng": self._cur_rng.bit_generator.state,
            },
        }

    def _restore(self, snap: dict) -> None:
        mujoco, model = self._mj, self._mj_model
        mujoco.mj_copyData(self._mj_data, model, snap["data"])
        for name, value in snap["model"].items():
            getattr(model, name)[:] = value
        for name, value in snap["retention"].items():
            setattr(self._retention, name, value)
        for a, vals in snap["arms"].items():
            s = self._arms[a]
            s.q_plan, s.qd_plan, s.offset, s.tau_ff, s.force_cmd, s.tau_tare = (v.copy() for v in vals[:6])
            s.grip_target, s.grip_limit = vals[6], vals[7]
        self._rng.bit_generator.state = snap["rng"]
        self._read_rng.bit_generator.state = snap["read_rng"]
        self._noise = {a: v.copy() for a, v in snap["noise"].items()}
        self._bias = {a: v.copy() for a, v in snap["bias"].items()}
        self._true_fz, self._true_peak_push, self._true_peak_pull, self._true_peak_raw, self._true_overrun_mm = snap["true"]
        self._stuck = dict(snap["stuck"])
        self._link_since = snap["link_since"]
        self._stop_requested = False
        self._tick_hook = None
        hard = snap["hard"]
        self._play = {a: v.copy() for a, v in hard["play"].items()}
        self._hyst_off = {a: v.copy() for a, v in hard["hyst_off"].items()}
        self._tare_noise = {a: v.copy() for a, v in hard["tare_noise"].items()}
        self._cam_hist.clear()
        for t, poses in hard["cam_hist"]:
            self._cam_hist.append((t, {n: (p.copy(), r.copy()) for n, (p, r) in poses.items()}))
        self._frame_count = hard["frame_count"]
        readings = snap["readings"]
        self._gap = {a: v.copy() for a, v in readings["gap"].items()}
        self._ctrl_cmd = {a: v.copy() for a, v in readings["ctrl_cmd"].items()}
        self._cur_noise = {a: v.copy() for a, v in readings["cur_noise"].items()}
        self._cur_rng.bit_generator.state = readings["cur_rng"]

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._retention.release(self._mj_model, self._mj_data)


def play_operator(x: np.ndarray, state: np.ndarray, half_width: float) -> np.ndarray:
    """A play (backlash) operator per axis: the output follows the input only once the input
    leaves the band of +-half_width around it.

    The station updates it each time the force estimate is read: on every control tick of a
    guarded move and at every ``robot_state``. So during insertions and probes it follows the
    trajectory. Between reads, for example during a free move without a force limit, a reversal
    of the force can go unseen."""
    return np.where(x - state > half_width, x - half_width, np.where(x - state < -half_width, x + half_width, state))


def _downsample(rgb: np.ndarray, k: int) -> np.ndarray:
    """Mean of each k x k block, rounded half up, in integer arithmetic."""
    if k == 1:
        return rgb.copy()
    h, w = rgb.shape[0] // k, rgb.shape[1] // k
    blocks = rgb[: h * k, : w * k].astype(np.uint32).reshape(h, k, w, k, 3).sum(axis=(1, 3))
    return ((blocks + (k * k) // 2) // (k * k)).astype(np.uint8)


def _log_rot(r: np.ndarray) -> np.ndarray:
    """Rotation vector of a rotation matrix."""
    cos = float(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cos)
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / (2.0 * math.sin(angle))
    return axis * angle


def _exp_rot(rv: np.ndarray) -> np.ndarray:
    """Rotation matrix of a rotation vector."""
    angle = float(np.linalg.norm(rv))
    if angle < 1e-12:
        return np.eye(3)
    k = rv / angle
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(angle) * kx + (1.0 - math.cos(angle)) * (kx @ kx)
