"""The oracle plug cycle and the do-nothing program. Privileged.

``oracle_cycle`` picks the plug from the tray, inserts it into the target port and lets go,
reading the plug's true pose to do it. It is the reference a skill is measured against, and
the cycle the stand-in contact policy will imitate in S5. ``do_nothing`` leaves the plug in
the tray, which is the baseline every check must beat.
"""

from __future__ import annotations

import numpy as np

from ...config import load_experiment
from ..base import Pose
from . import scene
from .inject import ARM, CLEARANCE_M, EXTRA_TRAVEL_M, INSERT_SPEED, PLUG_TO_TOOL, RELEASE_LIFT_M, _insertion_flag, _rot_z
from .station import TOP_DOWN_SEED, MjPlugStation
from .truth import truth

APPROACH_M = 0.045           # hover height above the grasp point
LIFT_M = 0.045
FREE_SPEED = 0.25            # m/s between stations
NEAR_SPEED = 0.04            # m/s for the last centimeters


def _plug_pose(station: MjPlugStation) -> tuple[np.ndarray, np.ndarray]:
    """True shoulder position and rotation of the plug. Privileged."""
    mujoco, model, data, ids = station._mj, station._mj_model, station._mj_data, station._lay.ids
    mujoco.mj_kinematics(model, data)
    site = ids["site:plug_shoulder"]
    return data.site_xpos[site].copy(), data.site_xmat[site].reshape(3, 3).copy()


def _tool_for_plug(shoulder: np.ndarray, plug_rot: np.ndarray, rot_g: np.ndarray, off_g: np.ndarray) -> Pose:
    return Pose.from_matrix(shoulder + plug_rot @ off_g, plug_rot @ rot_g)


def oracle_cycle(seed: int, connector: str, *, release: bool = True, noise: bool = True) -> tuple[MjPlugStation, dict]:
    """Pick from the tray, insert into the target port, release. Returns the station and a summary with truth."""
    exp = load_experiment()
    station = MjPlugStation(seed, connector, noise=noise)
    cfg, lay = station._cfg, station._lay
    depth = lay.geom.full_depth
    t0 = float(station._mj_data.time)
    steps: list[tuple[str, bool]] = []

    # 1. pick: come down over the plug where it stands in the tray, close, lift
    station.wait(0.3)
    shoulder, plug_rot = _plug_pose(station)
    yaw = float(np.arctan2(plug_rot[1, 0], plug_rot[0, 0]))
    upright = _rot_z(yaw)
    grasp_off = np.array([0.0, 0.0, scene.GRASP_HEIGHT])
    grasp = _tool_for_plug(shoulder, upright, PLUG_TO_TOOL, grasp_off)
    hover = Pose((grasp.pos[0], grasp.pos[1], grasp.pos[2] + APPROACH_M), grasp.quat)
    # go to a top-down posture first, so the long move ends near the right IK branch
    q_top, _, _ = station.solve_ik(ARM, hover, seed_q=np.array(TOP_DOWN_SEED))
    station._arms[ARM].q_plan = station._arms[ARM].q_plan  # no-op, the move below plans from the current joints
    steps.append(("to tray", _joint_move(station, q_top)))
    opening = min(1.0, (lay.geom.grip_width / 2 + 0.008) / 0.0375)
    station.set_gripper(ARM, opening)
    steps.append(("down to plug", station.move_to_pose(ARM, grasp, NEAR_SPEED).ok))
    station.set_gripper(ARM, 0.0)
    station.wait(0.2)
    steps.append(("lift", station.move_to_pose(ARM, hover, NEAR_SPEED).ok))

    # 2. carry: put the plug straight above the target port, using where it really sits in the fingers
    shoulder, plug_rot = _plug_pose(station)
    tool_pos, tool_rot = station._fk(ARM, station._q(ARM))
    rot_g = plug_rot.T @ tool_rot                      # tool in the plug frame, as grasped
    off_g = plug_rot.T @ (tool_pos - shoulder)
    port_pos, board_yaw = scene.port_pose_world(cfg, lay, cfg.target_port)
    above = _tool_for_plug(port_pos + np.array([0.0, 0.0, depth + CLEARANCE_M + 0.03]), _rot_z(board_yaw), rot_g, off_g)
    start = _tool_for_plug(port_pos + np.array([0.0, 0.0, depth + CLEARANCE_M]), _rot_z(board_yaw), rot_g, off_g)
    steps.append(("to port", station.move_to_pose(ARM, above, FREE_SPEED).ok))
    steps.append(("pre-insert", station.move_to_pose(ARM, start, NEAR_SPEED).ok))
    # close the last error with the plug's true pose
    shoulder, plug_rot = _plug_pose(station)
    err = (port_pos + np.array([0.0, 0.0, depth + CLEARANCE_M])) - shoulder
    pose = station.robot_state().arms[ARM].tool_pose
    trim = Pose((pose.pos[0] + err[0], pose.pos[1] + err[1], pose.pos[2] + err[2]), start.quat)
    station.move_to_pose(ARM, trim, 0.01)
    station.zero_force_estimate(ARM)

    # 3. insert: a force-guarded descent, as in every injected state
    nominal = station.nominal()
    tool_z = float(station.robot_state().arms[ARM].tool_pose.pos[2])
    expected = tool_z - float(nominal.seated_tool_pose(cfg.target_port).pos[2])
    max_travel = CLEARANCE_M + depth + EXTRA_TRAVEL_M
    result = station.move_relative(ARM, [0.0, 0.0, -max_travel], "base", INSERT_SPEED, exp.push_force_cap_n(connector), max_travel)
    station.wait(0.2)
    flag = _insertion_flag(result, expected)
    held = truth(station)

    # 4. release and back away
    released, release_ref = None, None
    if release:
        release_ref = held["metrics"]["plug_pos"]
        station.set_gripper(ARM, 1.0)
        pose = station.robot_state().arms[ARM].tool_pose
        station.move_to_pose(ARM, Pose((pose.pos[0], pose.pos[1], pose.pos[2] + RELEASE_LIFT_M), pose.quat), 0.03)
        station.wait(exp.predicates["released_stable"]["settle_s"])
        released = truth(station, release_ref=release_ref)

    return station, {
        "seed": int(seed), "connector": connector, "flag": flag, "steps": steps,
        "insert_stopped_by": result.stopped_by, "insert_travel_mm": result.travel_m * 1000.0,
        "truth_held": held, "truth_released": released,
        "sim_seconds": float(station._mj_data.time) - t0,
    }


def _joint_move(station: MjPlugStation, q_goal: np.ndarray, seconds: float = 1.6) -> bool:
    """Move the right arm to joint angles along a minimum-jerk path."""
    from .station import CONTROL_DT

    st = station._arms[ARM]
    q0 = st.q_plan.copy()
    ticks = max(2, int(round(seconds / CONTROL_DT)))
    qd_prev = np.zeros(6)
    for k in range(1, ticks + 1):
        u = k / ticks
        s = 10 * u**3 - 15 * u**4 + 6 * u**5
        q_new = q0 + s * (q_goal - q0)
        st.qd_plan = (q_new - st.q_plan) / CONTROL_DT
        qdd = (st.qd_plan - qd_prev) / CONTROL_DT
        qd_prev = st.qd_plan
        st.q_plan = q_new
        station._integrate_hold(ARM)
        station._integrate_hold("left")
        station._command_all()
        station._command(ARM, qdd_plan=qdd)
        station._tick()
    st.qd_plan = np.zeros(6)
    station._hold(0.5)
    return bool(np.abs(station._q(ARM) - q_goal).max() < 0.02)


def do_nothing(seed: int, connector: str, seconds: float = 1.0) -> tuple[MjPlugStation, dict]:
    """The program that does nothing. The plug stays in the tray."""
    station = MjPlugStation(seed, connector)
    station.wait(seconds)
    return station, {"seed": int(seed), "connector": connector, "truth": truth(station)}
