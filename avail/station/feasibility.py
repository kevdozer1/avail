"""S0 feasibility spike: does the dual-YAM station load, render and move on this machine?

Loads ASPIRE's station model from its own folder (the XML references its meshes
relative to itself), parks both arms at the profile's ready joints, renders the 3
cameras, moves the right arm's grasp site 50 mm straight down and back with IK,
and prints the commanded actuator forces as the joint torque estimate.

Run it with ``uv run avail smoke``. ``tests/test_s0_smoke.py`` runs the same code.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .mj_plug.aspire import ARMS, CAMERAS, ready_joints, station_spec

MOVE_MM = 50.0
TRACK_GATE_MM = 2.0


def load_station():
    """The ASPIRE station as (model, data, ready) with both arms at their ready joints."""
    import mujoco

    model = station_spec().compile()
    data = mujoco.MjData(model)
    ready = {arm: list(ready_joints()[arm]) for arm in ARMS}
    for arm in ARMS:
        for i, q in enumerate(ready[arm], start=1):
            name = f"{arm}_joint{i}"
            data.qpos[model.joint(name).qposadr[0]] = q
            data.ctrl[model.actuator(name).id] = q
    mujoco.mj_forward(model, data)
    return model, data, ready


def _arm_ids(model, arm: str) -> dict:
    joints = [f"{arm}_joint{i}" for i in range(1, 7)]
    return {
        "qadr": np.array([model.joint(j).qposadr[0] for j in joints]),
        "dadr": np.array([model.joint(j).dofadr[0] for j in joints]),
        "act": np.array([model.actuator(j).id for j in joints]),
        "site": model.site(f"{arm}_grasp_site").id,
    }


def render_cameras(model, data, out_dir: Path, width: int = 640, height: int = 480) -> dict[str, dict]:
    """Render each station camera to ``smoke_<camera>.png``. Returns per-image statistics."""
    import mujoco
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict] = {}
    renderer = mujoco.Renderer(model, height, width)  # height first
    try:
        for cam in CAMERAS:
            renderer.update_scene(data, camera=cam)
            rgb = renderer.render()
            path = out_dir / f"smoke_{cam}.png"
            Image.fromarray(rgb).save(path)
            stats[cam] = {
                "path": str(path),
                "shape": list(rgb.shape),
                "mean": round(float(rgb.mean()), 2),
                "std": round(float(rgb.std()), 2),
                "distinct_colors": int(len(np.unique(rgb.reshape(-1, 3)[::7], axis=0))),
            }
    finally:
        renderer.close()
    return stats


def move_down_and_back(model, data, arm: str = "right", distance_mm: float = MOVE_MM,
                       speed_mm_s: float = 50.0, closed_loop: bool = True, solver: str = "mink") -> dict:
    """Move the arm's grasp site straight down by ``distance_mm`` and back. Report tracking error in mm.

    The joint position servos in this model are soft (40 Nm/rad at the shoulder, 10 at the
    wrist) and each joint has 0.1 Nm of friction loss, so a plain open-loop joint target
    leaves the tool short. ``closed_loop`` adds a velocity feedforward and a slow per-joint
    integral on the measured joint error, which is how the station's motion skills will run.
    """
    import mujoco

    ids = _arm_ids(model, arm)
    control_dt = 0.01
    substeps = max(1, int(round(control_dt / model.opt.timestep)))
    start = data.site_xpos[ids["site"]].copy()
    rot = data.site_xmat[ids["site"]].reshape(3, 3).copy()
    goal_down = start + np.array([0.0, 0.0, -distance_mm / 1000.0])
    move_s = distance_mm / speed_mm_s
    hold_s = 1.0

    if solver == "mink":
        import mink

        configuration = mink.Configuration(model)
        configuration.update(q=data.qpos.copy())
        frame = mink.FrameTask(f"{arm}_grasp_site", "site", position_cost=1.0, orientation_cost=1.0, lm_damping=1e-4)
        posture = mink.PostureTask(model, cost=1e-4)
        posture.set_target(data.qpos.copy())
        others = [d for d in range(model.nv) if d not in set(ids["dadr"].tolist())]
        freeze = mink.DofFreezingTask(model, dof_indices=others)
        limits = [mink.ConfigurationLimit(model)]
        target_rot = mink.SO3.from_matrix(rot)

        def solve(target_pos: np.ndarray) -> np.ndarray:
            frame.set_target(mink.SE3.from_rotation_and_translation(target_rot, target_pos))
            for _ in range(2):
                vel = mink.solve_ik(configuration, [frame, posture], control_dt, "daqp", damping=1e-6,
                                    limits=limits, constraints=[freeze])
                configuration.integrate_inplace(vel, control_dt)
            return configuration.q[ids["qadr"]].copy()
    else:
        q_cmd = data.qpos[ids["qadr"]].copy()
        scratch = mujoco.MjData(model)

        def solve(target_pos: np.ndarray) -> np.ndarray:
            nonlocal q_cmd
            for _ in range(3):
                scratch.qpos[:] = data.qpos
                scratch.qpos[ids["qadr"]] = q_cmd
                mujoco.mj_kinematics(model, scratch)
                mujoco.mj_comPos(model, scratch)
                jacp = np.zeros((3, model.nv))
                jacr = np.zeros((3, model.nv))
                mujoco.mj_jacSite(model, scratch, jacp, jacr, ids["site"])
                jac = np.vstack([jacp[:, ids["dadr"]], jacr[:, ids["dadr"]]])
                err_p = target_pos - scratch.site_xpos[ids["site"]]
                cur = scratch.site_xmat[ids["site"]].reshape(3, 3)
                err_r = 0.5 * (np.cross(cur[:, 0], rot[:, 0]) + np.cross(cur[:, 1], rot[:, 1])
                               + np.cross(cur[:, 2], rot[:, 2]))
                err = np.concatenate([err_p, err_r])
                dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-6 * np.eye(6), err)
                q_cmd = q_cmd + dq
            return q_cmd.copy()

    # servo gains straight from the model: torque = kp * (ctrl - q) - kd * qdot
    kp = model.actuator_gainprm[ids["act"], 0].copy()
    kd = -model.actuator_biasprm[ids["act"], 2].copy()
    ki, deadband, clamp = 5.0, 2e-4, 0.15            # 1/s, rad, rad
    q_plan_prev = data.qpos[ids["qadr"]].copy()
    qd_plan_prev = np.zeros(6)
    mass = np.zeros((model.nv, model.nv))
    # bumpless start: the offset the servo is already holding against gravity and friction
    offset = (data.ctrl[ids["act"]] - data.qpos[ids["qadr"]]) if closed_loop else np.zeros(6)
    trace: list[dict] = []

    def run_phase(p0: np.ndarray, p1: np.ndarray, seconds: float, label: str) -> None:
        nonlocal offset, q_plan_prev, qd_plan_prev
        ticks = max(1, int(round(seconds / control_dt)))
        for k in range(1, ticks + 1):
            u = k / ticks
            s = 10 * u**3 - 15 * u**4 + 6 * u**5          # minimum-jerk profile
            desired = p0 + s * (p1 - p0)
            q_plan = solve(desired)
            qd_plan = (q_plan - q_plan_prev) / control_dt
            qdd_plan = (qd_plan - qd_plan_prev) / control_dt
            q_plan_prev, qd_plan_prev = q_plan, qd_plan
            if closed_loop:
                err = q_plan - data.qpos[ids["qadr"]]
                err = np.where(np.abs(err) < deadband, 0.0, err)
                offset = np.clip(offset + ki * control_dt * err, -clamp, clamp)   # slow integral per joint
                mujoco.mj_fullM(model, data, mass)                                # MuJoCo 3.13 signature
                inertia = mass[np.ix_(ids["dadr"], ids["dadr"])] @ qdd_plan       # the arm's own model
                command = q_plan + (kd * qd_plan + inertia) / kp + offset         # velocity and inertia feedforward
            else:
                command = q_plan
            data.ctrl[ids["act"]] = command
            for _ in range(substeps):
                mujoco.mj_step(model, data)
            actual = data.site_xpos[ids["site"]].copy()
            trace.append({"phase": label, "err_mm": float(np.linalg.norm(desired - actual) * 1000.0),
                          "lateral_mm": float(np.linalg.norm((desired - actual)[:2]) * 1000.0)})

    run_phase(start, goal_down, move_s, "down")
    run_phase(goal_down, goal_down, hold_s, "hold_down")
    at_down = data.site_xpos[ids["site"]].copy()
    run_phase(goal_down, start, move_s, "up")
    run_phase(start, start, hold_s, "hold_up")
    at_up = data.site_xpos[ids["site"]].copy()

    def phase_max(label: str, key: str) -> float:
        return max(t[key] for t in trace if t["phase"] == label)

    return {
        "arm": arm,
        "solver": solver,
        "closed_loop": closed_loop,
        "distance_mm": distance_mm,
        "speed_mm_s": speed_mm_s,
        "travel_mm": round(float((start[2] - at_down[2]) * 1000.0), 3),
        "end_error_down_mm": round(float(np.linalg.norm(goal_down - at_down) * 1000.0), 3),
        "end_error_back_mm": round(float(np.linalg.norm(start - at_up) * 1000.0), 3),
        "max_error_moving_mm": round(max(phase_max("down", "err_mm"), phase_max("up", "err_mm")), 3),
        "max_lateral_moving_mm": round(max(phase_max("down", "lateral_mm"), phase_max("up", "lateral_mm")), 3),
    }


def torque_estimate(model, data, arm: str = "right") -> dict[str, float]:
    """Commanded actuator forces for one arm, the stand-in for a motor-current torque estimate (Nm)."""
    ids = _arm_ids(model, arm)
    return {f"{arm}_joint{i + 1}": round(float(data.actuator_force[a]), 4) for i, a in enumerate(ids["act"])}


def run_feasibility(out_dir: Path | None = None, render: bool = True) -> dict:
    """The whole spike. Returns a dict of numbers for the report and the test."""
    import mujoco

    from ..paths import check_disk_floor, runs_dir

    result: dict = {"mujoco": mujoco.__version__}
    model, data, ready = load_station()
    result["model"] = {"nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu), "ngeom": int(model.ngeom),
                       "ncam": int(model.ncam), "timestep_s": float(model.opt.timestep)}
    result["ready_joints"] = ready

    t0 = time.perf_counter()
    for _ in range(200):
        mujoco.mj_step(model, data)
    result["steps_per_s_raw"] = round(200 / (time.perf_counter() - t0))
    drift = {arm: float(np.abs(data.qpos[_arm_ids(model, arm)["qadr"]] - np.array(ready[arm])).max()) for arm in ARMS}
    result["max_joint_drift_rad_after_200_steps"] = {k: round(v, 5) for k, v in drift.items()}

    if render:
        if out_dir is None:
            check_disk_floor()
            out_dir = runs_dir()
        result["images"] = render_cameras(model, data, Path(out_dir))

    result["torque_estimate_nm_at_ready"] = torque_estimate(model, data, "right")
    try:
        import mink  # noqa: F401
        solver = "mink"
    except Exception as exc:  # recorded, then damped least squares takes over
        result["mink_error"] = f"{type(exc).__name__}: {exc}"
        solver = "dls"
    result["ik_solver"] = solver

    # open loop on a copy, to show what the soft servos and joint friction leave behind
    probe = mujoco.MjData(model)
    probe.qpos[:], probe.qvel[:], probe.ctrl[:] = data.qpos, data.qvel, data.ctrl
    mujoco.mj_forward(model, probe)
    result["move_open_loop"] = move_down_and_back(model, probe, closed_loop=False, solver=solver)
    slow = mujoco.MjData(model)
    slow.qpos[:], slow.qvel[:], slow.ctrl[:] = data.qpos, data.qvel, data.ctrl
    mujoco.mj_forward(model, slow)
    result["move_25mm_s"] = move_down_and_back(model, slow, closed_loop=True, solver=solver, speed_mm_s=25.0)
    result["move"] = move_down_and_back(model, data, closed_loop=True, solver=solver)
    result["torque_estimate_nm_after_move"] = torque_estimate(model, data, "right")
    worst = max(result["move"]["end_error_down_mm"], result["move"]["end_error_back_mm"])
    result["tracking_gate_mm"] = TRACK_GATE_MM
    result["tracking_ok"] = bool(worst <= TRACK_GATE_MM)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="avail smoke", description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None, help="folder for the 3 smoke images (default AVAIL_DATA/runs)")
    parser.add_argument("--no-render", action="store_true", help="skip the camera renders")
    args = parser.parse_args(argv)
    result = run_feasibility(out_dir=args.out, render=not args.no_render)
    print(json.dumps(result, indent=2), flush=True)
    move = result["move"]
    print(f"\n50 mm move, closed loop: end error {move['end_error_down_mm']} mm down, "
          f"{move['end_error_back_mm']} mm back (gate {TRACK_GATE_MM} mm). "
          f"{'PASS' if result['tracking_ok'] else 'FAIL'}", flush=True)
    return 0 if result["tracking_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
