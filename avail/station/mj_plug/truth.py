"""Ground truth for the plug station. Privileged.

Only the evaluator, the audit sampler and ``OracleHuman`` may import this module
(CLAUDE.md rule 10). Evidence sources, witnesses, fork policies and agents never do.
tests/test_truth_wall.py enforces the boundary.
"""

from __future__ import annotations

import math

import numpy as np

from .scene import TABLE_TOP_Z, port_pose_world

LATERAL_SEARCH_M = 0.02


def _latch_engage_mm(station) -> float:
    """Shortfall below which the latch has snapped in: where the engaged detent turns from push-out to pull-in."""
    table = station._retention.detent_engaged
    xs, ys = table.xs, table.ys
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if y0 >= 0.0 > y1 and x1 > x0:
            return x0 + (x1 - x0) * (y0 / (y0 - y1)) if y0 != y1 else x0
    return xs[-1]


def plug_metrics(station) -> dict:
    """Where the plug really is, relative to the nearest port and to the target port."""
    mujoco = station._mj
    model, data, lay, cfg = station._mj_model, station._mj_data, station._lay, station._cfg
    mujoco.mj_kinematics(model, data)
    ids = lay.ids
    tip = data.site_xpos[ids["site:plug_tip"]].copy()
    shoulder = data.site_xpos[ids["site:plug_shoulder"]].copy()
    axis = data.site_xmat[ids["site:plug_shoulder"]].reshape(3, 3)[:, 2]
    tilt = math.degrees(math.acos(float(np.clip(axis[2], -1.0, 1.0))))
    face_z = TABLE_TOP_Z + lay.geom.face_z
    depth_mm = (face_z - float(tip[2])) * 1000.0

    c, s = math.cos(cfg.board_yaw), math.sin(cfg.board_yaw)
    best_port, best_d, in_bore, lateral_mm = 0, float("inf"), False, float("nan")
    ox, oy = lay.geom.opening_half
    for port in range(1, len(lay.ports_board) + 1):
        center, _ = port_pose_world(cfg, lay, port)
        dx, dy = float(tip[0] - center[0]), float(tip[1] - center[1])
        d = math.hypot(dx, dy)
        if d < best_d:
            lx, ly = c * dx + s * dy, -s * dx + c * dy
            best_port, best_d = port, d
            in_bore = depth_mm > 0.0 and abs(lx) <= ox + 0.0004 and abs(ly) <= oy + 0.0004
            sx, sy = float(shoulder[0] - center[0]), float(shoulder[1] - center[1])
            lateral_mm = math.hypot(sx, sy) * 1000.0
    full = station._spec.full_depth_mm
    shortfall_mm = full - depth_mm if in_bore else float("nan")

    arm_roots = {model.body(f"{arm}_arm").id for arm in ("left", "right")}
    plug = ids["body:plug"]
    finger_bodies = set()
    for i in range(data.ncon):
        con = data.contact[i]
        b1, b2 = int(model.geom_bodyid[con.geom1]), int(model.geom_bodyid[con.geom2])
        if plug not in (b1, b2):
            continue
        other = b2 if b1 == plug else b1
        if int(model.body_rootid[other]) in arm_roots and con.dist < 0.0005:
            finger_bodies.add(other)
    va = ids["plug_dadr"]
    speed = float(np.linalg.norm(data.qvel[va:va + 3]))
    gripper_open = {
        arm: float(data.qpos[lay.arm[arm]["finger_qadr"]]) > 0.85 * 0.0375 for arm in ("left", "right")
    }
    return {
        "nearest_port": best_port,
        "in_bore": bool(in_bore),
        "depth_mm": float(depth_mm),
        "shortfall_mm": float(shortfall_mm),
        "lateral_mm": float(lateral_mm),
        "tilt_deg": float(tilt),
        "latch_blocked": bool(station._retention.latch_blocked),
        "finger_contacts": len(finger_bodies),
        "plug_speed_m_s": speed,
        "gripper_open": gripper_open,
        "plug_pos": [float(v) for v in shoulder],
        "true_force_z_n": float(station._true_fz),
    }


def truth(station, target_port: int | None = None, release_ref: list[float] | None = None) -> dict:
    """Each predicate's true value for the plug and the target port, with the metrics behind it.

    ``release_ref`` is the plug position recorded when a release was commanded. With it,
    ``released_stable`` also requires that the plug stayed within tolerance of that position.
    """
    exp, spec, connector = station._exp, station._spec, station.connector
    target = int(target_port if target_port is not None else station._cfg.target_port)
    m = plug_metrics(station)
    tol = exp.predicates["seated"]
    at_port = m["in_bore"] and m["nearest_port"] == target
    if spec.seated_min_depth_mm is not None:
        deep_enough = m["depth_mm"] >= spec.seated_min_depth_mm
    else:
        deep_enough = m["in_bore"] and m["shortfall_mm"] <= tol["depth_tol_mm"]
    straight = m["tilt_deg"] <= tol["tilt_tol_deg"] and m["lateral_mm"] <= tol["lateral_tol_mm"]
    seated_anywhere = bool(m["in_bore"] and deep_enough and straight)
    seated = bool(at_port and seated_anywhere)
    if connector == "latch":
        engaged = bool(m["in_bore"] and not m["latch_blocked"] and m["shortfall_mm"] <= _latch_engage_mm(station))
        latched = bool(seated and engaged)
    else:
        engaged, latched = None, None
    held = m["finger_contacts"] >= 2
    still = m["plug_speed_m_s"] < 0.002
    let_go = (not held) and m["gripper_open"]["right"]
    stayed = True
    if release_ref is not None:
        stayed = float(np.linalg.norm(np.array(m["plug_pos"]) - np.array(release_ref))) * 1000.0 <= 0.5
    released_stable = bool(let_go and still and stayed and seated_anywhere)
    return {
        "target_port": target,
        "predicates": {
            "at_port": bool(at_port),
            "seated": seated,
            "latched": latched,
            "held": bool(held),
            "released_stable": released_stable,
            "removed": bool(not (m["in_bore"] and m["nearest_port"] == target)),
        },
        "latch_engaged": engaged,
        "metrics": m,
    }
