"""The cable loop's scene: the plug station with a 5-slot rack of cables beside the board.

Built on ``scene.build_spec`` with parts added, so the board, the arms, the cameras and the plug are
the station's own. The station has 1 physical plug, a free body. In the cable loop it stands for the
cable the arm is handling. The other 4 cables are visual bodies that never collide, posed per step:
upright in their rack slot, or where they ended (seated in a port, or wherever they came to rest).
The physical tray pocket is invisible and moves to the slot the arm picks from.

Simulator side. Evidence sources never import this module.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from . import scene

N_CABLES = 5
SLOT_PITCH = 0.030                          # rack slots along the tray's x axis
RACK_CENTER_X = 0.54                        # nominal rack center, base frame. The tray's own center is scene.TRAY_CENTER_XY
CABLE_NAMES = ("red", "green", "blue", "yellow", "white")
CABLE_RGBA = (
    (0.82, 0.16, 0.14, 1.0),
    (0.14, 0.62, 0.22, 1.0),
    (0.16, 0.34, 0.86, 1.0),
    (0.95, 0.78, 0.10, 1.0),
    (0.93, 0.93, 0.90, 1.0),
)
HIDDEN = (0.0, 0.0, -1.0)                   # a visual cable that is not shown sits below the floor


def slot_center_rack(k: int) -> tuple[float, float]:
    """Slot k's center in the rack frame (rack center, x along the slots)."""
    return ((k - (N_CABLES - 1) / 2) * SLOT_PITCH, 0.0)


@lru_cache(maxsize=4)
def build_cable_model(connector: str, profile: str = "easy"):
    """Compile the cable-loop station for 1 connector type. Returns (model, layout). Cached per process."""
    import mujoco as mj

    spec, geom = scene.build_spec(connector, profile)
    spec.modelname = f"avail_cable_loop_{connector}" + ("" if profile == "easy" else f"_{profile}")
    box = mj.mjtGeom.mjGEOM_BOX
    world = spec.worldbody
    bx, by, bz = geom.body
    tx, ty = geom.tongue
    depth = geom.full_depth

    # the physical tray pocket is invisible: the rack below is what a camera sees
    tray = [b for b in spec.bodies if b.name == "tray"][0]
    for g in tray.geoms:
        g.group = 3

    # the rack: a plate and 1 pocket outline per slot, visual only
    tz = 2 * scene.TRAY_BLOCK_HALF[2]
    rack = world.add_body(name="cable_rack", pos=[RACK_CENTER_X, scene.TRAY_CENTER_XY[1], scene.TABLE_TOP_Z])
    half_x = (N_CABLES * SLOT_PITCH) / 2 + 0.008
    rack.add_geom(name="rack_plate", type=box, pos=[0, 0, (tz - scene.TRAY_POCKET_DEPTH) / 2],
                  size=[half_x, scene.TRAY_BLOCK_HALF[1], (tz - scene.TRAY_POCKET_DEPTH) / 2],
                  material="tray_mat", contype=0, conaffinity=0)
    ox, oy = tx / 2 + 0.0008, ty / 2 + 0.0008
    wall = 0.003
    zc, hz = tz - scene.TRAY_POCKET_DEPTH / 2, scene.TRAY_POCKET_DEPTH / 2
    for k in range(N_CABLES):
        cx, cy = slot_center_rack(k)
        for name, pos, size in (
            ("xp", (cx + ox + wall / 2, cy, zc), (wall / 2, oy + wall, hz)),
            ("xn", (cx - ox - wall / 2, cy, zc), (wall / 2, oy + wall, hz)),
            ("yp", (cx, cy + oy + wall / 2, zc), (ox, wall / 2, hz)),
            ("yn", (cx, cy - oy - wall / 2, zc), (ox, wall / 2, hz)),
        ):
            rack.add_geom(name=f"rack_slot{k}_{name}", type=box, pos=list(pos), size=list(size), material="tray_mat",
                          contype=0, conaffinity=0)

    # 5 visual cables, each the shape of the plug, its own color, with a stub of black cable
    for k in range(N_CABLES):
        spec.add_material(name=f"vcable{k}_mat", rgba=list(CABLE_RGBA[k]))
        body = world.add_body(name=f"vcable{k}", pos=list(HIDDEN))
        body.add_geom(name=f"vcable{k}_body", type=box, pos=[0, 0, bz / 2], size=[bx / 2, by / 2, bz / 2],
                      material=f"vcable{k}_mat", contype=0, conaffinity=0, mass=0.0)
        body.add_geom(name=f"vcable{k}_tongue", type=box, pos=[0, 0, -depth / 2], size=[tx / 2, ty / 2, depth / 2],
                      material="tongue_mat", contype=0, conaffinity=0, mass=0.0)
        body.add_geom(name=f"vcable{k}_cable", type=mj.mjtGeom.mjGEOM_CAPSULE,
                      fromto=[0, -by / 2, bz - 0.006, 0, -by / 2 - 0.028, bz - 0.001], size=[0.0025, 0, 0],
                      contype=0, conaffinity=0, mass=0.0, material="cable_mat")
        if connector == "latch":
            body.add_geom(name=f"vcable{k}_tab", type=box, pos=[0, by / 2 + 0.0008, 0.0055], size=[0.0025, 0.0008, 0.0045],
                          contype=0, conaffinity=0, mass=0.0, material="latch_tab_mat")

    model = spec.compile()
    model, lay = scene.layout_of(model, geom, connector, profile)
    lay.ids["body:cable_rack"] = model.body("cable_rack").id
    for k in range(N_CABLES):
        lay.ids[f"body:vcable{k}"] = model.body(f"vcable{k}").id
        lay.ids[f"mat:vcable{k}"] = model.material(f"vcable{k}_mat").id
    lay.base["body_pos"] = model.body_pos.copy()
    lay.base["body_quat"] = model.body_quat.copy()
    return model, lay


def rack_pose(cfg) -> tuple[np.ndarray, float]:
    """The rack's true center (x, y, table top) and yaw for a seed's scene: the tray's offset and yaw move it."""
    x = RACK_CENTER_X + cfg.tray_xy_yaw[0]
    y = scene.TRAY_CENTER_XY[1] + cfg.tray_xy_yaw[1]
    return np.array([x, y, scene.TABLE_TOP_Z]), float(cfg.tray_xy_yaw[2])


def slot_world(cfg, k: int) -> tuple[np.ndarray, float]:
    """Slot k's pocket center on the table (z at the table top) and the rack's yaw. Privileged, true pose."""
    center, yaw = rack_pose(cfg)
    sx, sy = slot_center_rack(k)
    c, s = math.cos(yaw), math.sin(yaw)
    return center + np.array([c * sx - s * sy, s * sx + c * sy, 0.0]), yaw


def shoulder_z_in_slot(lay) -> float:
    """Height of a cable's shoulder plane when it stands in a slot."""
    return scene.TABLE_TOP_Z + 2 * scene.TRAY_BLOCK_HALF[2] - scene.TRAY_POCKET_DEPTH + lay.geom.full_depth + 0.0005
