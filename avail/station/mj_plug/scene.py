"""Compose the plug station: ASPIRE's dual-YAM model plus a generated board, sockets, a plug and a tray.

``build_model(connector)`` compiles one model per connector type and caches it per process.
``sample_scene(seed, connector)`` draws everything that varies per seed. ``apply_scene`` writes
that sample into the model in place, starting from the stored base values, so the same seed
always gives the same model whatever ran before it.

Frames. The board frame sits on the table top at the board center, z up. The plug frame sits
at the center of the plug's shoulder plane, z up along the plug axis, x across the faces the
fingers squeeze. Insertion is straight down in v0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from ...config import ConnectorSpec, load_experiment
from . import aspire

MM = 1e-3
TABLE_TOP_Z = aspire.TABLE_TOP_Z
BOARD_CENTER_XY = (0.50, -0.12)            # nominal board center on the table, base frame
TRAY_CENTER_XY = (0.52, -0.36)             # nominal tray center
PLATE_HALF_XY = (0.045, 0.090)
PLATE_T = 0.002
FLOOR_T = 0.0015
WALL_T = 0.003
BORE_EXTRA = 0.0005                        # the bore is this much deeper than the tongue, so the shoulder is the stop
NOSE_LEN = 0.0012                          # tapered tip of the tongue
NOSE_INSET = 0.0006
CHAMFER_PLATE_T = 0.0004
GRASP_HEIGHT = 0.016                       # nominal grasp point above the shoulder plane
PAD_OFFSET_Y = 0.044                       # fingertip pads relative to ASPIRE's grasp site, along the site's y
TOOL_SITE = {arm: f"{arm}_tool" for arm in aspire.ARMS}
TRAY_POCKET_DEPTH = 0.008
TRAY_BLOCK_HALF = (0.030, 0.025, 0.006)
N_LIGHTS = 4
N_CABLES = 3
N_TABLE_MATS = 4
N_BOARD_MATS = 4
PLUG_COLORS = (
    (0.93, 0.93, 0.90, 1.0),               # off white
    (0.95, 0.78, 0.12, 1.0),               # yellow
    (0.20, 0.42, 0.78, 1.0),               # blue
    (0.62, 0.64, 0.66, 1.0),               # gray
)

# solver settings that made sub-millimeter insertion behave in Kevin's earlier rack cell
TIMESTEP = 0.001
CONTACT_SOLREF = (0.004, 1.0)
CONTACT_SOLIMP = (0.95, 0.99, 0.001, 0.5, 2.0)
SOCKET_FRICTION = (0.3, 0.005, 0.0001)
PLUG_BODY_FRICTION = (0.8, 0.005, 0.0001)

# The ASPIRE plant squeezes with 100 N/m, about 0.6 N on a 12 mm plug. The real gripper motor is
# rated 3.75 Nm in the station profile. These values let the fingers hold a plug through a 15 N
# tug. ASSUMPTION, recorded in reports/S2.md.
FINGER_KP = 3000.0                         # N/m
FINGER_KD = 80.0                           # N s/m
FINGER_FORCE_MAX = 25.0                    # N


@dataclass(frozen=True)
class ConnectorGeom:
    """Plug and socket dimensions in meters."""

    name: str
    full_depth: float
    tongue: tuple[float, float]            # x, y cross-section
    body: tuple[float, float, float]       # x, y, z
    clearance: float
    chamfer: float

    @property
    def opening_half(self) -> tuple[float, float]:
        return (self.tongue[0] / 2 + self.clearance, self.tongue[1] / 2 + self.clearance)

    @property
    def housing_half(self) -> tuple[float, float]:
        ox, oy = self.opening_half
        return (ox + self.chamfer + WALL_T, oy + self.chamfer + WALL_T)

    @property
    def housing_height(self) -> float:
        return self.full_depth + BORE_EXTRA + FLOOR_T

    @property
    def face_z(self) -> float:
        """Height of the socket face above the table top."""
        return PLATE_T + self.housing_height

    @property
    def grip_width(self) -> float:
        return self.body[0]


def connector_geom(name: str, spec: ConnectorSpec | None = None) -> ConnectorGeom:
    spec = spec or load_experiment().connector(name)
    if spec.geometry_mm is None:
        raise ValueError(f"connector {name!r} has no geometry_mm in config/experiment.yaml, so it cannot be built")
    g = spec.geometry_mm
    return ConnectorGeom(
        name=name,
        full_depth=spec.full_depth_mm * MM,
        tongue=(g.tongue_xy[0] * MM, g.tongue_xy[1] * MM),
        body=(g.body_xyz[0] * MM, g.body_xyz[1] * MM, g.body_xyz[2] * MM),
        clearance=g.clearance * MM,
        chamfer=g.chamfer * MM,
    )


def board_layout() -> tuple[tuple[float, float], ...]:
    """Port centers in the board frame, port 1 first. 2 rows of 4 by default."""
    cfg = load_experiment().station.board
    pitch = cfg.pitch_mm * MM
    xs = [(r - (cfg.rows - 1) / 2) * pitch for r in range(cfg.rows)]
    ys = [(c - (cfg.cols - 1) / 2) * pitch for c in range(cfg.cols)]
    return tuple((x, y) for x in xs for y in ys)


N_PORTS = len(board_layout())


# ----------------------------------------------------------------------------- the per-seed sample
@dataclass(frozen=True)
class SceneConfig:
    """Everything that varies per seed. Values are rounded so the same seed is the same bytes."""

    seed: int
    connector: str
    board_xy: tuple[float, float]                      # offset from nominal, m
    board_yaw: float                                   # rad
    target_port: int                                   # 1..N_PORTS
    plug_color: int
    table_mat: int
    board_mat: int
    lights: tuple[tuple[int, float, float, float, float], ...]     # active, x, y, z, diffuse
    camera_jitter: tuple[tuple[float, float, float, float, float, float], ...]   # per camera: dx dy dz rx ry rz
    cables: tuple[tuple[int, float, float, float], ...]            # active, x, y, yaw in the base frame
    tray_xy_yaw: tuple[float, float, float]            # tray offset and yaw
    plug_in_tray: tuple[float, float, float]           # plug offset and yaw in its pocket
    # what the robot does not know about this episode
    localization_error: tuple[float, float, float]     # estimated minus true port position, m
    grasp_offset: tuple[float, float, float, float]    # along the plug axis, across the fingers, along the fingers, tilt rad
    force_bias: tuple[float, float, float]             # N, the force estimate's bias at the start of the episode
    yaw_error: float = 0.0                             # estimated minus true board yaw, rad


def _r(value: float, digits: int = 6) -> float:
    return float(round(float(value), digits))


def sample_scene(seed: int, connector: str) -> SceneConfig:
    """The scene for a seed. Pure function of (seed, connector)."""
    cfg = load_experiment()
    names = sorted(cfg.connectors)
    if connector not in names:
        raise KeyError(f"unknown connector {connector!r}. Known: {', '.join(names)}")
    rnd = cfg.station.randomization
    noise = cfg.station.noise
    rng = np.random.default_rng([int(seed), names.index(connector), 2026_11])

    board_xy = tuple(_r(v) for v in rng.uniform(-rnd.board_xy_mm, rnd.board_xy_mm, size=2) * MM)
    board_yaw = _r(math.radians(rng.uniform(-rnd.board_yaw_deg, rnd.board_yaw_deg)))
    target_port = int(rng.integers(1, N_PORTS + 1))
    plug_color = int(rng.integers(0, len(PLUG_COLORS)))
    table_mat = int(rng.integers(0, N_TABLE_MATS))
    board_mat = int(rng.integers(0, N_BOARD_MATS))

    n_lights = int(rng.integers(rnd.lights[0], rnd.lights[1] + 1))
    active = np.zeros(N_LIGHTS, dtype=int)
    active[rng.permutation(N_LIGHTS)[:n_lights]] = 1
    lights = []
    for i in range(N_LIGHTS):
        jitter = rng.uniform(-0.25, 0.25, size=3) * np.array([1.0, 1.0, 0.4])
        diffuse = rng.uniform(0.35, 0.8)
        lights.append((int(active[i]), _r(jitter[0]), _r(jitter[1]), _r(jitter[2]), _r(diffuse)))

    camera_jitter = []
    for _ in aspire.CAMERAS:
        dpos = rng.uniform(-rnd.camera_xyz_mm, rnd.camera_xyz_mm, size=3) * MM
        drot = np.radians(rng.uniform(-rnd.camera_rot_deg, rnd.camera_rot_deg, size=3))
        camera_jitter.append(tuple(_r(v) for v in (*dpos, *drot)))

    n_cables = int(rng.integers(rnd.distractor_cables[0], rnd.distractor_cables[1] + 1))
    cables = []
    for i in range(N_CABLES):
        x = rng.uniform(0.36, 0.72)
        y = rng.uniform(-0.02, 0.22) if rng.random() < 0.5 else rng.uniform(-0.52, -0.26)
        cables.append((int(i < n_cables), _r(x), _r(y), _r(rng.uniform(-math.pi, math.pi))))

    tray = (_r(rng.uniform(-0.01, 0.01)), _r(rng.uniform(-0.01, 0.01)), _r(math.radians(rng.uniform(-8, 8))))
    plug_in_tray = (_r(rng.uniform(-0.0006, 0.0006)), _r(rng.uniform(-0.0006, 0.0006)), _r(math.radians(rng.uniform(-3, 3))))

    sigma = noise.port_localization_sigma_mm * MM
    loc = rng.normal(0.0, sigma, size=3) * np.array([1.0, 1.0, 0.5])
    grasp = (rng.normal(0.0, 0.0010), rng.normal(0.0, 0.0004), rng.normal(0.0, 0.0006), math.radians(rng.normal(0.0, 0.5)))
    # Closing a parallel gripper centers the plug between the pads, so there is no offset across the fingers.
    grasp = (float(np.clip(grasp[0], -0.003, 0.003)), 0.0,
             float(np.clip(grasp[2], -0.002, 0.002)), float(np.clip(grasp[3], -math.radians(1.5), math.radians(1.5))))
    bias = rng.normal(0.0, 0.3, size=3)
    # the board's yaw is localized no better than its position: the same sigma over a 100 mm lever
    yaw_error = float(np.clip(rng.normal(0.0, sigma / 0.1), -3 * sigma / 0.1, 3 * sigma / 0.1))

    return SceneConfig(
        seed=int(seed), connector=connector, board_xy=board_xy, board_yaw=board_yaw, target_port=target_port,
        plug_color=plug_color, table_mat=table_mat, board_mat=board_mat, lights=tuple(lights),
        camera_jitter=tuple(camera_jitter), cables=tuple(cables), tray_xy_yaw=tray, plug_in_tray=plug_in_tray,
        localization_error=tuple(_r(v) for v in loc), grasp_offset=tuple(_r(v) for v in grasp),
        force_bias=tuple(_r(v, 4) for v in bias), yaw_error=_r(yaw_error),
    )


# ----------------------------------------------------------------------------- building the model
@dataclass
class Layout:
    """Ids and base values of everything the station reads or rewrites."""

    connector: str
    geom: ConnectorGeom
    ports_board: tuple[tuple[float, float], ...]
    ids: dict[str, int] = field(default_factory=dict)
    arm: dict[str, dict] = field(default_factory=dict)
    base: dict[str, np.ndarray] = field(default_factory=dict)
    profile: str = "easy"


def _quat_y(deg: float) -> list[float]:
    h = math.radians(deg) / 2
    return [math.cos(h), 0.0, math.sin(h), 0.0]


def _quat_x(deg: float) -> list[float]:
    h = math.radians(deg) / 2
    return [math.cos(h), math.sin(h), 0.0, 0.0]


def _quat_z_rad(rad: float) -> np.ndarray:
    return np.array([math.cos(rad / 2), 0.0, 0.0, math.sin(rad / 2)])


def _look_at_quat(pos, target) -> list[float]:
    """Quaternion of a MuJoCo camera at ``pos`` looking at ``target`` with world z up."""
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    z = pos - target
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    from ..base import Pose

    return list(Pose.from_matrix(pos, np.column_stack([x, y, z])).quat)


def _add_bore(body, mj, prefix: str, center: tuple[float, float], z_floor_top: float, z_face: float,
              opening_half: tuple[float, float], chamfer: float, material: str, contact: dict) -> None:
    """Walls, rim and chamfer plates of one rectangular bore, in the parent body's frame."""
    px, py = center
    ox, oy = opening_half
    ch, t = chamfer, WALL_T
    z1 = z_face - ch                                    # top of the straight bore
    mid, half = (z_floor_top + z1) / 2, (z1 - z_floor_top) / 2
    box = mj.mjtGeom.mjGEOM_BOX
    common = dict(type=box, material=material, **contact)
    wall = (ch + t) / 2
    body.add_geom(name=f"{prefix}_wall_xp", pos=[px + ox + wall, py, mid], size=[wall, oy + ch + t, half], **common)
    body.add_geom(name=f"{prefix}_wall_xn", pos=[px - ox - wall, py, mid], size=[wall, oy + ch + t, half], **common)
    body.add_geom(name=f"{prefix}_wall_yp", pos=[px, py + oy + wall, mid], size=[ox, wall, half], **common)
    body.add_geom(name=f"{prefix}_wall_yn", pos=[px, py - oy - wall, mid], size=[ox, wall, half], **common)
    if ch <= 0:
        return
    zr = z_face - ch / 2
    body.add_geom(name=f"{prefix}_rim_xp", pos=[px + ox + ch + t / 2, py, zr], size=[t / 2, oy + ch + t, ch / 2], **common)
    body.add_geom(name=f"{prefix}_rim_xn", pos=[px - ox - ch - t / 2, py, zr], size=[t / 2, oy + ch + t, ch / 2], **common)
    body.add_geom(name=f"{prefix}_rim_yp", pos=[px, py + oy + ch + t / 2, zr], size=[ox + ch, t / 2, ch / 2], **common)
    body.add_geom(name=f"{prefix}_rim_yn", pos=[px, py - oy - ch - t / 2, zr], size=[ox + ch, t / 2, ch / 2], **common)
    # 45 degree lead-in plates. Each plate's inner face runs from the bore edge up to the rim edge.
    half_len, th = ch * math.sqrt(2) / 2, CHAMFER_PLATE_T
    off = th / math.sqrt(2)
    body.add_geom(name=f"{prefix}_ch_xp", pos=[px + ox + ch / 2 + off, py, zr - off], quat=_quat_y(-45),
                  size=[half_len, oy + ch, th], **common)
    body.add_geom(name=f"{prefix}_ch_xn", pos=[px - ox - ch / 2 - off, py, zr - off], quat=_quat_y(45),
                  size=[half_len, oy + ch, th], **common)
    body.add_geom(name=f"{prefix}_ch_yp", pos=[px, py + oy + ch / 2 + off, zr - off], quat=_quat_x(45),
                  size=[ox + ch, half_len, th], **common)
    body.add_geom(name=f"{prefix}_ch_yn", pos=[px, py - oy - ch / 2 - off, zr - off], quat=_quat_x(-45),
                  size=[ox + ch, half_len, th], **common)


JAM_STOP_HALF_T = 0.00025                  # half the thickness of hard mode's jam stop
JAM_BITS = 2                               # contype bit that only the jam stop and the plug's tongue and nose share


def build_spec(connector: str, profile: str = "easy"):
    """The composed, uncompiled spec for one connector type.

    ``profile`` "hard" (S4b) adds 2 things and changes nothing else: a jam stop, a thin plate that
    sits in the target bore at the jam depth and collides with the plug's tongue and nose only,
    and a cable that leaves the plug's top and hangs down beside it, visual only, in place of the
    28 mm stub.
    """
    import mujoco as mj

    if profile not in ("easy", "hard"):
        raise ValueError(f"profile must be 'easy' or 'hard', got {profile!r}")
    hard = profile == "hard"
    geom = connector_geom(connector)
    spec = aspire.station_spec()
    spec.modelname = f"avail_plug_station_{connector}" + ("_hard" if hard else "")

    opt = spec.option
    opt.timestep = TIMESTEP
    opt.cone = mj.mjtCone.mjCONE_ELLIPTIC
    opt.impratio = 10.0
    opt.noslip_iterations = 3
    opt.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST

    # fingers that can hold a plug through a tug
    for act in spec.actuators:
        if act.name in aspire.GRIPPER_ACTUATOR.values():
            gain, bias = list(act.gainprm), list(act.biasprm)
            gain[0] = FINGER_KP
            bias[1], bias[2] = -FINGER_KP, -FINGER_KD
            act.gainprm, act.biasprm = gain, bias
            act.forcelimited = mj.mjtLimited.mjLIMITED_TRUE
            act.forcerange = [-FINGER_FORCE_MAX, FINGER_FORCE_MAX]

    # materials: a small set for the table and the board, picked per seed
    flat, checker = mj.mjtBuiltin.mjBUILTIN_FLAT, mj.mjtBuiltin.mjBUILTIN_CHECKER
    rgb_role = mj.mjtTextureRole.mjTEXROLE_RGB

    def material(name: str, rgb1, rgb2=None, builtin=flat, repeat=(1.0, 1.0), speckle: float = 0.0, markrgb=(0.5, 0.5, 0.5)):
        tex = spec.add_texture(name=f"{name}_tex", type=mj.mjtTexture.mjTEXTURE_2D, builtin=builtin,
                               rgb1=list(rgb1), rgb2=list(rgb2 or rgb1), width=256, height=256)
        if speckle > 0:
            tex.mark = mj.mjtMark.mjMARK_RANDOM
            tex.random = speckle
            tex.markrgb = list(markrgb)
        mat = spec.add_material(name=name, texrepeat=list(repeat), texuniform=True, reflectance=0.05)
        textures = list(mat.textures)
        textures[rgb_role] = f"{name}_tex"
        mat.textures = textures
        return mat

    material("table_mat0", (0.96, 0.96, 0.96))
    material("table_mat1", (0.80, 0.70, 0.55), speckle=0.08, markrgb=(0.62, 0.50, 0.36), repeat=(3, 3))
    material("table_mat2", (0.42, 0.45, 0.48), speckle=0.05, markrgb=(0.30, 0.32, 0.34), repeat=(4, 4))
    material("table_mat3", (0.20, 0.36, 0.30), (0.24, 0.42, 0.35), builtin=checker, repeat=(8, 8))
    material("board_mat0", (0.10, 0.36, 0.16), speckle=0.04, markrgb=(0.16, 0.46, 0.22), repeat=(6, 6))
    material("board_mat1", (0.12, 0.20, 0.45), speckle=0.04, markrgb=(0.18, 0.28, 0.58), repeat=(6, 6))
    material("board_mat2", (0.70, 0.66, 0.56), speckle=0.05, markrgb=(0.55, 0.50, 0.42), repeat=(6, 6))
    material("board_mat3", (0.30, 0.30, 0.32))
    spec.add_material(name="housing_mat", rgba=[0.09, 0.09, 0.10, 1.0], reflectance=0.0, specular=0.3, shininess=0.4)
    spec.add_material(name="tray_mat", rgba=[0.55, 0.55, 0.58, 1.0])
    spec.add_material(name="tongue_mat", rgba=[0.80, 0.80, 0.78, 1.0], specular=0.6, shininess=0.6)
    spec.add_material(name="plug_body_mat", rgba=list(PLUG_COLORS[0]))
    spec.add_material(name="cable_mat", rgba=[0.08, 0.08, 0.09, 1.0])
    spec.add_material(name="latch_tab_mat", rgba=[0.85, 0.85, 0.82, 1.0])

    # The fingertip pads of this gripper sit 44 mm to one side of the wrist axis, and ASPIRE's
    # grasp site sits on the axis. AVAIL's tool point is between the pad tips, which is where a
    # small part is actually held. Same orientation as the grasp site.
    for arm in aspire.ARMS:
        tcp = [b for b in spec.bodies if b.name == f"{arm}_tcp"][0]
        grasp = [s for s in tcp.sites if s.name == aspire.GRASP_SITE[arm]][0]
        tcp.add_site(name=TOOL_SITE[arm], pos=[grasp.pos[0], grasp.pos[1] + PAD_OFFSET_Y, grasp.pos[2]],
                     size=[0.002, 0.002, 0.002], rgba=[1, 0.5, 0, 1], group=5)

    # ASPIRE draws its frame markers as colored balls. A real camera does not see them.
    for site in spec.sites:
        site.group = 5
        site.rgba = [0.0, 0.0, 0.0, 0.0]

    # name the table so its material can be switched
    table_body = [b for b in spec.bodies if b.name == "play_table"][0]
    table_geom = table_body.first_geom()
    table_geom.name = "play_table_top"
    table_geom.material = "table_mat0"

    world = spec.worldbody
    contact = dict(friction=list(SOCKET_FRICTION), solref=list(CONTACT_SOLREF), solimp=list(CONTACT_SOLIMP), condim=4)
    box = mj.mjtGeom.mjGEOM_BOX

    # the board: a thin plate carrying one socket housing per port
    board = world.add_body(name="board", pos=[BOARD_CENTER_XY[0], BOARD_CENTER_XY[1], TABLE_TOP_Z])
    board.add_geom(name="board_plate", type=box, pos=[0, 0, PLATE_T / 2], size=[PLATE_HALF_XY[0], PLATE_HALF_XY[1], PLATE_T / 2],
                   material="board_mat0", **contact)
    hx, hy = geom.housing_half
    for i, (px, py) in enumerate(board_layout(), start=1):
        board.add_geom(name=f"port{i}_floor", type=box, pos=[px, py, PLATE_T + FLOOR_T / 2], size=[hx, hy, FLOOR_T / 2],
                       material="housing_mat", **contact)
        _add_bore(board, mj, f"port{i}", (px, py), PLATE_T + FLOOR_T, geom.face_z, geom.opening_half, geom.chamfer,
                  "housing_mat", contact)
        board.add_site(name=f"port{i}", pos=[px, py, geom.face_z], size=[0.001, 0.001, 0.001], group=5)
    if hard:
        # Placed in the target bore per seed, and far below the table otherwise. It is a body of its
        # own because MuJoCo's collision tree of a body with many geoms does not follow a geom
        # moved after compiling. A body moved after compiling is fine.
        ox, oy = geom.opening_half
        stop = board.add_body(name="jam_stop_body", pos=[0.0, 0.0, -1.0])
        stop.add_geom(name="jam_stop", type=box, pos=[0.0, 0.0, 0.0], size=[ox, oy, JAM_STOP_HALF_T],
                      material="housing_mat", contype=0, conaffinity=JAM_BITS, condim=4,
                      friction=list(SOCKET_FRICTION), solref=list(CONTACT_SOLREF), solimp=list(CONTACT_SOLIMP))

    # the tray: one loose pocket that holds the plug upright
    tray = world.add_body(name="tray", pos=[TRAY_CENTER_XY[0], TRAY_CENTER_XY[1], TABLE_TOP_Z])
    tz = 2 * TRAY_BLOCK_HALF[2]
    tray.add_geom(name="tray_floor", type=box, pos=[0, 0, (tz - TRAY_POCKET_DEPTH) / 2],
                  size=[TRAY_BLOCK_HALF[0], TRAY_BLOCK_HALF[1], (tz - TRAY_POCKET_DEPTH) / 2], material="tray_mat", **contact)
    pocket_half = (geom.tongue[0] / 2 + 0.0008, geom.tongue[1] / 2 + 0.0008)
    _add_bore(tray, mj, "tray_pocket", (0.0, 0.0), tz - TRAY_POCKET_DEPTH, tz, pocket_half, 0.0, "tray_mat", contact)
    tray.add_site(name="tray_pocket", pos=[0, 0, tz], size=[0.001, 0.001, 0.001], group=5)

    # the plug: a free body. Mass is mostly in the body the fingers hold.
    bx, by, bz = geom.body
    tx, ty = geom.tongue
    depth = geom.full_depth
    plug = world.add_body(name="plug", pos=[TRAY_CENTER_XY[0], TRAY_CENTER_XY[1], TABLE_TOP_Z + 0.2])
    plug.add_freejoint(name="plug_free")
    plug_contact = dict(solref=list(CONTACT_SOLREF), solimp=list(CONTACT_SOLIMP), condim=4)
    plug.add_geom(name="plug_body", type=box, pos=[0, 0, bz / 2], size=[bx / 2, by / 2, bz / 2], mass=0.020,
                  material="plug_body_mat", friction=list(PLUG_BODY_FRICTION), **plug_contact)
    tip_bits = dict(contype=1 | JAM_BITS) if hard else {}
    plug.add_geom(name="plug_tongue", type=box, pos=[0, 0, -(depth - NOSE_LEN) / 2], size=[tx / 2, ty / 2, (depth - NOSE_LEN) / 2],
                  mass=0.004, material="tongue_mat", friction=list(SOCKET_FRICTION), **plug_contact, **tip_bits)
    plug.add_geom(name="plug_nose", type=box, pos=[0, 0, -depth + NOSE_LEN / 2],
                  size=[tx / 2 - NOSE_INSET, ty / 2 - NOSE_INSET, NOSE_LEN / 2], mass=0.001, material="tongue_mat",
                  friction=list(SOCKET_FRICTION), **plug_contact, **tip_bits)
    if hard:                                # the hanging cable, segments placed per seed by apply_scene
        from .hard import CABLE_SEGMENTS

        for i in range(CABLE_SEGMENTS):
            plug.add_geom(name=f"plug_cable_h{i}", type=mj.mjtGeom.mjGEOM_CAPSULE,
                          fromto=[0, 0, bz + 0.001 * i, 0, 0, bz + 0.001 * (i + 1)], size=[0.002, 0, 0],
                          contype=0, conaffinity=0, mass=0.0, material="cable_mat")
    else:
        plug.add_geom(name="plug_cable", type=mj.mjtGeom.mjGEOM_CAPSULE, fromto=[0, -by / 2, bz - 0.006, 0, -by / 2 - 0.028, bz - 0.001],
                      size=[0.0025, 0, 0], contype=0, conaffinity=0, mass=0.0, material="cable_mat")
    if connector == "latch":
        plug.add_geom(name="plug_latch_tab", type=box, pos=[0, by / 2 + 0.0008, 0.0055], size=[0.0025, 0.0008, 0.0045],
                      contype=0, conaffinity=0, mass=0.0, material="latch_tab_mat")
    plug.add_site(name="plug_tip", pos=[0, 0, -depth], size=[0.0008, 0.0008, 0.0008], group=5)
    plug.add_site(name="plug_shoulder", pos=[0, 0, 0], size=[0.0008, 0.0008, 0.0008], group=5)
    plug.add_site(name="plug_grasp", pos=[0, 0, GRASP_HEIGHT], size=[0.0008, 0.0008, 0.0008], group=5)

    # distractor cables: visual only, pre-built, shown or hidden per seed
    for i in range(N_CABLES):
        cable = world.add_body(name=f"cable{i}", pos=[0.6, 0.3 + 0.05 * i, TABLE_TOP_Z + 0.004])
        pts = _cable_points(i)
        for k in range(len(pts) - 1):
            a, b = pts[k], pts[k + 1]
            cable.add_geom(name=f"cable{i}_seg{k}", type=mj.mjtGeom.mjGEOM_CAPSULE, fromto=[a[0], a[1], 0, b[0], b[1], 0],
                           size=[0.003, 0, 0], contype=0, conaffinity=0, mass=0.0, material="cable_mat")

    # extra lights. The station's own directional light is light 0.
    for i, (lx, ly) in enumerate(((0.95, -0.45), (0.95, 0.40), (0.15, -0.55)), start=1):
        world.add_light(name=f"aux_light{i}", pos=[lx, ly, TABLE_TOP_Z + 0.75], dir=[0.5 - lx, -0.1 - ly, -0.75],
                        diffuse=[0.5, 0.5, 0.5], specular=[0.1, 0.1, 0.1], castshadow=False)
    # No shadows anywhere. On this GPU the shadow map makes a few pixels differ by 1 level between
    # identical renders, which breaks the byte-for-byte rule and any cache keyed on image bytes.
    for light in spec.lights:
        light.castshadow = False

    # a camera for clips a person watches. It is not one of the station's sensors.
    cam_pos = [BOARD_CENTER_XY[0] + 0.19, BOARD_CENTER_XY[1] - 0.21, TABLE_TOP_Z + 0.11]
    cam_target = [BOARD_CENTER_XY[0], BOARD_CENTER_XY[1], TABLE_TOP_Z + 0.03]
    world.add_camera(name="video_cam", pos=cam_pos, quat=_look_at_quat(cam_pos, cam_target), fovy=40.0)
    return spec, geom


def _cable_points(index: int) -> list[tuple[float, float]]:
    """A fixed wavy polyline per cable index, in the cable body's frame."""
    n, length = 9, 0.20 + 0.04 * index
    amp, waves = 0.018 + 0.006 * index, 1.0 + 0.5 * index
    return [((k / (n - 1) - 0.5) * length, amp * math.sin(2 * math.pi * waves * k / (n - 1))) for k in range(n)]


@lru_cache(maxsize=4)
def build_model(connector: str, profile: str = "easy"):
    """Compile the station for one connector type and profile. Returns (model, layout). Cached per process."""
    spec, geom = build_spec(connector, profile)
    return layout_of(spec.compile(), geom, connector, profile)


def layout_of(model, geom: ConnectorGeom, connector: str, profile: str = "easy"):
    """The layout of a compiled station model: the ids and base values the station reads or rewrites.
    Returns (model, layout). A model built on ``build_spec`` with parts added (the cable loop) uses it too."""
    lay = Layout(connector=connector, geom=geom, ports_board=board_layout())
    lay.profile = profile
    ids = lay.ids
    for name in ("board", "tray", "plug", *[f"cable{i}" for i in range(N_CABLES)]):
        ids[f"body:{name}"] = model.body(name).id
    for name in ("plug_tip", "plug_shoulder", "plug_grasp", "tray_pocket", *[f"port{i}" for i in range(1, N_PORTS + 1)]):
        ids[f"site:{name}"] = model.site(name).id
    for name in ("play_table_top", "board_plate", "plug_body"):
        ids[f"geom:{name}"] = model.geom(name).id
    if connector == "latch":
        ids["geom:plug_latch_tab"] = model.geom("plug_latch_tab").id
    for name in (*aspire.CAMERAS, "video_cam"):
        ids[f"cam:{name}"] = model.camera(name).id
    for i in range(N_TABLE_MATS):
        ids[f"mat:table{i}"] = model.material(f"table_mat{i}").id
    for i in range(N_BOARD_MATS):
        ids[f"mat:board{i}"] = model.material(f"board_mat{i}").id
    ids["mat:plug_body"] = model.material("plug_body_mat").id
    if profile == "hard":
        from .hard import CABLE_SEGMENTS

        ids["geom:jam_stop"] = model.geom("jam_stop").id
        ids["body:jam_stop"] = model.body("jam_stop_body").id
        for i in range(CABLE_SEGMENTS):
            ids[f"geom:plug_cable_h{i}"] = model.geom(f"plug_cable_h{i}").id
    joint = model.joint("plug_free")
    ids["plug_qadr"] = int(joint.qposadr[0])
    ids["plug_dadr"] = int(joint.dofadr[0])
    for arm in aspire.ARMS:
        joints = aspire.ARM_JOINTS[arm]
        act = np.array([model.actuator(j).id for j in joints])
        finger = model.joint(aspire.FINGER_JOINT[arm])
        lay.arm[arm] = {
            "qadr": np.array([model.joint(j).qposadr[0] for j in joints]),
            "dadr": np.array([model.joint(j).dofadr[0] for j in joints]),
            "act": act,
            "kp": model.actuator_gainprm[act, 0].copy(),
            "kd": -model.actuator_biasprm[act, 2].copy(),
            "site": model.site(TOOL_SITE[arm]).id,
            "grip_act": model.actuator(aspire.GRIPPER_ACTUATOR[arm]).id,
            "finger_qadr": int(finger.qposadr[0]),
            "finger_dadr": int(finger.dofadr[0]),
            "finger2_qadr": int(model.joint(f"{arm}_right_finger").qposadr[0]),
            "cam": model.camera(aspire.WRIST_CAMERA[arm]).id,
            "lower": model.jnt_range[[model.joint(j).id for j in joints], 0].copy(),
            "upper": model.jnt_range[[model.joint(j).id for j in joints], 1].copy(),
        }
    # base values of everything apply_scene rewrites
    lay.base = {
        "body_pos": model.body_pos.copy(), "body_quat": model.body_quat.copy(),
        "cam_pos": model.cam_pos.copy(), "cam_quat": model.cam_quat.copy(),
        "light_pos": model.light_pos.copy(), "light_dir": model.light_dir.copy(),
        "light_diffuse": model.light_diffuse.copy(), "light_active": model.light_active.copy(),
        "geom_matid": model.geom_matid.copy(), "geom_rgba": model.geom_rgba.copy(), "geom_pos": model.geom_pos.copy(),
        "mat_rgba": model.mat_rgba.copy(),
        "dof_frictionloss": model.dof_frictionloss.copy(), "dof_damping": model.dof_damping.copy(),
        "dof_armature": model.dof_armature.copy(), "dof_solimp": model.dof_solimp.copy(),
    }
    if profile == "hard":
        lay.base.update({"geom_quat": model.geom_quat.copy(), "geom_size": model.geom_size.copy(),
                         "geom_rbound": model.geom_rbound.copy()})
    return model, lay


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _rotvec_quat(rv) -> np.ndarray:
    rv = np.asarray(rv, float)
    angle = float(np.linalg.norm(rv))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = rv / angle
    return np.array([math.cos(angle / 2), *(math.sin(angle / 2) * axis)])


def apply_scene(model, lay: Layout, cfg: SceneConfig, hard=None, jam_shortfall_mm: float | None = None) -> None:
    """Write a scene sample into the model in place, starting from the stored base values.

    A hard-mode model also takes the seed's hard-mode draws (``hard``) and, for a jammed short
    plug, the shortfall at which the jam stop holds the plug's tip in the target port.
    """
    if cfg.connector != lay.connector:
        raise ValueError(f"scene is for connector {cfg.connector!r} and the model is for {lay.connector!r}")
    for name, base in lay.base.items():
        getattr(model, name)[:] = base
    ids = lay.ids

    board = ids["body:board"]
    model.body_pos[board, 0] += cfg.board_xy[0]
    model.body_pos[board, 1] += cfg.board_xy[1]
    model.body_quat[board] = _quat_z_rad(cfg.board_yaw)

    tray = ids["body:tray"]
    model.body_pos[tray, 0] += cfg.tray_xy_yaw[0]
    model.body_pos[tray, 1] += cfg.tray_xy_yaw[1]
    model.body_quat[tray] = _quat_z_rad(cfg.tray_xy_yaw[2])

    model.geom_matid[ids["geom:play_table_top"]] = ids[f"mat:table{cfg.table_mat}"]
    model.geom_matid[ids["geom:board_plate"]] = ids[f"mat:board{cfg.board_mat}"]
    model.mat_rgba[ids["mat:plug_body"]] = PLUG_COLORS[cfg.plug_color]

    for i, (active, dx, dy, dz, diffuse) in enumerate(cfg.lights):
        model.light_active[i] = bool(active)
        model.light_pos[i] += (dx, dy, dz)
        model.light_diffuse[i] = diffuse

    for cam_name, jitter in zip(aspire.CAMERAS, cfg.camera_jitter):
        cam = ids[f"cam:{cam_name}"]
        model.cam_pos[cam] += jitter[:3]
        model.cam_quat[cam] = _quat_mul(model.cam_quat[cam], _rotvec_quat(jitter[3:]))

    for i, (active, x, y, yaw) in enumerate(cfg.cables):
        body = ids[f"body:cable{i}"]
        model.body_pos[body] = (x, y, TABLE_TOP_Z + 0.0035) if active else (x, y, -1.0)   # hidden below the floor
        model.body_quat[body] = _quat_z_rad(yaw)

    if lay.profile == "hard":
        if hard is None:
            raise ValueError("a hard-mode model needs the seed's hard-mode draws")
        _place_cable(model, lay, hard)
        stop = ids["body:jam_stop"]
        if jam_shortfall_mm is not None:            # the stop's top face is where the plug's tip comes to rest
            px, py = lay.ports_board[cfg.target_port - 1]
            z_tip = lay.geom.face_z - (lay.geom.full_depth - float(jam_shortfall_mm) * MM)
            model.body_pos[stop] = (px, py, z_tip - JAM_STOP_HALF_T)
    elif hard is not None or jam_shortfall_mm is not None:
        raise ValueError("hard-mode draws were given for an easy-mode model")


def _place_cable(model, lay: Layout, hard) -> None:
    """Lay the hanging cable's segments along this seed's route. Visual only."""
    from .hard import CABLE_SEGMENTS, cable_points

    radius = load_experiment().hard().plug_cable.diameter_mm * MM / 2.0
    pts = np.asarray(cable_points(hard, lay.connector, lay.geom))
    for i in range(CABLE_SEGMENTS):
        g = lay.ids[f"geom:plug_cable_h{i}"]
        a, b = pts[i], pts[i + 1]
        d = b - a
        length = float(np.linalg.norm(d))
        z = d / length
        axis = np.cross([0.0, 0.0, 1.0], z)
        s, c = float(np.linalg.norm(axis)), float(z[2])
        if s < 1e-12:
            quat = np.array([1.0, 0.0, 0.0, 0.0]) if c > 0 else np.array([0.0, 1.0, 0.0, 0.0])
        else:
            angle = math.atan2(s, c)
            quat = np.array([math.cos(angle / 2), *(math.sin(angle / 2) * axis / s)])
        model.geom_pos[g] = 0.5 * (a + b)
        model.geom_quat[g] = quat
        model.geom_size[g, 0] = radius
        model.geom_size[g, 1] = 0.5 * length
        model.geom_rbound[g] = radius + 0.5 * length


def port_pose_world(model_or_cfg, lay: Layout, port: int, cfg: SceneConfig | None = None) -> tuple[np.ndarray, float]:
    """True position of a port's face center in the base frame, and the board yaw. Privileged."""
    cfg = cfg if cfg is not None else model_or_cfg
    px, py = lay.ports_board[port - 1]
    c, s = math.cos(cfg.board_yaw), math.sin(cfg.board_yaw)
    x = BOARD_CENTER_XY[0] + cfg.board_xy[0] + c * px - s * py
    y = BOARD_CENTER_XY[1] + cfg.board_xy[1] + s * px + c * py
    return np.array([x, y, TABLE_TOP_Z + lay.geom.face_z]), cfg.board_yaw
