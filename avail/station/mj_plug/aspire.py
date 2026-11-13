"""ASPIRE's dual-YAM station model, read from ``third_party/`` and never copied.

The station XML is Apache-2.0 and lives in the ASPIRE clone (sessions/S0.md step
4). It is loaded from its own folder so its mesh paths resolve. Everything AVAIL
changes is changed in memory on the parsed spec.
"""

from __future__ import annotations

from functools import lru_cache

from ...paths import aspire_station_profile, aspire_station_xml

ARMS = ("left", "right")
ARM_JOINTS = {arm: tuple(f"{arm}_joint{i}" for i in range(1, 7)) for arm in ARMS}
GRIPPER_ACTUATOR = {arm: f"{arm}_gripper" for arm in ARMS}
FINGER_JOINT = {arm: f"{arm}_left_finger" for arm in ARMS}          # the actuated finger. The other is coupled
GRASP_SITE = {arm: f"{arm}_grasp_site" for arm in ARMS}
WRIST_CAMERA = {arm: f"{arm}_camera_d405" for arm in ARMS}
TOP_CAMERA = "top_camera_d405"
CAMERAS = (TOP_CAMERA, WRIST_CAMERA["left"], WRIST_CAMERA["right"])
FINGER_OPEN_M = 0.0375                                              # finger travel from closed to open, per finger
TABLE_TOP_Z = 0.75

# The overhead camera body points +Z at the table and MuJoCo cameras look along -Z. The wrist
# cameras carry a half-turn that accounts for this and the overhead camera does not, so as
# written it renders the sky. This half-turn about Y is applied in memory.
TOP_CAMERA_QUAT = (0.0, 0.0, 1.0, 0.0)


def require_station_files():
    xml, profile = aspire_station_xml(), aspire_station_profile()
    missing = [str(p) for p in (xml, profile) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "ASPIRE station files not found: " + ", ".join(missing) + ". "
            "Clone https://github.com/NVlabs/ASPIRE into third_party/ASPIRE at commit fa00f0e (sessions/S0.md step 4)."
        )
    return xml, profile


def station_spec():
    """The station as an editable ``mujoco.MjSpec`` with the overhead camera fixed."""
    import mujoco

    xml, _ = require_station_files()
    spec = mujoco.MjSpec.from_file(str(xml))
    top = [c for c in spec.cameras if c.name == TOP_CAMERA]
    if len(top) != 1:
        raise RuntimeError(f"expected exactly 1 camera named {TOP_CAMERA} in {xml}, found {len(top)}")
    top[0].quat = list(TOP_CAMERA_QUAT)
    return spec


@lru_cache(maxsize=1)
def ready_joints() -> dict[str, tuple[float, ...]]:
    """The profile's ready joints per arm, in radians."""
    import yaml

    _, profile_path = require_station_files()
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    return {arm: tuple(float(v) for v in profile["arms"][arm]["ready_joints"]) for arm in ARMS}
