"""The Station protocol and the plain data that crosses it.

A Station is what a probe or an evidence source is allowed to touch: cameras,
proprioception and guarded motion. Nothing here holds an object pose, and
nothing here imports a simulator. Three backends implement the protocol: the
MuJoCo plug station, a client of a CAP MCP server, and the SO-101. A probe
written against ``Station`` runs on all 3.

These types moved here from ``avail.station.base`` in S5, so the core can use
them without importing the testbed. ``avail.station.base`` still offers every
name under its old address, next to the plug station's taught geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = ["STOP_REASONS", "ArmState", "CameraFrame", "MotionResult", "Observation", "Pose", "RobotState", "Station"]

# why a motion ended
STOP_REASONS = ("target", "travel", "force", "stall", "timeout", "stopped", "error")


@dataclass(frozen=True)
class Pose:
    """A rigid pose in the station base frame. Position in meters, quaternion as (w, x, y, z)."""

    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]

    def position(self) -> np.ndarray:
        return np.asarray(self.pos, dtype=float)

    def rotation(self) -> np.ndarray:
        """The 3 x 3 rotation matrix of this pose."""
        w, x, y, z = (float(v) for v in self.quat)
        n = float(np.sqrt(w * w + x * x + y * y + z * z)) or 1.0
        w, x, y, z = w / n, x / n, y / n, z / n
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])

    @staticmethod
    def from_matrix(pos, rot) -> "Pose":
        """Build a pose from a position and a 3 x 3 rotation matrix."""
        m = np.asarray(rot, dtype=float)
        t = float(np.trace(m))
        if t > 0:
            s = np.sqrt(t + 1.0) * 2
            q = ((0.25 * s), (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            q = ((m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s)
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            q = ((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s)
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            q = ((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s)
        p = np.asarray(pos, dtype=float)
        return Pose((float(p[0]), float(p[1]), float(p[2])), tuple(float(v) for v in q))


@dataclass
class CameraFrame:
    """One camera image with the calibration the station believes in.

    ``intrinsics`` is the 3 x 3 pinhole matrix in pixels. ``extrinsics`` is the 4 x 4
    camera-to-base transform from the station's calibration (OpenCV convention: +Z
    forward, +X right, +Y down). The real camera may sit a few millimeters away from it.
    """

    name: str
    rgb: np.ndarray                 # H x W x 3, uint8
    intrinsics: np.ndarray          # 3 x 3
    extrinsics: np.ndarray          # 4 x 4, nominal
    ts: float

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])


@dataclass
class Observation:
    """Camera images and a timestamp. No object pose."""

    frames: dict[str, CameraFrame]
    ts: float


@dataclass
class ArmState:
    """Proprioception for one arm. Forces are estimates from motor torque. No object pose."""

    joint_pos: np.ndarray           # rad
    joint_vel: np.ndarray           # rad/s
    joint_torque_est: np.ndarray    # Nm, commanded motor torque with noise
    gripper_pos: float              # 0 closed to 1 open
    gripper_torque: float           # the gripper motor's effort, in the backend's unit (N for a linear gripper)
    tool_pose: Pose                 # from forward kinematics
    tool_force_est: np.ndarray      # N, base frame, the force the tool applies to the world, from joint torques


@dataclass
class RobotState:
    arms: dict[str, ArmState]
    ts: float


@dataclass
class MotionResult:
    """What a motion did. ``travel_m`` is measured tool travel along the commanded direction."""

    ok: bool
    stopped_by: str                              # one of STOP_REASONS
    travel_m: float = 0.0
    duration_s: float = 0.0
    force_at_stop_n: float | None = None         # estimated tool force along the motion at the end
    peak_force_n: float | None = None            # largest estimated tool force along the motion
    trace: dict[str, list[float]] = field(default_factory=dict)   # t_s, travel_mm, force_n for guarded moves
    note: str = ""

    def __post_init__(self) -> None:
        if self.stopped_by not in STOP_REASONS:
            raise ValueError(f"stopped_by must be one of {STOP_REASONS}, got {self.stopped_by!r}")


@runtime_checkable
class Station(Protocol):
    """Everything a probe or an evidence source may do to a robot cell."""

    station_id: str

    def observe(self, cameras: list[str] | None = None) -> Observation:
        """Fresh camera images. ``cameras`` limits which ones are captured."""
        ...

    def robot_state(self) -> RobotState:
        """Joints, a joint torque estimate, gripper position and torque, and the tool pose per arm."""
        ...

    def move_to_pose(self, arm: str, pose: Pose, speed: float, max_force: float | None = None) -> MotionResult:
        """Move the tool to ``pose`` in a straight line at ``speed`` (m/s). Stops early above ``max_force`` (N)."""
        ...

    def move_relative(self, arm: str, delta_m, frame: str, speed: float, max_force: float,
                      max_travel: float) -> MotionResult:
        """A guarded move along ``delta_m`` (meters, in ``frame``: 'base' or 'tool').

        Stops at whichever comes first: the estimated force along the motion reaches
        ``max_force``, or the measured travel reaches ``max_travel`` or the length of
        ``delta_m``. The result says which.
        """
        ...

    def set_gripper(self, arm: str, position: float, max_torque: float | None = None) -> MotionResult:
        """Gripper to ``position`` (0 closed to 1 open), with an optional effort limit."""
        ...

    def stop(self) -> None:
        """Stop all motion and hold."""
        ...

    def wait(self, seconds: float) -> None:
        """Hold still for ``seconds``. On a real cell this is a sleep. In simulation it advances time."""
        ...
