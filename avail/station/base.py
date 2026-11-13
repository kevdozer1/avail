"""The plug station's taught geometry, and the Station protocol under its old address.

The Station protocol and the plain data that crosses it (``Pose``, ``CameraFrame``,
``Observation``, ``ArmState``, ``RobotState``, ``MotionResult``, ``STOP_REASONS``) live in
``avail.core.robot`` since S5, so the core can use them without importing the testbed.
They are imported here under their old names, the same objects, so every
``from avail.station.base import ...`` keeps working.

What stays here is testbed knowledge: ``NominalGeometry``, what a deployment of the plug
station would know about it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core.robot import STOP_REASONS, ArmState, CameraFrame, MotionResult, Observation, Pose, RobotState, Station

__all__ = ["STOP_REASONS", "ArmState", "CameraFrame", "MotionResult", "NominalGeometry", "Observation", "Pose",
           "RobotState", "Station"]


@dataclass(frozen=True)
class NominalGeometry:
    """What a deployment would know about the station: taught poses and part dimensions.

    Positions come from a localization that is off by about a millimeter (config
    ``station.noise.port_localization_sigma_mm``). Nothing here is ground truth.
    """

    connector: str
    full_depth_m: float
    grasp_height_m: float
    board_xy: tuple[float, float]             # estimated board center, base frame
    board_yaw: float                          # estimated
    face_z: float                             # estimated height of the socket faces, base frame
    ports_board: tuple[tuple[float, float], ...]
    pitch_m: float
    plug_body_m: tuple[float, float, float] = (0.0, 0.0, 0.0)     # x across the fingers, y, z along the plug axis
    plug_tongue_m: tuple[float, float] = (0.0, 0.0)               # cross-section of the part that goes in
    housing_half_m: tuple[float, float] = (0.0, 0.0)              # half-size of a socket housing on the board
    housing_height_m: float = 0.0
    finger_travel_m: float = 0.0                                  # one finger's travel from closed to open
    pad_thickness_m: float = 0.0
    holding_arm: str = "right"
    free_arm: str = "left"

    def expected_jaw_pos(self) -> float:
        """Gripper reading (0 closed to 1 open) expected while the fingers hold the plug."""
        if self.finger_travel_m <= 0:
            return float("nan")
        return max(0.0, (self.plug_body_m[0] / 2 - self.pad_thickness_m) / self.finger_travel_m)

    def port_position(self, port: int) -> np.ndarray:
        """Estimated face center of a port, base frame."""
        px, py = self.ports_board[port - 1]
        c, s = math.cos(self.board_yaw), math.sin(self.board_yaw)
        return np.array([self.board_xy[0] + c * px - s * py, self.board_xy[1] + s * px + c * py, self.face_z])

    def tool_rotation(self) -> np.ndarray:
        """Tool orientation for a plug held straight above the board: tool z down, tool x across the fingers."""
        c, s = math.cos(self.board_yaw), math.sin(self.board_yaw)
        return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])

    def seated_tool_pose(self, port: int) -> Pose:
        """The taught tool pose with the plug seated in ``port``."""
        pos = self.port_position(port) + np.array([0.0, 0.0, self.grasp_height_m])
        return Pose.from_matrix(pos, self.tool_rotation())

    def pre_insert_tool_pose(self, port: int, clearance_m: float) -> Pose:
        """The taught tool pose with the plug's tip ``clearance_m`` above the socket face of ``port``."""
        pos = self.port_position(port) + np.array([0.0, 0.0, self.grasp_height_m + self.full_depth_m + clearance_m])
        return Pose.from_matrix(pos, self.tool_rotation())

    def nearest_port(self, xy) -> tuple[int, float]:
        """The port whose estimated center is closest to a base-frame point. Returns (port, distance m)."""
        best, dist = 0, float("inf")
        for port in range(1, len(self.ports_board) + 1):
            d = float(np.linalg.norm(np.asarray(xy, float)[:2] - self.port_position(port)[:2]))
            if d < dist:
                best, dist = port, d
        return best, dist
