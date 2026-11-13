"""vision.cable_tag: which cable is in the jaws? From the holding arm's wrist camera.

1 check, 1 file. Each cable of a task carries its own color, and the task tells the check the color
of the cable a claim names and the colors of all of them (``params["cable"]``, a palette taught on
dev seeds). The check projects the tool point, which sits inside the held part, into the claim-time
wrist frame with the frame's own calibration, and reads the color there: the median of a small
window. It compares the reading's direction in RGB with each palette color's, so a cable in shadow
is still its own color: the angle between them says which cable is in the jaws. A reading too dark
to have a direction (the fingers, a shadow) or far in angle from every cable says no cable is seen.

Why the angle. The 1st version compared RGB distances. A seed's lighting scales a rendered color,
and on the dev shift dev1 (2026-11-12) the white cable in dim light read 0.4 to 0.6 from white, so the
check saw no cable on about 1 pick in 5 and a person was asked. On 100 clean picks over 20 dev seeds
(23,900 to 23,919), each read in the rack and over its port, a cable's reading stayed within 1.1
degrees of its own color and 28 degrees or more from every other, with its brightest channel at 0.40
or more. The thresholds below were set from those picks before the next dev shift.

What it cannot see: whether the part hangs straight. A part cocked about the line between the pads
turns toward or away from this camera, and on dev seeds the angle the check read in the image stayed
at 5 to 7 degrees for parts cocked by 0 to 14 degrees. So it reports no angle. A cocked part shows
instead in where the arm holds it over the port (``passive.hover_pose``).

Features: ``present``, ``match``, ``mismatch``, ``nearest`` (palette index), ``angle_deg`` (to the
nearest color), ``expected_angle_deg``, ``brightness`` (the reading's brightest channel), ``color_dist``
and ``expected_dist`` (RGB distances, for reference), ``in_view``. Score: ``-expected_angle_deg``.
It moves nothing and costs a frame's worth of time.
"""

from __future__ import annotations

import numpy as np

from ..schemas import Claim, Cost, Evidence
from .base import make_evidence, register

WINDOW_PX = 7              # half-size of the window read at the tool point
BRIGHT_MIN = 0.15          # a reading whose brightest channel is under this has no color: the fingers or a shadow
PRESENT_DEG = 10.0         # a reading this close in angle to a cable's color is that cable (clean picks: 1.1 at most)
MARGIN_DEG = 10.0          # and nearer the named one than any other by this much to call a match (colors: 28 apart)


def angles_deg(color: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """The angle in degrees between a reading and each palette color, as directions in RGB."""
    c = np.asarray(color, float)
    p = np.asarray(palette, float)
    cos = (p @ c) / (np.linalg.norm(p, axis=1) * max(float(np.linalg.norm(c)), 1e-12))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def project(frame, point: np.ndarray) -> np.ndarray | None:
    """Pixel of a base-frame point in a frame, from the frame's calibration. None when behind the camera."""
    t = np.asarray(frame.extrinsics, float)
    cam = t[:3, :3].T @ (np.asarray(point, float) - t[:3, 3])
    if cam[2] <= 1e-6:
        return None
    uv = np.asarray(frame.intrinsics, float) @ (cam / cam[2])
    return uv[:2]


def tool_color(frame, point: np.ndarray) -> np.ndarray | None:
    """The color the frame shows at a base-frame point, RGB from 0 to 1: the median of a small window. None
    when the point is not in view. Teaching the palette reads colors with this function too."""
    uv = project(frame, point)
    h, w = frame.rgb.shape[:2]
    if uv is None or not (0 <= uv[0] < w and 0 <= uv[1] < h):
        return None
    rgb = frame.rgb.astype(float) / 255.0
    u, v = int(round(uv[0])), int(round(uv[1]))
    win = rgb[max(0, v - WINDOW_PX):v + WINDOW_PX + 1, max(0, u - WINDOW_PX):u + WINDOW_PX + 1].reshape(-1, 3)
    return np.median(win, axis=0)


class CableTag:
    source_id = "vision.cable_tag"
    kind = "passive_vision"
    predicates = {"held", "at_port"}

    def estimate_cost(self, claim: Claim) -> Cost:
        return Cost(seconds=0.05, usd=0.0, moved_robot=False)

    def collect(self, station, claim: Claim, params: dict) -> Evidence:
        nominal = params["nominal"]
        arm = params.get("arm", nominal.holding_arm)
        cable = params["cable"]
        key = f"claim/{arm}_camera_d405"
        frame = params["frames"].get(key) if params.get("frames") is not None else None
        if frame is None:
            frame = station.observe([f"{arm}_camera_d405"]).frames[f"{arm}_camera_d405"]
        tool = station.robot_state().arms[arm].tool_pose
        features: dict[str, float] = {"in_view": 0.0, "present": 0.0, "match": 0.0, "mismatch": 0.0}
        color = tool_color(frame, tool.position())
        if color is None:
            return make_evidence(self, claim, features, score=None, seconds=0.05)
        features["in_view"] = 1.0
        palette = np.asarray(cable["palette"], float)
        dist = np.linalg.norm(palette - color, axis=1)
        ang = angles_deg(color, palette)
        nearest = int(np.argmin(ang))
        expected = int(cable["index"])
        others = np.delete(ang, expected)
        brightness = float(np.max(color))
        present = bool(brightness >= BRIGHT_MIN and ang[nearest] <= PRESENT_DEG)
        features.update({"nearest": float(nearest), "angle_deg": float(ang[nearest]), "expected_angle_deg": float(ang[expected]),
                         "brightness": brightness, "color_dist": float(dist[nearest]), "expected_dist": float(dist[expected]),
                         "present": float(present),
                         "match": float(present and nearest == expected and float(others.min()) - ang[expected] >= MARGIN_DEG),
                         "mismatch": float(present and nearest != expected and ang[expected] - ang[nearest] >= MARGIN_DEG)})
        return make_evidence(self, claim, features, score=-float(ang[expected]), seconds=0.05)


register(CableTag())
