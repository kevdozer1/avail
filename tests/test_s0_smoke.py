"""S0 feasibility: the dual-YAM station loads, renders 3 cameras and tracks a 50 mm move.

Needs the ASPIRE clone under third_party/ (sessions/S0.md step 4). Skipped without it.
"""

from __future__ import annotations

import numpy as np
import pytest

from avail.paths import aspire_station_profile, aspire_station_xml

pytestmark = pytest.mark.skipif(
    not (aspire_station_xml().is_file() and aspire_station_profile().is_file()),
    reason="third_party/ASPIRE is not cloned",
)


@pytest.fixture(scope="module")
def spike(tmp_path_factory):
    from avail.station.feasibility import run_feasibility

    return run_feasibility(out_dir=tmp_path_factory.mktemp("smoke"), render=True)


def test_station_loads_with_two_arms_and_three_cameras(spike):
    assert spike["model"]["ncam"] == 3
    assert spike["model"]["nu"] == 14                      # 6 joints and 1 gripper per arm
    assert spike["max_joint_drift_rad_after_200_steps"]["right"] < 0.01
    assert spike["max_joint_drift_rad_after_200_steps"]["left"] < 0.01


def test_three_smoke_images_show_a_scene(spike):
    from PIL import Image

    assert set(spike["images"]) == {"top_camera_d405", "left_camera_d405", "right_camera_d405"}
    for cam, info in spike["images"].items():
        rgb = np.asarray(Image.open(info["path"]))
        assert rgb.shape == (480, 640, 3), cam
        # a blank or gradient-only frame has few distinct colors. The sky alone gave 111.
        assert info["distinct_colors"] > 500, f"{cam} looks empty: {info}"
        assert info["std"] > 20, cam


def test_overhead_camera_sees_the_table_not_the_sky(spike):
    from PIL import Image

    rgb = np.asarray(Image.open(spike["images"]["top_camera_d405"]["path"])).astype(int)
    white = (rgb.min(axis=2) > 200).mean()               # the white play table
    assert white > 0.15, f"only {white:.1%} of the overhead view is table"


def test_50_mm_move_tracks_within_2_mm(spike):
    move = spike["move"]
    assert abs(move["travel_mm"] - 50.0) < 2.0
    assert move["end_error_down_mm"] < 2.0
    assert move["end_error_back_mm"] < 2.0
    assert move["max_lateral_moving_mm"] < 2.0
    assert spike["tracking_ok"] is True


def test_open_loop_move_shows_why_the_servo_needs_help(spike):
    """The soft joint servos alone leave the tool millimeters off. This is the under-travel the station keeps."""
    assert spike["move_open_loop"]["end_error_down_mm"] > spike["move"]["end_error_down_mm"]


def test_torque_estimate_is_the_commanded_actuator_force(spike):
    est = spike["torque_estimate_nm_after_move"]
    assert set(est) == {f"right_joint{i}" for i in range(1, 7)}
    assert all(np.isfinite(v) and abs(v) < 28.0 for v in est.values())
