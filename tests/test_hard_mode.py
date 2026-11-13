"""S4b hard mode: the profile, its seed ranges, the webcam model, jams, lots and the force estimate.

Station tests build dev seeds only. Easy mode must not change: the easy model has none of hard
mode's parts, and an easy station never draws from hard mode's streams.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from avail import config, seeds
from avail.station.mj_plug import webcam
from avail.station.mj_plug.inject import ErrorSpec, sample_error

DEV = 6100                              # a dev seed for hard-mode station tests


# ----------------------------------------------------------------------------- config and seeds
def test_hard_profile_parses_with_the_brief_values():
    hard = config.load_experiment().hard()
    cam = hard.wrist_camera
    assert tuple(cam.sensor_px) == (1920, 1080) and cam.diagonal_fov_deg == 130 and cam.projection == "equidistant"
    assert cam.max_exposure_s == pytest.approx(1 / 30, abs=1e-4)
    assert tuple(cam.working_px) == (640, 360) and cam.chroma_subsampling == "4:2:2"
    assert hard.force.detent_lot_pct == 30
    assert set(hard.jams.classes) == {"N1", "N2", "N3", "N4"} and 0 < hard.jams.share < 1


def test_hard_ranges_are_their_own_and_the_test_range_is_locked(tmp_path):
    budget = config.load_experiment().seeds
    assert budget.split_of(60000) == "zoo_fit_hard" and budget.split_of(63999) == "zoo_calibration_hard"
    assert budget.split_of(64000) == "zoo_test_hard" and "zoo_test_hard" in config.TEST_SPLITS
    assert seeds.FINAL_RUNS["E1bE2b"] == "zoo_test_hard"
    with pytest.raises(seeds.TestSeedError):
        seeds.assert_simulable(64000)
    with pytest.raises(seeds.TestSeedError):          # E1E2 opens the easy test range only
        seeds.guard([64000], purpose="unit test", final_run="E1E2", record_path=tmp_path / "reads.jsonl")
    with pytest.raises(seeds.TestSeedError):          # and E1bE2b the hard one only
        seeds.guard([16000], purpose="unit test", final_run="E1bE2b", record_path=tmp_path / "reads.jsonl")
    assert not (tmp_path / "reads.jsonl").exists()


def test_a_seed_is_built_in_its_own_mode_only():
    assert seeds.profile_of_seed(60000) == "hard" and seeds.profile_of_seed(10000) == "easy"
    assert seeds.profile_of_seed(DEV) is None
    with pytest.raises(seeds.TestSeedError):
        seeds.assert_profile(60000, "easy")
    with pytest.raises(seeds.TestSeedError):
        seeds.assert_profile(10000, "hard")
    seeds.assert_profile(DEV, "easy")
    seeds.assert_profile(DEV, "hard")


def test_hard_draws_leave_the_easy_draws_alone():
    for cls in ("P0", "N1", "N2", "N3", "N4", "N6"):
        easy = sample_error(DEV, "fric", cls)
        hard = sample_error(DEV, "fric", cls, profile="hard")
        assert replace(hard, jam_kind=None, wedge_hold_n=None) == easy
        if cls not in ("N1", "N2", "N3", "N4"):
            assert hard.jam_kind is None


def test_the_jam_share_is_near_the_config_share():
    share = config.load_experiment().hard().jams.share
    kinds = [sample_error(s, "latch", "N2", profile="hard").jam_kind for s in range(6000, 6400)]
    jammed = np.mean([k is not None for k in kinds])
    assert abs(jammed - share) < 0.08
    assert {"burr", "wedge"} <= set(k for k in kinds if k)


# ----------------------------------------------------------------------------- the webcam model
def _cam(seed: int = 0) -> webcam.Webcam:
    spec = config.load_experiment().hard().wrist_camera
    return webcam.Webcam(spec, webcam.draw_unit(spec, np.random.default_rng(seed)))


def test_distort_and_undistort_are_inverse():
    k = (-0.04, 0.008, 0.0, 0.0)
    theta = np.linspace(0.0, 1.2, 50)
    assert np.allclose(webcam.undistort(webcam.distort(theta, k), k), theta, atol=1e-10)


def test_the_lens_maps_match_the_exact_formulas():
    cam = _cam(3)
    unit, plan = cam.unit, cam.plan
    raw_x, raw_y = cam._raw_from_pinhole()
    rng = np.random.default_rng(4)
    us, vs = rng.integers(0, 1920, 500), rng.integers(0, 1080, 500)
    dx, dy = us - unit.cx, vs - unit.cy
    theta = webcam.undistort(np.hypot(dx, dy) / unit.f, unit.k)
    phi = np.arctan2(dy, dx)
    ex = plan["f"] * np.tan(theta) * np.cos(phi) + (plan["width"] - 1) / 2
    ey = plan["f"] * np.tan(theta) * np.sin(phi) + (plan["height"] - 1) / 2
    assert np.abs(raw_x[vs, us] - ex).max() < 0.01 and np.abs(raw_y[vs, us] - ey).max() < 0.01


def test_every_rectified_pixel_reads_a_rendered_ray():
    for s in range(4):
        cam = _cam(s)
        raw_x, raw_y = cam._raw_from_pinhole()
        (rx, ry), _ = cam._rect_from_raw()
        inside = (raw_x >= 0) & (raw_x <= cam.plan["width"] - 1) & (raw_y >= 0) & (raw_y <= cam.plan["height"] - 1)
        u = np.clip(np.rint(rx).astype(int), 0, 1919)
        v = np.clip(np.rint(ry).astype(int), 0, 1079)
        assert inside[v, u].all()


def test_a_point_lands_where_the_working_intrinsics_say_within_the_calibration_error():
    cam = _cam(5)
    plan = cam.plan
    point = np.array([0.012, 0.035, 0.15])                     # camera frame, meters, about 15 degrees off axis
    pin = np.zeros((plan["height"], plan["width"], 3), dtype=np.uint8)
    pu = plan["f"] * point[0] / point[2] + (plan["width"] - 1) / 2
    pv = plan["f"] * point[1] / point[2] + (plan["height"] - 1) / 2
    import cv2

    cv2.circle(pin, (int(round(pu * 8)), int(round(pv * 8))), 40, (255, 255, 255), -1, lineType=cv2.LINE_AA, shift=3)
    rgb, k, _ = cam.process(pin, np.random.default_rng(0), distance_m=0.5)
    luma = rgb.astype(float).mean(axis=2)
    mask = luma > 0.5 * luma.max()
    yy, xx = np.nonzero(mask)
    w = luma[mask]
    found = np.array([np.sum(xx * w) / w.sum(), np.sum(yy * w) / w.sum()])
    expected = np.array([k[0, 0] * point[0] / point[2] + k[0, 2], k[1, 1] * point[1] / point[2] + k[1, 2]])
    assert np.linalg.norm(found - expected) < 1.5             # working pixels: the calibration is close, not exact


def test_a_dim_scene_gets_more_gain_and_more_noise():
    cam = _cam(6)
    plan = cam.plan
    out = {}
    for name, level in (("bright", 200), ("dim", 18)):
        pin = np.full((plan["height"], plan["width"], 3), level, dtype=np.uint8)
        rgb, _, info = cam.process(pin, np.random.default_rng(1), distance_m=0.5)
        center = rgb[120:240, 220:420].astype(float).mean(axis=2)
        out[name] = (info.gain, info.exposure_s, float(center.std()))
    assert out["dim"][0] > out["bright"][0] and out["dim"][1] >= out["bright"][1]
    assert out["dim"][2] > out["bright"][2]


def test_rolling_shutter_shows_early_rows_where_the_content_was_earlier():
    cam = _cam(7)
    h, w = cam.sensor_h, cam.sensor_w
    lin = np.zeros((h, w, 3), dtype=np.float32)
    col = 967
    lin[:, col - 1:col + 2] = 1.0                              # a vertical line where the content is now
    # the reference point, on the row through the optical center, moved +x by 60 raw pixels over the last 0.1 s
    times = np.linspace(-0.1, 0.0, 26)
    target_px = col - 60.0 * (-times / 0.1)
    theta = webcam.undistort((target_px - cam.unit.cx) / cam.unit.f, cam.unit.k)
    pts = np.stack([np.tan(theta) * 0.5, np.zeros_like(theta), np.full_like(theta, 0.5)], axis=1)
    assert np.allclose(cam.project_raw(pts)[:, 0], target_px, atol=1e-6)
    info = webcam.FrameInfo()
    out = cam._apply_motion(lin, webcam.Motion(times=times, points_cam=pts), exposure=0.005, info=info)
    top = float(np.average(np.arange(w), weights=out[0, :, 0] + 1e-12))
    bottom = float(np.average(np.arange(w), weights=out[h - 1, :, 0] + 1e-12))
    assert info.shutter_shift_px > 1.0
    assert top < bottom <= col + 0.5                           # the row read earliest shows the content farthest back
    assert col - top > 5.0


def test_the_defocus_follows_the_thin_lens():
    cam = _cam(0)
    near, focus = cam.defocus_sigma_px(0.15), cam.defocus_sigma_px(cam.spec.focus_distance_m)
    assert focus < 1e-9 < near and 0.5 < near < 5.0


# ----------------------------------------------------------------------------- the hard station (dev seeds)
@pytest.fixture(scope="module")
def mujoco_ok():
    pytest.importorskip("mujoco")
    from avail.station.mj_plug import aspire

    try:
        aspire.require_station_files()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    return True


def test_hard_frames_are_exact_and_survive_a_snapshot(mujoco_ok):
    from avail.zoo.harness import prepare

    a = prepare(DEV, "latch", "N1", profile="hard")
    b = prepare(DEV, "latch", "N1", profile="hard")
    try:
        fa = a.params["frames"].get("claim/right_camera_d405")
        fb = b.params["frames"].get("claim/right_camera_d405")
        assert fa.rgb.shape == (360, 640, 3) and np.array_equal(fa.rgb, fb.rgb)
        assert np.allclose(fa.intrinsics, fb.intrinsics)
        st = a.station
        snap = st._snapshot()
        now1 = st.observe(["right_camera_d405"]).frames["right_camera_d405"].rgb
        st._restore(snap)
        now2 = st.observe(["right_camera_d405"]).frames["right_camera_d405"].rgb
        assert np.array_equal(now1, now2)
        assert a.station._frame_info["right_camera_d405"]["exposure_s"] > 0
    finally:
        a.close()
        b.close()


def test_the_easy_model_has_no_hard_parts(mujoco_ok):
    from avail.station.mj_plug import scene

    model, lay = scene.build_model("fric", "easy")
    names = {model.geom(i).name for i in range(model.ngeom)}
    assert "jam_stop" not in names and not any(n.startswith("plug_cable_h") for n in names) and "plug_cable" in names
    hmodel, hlay = scene.build_model("fric", "hard")
    hnames = {hmodel.geom(i).name for i in range(hmodel.ngeom)}
    assert "jam_stop" in hnames and "plug_cable" not in hnames and lay.profile == "easy" and hlay.profile == "hard"


@pytest.mark.parametrize("connector", ["fric", "latch"])
def test_a_jam_holds_the_plug_short_and_a_push_cannot_pass_it(mujoco_ok, connector):
    from avail.evidence import base
    from avail.zoo.harness import prepare

    spec = ErrorSpec("N2", shortfall_mm=1.5, jam_kind="burr")
    p = prepare(DEV, connector, spec=spec, profile="hard")
    try:
        assert p.outcome.flag and p.outcome.landed_class == "N2"
        assert abs(p.outcome.truth["metrics"]["shortfall_mm"] - 1.5) < 0.3
        base.get("probe.push_seat").collect(p.view, p.claim, p.fresh_params())
        assert not p.label()
        assert abs(p.truth()["metrics"]["shortfall_mm"] - 1.5) < 0.3
    finally:
        p.close()


def test_a_burr_in_the_detent_push_out_zone_keeps_its_n1_plug_at_the_stop(mujoco_ok):
    """Review finding: without the arm holding at the stop, the latch detent pushed burr N1 plugs back to N2 depth."""
    from avail.zoo.harness import prepare

    p = prepare(6001, "latch", spec=ErrorSpec("N1", shortfall_mm=0.743, jam_kind="burr"), profile="hard")
    try:
        assert p.outcome.landed_class == "N1"
        assert abs(p.outcome.truth["metrics"]["shortfall_mm"] - 0.743) < 0.15
    finally:
        p.close()


def test_a_burr_lets_go_of_a_pull_and_a_strong_wedge_holds_it(mujoco_ok):
    from avail.evidence import base
    from avail.zoo.harness import prepare

    exp = config.load_experiment()
    cap = exp.tug_force_cap_n("fric")
    slips = {}
    for kind, hold in (("burr", None), ("wedge", 2.0 * cap)):
        p = prepare(DEV, "fric", spec=ErrorSpec("N2", shortfall_mm=1.5, jam_kind=kind, wedge_hold_n=hold), profile="hard")
        try:
            ev = base.get("probe.tug").collect(p.view, p.claim, p.fresh_params())
            slips[kind] = ev.features.get("slipped")
        finally:
            p.close()
    assert slips["burr"] == 1.0 and slips["wedge"] == 0.0


def test_the_lot_factor_scales_the_detent(mujoco_ok):
    from avail.station.mj_plug.station import MjPlugStation

    st = MjPlugStation(DEV, "latch", profile="hard")
    try:
        factor = st._hard.lot_factor
        spec = config.load_experiment().connector("latch")
        assert 0.7 <= factor <= 1.3
        assert st._retention.detent_engaged.peak_positive == pytest.approx(factor * max(v for _, v in spec.detent_n))
    finally:
        st.close()


def test_the_play_holds_the_estimate_until_the_force_turns():
    from avail.station.mj_plug.station import play_operator

    h = config.load_experiment().hard().force.hysteresis_n
    state = np.zeros(3)
    # the output moves only when the input leaves the band of +-h around it
    for x, want in ((0.5 * h, 0.0), (2.0 * h, 1.0 * h), (1.5 * h, 1.0 * h), (-1.5 * h, -0.5 * h)):
        state = play_operator(np.array([x, 0.0, 0.0]), state, h)
        assert state[0] == pytest.approx(want)


def test_the_demo_reads_both_shipped_witnesses_and_the_reported_numbers():
    """The demo's hard-mode half needs files that ship with the repo. Light: no simulation."""
    from avail import demo
    from avail.witness import store

    rep = demo._reported()
    assert rep is not None and set(rep["connectors"]) == {"fric", "latch"}
    for connector in ("fric", "latch"):
        assert {"easy", "hard"} <= set(rep["connectors"][connector])
        easy, _ = store.load_shipped(connector)
        hard, payload = store.load_shipped(f"{connector}_hard")
        assert hard.station_id == "mj_plug_hard" and easy.station_id == "mj_plug"
        assert payload["provenance"].get("profile") == "hard"
        html = demo._reported_html(rep, connector)
        assert "hard" in html and "easy" in html
