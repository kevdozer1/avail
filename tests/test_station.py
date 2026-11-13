"""The plug station: retention tables, scene determinism, the Station protocol, injection and truth.

The acceptance checks of docs/02_plug_station.md run on 50 dev seeds through
``avail station accept``. The tests here pin the same behavior on a few fixed dev seeds so a
change that breaks the station fails the suite within a minute. All seeds are dev seeds.
"""

from __future__ import annotations

import dataclasses
import hashlib
import subprocess
import sys

import numpy as np
import pytest

from avail.config import load_experiment
from avail.paths import REPO_ROOT, aspire_station_xml

pytestmark = pytest.mark.skipif(not aspire_station_xml().is_file(), reason="third_party/ASPIRE is not cloned")

CONNECTORS = ("fric", "latch")


# ------------------------------------------------------------------ retention tables
@pytest.mark.parametrize("connector", CONNECTORS)
def test_force_tables_give_the_configured_peaks(connector):
    from avail.station.mj_plug.retention import Table

    spec = load_experiment().connector(connector)
    friction = Table(spec.axial_friction_n, hold_ends=True)(spec.full_depth_mm)
    detent = Table(spec.detent_n)
    assert friction + detent.peak_negative == pytest.approx(spec.insertion_peak_n, abs=0.05)
    assert friction + detent.peak_positive == pytest.approx(spec.retention_seated_n, abs=0.05)
    lo, hi = spec.retention_2mm_short_n
    two_short = Table(spec.axial_friction_n, hold_ends=True)(spec.full_depth_mm - 2.0)
    assert lo <= two_short <= hi


def test_blocked_latch_holds_with_about_1_newton():
    from avail.station.mj_plug.retention import Table

    spec = load_experiment().connector("latch")
    friction = Table(spec.axial_friction_n, hold_ends=True)(spec.full_depth_mm)
    blocked = Table(spec.detent_latch_blocked_n)
    assert friction + blocked.peak_positive == pytest.approx(spec.retention_latch_blocked_n, abs=0.05)
    assert blocked.peak_negative == 0.0                      # no barrier, so no click on the way in


def test_table_is_piecewise_linear_and_zero_outside():
    from avail.station.mj_plug.retention import Table

    t = Table([(0.0, 0.0), (1.0, 2.0), (2.0, -4.0)])
    assert t(0.5) == pytest.approx(1.0)
    assert t(1.5) == pytest.approx(-1.0)
    assert t(-1.0) == 0.0 and t(3.0) == 0.0
    assert Table([(0.0, 1.0), (1.0, 2.0)], hold_ends=True)(5.0) == 2.0


# ------------------------------------------------------------------ scene
def test_scene_sample_is_a_pure_function_of_seed_and_connector():
    from avail.station.mj_plug.scene import sample_scene

    a, b = sample_scene(42, "fric"), sample_scene(42, "fric")
    assert a == b
    assert sample_scene(43, "fric") != a
    assert sample_scene(42, "latch") != dataclasses.replace(a, connector="latch")


def test_scene_sample_stays_inside_the_configured_ranges():
    from avail.station.mj_plug.scene import N_PORTS, sample_scene

    rnd = load_experiment().station.randomization
    for seed in range(60):
        cfg = sample_scene(seed, "latch")
        assert all(abs(v) <= rnd.board_xy_mm / 1000 + 1e-9 for v in cfg.board_xy)
        assert abs(np.degrees(cfg.board_yaw)) <= rnd.board_yaw_deg + 1e-6
        assert 1 <= cfg.target_port <= N_PORTS
        assert rnd.lights[0] <= sum(l[0] for l in cfg.lights) <= rnd.lights[1]
        assert rnd.distractor_cables[0] <= sum(c[0] for c in cfg.cables) <= rnd.distractor_cables[1]
        for jitter in cfg.camera_jitter:
            assert all(abs(v) <= rnd.camera_xyz_mm / 1000 + 1e-9 for v in jitter[:3])
            assert all(abs(np.degrees(v)) <= rnd.camera_rot_deg + 1e-6 for v in jitter[3:])
        assert cfg.grasp_offset[1] == 0.0                   # a parallel gripper centers the plug


def test_applying_a_scene_does_not_depend_on_what_was_applied_before():
    from avail.station.mj_plug import scene

    model, lay = scene.build_model("fric")
    names = ("body_pos", "body_quat", "cam_pos", "cam_quat", "light_pos", "light_active", "geom_matid", "mat_rgba")

    def snapshot(seed):
        scene.apply_scene(model, lay, scene.sample_scene(seed, "fric"))
        return {n: getattr(model, n).copy() for n in names}

    first = snapshot(11)
    snapshot(12)
    again = snapshot(11)
    for n in names:
        assert np.array_equal(first[n], again[n]), n


def test_no_light_casts_a_shadow():
    """Shadows made identical renders differ by 1 level on this GPU, which breaks the byte-for-byte rule."""
    from avail.station.mj_plug import scene

    model, _ = scene.build_model("fric")
    assert not model.light_castshadow.any()


# ------------------------------------------------------------------ the Station protocol
@pytest.fixture(scope="module")
def fresh_station():
    from avail.station.mj_plug.station import MjPlugStation

    station = MjPlugStation(7, "fric")
    yield station
    station.close()


def test_station_satisfies_the_protocol(fresh_station):
    from avail.station.base import Station

    assert isinstance(fresh_station, Station)
    assert fresh_station.station_id == "mj_plug"


def test_observation_has_3_frames_with_calibration_and_no_object_pose(fresh_station):
    from avail.station.base import ArmState, CameraFrame, Observation, RobotState

    obs = fresh_station.observe()
    assert sorted(obs.frames) == sorted(load_experiment().station.cameras)
    for frame in obs.frames.values():
        assert frame.rgb.shape == (480, 640, 3) and frame.rgb.dtype == np.uint8
        assert frame.intrinsics.shape == (3, 3) and frame.extrinsics.shape == (4, 4)
        assert np.allclose(frame.extrinsics[:3, :3] @ frame.extrinsics[:3, :3].T, np.eye(3), atol=1e-6)
    # CLAUDE.md rule 10: neither record may carry an object pose
    banned = ("plug", "object", "port", "socket", "truth", "depth", "shortfall")
    for cls in (Observation, CameraFrame, RobotState, ArmState):
        for f in dataclasses.fields(cls):
            assert not any(word in f.name for word in banned), f"{cls.__name__}.{f.name}"
    with pytest.raises(KeyError):
        fresh_station.observe(["no_such_camera"])


def test_robot_state_reports_both_arms(fresh_station):
    state = fresh_station.robot_state()
    assert set(state.arms) == {"left", "right"}
    arm = state.arms["right"]
    assert arm.joint_pos.shape == (6,) and arm.joint_vel.shape == (6,) and arm.joint_torque_est.shape == (6,)
    assert 0.0 <= arm.gripper_pos <= 1.0
    assert arm.tool_force_est.shape == (3,)
    assert np.isfinite(arm.tool_force_est).all()


def test_guarded_move_rejects_bad_arguments(fresh_station):
    with pytest.raises(ValueError):
        fresh_station.move_relative("right", [0, 0, 0], "base", 0.005, 5.0, 0.005)
    with pytest.raises(ValueError):
        fresh_station.move_relative("right", [0, 0, -0.01], "elsewhere", 0.005, 5.0, 0.005)
    with pytest.raises(ValueError):
        fresh_station.move_relative("right", [0, 0, -0.01], "base", 0.005, -1.0, 0.005)


def test_nominal_geometry_carries_the_localization_error_and_no_more(fresh_station):
    from avail.station.mj_plug import scene

    nominal = fresh_station.nominal()
    cfg, lay = fresh_station._cfg, fresh_station._lay
    sigma = load_experiment().station.noise.port_localization_sigma_mm / 1000
    assert nominal.board_yaw == pytest.approx(cfg.board_yaw + cfg.yaw_error)
    assert abs(cfg.yaw_error) <= 3 * sigma / 0.1 + 1e-9
    for port in (1, 8):
        true_pos, _ = scene.port_pose_world(cfg, lay, port)
        err = nominal.port_position(port) - true_pos
        # the board center is off by the localization error and the yaw error swings the far ports a little more
        assert np.linalg.norm(err[:2] - np.array(cfg.localization_error[:2])) <= abs(cfg.yaw_error) * 0.07 + 1e-9
        assert err[2] == pytest.approx(cfg.localization_error[2])
    assert nominal.nearest_port(nominal.port_position(5)[:2])[0] == 5
    seated = nominal.seated_tool_pose(3)
    assert seated.rotation()[2, 2] == pytest.approx(-1.0)   # the tool points down


# ------------------------------------------------------------------ determinism
def _state_hash(seed: int, connector: str) -> str:
    from avail.station.mj_plug.inject import sample_error, scripted_insert

    station, outcome = scripted_insert(seed, connector, sample_error(seed, connector, "N2"))
    obs = station.observe()
    h = hashlib.sha256()
    h.update(station._mj_data.qpos.tobytes())
    h.update(station._mj_data.qvel.tobytes())
    for name in sorted(obs.frames):
        h.update(obs.frames[name].rgb.tobytes())
    h.update(np.array(outcome.insert.trace["force_n"]).tobytes())
    station.close()
    return h.hexdigest()


def test_same_seed_gives_the_same_state_and_the_same_pixels_byte_for_byte():
    a = _state_hash(31, "fric")
    _state_hash(32, "fric")                                  # something else in between
    b = _state_hash(31, "fric")
    assert a == b


def test_same_seed_gives_the_same_bytes_in_another_process():
    here = _state_hash(33, "latch")
    code = "from tests.test_station import _state_hash; print(_state_hash(33, 'latch'))"
    done = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().splitlines()[-1] == here


def test_first_observation_is_identical_for_the_same_seed():
    from avail.station.mj_plug.station import MjPlugStation

    frames = []
    for _ in range(2):
        station = MjPlugStation(55, "latch")
        obs = station.observe()
        frames.append(b"".join(obs.frames[k].rgb.tobytes() for k in sorted(obs.frames)))
        station.close()
    assert frames[0] == frames[1]


def test_snapshot_and_restore_replay_a_probe_exactly():
    from avail.station.mj_plug.inject import ARM, sample_error, scripted_insert

    station, _ = scripted_insert(34, "latch", sample_error(34, "latch", "N2"))
    snap = station._snapshot()

    def probe():
        r = station.move_relative(ARM, [0, 0, -0.006], "base", 0.005, 9.0, 0.006)
        return r.travel_m, r.stopped_by, r.force_at_stop_n, station._mj_data.qpos.tobytes()

    first = probe()
    station._restore(snap)
    second = probe()
    station.close()
    assert first == second


# ------------------------------------------------------------------ injection and truth
@pytest.mark.parametrize("connector,cls", [("fric", "P0"), ("fric", "N1"), ("fric", "N2"), ("latch", "N3"),
                                           ("latch", "N5"), ("latch", "N6"), ("fric", "N7"), ("latch", "N8")])
def test_injected_class_lands_in_its_class(connector, cls):
    from avail.station.mj_plug.inject import sample_error, scripted_insert

    station, outcome = scripted_insert(21, connector, sample_error(21, connector, cls))
    station.close()
    assert outcome.landed_class == cls
    primary = "latched" if connector == "latch" else "seated"
    assert outcome.truth["predicates"][primary] is (cls == "P0")
    assert outcome.claim_predicate == primary


def test_a_short_plug_fools_the_skill_and_the_arm_is_left_holding_it():
    from avail.station.mj_plug.inject import ErrorSpec, scripted_insert

    station, outcome = scripted_insert(22, "fric", ErrorSpec("N2", shortfall_mm=1.5))
    station.close()
    m = outcome.truth["metrics"]
    assert outcome.flag is True                              # the skill says done
    assert outcome.insert.stopped_by == "force"              # its force threshold tripped on the stuck joint
    assert 1.0 <= m["shortfall_mm"] < 2.0                    # and the plug is short
    assert outcome.truth["predicates"]["held"] is True
    assert outcome.truth["predicates"]["seated"] is False


def test_wrong_port_is_seated_somewhere_else():
    from avail.station.mj_plug.inject import sample_error, scripted_insert

    spec = sample_error(23, "fric", "N6")
    station, outcome = scripted_insert(23, "fric", spec)
    station.close()
    m = outcome.truth["metrics"]
    assert m["nearest_port"] == spec.wrong_port != outcome.target_port
    assert outcome.truth["predicates"]["at_port"] is False
    assert outcome.truth["predicates"]["seated"] is False


def test_released_twin_and_not_released_differ_only_in_the_release():
    from avail.station.mj_plug.inject import ErrorSpec, scripted_insert

    station, good = scripted_insert(24, "latch", ErrorSpec("P0", release=True))
    station.close()
    station, bad = scripted_insert(24, "latch", ErrorSpec("N9"))
    station.close()
    assert good.claim_predicate == bad.claim_predicate == "released_stable"
    assert good.truth["predicates"]["released_stable"] is True
    assert bad.truth["predicates"]["released_stable"] is False
    assert bad.truth["predicates"]["latched"] is True        # the plug itself is fine, the fingers never opened
    assert good.landed_class == "P0" and bad.landed_class == "N9"


def test_class_mix_matches_the_config():
    from avail.station.mj_plug.inject import class_mix

    zoo = load_experiment().zoo
    for connector in CONNECTORS:
        mix = class_mix(connector)
        assert sum(mix.values()) == pytest.approx(1.0)
        assert mix["P0"] == pytest.approx(zoo.positive_share)
        near = sum(v for k, v in mix.items() if k in ("N1", "N2", "N3", "N5"))
        assert near == pytest.approx(zoo.near_miss_share_of_negatives * (1 - zoo.positive_share))
    assert "N5" not in class_mix("fric") and "N5" in class_mix("latch")


def test_sampled_classes_follow_the_mix():
    from avail.station.mj_plug.inject import class_mix, sample_error

    draws = [sample_error(seed, "latch").cls for seed in range(3000)]
    mix = class_mix("latch")
    for cls, share in mix.items():
        got = draws.count(cls) / len(draws)
        assert abs(got - share) < 4 * (share * (1 - share) / len(draws)) ** 0.5 + 0.004, cls


# ------------------------------------------------------------------ the gate, on a few seeds
@pytest.mark.slow
@pytest.mark.parametrize("connector", CONNECTORS)
def test_oracle_cycle_seats_and_do_nothing_does_not(connector):
    from avail.station.mj_plug.oracle import do_nothing, oracle_cycle

    primary = "latched" if connector == "latch" else "seated"
    station, res = oracle_cycle(201, connector)
    station.close()
    assert res["truth_held"]["predicates"][primary] is True
    assert res["truth_released"]["predicates"]["released_stable"] is True
    station, idle = do_nothing(201, connector)
    station.close()
    assert idle["truth"]["predicates"][primary] is False


@pytest.mark.slow
@pytest.mark.parametrize("connector", CONNECTORS)
def test_seated_plug_holds_a_pull_and_a_short_one_is_repaired_by_a_capped_push(connector):
    from avail.station.mj_plug import accept

    held = accept.job(("seated_pull", connector, 202))
    force_n, limit_mm = accept.SEATED_PULL[connector]
    assert held["landed"] == "P0"
    assert held["disp_at_force_mm"] is not None and held["disp_at_force_mm"] < limit_mm
    assert held["still_ok"] is True
    peak = load_experiment().connector(connector).insertion_peak_n
    assert abs(held["true_peak_n"] - peak) <= 0.2 * peak
    assert held["true_peak_n"] - held["true_min_after_n"] > 0.3 * peak       # the click: a peak, then a drop
    pushed = accept.job(("push_repair", connector, 202))
    assert pushed["landed"] == "N2" and pushed["seated_after"] is True
    cap = load_experiment().push_force_cap_n(connector)
    assert pushed["true_peak_push_n"] <= cap + 2.5          # the real force may pass the cap by the noise, no more


@pytest.mark.slow
def test_latch_two_millimeters_short_comes_out_under_a_2_newton_pull():
    from avail.station.mj_plug import accept

    res = accept.job(("short_pull", "latch", 203))
    assert res["plug_moved_mm"] > accept.SHORT_PULL_MOVE_MM
    assert res["max_true_pull_n"] <= accept.SHORT_PULL_N


def test_the_station_refuses_a_test_seed(monkeypatch):
    """CLAUDE.md rule 12, at the lowest level: no code path can simulate a test seed before its final run."""
    from avail import seeds
    from avail.station.mj_plug.station import MjPlugStation

    monkeypatch.delenv(seeds.FINAL_RUN_ENV, raising=False)
    with pytest.raises(seeds.TestSeedError):
        MjPlugStation(16000, "fric")
    with pytest.raises(seeds.TestSeedError):
        MjPlugStation(30005, "latch")


# ------------------------------------------------------------------ regressions from the review of 2026-11-11
def test_a_snapshot_replays_the_live_state_exactly():
    """A probe from the live state and the same probe after a restore give the same bytes."""
    from avail.station.mj_plug.inject import ARM, sample_error, scripted_insert

    station, _ = scripted_insert(41, "latch", sample_error(41, "latch", "P0"))
    snap = station._snapshot()

    def probe():
        r = station.move_relative(ARM, [0, 0, -0.006], "base", 0.005, 9.0, 0.006)
        return r.travel_m, r.stopped_by, station._mj_data.qpos.tobytes(), station.read_signal("link")["link_up"]

    live = probe()
    station._restore(snap)
    replay = probe()
    station.close()
    assert live == replay


def test_a_frame_depends_on_the_camera_and_the_state_only():
    """The renderer's output lagged a camera switch by 1 frame. Each camera is now rendered twice."""
    from avail.station.mj_plug.station import MjPlugStation

    station = MjPlugStation(42, "fric")
    a = station.observe(["top_camera_d405", "left_camera_d405", "right_camera_d405"])
    b = station.observe(["right_camera_d405", "top_camera_d405", "left_camera_d405"])
    c = station.observe()
    station.close()
    for cam in a.frames:
        assert np.array_equal(a.frames[cam].rgb, b.frames[cam].rgb), cam
        assert np.array_equal(a.frames[cam].rgb, c.frames[cam].rgb), cam


def test_two_live_stations_do_not_share_a_model():
    from avail.station.mj_plug.station import MjPlugStation

    a = MjPlugStation(43, "fric")
    board = a._lay.ids["body:board"]
    before = a._mj_model.body_pos[board].copy()
    b = MjPlugStation(44, "fric")
    after = a._mj_model.body_pos[board].copy()
    assert a._mj_model is not b._mj_model
    assert np.array_equal(before, after)
    a.close()
    b.close()


def _insert_job(args):
    seed, connector, cls = args
    from avail.station.mj_plug.inject import ARM, sample_error, scripted_insert

    station, out = scripted_insert(seed, connector, sample_error(seed, connector, cls))
    state = station.robot_state().arms[ARM]
    rot = state.tool_pose.rotation()
    row = {"cls": cls, "short": out.truth["metrics"]["shortfall_mm"], "travel": out.insert.travel_m * 1000.0,
           "plug_tilt": out.truth["metrics"]["tilt_deg"], "tool_tilt": float(np.degrees(np.arccos(np.clip(-rot[2, 2], -1, 1)))),
           "left_qd": float(np.linalg.norm(station._qd("left"))), "jaw": state.gripper_pos}
    station.close()
    return row


@pytest.mark.slow
def test_the_insertion_record_does_not_carry_the_shortfall():
    """Regression: the descent started at the true socket height, so travel gave the shortfall to 0.09 mm.

    It now starts at the taught height, which carries this episode's localization error, so
    travel says no more than the taught geometry allows.
    """
    from concurrent.futures import ProcessPoolExecutor

    jobs = [(s, "fric", c) for s in range(3800, 3812) for c in ("P0", "N1", "N2", "N3")]
    with ProcessPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(_insert_job, jobs))
    short = np.array([r["short"] for r in rows])
    travel = np.array([r["travel"] for r in rows])
    assert np.std(travel + short) > 0.4                      # it was 0.088 mm
    assert abs(np.corrcoef(travel, short)[0, 1]) < 0.95


@pytest.mark.slow
def test_a_tilted_jam_is_cocked_in_the_fingers_with_the_tool_straight():
    """Regression: N7 tilted the whole hand, so the robot's own pose showed the tilt exactly."""
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(_insert_job, [(s, "latch", "N7") for s in range(3820, 3826)]))
    assert all(r["tool_tilt"] < 1.0 for r in rows)
    assert all(r["plug_tilt"] > 2.0 for r in rows)


def test_the_free_arm_is_at_rest_and_the_gripper_reads_at_its_resolution():
    """Regression: the free arm sagged and recovered after reset, so its velocity carried the time
    since reset. And the gripper read micrometers, which a real encoder does not resolve."""
    row = _insert_job((3830, "latch", "N2"))
    assert row["left_qd"] < 1e-6
    res = load_experiment().station.noise.gripper_position_resolution_mm / (0.0375 * 1000.0)
    assert abs(row["jaw"] / res - round(row["jaw"] / res)) < 1e-6


def test_a_travel_stop_brakes_at_the_cap():
    """Regression: a pull with nothing holding the tool back coasted 0.5 mm past its travel cap."""
    from avail.station.mj_plug.inject import ARM, sample_error, scripted_insert

    station, _ = scripted_insert(3831, "latch", sample_error(3831, "latch", "N4"))
    p0 = station._fk(ARM, station._q(ARM))[0]
    farthest = [0.0]

    def hook():
        farthest[0] = max(farthest[0], float(station._fk(ARM, station._q(ARM))[0][2] - p0[2]))

    station._tick_hook = hook
    r = station.move_relative(ARM, [0, 0, 0.003], "base", 0.003, 15.0, 0.003)
    station._tick_hook = None
    station.close()
    assert r.stopped_by == "travel"
    assert farthest[0] * 1000.0 <= 3.0 + 0.2
