"""S4c hard mode v2: the servo's joint and current readings on top of hard mode.

Station tests build dev seeds only. Hard mode v1 must not change: a v1 station reads its joints
and torques exactly, as in S4b.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from avail import config, seeds
from avail.station.mj_plug.inject import ARM, ErrorSpec

DEV = 6500                              # a dev seed for hard-mode v2 station tests


# ----------------------------------------------------------------------------- config and seeds
def test_v2_is_hard_mode_plus_the_servo_readings():
    exp = config.load_experiment()
    hard, hard2 = exp.hard(), exp.hard("hard2")
    assert hard2.base == "hard" and hard.joint_readings is None and hard.current_readings is None
    for section in ("wrist_camera", "plug_cable", "jams", "force"):
        assert getattr(hard2, section) == getattr(hard, section)
    j, g, c = hard2.joint_readings, hard2.gear_lash, hard2.current_readings
    assert j.counts_per_turn == 4096 and tuple(g.lash_deg) == (0.5, 1.0) and g.motor_speed_rad_s == pytest.approx(4.72)
    assert c.count_ma == 6.5 and c.stall_current_a == 2.7 and c.noise_counts_sd >= 0
    assert hard.gear_lash is None


def test_v2_ranges_are_their_own_and_the_test_range_is_locked(tmp_path):
    budget = config.load_experiment().seeds
    assert budget.split_of(66000) == "zoo_fit_hard2" and budget.split_of(69999) == "zoo_calibration_hard2"
    assert budget.split_of(70000) == "zoo_test_hard2" and "zoo_test_hard2" in config.TEST_SPLITS
    assert seeds.FINAL_RUNS["E1cE2c"] == "zoo_test_hard2"
    with pytest.raises(seeds.TestSeedError):
        seeds.assert_simulable(70000)
    with pytest.raises(seeds.TestSeedError):          # the v1 final run does not open the v2 test range
        seeds.guard([70000], purpose="unit test", final_run="E1bE2b", record_path=tmp_path / "reads.jsonl")
    with pytest.raises(seeds.TestSeedError):          # and the v2 final run opens nothing else
        seeds.guard([64000], purpose="unit test", final_run="E1cE2c", record_path=tmp_path / "reads.jsonl")
    assert not (tmp_path / "reads.jsonl").exists()


def test_a_stored_read_of_a_read_split_passes_a_changed_config_and_is_recorded(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(seeds, "RETIRED_FINAL_RUNS", {})      # E1bE2b is retired now. This tests the guard as it was
    record = tmp_path / "reads.jsonl"
    record.write_text(json.dumps({"split": "zoo_test_hard", "config_hash": "000000000000", "final_run": "E1bE2b"}) + "\n",
                      encoding="utf-8")
    with pytest.raises(seeds.TestSeedError):          # simulating it again under a new config stays refused
        seeds.guard([64000], purpose="unit test", final_run="E1bE2b", record_path=record)
    seeds.guard([64000], purpose="unit test, stored frames", final_run="E1bE2b", record_path=record, stored=True)
    last = json.loads(record.read_text(encoding="utf-8").splitlines()[-1])
    assert last["stored"] is True and last["split"] == "zoo_test_hard"
    with pytest.raises(seeds.TestSeedError):          # a stored read still needs the final run's name
        seeds.guard([64000], purpose="unit test", record_path=record, stored=True)
    with pytest.raises(seeds.TestSeedError):          # and never opens simulation
        seeds.assert_simulable(64000)


def test_a_v2_seed_is_built_in_v2_only():
    from avail.zoo import store

    assert seeds.profile_of_seed(66000) == "hard2" and seeds.profile_of_seed(60000) == "hard"
    with pytest.raises(seeds.TestSeedError):
        seeds.assert_profile(66000, "hard")
    with pytest.raises(seeds.TestSeedError):
        seeds.assert_profile(60000, "hard2")
    seeds.assert_profile(DEV, "hard2")
    assert store.profile_of_split("fit_hard2") == "hard2" and store.profile_of_split("dev_hard2") == "hard2"
    assert store.profile_of_split("test_hard") == "hard"


def test_a_profile_cannot_name_a_base_that_has_a_base():
    raw = config._read_yaml(config.EXPERIMENT_YAML)
    raw["profiles"]["hard3"] = {"base": "hard2"}
    with pytest.raises(Exception):
        config.ExperimentConfig.model_validate(raw)


# ----------------------------------------------------------------------------- the v2 station (dev seeds)
@pytest.fixture(scope="module")
def mujoco_ok():
    pytest.importorskip("mujoco")
    from avail.station.mj_plug import aspire

    try:
        aspire.require_station_files()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    return True


def test_v1_reads_exactly_and_v2_reads_on_the_count_grid(mujoco_ok):
    from avail.station.mj_plug.station import MjPlugStation

    v1 = MjPlugStation(DEV, "latch", profile="hard")
    v2 = MjPlugStation(DEV, "latch", profile="hard2")
    try:
        assert not v1._sense_on and not v1._cur_on and not v1._lash_on
        assert np.array_equal(v1.robot_state().arms[ARM].joint_pos, v1._q(ARM))
        assert np.array_equal(v1._motor_torque_read(ARM), v1._motor_torque(ARM))
        assert v2._sense_on and v2._cur_on and v2._lash_on
        q, read = v2._q(ARM), v2._q_read(ARM)
        step = 2 * math.pi / 4096
        counts = read / step + v2._enc_phase[ARM]
        assert np.allclose(counts, np.round(counts), atol=1e-6)          # on the grid
        assert np.all(np.abs(read - q) <= step / 2 + 1e-12)                # the encoder reads the output: within half a count
        weight = v2._mj_data.qfrc_gravcomp[v2._arms[ARM].ids["dadr"]]
        unit = v2._tau_peak[ARM] * v2._cur_count
        torque = (v2._motor_torque_read(ARM) + weight) / unit
        assert np.allclose(torque, np.round(torque), atol=1e-6)
    finally:
        v1.close()
        v2.close()


def test_a_servo_whose_torque_turns_passes_none_until_its_motor_crosses_the_lash(mujoco_ok):
    from avail.station.mj_plug.station import MjPlugStation

    st = MjPlugStation(DEV, "fric", profile="hard2")
    try:
        ids, data = st._arms[ARM].ids, st._mj_data
        st._mj.mj_forward(st._mj_model, data)
        j = 0                                                           # the base joint carries no weight
        half = 0.5 * st._lash[ARM][j]
        st._gap[ARM][j] = -half                                         # gears touching on the minus side
        st._ctrl_cmd[ARM][j] = data.qpos[ids["qadr"][j]] + 0.05          # the servo now pushes plus
        q, qd = data.qpos[ids["qadr"][j]], data.qvel[ids["dadr"][j]]
        free = q + ids["kd"][j] * qd / ids["kp"][j]
        st._apply_lash()
        assert data.ctrl[ids["act"][j]] == pytest.approx(free)          # no torque while the motor crosses
        assert st._gap[ARM][j] == pytest.approx(-half + st._lash_speed * st._mj_model.opt.timestep)
        steps = int(np.ceil(2 * half / (st._lash_speed * st._mj_model.opt.timestep))) + 1
        for _ in range(steps):
            st._apply_lash()
        assert st._gap[ARM][j] == pytest.approx(half)
        assert data.ctrl[ids["act"][j]] == pytest.approx(st._ctrl_cmd[ARM][j])   # touching again: the target holds
    finally:
        st.close()


def test_v2_insertions_are_exact_repeats_and_survive_a_snapshot(mujoco_ok):
    from avail.zoo.harness import prepare

    a = prepare(DEV, "latch", "N1", profile="hard2", render=False)
    b = prepare(DEV, "latch", "N1", profile="hard2", render=False)
    try:
        ta, tb = a.outcome.insert.trace, b.outcome.insert.trace
        assert ta["travel_mm"] == tb["travel_mm"] and ta["force_n"] == tb["force_n"]
        travel = np.asarray(ta["travel_mm"])
        assert len(np.unique(np.round(travel, 9))) < 0.5 * len(travel)    # a reading moves in counts
        st = a.station
        snap = st._snapshot()
        r1 = st.move_relative(ARM, [0, 0, -0.002], "base", 0.003, 3.0, 0.002)
        st._restore(snap)
        r2 = st.move_relative(ARM, [0, 0, -0.002], "base", 0.003, 3.0, 0.002)
        assert r1.trace["travel_mm"] == r2.trace["travel_mm"] and r1.trace["force_n"] == r2.trace["force_n"]
    finally:
        a.close()
        b.close()


def test_v2_truth_ignores_the_readings(mujoco_ok):
    from avail.zoo.harness import prepare

    p = prepare(DEV, "fric", spec=ErrorSpec("N2", shortfall_mm=1.5), profile="hard2", render=False)
    try:
        before = p.truth()
        pose_before = p.station.robot_state().arms[ARM].tool_pose.pos
        p.station._enc_phase[ARM] = (p.station._enc_phase[ARM] + 0.5) % 1.0      # the count grid moved half a count
        assert p.truth() == before
        assert np.linalg.norm(np.subtract(p.station.robot_state().arms[ARM].tool_pose.pos, pose_before)) > 1e-5
    finally:
        p.close()
