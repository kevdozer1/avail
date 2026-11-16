"""S5: the cable loop testbed: its scene, its cell, its failure draws, its 3 new checks and its replay.

Shift dev seeds only (20,000 to 23,999). Light tests, as the change policy asks of a testbed: the
cell runs, a check reads what it should, a replay shows what ran.
"""

from __future__ import annotations

import numpy as np
import pytest

SEED = 20001                    # a shift dev seed whose first cable meets no failure draw


@pytest.fixture(scope="module")
def mujoco_ok():
    pytest.importorskip("mujoco")
    from avail.station.mj_plug import aspire

    try:
        aspire.require_station_files()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    return True


@pytest.fixture(scope="module")
def latch_witnesses(mujoco_ok):
    """The cable loop's witnesses for latch. They are built from the cable zoo's tables under AVAIL_DATA, so a fresh
    clone without that data skips the tests that need them."""
    from avail.core.task import load_task
    from avail.station.mj_plug import cable_shift

    try:
        return cable_shift.make_witnesses(load_task("cable_loop"), "latch", 0.01)
    except FileNotFoundError as exc:
        pytest.skip(f"needs the cable zoo's tables under AVAIL_DATA: {exc}")


def _claim(pred, item, port):
    from avail.schemas import Claim, ClaimContext, Predicate

    return Claim(claim_id="c", run_id="test", cycle_id="c0", step_id=f"{item}.x", raised_by="skill",
                 predicate=Predicate(name=pred, target={"item": item, "port": port}, definition="test"),
                 context=ClaimContext(task="cable_loop", stage="test", attempt=0))


def _ev(cell, source_id, pred, item, step):
    from avail.evidence import base

    return base.get(source_id).collect(cell.view, _claim(pred, item, cell.port_of(item)), cell.claim_params(item, step))


def test_the_scene_has_a_rack_and_5_visual_cables(mujoco_ok):
    from avail.station.mj_plug.cable_scene import N_CABLES, build_cable_model

    model, lay = build_cable_model("latch")
    for k in range(N_CABLES):
        assert f"body:vcable{k}" in lay.ids
        assert model.geom_contype[model.geom(f"vcable{k}_body").id] == 0      # a visual cable never collides
    assert "body:cable_rack" in lay.ids and model.geom_group[model.geom("tray_floor").id] == 3


def test_failure_draws_are_a_function_of_seed_item_step_and_attempt(mujoco_ok):
    from avail.station.mj_plug.cable_cell import CableCell

    a, b = CableCell(SEED, "latch"), CableCell(SEED, "latch")
    try:
        for item, step, attempt in ((0, "pick", 0), (3, "insert", 1), (4, "release", 0)):
            assert a._draw(item, step, attempt).random() == b._draw(item, step, attempt).random()
        assert a._draw(0, "pick", 0).random() != a._draw(0, "pick", 1).random()
        assert [c.port for c in a.cables] == [c.port for c in b.cables] and len({c.port for c in a.cables}) == 5
    finally:
        a.close()
        b.close()


def test_a_cable_goes_through_the_4_steps_and_the_checks_read_it(mujoco_ok):
    from avail.evidence import cable_tag, hover_pose, jaw_hold  # noqa: F401  they register
    from avail.station.mj_plug.cable_cell import CableCell

    cell = CableCell(SEED, "latch")
    try:
        r = cell.run_skill("pick", 0)
        assert r.failure is None and cell.truth_of(0, "held")
        jaw = _ev(cell, "passive.jaw_hold", "held", 0, "pick").features
        tag = _ev(cell, "vision.cable_tag", "held", 0, "pick").features
        assert jaw["in_band"] == 1.0 and jaw["closed_empty"] == 0.0
        assert tag["match"] == 1.0 and tag["mismatch"] == 0.0
        other = _ev(cell, "vision.cable_tag", "held", 2, "pick").features      # the claim names another cable
        assert other["mismatch"] == 1.0 and other["match"] == 0.0
        cell.run_skill("carry", 0)
        assert cell.truth_of(0, "at_port")
        hover = _ev(cell, "passive.hover_pose", "at_port", 0, "carry").features
        assert hover["port_match"] == 1.0 and hover["in_band"] == 1.0
        cell.run_skill("insert", 0)
        assert cell.truth_of(0, "latched")
        cell.run_skill("release", 0)
        assert cell.truth_of(0, "released_stable") and cell.item_success(0)
        jaw = _ev(cell, "passive.jaw_hold", "released_stable", 0, "release").features
        assert jaw["open"] == 1.0
        assert len(cell.frames) > 100 and {f["tag"] for f in cell.frames} == {"autonomous"}
    finally:
        cell.close()


def test_empty_jaws_read_as_empty(mujoco_ok):
    from avail.evidence import jaw_hold  # noqa: F401
    from avail.station.mj_plug.cable_cell import CableCell

    cell = CableCell(SEED, "fric")
    try:
        cell.run_skill("pick", 0)
        st = cell.station
        st.set_gripper("right", 1.0)
        st.wait(0.4)
        st.set_gripper("right", 0.0)
        st.wait(0.4)
        assert not cell.truth_of(0, "held")
        assert _ev(cell, "passive.jaw_hold", "held", 0, "pick").features["closed_empty"] == 1.0
    finally:
        cell.close()


def test_the_white_cable_in_dim_light_is_still_white(mujoco_ok):
    """Dev seed 20000 lights its scene dimly: its white cable read 0.54 from white in RGB, and the check saw no cable.
    The check compares directions in RGB, so the cable in the jaws reads as itself."""
    from avail.evidence import cable_tag  # noqa: F401
    from avail.station.mj_plug.cable_cell import CableCell

    cell = CableCell(20000, "latch")
    try:
        cell._pick_from_slot(4)
        tag = _ev(cell, "vision.cable_tag", "held", 4, "pick").features
        assert tag["expected_dist"] > 0.30                                      # far in RGB, as on the dev shift
        assert tag["present"] == 1.0 and tag["match"] == 1.0 and tag["expected_angle_deg"] < 5.0
        assert _ev(cell, "vision.cable_tag", "held", 3, "pick").features["mismatch"] == 1.0   # it is not the yellow one
    finally:
        cell.close()


def test_a_chosen_takeover_is_1_call_for_help_and_labels_the_state_it_found(latch_witnesses):
    """Dev seed 20007, latch, avail_simop: cable 4's insertion is rejected twice, the person chooses to take over
    and seats it. That is 1 call for help with 1 wait. The takeover's label is what the person found, a plug that
    was not latched, so it agrees with the evidence it is stored with, and the done gate takes their sign-off."""
    from avail.core.task import load_task
    from avail.shift.driver import CycleDriver
    from avail.station.mj_plug import cable_shift

    events = []

    class Ledger:
        def append(self, kind, payload):
            events.append((kind, payload))

    task = load_task("cable_loop")
    cell = cable_shift.make_cell(20007, "latch")
    try:
        driver = CycleDriver(task, cell, "avail_simop", witnesses=latch_witnesses,
                             human=cable_shift.make_human(cell, 20007), ledger=Ledger(),
                             evaluator=cable_shift.evaluator(cell))
        log = driver.run()
        truth = cable_shift.evaluate(cell)
    finally:
        cable_shift.close(cell)
    # cable 5's insertion. Since the S5c fix (2026-11-14) the calibrated release witness also calls on other cables
    ivs = [p for kind, p in events if kind == "intervention" and (p.item, p.step) == ("4", "insert")]
    assert [(iv.question, iv.outcome) for iv in ivs] == [("choose", "unresolved"), ("takeover", "taken_over_done")]
    assert ivs[0].seconds_waiting > 0.0                                               # 1 call for help, 1 wait
    assert ivs[1].truth_label is False and ivs[1].seconds_waiting == 0.0
    assert ivs[1].handoff_frame is not None and ivs[1].handback_frame > ivs[1].handoff_frame
    assert log.items[4].declared_done and truth[4] and all(truth.values())
    assist = next(p for kind, p in events if kind == "assist" and p.assist_id == ivs[1].assist_id)
    assert assist.question == "takeover" and assist.truth_label is True               # the sign-off after the handback


def test_the_person_fixes_a_blocked_latch(latch_witnesses):
    """Dev seed 20052, latch, avail_simop: cable 3's latch is blocked, the insertion is rejected twice, and the person
    takes over. They take the plug out, clear the latch and seat it. On dev1, when the latch was cleared with the
    plug still in, the person could not pull it out and missed the port in 5 of 8 such takeovers."""
    from avail.core.task import load_task
    from avail.shift.driver import CycleDriver
    from avail.station.mj_plug import cable_shift

    events = []

    class Ledger:
        def append(self, kind, payload):
            events.append((kind, payload))

    task = load_task("cable_loop")
    cell = cable_shift.make_cell(20052, "latch")
    try:
        CycleDriver(task, cell, "avail_simop", witnesses=latch_witnesses,
                    human=cable_shift.make_human(cell, 20052), ledger=Ledger(), evaluator=cable_shift.evaluator(cell)).run()
        truth = cable_shift.evaluate(cell)
    finally:
        cable_shift.close(cell)
    takeovers = [p for kind, p in events if kind == "intervention" and p.question == "takeover"]
    assert [(iv.item, iv.step, iv.outcome, iv.truth_label) for iv in takeovers] == [("3", "insert", "taken_over_done", False)]
    assert truth[3]


def test_a_replay_shows_what_ran(mujoco_ok):
    import pandas as pd

    from avail.station.mj_plug.cable_cell import CableCell
    from avail.station.mj_plug.cable_replay import Replay

    cell = CableCell(SEED, "latch")
    try:
        for step in ("pick", "carry"):
            cell.run_skill(step, 0)
        live = cell.station.observe(["right_camera_d405"]).frames["right_camera_d405"].rgb
        frames, scene = pd.DataFrame(cell.frames), list(cell.scene_events)
    finally:
        cell.close()
    rep = Replay(SEED, "latch", frames, scene)
    try:
        again = rep.render(len(frames) - 1, camera="right_camera_d405")
    finally:
        rep.close()
    assert float(np.abs(live.astype(int) - again.astype(int)).mean()) < 1.0
