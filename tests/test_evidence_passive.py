"""The passive evidence sources and the instrumented one: skill_flag, pose_residual, jaw_state, click, link.

One smoke test per source runs on real injected states of the simulated plug station. This file
is evaluator-side: it builds states with ``avail.zoo.harness.prepare`` and reads ground truth
to check what the sources said. The sources themselves never see it. Every seed is a dev seed
between 3000 and 4999.

The small tests at the top need no simulator. They pin the sign conventions and the click rule
on traces built by hand.
"""

from __future__ import annotations

import ast
import math
import sys
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pytest

from avail.evidence import instrumented, passive
from avail.evidence.base import REGISTRY, StationView
from avail.paths import REPO_ROOT, aspire_station_xml
from avail.schemas import EVIDENCE_KINDS, PREDICATES, Claim, ClaimContext, Cost, Evidence, Predicate
from avail.station.base import NominalGeometry, Pose
from avail.walls import scan_source

SOURCES = (passive.SKILL_FLAG, passive.POSE_RESIDUAL, passive.JAW_STATE, passive.CLICK, instrumented.LINK)
SOURCE_FILES = ("avail/evidence/passive.py", "avail/evidence/instrumented.py")
CONNECTORS = ("fric", "latch")
CLICK_SEEDS = tuple(range(3920, 3930))        # 10 dev seeds. None of them is a released twin
RELEASED_TWIN_SEED = {"fric": 3946, "latch": 3942}     # dev seeds whose P0 state lets go and backs away

needs_station = pytest.mark.skipif(not aspire_station_xml().is_file(), reason="third_party/ASPIRE is not cloned")


def make_claim(predicate: str = "seated", port: int = 3) -> Claim:
    return Claim(claim_id="claim_test", run_id="test", cycle_id="c0", step_id="insert",
                 predicate=Predicate(name=predicate, target={"plug": "p1", "port": port}),
                 raised_by="skill", context=ClaimContext(task="plug_cycle", stage="insert", attempt=0))


# ============================================================================ no simulator needed
def test_the_5_sources_are_registered_with_their_kind_and_a_zero_cost():
    expected = {"passive.skill_flag": "passive_proprio", "passive.pose_residual": "passive_proprio",
                "passive.jaw_state": "passive_proprio", "passive.click": "passive_proprio",
                "instr.link": "instrumented"}
    assert {s.source_id: s.kind for s in SOURCES} == expected
    for source in SOURCES:
        assert REGISTRY[source.source_id] is source
        assert source.kind in EVIDENCE_KINDS
        assert source.predicates and source.predicates <= set(PREDICATES)
        assert source.estimate_cost(make_claim()) == Cost.zero()


@pytest.mark.parametrize("rel_path", SOURCE_FILES)
def test_source_file_passes_the_truth_wall(rel_path):
    source = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
    assert scan_source(source, rel_path) == []


@pytest.mark.parametrize("rel_path", SOURCE_FILES)
def test_source_file_imports_only_what_a_source_may(rel_path):
    """The standard library, numpy, scipy, cv2, PIL, and 5 corners of avail. Nothing from the simulator."""
    allowed_avail = ("avail.schemas", "avail.station.base", "avail.evidence.base", "avail.config", "avail.stats")
    allowed_other = set(sys.stdlib_module_names) | {"numpy", "scipy", "cv2", "PIL"}
    package = rel_path[:-len(".py")].split("/")[:-1]
    imported = []
    for node in ast.walk(ast.parse((REPO_ROOT / rel_path).read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = package[:len(package) - (node.level - 1)] if node.level else []
            imported.append(".".join([*base, *([node.module] if node.module else [])]))
    assert imported, "the scan found no imports, so it checked nothing"
    for name in imported:
        if name.split(".")[0] == "avail":
            assert any(name == ok or name.startswith(ok + ".") for ok in allowed_avail), name
        else:
            assert name.split(".")[0] in allowed_other, name


def hand_nominal() -> NominalGeometry:
    ports = tuple((x, y) for x in (-0.02, 0.02) for y in (-0.06, -0.02, 0.02, 0.06))
    return NominalGeometry(connector="fric", full_depth_m=0.012, grasp_height_m=0.016, board_xy=(0.5, -0.12),
                           board_yaw=0.05, face_z=0.7675, ports_board=ports, pitch_m=0.04,
                           plug_body_m=(0.012, 0.018, 0.032), finger_travel_m=0.0375, pad_thickness_m=0.001)


def test_pose_residual_signs_and_units_on_a_hand_made_geometry():
    nominal = hand_nominal()
    taught = nominal.seated_tool_pose(3)
    at_taught = passive.pose_residual_features(taught, nominal, 3)
    assert at_taught["axial_mm"] == pytest.approx(0.0, abs=1e-9)
    assert at_taught["lateral_mm"] == pytest.approx(0.0, abs=1e-9)
    assert at_taught["tilt_deg"] == pytest.approx(0.0, abs=1e-4)
    assert (at_taught["port_index_est"], at_taught["port_match"]) == (3.0, 1.0)

    # 2.5 mm above the taught pose and 1 mm to the side: the tool stopped short, so axial is positive
    high = Pose((taught.pos[0] + 0.001, taught.pos[1], taught.pos[2] + 0.0025), taught.quat)
    feats = passive.pose_residual_features(high, nominal, 3)
    assert feats["axial_mm"] == pytest.approx(2.5, abs=1e-6)
    assert feats["lateral_mm"] == pytest.approx(1.0, abs=1e-6)
    assert feats["port_match"] == 1.0

    # 1.5 mm below the taught pose reads negative
    low = Pose((taught.pos[0], taught.pos[1], taught.pos[2] - 0.0015), taught.quat)
    assert passive.pose_residual_features(low, nominal, 3)["axial_mm"] == pytest.approx(-1.5, abs=1e-6)

    # seated in the neighbor: the estimate names the neighbor and the lateral error is the pitch
    neighbor = passive.pose_residual_features(nominal.seated_tool_pose(4), nominal, 3)
    assert (neighbor["port_index_est"], neighbor["port_match"]) == (4.0, 0.0)
    assert neighbor["lateral_mm"] == pytest.approx(40.0, abs=1e-6)

    # a tool tilted 5 degrees about its own x axis
    c, s = math.cos(math.radians(5.0)), math.sin(math.radians(5.0))
    tilted = Pose.from_matrix(taught.position(), taught.rotation() @ np.array([[1, 0, 0], [0, c, -s], [0, s, c]]))
    assert passive.pose_residual_features(tilted, nominal, 3)["tilt_deg"] == pytest.approx(5.0, abs=1e-4)

    # with no claimed port only the parts that need none are measured
    assert set(passive.pose_residual_features(taught, nominal, None)) == {"tilt_deg", "port_index_est"}


def test_pose_residual_score_is_the_formula_in_the_docstring():
    def score(predicate="seated", **changes):
        features = {"axial_mm": 0.0, "lateral_mm": 0.0, "tilt_deg": 0.0, "port_match": 1.0, **changes}
        return passive.pose_residual_score(features, predicate, 12.0)

    assert score() == 0.0
    assert score(axial_mm=1.5) == pytest.approx(-1.5)                # a tool that stopped high, 1 for 1
    assert score(axial_mm=-1.5) == pytest.approx(1.5)                # a little low counts for the claim
    assert score(axial_mm=-3.0) == pytest.approx(-1.0)               # -(-3 + 4 * 1): past 2 mm low it falls 3 for 1
    assert score(lateral_mm=2.9) == 0.0 and score(lateral_mm=5.0) == pytest.approx(-2.0)
    assert score(tilt_deg=1.9) == 0.0 and score(tilt_deg=5.0) == pytest.approx(-6.0)
    assert score(port_match=0.0) == pytest.approx(-25.0)
    assert score(predicate="latched", axial_mm=1.5) == score(axial_mm=1.5)
    # at_port: a plug in the mouth of the right port still counts, a wrong port does not
    assert score(predicate="at_port", axial_mm=9.0) == 0.0
    assert score(predicate="at_port", axial_mm=14.0) == pytest.approx(-2.0)
    assert score(predicate="at_port", port_match=0.0, lateral_mm=40.0) == pytest.approx(-62.0)
    # no score for a predicate the pose does not bear on, or when a term was not measured
    assert score(predicate="released_stable") is None
    assert passive.pose_residual_score({"tilt_deg": 0.0, "port_index_est": 3.0}, "seated", 12.0) is None


def hand_insert(jump_mm: float = 0.0, noise_mm: float = 0.0, stopped_by: str = "force", rng_seed: int = 0) -> dict:
    """An insertion record built by hand at 250 Hz: approach at 8 mm/s, 0.3 s against a barrier at 3 mm/s,
    an optional jump inside 24 ms, then 0.2 s on the hard stop. A travel stop ends while still moving."""
    dt = 0.004
    steps = [8.0 * dt] * 625 + [3.0 * dt] * 75
    force = [1.5] * 625 + list(np.linspace(1.5, 4.5, 75))
    steps += [jump_mm / 6.0] * 6 if jump_mm else [3.0 * dt] * 6
    force += list(np.linspace(4.5, 2.5, 6)) if jump_mm else [4.5] * 6
    if stopped_by == "force":
        steps += [0.0] * 50
        force += list(np.linspace(force[-1], 6.0, 50))
    travel = np.cumsum(steps)
    if noise_mm:
        travel = travel + np.random.default_rng(rng_seed).normal(0.0, noise_mm, len(travel))
    t = dt * np.arange(1, len(travel) + 1)
    return {"flag": True, "stopped_by": stopped_by, "travel_mm": float(travel[-1]), "expected_travel_mm": 22.0,
            "force_at_stop_n": float(force[-1]), "peak_force_n": float(max(force)),
            "trace": {"t_s": t.tolist(), "travel_mm": travel.tolist(), "force_n": [float(v) for v in force]}}


def test_click_features_on_hand_made_traces():
    quiet = passive.click_features(hand_insert())
    assert quiet["snap_mm"] == pytest.approx(0.0, abs=0.01) and quiet["clicked"] == 0.0
    assert quiet["snap_speed_mm_s"] == pytest.approx(8.0, abs=0.1)        # nothing faster than the approach

    record = hand_insert(jump_mm=0.55)
    heard = passive.click_features(record)
    assert 0.25 < heard["snap_mm"] < 0.55 - 8.0 * 0.024 + 1e-9             # the jump, less the commanded 0.192 mm
    assert heard["clicked"] == 1.0
    assert heard["snap_speed_mm_s"] == pytest.approx(0.55 / 0.024, rel=0.01)
    assert heard["peak_n"] == pytest.approx(4.5, abs=0.05) and heard["drop_n"] == pytest.approx(2.0, abs=0.05)
    assert heard["peak_depth_mm"] == pytest.approx(0.55, abs=0.02)         # the peak sat right before the jump
    assert quiet["drop_n"] == pytest.approx(0.0, abs=1e-9)                 # flat force where nothing jumped
    assert heard["final_force_n"] == 6.0
    assert heard["travel_short_mm"] == pytest.approx(22.0 - record["travel_mm"])

    # the commanded speed is used when the skill recorded it: a faster command explains more of the gain
    assert passive.click_features({**record, "speed_mm_s": 8.0})["snap_mm"] == pytest.approx(heard["snap_mm"], abs=1e-6)
    faster = passive.click_features({**record, "speed_mm_s": 12.0})
    assert faster["snap_mm"] == pytest.approx(heard["snap_mm"] - 4.0 * 0.024, abs=1e-6)

    # the same jump in an insertion that ran to its travel cap is a slip
    slip = passive.click_features(hand_insert(jump_mm=0.55, stopped_by="travel"))
    assert slip["snap_mm"] > 0.25 and slip["clicked"] == 0.0


def test_click_rule_holds_under_encoder_noise_of_50_micrometers():
    """The sim's travel trace is noise free. A real encoder is not, and noise must not pass for a jump."""
    def clicks(jump_mm: float) -> float:
        records = (hand_insert(jump_mm=jump_mm, noise_mm=0.05, rng_seed=k) for k in range(20))
        return sum(passive.click_features(record)["clicked"] for record in records)

    assert clicks(0.0) == 0
    assert clicks(0.7) == 20


@pytest.mark.parametrize("trace", [
    None, "text", {}, {"t_s": [0.0, 0.004], "travel_mm": [0.0, 0.03], "force_n": [0.0, 0.1]},
    {"t_s": [0.004 * k for k in range(40)], "travel_mm": [0.03 * k for k in range(39)], "force_n": [1.0] * 40},
    {"t_s": [0.004 * k for k in range(40)], "travel_mm": [float("nan")] * 40, "force_n": [1.0] * 40},
    {"t_s": [0.0] * 40, "travel_mm": [0.03 * k for k in range(40)], "force_n": [1.0] * 40},
    {"t_s": ["a"] * 40, "travel_mm": ["b"] * 40, "force_n": ["c"] * 40},
])
def test_click_features_leave_out_what_a_broken_trace_cannot_give(trace):
    record = {"travel_mm": 20.0, "expected_travel_mm": 22.0, "force_at_stop_n": 6.0, "stopped_by": "force",
              "trace": trace}
    features = passive.click_features(record)
    assert features["travel_short_mm"] == pytest.approx(2.0)
    assert "snap_mm" not in features and "clicked" not in features and "peak_n" not in features
    assert all(math.isfinite(v) for v in features.values())
    evidence = passive.CLICK.collect(None, make_claim(), {"insert": record})
    assert evidence.score is None and "clicked" not in evidence.features


@pytest.mark.parametrize("flag, expected", [(True, 1.0), (False, 0.0), (np.bool_(True), 1.0), (1, 1.0), (0, 0.0)])
def test_skill_flag_reads_true_and_false(flag, expected):
    evidence = passive.SKILL_FLAG.collect(None, make_claim(), {"insert": {"flag": flag}})
    assert evidence.features == {"flag": expected} and evidence.score == expected


@pytest.mark.parametrize("params", [{}, {"insert": None}, {"insert": {}}, {"insert": {"flag": "yes"}},
                                    {"insert": {"flag": 0.7}}, {"insert": {"flag": None}}])
def test_skill_flag_with_no_usable_flag_gives_no_features(params):
    evidence = passive.SKILL_FLAG.collect(None, make_claim(), params)
    assert evidence.features == {} and evidence.score is None


# ============================================================================ on real injected states
@dataclass
class Collected:
    """What the 5 sources said about one injected state, with the ground truth they never saw."""

    claim: Claim
    label: bool                          # truth of the claimed predicate
    truth: dict
    ev: dict[str, Evidence]              # the claim and the params as the harness made them
    ev_no_insert: dict[str, Evidence]    # the same with no 'insert' record
    ev_empty: dict[str, Evidence]        # the same with no params at all
    ev_at_port: Evidence                 # pose_residual asked about at_port
    ev_unwired: tuple[Evidence, Evidence]   # link on a station with no signals: through a view, and bare
    raw_click: dict[str, float]          # click_features as computed, before the record drops anything
    raw_pose: dict[str, float]           # pose_residual_features as computed
    station_untouched: bool              # station time, tool pose and gripper are as they were


class _NoSignals:
    """A station that reports robot state and no signals."""

    station_id = "bare"

    def __init__(self, state):
        self._state = state

    def robot_state(self):
        return self._state


@lru_cache(maxsize=None)
def collected(connector: str, cls: str, seed: int) -> Collected:
    """Inject one state, run every source on it in every way the tests need, and close the station."""
    from avail.zoo.harness import prepare

    assert 3000 <= seed <= 4999, "this file uses dev seeds 3000 to 4999 only"
    prepared = prepare(seed, connector, cls, render=False)
    try:
        view, claim, params = prepared.view, prepared.claim, prepared.params
        before = view.robot_state()
        qpos_before = prepared.station._mj_data.qpos.copy()      # the test may look past the wall
        without = {k: v for k, v in params.items() if k != "insert"}
        at_port = claim.model_copy(update={"predicate": Predicate(name="at_port", target=dict(claim.predicate.target))})
        bare = _NoSignals(before)
        out = Collected(
            claim=claim,
            ev={s.source_id: s.collect(view, claim, params) for s in SOURCES},
            ev_no_insert={s.source_id: s.collect(view, claim, without) for s in SOURCES},
            ev_empty={s.source_id: s.collect(view, claim, {}) for s in SOURCES},
            ev_at_port=passive.POSE_RESIDUAL.collect(view, at_port, params),
            ev_unwired=(instrumented.LINK.collect(StationView(bare), claim, params),
                        instrumented.LINK.collect(bare, claim, params)),
            raw_click=passive.click_features(params["insert"]),
            raw_pose=passive.pose_residual_features(before.arms[params["arm"]].tool_pose, params["nominal"],
                                                    claim.predicate.target["port"]),
            label=prepared.label(), truth=prepared.truth(), station_untouched=False,
        )
        after = view.robot_state()
        arm = params["arm"]
        # the gripper reading carries fresh noise on every read, so "untouched" is judged on the joints
        out.station_untouched = bool(
            after.ts == before.ts and after.arms[arm].tool_pose == before.arms[arm].tool_pose
            and np.array_equal(prepared.station._mj_data.qpos, qpos_before))
        return out
    finally:
        prepared.close()


def feature(state: Collected, source_id: str, name: str) -> float:
    return state.ev[source_id].features[name]


@needs_station
@pytest.mark.parametrize("connector", CONNECTORS)
def test_skill_flag_is_1_on_a_seated_plug_and_is_fooled_by_a_short_one(connector):
    seated, short = collected(connector, "P0", 3920), collected(connector, "N2", 3920)
    assert seated.label is True and short.label is False
    assert seated.ev["passive.skill_flag"].features == {"flag": 1.0}
    assert seated.ev["passive.skill_flag"].score == 1.0
    # the baseline everybody trusts today: the skill says done on a plug that is 1 to 2 mm short
    assert short.ev["passive.skill_flag"].features == {"flag": 1.0}


@needs_station
def test_pose_residual_names_the_wrong_port_and_reads_a_shortfall_of_2_to_4_mm():
    wrong, short, seated = collected("fric", "N6", 3930), collected("latch", "N3", 3931), collected("fric", "P0", 3920)
    claimed = wrong.claim.predicate.target["port"]
    went_into = wrong.truth["metrics"]["nearest_port"]
    assert went_into != claimed
    assert feature(wrong, "passive.pose_residual", "port_match") == 0.0
    assert feature(wrong, "passive.pose_residual", "port_index_est") == went_into
    assert feature(wrong, "passive.pose_residual", "lateral_mm") > 30.0          # the pitch is 40 mm

    true_shortfall = short.truth["metrics"]["shortfall_mm"]
    assert 2.0 <= true_shortfall <= 4.0
    assert abs(feature(short, "passive.pose_residual", "axial_mm") - true_shortfall) <= 3.5
    assert feature(short, "passive.pose_residual", "port_match") == 1.0

    assert feature(seated, "passive.pose_residual", "port_match") == 1.0
    assert abs(feature(seated, "passive.pose_residual", "axial_mm")) <= 3.5
    assert feature(seated, "passive.pose_residual", "tilt_deg") < 2.0
    assert seated.ev["passive.pose_residual"].score > wrong.ev["passive.pose_residual"].score + 20.0

    # asked about at_port, a short plug in the right port passes and a plug in the neighbor does not
    assert short.truth["predicates"]["at_port"] is True and wrong.truth["predicates"]["at_port"] is False
    assert short.ev_at_port.score > wrong.ev_at_port.score + 20.0


@needs_station
def test_jaw_state_separates_a_dropped_plug_from_a_held_one_and_an_open_hand_from_a_closed_one():
    held, dropped, short = collected("fric", "P0", 3920), collected("fric", "N8", 3932), collected("fric", "N2", 3920)
    assert held.truth["predicates"]["held"] is True and dropped.truth["predicates"]["held"] is False
    assert abs(feature(held, "passive.jaw_state", "jaw_open_error")) < passive.JAW_BAND
    assert feature(dropped, "passive.jaw_state", "jaw_open_error") > 0.5
    assert feature(held, "passive.jaw_state", "jaw_torque") > feature(dropped, "passive.jaw_state", "jaw_torque") + 5.0
    assert held.ev["passive.jaw_state"].score == 0.0
    assert dropped.ev["passive.jaw_state"].score < -0.5
    # it says nothing about depth: a plug 1 to 2 mm short scores exactly like a seated one
    assert short.ev["passive.jaw_state"].score == held.ev["passive.jaw_state"].score

    # the release slice: the fingers opened on the released twin and never opened on N9
    for connector in CONNECTORS:
        released = collected(connector, "P0", RELEASED_TWIN_SEED[connector])
        stuck = collected(connector, "N9", 3934)
        assert released.claim.predicate.name == stuck.claim.predicate.name == "released_stable"
        assert released.label is True and stuck.label is False
        assert released.ev["passive.jaw_state"].score > 0.5
        assert stuck.ev["passive.jaw_state"].score == 0.0


@needs_station
@pytest.mark.parametrize("connector", CONNECTORS)
def test_click_is_heard_on_seated_plugs_and_not_on_plugs_1_to_2_mm_short(connector):
    seated = [collected(connector, "P0", seed) for seed in CLICK_SEEDS]
    short = [collected(connector, "N2", seed) for seed in CLICK_SEEDS]
    assert sum(s.label for s in seated) >= 9, "the injected P0 states are not seated, so this test checks nothing"
    assert not any(s.label for s in short)
    heard_seated = sum(feature(s, "passive.click", "clicked") for s in seated)
    heard_short = sum(feature(s, "passive.click", "clicked") for s in short)
    assert heard_seated >= 9, f"clicked on {heard_seated:.0f} of 10 seated {connector} states"
    assert heard_short <= 2, f"clicked on {heard_short:.0f} of 10 short {connector} states"
    # the jump is the signal: every seated state shows more of it than every short one
    assert min(feature(s, "passive.click", "snap_mm") for s in seated if s.label) > \
        max(feature(s, "passive.click", "snap_mm") for s in short) + 0.1


@needs_station
def test_link_is_up_when_the_contacts_mate_and_says_nothing_about_the_latch():
    for connector in CONNECTORS:
        seated, short = collected(connector, "P0", 3920), collected(connector, "N2", 3920)
        assert seated.ev["instr.link"].features == {"link_up": 1.0} and seated.ev["instr.link"].score == 1.0
        assert short.ev["instr.link"].features == {"link_up": 0.0} and short.ev["instr.link"].score == 0.0
    unlatched = collected("latch", "N5", 3933)
    assert unlatched.claim.predicate.name == "latched" and unlatched.label is False
    assert unlatched.truth["predicates"]["seated"] is True           # fully in, latch blocked
    assert unlatched.ev["instr.link"].features == {"link_up": 1.0}

    # a station that is not instrumented: evidence with no features and no score, through a view or bare
    for evidence in seated.ev_unwired:
        assert evidence.source_id == "instr.link" and evidence.features == {} and evidence.score is None


def every_state() -> list[Collected]:
    states = [collected(c, cls, seed) for c in CONNECTORS for cls in ("P0", "N2") for seed in CLICK_SEEDS]
    states += [collected("fric", "N6", 3930), collected("latch", "N3", 3931), collected("fric", "N8", 3932),
               collected("latch", "N5", 3933)]
    states += [collected(c, "N9", 3934) for c in CONNECTORS]
    states += [collected(c, "P0", RELEASED_TWIN_SEED[c]) for c in CONNECTORS]
    return states


@needs_station
def test_every_source_answers_a_release_claim():
    for connector in CONNECTORS:
        for state in (collected(connector, "N9", 3934), collected(connector, "P0", RELEASED_TWIN_SEED[connector])):
            assert state.claim.predicate.name == "released_stable"
            for source in SOURCES:
                evidence = state.ev[source.source_id]
                assert isinstance(evidence, Evidence)
                assert evidence.claim_id == state.claim.claim_id
                assert (evidence.source_id, evidence.kind) == (source.source_id, source.kind)
            # each source still measures what it can
            assert state.ev["passive.skill_flag"].features == {"flag": 1.0}
            assert feature(state, "passive.click", "clicked") == 1.0     # the insertion before the release clicked
            assert feature(state, "passive.pose_residual", "port_match") == 1.0
            assert state.ev["instr.link"].features == {"link_up": 1.0}
            # and only the sources that bear on a release give a score
            assert state.ev["passive.pose_residual"].score is None and state.ev["passive.click"].score is None
            assert state.ev["passive.jaw_state"].score is not None


@needs_station
def test_a_missing_insert_record_gives_evidence_with_no_features():
    for state in every_state():
        for source_id in ("passive.skill_flag", "passive.click"):
            assert state.ev[source_id].features, "with the record these sources do measure something"
            for evidence in (state.ev_no_insert[source_id], state.ev_empty[source_id]):
                assert isinstance(evidence, Evidence) and evidence.features == {} and evidence.score is None
        # the sources that read the station do not need the record
        for source_id in ("passive.pose_residual", "instr.link"):
            assert state.ev_no_insert[source_id].features == state.ev[source_id].features
        # a second read of the gripper carries its own noise: 0.02 of travel is 0.75 mm, 3 N is 10 sigma
        with_record, without = state.ev["passive.jaw_state"].features, state.ev_no_insert["passive.jaw_state"].features
        assert with_record.keys() == without.keys()
        for name, tol in (("jaw_pos", 0.02), ("jaw_open_error", 0.02), ("jaw_torque", 3.0)):
            assert without[name] == pytest.approx(with_record[name], abs=tol), name
        # with no params at all there is no taught geometry and no arm name. The link still reads
        assert state.ev_empty["passive.pose_residual"].features == {}
        assert state.ev_empty["passive.jaw_state"].features == {}
        assert state.ev_empty["instr.link"].features == state.ev["instr.link"].features


@needs_station
def test_features_are_finite_and_nothing_was_spent_or_moved():
    seen = set()
    for state in every_state():
        assert state.station_untouched
        # what the sources computed was finite as computed, and the record dropped nothing
        assert all(math.isfinite(v) for v in (*state.raw_click.values(), *state.raw_pose.values()))
        assert state.ev["passive.click"].features == state.raw_click
        assert state.ev["passive.pose_residual"].features == state.raw_pose
        records = [*state.ev.values(), *state.ev_no_insert.values(), *state.ev_empty.values(), state.ev_at_port,
                   *state.ev_unwired]
        for evidence in records:
            for name, value in evidence.features.items():
                assert isinstance(value, float) and math.isfinite(value), (evidence.source_id, name, value)
            assert evidence.score is None or math.isfinite(evidence.score)
            assert evidence.cost == Cost.zero()
            assert evidence.side_effect == "none" and evidence.artifacts == []
        seen |= {(sid, name) for sid, evidence in state.ev.items() for name in evidence.features}
    # every feature the docstrings promise showed up on at least 1 real state
    assert seen == {
        ("passive.skill_flag", "flag"),
        *(("passive.pose_residual", n) for n in ("axial_mm", "lateral_mm", "port_index_est", "port_match", "tilt_deg")),
        *(("passive.jaw_state", n) for n in ("jaw_pos", "jaw_torque", "jaw_open_error")),
        *(("passive.click", n) for n in ("peak_n", "drop_n", "peak_depth_mm", "snap_mm", "snap_speed_mm_s",
                                         "final_force_n", "travel_short_mm", "clicked")),
        ("instr.link", "link_up"),
    }
