"""The contact probes: push_seat, tug and wiggle. One smoke test each, on real injected states.

Evaluator-side: states come from ``avail.zoo.harness.prepare`` and ground truth is read after
each probe to check what it did. The probes themselves see a ``StationView`` only. Every seed
is a dev seed between 3000 and 4999.
"""

from __future__ import annotations

import pytest

from avail.config import load_experiment
from avail.evidence import probes
from avail.evidence.base import StationView
from avail.paths import REPO_ROOT, aspire_station_xml
from avail.walls import scan_source

pytestmark = pytest.mark.skipif(not aspire_station_xml().is_file(), reason="third_party/ASPIRE is not cloned")


class RecordingView:
    """A StationView that records every guarded move it is asked for."""

    def __init__(self, view: StationView):
        self._view = view
        self.calls: list[tuple] = []

    def __getattr__(self, name):
        return getattr(self._view, name)

    def move_relative(self, arm, delta_m, frame, speed, max_force, max_travel):
        self.calls.append((max_force, max_travel))
        return self._view.move_relative(arm, delta_m, frame, speed, max_force, max_travel)


def run(source, seed: int, connector: str, cls: str):
    """Collect one probe on an injected state. Returns (evidence, truth label after, caps asked for, connector)."""
    from avail.zoo.harness import prepare

    prepared = prepare(seed, connector, cls, render=False)
    view = RecordingView(prepared.view)
    ev = source.collect(view, prepared.claim, prepared.fresh_params())
    label = prepared.label()
    prepared.close()
    return ev, label, view.calls


def caps_ok(calls: list[tuple], connector: str) -> bool:
    cfg = load_experiment()
    force = max(cfg.push_force_cap_n(connector), cfg.tug_force_cap_n(connector), cfg.probes.wiggle.force_cap_n)
    travel = max(cfg.probes.push_seat.travel_cap_mm, cfg.probes.tug.travel_cap_mm, cfg.probes.wiggle.travel_cap_mm) / 1000.0
    return all(f <= force + 1e-9 and t <= travel + 1e-9 for f, t in calls)


def test_push_on_a_seated_plug_meets_the_stop_at_once():
    ev, label, calls = run(probes.PUSH_SEAT, 3400, "latch", "P0")
    assert ev.features["travel_mm"] < 0.4 and ev.features["stopped_by_force"] == 1.0
    assert ev.score > 0.5 and label is True and ev.cost.moved_robot
    assert caps_ok(calls, "latch")
    assert calls[0][0] == pytest.approx(load_experiment().push_force_cap_n("latch"))


def test_push_on_a_short_plug_seats_it_and_says_so():
    ev, label, calls = run(probes.PUSH_SEAT, 3401, "fric", "N2")
    assert ev.features["clicked"] == 1.0 and ev.side_effect == "repaired"
    assert label is True                                     # the repair the push probe relies on
    assert ev.score > 0.5 and caps_ok(calls, "fric")


def test_push_on_a_plug_in_the_mouth_scores_low():
    ev, label, _ = run(probes.PUSH_SEAT, 3402, "latch", "N4")
    assert ev.score < 0.2 and label is False


def test_tug_holds_a_seated_plug_and_leaves_it_seated():
    ev, label, calls = run(probes.TUG, 3403, "latch", "P0")
    assert ev.features["slipped"] == 0.0 and ev.features["stopped_by_force"] == 1.0
    assert ev.score > 0.5 and label is True and caps_ok(calls, "latch")


def test_tug_pulls_an_unlatched_plug_and_it_goes_back_in_without_a_click():
    ev, label, _ = run(probes.TUG, 3404, "latch", "N5")
    assert ev.features["slipped"] == 1.0 and ev.side_effect == "disturbed"
    assert ev.features.get("reseat_clicked") == 0.0
    assert ev.score < 0.2 and label is False                 # the latch is still blocked


def test_tug_on_a_short_plug_reseats_it_with_a_click():
    ev, label, _ = run(probes.TUG, 3405, "fric", "N2")
    assert ev.features["slipped"] == 1.0 and ev.features.get("reseat_clicked") == 1.0
    assert label is True and ev.score > 0.5


def test_wiggle_reports_play_and_comes_back():
    ev, label, calls = run(probes.WIGGLE, 3406, "fric", "P0")
    for key in ("lateral_compliance_mm_per_n", "travel_xp_mm", "travel_yn_mm", "return_error_mm"):
        assert key in ev.features
    assert ev.features["return_error_mm"] < 0.5 and label is True
    assert all(f <= load_experiment().probes.wiggle.force_cap_n + 1e-9 for f, _ in calls)


def test_a_probe_refuses_a_claim_it_does_not_bear_on():
    from avail.schemas import Claim, ClaimContext, Predicate

    claim = Claim(claim_id="c", run_id="r", cycle_id="y", step_id="s", raised_by="skill",
                  predicate=Predicate(name="released_stable", target={"plug": "p1"}),
                  context=ClaimContext(task="t", stage="s", attempt=0))
    ev = probes.PUSH_SEAT.collect(None, claim, {})
    assert ev.features == {} and ev.score is None


def test_probes_stay_behind_the_truth_wall():
    path = REPO_ROOT / "avail" / "evidence" / "probes.py"
    assert scan_source(path.read_text(encoding="utf-8"), "avail/evidence/probes.py") == []
