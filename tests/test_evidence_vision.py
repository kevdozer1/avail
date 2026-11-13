"""vision.gap: the classical gap between the plug shoulder and the socket face, from 1 frame.

Evaluator-side: states come from ``avail.zoo.harness.prepare`` and truth is read to check the
estimate. Every seed is a dev seed between 3000 and 4999.
"""

from __future__ import annotations

import numpy as np
import pytest

from avail.evidence import vision
from avail.evidence.base import FrameStore
from avail.paths import REPO_ROOT, aspire_station_xml
from avail.walls import scan_source

pytestmark = pytest.mark.skipif(not aspire_station_xml().is_file(), reason="third_party/ASPIRE is not cloned")
SOURCE = vision.VisionGap()


def gap_on(seed: int, connector: str, cls: str, view: str = "claim"):
    from avail.zoo.harness import prepare

    prepared = prepare(seed, connector, cls)
    params = prepared.fresh_params()
    params["view"] = view
    ev = SOURCE.collect(prepared.view, prepared.claim, params)
    short = prepared.outcome.truth["metrics"]["shortfall_mm"]
    keys = params["frames"].keys()
    prepared.close()
    return ev, short, keys


def test_a_seated_plug_shows_no_gap_and_a_short_one_shows_its_shortfall():
    seated = [gap_on(s, "latch", "P0")[0].features["gap_mm_est"] for s in range(3500, 3503)]
    short = [gap_on(s, "latch", "N3") for s in range(3500, 3503)]
    assert np.median(seated) < 0.6
    gaps = [ev.features["gap_mm_est"] for ev, _, _ in short]
    assert np.median(gaps) > np.median(seated) + 1.5
    # within 0.5 mm of the true shortfall on these clean renders
    assert all(abs(ev.features["gap_mm_est"] - s) < 0.5 for ev, s, _ in short)


def test_an_unlatched_plug_looks_seated():
    ev, _, _ = gap_on(3503, "latch", "N5")
    assert ev.features["gap_mm_est"] < 0.6                   # vision cannot see a latch


def test_a_plug_in_the_wrong_port_leaves_the_claimed_socket_empty():
    ev, _, _ = gap_on(3504, "latch", "N6")
    assert ev.features.get("plug_found") == 0.0
    assert ev.score == pytest.approx(-12.0)                  # the whole tongue counts as showing


def test_the_now_view_takes_a_fresh_frame_and_keeps_it():
    ev, _, keys = gap_on(3505, "fric", "P0", view="now")
    assert "now/right_camera_d405" in keys and "gap_mm_est" in ev.features


def test_a_missing_frame_gives_an_empty_evidence():
    from avail.zoo.harness import prepare

    prepared = prepare(3506, "fric", "P0", render=False)
    params = dict(prepared.params)
    params["frames"] = FrameStore()
    ev = SOURCE.collect(prepared.view, prepared.claim, params)
    prepared.close()
    assert ev.features == {} and ev.score is None


def test_the_same_frame_gives_the_same_features():
    from avail.zoo.harness import prepare

    prepared = prepare(3507, "latch", "N2")
    nominal = prepared.params["nominal"]
    frame = prepared.params["frames"].get(f"claim/{nominal.holding_arm}_camera_d405")
    a = vision.measure_gap(frame, nominal, prepared.outcome.target_port)
    b = vision.measure_gap(frame, nominal, prepared.outcome.target_port)
    prepared.close()
    assert a == b


def test_vision_stays_behind_the_truth_wall():
    path = REPO_ROOT / "avail" / "evidence" / "vision.py"
    assert scan_source(path.read_text(encoding="utf-8"), "avail/evidence/vision.py") == []
