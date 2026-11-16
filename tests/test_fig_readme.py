"""The README figure's waffles (S5c): 1 square per cycle, colored by what was declared against what was true."""

from __future__ import annotations

import json

from avail.analysis import fig_readme


def _end(seed, arm, uncaught=0, declared=True, true=True, wall=100.0, assists=0):
    return {"kind": "cycle_end", "payload": {"seed": seed, "arm": arm, "uncaught": uncaught, "declared_success": declared,
                                             "true_success": true, "wall_s": wall, "assists": assists, "usd": 0.0}}


def _ledger(folder, events):
    folder.mkdir(parents=True)
    (folder / "ledger.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")


def test_each_cycle_is_green_red_or_grey_and_the_times_are_multiples_of_the_skills(tmp_path):
    _ledger(tmp_path / "fig_final_flag_latch_b72000", [
        _end(72000, "flag"), _end(72001, "flag", uncaught=1, true=False), _end(72002, "flag", declared=False, true=False)])
    _ledger(tmp_path / "fig_final_avail_human_latch_b72000", [
        _end(72000, "avail_human", wall=150.0, assists=1), _end(72001, "avail_human", wall=200.0, assists=2),
        _end(72001, "avail_human", wall=250.0, assists=3)])          # a rerun of a seed: the last one counts
    s = fig_readme.arm_summary(tmp_path)
    assert s["skill_done"]["outcomes"] == ["green", "red", "grey"]
    # avail_human is E3's and FIG's recorded name for avail_simop: its ledgers read as avail_simop
    assert (s["avail_simop"]["n"], s["avail_simop"]["time_x"]) == (2, 2.0)        # (150 + 250) / 2 over 100
    assert s["avail_simop"]["assists_per_100"] == 200.0
    assert s["avail_simop"]["title"] == "AVAIL + simulated operator"
    assert "agent_looks" not in s                                                  # an arm with no ledger is left out


def test_avails_reason_names_the_stage_that_decided():
    ev = {"passive.click": {"clicked": 0.0}, "passive.pose_residual": {"axial_mm": 0.1}, "probe.tug": {"disp_mm": 1.03}}
    assert fig_readme._reason(ev, 1) == "no latch click"
    assert fig_readme._reason(ev, 3) == "tug moved 1.0 mm"
    assert fig_readme._reason({"passive.pose_residual": {"axial_mm": 2.9}, "passive.click": {"clicked": 0.0}}, 1) \
        == "plug 2.9 mm short"
