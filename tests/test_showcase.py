"""The showcase (S5c): the written cycle-selection rule, the paired cycles and the cost line, from fake ledgers."""

from __future__ import annotations

import json

from avail.analysis import showcase


def _end(seed, arm, uncaught=0, declared=True, true=True, wall=100.0, assists=0, wait=0.0, handling=0.0):
    return {"kind": "cycle_end", "payload": {
        "seed": seed, "arm": arm, "cycle_id": f"{arm}_c{seed}", "uncaught": uncaught, "declared_success": declared,
        "true_success": true, "wall_s": wall, "assists": assists, "items": [],
        "time_s": {"motion": wall - wait - handling, "verify": 0.0, "model": 0.0, "wait": wait, "handling": handling}}}


def _ledger(folder, events):
    folder.mkdir(parents=True)
    (folder / "ledger.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")


def test_the_rule_takes_the_lowest_seed_with_a_bad_skill_cycle_a_clean_operator_cycle_and_a_call(tmp_path):
    _ledger(tmp_path / "dev_flag_latch_b20400", [_end(20400, "flag"), _end(20401, "flag", uncaught=1),
                                                 _end(20402, "flag", uncaught=2), _end(20403, "flag", uncaught=1)])
    _ledger(tmp_path / "dev_avail_simop_latch_b20400", [
        _end(20400, "avail_simop", assists=1),
        _end(20401, "avail_simop"),                           # clean, but no call for help
        _end(20402, "avail_simop", uncaught=1, true=False),   # not clean
        _end(20403, "avail_simop", assists=2)])               # the 1 that meets clause 1
    pairs = showcase.paired(tmp_path)
    assert [p[0] for p in pairs] == [20400, 20401, 20402, 20403]
    assert showcase.pick(pairs) == (20403, "1")
    assert showcase.pick(pairs[:2]) == (20401, "3a")          # no call anywhere: the call condition is dropped


def test_the_operators_old_arm_name_reads_as_avail_simop(tmp_path):
    _ledger(tmp_path / "e3_avail_human_latch_b30000", [_end(30000, "avail_human", assists=1)])
    assert list(showcase.cycles(tmp_path, "avail_simop")) == [30000]


def test_the_cost_line():
    cs = [_end(1, "avail_simop", wall=150.0, assists=1, wait=20.0, handling=40.0)["payload"],
          _end(2, "avail_simop", wall=170.0, assists=2, wait=40.0, handling=20.0)["payload"]]
    c = showcase.cost(cs, base_s=80.0)
    assert c["calls_per_100"] == 150.0 and c["time_x"] == 2.0 and c["robots_per_operator"] == 160.0 / 60.0
