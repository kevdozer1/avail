"""The shift card's per-item figures and time split (S5b): description only, from cycle_end events alone."""

from __future__ import annotations

import pytest

from avail.shift.card import arm_card, md


def _end(assists: int, wait: float, steps: list[tuple[bool, bool]]) -> dict:
    items = [{"item": k, "true": True, "declared": True} for k in range(5)]
    return {"cycle_id": "c", "uncaught": 0, "items": items, "true_success": True, "declared_success": True,
            "assists": assists, "takeovers": 0, "usd": 0.0,
            "time_s": {"motion": 90.0, "verify": 1.0, "model": 0.0, "wait": wait, "handling": 0.0},
            "steps": [{"declared": d, "true": t} for d, t in steps]}


def test_per_item_figures_and_bad_steps():
    ends = [_end(2, 40.0, [(True, True)] * 19 + [(True, False)]), _end(0, 0.0, [(True, True)] * 20)]
    card = arm_card(ends, [], 0.005)
    assert card["assists_per_100_items"] == pytest.approx(100 * 2 / 10)            # 2 calls over 10 items
    assert card["seconds_per_item"] == pytest.approx((91 + 40 + 91) / 10)
    assert card["time_split_s"]["wait"] == pytest.approx(20.0)                       # per cycle
    assert (card["bad_steps_through"]["k"], card["bad_steps_through"]["n"]) == (1, 40)
    text = md({"arms": {"avail_human latch": card}})
    assert "description only" in text and "| avail_human latch | 20.0 | 22.2 |" in text


def test_the_e3_witness_must_come_from_the_cable_loops_calibration():
    from types import SimpleNamespace

    from avail.station.mj_plug import cable_shift

    names = [d.name for d in cable_shift.cable_stages()]
    zoo = SimpleNamespace(alpha=0.01, stage_defs=[SimpleNamespace(name=n) for n in names],
                          calibration={"split_id": "zoo_calibration"})
    assert not cable_shift._witness_matches(zoo, 0.01)                             # the zoo's 1-plug states: stale
    zoo.calibration = {"split_id": "cable_calibration"}
    assert cable_shift._witness_matches(zoo, 0.01)
