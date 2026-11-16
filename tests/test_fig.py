"""The figure run FIG (S5c): the agent_looks arm, the plan and its judge estimate, the judge's limits, the seed range.

The arm runs on a toy task and a fake cell with a scripted judge, so nothing here needs the simulator or a key."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from avail.core.task import TaskSpec
from avail.evidence.judge import build_prompt, typical_views
from avail.schemas import Claim, ClaimContext, Predicate, new_id
from avail.shift import cli
from avail.shift.driver import CycleDriver

TASK = TaskSpec.model_validate({
    "task": "toy", "title": "Pick 1 widget and place it", "items": {"kind": "widget", "count": 1},
    "steps": [{"name": "pick", "claim": "held"}, {"name": "place", "claim": "placed"}],
    "predicates": {p: {"done_means": f"The widget is {p} where the task says it goes.", "checks": ["passive.jaw_hold"],
                       "decide": {"kind": "rule", "accept_all": [{"feature": "passive.jaw_hold/in_band", "above": 0.5}]}}
                   for p in ("held", "placed")},
    "fork": {"max_attempts": 2},
})


class Cell:
    variant, n_items = "v1", 1
    view = None

    def __init__(self, flag=True):
        self.flag, self.tries = flag, []

    def target_of(self, item):
        return {}

    def run_skill(self, step, item, attempt, marks=None):
        self.tries.append(step)
        return SimpleNamespace(ok=True, flag=self.flag, seconds=1.0, failure=None)

    def claim_params(self, item, step, result):
        return {"step": step}

    def now(self):
        return 0.0


class Judge:
    """Says yes or no from a list, 1 answer per call."""

    def __init__(self, answers):
        self.answers, self.asked = list(answers), []

    def collect(self, view, claim, params):
        self.asked.append(params["step"])
        holds = self.answers.pop(0)
        return SimpleNamespace(features={"holds": 1.0 if holds else 0.0}, cost=SimpleNamespace(seconds=2.0, usd=0.001))


def test_the_judge_decides_every_step_and_the_flag_does_not():
    cell, judge = Cell(flag=False), Judge([True, True])          # the skill says it failed both steps: the judge says yes
    log = CycleDriver(TASK, cell, "agent_looks", judge=judge).run()
    assert judge.asked == ["pick", "place"] and log.items[0].declared_done
    assert log.model_s == 4.0 and log.usd == pytest.approx(0.002)


def test_a_no_is_retried_with_the_flag_arms_tries_then_left_open():
    cell, judge = Cell(flag=True), Judge([False, False])        # the skill says done, the judge says no twice
    log = CycleDriver(TASK, cell, "agent_looks", judge=judge).run()
    assert cell.tries == ["pick", "pick"] and not log.items[0].declared_done and log.items[0].unresolved == "pick"


def test_the_prompt_carries_the_robots_readings_only_when_given():
    claim = Claim(claim_id=new_id("claim"), run_id="r", cycle_id="c", step_id="0.pick",
                  predicate=Predicate(name="held", target={"item": 0}, definition="The gripper holds the cable."),
                  raised_by="skill", context=ClaimContext(task="toy", stage="pick", attempt=0))
    views = typical_views("crop_ref")
    plain = build_prompt(claim, "latch", "crop_ref", views)
    told = build_prompt(claim, "latch", "crop_ref", views, state=["Gripper reading 0.31."])
    assert "own readings" not in plain and "- Gripper reading 0.31." in told
    assert told.index("own readings") < told.index("Answer with 1 JSON object")


def test_the_plan_runs_every_arm_on_the_same_100_seeds_and_prints_the_estimate_first(capsys):
    plan = cli.fig_plan()
    assert list(plan["arms"]) == ["skill_done", "agent_looks", "avail", "avail_human"]
    assert {a: (s["driver_arm"], s["seeds"]) for a, s in plan["arms"].items()} == {
        "skill_done": ("flag", [72000, 72099]), "agent_looks": ("agent_looks", [72000, 72099]),
        "avail": ("avail", [72000, 72099]), "avail_human": ("avail_human", [72000, 72099])}
    assert plan["judge"]["model"] == "luna" and plan["judge"]["cap_usd"] == 6.0
    assert plan["judge"]["estimate_usd"] <= 6.0 and plan["judge"]["cycles"] == 100


def test_an_estimate_over_the_cap_cuts_the_judge_arm_and_never_below_40(monkeypatch):
    import avail.config

    real = avail.config.load_experiment()

    def with_price(usd):
        exp = real.model_copy(deep=True)
        exp.fig["judge_usd_per_call"] = usd
        monkeypatch.setattr(avail.config, "load_experiment", lambda *a, **k: exp)
        return cli.fig_plan()

    cut = with_price(0.0075)                                     # 40 calls: $0.30 a cycle, $6 buys 20 cycles
    assert cut["judge"]["cycles"] == 20 and cut["arms"]["agent_looks"]["seeds"] == [72000, 72019]
    assert cut["arms"]["avail"]["cycles"] == 100                 # only the judge arm is cut
    some = with_price(0.0025)                                    # $0.10 a cycle: 60 cycles
    assert some["judge"]["cycles"] == 60


def test_the_judge_may_not_be_fable_or_cost_over_3_dollars():
    assert "Fable" in cli.fig_judge_problem("cable_loop", "fable")
    assert cli.fig_judge_problem("cable_loop", "no_such_model")
    said = cli.fig_judge_problem("cable_loop", "luna")
    assert said is None or "cannot be called now" in said      # on a machine with no key the router says why


def test_only_fig_reads_the_figure_seeds(tmp_path):
    from avail.seeds import TestSeedError, guard

    record = tmp_path / "reads.jsonl"
    with pytest.raises(TestSeedError):
        guard([72000], purpose="test", final_run="E3", record_path=record)
    with pytest.raises(TestSeedError):
        guard([72000], purpose="test", record_path=record)
    with pytest.raises(TestSeedError, match="retired"):            # FIG read its seeds on 2026-11-13
        guard([72000, 72099], purpose="test", final_run="FIG", record_path=record)


def test_run_refuses_the_fig_final_run():
    with pytest.raises(SystemExit):
        cli.main(["run", "--variant", "latch", "--arm", "flag", "--seeds", "72000:72001", "--shift", "x",
                  "--final-run", "FIG"])
