"""The E3 final run's plan and the check that runs before it reads a test seed (S5).

``avail shift e3`` reads the shift test seeds once, arm by arm. The seed guard refuses a changed
config once E3 has read its split, so whatever the run needs from the config must be right before
the 1st arm. The vlm arm needs its judge's spend cap, so the command refuses to start a vlm arm
its cap cannot cover, before any seed is read. No cycle runs in these tests.
"""

from __future__ import annotations

import json

import pytest

from avail.shift import cli
from avail.station.mj_plug import cable_shift as _cable_shift  # noqa: F401  imported before any test patches the
                                                               # config loader, which it binds at import

E3_CONFIG_PROBLEM = cli.e3_config_problem          # the real check, before the fixture below stands it in
LOAD_EXPERIMENT = __import__("avail.config", fromlist=["load_experiment"]).load_experiment  # the real loader


@pytest.fixture(autouse=True)
def shifts_in_tmp(monkeypatch, tmp_path):
    """No test here writes under AVAIL_DATA: the E3 setup record and the shift folders go to a temporary folder, and
    E3 has read nothing unless a test says so."""
    monkeypatch.setattr(cli, "_shift_dir", lambda name: tmp_path / "shifts" / name)
    monkeypatch.setattr(cli, "e3_started", lambda: False)
    monkeypatch.setattr(cli, "prepare_problem", lambda *a: None)              # all prepared, unless a test says
    monkeypatch.setattr(cli, "e3_config_problem", lambda: None)
    monkeypatch.setattr(cli, "station_problem", lambda *a: None)              # the robot model in place
    monkeypatch.setattr(cli, "judge_problem", lambda *a: None)                # the judge callable
    return tmp_path / "shifts"


def _config(monkeypatch, caps: dict | None = None, drop_e3: tuple = (), e3: dict | None = None):
    """The live config with these spend caps, without these e3 keys and with these e3 values, so a test does not
    depend on the caps Kevin sets in config/experiment.yaml."""
    import avail.config

    exp = LOAD_EXPERIMENT()                                                   # the live config, every call
    block = {**{k: v for k, v in exp.e3.items() if k not in drop_e3}, **(e3 or {})}
    patched = exp.model_copy(update={"spend_caps_usd": {**exp.spend_caps_usd, **(caps or {})}, "e3": block})
    monkeypatch.setattr(avail.config, "load_experiment", lambda *a, **k: patched)
    from avail.station.mj_plug import cable_shift                             # it took load_experiment at import

    monkeypatch.setattr(cable_shift, "load_experiment", lambda *a, **k: patched)
    return patched


NO_CAPS = {"E3_vlm_arm_latch": 0.0, "E3_vlm_arm_fric": 0.0}
WITH_VLM = {"arms": ["flag", "vlm", "avail", "avail_human"]}   # E3 has no judge arm since S5b: these tests add it


def test_the_plan_is_what_config_writes(monkeypatch):
    plan = cli.e3_plan("latch")
    assert list(plan["arms"]) == ["flag", "avail", "avail_human"]                # Kevin, S5b: no judge arm
    assert "vlm_cost" not in plan and "vlm_caps_short" not in plan            # so no cap and no cost to check
    _config(monkeypatch, e3=WITH_VLM)
    plan = cli.e3_plan("latch")
    arms = plan["arms"]
    assert list(arms) == ["flag", "vlm", "avail", "avail_human"]
    assert {a: s["cycles"] for a, s in arms.items()} == {"flag": 1000, "vlm": 300, "avail": 1000, "avail_human": 5000}
    assert all(s["seeds"][0] == 30000 for s in arms.values())                  # the arms share seeds
    assert arms["avail_human"]["seeds"] == [30000, 34999]
    assert plan["alpha"] == 0.01 and plan["split"] == "shift_test_e3"
    assert cli.e3_plan("fric")["arms"]["avail_human"]["cycles"] == 1000
    cost = plan["vlm_cost"]
    assert cost["session"] == "E3_vlm_arm_latch" and cost["estimate_usd"] == pytest.approx(300 * cost["usd_per_cycle"])
    assert cli.e3_plan("fric")["vlm_cost"]["session"] == "E3_vlm_arm_fric"         # each variant has its own session
    assert cost["judge"] == {"model": "fable", "prompt": "crop_ref"}              # the judge is config, which is hashed


def test_every_variants_cap_is_asked_for_before_the_1st_read(monkeypatch):
    """The config freezes at E3's 1st read, whichever variant makes it, so the plan names every short cap."""
    _config(monkeypatch, NO_CAPS, e3=WITH_VLM)
    plan = cli.e3_plan("latch")
    assert plan["vlm_caps_short"]["fric"]["session"] == "E3_vlm_arm_fric"
    warned = cli.other_caps_warning(plan)
    assert "set every variant's vlm cap now" in warned and "fric needs" in warned
    assert "latch needs" not in warned                                          # this arm's own cap is refused instead
    assert cli.other_caps_warning({"vlm_cost": {}}) is None
    plan.pop("vlm_cost")                                                        # --arm left this variant's vlm arm out
    assert "latch needs spend_caps_usd.E3_vlm_arm_latch of at least $70" in cli.other_caps_warning(plan)


def test_caps_that_cover_both_arms_clear_every_cap_message(monkeypatch):
    _config(monkeypatch, {"E3_vlm_arm_latch": 70.0, "E3_vlm_arm_fric": 70.0}, e3=WITH_VLM)   # 1.1 x $63: $69.30
    plan = cli.e3_plan("latch")
    assert "vlm_caps_short" not in plan
    assert cli.vlm_cost_problem(plan) is None and cli.other_caps_warning(plan) is None


def test_no_cost_per_cycle_names_the_missing_cost_and_no_cap(monkeypatch):
    """With no e3.vlm_usd_per_cycle, no cap can be checked, and raising one would clear nothing."""
    _config(monkeypatch, {"E3_vlm_arm_latch": 500.0, "E3_vlm_arm_fric": 500.0}, drop_e3=("vlm_usd_per_cycle",), e3=WITH_VLM)
    plan = cli.e3_plan("latch")
    assert "e3.vlm_usd_per_cycle" in cli.vlm_cost_problem(plan)
    assert cli.other_caps_warning(plan) is None                                  # this arm's refusal already says it
    plan.pop("vlm_cost")                                                        # --arm left the vlm arm out
    warned = cli.other_caps_warning(plan)
    assert "e3.vlm_usd_per_cycle" in warned and "spend_caps_usd" not in warned
    assert "no vlm arm can run" in cli.other_caps_warning(plan, started=True)
    started = cli.vlm_cost_problem(cli.e3_plan("latch"), started=True)         # the refusal names the cause after
    assert "e3.vlm_usd_per_cycle" in started and "--arm" in started            # the start too


def test_once_e3_has_read_its_seeds_a_short_cap_is_frozen(monkeypatch):
    """After E3's 1st read the seed guard refuses a changed config, so raising a cap is no longer an option."""
    _config(monkeypatch, NO_CAPS, e3=WITH_VLM)
    plan = cli.e3_plan("latch")
    said = cli.vlm_cost_problem(plan, started=True)
    assert "frozen" in said and "cannot run" in said and "Raise" not in said
    warned = cli.other_caps_warning(plan, started=True)
    assert "frozen" in warned and "fric (E3_vlm_arm_fric)" in warned and "set every" not in warned
    assert "frozen" in cli.vlm_cost_problem(_plan(None), started=True)
    assert cli.vlm_cost_problem(_plan(70.0, per_cycle=0.21), started=True) is None   # a cap that covers its arm runs


def _plan(cap, per_cycle=0.2, cycles=300):
    return {"arms": {"vlm": {"cycles": cycles}},
            "vlm_cost": {"session": "E3_vlm_arm_latch", "cap_usd": cap, "usd_per_cycle": per_cycle,
                         "estimate_usd": round(per_cycle * cycles, 2)}}


def test_a_vlm_arm_its_cap_cannot_cover_is_refused():
    assert "Raise the cap to at least $66" in cli.vlm_cost_problem(_plan(0.0))
    asked = cli.vlm_cost_problem(_plan(0.0, per_cycle=0.21))                     # 1.1 x $63 is $69.30: ask for $70
    assert "at least $70" in asked and cli.vlm_cost_problem(_plan(70.0, per_cycle=0.21)) is None
    assert cli.vlm_cost_problem(_plan(60.0)) is not None                        # no 10% margin over $60
    assert cli.vlm_cost_problem(_plan(66.0)) is None
    assert "no spend cap" in cli.vlm_cost_problem(_plan(None))
    assert cli.vlm_cost_problem(_plan(1000.0, per_cycle=float("nan"))) is not None
    assert "e3.vlm_usd_per_cycle" in cli.vlm_cost_problem(_plan(None, per_cycle=float("nan")))  # the cost comes 1st
    assert cli.vlm_cost_problem({"arms": {"flag": {"cycles": 10}}}) is None    # no vlm arm, nothing to pay


def test_the_run_refuses_before_any_seed_is_read(monkeypatch, capsys):
    def no_run(*args, **kwargs):
        raise AssertionError("run_shift must not be reached")

    import avail.seeds
    import avail.shift.runner

    monkeypatch.setattr(avail.shift.runner, "run_shift", no_run)
    monkeypatch.setattr(avail.seeds, "guard", no_run)
    monkeypatch.setattr(cli, "e3_plan", lambda variant: {**_plan(0.0), "arms": {"flag": {"cycles": 5, "seeds": [30000, 30004]},
                                                                               "vlm": {"cycles": 300, "seeds": [30000, 30299]}}})
    with pytest.raises(SystemExit):
        cli.main(["e3", "--variant", "latch", "--final-run", "E3"])
    assert "before the 1st arm" in capsys.readouterr().err
    with pytest.raises(SystemExit):                                            # no final-run name: refused too
        cli.main(["e3", "--variant", "latch"])
    with pytest.raises(SystemExit):
        cli.main(["e3", "--variant", "latch", "--arm", "nonesuch", "--final-run", "E3"])


def test_leaving_the_vlm_arm_out_needs_no_cap(monkeypatch):
    ran = []
    import avail.shift.runner

    monkeypatch.setattr(avail.shift.runner, "run_shift", lambda *a, **k: ran.append((a[2], len(a[3]), k["final_run"])))
    monkeypatch.setattr(cli, "e3_plan", lambda variant: {**_plan(0.0), "arms": {"flag": {"cycles": 5, "seeds": [30000, 30004]},
                                                                               "vlm": {"cycles": 300, "seeds": [30000, 30299]},
                                                                               "avail_human": {"cycles": 7, "seeds": [30000, 30006]}}})
    assert cli.main(["e3", "--variant", "latch", "--arm", "avail_human", "--arm", "flag", "--final-run", "E3"]) == 0
    assert ran == [("flag", 5, "E3"), ("avail_human", 7, "E3")]                 # in the plan's order, vlm left out


def test_the_vlm_arm_runs_last(monkeypatch):
    """A judge call refused at its cap stops the command, so the arms that need no judge come 1st."""
    ran = []
    import avail.shift.runner

    monkeypatch.setattr(avail.shift.runner, "run_shift", lambda *a, **k: ran.append(a[2]))
    monkeypatch.setattr(cli, "e3_plan", lambda variant: {**_plan(100.0), "arms": {
        "flag": {"cycles": 5, "seeds": [30000, 30004]}, "vlm": {"cycles": 300, "seeds": [30000, 30299]},
        "avail": {"cycles": 5, "seeds": [30000, 30004]}, "avail_human": {"cycles": 7, "seeds": [30000, 30006]}}})
    assert cli.main(["e3", "--variant", "latch", "--final-run", "E3"]) == 0
    assert ran == ["flag", "avail", "avail_human", "vlm"]


def test_a_changed_setup_stops_the_next_arm(monkeypatch, shifts_in_tmp):
    """The seed guard hashes config/experiment.yaml only. The task file and the testbed's setup files are recorded at
    E3's 1st run, and an arm of either variant that finds 1 of them changed is refused."""
    ran = []
    import avail.shift.runner

    monkeypatch.setattr(avail.shift.runner, "run_shift", lambda *a, **k: ran.append((a[1], a[2])))
    monkeypatch.setattr(cli, "e3_plan", lambda variant: {"variant": variant, "arms": {"flag": {"cycles": 5, "seeds": [30000, 30004]}}})
    hashes = {"tasks/cable_loop.yaml": "aaaaaaaaaaaa", "config/cable_cell.yaml": "bbbbbbbbbbbb"}
    monkeypatch.setattr(cli, "setup_hashes", lambda task: dict(hashes))
    assert cli.main(["e3", "--variant", "latch", "--final-run", "E3"]) == 0          # the 1st run records the setup
    assert json.loads((shifts_in_tmp / "e3_setup.json").read_text(encoding="utf-8"))["files"] == hashes
    assert cli.main(["e3", "--variant", "fric", "--final-run", "E3"]) == 0           # the same setup: the next arm runs
    hashes["config/cable_cell.yaml"] = "cccccccccccc"
    with pytest.raises(SystemExit):
        cli.main(["e3", "--variant", "latch", "--final-run", "E3"])
    assert ran == [("latch", "flag"), ("fric", "flag")]


def test_a_setup_changed_while_an_arm_runs_stops_the_next_arm(monkeypatch):
    """The setup is checked again before each arm, so an edit made while 1 arm runs reaches no later arm."""
    ran = []
    hashes = {"tasks/cable_loop.yaml": "aaaaaaaaaaaa"}
    import avail.shift.runner

    def run(*a, **k):
        ran.append(a[2])
        hashes["tasks/cable_loop.yaml"] = "dddddddddddd"                       # the task file is edited mid-arm

    monkeypatch.setattr(avail.shift.runner, "run_shift", run)
    monkeypatch.setattr(cli, "e3_plan", lambda variant: {"variant": variant, "arms": {
        "flag": {"cycles": 5, "seeds": [30000, 30004]}, "avail": {"cycles": 5, "seeds": [30000, 30004]}}})
    monkeypatch.setattr(cli, "setup_hashes", lambda task: dict(hashes))
    with pytest.raises(SystemExit):
        cli.main(["e3", "--variant", "latch", "--final-run", "E3"])
    assert ran == ["flag"]


def test_a_variant_outside_e3_is_refused_before_anything(monkeypatch, capsys):
    """The variant check comes before the plan, the hashes, the witness check and any read, so its refusal is the
    1st thing that can stop the command."""
    import avail.shift.runner

    def ran(*a, **k):
        raise AssertionError("something ran before the variant check")

    monkeypatch.setattr(avail.shift.runner, "run_shift", ran)
    for name in ("e3_plan", "setup_hashes", "e3_started", "prepare_problem", "e3_config_problem", "station_problem",
                 "judge_problem"):
        monkeypatch.setattr(cli, name, ran)
    for argv in (["e3", "--variant", "prong", "--final-run", "E3"], ["e3", "--variant", "prong", "--plan"]):
        with pytest.raises(SystemExit):
            cli.main(argv)
        assert "e3.variants" in capsys.readouterr().err


def test_a_missing_witness_is_refused_before_any_seed_is_read(monkeypatch, capsys, shifts_in_tmp):
    """A final run never builds a witness, so a missing 1 must stop E3 before its 1st arm, with nothing recorded."""
    import avail.shift.runner

    monkeypatch.setattr(avail.shift.runner, "run_shift", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran")))
    monkeypatch.setattr(cli, "prepare_problem", lambda *a: "fric has no seated witness")
    monkeypatch.setattr(cli, "e3_plan", lambda variant: {"variant": variant, "arms": {"flag": {"cycles": 5, "seeds": [30000, 30004]}}})
    with pytest.raises(SystemExit):
        cli.main(["e3", "--variant", "latch", "--final-run", "E3"])
    assert "fric has no seated witness" in capsys.readouterr().err
    assert not (shifts_in_tmp / "e3_setup.json").exists()
    assert cli.main(["e3", "--variant", "latch", "--plan"]) == 0
    assert "before the run: fric has no seated witness" in capsys.readouterr().out


def test_the_testbed_finds_what_prepare_has_not_made(monkeypatch, tmp_path):
    """A missing or stale witness, or a missing judge reference picture, is named before E3's 1st arm, and the check
    creates no folder. The paths are the real ones, under a scratch data root."""
    from types import SimpleNamespace

    import avail.witness.store
    from avail.station.mj_plug import cable_shift

    monkeypatch.setenv("AVAIL_DATA", str(tmp_path))
    _config(monkeypatch, e3=WITH_VLM)                                          # a judge arm needs its pictures
    refs = cable_shift.judge_reference_files()
    assert set(refs) == {"latch", "fric"} and all(tmp_path in path.parents for path in refs.values())
    said = cable_shift.prepare_problem(["latch", "fric"], 0.01)
    assert "latch has no latched witness" in said and "fric has no seated witness" in said
    assert "latch has no judge reference picture" in said and "fric has no judge reference picture" in said
    assert "avail shift prepare" not in said and "put back" not in said                 # the advice is the CLI's
    assert not (tmp_path / "witness_e3").exists() and not (tmp_path / "cache").exists()  # it creates no folder
    for path in refs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
    for rel in ("latch/latched/v1.json", "latch/latched/v2.json", "fric/seated/v1.json"):
        (tmp_path / "witness_e3" / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "witness_e3" / rel).write_text("{}", encoding="utf-8")
    names = [d.name for d in cable_shift.cable_stages()]
    alphas = {"latch": 0.01, "fric": 0.02}                                      # fric's was built at another alpha

    def load(connector, predicate, version=None, root=None):
        return SimpleNamespace(alpha=alphas[connector], stage_defs=[SimpleNamespace(name=n) for n in names],
                               calibration={"split_id": "cable_calibration"}), {}

    monkeypatch.setattr(avail.witness.store, "load", load)
    said = cable_shift.prepare_problem(["latch", "fric"], 0.01)
    assert "fric seated witness (v1)" in said and "latch" not in said
    alphas["fric"] = 0.01
    assert cable_shift.prepare_problem(["latch", "fric"], 0.01) is None
    monkeypatch.setattr(cable_shift, "judge_reference_files", lambda: {**refs, "fric": tmp_path / "never_built.png"})
    assert "fric has no judge reference picture" in cable_shift.prepare_problem(["latch", "fric"], 0.01)


def test_the_judge_reference_is_needed_only_by_a_crop_ref_judge_arm(monkeypatch):
    from avail.station.mj_plug import cable_shift

    assert cable_shift.judge_reference_files() == {}                           # S5b: E3 has no judge arm
    _config(monkeypatch, e3=WITH_VLM)
    assert set(cable_shift.judge_reference_files()) == {"latch", "fric"}          # fable on crop_ref
    _config(monkeypatch, e3={**WITH_VLM, "vlm_judge": {"model": "fable", "prompt": "full"}})
    assert cable_shift.judge_reference_files() == {}
    _config(monkeypatch, e3={"arms": ["flag", "avail", "avail_human"]})
    assert cable_shift.judge_reference_files() == {}


def test_a_final_run_never_builds_a_judge_reference(monkeypatch, tmp_path):
    """As with witnesses: a picture built mid-run would be an input the setup record never saw."""
    import avail.zoo.judges
    from avail.seeds import FINAL_RUN_ENV
    from avail.station.mj_plug import cable_shift

    monkeypatch.setattr(avail.zoo.judges, "reference_path", lambda connector, profile="easy", create=True: tmp_path / "none.png")
    monkeypatch.setattr(avail.zoo.judges, "reference_image", lambda *a, **k: (_ for _ in ()).throw(AssertionError("built")))
    monkeypatch.setenv(FINAL_RUN_ENV, "E3")
    with pytest.raises(RuntimeError, match="avail shift prepare"):
        cable_shift.make_judge().collect(None, None, {"connector": "fric"})


def test_a_config_changed_after_e3_started_is_the_only_thing_said(monkeypatch, capsys, shifts_in_tmp):
    """The seed guard will refuse every arm, so the cap, witness and setup messages, which read the config as it is
    now, are held back."""
    import avail.shift.runner

    monkeypatch.setattr(avail.shift.runner, "run_shift", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran")))
    monkeypatch.setattr(cli, "e3_started", lambda: True)
    monkeypatch.setattr(cli, "e3_config_problem", lambda: "config/experiment.yaml changed since E3 started")
    monkeypatch.setattr(cli, "prepare_problem", lambda *a: "fric has no seated witness")   # each would speak
    shifts_in_tmp.mkdir(parents=True, exist_ok=True)
    (shifts_in_tmp / "e3_setup.json").write_text(json.dumps({"files": {"tasks/cable_loop.yaml": "x"}}),
                                                 encoding="utf-8")
    short = {"fric": {"session": "E3_vlm_arm_fric", "cap_usd": 0.0, "estimate_usd": 63.0}}
    monkeypatch.setattr(cli, "e3_plan", lambda variant: {**_plan(0.0), "variant": variant, "vlm_caps_short": short,
                                                         "arms": {"flag": {"cycles": 5, "seeds": [30000, 30004]},
                                                                  "vlm": {"cycles": 300, "seeds": [30000, 30299]}}})
    plan = cli.e3_plan("latch")
    assert cli.vlm_cost_problem(plan, True) and cli.other_caps_warning(plan, True)  # each would speak if asked
    assert cli.main(["e3", "--variant", "latch", "--plan"]) == 0
    out = capsys.readouterr().out
    said = out.split("before the run")[1:]
    assert len(said) == 1 and "config/experiment.yaml changed since E3 started" in said[0]
    with pytest.raises(SystemExit):
        cli.main(["e3", "--variant", "latch", "--final-run", "E3"])
    err = capsys.readouterr().err
    assert "config/experiment.yaml changed since E3 started" in err and "The arms share seeds" not in err


def test_what_to_do_about_a_missing_witness_depends_on_the_start(monkeypatch, capsys):
    monkeypatch.setattr(cli, "prepare_problem", lambda *a: "fric has no seated witness")
    monkeypatch.setattr(cli, "e3_plan", lambda variant: {"variant": variant, "arms": {"flag": {"cycles": 5, "seeds": [30000, 30004]}}})
    assert cli.main(["e3", "--variant", "latch", "--plan"]) == 0
    before = capsys.readouterr().out
    assert "avail shift prepare" in before and "put back" not in before
    monkeypatch.setattr(cli, "e3_started", lambda: True)                       # after the 1st read, prepare would
    assert cli.main(["e3", "--variant", "latch", "--plan"]) == 0             # change the recorded setup
    after = capsys.readouterr().out
    assert "put back what changed" in after and "avail shift prepare" not in after


def test_shift_run_cannot_read_the_e3_split(monkeypatch, tmp_path):
    """Only `avail shift e3` reads E3's seeds: it records the setup that every cycle is then checked against."""
    import avail.seeds
    from avail.shift import runner

    monkeypatch.setattr(avail.seeds, "guard", lambda *a, **k: (_ for _ in ()).throw(AssertionError("read")))
    with pytest.raises(SystemExit):
        cli.main(["run", "--variant", "latch", "--arm", "flag", "--seeds", "30000:30010", "--shift", "x",
                  "--final-run", "E3"])
    with pytest.raises(ValueError, match="expected_setup"):             # nor can any caller of the runner
        runner.run_shift("cable_loop", "latch", "flag", [30000], shift="x", run_root=tmp_path, final_run="E3")
    with pytest.raises(ValueError, match="names no testbed"):
        runner.run_shift("plug_insert", "latch", "flag", [20000], shift="x", run_root=tmp_path)


def test_a_task_with_no_testbed_is_refused_before_anything(monkeypatch, capsys):
    monkeypatch.setattr(cli, "e3_plan", lambda *a: (_ for _ in ()).throw(AssertionError("planned")))
    for argv in (["e3", "--task", "plug_insert", "--variant", "latch", "--plan"],
                 ["e3", "--task", "plug_insert", "--variant", "latch", "--arm", "flag", "--final-run", "E3"]):
        with pytest.raises(SystemExit):
            cli.main(argv)
        assert "names no testbed" in capsys.readouterr().err


def test_a_started_e3_with_no_setup_record_is_refused(monkeypatch, capsys, shifts_in_tmp):
    import avail.shift.runner

    monkeypatch.setattr(avail.shift.runner, "run_shift", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran")))
    monkeypatch.setattr(cli, "e3_started", lambda: True)
    monkeypatch.setattr(cli, "e3_plan", lambda variant: {"variant": variant, "arms": {"flag": {"cycles": 5, "seeds": [30000, 30004]}}})
    with pytest.raises(SystemExit):
        cli.main(["e3", "--variant", "latch", "--final-run", "E3"])
    assert "setup record" in capsys.readouterr().err
    assert not (shifts_in_tmp / "e3_setup.json").exists()                     # no new record stands in for it


def test_the_final_run_says_what_a_short_cap_will_cost(monkeypatch, capsys):
    """In the final run it is too late to set a cap: the 1st read freezes the config. So it says what is lost."""
    import avail.shift.runner

    seen = []
    monkeypatch.setattr(avail.shift.runner, "run_shift", lambda *a, **k: seen.append(k.get("expected_setup")))
    _config(monkeypatch, {"E3_vlm_arm_latch": 70.0, "E3_vlm_arm_fric": 0.0}, e3=WITH_VLM)
    assert cli.main(["e3", "--variant", "latch", "--final-run", "E3"]) == 0
    out = capsys.readouterr().out
    assert "will never run: fric (spend_caps_usd.E3_vlm_arm_fric is $0.00, its arm needs at least $70)" in out
    assert "set every variant's vlm cap now" not in out
    assert len(seen) == 4 and all(s == seen[0] and s for s in seen)            # every arm carries the record


def test_a_cycle_counts_only_under_the_recorded_setup(monkeypatch, tmp_path):
    """The runner checks a final run's setup before each cycle and before the cycle's end is written."""
    import sys
    import types

    import avail.core.task
    from avail.shift import runner, setup_record

    now = {"tasks/cable_loop.yaml": "aaaaaaaaaaaa", setup_record.CODE_KEY: "cccccccccccc"}
    monkeypatch.setattr(setup_record, "setup_hashes", lambda task: dict(now))
    runner.same_setup(None, "cable_loop", "x", "before seed 1")                  # a dev shift: no check
    runner.same_setup(dict(now), "cable_loop", "x", "before seed 1")             # the same setup: no problem
    with pytest.raises(RuntimeError, match="code: avail"):
        runner.same_setup({**now, setup_record.CODE_KEY: "dddddddddddd"}, "cable_loop", "x", "while seed 1 ran")
    bed = types.ModuleType("fake_testbed")
    bed.make_cell = lambda *a, **k: (_ for _ in ()).throw(AssertionError("a cycle started"))
    monkeypatch.setitem(sys.modules, "fake_testbed", bed)
    monkeypatch.setattr(avail.core.task, "load_task", lambda name: types.SimpleNamespace(testbed="fake_testbed"))
    args = ("cable_loop", "flag", "latch", "easy", "x_b20000", [20000], str(tmp_path), None, 0.01,
            {**now, "tasks/cable_loop.yaml": "bbbbbbbbbbbb"})
    with pytest.raises(RuntimeError, match="tasks/cable_loop.yaml changed before seed 20000"):
        runner.run_block(args)


def test_a_cycle_that_ran_under_a_change_gets_no_end(monkeypatch, tmp_path):
    """The 2nd check, before the cycle's end is written: a file edited while the cycle ran cuts it off."""
    import types

    import avail.shift.driver
    from avail.shift import runner, setup_record

    now = {"tasks/cable_loop.yaml": "aaaaaaaaaaaa"}
    expected, ended, closed = dict(now), [], []

    class Driver:
        def __init__(self, *a, **k):
            pass

        def run(self):
            now["tasks/cable_loop.yaml"] = "bbbbbbbbbbbb"                      # the task file is edited mid-cycle
            return types.SimpleNamespace(cycle_id="x_b20000_c20000")

    bed = types.ModuleType("fake_testbed_2")
    bed.make_cell = lambda *a, **k: "cell"
    bed.evaluator = lambda cell: None
    bed.evaluate = lambda cell: {}
    bed.close = closed.append
    import sys

    monkeypatch.setitem(sys.modules, "fake_testbed_2", bed)
    import avail.core.task

    monkeypatch.setattr(avail.core.task, "load_task", lambda name: types.SimpleNamespace(testbed="fake_testbed_2"))
    monkeypatch.setattr(avail.shift.driver, "CycleDriver", Driver)
    monkeypatch.setattr(setup_record, "setup_hashes", lambda task: dict(now))
    monkeypatch.setattr(runner, "_cycle_end", lambda *a, **k: ended.append(a))
    with pytest.raises(RuntimeError, match="changed while seed 20000 ran .*reruns it under a new id"):
        runner.run_block(("cable_loop", "flag", "latch", "easy", "x_b20000", [20000], str(tmp_path), None, 0.01, expected))
    assert ended == [] and closed == ["cell"]                                  # no end written, the cell closed


def test_every_block_of_a_final_run_carries_the_record(monkeypatch, tmp_path):
    import avail.seeds
    from avail.seeds import FINAL_RUN_ENV
    from avail.shift import runner

    monkeypatch.setenv(FINAL_RUN_ENV, "")                                      # restored after the test
    monkeypatch.setattr(avail.seeds, "guard", lambda *a, **k: None)            # no read is recorded here
    seen = []
    monkeypatch.setattr(runner, "run_block", lambda a: seen.append(a) or {"run_id": a[4], "cycles": 0, "skipped": 0,
                                                                            "wall_s": 0.0})
    record = {"tasks/cable_loop.yaml": "aaaaaaaaaaaa"}
    runner.run_shift("cable_loop", "latch", "flag", list(range(30000, 30025)), shift="x", run_root=tmp_path,
                     workers=1, final_run="E3", expected_setup=record)
    assert len(seen) == 3 and all(a[-1] == record for a in seen)               # 3 blocks of up to 10 seeds


def test_the_refusal_says_whether_the_cycle_ran():
    from avail.shift import runner, setup_record

    with pytest.raises(RuntimeError) as before:
        runner.same_setup({"x": "1"}, "cable_loop", "r", "before seed 1")
    assert "new id" not in str(before.value) and "goes on from that seed" in str(before.value)
    with pytest.raises(RuntimeError) as during:
        runner.same_setup({"x": "1"}, "cable_loop", "r", "while seed 1 ran", ran=True)
    assert "reruns it under a new id" in str(during.value)
    assert setup_record.CODE_KEY in str(before.value)                          # the record names the code too


def test_a_missing_robot_model_is_said_on_its_own(monkeypatch, capsys):
    """git does not track the robot model, so a worktree may lack it. prepare cannot make it, so its advice is not added."""
    import avail.shift.runner

    monkeypatch.setattr(avail.shift.runner, "run_shift", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran")))
    monkeypatch.setattr(cli, "station_problem", lambda *a: "the station's robot model is missing (x)")
    monkeypatch.setattr(cli, "e3_plan", lambda variant: {"variant": variant, "arms": {"flag": {"cycles": 5, "seeds": [30000, 30004]}}})
    assert cli.main(["e3", "--variant", "latch", "--plan"]) == 0
    out = capsys.readouterr().out
    assert "before the run: the station's robot model is missing (x)\n" in out and "avail shift prepare" not in out
    with pytest.raises(SystemExit):
        cli.main(["e3", "--variant", "latch", "--final-run", "E3"])
    assert "robot model is missing" in capsys.readouterr().err


def _no_aspire() -> bool:
    from avail.paths import aspire_station_xml
    return not aspire_station_xml().is_file()


needs_aspire = pytest.mark.skipif(_no_aspire(), reason="needs the ASPIRE clone: uv run python scripts/setup_aspire.py")


@needs_aspire
def test_the_testbed_names_its_robot_model(monkeypatch, tmp_path):
    import avail.paths
    from avail.station.mj_plug import cable_shift

    assert cable_shift.station_problem() is None                               # the repo's model is in place
    files = cable_shift.station_files()
    names = {f.name for f in files}
    assert {"station.xml", "station.yaml"} <= names and any(n.endswith(".stl") for n in names)
    monkeypatch.setattr(avail.paths, "aspire_station_xml", lambda: tmp_path / "none" / "station.xml")
    assert "robot model is missing" in cable_shift.station_problem()
    assert all(f.name != "station.xml" for f in cable_shift.station_files())    # only files that exist are listed


def test_a_judge_that_cannot_be_called_is_refused_before_the_read(monkeypatch, capsys):
    """The vlm arm runs last, so a missing key would show only after the other arms read the split."""
    import avail.shift.runner

    monkeypatch.setattr(avail.shift.runner, "run_shift", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran")))
    monkeypatch.setattr(cli, "judge_problem", lambda *a: "the E3 judge fable cannot be called now: no key")
    monkeypatch.setattr(cli, "e3_plan", lambda variant: {**_plan(100.0), "variant": variant, "arms": {
        "flag": {"cycles": 5, "seeds": [30000, 30004]}, "vlm": {"cycles": 300, "seeds": [30000, 30299]}}})
    assert cli.main(["e3", "--variant", "latch", "--plan"]) == 0
    assert "before the run: the E3 judge fable cannot be called now: no key" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["e3", "--variant", "latch", "--final-run", "E3"])
    assert "cannot be called now" in capsys.readouterr().err
    monkeypatch.setattr(cli, "judge_problem", lambda *a: (_ for _ in ()).throw(AssertionError("asked")))
    ran = []
    monkeypatch.setattr(avail.shift.runner, "run_shift", lambda *a, **k: ran.append(a[2]))
    assert cli.main(["e3", "--variant", "latch", "--arm", "flag", "--final-run", "E3"]) == 0   # no vlm arm, no question
    assert ran == ["flag"]


def test_the_testbed_says_why_its_judge_cannot_be_called(monkeypatch):
    import avail.router.smoke
    from avail.station.mj_plug import cable_shift

    monkeypatch.setattr(avail.router.smoke, "skip_reason", lambda entry: "no key: set OPENROUTER_API_KEY in .env")
    assert cable_shift.judge_problem() == ("the E3 judge fable cannot be called now: no key: set OPENROUTER_API_KEY "
                                           "in .env")
    monkeypatch.setattr(avail.router.smoke, "skip_reason", lambda entry: None)
    assert cable_shift.judge_problem() is None
    _config(monkeypatch, e3={"vlm_judge": {"model": "nonesuch", "prompt": "crop_ref"}})
    assert "has no model nonesuch" in cable_shift.judge_problem()


def test_a_key_the_router_cannot_send_is_named_before_the_read(monkeypatch):
    """A key with a space or a character that is not ASCII is present, and still refused at the judge's 1st call."""
    from avail.router.models import default_models
    from avail.router.smoke import skip_reason

    entry = default_models().models["fable"]
    stand_in = "sk-or-v1-0123456789abcdef"                                      # not a key: a value of the same shape
    monkeypatch.setenv("OPENROUTER_API_KEY", stand_in + "​")              # a zero-width space survives strip()
    said = skip_reason(entry)
    assert said and "OPENROUTER_API_KEY" in said and stand_in not in said      # names the variable, never the value
    assert "comes from the environment" in said and "env.ps1" in said         # where a worktree's run takes it from
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1 0123456789")
    assert "a space" in skip_reason(entry)
    monkeypatch.setenv("OPENROUTER_API_KEY", stand_in)
    assert skip_reason(entry) is None


def test_a_malformed_key_from_dotenv_is_sent_back_to_dotenv(monkeypatch):
    from avail import config
    from avail.router.core import _check_key

    monkeypatch.delenv("STAND_IN_KEY", raising=False)
    monkeypatch.setattr(config, "dotenv_values", lambda *a, **k: {"STAND_IN_KEY": "abc def"})
    with pytest.raises(config.MissingCredentialError) as caught:
        _check_key("abc def", ["STAND_IN_KEY"], "fable", "openrouter")
    assert f"Re-enter the key in {config.ENV_FILE}" in str(caught.value) and "abc def" not in str(caught.value)


def test_the_code_fingerprint_follows_the_code(monkeypatch, tmp_path):
    from avail.shift import setup_record

    (tmp_path / "shift").mkdir()
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "shift" / "a.py").write_bytes(b"x = 1\n")
    monkeypatch.setattr(setup_record, "PACKAGE", tmp_path)
    before = setup_record.code_fingerprint()
    (tmp_path / "__pycache__" / "a.py").write_bytes(b"cached")                 # not code that runs
    (tmp_path / "shift" / "a.py").write_bytes(b"x = 1\r\n")                    # line endings do not count
    assert setup_record.code_fingerprint() == before
    (tmp_path / "shift" / "a.py").write_bytes(b"x = 2\n")
    assert setup_record.code_fingerprint() != before


def test_the_config_check_compares_with_the_config_e3_read_under(monkeypatch):
    import avail.config
    import avail.seeds

    reads = [{"split": "zoo_test_hard2", "config_hash": "zzzzzzzzzzzz"}]          # another split's reads do not count
    monkeypatch.setattr(avail.seeds, "test_seed_reads", lambda *a, **k: list(reads))
    monkeypatch.setattr(avail.config, "config_hash", lambda *a, **k: "bbbbbbbbbbbb")
    check = E3_CONFIG_PROBLEM
    assert check() is None
    reads.append({"split": "shift_test_e3", "config_hash": "bbbbbbbbbbbb"})
    assert check() is None                                                    # the same config: no problem
    reads.append({"split": "shift_test_e3", "config_hash": "aaaaaaaaaaaa"})
    said = check()
    assert "aaaaaaaaaaaa" in said and "bbbbbbbbbbbb" in said and "Put the config back" in said


@needs_aspire
def test_the_setup_covers_the_task_file_and_the_testbeds_files():
    hashes = cli.setup_hashes("cable_loop")
    from avail.shift.setup_record import CODE_KEY

    repo = {"tasks/cable_loop.yaml", "config/experiment.yaml", "config/cable_cell.yaml",
            "avail/station/mj_plug/cable_palette.json", "config/models.yaml", CODE_KEY}
    assert repo <= set(hashes)
    # besides those, only the witness and the judge reference each E3 variant loads, once `avail shift prepare` has
    # built them, and the station's robot model under third_party/ASPIRE
    assert all((k.startswith("AVAIL_DATA/witness_e3/") and k.endswith(".json"))
               or (k.startswith("AVAIL_DATA/cache/reference/") and k.endswith(".png"))
               or k.startswith("third_party/ASPIRE/") for k in set(hashes) - repo)
    assert any(k.endswith("/station.xml") for k in hashes) and any(k.endswith(".stl") for k in hashes)  # the robot
    assert all(len(h) == 12 for h in hashes.values())


def test_the_setup_names_the_latest_witness_of_each_variant(monkeypatch, tmp_path):
    from avail.station.mj_plug import cable_shift

    monkeypatch.setattr(cable_shift, "data_root", lambda create=False: tmp_path)
    for name in ("v1.json", "v3.json"):
        path = tmp_path / "witness_e3" / "latch" / "latched" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    files = cable_shift.setup_files()
    witnesses = [f for f in files if "witness_e3" in str(f)]
    assert [f.name for f in witnesses] == ["v3.json"]                          # fric has none built: nothing to name


def test_the_plan_alone_reads_nothing(monkeypatch, capsys):
    import avail.seeds

    monkeypatch.setattr(avail.seeds, "guard", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no read")))
    assert cli.main(["e3", "--variant", "latch", "--plan"]) == 0
    out = capsys.readouterr().out
    assert '"avail_human"' in out and "5000" in out
