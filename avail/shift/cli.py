"""``avail shift``: run a task's cycles on its testbed, and report them from the ledger alone.

    avail shift prepare --task cable_loop --variant latch            build the witnesses the cycles use (no test seed)
    avail shift run --task cable_loop --variant latch --arm avail_human --seeds 20000:20100 --shift dev1 --workers 2
    avail shift report --shift dev1                                   the shift card and the E3 table, from the ledgers
    avail shift export --shift dev1                                   intervention episodes, LeRobot-style, and labels
    avail shift e3 --variant latch --plan                            the E3 final run: its arms, seeds and cost, no cycle
    avail shift e3 --variant latch --final-run E3                    the E3 final run itself. Kevin starts it

Shift dev seeds are 20,000 to 23,999. The E3 test seeds (30,000 to 37,999) are read once, by
``avail shift e3 ... --final-run E3`` alone: ``run`` refuses ``--final-run E3``. Every run is resumable:
run the same command again to pick up where it stopped.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .setup_record import changed_keys, setup_hashes


def _seeds(text: str) -> list[int]:
    lo, hi = text.split(":")
    return list(range(int(lo), int(hi)))


def _shift_dir(name: str) -> Path:
    from ..paths import runs_dir

    return runs_dir() / "shifts" / name


FINAL_RUN = "E3"
MARGIN = 1.1                                     # a vlm cap must cover its estimate with 10% to spare


def judge_session(variant: str) -> str:
    """The spend session one variant's E3 vlm arm is charged to. Each variant has its own, so the 2nd variant's arm
    cannot find its cap spent by the 1st: a session's cap counts everything ever charged to it."""
    return f"{FINAL_RUN}_vlm_arm_{variant}"


def e3_plan(variant: str) -> dict:
    """The E3 final run as config writes it: arms, cycles per arm, the seeds they share, and the vlm arm's judge,
    session and cost."""
    from ..config import load_experiment

    exp = load_experiment()
    e3, seeds = exp.e3, exp.seeds.shift_test_e3
    lo = int(seeds[0])
    arms = {}
    for arm in e3["arms"]:
        n = int(e3["cycles_per_arm"])
        if arm == "avail_human":
            n = int(e3["avail_human_cycles"].get(variant, e3["avail_human_cycles"]["default"]))
        if arm == "vlm":
            n = int(e3["vlm_arm_cycles"])
        arms[arm] = {"cycles": n, "seeds": [lo, lo + n - 1]}
    plan = {"variant": variant, "arms": arms, "alpha": exp.customer_bar["alpha_for_bar"], "split": "shift_test_e3"}
    per_cycle = float(e3.get("vlm_usd_per_cycle", float("nan")))
    if "vlm" in arms:
        session = judge_session(variant)
        plan["vlm_cost"] = {"session": session, "cap_usd": exp.spend_caps_usd.get(session),
                            "judge": dict(e3.get("vlm_judge", {})), "usd_per_cycle": per_cycle,
                            "estimate_usd": round(per_cycle * arms["vlm"]["cycles"], 2)}
    # the config freezes at E3's 1st read, whichever variant makes it, so every variant's cap is set before it,
    # this one's too, even when this command leaves its vlm arm out
    short = {}
    if "vlm" in e3["arms"]:
        need = round(per_cycle * int(e3["vlm_arm_cycles"]), 2)
        for each in dict.fromkeys([*e3.get("variants", []), variant]):
            cap = exp.spend_caps_usd.get(judge_session(each))
            if not need == need or cap is None or float(cap) < MARGIN * need:
                short[each] = {"session": judge_session(each), "cap_usd": cap, "estimate_usd": need}
    if short:
        plan["vlm_caps_short"] = short
    return plan


def vlm_cost_problem(plan: dict, started: bool = False) -> str | None:
    """Why this command's vlm arm cannot start, or None. Checked before any test seed is read, since the seed guard
    refuses a changed config once the split has been read. ``started``: E3 has read its split already, so its caps
    are frozen and a short cap can only mean leaving the vlm arm out."""
    cost = plan.get("vlm_cost")
    if cost is None:
        return None
    cap, need = cost["cap_usd"], cost["estimate_usd"]
    frozen = ("E3 has started, so its config is frozen: this vlm arm cannot run. Leave it out by naming the other arms "
              "with --arm")
    if not need == need:                                       # NaN: no measured cost per cycle, so no cap can be sized
        if started:
            return ("E3 has started with no e3.vlm_usd_per_cycle in its frozen config, so no vlm arm can run. Leave this "
                    "one out by naming the other arms with --arm")
        return ("config has no e3.vlm_usd_per_cycle, so the vlm arm's cost is unknown. Measure it on dev seeds and add "
                "it before the 1st arm, or leave the vlm arm out by naming the other arms with --arm")
    if cap is None:
        return frozen if started else (f"config has no spend cap for {cost['session']}. Add spend_caps_usd."
                                       f"{cost['session']} before the 1st arm")
    if float(cap) < MARGIN * need:                             # a cap without a 10% margin over the estimate
        said = (f"the vlm arm's {plan['arms']['vlm']['cycles']} cycles are estimated at ${need:.2f} "
                f"(${cost['usd_per_cycle']:.3f} per cycle, e3.vlm_usd_per_cycle), and spend_caps_usd.{cost['session']} "
                f"is ${float(cap):.2f}. ")
        if started:
            return said + frozen
        return said + (f"Raise the cap to at least ${math.ceil(MARGIN * need)} before the 1st arm, or leave the vlm arm "
                       "out by naming the other arms with --arm")
    return None


def e3_started() -> bool:
    """Whether E3 has read its split: from then on the seed guard refuses a changed config, so its caps are frozen."""
    from ..seeds import FINAL_RUNS, test_seed_reads

    return any(r.get("split") == FINAL_RUNS[FINAL_RUN] for r in test_seed_reads())


def e3_config_problem() -> str | None:
    """Why the seed guard would refuse every E3 arm, or None: config/experiment.yaml is no longer the config E3 read its
    split under. The cap messages judge the config as it is now, so they hold only while this is None."""
    from ..seeds import config_drift

    drift = config_drift(FINAL_RUN)
    if drift:
        then, now = drift
        return (f"config/experiment.yaml changed since E3 started: E3 read its split under {', '.join(then)} and the "
                f"config is now {now}, so the seed guard will refuse every arm. Put the config back as it was")
    return None


def station_problem(task_name: str) -> str | None:
    """Why a final run could not build its testbed's world, or None: the testbed's ``station_problem`` hook, when it has
    one, for files no command here makes, such as a robot model that git does not track."""
    import importlib

    from ..core.task import load_task

    task = load_task(task_name)
    bed = importlib.import_module(task.testbed) if task.testbed else None
    hook = getattr(bed, "station_problem", None)
    return hook() if hook else None


def judge_problem(task_name: str) -> str | None:
    """Why the vlm arm's judge could not be called, or None: the testbed's ``judge_problem`` hook, when it has one. The
    vlm arm runs last, so a missing key would otherwise surface only after the other arms had read the split."""
    import importlib

    from ..core.task import load_task

    task = load_task(task_name)
    bed = importlib.import_module(task.testbed) if task.testbed else None
    hook = getattr(bed, "judge_problem", None)
    return hook() if hook else None


def prepare_problem(task_name: str, variants: list[str], alpha: float | None) -> str | None:
    """Why a final run would stop for want of something ``avail shift prepare`` makes, or None: the testbed's
    ``prepare_problem`` hook, when it has one. A final run builds no witness and no reference, so 1 missing would stop E3
    after an earlier arm had read the split, and making it then would change the setup the E3 record holds."""
    import importlib

    from ..core.task import load_task

    task = load_task(task_name)
    bed = importlib.import_module(task.testbed) if task.testbed else None
    hook = getattr(bed, "prepare_problem", None)
    return hook(list(variants), float(alpha)) if hook else None


def setup_problem(record: Path, now: dict[str, str]) -> str | None:
    """Why E3 cannot go on with this setup, or None. The 1st run writes ``record``. Every later arm, of either
    variant, must find the same files and code with the same hashes, since the arms share seeds. Checked before each
    arm here, and before each cycle by the shift runner."""
    if not record.is_file():
        return None
    changed = changed_keys(json.loads(record.read_text(encoding="utf-8"))["files"], now)
    if changed:
        return (f"{', '.join(changed)} changed since E3 started ({record}). The arms share seeds: put them back as they "
                "were, or E3 needs new, unread test seeds")
    return None


def other_caps_warning(plan: dict, started: bool = False, final: bool = False) -> str | None:
    """A warning for each variant whose vlm cap would not cover its arm, besides a vlm arm this command runs (that one
    is refused): the config cannot change after E3's 1st read. ``final``: this command is about to make that read, so
    the warning says what it will cost, as it is too late to set a cap."""
    short = dict(plan.get("vlm_caps_short", {}))
    if "vlm_cost" in plan:
        short.pop(plan.get("variant"), None)
    if not short:
        return None
    if any(s["estimate_usd"] != s["estimate_usd"] for s in short.values()):  # NaN: no cost per cycle, whatever the caps
        if "vlm_cost" in plan:
            return None                                                     # this arm's refusal already says so
        if started:
            return "E3 has started with no e3.vlm_usd_per_cycle in its frozen config, so no vlm arm can run"
        if final:
            return "this run's 1st read freezes a config with no e3.vlm_usd_per_cycle, so no vlm arm will ever run"
        return ("config has no e3.vlm_usd_per_cycle, so no vlm arm's cost is known. The config freezes at E3's 1st read: "
                "measure the cost on dev seeds and add it before the 1st arm, or no vlm arm can run")
    if started:
        return ("E3 has started, so its config is frozen, and these vlm arms cannot run with their caps: "
                + ", ".join(f"{v} ({s['session']})" for v, s in short.items()))
    if final:
        return ("this run's 1st read freezes the config with these vlm caps short, so these vlm arms will never run: "
                + ", ".join(f"{v} (spend_caps_usd.{s['session']} is "
                            + ("missing" if s["cap_usd"] is None else f"${float(s['cap_usd']):.2f}")
                            + f", its arm needs at least ${math.ceil(MARGIN * s['estimate_usd'])})"
                            for v, s in short.items()))
    parts = [f"{v} needs spend_caps_usd.{s['session']} of at least ${math.ceil(MARGIN * s['estimate_usd'])}"
             for v, s in short.items()]
    return ("the config freezes at E3's 1st read, so set every variant's vlm cap now: " + ", ".join(parts)
            + ". Leave a variant's vlm arm out only if it will never run")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="avail shift", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_prep = sub.add_parser("prepare", help="build the witnesses the cycles use")
    p_prep.add_argument("--task", default="cable_loop")
    p_prep.add_argument("--variant", action="append", required=True, help="repeat for several")
    p_run = sub.add_parser("run", help="run cycles of 1 arm")
    p_run.add_argument("--task", default="cable_loop")
    p_run.add_argument("--variant", required=True)
    p_run.add_argument("--arm", required=True, choices=["flag", "vlm", "avail", "avail_human"])
    p_run.add_argument("--seeds", type=_seeds, required=True, help="low:high, high excluded")
    p_run.add_argument("--shift", required=True, help="the shift's name: its folder under AVAIL_DATA/runs/shifts")
    p_run.add_argument("--workers", type=int, default=2)
    p_run.add_argument("--final-run", default=None)
    p_tools = sub.add_parser("tools", help="cycles run by the scripted agent through the tool server alone")
    p_tools.add_argument("--task", default="cable_loop")
    p_tools.add_argument("--variant", required=True)
    p_tools.add_argument("--seeds", type=_seeds, required=True)
    p_tools.add_argument("--shift", required=True)
    p_rep = sub.add_parser("report", help="the shift card from the ledgers")
    p_rep.add_argument("--shift", required=True)
    p_exp = sub.add_parser("export", help="intervention episodes and labels")
    p_exp.add_argument("--shift", required=True)
    p_e3 = sub.add_parser("e3", help="the E3 final run")
    p_e3.add_argument("--variant", required=True, help="the cell's variant, for the plug station the connector type")
    p_e3.add_argument("--task", default="cable_loop")
    p_e3.add_argument("--plan", action="store_true", help="print the plan and stop")
    p_e3.add_argument("--final-run", default=None)
    p_e3.add_argument("--workers", type=int, default=4)
    p_e3.add_argument("--arm", action="append", default=None, help="run only these arms (repeat)")
    args = parser.parse_args(argv)

    if args.command == "prepare":
        import importlib

        from ..config import load_experiment
        from ..core.task import load_task

        task = load_task(args.task)
        bed = importlib.import_module(task.testbed)
        alpha = float(load_experiment().customer_bar["alpha_for_bar"])
        for variant in args.variant:
            if hasattr(bed, "prepare"):                       # the testbed's own list of what a final run will read
                made = bed.prepare(task, variant, alpha)
            else:
                made = {p: type(w).__name__ for p, w in bed.make_witnesses(task, variant, alpha).items()}
            print(f"{args.task} {variant}: " + ", ".join(f"{k} {v}" for k, v in made.items()), flush=True)
        return 0
    if args.command == "run":
        if args.final_run == FINAL_RUN:
            parser.error("the E3 split is read by `avail shift e3 --variant <variant> --final-run E3` alone, which checks "
                         "the caps, the witnesses and the setup record before any seed is read")
        from .runner import run_shift

        summary = run_shift(args.task, args.variant, args.arm, args.seeds, shift=args.shift, run_root=_shift_dir(args.shift),
                            workers=args.workers, final_run=args.final_run)
        print(json.dumps(summary, indent=1), flush=True)
        return 0
    if args.command == "tools":
        import importlib

        from ..core.task import load_task

        bed = importlib.import_module(load_task(args.task).testbed)
        out = bed.tool_session(args.task, args.variant, args.seeds, shift=args.shift, run_root=_shift_dir(args.shift))
        print(json.dumps(out, indent=1, default=str), flush=True)
        return 0
    if args.command == "report":
        from . import card

        path = card.write(_shift_dir(args.shift))
        print(card.md(json.loads(path.read_text(encoding="utf-8"))), flush=True)
        print(f"\nwritten to {path}", flush=True)
        return 0
    if args.command == "export":
        from ..flywheel.export import export_interventions

        out = export_interventions(_shift_dir(args.shift), _shift_dir(args.shift) / "export")
        print(json.dumps(out, indent=1, default=str), flush=True)
        return 0
    if args.command == "e3":
        from ..config import load_experiment

        variants = [str(v) for v in load_experiment().e3.get("variants", [])]
        if args.variant not in variants:
            parser.error(f"E3 runs the variants in config e3.variants ({', '.join(variants)}), and {args.variant} is not "
                         "one of them")
        from ..core.task import load_task

        if not load_task(args.task).testbed:
            parser.error(f"{args.task} names no testbed, so no shift can run it. E3 runs a task whose file names its "
                         "testbed")
        plan = e3_plan(args.variant)
        if args.arm:
            unknown = [a for a in args.arm if a not in plan["arms"]]
            if unknown:
                parser.error(f"E3 has no arm {', '.join(unknown)}. Its arms: {', '.join(plan['arms'])}")
            plan["arms"] = {a: s for a, s in plan["arms"].items() if a in args.arm}
            if "vlm" not in plan["arms"]:
                plan.pop("vlm_cost", None)
        plan["setup_files"] = setup_hashes(args.task)
        print(json.dumps(plan, indent=1), flush=True)
        started = e3_started()
        # the cap, witness and setup messages judge the config as it is now (its caps, alpha, variants, arms and judge),
        # which after E3's 1st read must be the config it read under: when it is not, the seed guard refuses every arm,
        # and that is the only thing worth saying
        config_changed = e3_config_problem() if started else None
        problem = None if config_changed else vlm_cost_problem(plan, started)
        warning = None if config_changed else other_caps_warning(plan, started, final=not args.plan)
        # every variant's witness and reference, since the record names them all: 1 made after the 1st arm changes it
        station = station_problem(args.task)               # files no command here makes: its own advice, none added
        judge = judge_problem(args.task) if "vlm" in plan["arms"] else None    # a key missing would show 12 h in
        missing = None if config_changed else prepare_problem(args.task, variants, plan.get("alpha"))
        if missing and started:
            missing += (". E3 has started, so a new witness or reference picture would change the setup its record holds: "
                        "put back what changed")
        elif missing:
            missing += (". Run `avail shift prepare --variant <variant>` for each before E3's 1st arm: a final run builds "
                        "none of these")
        record = _shift_dir("e3_setup.json")              # shared by every variant: they read the same seeds
        changed = None if config_changed else setup_problem(record, plan["setup_files"])
        # a read with no record cannot be checked: the 1st arms' setup is unknown, and a new record would not be it
        unrecorded = (None if config_changed or not started or record.is_file() else
                      f"E3 has read its split and its setup record {record} is missing, so what its arms ran under cannot "
                      "be checked. Put the record back, or E3 needs new, unread test seeds")
        if args.plan:
            for line in (config_changed, unrecorded, station, judge, problem, missing, warning, changed):
                if line:
                    print(f"\nbefore the run: {line}", flush=True)
            return 0
        if args.final_run != FINAL_RUN:
            parser.error("the E3 final run reads the shift test seeds once. Pass --final-run E3 to start it")
        if config_changed:
            parser.error(config_changed)
        if unrecorded:
            parser.error(unrecorded)
        if station:
            parser.error(station)
        if judge:
            parser.error(judge)
        if problem:
            parser.error(problem)
        if missing:
            parser.error(missing)
        if changed:
            parser.error(changed)
        if warning:
            print(f"\nwarning: {warning}", flush=True)
        if not record.is_file():
            record.parent.mkdir(parents=True, exist_ok=True)
            record.write_text(json.dumps({"task": args.task, "files": plan["setup_files"]}, indent=1),
                              encoding="utf-8", newline="\n")
        from .runner import run_shift

        expected = json.loads(record.read_text(encoding="utf-8"))["files"]   # every cycle is checked against it
        # the vlm arm runs last: a judge call refused at its cap stops the command, and the other arms come 1st
        for arm, spec in sorted(plan["arms"].items(), key=lambda kv: kv[0] == "vlm"):
            changed = setup_problem(record, setup_hashes(args.task))    # a file edited since the last cycle's check
            if changed:
                parser.error(changed)
            lo, hi = spec["seeds"]
            run_shift(args.task, args.variant, arm, list(range(lo, hi + 1)), shift=f"e3_final_{args.variant}",
                      run_root=_shift_dir(f"e3_final_{args.variant}"), workers=args.workers, final_run=FINAL_RUN,
                      expected_setup=expected)
        return 0
    return 2
