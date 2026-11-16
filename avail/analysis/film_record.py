"""Record 1 cycle for the showcase film: every event on the sim clock, and the frame log at the film's rate. Evaluator-side.

    uv run python -m avail.analysis.film_record --seed 20201 --out <AVAIL_DATA>/runs/showcase/film

The film's overlays are drawn from this log, so the log comes first (the robot demo video style guide). For 1 seed it
runs the cycle twice, the same failure draws each time: ``flag`` (each step's own flag, "Skill says done") and
``avail_simop`` ("AVAIL", the simulated operator behind the fork). The task and its witnesses are the ones E3 tested:
the task file as it stood in E3's frozen worktree (``archive/e3_code``), whose ``held``, ``at_port`` and
``released_stable`` are its hand-set rules, and the insertion witness v3.

Logged on the sim clock (``t``), each with what the code passed: every skill call (``CableCell.run_skill``) and the
robot calls inside it (``move_to_pose``, ``move_relative``, ``set_gripper``, ``wait``), every check
(``witness.verify``) with its verdict and evidence, every call to the operator (``OracleHuman.ask``), and the driver's
own records (claims, verdicts, forks, assists, interventions). The operator's time to answer and a takeover's handling
are not on the sim clock, so each is logged as a hold at the moment it falls. The frame log runs at 30 Hz, so a film at
30 frames a second has a recorded state for every frame at 1x. Nothing here changes what the cycle does: the wrappers
only note the time and pass every call through.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ARMS = ("flag", "avail_simop")
FILM_HZ = 30.0


def _jsonable(x):
    if hasattr(x, "model_dump"):
        return x.model_dump(mode="json")
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, float) or isinstance(x, int) or isinstance(x, str) or x is None or isinstance(x, bool):
        return x
    try:
        return float(x)
    except Exception:  # noqa: BLE001
        return str(x)


def e3_task():
    """The cable loop's task file as E3 ran it, from its frozen worktree."""
    import yaml

    from ..core.task import TaskSpec
    from ..paths import REPO_ROOT

    path = REPO_ROOT / "archive" / "e3_code" / "tasks" / "cable_loop.yaml"
    return TaskSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def record(seed: int, arm: str, out: Path) -> dict:
    from ..seeds import assert_simulable
    from ..shift.driver import CycleDriver
    from ..station.mj_plug import cable_cell, cable_shift

    assert_simulable(seed)                              # dev seeds only: a test seed is refused here
    cable_cell.FRAME_DT = 1.0 / FILM_HZ                 # the frame log at the film's rate. Logging moves nothing
    task = e3_task()
    cell = cable_shift.make_cell(seed, "latch")
    st = cell.station
    events: list[dict] = []
    depth = {"skill": 0}

    def now() -> float:
        return float(cell.now())

    def note(kind: str, **fields) -> None:
        events.append({"t": now(), "kind": kind, **_jsonable(fields)})

    # robot calls inside a skill: the substeps
    for name in ("move_to_pose", "move_relative", "set_gripper", "wait"):
        fn = getattr(st, name)

        def wrapped(*a, _fn=fn, _name=name, **k):
            t0 = now()
            res = _fn(*a, **k)
            if depth["skill"]:
                args = [x for x in a[1:]] if a else []
                events.append({"t": t0, "t_end": now(), "kind": "substep", "fn": _name,
                               "args": _jsonable(args[:2]), "kwargs": _jsonable({kk: vv for kk, vv in k.items()})})
            return res
        setattr(st, name, wrapped)

    skill = cell.run_skill

    def run_skill(step, item, attempt=0, **k):
        t0 = now()
        depth["skill"] += 1
        try:
            res = skill(step, item, attempt, **k)
        finally:
            depth["skill"] -= 1
        events.append({"t": t0, "t_end": now(), "kind": "skill", "fn": "run_skill", "step": step, "item": int(item),
                       "attempt": int(attempt), "flag": bool(res.flag), "ok": bool(res.ok), "failure": res.failure})
        return res
    cell.run_skill = run_skill

    witnesses = cable_shift.make_witnesses(task, "latch", 0.01) if arm != "flag" else {}
    for predicate, w in witnesses.items():
        verify = w.verify

        def checked(view, claim, params, _verify=verify, _p=predicate):
            t0 = now()
            verdict, evidence = _verify(view, claim, params)
            events.append({"t": t0, "t_end": now(), "kind": "check", "fn": "verify", "predicate": _p,
                           "step_id": claim.step_id, "attempt": claim.context.attempt,
                           "witness": type(w).__name__, "version": getattr(w, "version", ""),
                           "decision": verdict.decision, "stage": verdict.stage_reached,
                           # the time the driver counts for a check that moves nothing: each source's stated cost
                           "cost_s": float(sum(e.cost.seconds for e in evidence if not e.cost.moved_robot)),
                           "evidence": {e.source_id: _jsonable(e.features) for e in evidence}})
            return verdict, evidence
        w.verify = checked

    human = cable_shift.make_human(cell, seed) if arm != "flag" else None
    if human is not None:
        ask = human.ask

        def asked(request, _ask=ask):
            t0 = now()
            ans = _ask(request)
            events.append({"t": t0, "t_end": now(), "kind": "operator", "fn": "ask", "question": request.question,
                           "item": int(request.context["item"]), "step": request.context["step"],
                           "answer": str(ans.answer), "seconds_to_answer": float(ans.seconds_to_answer),
                           "seconds_handling": float(ans.seconds_handling), "truth_label": ans.truth_label})
            return ans
        human.ask = asked

    class Ledger:
        def append(self, kind, payload):
            note(f"ledger.{kind}", payload=payload)

    plan = {"cables": [{"cable": k + 1, "port": int(c.port)} for k, c in enumerate(cell.cables)]}
    note("plan", **plan)
    driver = CycleDriver(task, cell, arm, witnesses=witnesses, human=human, ledger=Ledger(), run_id="film",
                         cycle_id=f"film_{arm}_{seed}", evaluator=cable_shift.evaluator(cell))
    try:
        log = driver.run()
        truth = cable_shift.evaluate(cell)
        note("end", items=[{"item": it.item, "declared": bool(it.declared_done), "true": bool(truth.get(it.item)),
                            "unresolved": it.unresolved} for it in log.items],
             time_s={"motion": log.motion_s, "verify": log.verify_s, "wait": log.wait_s, "handling": log.handling_s},
             assists=log.assists, takeovers=log.takeovers, audit=driver.audit)
        folder = Path(out) / arm
        folder.mkdir(parents=True, exist_ok=True)
        cable_shift.frames_table(cell).to_parquet(folder / "frames.parquet", index=False)
        (folder / "scene.json").write_text(json.dumps(cable_shift.scene_events(cell)), encoding="utf-8")
        (folder / "events.json").write_text(json.dumps({"seed": seed, "arm": arm, "film_hz": FILM_HZ,
                                                       "events": events}, default=str), encoding="utf-8")
    finally:
        cable_shift.close(cell)
    return {"arm": arm, "events": len(events), "declared": sum(it.declared_done for it in log.items),
            "true": sum(bool(v) for v in truth.values()), "assists": log.assists}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m avail.analysis.film_record")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    for arm in ARMS:
        print(json.dumps(record(args.seed, arm, Path(args.out))), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
