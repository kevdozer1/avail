"""Run a shift: N cycles of a task, 1 arm, back to back, resumable, in parallel workers.

Core code. The testbed is named by the task file (``testbed:``, a module path) and imported by
name, so the core imports no simulator. The testbed module provides:

- ``make_cell(seed, variant, profile) -> cell`` (the cycle driver's cell interface, see ``driver.py``)
- ``make_witnesses(task, variant, alpha) -> {predicate: witness}``
- ``make_human(cell, seed) -> human`` (``avail.fork.human.Human``)
- ``make_judge() -> evidence source`` (the vlm arm's judge), may raise when no key is set
- ``make_fig_judge(session) -> evidence source`` (the agent_looks arm's judge), only for that arm
- ``evaluate(cell) -> {item: bool}``: the evaluator's truth on each item at the end of the cycle
- ``frames_table(cell) -> DataFrame`` and ``scene_events(cell) -> list``: the cycle's frame log
- ``close(cell)``

Cycles run in blocks. Each block is 1 ledger run, ``<shift>_<arm>_<variant>_b<first seed>``, so
workers never share a ledger, and a block whose ledger already holds a ``cycle_end`` for a seed
skips that seed: a stopped shift starts again where it stopped. A cycle that was cut off runs again
under a new id (``_r1``), so its records stay apart from the rerun's, and only ended cycles count. The
seed guard runs before any cycle: a shift test seed needs its final run's name, and the read is recorded.
A final run also carries the setup it recorded (``setup_record.py``): each cycle is checked against it
before it starts and before its end is written, so a cycle that ran under another setup never counts.
"""

from __future__ import annotations

import importlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

BLOCK = 10


def _block_runs(shift: str, arm: str, variant: str, seeds: list[int], block: int = BLOCK) -> list[tuple[str, list[int]]]:
    out = []
    for i in range(0, len(seeds), block):
        part = seeds[i:i + block]
        out.append((f"{shift}_{arm}_{variant}_b{part[0]}", part))
    return out


def done_seeds(run_dir: Path) -> set[int]:
    """Seeds whose cycle already ended in a run's ledger."""
    from ..ledger.reader import iter_events

    path = Path(run_dir) / "ledger.jsonl"
    if not path.is_file():
        return set()
    return {int(e["payload"]["seed"]) for e in iter_events(path, kinds=["cycle_end"]) if "seed" in e["payload"]}


def started_cycles(run_dir: Path) -> set[str]:
    """Cycle ids that a run's ledger saw start, ended or not."""
    from ..ledger.reader import iter_events

    path = Path(run_dir) / "ledger.jsonl"
    if not path.is_file():
        return set()
    return {str(e["payload"].get("cycle_id")) for e in iter_events(path, kinds=["cycle_start"])}


def cycle_id_for(run_id: str, seed: int, started: set[str]) -> str:
    """``<run>_c<seed>``, or ``<run>_c<seed>_r<k>`` when a stopped run already started that cycle k times, so a
    rerun's records never mix with those of the cycle that was cut off."""
    base = f"{run_id}_c{seed}"
    k = sum(1 for c in started if c == base or c.startswith(base + "_r"))
    return base if k == 0 else f"{base}_r{k}"


def same_setup(expected: dict | None, task_name: str, run_id: str, when: str, ran: bool = False) -> None:
    """Raise when a final run's setup differs from the 1 it recorded, ``expected``. None: no check (a dev shift).
    ``ran``: the cycle has run, so it is cut off and reruns under a new id."""
    if expected is None:
        return
    from .setup_record import changed_keys, setup_hashes

    changed = changed_keys(expected, setup_hashes(task_name))
    if changed:
        then = ("so that cycle does not count. Put them back as they were and run the same command again, which reruns "
                "it under a new id" if ran else
                "so the run stopped before that seed. Put them back as they were and run the same command again, which "
                "goes on from that seed")
        raise RuntimeError(f"{', '.join(changed)} changed {when} ({run_id}), {then}, or the final run needs new, unread "
                           "test seeds")


def run_block(args: tuple) -> dict:
    """Worker: the cycles of 1 block, written to their own ledger run."""
    (task_name, arm, variant, profile, run_id, seeds, run_root, final_run, alpha, expected) = args
    from ..core.task import load_task
    from ..ledger.writer import LedgerWriter
    from ..seeds import assert_simulable, unlock_final_run
    from .driver import CycleDriver

    unlock_final_run(final_run)
    task = load_task(task_name)
    bed = importlib.import_module(task.testbed)
    run_dir = Path(run_root) / run_id
    skip = done_seeds(run_dir)
    started = started_cycles(run_dir)
    todo = [s for s in seeds if s not in skip]
    out = {"run_id": run_id, "cycles": 0, "skipped": len(seeds) - len(todo), "wall_s": 0.0}
    if not todo:
        return out
    t0 = time.perf_counter()
    judge = None
    if arm == "vlm":                    # a final run's judge calls go to that run's own session, 1 per variant
        judge = bed.make_judge(session=f"{final_run}_vlm_arm_{variant}") if final_run else bed.make_judge()
    elif arm == "agent_looks":          # the figure run's judge, charged to its own session
        judge = bed.make_fig_judge(session=f"{final_run or 'FIG'}_agent_looks")
    with LedgerWriter(run_id, folder=run_dir) as ledger:
        for seed in todo:
            same_setup(expected, task_name, run_id, f"before seed {seed}")
            assert_simulable(seed)
            cell = bed.make_cell(seed, variant, profile)
            try:
                witnesses = bed.make_witnesses(task, variant, alpha) if arm in ("avail", "avail_human") else {}
                human = bed.make_human(cell, seed) if arm == "avail_human" else None
                driver = CycleDriver(task, cell, arm, witnesses=witnesses, human=human, ledger=ledger, judge=judge,
                                     run_id=run_id, cycle_id=cycle_id_for(run_id, seed, started),
                                     evaluator=bed.evaluator(cell))
                log = driver.run()
                truth = bed.evaluate(cell)
                same_setup(expected, task_name, run_id, f"while seed {seed} ran", ran=True)   # before its end is written
                _cycle_end(ledger, driver, log, truth, seed, arm, variant, task)
                frames = bed.frames_table(cell)
                fdir = run_dir / "frames"
                fdir.mkdir(parents=True, exist_ok=True)
                frames.to_parquet(fdir / f"{log.cycle_id}.parquet", index=False)
                (fdir / f"{log.cycle_id}.scene.json").write_text(json.dumps(bed.scene_events(cell)), encoding="utf-8")
                out["cycles"] += 1
            finally:
                bed.close(cell)
    out["wall_s"] = round(time.perf_counter() - t0, 1)
    return out


def _cycle_end(ledger, driver, log, truth: dict, seed: int, arm: str, variant: str, task) -> None:
    """The cycle_end event: what was declared, what was true (sim, from the evaluator), and the time it took."""
    from ..schemas import CycleEnd, CyclePredicate, Predicate

    items = []
    preds = []
    for it in log.items:
        ok_true = bool(truth.get(it.item, False))
        items.append({"item": it.item, "declared": bool(it.declared_done), "true": ok_true,
                      "uncaught": bool(it.declared_done and not ok_true), "unresolved": it.unresolved,
                      "attempts": it.attempts, "failures": it.failures, "interventions": it.interventions,
                      "takeovers": it.takeovers})
        for step, predicate in task.required(variant):
            rec = driver.gate.settled(it.item, predicate) or {}
            preds.append(CyclePredicate(predicate=Predicate(name=predicate, target={"item": it.item},
                                                            definition=task.predicates[predicate].done_means),
                                        verdict_id=rec.get("verdict_id"), signoff_assist_id=rec.get("signoff_assist_id")))
    declared = all(i["declared"] for i in items)
    payload = CycleEnd(declared_success=bool(declared), predicates=preds, wall_s=float(log.shift_s),
                       true_success=bool(all(i["true"] for i in items)),
                       cycle_id=log.cycle_id, seed=int(seed), arm=arm, variant=variant, task=task.task, items=items,
                       uncaught=sum(i["uncaught"] for i in items), declared_items=sum(i["declared"] for i in items),
                       true_items=sum(i["true"] for i in items), assists=log.assists, takeovers=log.takeovers,
                       forks=log.forks, usd=log.usd,
                       time_s={"motion": log.motion_s, "verify": log.verify_s, "model": log.model_s,
                               "wait": log.wait_s, "handling": log.handling_s},
                       steps=list(driver.audit))     # evaluator-side: each step declared done or not, and whether it was
    ledger.append("cycle_end", payload)


def run_shift(task_name: str, variant: str, arm: str, seeds: list[int], *, shift: str, run_root, workers: int = 2,
              profile: str = "easy", final_run: str | None = None, alpha: float | None = None,
              expected_setup: dict | None = None) -> dict:
    """Run every cycle of a shift for 1 arm. Returns how many ran, skipped and how long it took.

    ``expected_setup``: the setup hashes a final run recorded (``setup_record.setup_hashes``). A final run must pass
    them, and every cycle is checked against them."""
    from ..config import load_experiment
    from ..core.task import load_task
    from ..seeds import guard, unlock_final_run

    # refused before the guard records a read: a final run with no recorded setup, or a task no worker could run
    if final_run is not None and expected_setup is None:
        raise ValueError(f"the final run {final_run} passes the setup it recorded (expected_setup), so that each cycle "
                         "can be checked against it. Start it through its own command, such as `avail shift e3`")
    testbed = load_task(task_name).testbed
    if not testbed:
        raise ValueError(f"{task_name} names no testbed, so no shift can run it")
    importlib.import_module(testbed)
    seeds = [int(s) for s in seeds]
    guard(seeds, purpose=f"shift {shift}, {task_name} {variant} {arm}", final_run=final_run)
    unlock_final_run(final_run)
    if alpha is None:
        alpha = float(load_experiment().customer_bar["alpha_for_bar"])
    blocks = _block_runs(shift, arm, variant, seeds)
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    done = []
    args = [(task_name, arm, variant, profile, run_id, part, str(run_root), final_run, alpha, expected_setup)
            for run_id, part in blocks]
    if workers <= 1:
        for a in args:
            done.append(run_block(a))
            print(f"  {done[-1]['run_id']}: {done[-1]['cycles']} cycles, {done[-1]['skipped']} skipped, "
                  f"{done[-1]['wall_s']} s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as pool:
            futures = [pool.submit(run_block, a) for a in args]
            for fut in as_completed(futures):
                info = fut.result()
                done.append(info)
                print(f"  {info['run_id']}: {info['cycles']} cycles, {info['skipped']} skipped, {info['wall_s']} s", flush=True)
    return {"shift": shift, "arm": arm, "variant": variant, "cycles": sum(d["cycles"] for d in done),
            "skipped": sum(d["skipped"] for d in done), "wall_s": round(time.perf_counter() - t0, 1),
            "runs": [d["run_id"] for d in done]}
