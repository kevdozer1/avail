"""The cable loop's hand-set checks, as fit and calibration tables for calibrated witnesses (S5c fix). Evaluator-side.

    uv run python -m avail.station.mj_plug.check_zoo --variant latch --split check_fit --workers 6

E3 failed H3's defect bar (2026-11-13), and every 1 of its 31 uncaught failures passed ``released_stable``,
a rule whose thresholds were set by hand on dev renders. ``held`` and ``at_port`` were hand-set the same
way. This module builds the tables that replace all 3 rules with witnesses calibrated the way the insertion
witness is (``cable_zoo.py``), on the seed ranges ``check_fit`` (28,000 to 28,999) and ``check_calibration``
(29,000 to 29,999), which are no test range.

Each seed runs 1 cycle of the cell with its own failure draws. Each step is tried up to the task's
``fork.max_attempts`` times while truth says it failed, then the simulated operator takes over (its scripted
correction, ``OracleHuman.take_over``). That stands for the gate and the operator of the ``avail_simop`` arm,
so the claims in the table are the ones that arm meets: retried ones, and releases after an insertion that was
fixed by a retry or a takeover. Every claim of ``held`` (pick), ``at_port`` (carry) and ``released_stable``
(release) gets the sources its witness reads, in pass ``A`` (none of them moves the robot), with truth read
after each. Insertions are decided by truth too, and get no row: their witness is ``cable_zoo``'s. Truth is
used only for the labels and for which steps to retry or take over. No source ever sees it.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import SimpleNamespace

SPLITS = ("check_fit", "check_calibration", "check_fit_drag", "check_calibration_drag")
#: the enrichment splits (docs/07, 2026-11-13): the release's own failure, the opening's tip and the lift's drag,
#: given to every 1st release try, and only released_stable's rows kept. They join their table's natural rows
DRAG_SPLITS = {"check_fit_drag": "check_fit", "check_calibration_drag": "check_calibration"}
RELEASE_TIP_DEG = (3.0, 14.0)                # set on dev seeds 21,300 to 21,359 before any enrichment seed ran
BLOCK = 10
#: the sources each hand-set predicate's witness reads, in the order its cascade collects them
SOURCES = {
    "held": ("passive.jaw_hold", "vision.cable_tag"),
    "at_port": ("passive.hover_pose", "passive.hover_offset", "passive.jaw_hold", "vision.cable_tag"),
    "released_stable": ("passive.jaw_hold", "passive.release_record", "passive.release_lift", "vision.gap@now",
                        "probe.release_tug"),
}
#: sources that move the arm go in pass B, after every pass-A source, as the cascade runs them (S5c, 2026-11-13)
PASS_B = ("vision.release_relook", "probe.release_tug")


def _collect(cell, claim, params: dict, key: str, row: dict, item: int, predicate: str) -> None:
    from ...evidence import base

    source_id, _, viewname = key.partition("@")
    call = dict(params)
    if viewname:
        call["view"] = viewname
    col = f"{'B' if key in PASS_B else 'A'}/{key}"
    try:
        ev = base.get(source_id).collect(cell.view, claim, call)
    except Exception as exc:                                # a broken source must not sink the block
        row[f"{col}/_error"] = f"{type(exc).__name__}: {exc}"[:200]
        row[f"{col}/_y"] = bool(cell.truth_of(item, predicate))
        return
    for name, value in ev.features.items():
        row[f"{col}/{name}"] = float(value)
    row[f"{col}/_score"] = ev.score
    row[f"{col}/_seconds"] = ev.cost.seconds
    row[f"{col}/_y"] = bool(cell.truth_of(item, predicate))


def claim_rows(seed: int, variant: str, tip: tuple[float, float] | None = None, only: str | None = None) -> list[dict]:
    """Every claim of held, at_port and released_stable in 1 cycle on this seed, with features and truth labels.
    ``tip``: the enrichment's release tip range in degrees (``RELEASE_TIP_DEG``). ``only``: keep 1 predicate's rows."""
    from ...analysis.run import primary_predicate
    from ...core.task import load_task
    from ...schemas import Claim, ClaimContext, Predicate, new_id
    from .cable_cell import CableCell
    from .oracle_human import OracleHuman

    task = load_task("cable_loop")
    tries = int(task.fork.get("max_attempts", 2))
    cell = CableCell(seed, variant)
    cell.release_tip = tip
    operator = OracleHuman(cell, seed)
    rows: list[dict] = []
    try:
        for item in range(cell.n_items):
            for step in task.steps:
                predicate = step.claim_for(variant)
                for attempt in range(tries):
                    w0 = time.perf_counter()
                    res = cell.run_skill(step.name, item, attempt)
                    if predicate in SOURCES and (only is None or predicate == only):
                        params = cell.claim_params(item, step.name, res)
                        claim = Claim(claim_id=new_id("claim"), run_id=f"{variant}_check_zoo", cycle_id=f"c{seed}",
                                      step_id=f"{item}.{step.name}",
                                      predicate=Predicate(name=predicate, target={"item": int(item), **cell.target_of(item)},
                                                          definition=task.predicates[predicate].done_means),
                                      raised_by="skill",
                                      context=ClaimContext(task="cable_loop", stage=step.name, attempt=attempt))
                        row = {"seed": int(seed), "connector": variant, "item": int(item), "step": step.name,
                               "attempt": int(attempt), "failure": res.failure or "", "claim_predicate": predicate,
                               "y0": bool(cell.truth_of(item, predicate)), "skill_flag": bool(res.flag)}
                        for key in SOURCES[predicate]:
                            _collect(cell, claim, params, key, row, item, predicate)
                        row["wall_s"] = round(time.perf_counter() - w0, 3)
                        rows.append(row)
                    if cell.truth_of(item, predicate):
                        break
                else:                                       # every try failed: the operator takes over
                    request = SimpleNamespace(question="takeover",
                                              context={"item": item, "step": step.name, "predicate": predicate})
                    operator.take_over(cell, request)
            cell.settle_item(item)
    finally:
        cell.close()
    return rows


def _part_path(variant: str, split: str, block: list[int]):
    from ...zoo import store

    folder = store.split_dir(variant, split) / "features"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{store.part_name(block, BLOCK)}.parquet"


def merge(variant: str, split: str):
    """The parts concatenated into ``features.parquet``. The key is the seed, the cable, the step and the attempt."""
    import pandas as pd

    from ...zoo import store

    folder = store.split_dir(variant, split)
    parts = sorted((folder / "features").glob("part_*.parquet"))
    table = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True) if parts else pd.DataFrame()
    key = ["seed", "item", "step", "attempt"]
    if len(table) and table.duplicated(key).any():
        raise RuntimeError(f"duplicate claims in the parts under {folder}")
    if len(table):
        table = table.sort_values(key).reset_index(drop=True)
    tmp = folder / "features.parquet.tmp"
    table.to_parquet(tmp, index=False)
    tmp.replace(folder / "features.parquet")
    return table


def run_block(args: tuple) -> dict:
    """Worker: 1 block of seeds, written whole or not at all."""
    from ...seeds import assert_simulable
    from ...zoo import store

    variant, split, block = args
    t0 = time.perf_counter()
    rows = []
    drag = split in DRAG_SPLITS
    for seed in block:
        assert_simulable(seed)
        for row in claim_rows(seed, variant, tip=RELEASE_TIP_DEG if drag else None,
                              only="released_stable" if drag else None):
            row["split"] = split
            rows.append(row)
    store.write_table_atomic(rows, _part_path(variant, split, block))
    return {"block": store.part_name(block, BLOCK), "rows": len(rows), "wall_s": round(time.perf_counter() - t0, 1)}


def summary(table) -> dict:
    """Per predicate: claims, bad claims at the claim, the base rate, retried claims."""
    out = {}
    for predicate, rows in table.groupby("claim_predicate"):
        bad = int((~rows["y0"].astype(bool)).sum())
        out[predicate] = {"claims": int(len(rows)), "bad": bad, "base_rate": round(bad / max(1, len(rows)), 5),
                          "retried": int((rows["attempt"] > 0).sum()),
                          "bad_by_failure": {str(k): int(v) for k, v in
                                             rows[~rows["y0"].astype(bool)]["failure"].replace("", "none").value_counts().items()}}
    return out


def run(variant: str, split: str, seeds: list[int] | None = None, workers: int = 4) -> dict:
    """All blocks of a split that have no part yet, then the parts merged into ``features.parquet``."""
    from ...seeds import guard
    from ...zoo import store

    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    seeds = list(seeds) if seeds is not None else list(store.split_seeds(split))
    store.check_seeds_in_split(seeds, split)
    guard(seeds, purpose=f"cable-loop check claims, {variant} {split}")       # no test seed, nothing recorded
    todo = [b for b in store.blocks(seeds, BLOCK) if not _part_path(variant, split, b).is_file()]
    t0, done = time.perf_counter(), 0
    if todo:
        with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as pool:
            futures = [pool.submit(run_block, (variant, split, b)) for b in todo]
            for fut in as_completed(futures):
                info = fut.result()
                done += 1
                print(f"  {variant} {split} {info['block']}: {info['rows']} claims in {info['wall_s']} s "
                      f"({done}/{len(todo)})", flush=True)
    table = merge(variant, split)
    errors = {c: int(table[c].notna().sum()) for c in table.columns if c.endswith("/_error") and table[c].notna().any()}
    out = {"variant": variant, "split": split, "seeds": len(seeds), "claims": int(len(table)),
           "by_predicate": summary(table), "source_errors": errors, "wall_s": round(time.perf_counter() - t0, 1)}
    store.write_manifest(variant, split, {"features": out})
    return out


def main(argv: list[str] | None = None) -> int:
    import json

    parser = argparse.ArgumentParser(prog="python -m avail.station.mj_plug.check_zoo")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--seeds", default=None, help="low:high, high excluded. Default: the whole split")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    seeds = None
    if args.seeds:
        lo, hi = args.seeds.split(":")
        seeds = list(range(int(lo), int(hi)))
    print(json.dumps(run(args.variant, args.split, seeds, args.workers), indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
