"""The cable loop's own insertions, as a fit and a calibration table for E3's insertion witness (S5b). Evaluator-side.

    uv run python -m avail.station.mj_plug.cable_zoo --variant latch --split cable_fit --workers 4

E3's witness was fit and calibrated on the zoo's 1-plug states. On the cable loop it met a state the zoo has no class
for (dev2, seed 20,132: a retried insertion 0.3 mm short of the latch), so this module builds tables from the cable
loop itself, on the seed ranges ``cable_fit`` and ``cable_calibration`` (config/experiment.yaml), which are no test
range.

Each seed runs 1 cycle of the cell with its own failure draws. Pick and carry are retried until truth says they
holds, up to 3 tries, which stands for the checks and the person that stand before an insertion on E3's AVAIL arms.
Each insertion claim then gets the cascade's sources in the cascade's order, as ``Cascade.verify`` collects them on a
live cell: the passive signals (pass ``A``), then the push and then the tug (pass ``B``), each on what the last one
left, with truth read again after each. An insertion that was bad at its claim is tried once more, as a rejected
1st try is on E3, so retried insertions are in the table too. Truth is used only for the labels and for which steps
to retry: no source ever sees it. Columns follow the zoo's (``avail.zoo.features``): ``<pass>/<source>/<feature>``,
``_y`` the truth after the source acted, ``y0`` the truth at the claim.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

PASSIVE = ("passive.skill_flag", "passive.pose_residual", "passive.jaw_state", "passive.click")
PROBES = ("probe.push_seat", "probe.tug")
SPLITS = ("cable_fit", "cable_calibration")
BLOCK = 10
TRIES = 3


def _step_until(cell, step: str, item: int, predicate: str | None) -> bool:
    """A step tried until truth says its predicate holds, up to TRIES times. Evaluator-side."""
    for attempt in range(TRIES):
        cell.run_skill(step, item, attempt)
        if predicate is None or cell.truth_of(item, predicate):
            return True
    return False


def _collect(cell, claim, params: dict, source_id: str, row: dict, prefix: str, item: int, predicate: str) -> None:
    from ...evidence import base

    key = f"{prefix}/{source_id}"
    try:
        ev = base.get(source_id).collect(cell.view, claim, dict(params))
    except Exception as exc:                                # a broken source must not sink the block
        row[f"{key}/_error"] = f"{type(exc).__name__}: {exc}"[:200]
        row[f"{key}/_y"] = bool(cell.truth_of(item, predicate))
        return
    for name, value in ev.features.items():
        row[f"{key}/{name}"] = float(value)
    row[f"{key}/_score"] = ev.score
    row[f"{key}/_seconds"] = ev.cost.seconds
    row[f"{key}/_y"] = bool(cell.truth_of(item, predicate))


def insertion_rows(seed: int, variant: str) -> list[dict]:
    """Every insertion claim of 1 cycle on this seed, with the cascade's features and the truth labels."""
    from ...analysis.run import primary_predicate
    from ...core.task import load_task
    from ...schemas import Claim, ClaimContext, Predicate, new_id
    from .cable_cell import CableCell

    task = load_task("cable_loop")
    predicate = primary_predicate(variant)
    definition = task.predicates[predicate].done_means
    cell = CableCell(seed, variant)
    rows: list[dict] = []
    try:
        for item in range(cell.n_items):
            if not _step_until(cell, "pick", item, "held") or not _step_until(cell, "carry", item, "at_port"):
                continue
            for attempt in (0, 1):
                w0 = time.perf_counter()
                res = cell.run_skill("insert", item, attempt)
                params = cell.claim_params(item, "insert", res)
                claim = Claim(claim_id=new_id("claim"), run_id=f"{variant}_cable_zoo", cycle_id=f"c{seed}",
                              step_id=f"{item}.insert",
                              predicate=Predicate(name=predicate, target={"item": int(item), **cell.target_of(item)},
                                                  definition=definition),
                              raised_by="skill", context=ClaimContext(task="cable_loop", stage="insert", attempt=attempt))
                y0 = bool(cell.truth_of(item, predicate))
                row = {"seed": int(seed), "connector": variant, "item": int(item), "attempt": int(attempt),
                       "failure": res.failure or "", "claim_predicate": predicate, "y0": y0,
                       "skill_flag": bool(res.flag)}
                for source_id in PASSIVE:                   # pass A: nothing has acted
                    _collect(cell, claim, params, source_id, row, "A", item, predicate)
                for source_id in PROBES:                    # pass B: the push, then the tug on what it left
                    _collect(cell, claim, params, source_id, row, "B", item, predicate)
                row["wall_s"] = round(time.perf_counter() - w0, 3)
                rows.append(row)
                if y0:
                    break
            _step_until(cell, "release", item, None)
    finally:
        cell.close()
    return rows


def _part_path(variant: str, split: str, block: list[int]):
    from ...zoo import store

    folder = store.split_dir(variant, split) / "features"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{store.part_name(block, BLOCK)}.parquet"


def merge(variant: str, split: str):
    """The parts concatenated into ``features.parquet``. A seed has 1 row per insertion claim, so the key is the seed,
    the cable and the attempt (the zoo's own merge allows 1 row per seed)."""
    import pandas as pd

    from ...zoo import store

    folder = store.split_dir(variant, split)
    parts = sorted((folder / "features").glob("part_*.parquet"))
    table = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True) if parts else pd.DataFrame()
    if len(table) and table.duplicated(["seed", "item", "attempt"]).any():
        raise RuntimeError(f"duplicate claims in the parts under {folder}")
    if len(table):
        table = table.sort_values(["seed", "item", "attempt"]).reset_index(drop=True)
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
    for seed in block:
        assert_simulable(seed)
        for row in insertion_rows(seed, variant):
            row["split"] = split
            rows.append(row)
    store.write_table_atomic(rows, _part_path(variant, split, block))
    return {"block": store.part_name(block, BLOCK), "rows": len(rows), "wall_s": round(time.perf_counter() - t0, 1)}


def run(variant: str, split: str, seeds: list[int] | None = None, workers: int = 4) -> dict:
    """All blocks of a split that have no part yet, then the parts merged into ``features.parquet``."""
    from ...seeds import guard
    from ...zoo import store

    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    seeds = list(seeds) if seeds is not None else list(store.split_seeds(split))
    store.check_seeds_in_split(seeds, split)
    guard(seeds, purpose=f"cable-loop insertions, {variant} {split}")       # no test seed, nothing recorded
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
    summary = {"variant": variant, "split": split, "seeds": len(seeds), "claims": int(len(table)),
               "bad_at_claim": int((~table["y0"].astype(bool)).sum()), "retried": int((table["attempt"] > 0).sum()),
               "source_errors": errors, "wall_s": round(time.perf_counter() - t0, 1)}
    store.write_manifest(variant, split, {"features": summary})
    return summary


def main(argv: list[str] | None = None) -> int:
    import json

    parser = argparse.ArgumentParser(prog="python -m avail.station.mj_plug.cable_zoo")
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
