"""The ladder's harm run (2026-11-16, docs/07 step 2): every rung forced on the 1st try's claim of its step, on dev seeds.

Each seed runs 1 cycle of the cell the way the check tables do: a step is tried while truth says it failed, up to the
task's tries, then the simulated operator takes over. On the 1st try of every step, before anything else, the step's
rung from the task file's ladder runs whether or not the claim holds, with truth read just before and just after it.
Truth only labels. The harm of a rung is the share of good claims (true before) it turned bad (false after), with a
Wilson 95% interval. A rung whose upper bound is over ``HARM_MAX`` is dropped before the gate (Kevin's brief). The share
of bad claims a rung made good is reported beside it, description only.

Usage::

    uv run python -m avail.station.mj_plug.rung_harm --seeds 22200:22700 --workers 6
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

HARM_MAX = 0.005
BLOCK = 10


def out_dir() -> Path:
    from ...paths import data_root

    return data_root(create=True) / "runs" / "analysis" / "overnight2" / "rung_harm"


def rows_for_seed(seed: int, variant: str = "latch") -> list[dict]:
    from ...core.task import load_task
    from .cable_cell import CableCell
    from .oracle_human import OracleHuman

    task = load_task("cable_loop")
    ladder = dict(task.fork.get("ladder", {}))
    tries = int(task.fork.get("max_attempts", 2))
    cell = CableCell(seed, variant)
    operator = OracleHuman(cell, seed)
    rows: list[dict] = []
    try:
        for item in range(cell.n_items):
            for step in task.steps:
                predicate = step.claim_for(variant)
                for attempt in range(tries):
                    res = cell.run_skill(step.name, item, attempt)
                    if attempt == 0:
                        for rung in ladder.get(predicate, []):
                            before = bool(cell.truth_of(item, predicate))
                            t0, w0 = cell.now(), time.perf_counter()
                            out = cell.rung_begin(rung, item, step.name)
                            cell.rung_end(rung, item, step.name)
                            rows.append({"seed": int(seed), "item": int(item), "step": step.name, "predicate": predicate,
                                         "rung": rung, "failure": res.failure or "", "true_before": before,
                                         "true_after": bool(cell.truth_of(item, predicate)),
                                         "rung_ok": bool(getattr(out, "ok", True)), "sim_s": round(cell.now() - t0, 3),
                                         "wall_s": round(time.perf_counter() - w0, 3)})
                    if cell.truth_of(item, predicate):
                        break
                else:
                    operator.take_over(cell, SimpleNamespace(question="takeover", context={
                        "item": item, "step": step.name, "predicate": predicate}))
            cell.settle_item(item)
    finally:
        cell.close()
    return rows


def run_block(block: list[int]) -> dict:
    import pandas as pd

    from ...seeds import assert_simulable

    t0 = time.perf_counter()
    rows = []
    for seed in block:
        assert_simulable(seed)
        rows += rows_for_seed(seed)
    path = out_dir() / f"part_{block[0]:06d}_{block[-1]:06d}.parquet"
    tmp = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    tmp.replace(path)
    return {"block": path.name, "rows": len(rows), "wall_s": round(time.perf_counter() - t0, 1)}


def summary() -> dict:
    import pandas as pd

    from ...stats import wilson

    parts = sorted(out_dir().glob("part_*.parquet"))
    t = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    out = {"seeds": int(t["seed"].nunique()), "harm_max": HARM_MAX, "rungs": {}}
    for rung, r in t.groupby("rung"):
        good = r[r["true_before"]]
        bad = r[~r["true_before"]]
        harm = int((~good["true_after"]).sum())
        lo, hi = wilson(harm, len(good))
        fixed = int(bad["true_after"].sum())
        flo, fhi = wilson(fixed, len(bad)) if len(bad) else (float("nan"), float("nan"))
        out["rungs"][rung] = {"predicate": str(r["predicate"].iloc[0]), "applied": int(len(r)),
                              "good": int(len(good)), "harmed": harm, "harm": harm / max(1, len(good)),
                              "harm_lo": lo, "harm_hi": hi, "kept": bool(hi <= HARM_MAX),
                              "bad": int(len(bad)), "made_good": fixed, "made_good_lo": flo, "made_good_hi": fhi,
                              "sim_s_median": float(r["sim_s"].median()),
                              "harmed_rows": good[~good["true_after"]][["seed", "item", "failure"]].to_dict("records")}
    return out


def main(argv: list[str] | None = None) -> int:
    from ...seeds import guard

    ap = argparse.ArgumentParser(prog="python -m avail.station.mj_plug.rung_harm")
    ap.add_argument("--seeds", required=True, help="low:high, high excluded. Dev seeds only")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args(argv)
    if not args.summary_only:
        lo, hi = (int(v) for v in args.seeds.split(":"))
        seeds = list(range(lo, hi))
        guard(seeds, purpose="the ladder's harm run")                  # refuses a test seed, records nothing for dev
        out_dir().mkdir(parents=True, exist_ok=True)
        blocks = [seeds[i:i + BLOCK] for i in range(0, len(seeds), BLOCK)]
        todo = [b for b in blocks if not (out_dir() / f"part_{b[0]:06d}_{b[-1]:06d}.parquet").is_file()]
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1) as pool:
            futures = [pool.submit(run_block, b) for b in todo]
            for fut in as_completed(futures):
                info = fut.result()
                done += 1
                print(f"  {info['block']}: {info['rows']} rows in {info['wall_s']} s ({done}/{len(todo)})", flush=True)
    s = summary()
    (out_dir() / "summary.json").write_text(json.dumps(s, indent=1, default=float), encoding="utf-8", newline="\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "harmed_rows"} for k, v in s["rungs"].items()},
                     indent=1, default=float), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
