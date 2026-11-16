"""The flywheel (the night of 2026-11-16, step 5): 10 deployment days on dev seeds, recalibrated after each one.

Sim, with the simulated operator. The labels are recalibration only: no model and no scorer is retrained. Each witness
keeps its scorers. Its thresholds are set again on its calibration table plus every operator answer so far.

- **A day** is ``CYCLES`` closed-loop ``avail_simop`` cycles on its own dev seeds, run with the witnesses of the day
  before, read from the day's own witness folder (``AVAIL_WITNESS_ROOT``, which ``cable_shift`` reads when it is set).
- **After a day**, every intervention with a label (a confirm's yes or no, a takeover's look at the claim when the
  operator took the arm) becomes a calibration row of its predicate's witness: the features of the claim at the fork,
  under the table's column names, with the operator's label in every stage's label column. The witness's thresholds are
  set again at its own alpha, beta and crossing rule, and published as the next day's witness.
- **The held-out set** never enters calibration: the claims of the gate's 500 dev cycles (22,800 to 23,299), with their
  features from the gate's ledger and their truth from its evaluator-side labels. Each day's witnesses decide them
  again offline: the share of good claims sent to the operator and the bad claims accepted, per witness. A claim whose
  new decision would need a stage the gate's run never reached is counted apart.

Usage::

    uv run python -m avail.station.mj_plug.flywheel --start 5000 --days 10 --cycles 200 --workers 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

PREDICATES = ("held", "at_port", "released_stable", "latched")
ROOT_ENV = "AVAIL_WITNESS_ROOT"


def base() -> Path:
    from ...paths import data_root

    return data_root(create=True) / "runs" / "flywheel"


def day_root(day: int) -> Path:
    return base() / "witness" / f"day{day:02d}"


def _witnesses(root: Path) -> dict:
    """The day's witnesses, read from its folder."""
    from ...core.task import load_task
    from . import cable_shift

    old = os.environ.get(ROOT_ENV)
    os.environ[ROOT_ENV] = str(root)
    try:
        cable_shift.check_cascade.cache_clear()
        cable_shift.e3_cascade.cache_clear()
        return cable_shift.make_witnesses(load_task("cable_loop"), "latch", 0.01)
    finally:
        if old is None:
            os.environ.pop(ROOT_ENV, None)
        else:
            os.environ[ROOT_ENV] = old
        cable_shift.check_cascade.cache_clear()
        cable_shift.e3_cascade.cache_clear()


def seed_day0() -> Path:
    """Day 0's folder: a copy of the witnesses the gate ran, so day 1 runs exactly them."""
    from ...paths import data_root

    root = day_root(0)
    if not root.is_dir():
        shutil.copytree(data_root() / "witness_e3", root)
    return root


def _calibration(predicate: str):
    from ...analysis import e1
    from ...analysis.run import primary_predicate
    from . import cable_shift as cs

    if predicate == primary_predicate("latch"):
        return e1.main_slice(cs.cable_table("latch", cs.CALIBRATION_SPLIT), predicate)
    return cs.check_table("latch", cs.CHECK_CALIBRATION_SPLIT, predicate)


def labeled_rows(shift_dir: Path, cascade) -> list[dict]:
    """The day's labeled interventions on 1 witness's predicate, as calibration rows under its table's columns."""
    from ...ledger.reader import iter_events

    rows = []
    for ledger in sorted(Path(shift_dir).glob("*/ledger.jsonl")):
        for e in iter_events(ledger, kinds=["intervention"]):
            iv = e["payload"]
            if iv.get("predicate_name") != cascade.predicate or iv.get("truth_label") is None:
                continue
            feats = dict(iv.get("features") or {})
            row: dict = {"seed": -1, "_label_from": iv.get("question")}
            for sdef in cascade.stage_defs:
                row[sdef.label] = bool(iv["truth_label"])
                for key in sdef.sources:
                    source_id = key.partition("@")[0]
                    for name, value in feats.items():
                        if name.startswith(source_id + "/"):
                            row[f"{sdef.zoo_pass}/{key}/{name[len(source_id) + 1:]}"] = float(value)
            rows.append(row)
    return rows


def recalibrate(cascade, table, rows: list[dict]):
    """The witness's thresholds set again on its calibration table plus the new rows. Scorers stay as they are."""
    import pandas as pd

    from ...witness.cascade import calibrate_cascade

    cal = pd.concat([table, pd.DataFrame(rows)], ignore_index=True) if rows else table
    for sdef in cascade.stage_defs:
        if sdef.label in cal.columns:
            cal[sdef.label] = cal[sdef.label].astype("boolean")
    neg = list(cascade.calibration.get("neg_share_fit") or [1.0] * len(cascade.stages))
    pos = list(cascade.calibration.get("pos_share_fit") or [1.0] * len(cascade.stages))
    # the split name stays the witness's own, so the testbed reads the day's witness as built for its stages and split
    split = str(cascade.calibration.get("split_id", "unknown"))
    new = calibrate_cascade([s.scorer for s in cascade.stages], neg, pos, cal, cascade.connector, cascade.predicate,
                            float(cascade.alpha), float(cascade.beta), cascade.stage_defs, station_id=cascade.station_id,
                            skip_features=cascade.skip_features, version=cascade.version, split_id=split,
                            crossing=getattr(cascade, "crossing", "abstain"))
    new.calibration.update({k: v for k, v in cascade.calibration.items() if k not in new.calibration})
    new.calibration["flywheel_rows"] = len(rows)
    return new


def publish_day(day: int, cascades: dict) -> Path:
    """The day's witnesses, written as the latest version under the day's folder (store layout)."""
    from ...witness import store

    root = day_root(day)
    root.mkdir(parents=True, exist_ok=True)
    for c in cascades.values():
        store.publish(c, checks={}, provenance={"purpose": f"flywheel day {day}: recalibrated on calibration plus "
                                                           f"{c.calibration.get('flywheel_rows', 0)} operator labels"},
                      root=root)
    return root


# ----------------------------------------------------------------------------- the held-out replay
def held_out_claims(shift_dir: Path) -> list[dict]:
    """Every verified claim of a shift: its predicate, its features by source and its evaluator-side truth."""
    from ...ledger.reader import iter_events

    out = []
    for ledger in sorted(Path(shift_dir).glob("*/ledger.jsonl")):
        ev, verdicts, claims_of_cycle = defaultdict(dict), [], {}
        ends = {}
        for e in iter_events(ledger, kinds=["claim", "evidence", "verdict", "cycle_end"]):
            p, k = e["payload"], e["kind"]
            if k == "claim":
                claims_of_cycle[p["claim_id"]] = p
            elif k == "evidence":
                for name, value in (p.get("features") or {}).items():
                    ev[p["claim_id"]][f"{p['source_id']}/{name}"] = value
            elif k == "verdict":
                verdicts.append(p)
            else:
                ends[p["cycle_id"]] = p
        by_cycle = defaultdict(list)
        for v in verdicts:
            c = claims_of_cycle.get(v["claim_id"])
            if c is not None:
                by_cycle[c["cycle_id"]].append((v, c))
        for cid, pairs in by_cycle.items():
            audit = ends.get(cid, {}).get("claims", [])
            if len(audit) != len(pairs):
                continue                                   # a cycle cut off, or one without labels: left out
            for (v, c), a in zip(pairs, audit):
                out.append({"predicate": c["predicate"]["name"], "features": ev[v["claim_id"]], "true": bool(a["true"]),
                            "decision": v["decision"], "stage": int(v.get("stage_reached") or 0), "cycle": cid})
    return out


def replay(cascade, claim: dict) -> str:
    """The witness's decision on a recorded claim. ``missing`` when it would need a stage the run never reached."""
    from ...stats import decide

    feats: dict[str, float] = {}
    for k, stage in enumerate(cascade.stages):
        sdef = cascade.stage_defs[k]
        have = False
        for key in sdef.sources:
            source_id = key.partition("@")[0]
            for name, value in claim["features"].items():
                if name.startswith(source_id + "/"):
                    feats[f"{key}/{name[len(source_id) + 1:]}"] = float(value)
                    have = True
        if not have:
            return "missing"
        d = decide(stage.scorer.score_one(feats), stage.tau_reject, stage.tau_accept, getattr(cascade, "crossing", "abstain"))
        if d != "abstain":
            return d
    return "abstain"


def evaluate(cascades: dict, claims: list[dict], cycles: int) -> dict:
    from ...stats import wilson

    out = {}
    for pred, c in cascades.items():
        mine = [x for x in claims if x["predicate"] == pred]
        dec = [replay(c, x) for x in mine]
        good = [d for d, x in zip(dec, mine) if x["true"]]
        bad = [d for d, x in zip(dec, mine) if not x["true"]]
        k_op = sum(d == "abstain" for d in good)
        k_fa = sum(d == "accept" for d in bad)
        out[pred] = {"good": len(good), "good_to_operator": k_op, "good_to_operator_ci": wilson(k_op, len(good)),
                     "bad": len(bad), "bad_accepted": k_fa, "bad_accepted_ci": wilson(k_fa, len(bad)) if bad else None,
                     "missing": sum(d == "missing" for d in dec),
                     "to_operator_per_100_cycles": 100.0 * sum(d == "abstain" for d in dec) / cycles}
    return out


def run(start: int, days: int, cycles: int, workers: int, held_out: Path) -> dict:
    from ...analysis.e4 import gate as gate_numbers
    from ...seeds import guard
    from ...shift.runner import run_shift

    seed_day0()
    claims = held_out_claims(held_out)
    n_held = len({x["cycle"] for x in claims})
    summary_path = base() / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {"days": []}
    done = {d["day"] for d in summary["days"]}
    w0 = _witnesses(day_root(0))
    if not summary.get("day0"):
        summary["day0"] = {"held_out": evaluate({p: w0[p] for p in PREDICATES}, claims, n_held), "held_out_cycles": n_held}
    tables = {p: _calibration(p) for p in PREDICATES}
    rows: dict[str, list] = defaultdict(list)
    for d in sorted(done):                                 # a resumed run takes back the labels of the days it has
        for p in PREDICATES:
            rows[p] += labeled_rows(base() / "days" / f"day{d:02d}", w0[p])
    for day in range(1, days + 1):
        if day in done:
            continue
        seeds = list(range(start + (day - 1) * cycles, start + day * cycles))
        guard(seeds, purpose=f"flywheel day {day}")
        os.environ[ROOT_ENV] = str(day_root(day - 1))
        try:
            run_shift("cable_loop", "latch", "avail_simop", seeds, shift=f"flywheel_day{day:02d}",
                      run_root=base() / "days" / f"day{day:02d}", workers=workers)
        finally:
            os.environ.pop(ROOT_ENV, None)
        wit = _witnesses(day_root(day - 1))
        new = {}
        for p in PREDICATES:
            rows[p] += labeled_rows(base() / "days" / f"day{day:02d}", wit[p])
            new[p] = recalibrate(wit[p], tables[p], rows[p])
        publish_day(day, new)
        g = gate_numbers(base() / "days" / f"day{day:02d}")
        calls = g["calls_per_100"]
        summary["days"].append({"day": day, "seeds": [seeds[0], seeds[-1]], "cycles": g["cycles"],
                                "calls_per_100": calls, "calls_ub95": g["calls_per_100_ub95_bootstrap"],
                                "bad_cycles": g["bad_cycles"], "bad_cycles_ci": [g["bad_cycles_rate"]["lo"],
                                                                                 g["bad_cycles_rate"]["hi"]],
                                "labels_so_far": {p: len(rows[p]) for p in PREDICATES},
                                "held_out": evaluate(new, claims, n_held)})
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8", newline="\n")
        print(f"day {day}: {g['cycles']} cycles, {calls:.1f} calls per 100, {g['bad_cycles']} bad cycles, "
              f"labels {dict((p, len(rows[p])) for p in PREDICATES)}", flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    from ...paths import runs_dir

    ap = argparse.ArgumentParser(prog="python -m avail.station.mj_plug.flywheel")
    ap.add_argument("--start", type=int, default=5000)
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--cycles", type=int, default=200)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--held-out", type=Path, default=None)
    args = ap.parse_args(argv)
    held = args.held_out or (runs_dir() / "shifts" / "gate_e4")
    run(args.start, args.days, args.cycles, args.workers, held)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
