"""Run every free evidence source on every zoo state. Evaluator-side.

For each seed the injected state is rebuilt (the station is deterministic, and the rebuild is
checked against the states table), then two passes run from a snapshot of the claim-time state:

- ``A``, standalone. Each source acts on the untouched state, so the evidence table (E1) can
  score every check by itself.
- ``B``, the cascade order: push, then tug, then a closer look. Each stage sees what the
  stages before it left behind, which is what the cascade (E2) is calibrated on.

After every source that moves the robot, ground truth is read again. That label, taken after
the stage's action, is the one the stage is judged against (docs/04_calibration_math.md,
section 4). Sources get a ``StationView`` and never see any of this.

Columns are ``<pass>/<source>/<feature>``. Names that start with an underscore are the
evaluator's own: ``_y`` (truth of the claim after the source acted), ``_score`` (the source's
hand-made score), ``_seconds``, ``_side`` and the real forces.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..config import config_hash, load_experiment
from ..seeds import assert_simulable, guard, unlock_final_run
from . import store

PASSIVE = ("passive.skill_flag", "passive.pose_residual", "passive.jaw_state", "passive.click", "instr.link")
STANDALONE_PROBES = ("probe.push_seat", "probe.tug", "probe.wiggle")
CASCADE_PROBES = ("probe.push_seat", "probe.tug")
VISION = "vision.gap"


def _collect(prepared, source_id: str, params: dict, row: dict, prefix: str) -> None:
    """Run one source and write its features and the evaluator's notes into ``row``."""
    from ..evidence import base

    key = f"{prefix}/{source_id}" if "view" not in params or source_id != VISION else f"{prefix}/{source_id}@{params['view']}"
    station = prepared.station
    station._reset_true_peaks()
    try:
        source = base.get(source_id)
    except KeyError:
        row[f"{key}/_missing"] = 1.0
        return
    try:
        ev = source.collect(prepared.view, prepared.claim, params)
    except Exception as exc:                                 # a broken source must not sink the block
        row[f"{key}/_error"] = f"{type(exc).__name__}: {exc}"[:200]
        row[f"{key}/_y"] = prepared.label()                  # whatever it did, the state after it still has a label
        return
    for name, value in ev.features.items():
        row[f"{key}/{name}"] = float(value)
    row[f"{key}/_score"] = ev.score
    row[f"{key}/_seconds"] = ev.cost.seconds
    row[f"{key}/_side"] = ev.side_effect
    row[f"{key}/_y"] = prepared.label()
    if ev.cost.moved_robot:
        row[f"{key}/_true_peak_push_n"] = station._true_peak_push
        row[f"{key}/_true_peak_pull_n"] = station._true_peak_pull
        row[f"{key}/_true_peak_raw_n"] = station._true_peak_raw
        if math.isfinite(station._true_overrun_mm):          # the most any of its guarded moves ran past its own travel limit
            row[f"{key}/_true_overrun_mm"] = station._true_overrun_mm
        tr = prepared.truth()["metrics"]
        row[f"{key}/_shortfall_after_mm"] = tr["shortfall_mm"]


def feature_row(seed: int, connector: str, expected: dict | None = None, frames=None, on_prepared=None,
                profile: str = "easy") -> dict:
    """Every source's features for one seed. ``expected`` is the seed's row in the states table.

    ``on_prepared(prepared)`` is called once the state exists and before any source touches it.
    The block runner uses it to write the states row when the states pass has not been run.
    """
    from .harness import prepare

    w0 = time.perf_counter()
    prepared = prepare(seed, connector, profile=profile)
    if on_prepared is not None:
        on_prepared(prepared)
    out = prepared.outcome
    row: dict = {
        "seed": int(seed), "connector": connector, "cls": out.spec.cls, "landed_cls": out.landed_class,
        "claim_predicate": out.claim_predicate, "target_port": out.target_port, "y0": prepared.label(),
    }
    if expected is not None:                                # the rebuild must be the state that was generated
        same = (expected["landed_cls"] == out.landed_class
                and abs((expected["insert_travel_mm"] or 0.0) - out.insert.travel_m * 1000.0) < 1e-9)
        row["rebuild_matches"] = bool(same)
    station = prepared.station
    snap = station._snapshot()

    # pass A: every source alone, from the untouched claim-time state
    for source_id in PASSIVE:
        _collect(prepared, source_id, prepared.fresh_params(), row, "A")
    params = prepared.fresh_params()
    params["view"] = "claim"
    _collect(prepared, VISION, params, row, "A")
    for source_id in STANDALONE_PROBES:
        station._restore(snap)
        _collect(prepared, source_id, prepared.fresh_params(), row, "A")

    # pass B: the cascade order, each stage on what the last one left, then a fresh look
    station._restore(snap)
    params = prepared.fresh_params()
    for source_id in CASCADE_PROBES:
        _collect(prepared, source_id, params, row, "B")
    params["view"] = "now"
    _collect(prepared, VISION, params, row, "B")

    row["sim_seconds"] = float(station._mj_data.time)
    row["wall_s"] = round(time.perf_counter() - w0, 3)
    prepared.close()
    return row


def features_block(args: tuple) -> dict:
    """Worker: the features of one block of seeds, written as one part."""
    connector, split, seeds, root, final_run = args
    import pandas as pd

    unlock_final_run(final_run)
    for seed in seeds:
        assert_simulable(seed)
    t0 = time.perf_counter()
    states_part = store.part_path(connector, split, "states", seeds, root)
    expected = {}
    if states_part.is_file():
        table = pd.read_parquet(states_part)
        expected = {int(r["seed"]): r for r in table.to_dict("records")}
    zip_path = store.part_path(connector, split, "frames", seeds, root)
    # When the states pass was not run for this block, this pass writes the states part and the
    # claim-time frames too, from the same injected state. Nothing is simulated twice.
    write_states = not states_part.is_file()
    claim_frames = store.FrameZip(zip_path) if write_states else None
    rows, state_rows = [], []

    def record_state(prepared) -> None:
        from .generate import state_row, add_claim_frames

        srow = state_row(prepared.station, prepared.outcome, split)
        add_claim_frames(srow, prepared.station, prepared.outcome, prepared.params["frames"], claim_frames)
        srow["frames_zip"] = claim_frames.path.name
        state_rows.append(srow)

    profile = store.profile_of_split(split)
    try:
        for seed in seeds:
            row = feature_row(seed, connector, expected.get(int(seed)), None, record_state if write_states else None,
                              profile=profile)
            row["split"] = split
            rows.append(row)
    finally:
        if claim_frames is not None:
            claim_frames.close()
    if write_states:
        store.write_table_atomic(state_rows, states_part)
    store.write_table_atomic(rows, store.part_path(connector, split, "features", seeds, root))
    return {"block": store.part_name(seeds), "n": len(rows), "wall_s": round(time.perf_counter() - t0, 1),
            "wrote_states": write_states}


def run_features(connector: str, split: str, seeds: list[int] | None = None, workers: int = 8,
                 time_budget_s: float | None = None, root=None, final_run: str | None = None) -> dict:
    """Run the sources over a split. Blocks that already have a features part are skipped."""
    exp = load_experiment()
    if not exp.connector(connector).modeled:
        raise ValueError(f"connector {connector!r} is not modeled yet")
    all_seeds = list(seeds) if seeds is not None else list(store.split_seeds(split))
    store.check_seeds_in_split(all_seeds, split)
    guard(all_seeds, purpose=f"zoo features, {connector} {split}", final_run=final_run)
    unlock_final_run(final_run)                             # workers inherit it. None closes the door again
    todo, skipped = [], 0
    for block in store.blocks(all_seeds):
        if store.part_path(connector, split, "features", block, root).is_file():
            skipped += 1
        else:
            todo.append(block)
    t0 = time.perf_counter()
    done = 0
    if todo:
        # a fresh process per block: a worker grows about 3 MB per state, and the machine has little to spare
        with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as pool:
            pending: dict = {}
            queue = list(todo)

            def submit() -> None:
                while queue and len(pending) < workers:
                    block = queue.pop(0)
                    pending[pool.submit(features_block, (connector, split, block, root, final_run))] = block

            submit()
            while pending:
                for fut in as_completed(list(pending)):
                    pending.pop(fut)
                    info = fut.result()
                    done += 1
                    print(f"  {connector} {split} {info['block']}: {info['n']} states in {info['wall_s']} s "
                          f"({done}/{len(todo)})", flush=True)
                    break
                if time_budget_s is not None and time.perf_counter() - t0 > time_budget_s:
                    queue.clear()
                submit()
    left = len(todo) - done
    summary = {"connector": connector, "split": split, "blocks_done": done, "blocks_skipped": skipped,
               "blocks_left": left, "wall_s": round(time.perf_counter() - t0, 1), "config_hash": config_hash()}
    if left == 0:
        store.merge(connector, split, "states", root)
        table = store.merge(connector, split, "features", root)
        summary["rows"] = int(len(table))
        if "rebuild_matches" in table:
            summary["rebuild_matches"] = float(table["rebuild_matches"].mean())
        errors = [c for c in table.columns if c.endswith("/_error")]
        summary["source_errors"] = {c: int(table[c].notna().sum()) for c in errors if table[c].notna().any()}
        summary["median_wall_s_per_state"] = float(table["wall_s"].median())
        store.write_manifest(connector, split, {"features": summary}, root)
    return summary
