"""Generate zoo states: one injected insertion per seed, with truth and the claim-time frames.

Evaluator-side. Runs in worker processes, a block of seeds each, and writes one part per
block. A run can be stopped at any time and started again: blocks that already have a part
are skipped. ``time_budget_s`` lets a caller keep each call short.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..config import config_hash, load_experiment
from ..seeds import assert_simulable, guard, unlock_final_run
from . import store


def state_row(station, outcome, split: str) -> dict:
    """The states-table row for one injected state. Reads truth: evaluator only."""
    m, p, spec, cfg = outcome.truth["metrics"], outcome.truth["predicates"], outcome.spec, station._cfg
    label = p[outcome.claim_predicate]
    row = {
        "seed": outcome.seed, "connector": outcome.connector, "split": split,
        "cls": spec.cls, "landed_cls": outcome.landed_class, "claim_predicate": outcome.claim_predicate,
        "target_port": outcome.target_port, "label": bool(label),
        "spec_shortfall_mm": spec.shortfall_mm, "spec_wrong_port": spec.wrong_port,
        "spec_lateral_x_mm": spec.lateral_mm[0] if spec.lateral_mm else None,
        "spec_lateral_y_mm": spec.lateral_mm[1] if spec.lateral_mm else None,
        "spec_tilt_x_deg": spec.tilt_deg[0] if spec.tilt_deg else None,
        "spec_tilt_y_deg": spec.tilt_deg[1] if spec.tilt_deg else None,
        "spec_drop_mm": spec.drop_mm, "spec_release": bool(spec.release), "spec_latch_blocked": bool(spec.latch_blocked),
        "true_nearest_port": m["nearest_port"], "true_in_bore": m["in_bore"], "true_depth_mm": m["depth_mm"],
        "true_shortfall_mm": m["shortfall_mm"], "true_lateral_mm": m["lateral_mm"], "true_tilt_deg": m["tilt_deg"],
        "true_finger_contacts": m["finger_contacts"],
        "true_at_port": p["at_port"], "true_seated": p["seated"],
        "true_latched": p["latched"] if p["latched"] is not None else None,
        "true_held": p["held"], "true_released_stable": p["released_stable"],
        "skill_flag": bool(outcome.flag), "insert_stopped_by": outcome.insert.stopped_by,
        "insert_travel_mm": outcome.insert.travel_m * 1000.0, "expected_travel_mm": outcome.expected_travel_m * 1000.0,
        "insert_force_at_stop_n": outcome.insert.force_at_stop_n, "insert_peak_force_n": outcome.insert.peak_force_n,
        "insert_seconds": outcome.insert.duration_s,
        "true_peak_push_n": station._true_peak_push, "true_peak_raw_n": station._true_peak_raw,
        "scene_board_yaw_deg": cfg.board_yaw * 57.29577951308232, "scene_plug_color": cfg.plug_color,
        "scene_table_mat": cfg.table_mat, "scene_board_mat": cfg.board_mat,
        "scene_lights": sum(l[0] for l in cfg.lights), "scene_cables": sum(c[0] for c in cfg.cables),
        "localization_error_mm": (cfg.localization_error[0] ** 2 + cfg.localization_error[1] ** 2) ** 0.5 * 1000.0,
        "grasp_axial_mm": cfg.grasp_offset[0] * 1000.0,
        "sim_seconds": outcome.seconds,
    }
    if station._hard is not None:                           # hard mode's draws, for the evaluator only. Easy rows are as in S4
        hard = station._hard
        row.update({"profile": station.profile, "spec_jam_kind": spec.jam_kind, "spec_wedge_hold_n": spec.wedge_hold_n,
                    "hard_lot": hard.lot,
                    "hard_lot_factor": hard.lot_factor, "hard_cable_azimuth_deg": hard.cable_azimuth * 57.29577951308232,
                    "hard_cable_gap_mm": hard.cable_gap_m * 1000.0})
    if getattr(station, "_sense_on", False) and getattr(station, "_lash_on", False):   # hard mode v2's servo draws
        arm = outcome_arm(station)
        row.update({"sense_lash_deg": json.dumps([round(float(v) * 57.29577951308232, 4) for v in station._lash[arm]]),
                    "sense_phase_counts": json.dumps([round(float(v), 4) for v in station._enc_phase[arm]])})
    return row


def outcome_arm(station) -> str:
    """The arm that holds the plug in an injected state."""
    from ..station.mj_plug.inject import ARM

    return ARM


def add_claim_frames(row: dict, station, outcome, frame_store, frames) -> None:
    """Store the claim-time frames of a state with their calibration, and the taught port position.

    ``frame_store`` is the harness's FrameStore when the frames were already rendered, else None.
    """
    seed = outcome.seed
    if frame_store is not None and frame_store.keys("claim/"):
        items = {key.split("/", 1)[1]: frame_store.get(key) for key in frame_store.keys("claim/")}
    else:
        items = station.observe().frames
    for cam, frame in items.items():
        row[f"frame_{cam}"] = frames.add(f"{seed}_{cam}.jpg", frame.rgb)
        # the calibration that goes with the frame, so a judge can crop it later without the simulator
        row[f"calib_{cam}"] = json.dumps({"K": [round(float(v), 6) for v in frame.intrinsics.ravel()],
                                          "T": [round(float(v), 9) for v in frame.extrinsics.ravel()]})
    nominal = station.nominal()
    port = nominal.port_position(outcome.target_port)
    row["nominal_port_x"], row["nominal_port_y"], row["nominal_port_z"] = (float(v) for v in port)
    row["nominal_board_yaw"] = float(nominal.board_yaw)
    for cam, info in getattr(station, "_frame_info", {}).items():     # hard mode: what each webcam did for its claim frame
        row[f"caminfo_{cam}"] = json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in info.items()})


def generate_block(args: tuple) -> dict:
    """Worker: inject every seed of one block, write its states part and its frames zip."""
    connector, split, seeds, root, final_run = args
    from ..station.mj_plug.inject import scripted_insert

    unlock_final_run(final_run)
    for seed in seeds:
        assert_simulable(seed)
    profile = store.profile_of_split(split)
    t0 = time.perf_counter()
    frames = store.FrameZip(store.part_path(connector, split, "frames", seeds, root))
    rows = []
    try:
        for seed in seeds:
            w0 = time.perf_counter()
            station, outcome = scripted_insert(seed, connector, profile=profile)
            row = state_row(station, outcome, split)
            add_claim_frames(row, station, outcome, None, frames)
            row["frames_zip"] = frames.path.name
            row["wall_s"] = round(time.perf_counter() - w0, 3)
            rows.append(row)
            station.close()
    finally:
        frames.close()
    store.write_table_atomic(rows, store.part_path(connector, split, "states", seeds, root))
    return {"block": store.part_name(seeds), "n": len(rows), "wall_s": round(time.perf_counter() - t0, 1)}


def generate(connector: str, split: str, seeds: list[int] | None = None, workers: int = 8,
             time_budget_s: float | None = None, root=None, final_run: str | None = None) -> dict:
    """Generate the states of a split. Returns how many blocks were done, skipped and left."""
    exp = load_experiment()
    if not exp.connector(connector).modeled:
        raise ValueError(f"connector {connector!r} is not modeled yet (no geometry or force tables in the config)")
    all_seeds = list(seeds) if seeds is not None else list(store.split_seeds(split))
    store.check_seeds_in_split(all_seeds, split)
    guard(all_seeds, purpose=f"zoo states, {connector} {split}", final_run=final_run)
    unlock_final_run(final_run)                             # workers inherit it. None closes the door again
    todo, skipped = [], 0
    for block in store.blocks(all_seeds):
        if store.part_path(connector, split, "states", block, root).is_file():
            skipped += 1
        else:
            todo.append(block)
    t0 = time.perf_counter()
    done, left = 0, len(todo)
    if todo:
        # a fresh process per block: a worker grows about 3 MB per state, and the machine has little to spare
        with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as pool:
            pending = {}
            queue = list(todo)

            def submit() -> None:
                while queue and len(pending) < workers:
                    block = queue.pop(0)
                    pending[pool.submit(generate_block, (connector, split, block, root, final_run))] = block

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
                    queue.clear()                           # let the running blocks finish, start no more
                submit()
        left = len(todo) - done
    summary = {"connector": connector, "split": split, "blocks_done": done, "blocks_skipped": skipped,
               "blocks_left": left, "wall_s": round(time.perf_counter() - t0, 1), "config_hash": config_hash()}
    if left == 0:
        table = store.merge(connector, split, "states", root)
        summary["states"] = int(len(table))
        summary["class_counts"] = {k: int(v) for k, v in table["cls"].value_counts().sort_index().items()}
        summary["landed_in_class"] = float((table["cls"] == table["landed_cls"]).mean())
        summary["positive_share"] = float(table["label"].mean())
        store.write_manifest(connector, split, {"states": summary}, root)
    return summary
