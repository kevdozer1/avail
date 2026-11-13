"""S4b: the classical gap reading on hard mode's wrist camera, measured on dev seeds. Evaluator-side.

For each dev seed and each class in the list, the same injected state (same seed, class and
shortfall) is built in easy mode and in hard mode. ``vision.gap``'s ``measure_gap`` reads the
claim-time frame of the holding arm's wrist camera, and its reading is compared with the true
shortfall from ground truth. For each hard frame it also records:

- pixels per millimeter at the port: in the raw 1080p frame, across the line of sight, and in the
  working frame, as ``vision.gap`` computes it
- what the camera did: exposure, gain, the blur's sd, the MJPEG size
- occlusion: how much of the strip where ``vision.gap`` reads the shoulder edge (the plug body's
  near face, 0 to 3 mm above the shoulder, across the 2 flank bands) a finger or the cable hides,
  from a segmentation render of the same view

Nothing here tunes anything. The numbers go into reports/S4b.md as measured.
"""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

HOLD_CAMERA = "right_camera_d405"
FLANK_MARGIN_MM = 0.6                 # as vision.gap's flank bands
STRIP_MM = (0.3, 3.0)                 # heights above the shoulder that are sampled on the near face


def _near_face(station):
    """The plug body's face that looks at the holding wrist camera, from truth: (origin, across, out, up, half width, offset)."""
    m, d = station._mj_model, station._mj_data
    plug = station._lay.ids["body:plug"]
    origin = d.xpos[plug].copy()
    rot = d.xmat[plug].reshape(3, 3)
    x_p, y_p, z_p = rot[:, 0], rot[:, 1], rot[:, 2]
    cam = m.camera(HOLD_CAMERA).id
    c = d.cam_xpos[cam] - origin
    bx, by, _ = station._lay.geom.body
    tx, ty = station._lay.geom.tongue
    if abs(float(c @ y_p)) >= abs(float(c @ x_p)):
        n = y_p * (1.0 if c @ y_p >= 0 else -1.0)
        return origin, np.cross(n, z_p), n, z_p, bx / 2, by / 2, tx / 2
    n = x_p * (1.0 if c @ x_p >= 0 else -1.0)
    return origin, np.cross(n, z_p), n, z_p, by / 2, bx / 2, ty / 2


def occlusion(station) -> dict:
    """Share of the shoulder strip that a finger or the cable hides in the holding wrist camera. Hard mode."""
    mujoco = station._mj
    model, data = station._mj_model, station._mj_data
    cam = station._webcams[HOLD_CAMERA]
    plan = cam.plan
    mujoco.mj_kinematics(model, data)
    mujoco.mj_camlight(model, data)
    origin, a, n, z, wb, db, wt = _near_face(station)
    mm = 1e-3
    pts = []
    for side in (-1.0, 1.0):
        for s in np.linspace(wt + FLANK_MARGIN_MM * mm, wb - FLANK_MARGIN_MM * mm, 6):
            for h in np.linspace(STRIP_MM[0] * mm, STRIP_MM[1] * mm, 6):
                pts.append(origin + side * s * a + (db + 0.0001) * n + h * z)
    pts = np.array(pts)
    cid = model.camera(HOLD_CAMERA).id
    r_cv = data.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
    pc = (pts - data.cam_xpos[cid]) @ r_cv
    u = plan["f"] * pc[:, 0] / pc[:, 2] + (plan["width"] - 1) / 2.0
    v = plan["f"] * pc[:, 1] / pc[:, 2] + (plan["height"] - 1) / 2.0
    renderer = station._renderer
    renderer.enable_segmentation_rendering()
    try:
        seg = station._render_view(HOLD_CAMERA, plan["width"], plan["height"])
    finally:
        renderer.disable_segmentation_rendering()
    ids = station._lay.ids
    cable = {ids[k] for k in ids if k.startswith("geom:plug_cable_h")}
    plug_geoms = {model.geom(name).id for name in ("plug_body",)}
    if "geom:plug_latch_tab" in ids:
        plug_geoms.add(ids["geom:plug_latch_tab"])
    arm_roots = {model.body(f"{arm}_arm").id for arm in ("left", "right")}
    counts = {"plug": 0, "finger": 0, "cable": 0, "other": 0, "outside": 0}
    for uu, vv in zip(u, v):
        iu, iv = int(round(uu)), int(round(vv))
        if not (0 <= iu < plan["width"] and 0 <= iv < plan["height"]):
            counts["outside"] += 1
            continue
        obj, typ = int(seg[iv, iu, 0]), int(seg[iv, iu, 1])
        if typ != int(mujoco.mjtObj.mjOBJ_GEOM) or obj < 0:
            counts["other"] += 1
        elif obj in plug_geoms:
            counts["plug"] += 1
        elif obj in cable:
            counts["cable"] += 1
        elif int(model.body_rootid[model.geom_bodyid[obj]]) in arm_roots:
            counts["finger"] += 1
        else:
            counts["other"] += 1
    total = len(pts)
    return {f"strip_{k}": v / total for k, v in counts.items()}


def measure_state(args: tuple) -> list[dict]:
    """Worker: 1 dev seed and 1 class, in easy and in hard mode."""
    seed, connector, cls = args
    from ..evidence.vision import measure_gap
    from ..station.mj_plug.inject import sample_error
    from ..zoo.harness import prepare

    rows = []
    for profile in ("easy", "hard"):
        t0 = time.perf_counter()
        spec = sample_error(seed, connector, cls, profile=profile)
        prepared = prepare(seed, connector, spec=spec, profile=profile)
        try:
            frame = prepared.params["frames"].get(f"claim/{HOLD_CAMERA}")
            gap = measure_gap(frame, prepared.params["nominal"], prepared.outcome.target_port)
            m = prepared.outcome.truth["metrics"]
            row = {"seed": seed, "connector": connector, "cls": cls, "profile": profile, "landed": prepared.outcome.landed_class,
                   "jam_kind": spec.jam_kind, "true_shortfall_mm": m["shortfall_mm"], "true_tilt_deg": m["tilt_deg"],
                   "in_bore": m["in_bore"], "frame_h": int(frame.rgb.shape[0]), "frame_w": int(frame.rgb.shape[1])}
            row.update({f"gap_{k}": v for k, v in gap.items()})
            if profile == "hard":
                info = prepared.station._frame_info.get(HOLD_CAMERA, {})
                row.update({f"cam_{k}": v for k, v in info.items() if isinstance(v, (int, float))})
                row.update(occlusion(prepared.station))
                row["lot_factor"] = prepared.station._hard.lot_factor
                row["cable_azimuth_deg"] = math.degrees(prepared.station._hard.cable_azimuth)
            row["seconds"] = time.perf_counter() - t0
            rows.append(row)
        finally:
            prepared.close()
    return rows


def run(connector: str, seeds: list[int], classes: list[str], workers: int = 5, out_dir: Path | None = None) -> dict:
    """Measure every (seed, class) pair. Writes a parquet of rows and a JSON summary. Dev seeds only."""
    import pandas as pd

    from ..paths import runs_dir
    from ..seeds import guard

    guard(seeds, purpose=f"S4b camera measurement, {connector}")
    if any(not (0 <= s <= 9999) for s in seeds):
        raise ValueError("the camera measurement uses dev seeds only")
    jobs = [(int(s), connector, c) for s in seeds for c in classes]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=20) as pool:
        pending, queue = set(), list(jobs)
        while queue or pending:
            while queue and len(pending) < workers:
                pending.add(pool.submit(measure_state, queue.pop(0)))
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                rows += fut.result()
    table = pd.DataFrame(rows).sort_values(["seed", "cls", "profile"]).reset_index(drop=True)
    out_dir = Path(out_dir) if out_dir is not None else runs_dir() / "s4b_camera"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_dir / f"camera_{connector}.parquet", index=False)
    summary = summarize(table)
    (out_dir / f"camera_{connector}.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8",
                                                       newline="\n")
    return summary


def _q(x: np.ndarray, q: float) -> float:
    return float(np.quantile(x, q)) if len(x) else float("nan")


def summarize(table) -> dict:
    """Reading error and failure share per profile and class, and the hard camera's numbers."""
    out: dict = {"rows": int(len(table)), "by": {}}
    for (profile, cls), part in table.groupby(["profile", "cls"]):
        found = part["gap_plug_found"].fillna(0).to_numpy() > 0
        err = (part["gap_gap_mm_est"] - part["true_shortfall_mm"]).to_numpy(dtype=float)[found]
        err = err[np.isfinite(err)]
        out["by"][f"{profile}/{cls}"] = {
            "n": int(len(part)), "read_share": float(found.mean()), "n_read": int(found.sum()),
            "median_abs_err_mm": _q(np.abs(err), 0.5), "p90_abs_err_mm": _q(np.abs(err), 0.9),
            "median_err_mm": _q(err, 0.5), "p10_err_mm": _q(err, 0.1), "p90_err_mm": _q(err, 0.9),
            "px_per_mm_median": float(np.nanmedian(part["gap_px_per_mm"])) if "gap_px_per_mm" in part else float("nan"),
        }
    for profile, part in table.groupby("profile"):
        found = part["gap_plug_found"].fillna(0).to_numpy() > 0
        err = (part["gap_gap_mm_est"] - part["true_shortfall_mm"]).to_numpy(dtype=float)[found]
        err = err[np.isfinite(err)]
        out["by"][f"{profile}/all"] = {"n": int(len(part)), "read_share": float(found.mean()),
                                       "median_abs_err_mm": _q(np.abs(err), 0.5), "p90_abs_err_mm": _q(np.abs(err), 0.9),
                                       "px_per_mm_median": float(np.nanmedian(part["gap_px_per_mm"]))}
    hard = table[table["profile"] == "hard"]
    if len(hard):
        cols = [c for c in hard.columns if c.startswith("cam_") or c.startswith("strip_")]
        out["hard_camera"] = {c: {"median": float(np.nanmedian(hard[c])), "p10": float(np.nanquantile(hard[c], 0.1)),
                                  "p90": float(np.nanquantile(hard[c], 0.9))} for c in cols}
        for name in ("strip_finger", "strip_cable"):
            if name in hard:
                v = hard[name].to_numpy(dtype=float)
                out["hard_camera"][name]["share_over_10pct"] = float((v > 0.10).mean())
                out["hard_camera"][name]["share_over_50pct"] = float((v > 0.50).mean())
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="avail s4b camera", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--connector", required=True)
    parser.add_argument("--seeds", required=True, help="low:high, high excluded. Dev seeds")
    parser.add_argument("--classes", default=None, help="comma-separated. Default: P0, N1, N2, N3, and N5 for latch")
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args(argv)
    lo, hi = (int(v) for v in args.seeds.split(":"))
    classes = args.classes.split(",") if args.classes else ["P0", "N1", "N2", "N3"] + (["N5"] if args.connector == "latch" else [])
    summary = run(args.connector, list(range(lo, hi)), classes, workers=args.workers)
    print(json.dumps(summary, indent=1, default=float), flush=True)
    return 0
