"""The plug station's acceptance checks (docs/02_plug_station.md, the S2 gate). Evaluator-side.

Each check is a small job run over dev seeds in worker processes. The numbers go into
``reports/S2.md`` as they come out. Nothing here lowers or rewords a gate: a check either
meets the number in the doc or the table says it did not.

Run with ``uv run avail station accept --connector fric``.
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from statistics import median

import numpy as np

from ...config import load_experiment
from ...seeds import guard
from .inject import ARM, CLASSES, ErrorSpec, sample_error, scripted_insert
from .oracle import do_nothing, oracle_cycle
from .truth import truth

PULL_SPEED = 0.003
SEATED_PULL = {"latch": (5.0, 0.3), "fric": (3.0, 0.5), "prong": (3.0, 0.5)}      # N, mm limit
SHORT_PULL_N = 2.0
SHORT_PULL_MOVE_MM = 2.0
CHUNK = 10                      # jobs per worker process


def _primary(connector: str) -> str:
    return "latched" if connector == "latch" else "seated"


def _pull(station, est_cap_n: float, travel_m: float, stop_at_true_n: float | None = None) -> dict:
    """Pull the plug straight up. Reports how far the plug really moved and the real pull force. Privileged."""
    z0 = truth(station)["metrics"]["plug_pos"][2]
    log = {"max_true_pull": 0.0, "disp_at_target": None}

    def hook() -> None:
        pull = station._true_fz
        log["max_true_pull"] = max(log["max_true_pull"], pull)
        if stop_at_true_n is not None and log["disp_at_target"] is None and pull >= stop_at_true_n:
            data, ids = station._mj_data, station._lay.ids
            log["disp_at_target"] = (float(data.qpos[ids["plug_qadr"] + 2]) - z0) * 1000.0
            station.stop()

    station._tick_hook = hook
    try:
        result = station.move_relative(ARM, [0.0, 0.0, travel_m], "base", PULL_SPEED, est_cap_n, travel_m)
    finally:
        station._tick_hook = None
    z1 = truth(station)["metrics"]["plug_pos"][2]
    return {"stopped_by": result.stopped_by, "tool_travel_mm": result.travel_m * 1000.0,
            "plug_moved_mm": (z1 - z0) * 1000.0, "max_true_pull_n": log["max_true_pull"],
            "disp_at_target_mm": log["disp_at_target"], "est_force_at_stop_n": result.force_at_stop_n}


def job(args: tuple) -> dict:
    """One acceptance job. Runs in a worker process."""
    kind, connector, seed = args[0], args[1], int(args[2])
    exp = load_experiment()
    primary = _primary(connector)
    t0 = time.perf_counter()
    out: dict = {"kind": kind, "connector": connector, "seed": seed}
    if kind == "oracle":
        station, res = oracle_cycle(seed, connector)
        out["seated"] = bool(res["truth_held"]["predicates"][primary])
        out["released_ok"] = bool(res["truth_released"]["predicates"]["released_stable"]
                                  and res["truth_released"]["predicates"][primary])
        out["flag"] = bool(res["flag"])
        out["sim_seconds"] = res["sim_seconds"]
    elif kind == "nothing":
        station, res = do_nothing(seed, connector)
        out["seated"] = bool(res["truth"]["predicates"][primary])
    elif kind == "seated_pull":
        force_n, _ = SEATED_PULL[connector]
        log: list[tuple[float, float]] = []
        station, ins = _insert_with_log(seed, connector, ErrorSpec("P0"), log)
        out["landed"] = ins.landed_class
        out.update(_click_numbers(log, ins, exp.connector(connector)))
        if ins.landed_class == "P0":
            pull = _pull(station, est_cap_n=force_n + 4.0, travel_m=0.003, stop_at_true_n=force_n)
            out["disp_at_force_mm"] = pull["disp_at_target_mm"]
            out["max_true_pull_n"] = pull["max_true_pull_n"]
            out["still_ok"] = bool(truth(station)["predicates"][primary])
    elif kind == "short_pull":
        station, ins = scripted_insert(seed, connector, ErrorSpec("N2", shortfall_mm=2.0))
        out["landed"] = ins.landed_class
        out["shortfall_mm"] = ins.truth["metrics"]["shortfall_mm"]
        # The gate is about the connector: a real pull of at most 2 N. The arm's own force estimate is
        # too noisy to hold a 2 N limit, so the pull is stopped on the real force and the cap on the
        # estimate is set out of the way. The estimate-limited version is reported next to it.
        snap = station._snapshot()
        pull = _pull(station, est_cap_n=SHORT_PULL_N + 6.0, travel_m=0.003, stop_at_true_n=SHORT_PULL_N)
        out["plug_moved_mm"] = pull["plug_moved_mm"]
        out["max_true_pull_n"] = pull["max_true_pull_n"]
        out["stopped_by"] = pull["stopped_by"]
        station._restore(snap)
        est = _pull(station, est_cap_n=SHORT_PULL_N, travel_m=0.003)
        out["est_limited_moved_mm"] = est["plug_moved_mm"]
        out["est_limited_max_true_pull_n"] = est["max_true_pull_n"]
    elif kind == "push_repair":
        station, ins = scripted_insert(seed, connector, sample_error(seed, connector, "N2"))
        out["landed"] = ins.landed_class
        probe = exp.probes.push_seat
        res = station.move_relative(ARM, [0.0, 0.0, -probe.travel_cap_mm / 1000.0], "base", probe.speed_mm_s / 1000.0,
                                    exp.push_force_cap_n(connector), probe.travel_cap_mm / 1000.0)
        station.wait(0.2)
        out["push_travel_mm"] = res.travel_m * 1000.0
        out["push_stopped_by"] = res.stopped_by
        out["seated_after"] = bool(truth(station)["predicates"][primary])
        out["true_peak_push_n"] = station._true_peak_push
    elif kind.startswith("class:"):
        cls = kind.split(":", 1)[1]
        spec = sample_error(seed, connector, "P0" if cls == "P0r" else cls)
        if cls == "P0r":
            spec = ErrorSpec("P0", release=True)
        elif cls == "P0":
            spec = ErrorSpec("P0")
        station, ins = scripted_insert(seed, connector, spec)
        out["landed"] = ins.landed_class
        out["flag"] = bool(ins.flag)
        out["ok"] = ins.landed_class == spec.cls
    elif kind == "hash":
        station, ins = scripted_insert(seed, connector, sample_error(seed, connector))
        first = station.observe()
        h = hashlib.sha256()
        h.update(station._mj_data.qpos.tobytes())
        h.update(station._mj_data.qvel.tobytes())
        for name in sorted(first.frames):
            h.update(first.frames[name].rgb.tobytes())
        out["hash"] = h.hexdigest()
        out["landed"] = ins.landed_class
    else:
        raise ValueError(f"unknown acceptance job {kind!r}")
    station.close()
    out["wall_s"] = round(time.perf_counter() - t0, 2)
    return out


def _insert_with_log(seed: int, connector: str, spec: ErrorSpec, log: list):
    """A scripted insertion that also records (shortfall mm, real downward force N) each tick. Privileged."""

    def record(station) -> None:
        ret = station._retention
        if ret.captured:
            log.append((ret.shortfall_mm, -station._true_fz))

    return scripted_insert(seed, connector, spec, recorder=record)


def _click_numbers(log: list, ins, spec) -> dict:
    """Peak push through the detent and the drop after it, from the real force and from the skill's own trace."""
    out: dict = {}
    if not log:
        return out
    arr = np.array(log)
    crossover = _crossover_mm(spec)
    ramp = arr[(arr[:, 0] > crossover) & (arr[:, 0] <= 1.7)]
    snap = arr[(arr[:, 0] <= crossover) & (arr[:, 0] > 0.03)]
    if len(ramp):
        out["true_peak_n"] = float(ramp[:, 1].max())
        out["true_min_after_n"] = float(snap[:, 1].min()) if len(snap) else float("nan")
    travel = np.array(ins.insert.trace["travel_mm"])
    force = np.array(ins.insert.trace["force_n"])
    end = float(travel[-1])
    window = (travel > end - 1.7) & (travel < end - crossover)
    after = travel >= end - crossover
    if window.any() and after.any():
        peak_idx = np.where(window)[0][np.argmax(force[window])]
        out["est_peak_n"] = float(force[peak_idx])
        tail = np.where(after)[0]
        out["est_min_after_n"] = float(force[tail[: max(1, len(tail) // 2)]].min())
    return out


def _crossover_mm(spec) -> float:
    pts = spec.detent_n
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if y0 >= 0.0 > y1:
            return x0 + (x1 - x0) * (y0 / (y0 - y1)) if y0 != y1 else x0
    return 0.45


def _job_chunk(chunk: list[tuple]) -> list[dict]:
    return [job(args) for args in chunk]


def _run_chunks(chunks: list[list[tuple]], workers: int) -> list[list[dict]]:
    """Every chunk in a fresh worker process, at most ``workers`` at a time. Results in chunk order.

    A worker grows with every station it builds, so it retires after 1 chunk. Chunks go in only
    as workers free up: with ``pool.map`` queuing every job at once, a retired worker was never
    replaced on this Python and the run hung. The 2 determinism passes are separate chunks, so
    they run in 2 different processes.
    """
    out: list = [None] * len(chunks)
    queue = list(enumerate(chunks))
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as pool:
        pending: dict = {}
        while queue or pending:
            while queue and len(pending) < workers:
                i, chunk = queue.pop(0)
                pending[pool.submit(_job_chunk, chunk)] = i
            done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
            for fut in done:
                out[pending.pop(fut)] = fut.result()
    return out


def _share(flags: list[bool]) -> dict:
    from ...stats import wilson

    n, k = len(flags), int(sum(bool(f) for f in flags))
    lo, hi = wilson(k, n) if n else (float("nan"), float("nan"))
    return {"k": k, "n": n, "share": k / n if n else float("nan"), "lo": lo, "hi": hi}


def run_acceptance(connector: str, seeds: list[int], workers: int = 12, classes: list[str] | None = None) -> dict:
    """Run checks 1 to 7 for one connector over dev seeds. Returns every number."""
    guard(seeds, purpose=f"S2 acceptance checks, {connector}")
    exp = load_experiment()
    spec = exp.connector(connector)
    if classes is None:
        classes = ["P0", "P0r"] + [c for c in CLASSES[1:] if not (c == "N5" and connector != "latch")]
    jobs: list[tuple] = []
    for seed in seeds:
        jobs += [("oracle", connector, seed), ("nothing", connector, seed), ("seated_pull", connector, seed),
                 ("short_pull", connector, seed), ("push_repair", connector, seed)]
        jobs += [(f"class:{c}", connector, seed) for c in classes]
    hash_seeds = seeds[:6]
    hash_pass = [("hash", connector, s) for s in hash_seeds]
    chunks = [jobs[i:i + CHUNK] for i in range(0, len(jobs), CHUNK)] + [hash_pass, list(hash_pass)]
    t0 = time.perf_counter()
    results = [r for chunk in _run_chunks(chunks, workers) for r in chunk]
    wall = time.perf_counter() - t0
    by = {}
    for r in results:
        by.setdefault(r["kind"], []).append(r)

    force_n, limit_mm = SEATED_PULL[connector]
    seated_pull = [r for r in by["seated_pull"] if r.get("disp_at_force_mm") is not None]
    disp = [r["disp_at_force_mm"] for r in seated_pull]
    short = by["short_pull"]
    click = [r for r in by["seated_pull"] if "true_peak_n" in r]
    peak_cfg = spec.insertion_peak_n
    hashes = {}
    for r in by["hash"]:
        hashes.setdefault(r["seed"], set()).add(r["hash"])

    out = {
        "connector": connector, "seeds": [seeds[0], seeds[-1]], "n_seeds": len(seeds), "wall_s": round(wall, 1),
        "check1_oracle_seats": _share([r["seated"] for r in by["oracle"]]),
        "check1_oracle_released_ok": _share([r["released_ok"] for r in by["oracle"]]),
        "check1_do_nothing_seats": _share([r["seated"] for r in by["nothing"]]),
        "oracle_sim_seconds_median": median(r["sim_seconds"] for r in by["oracle"]),
        "check2_seated_pull": {
            "pull_n": force_n, "limit_mm": limit_mm, "n": len(disp),
            "max_mm": max(disp) if disp else float("nan"), "median_mm": median(disp) if disp else float("nan"),
            "under_limit": _share([d < limit_mm for d in disp]),
            "not_reached_force": len(by["seated_pull"]) - len(seated_pull),
            "still_seated_after": _share([r["still_ok"] for r in seated_pull]),
        },
        "check3_short_pull": {
            "cap_n": SHORT_PULL_N, "must_move_mm": SHORT_PULL_MOVE_MM, "n": len(short),
            "moved_enough": _share([r["plug_moved_mm"] > SHORT_PULL_MOVE_MM and r["max_true_pull_n"] <= SHORT_PULL_N for r in short]),
            "median_moved_mm": median(r["plug_moved_mm"] for r in short),
            "max_true_pull_n": max(r["max_true_pull_n"] for r in short),
            "est_limited_moved_enough": _share([r["est_limited_moved_mm"] > SHORT_PULL_MOVE_MM for r in short]),
            "est_limited_median_moved_mm": median(r["est_limited_moved_mm"] for r in short),
            "landed_n2": _share([r["landed"] == "N2" or abs(r["shortfall_mm"] - 2.0) < 0.2 for r in short]),
        },
        "check4_click": {
            "configured_peak_n": peak_cfg, "n": len(click),
            "true_peak_median_n": median(r["true_peak_n"] for r in click) if click else float("nan"),
            "true_peak_within_20pct": _share([abs(r["true_peak_n"] - peak_cfg) <= 0.2 * peak_cfg for r in click]),
            "true_drop_median_n": median(r["true_peak_n"] - r["true_min_after_n"] for r in click) if click else float("nan"),
            "true_drops": _share([r["true_peak_n"] - r["true_min_after_n"] > 0.3 * peak_cfg for r in click]),
            "est_peak_median_n": median(r["est_peak_n"] for r in click if "est_peak_n" in r) if click else float("nan"),
            "est_peak_within_20pct": _share([abs(r["est_peak_n"] - peak_cfg) <= 0.2 * peak_cfg for r in click if "est_peak_n" in r]),
            "est_drop_median_n": median(r["est_peak_n"] - r["est_min_after_n"] for r in click if "est_peak_n" in r) if click else float("nan"),
        },
        "check5_push_repairs_n2": {
            "seated_after": _share([r["seated_after"] for r in by["push_repair"] if r["landed"] == "N2"]),
            "started_as_n2": _share([r["landed"] == "N2" for r in by["push_repair"]]),
            "worst_true_push_n": max(r["true_peak_push_n"] for r in by["push_repair"]),
            "push_cap_n": exp.push_force_cap_n(connector),
        },
        "check6_classes": {c: _share([r["ok"] for r in by[f"class:{c}"]]) for c in classes},
        "check6_landed": {c: _counts([r["landed"] for r in by[f"class:{c}"]]) for c in classes},
        "skill_flag_true": {c: _share([r["flag"] for r in by[f"class:{c}"]]) for c in classes},
        "check7_determinism": {"seeds": hash_seeds, "runs_per_seed": 2,
                               "identical": all(len(v) == 1 for v in hashes.values())},
    }
    return out


def _counts(items: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        out[item] = out.get(item, 0) + 1
    return dict(sorted(out.items()))


def summarize(res: dict) -> str:
    """The acceptance table as text."""
    c1, c2, c3, c4, c5 = (res["check1_oracle_seats"], res["check2_seated_pull"], res["check3_short_pull"],
                          res["check4_click"], res["check5_push_repairs_n2"])
    lines = [
        f"connector {res['connector']}, dev seeds {res['seeds'][0]} to {res['seeds'][1]}, {res['wall_s']} s",
        f" 1  oracle seats           {c1['k']}/{c1['n']} (gate 95%)   do-nothing seats {res['check1_do_nothing_seats']['k']}/{res['check1_do_nothing_seats']['n']} (gate 0)",
        f"    oracle then release    {res['check1_oracle_released_ok']['k']}/{res['check1_oracle_released_ok']['n']} stay seated",
        f" 2  seated, {c2['pull_n']:.0f} N pull      max {c2['max_mm']:.3f} mm, median {c2['median_mm']:.3f} mm (gate < {c2['limit_mm']} mm), n {c2['n']}",
        f" 3  2 mm short, <= {c3['cap_n']:.0f} N pull moved > {c3['must_move_mm']:.0f} mm on {c3['moved_enough']['k']}/{c3['moved_enough']['n']}, median {c3['median_moved_mm']:.2f} mm, max real pull {c3['max_true_pull_n']:.2f} N",
        f"    same, cap on the estimate moved > {c3['must_move_mm']:.0f} mm on {c3['est_limited_moved_enough']['k']}/{c3['est_limited_moved_enough']['n']}, median {c3['est_limited_median_moved_mm']:.2f} mm",
        f" 4  insertion peak         real {c4['true_peak_median_n']:.2f} N vs configured {c4['configured_peak_n']:.1f} N, within 20% on {c4['true_peak_within_20pct']['k']}/{c4['true_peak_within_20pct']['n']}, drop {c4['true_drop_median_n']:.2f} N",
        f"    as the arm estimates   peak {c4['est_peak_median_n']:.2f} N, within 20% on {c4['est_peak_within_20pct']['k']}/{c4['est_peak_within_20pct']['n']}, drop {c4['est_drop_median_n']:.2f} N",
        f" 5  capped push on N2      seats {c5['seated_after']['k']}/{c5['seated_after']['n']} (gate 90%), worst real push {c5['worst_true_push_n']:.1f} N at cap {c5['push_cap_n']:.1f} N",
        " 6  class lands in class   " + ", ".join(f"{c} {v['k']}/{v['n']}" for c, v in res["check6_classes"].items()),
        f" 7  determinism            {'identical' if res['check7_determinism']['identical'] else 'DIFFERENT'} across 2 processes on {len(res['check7_determinism']['seeds'])} seeds",
    ]
    return "\n".join(lines)


def save(res: dict, path) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)
