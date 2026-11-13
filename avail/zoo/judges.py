"""Run the VLM judges over stored zoo frames. Evaluator-side.

    avail zoo judges estimate                                  the worst-case cost of a 400-state run per connector
    avail zoo judges run --connector latch --split dev --model fake --variant crop
    avail zoo judges show --connector latch --split dev --seed 1002 --variant crop

Nothing here simulates a state. A judge run reads the claim-time frames a zoo run stored, puts
them in front of ``judge.vlm`` (``avail/evidence/judge.py``) and writes 1 row per seed to
``AVAIL_DATA/zoo/<connector>/<split>/judges/<model>__<variant>.parquet``. The one exception is
``reference_image``, which injects 1 seated plug on a dev seed, once, and caches the picture.

The wall. This module reads the states table, and the table holds ground truth. Truth is used
for 2 things only: to stratify a subsample by injected class, and, in other modules, to score the
answers afterwards. What reaches the judge is built by ``state_inputs``: the stored frames with
their calibration, the claim (predicate, plug and port) and the taught port position. The class,
the truth columns and the file names stay here. The judges table holds no truth column either:
join it to the states table on ``seed`` to score it.

What a states table must hold for the cropped variants: ``calib_<camera>`` (JSON with ``K``, the
3 x 3 intrinsics, and ``T``, the 4 x 4 camera-to-base extrinsics, both row by row) next to each
``frame_<camera>``, and the taught position of the target port in ``nominal_port_x``,
``nominal_port_y``, ``nominal_port_z`` with ``nominal_board_yaw``. ``avail zoo`` writes them. A
table from before it did has none, and the wrist camera's pose at claim time cannot be rebuilt
without the simulator, so for such a table the crop falls back to the central square of each
frame (480 x 480 of a 640 x 480 frame, no resize). That square holds the port in the wrist and
overhead views of this station, but not at its center, and the text sent says so. The
``crop_rule`` column records which rule a row used: ``port``, ``center`` or ``none``.

Money. Every call goes through ``avail.router``. ``run`` prints the worst-case estimate before
it calls anything, refuses a paid model when its key is not set, and stops when the estimate
is above what the spend session has left. A rerun is served from the router's cache and costs
0, which is how a run resumes. Answers from the fake model are placeholders: the table marks
them, and they measure nothing about a judge.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

from ..config import MissingCredentialError, load_experiment
from ..evidence import judge
from ..evidence.base import FrameStore
from ..paths import cache_dir, runs_dir
from ..router import ModelCallError, Router, RouterError, load_spend_caps
from ..router.smoke import skip_reason
from ..schemas import Claim, ClaimContext, Predicate
from ..seeds import TestSeedError, guard
from ..station.base import CameraFrame, NominalGeometry
from . import store

HOLDING_ARM, FREE_ARM = "right", "left"       # every zoo state is inserted by the right arm
REFERENCE_SEED = 3000                          # the dev seed of the reference picture
DEFAULT_STATES = 60                            # sessions/S3.md: at most 60 states per model on dev seeds
MAX_CALL_FAILURES = 3                          # failed calls in a row before a run stops
ALLOCATIONS = ("proportional", "equal")
ROW_COLUMNS = ("seed", "model", "variant", "p_yes", "holds", "confidence", "usd", "latency_s", "cached",
               "placeholder", "json_ok", "n_images", "crop_rule", "predicate", "port", "reason", "json_error",
               "truncated", "original_usd", "original_latency_s", "request_hash", "error")


# ----------------------------------------------------------------------------- which models
def default_judge_models(table=None) -> list[str]:
    """The ``e1_minimum`` list of ``config/models.yaml``, plus ``fable`` when its entry is verified."""
    table = table if table is not None else Router().models
    keys = list(table.e1_minimum)
    fable = table.models.get("fable")
    if fable is not None and fable.verified and "fable" not in keys:
        keys.append("fable")
    return keys


# ----------------------------------------------------------------------------- the subsample
def _allocate(counts: dict[str, int], n: int, allocation: str) -> dict[str, int]:
    """How many states to take from each class. The shares add up to ``n``, or to everything when ``n`` is larger."""
    names = sorted(counts)
    total = sum(counts.values())
    if n >= total:
        return dict(counts)
    if allocation == "proportional":
        quota = {c: n * counts[c] / total for c in names}
        take = {c: int(math.floor(quota[c])) for c in names}
        rest = n - sum(take.values())
        for c in sorted(names, key=lambda c: (-(quota[c] - take[c]), c))[:rest]:
            take[c] += 1
        return take
    take = {c: 0 for c in names}
    left = n
    open_ = [c for c in names if counts[c] > 0]
    while left > 0 and open_:
        share = max(1, left // len(open_))
        for c in list(open_):
            add = min(share, counts[c] - take[c], left)
            take[c] += add
            left -= add
            if take[c] >= counts[c]:
                open_.remove(c)
            if left == 0:
                break
    return take


OPEN_SPLITS = ("dev", "dev_hard", "dev_hard2")   # the splits a judge run may read without a final run


def subsample(connector: str, split: str, n: int, seed: int = 0, *, allocation: str = "proportional",
              final_run: str | None = None, root=None, classes: list[str] | None = None) -> list[int]:
    """A stratified sample of ``n`` seeds from a split's states table, by injected class. Sorted.

    The same arguments on the same table give the same seeds every time. ``proportional`` keeps
    the class mix of the split, so a rate pooled over the sample needs no weights. ``equal``
    takes the same number from every class, which gives the small classes more states and
    means a pooled rate must be weighted by class. Only the dev splits are read without
    ``final_run``. ``classes`` keeps only those injected classes (S4b: the near misses).
    """
    if allocation not in ALLOCATIONS:
        raise ValueError(f"allocation must be one of {', '.join(ALLOCATIONS)}, got {allocation!r}")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"n must be a whole number from 1 up, got {n!r}")
    _guard_split(split, final_run, list(store.split_seeds(split)) if split not in OPEN_SPLITS else [],
                 f"judge subsample, {connector} {split}")
    table = store.load(connector, split, "states", root)
    if classes is not None:
        table = table[table["cls"].isin(list(classes))]
    by_class: dict[str, list[int]] = {}
    for cls, state_seed in zip(table["cls"], table["seed"]):
        by_class.setdefault(str(cls), []).append(int(state_seed))
    take = _allocate({c: len(v) for c, v in by_class.items()}, n, allocation)
    picked: list[int] = []
    for cls in sorted(by_class):
        seeds = sorted(by_class[cls])
        rng = np.random.default_rng([int(seed), zlib.crc32(f"{connector}/{split}/{cls}".encode("utf-8"))])
        picked += [int(s) for s in rng.choice(seeds, size=take[cls], replace=False)]
    return sorted(picked)


def _guard_split(split: str, final_run: str | None, seeds: list[int], purpose: str) -> None:
    """Only the dev splits are open. Any other split needs the name of the final run, which goes to the seed guard."""
    if split not in OPEN_SPLITS and not final_run:
        raise TestSeedError(
            f"{purpose}: the {split} split is read by a final judge run only. Judge work on other days uses "
            "a dev split. Pass final_run='E1E2' (or 'E1bE2b' for hard mode) once Kevin has approved the judge run.")
    # A judge reads the frames the final run stored and simulates nothing, so a config changed since
    # that run (a new profile added beside it) does not refuse it. It is still recorded.
    guard(seeds, purpose=purpose, final_run=final_run, stored=True)


# ----------------------------------------------------------------------------- the estimate
def _typical_claim(connector: str, predicate: str, port: int) -> Claim:
    return Claim(claim_id="estimate", run_id="estimate", cycle_id="estimate", step_id="insert",
                 predicate=Predicate(name=predicate, target={"plug": "p1", "port": port}), raised_by="skill",
                 context=ClaimContext(task="plug_cycle", stage="insert", attempt=0))


def _typical_text(connector: str, variant: str, views: int) -> str:
    """The longest text a zoo state of this connector sends, so the estimate does not run low."""
    board = load_experiment().station.board
    ports = board.rows * board.cols
    texts = [judge.build_prompt(_typical_claim(connector, predicate, ports), connector, variant,
                                judge.typical_views(variant, views), n_ports=ports)
             for predicate in ("latched" if connector == "latch" else "seated", "released_stable")]
    return max(texts, key=len)


def _typical_images(variant: str, views: int) -> list[np.ndarray]:
    """Blank images of the sizes a call sends: whole frames, or crops with the reference in front."""
    render = load_experiment().station.render
    if variant == "full":
        return [np.zeros((render.height, render.width, 3), dtype=np.uint8) for _ in range(views)]
    crop = np.zeros((judge.CROP_OUT_PX, judge.CROP_OUT_PX, 3), dtype=np.uint8)
    return [crop for _ in range(views + (1 if variant == "crop_ref" else 0))]


def estimate(connector: str, n_states: int, models: list[str] | None = None, variants: list[str] | None = None, *,
             views: int = 2, router: Router | None = None) -> dict:
    """The worst-case cost of a judge run, without calling anything.

    1 row per model and variant: ``calls``, ``usd_per_call`` (what the router would reserve for 1
    call with the real image count and sizes) and ``usd_total``, which is ``calls`` times
    ``usd_per_call``. ``total_usd`` is the sum over the rows. ``views`` is the number of camera
    views per call: 2 for stored claim-time frames, the wrist view and the overhead view.
    """
    if isinstance(n_states, bool) or not isinstance(n_states, int) or n_states < 0:
        raise ValueError(f"n_states must be a whole number from 0 up, got {n_states!r}")
    router = router if router is not None else Router()
    table = router.models
    keys = list(models) if models is not None else default_judge_models(table)
    kinds = list(variants) if variants is not None else list(judge.VARIANTS)
    rows = []
    for key in keys:
        entry = table.get(key)
        allowance = judge.output_allowance(key, table)
        blocked = skip_reason(entry) if entry.paid else None
        for variant in kinds:
            if variant not in judge.VARIANTS:
                raise ValueError(f"unknown judge variant {variant!r}. Known: {', '.join(judge.VARIANTS)}")
            images = _typical_images(variant, views)
            per_call = float(router.estimate(key, _typical_text(connector, variant, views), images=images,
                                             json_schema=judge.ANSWER_SCHEMA, max_tokens=allowance))
            rows.append({
                "connector": connector, "model": key, "variant": variant, "calls": int(n_states),
                "images": len(images), "image_px": f"{images[0].shape[1]}x{images[0].shape[0]}",
                "max_tokens": allowance, "usd_per_call": per_call, "usd_total": n_states * per_call,
                "placeholder": entry.adapter == "fake", "blocked": blocked,
            })
    return {"connector": connector, "n_states": int(n_states), "views": int(views), "rows": rows,
            "total_usd": float(sum(row["usd_total"] for row in rows))}


def format_estimate(result: dict) -> str:
    """The estimate as a text table, with a subtotal per model and the grand total."""
    lines = [f"VLM judge cost estimate: {result['connector']}, {result['n_states']} states, {result['views']} camera "
             "views per call. No call is made.",
             "usd per call is the worst case the router reserves before a call: every image at the model table's "
             "token count and the whole output allowance spent.", ""]
    head = ("model", "variant", "calls", "images", "image px", "out tokens", "usd per call", "usd total")
    body = []
    models = list(dict.fromkeys(row["model"] for row in result["rows"]))
    for model in models:
        mine = [row for row in result["rows"] if row["model"] == model]
        for row in mine:
            body.append((row["model"], row["variant"], str(row["calls"]), str(row["images"]), row["image_px"],
                         str(row["max_tokens"]), f"{row['usd_per_call']:.6f}", f"{row['usd_total']:.2f}"))
        if len(mine) > 1:
            body.append((model, "subtotal", str(sum(r["calls"] for r in mine)), "", "", "", "",
                         f"{sum(r['usd_total'] for r in mine):.2f}"))
    body.append(("total", "", str(sum(r["calls"] for r in result["rows"])), "", "", "", "",
                 f"{result['total_usd']:.2f}"))
    widths = [max(len(head[i]), *(len(b[i]) for b in body)) for i in range(len(head))]
    lines.append("  ".join(h.ljust(w) if i < 2 else h.rjust(w) for i, (h, w) in enumerate(zip(head, widths))))
    lines.append("  ".join("-" * w for w in widths))
    for b in body:
        lines.append("  ".join(c.ljust(w) if i < 2 else c.rjust(w) for i, (c, w) in enumerate(zip(b, widths))).rstrip())
    blocked = {row["model"]: row["blocked"] for row in result["rows"] if row["blocked"]}
    if blocked:
        lines += ["", "Cannot be called now:"] + [f"  {model}: {why}" for model, why in blocked.items()]
    if any(row["placeholder"] for row in result["rows"]):
        lines += ["", "The fake model costs 0 and answers with placeholders. Its rows price nothing real."]
    try:
        caps = load_spend_caps()
        named = ", ".join(f"{name} ${cap:.2f}" for name, cap in caps.sessions.items())
        lines += ["", f"Spend caps in config/experiment.yaml: {named}. Build total ${caps.total_usd:.2f}."]
    except RouterError:
        pass
    lines.append("The measured cost of a call is known only after a paid run. A 60-state dev run measures it.")
    return "\n".join(lines)


def print_estimate(connector: str, n_states: int, models: list[str] | None = None, variants: list[str] | None = None,
                   **kwargs) -> dict:
    """Print the estimate table. This is what Kevin sees before any paid run. Returns the estimate."""
    result = estimate(connector, n_states, models, variants, **kwargs)
    print(format_estimate(result), flush=True)
    return result


# ----------------------------------------------------------------------------- the reference picture
def reference_path(connector: str, profile: str = "easy", create: bool = True) -> Path:
    """Where the reference picture of a connector lives. ``create=False`` makes no folder, for a check that only reads."""
    tag = "" if profile == "easy" else f"_{profile}"
    name = f"{connector}{tag}_seed{REFERENCE_SEED}_{int(judge.CROP_MM)}mm_{judge.CROP_OUT_PX}px.png"
    return cache_dir(create) / "reference" / name


def reference_image(connector: str, *, rebuild: bool = False, profile: str = "easy") -> np.ndarray:
    """An RGB crop of a correctly seated plug of this connector type, for the ``crop_ref`` variant.

    Built once: a seated plug is injected on dev seed 3000, ground truth confirms the claim
    holds, and the holding arm's wrist view is cropped around the port exactly as the judge
    crops. The picture is cached as a PNG under ``AVAIL_DATA/cache/reference/`` and read from
    there afterwards, so every call sends the same bytes and the router's cache keeps hitting.
    This is the one function of this module that simulates. Hard mode (S4b) has its own
    reference, from the hard-mode wrist camera.
    """
    from PIL import Image

    path = reference_path(connector, profile)
    if path.is_file() and not rebuild:
        return np.asarray(Image.open(path).convert("RGB"))
    from ..station.mj_plug.inject import ErrorSpec
    from .harness import prepare

    guard([REFERENCE_SEED], purpose=f"judge reference picture, {connector} {profile}")
    prepared = prepare(REFERENCE_SEED, connector, spec=ErrorSpec("P0", release=False), profile=profile)
    try:
        if prepared.outcome.landed_class != "P0" or not prepared.label():
            raise RuntimeError(f"dev seed {REFERENCE_SEED} did not give a seated {connector} plug "
                               f"(it landed in class {prepared.outcome.landed_class}), so it cannot be the reference")
        frame = prepared.params["frames"].get(f"claim/{HOLDING_ARM}_camera_d405")
        port = prepared.claim.predicate.target["port"]
        crop = judge.crop_to_port(frame, prepared.params["nominal"], port) if frame is not None else None
    finally:
        prepared.close()
    if crop is None:
        raise RuntimeError(f"the wrist view of dev seed {REFERENCE_SEED} does not show the port of the {connector} plug")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    Image.fromarray(crop.rgb).save(tmp, format="PNG")
    os.replace(tmp, path)
    return np.asarray(Image.open(path).convert("RGB"))


# ----------------------------------------------------------------------------- from a stored state to the judge's inputs
def _number(value) -> float | None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _calibration(text) -> tuple[np.ndarray | None, np.ndarray | None]:
    """(K, T) from a ``calib_<camera>`` cell, or (None, None) when the cell holds none."""
    if not isinstance(text, str) or not text.strip():
        return None, None
    try:
        doc = json.loads(text)
        k = np.asarray(doc["K"], dtype=float).reshape(3, 3)
        t = np.asarray(doc["T"], dtype=float).reshape(4, 4)
    except (ValueError, KeyError, TypeError):
        return None, None
    if not (np.all(np.isfinite(k)) and np.all(np.isfinite(t))):
        return None, None
    return k, t


def nominal_of(row: dict, connector: str) -> NominalGeometry | None:
    """The taught geometry of a stored state, rebuilt from its row and the config. No simulator.

    The board pose is set so that the target port sits exactly where the row says it was
    taught. The plug and socket dimensions are left at 0: the judge reads the port position,
    the number of ports and the arm names, and nothing else. None when the row has no taught
    port position.
    """
    x, y, z, yaw = (_number(row.get(name)) for name in
                    ("nominal_port_x", "nominal_port_y", "nominal_port_z", "nominal_board_yaw"))
    port = row.get("target_port")
    if None in (x, y, z, yaw) or port is None:
        return None
    from ..station.mj_plug import scene             # constants and the board layout. Nothing is simulated

    exp = load_experiment()
    ports = scene.board_layout()
    port = int(port)
    if not 1 <= port <= len(ports):
        return None
    px, py = ports[port - 1]
    c, s = math.cos(yaw), math.sin(yaw)
    return NominalGeometry(
        connector=connector, full_depth_m=exp.connector(connector).full_depth_mm * 1e-3,
        grasp_height_m=scene.GRASP_HEIGHT, board_xy=(x - (c * px - s * py), y - (s * px + c * py)), board_yaw=yaw,
        face_z=z, ports_board=ports, pitch_m=exp.station.board.pitch_mm * 1e-3,
        holding_arm=HOLDING_ARM, free_arm=FREE_ARM,
    )


def claim_of(row: dict, connector: str) -> Claim:
    """The claim a stored state raised: the predicate, the plug and the port. As ``harness.make_claim`` builds it."""
    seed = int(row["seed"])
    return Claim(
        claim_id=f"zoo_{connector}_{seed}", run_id="zoo", cycle_id=f"seed{seed}", step_id="insert",
        predicate=Predicate(name=str(row["claim_predicate"]), target={"plug": "p1", "port": int(row["target_port"])}),
        raised_by="skill", context=ClaimContext(task="plug_cycle", stage="insert", attempt=0),
    )


def state_inputs(row: dict, connector: str, split: str, root=None) -> tuple[Claim, dict]:
    """Everything the judge gets for 1 stored state: the claim and the params. This is the whole handoff.

    The params hold the holding arm's wrist frame and the overhead frame as they were stored, each
    with its calibration when the table has it, the taught geometry, the connector type and the
    arm. No truth column, no class and no file name goes in.
    """
    seed = int(row["seed"])
    frames = FrameStore()
    for camera in load_experiment().station.cameras:
        entry = row.get(f"frame_{camera}")
        if not isinstance(entry, str) or not entry:
            continue
        role = judge.view_of(f"claim/{camera}", HOLDING_ARM, FREE_ARM).role
        if role not in (judge.ROLE_HOLDING, judge.ROLE_OVERHEAD):
            continue
        k, t = _calibration(row.get(f"calib_{camera}"))
        rgb = store.read_frame(connector, split, seed, entry, root)
        frames.put(f"claim/{camera}", CameraFrame(name=camera, rgb=rgb, intrinsics=k, extrinsics=t, ts=0.0))
    params = {"frames": frames, "nominal": nominal_of(row, connector), "connector": connector, "arm": HOLDING_ARM}
    return claim_of(row, connector), params


def _crop_rule(request: judge.JudgeRequest) -> str:
    if request.variant == "full" or not request.views:
        return "none"
    return "port" if all(view.crop == "port" for view in request.views) else "center"


# ----------------------------------------------------------------------------- the run
def judges_path(connector: str, split: str, model: str, variant: str, root=None) -> Path:
    return store.split_dir(connector, split, root) / "judges" / f"{model}__{variant}.parquet"


def _write(rows: list[dict], path: Path) -> None:
    """Write the judges table whole or not at all."""
    import pandas as pd

    frame = pd.DataFrame(rows, columns=list(ROW_COLUMNS))
    frame = frame.astype({"seed": "int64", "p_yes": "float64", "confidence": "float64", "usd": "float64",
                          "latency_s": "float64", "original_usd": "float64", "original_latency_s": "float64",
                          "cached": "bool", "placeholder": "bool", "json_ok": "bool", "truncated": "bool",
                          "n_images": "int64", "port": "int64", "holds": "boolean"})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _row(seed: int, model: str, variant: str, claim: Claim, result: judge.JudgeResult | None, placeholder: bool,
         error: str | None = None) -> dict:
    row = dict.fromkeys(ROW_COLUMNS)
    row.update(seed=int(seed), model=model, variant=variant, usd=0.0, latency_s=0.0, cached=False,
               placeholder=bool(placeholder), json_ok=False, n_images=0, crop_rule="none", truncated=False,
               predicate=claim.predicate.name, port=int(claim.predicate.target["port"]), error=error)
    if result is None:
        return row
    request, response = result.request, result.response
    row.update(n_images=len(request.images), crop_rule=_crop_rule(request), usd=float(result.usd),
               latency_s=float(result.seconds), error=request.problem)
    if response is None:
        return row
    row.update(
        p_yes=result.p_yes, holds=result.holds, confidence=result.confidence, reason=result.reason,
        json_ok=result.p_yes is not None, json_error=result.json_error, cached=bool(response.cached),
        truncated=bool(response.truncated), request_hash=response.request_hash,
        original_usd=response.original_usd,
        original_latency_s=None if response.original_latency_ms is None else response.original_latency_ms / 1000.0,
    )
    return row


def run(connector: str, split: str, seeds: list[int], model: str, variant: str, session: str | None = None,
        max_states: int = DEFAULT_STATES, approved: bool = False, *, final_run: str | None = None,
        router: Router | None = None, root=None, reference: np.ndarray | None = None, echo: bool = True) -> dict:
    """Ask 1 model, with 1 variant, about every given seed of a stored split. Writes the judges table.

    The table is ``AVAIL_DATA/zoo/<connector>/<split>/judges/<model>__<variant>.parquet``: 1 row per
    seed with ``seed``, ``model``, ``variant``, ``p_yes``, ``holds``, ``confidence``, ``usd``, ``latency_s``,
    ``cached``, ``placeholder`` and ``json_ok``, plus what the call sent (``n_images``, ``crop_rule``,
    ``predicate``, ``port``, ``request_hash``), the model's ``reason``, and ``json_error``, ``truncated`` and
    ``error``. ``usd`` and ``latency_s`` are what this run paid and waited, so they are near 0 for a
    cached answer. ``original_usd`` and ``original_latency_s`` are what the answer cost when the
    provider computed it: use those for the cost of a check. The file holds the rows of this
    run and replaces an older file. To add states, give the old seeds again: their answers
    come from the cache for 0.

    Order of the checks. The seed guard runs before anything is read: only the dev split is open,
    and any other split needs ``final_run``, which is passed to ``avail.seeds.guard``. A paid model
    whose key is not set, that has no spend session, or whose session needs an OK it does not
    have, is refused: the reason is printed and returned under ``refused``. Then the worst-case
    estimate is printed, and a paid run stops there when the estimate is above what the session
    has left. At most ``max_states`` seeds run, the leading ones of the list.

    A call that fails at the provider gets a row with ``error`` and the run goes on. 3 failed
    calls in a row, a cap or a missing key stop the run. The rows gathered so far are written
    and ``stopped`` says why. Run it again to pick up: answered states cost 0.
    """
    def say(text: str = "") -> None:
        if echo:
            print(text, flush=True)

    if variant not in judge.VARIANTS:
        raise ValueError(f"unknown judge variant {variant!r}. Known: {', '.join(judge.VARIANTS)}")
    seeds = list(dict.fromkeys(int(s) for s in seeds))
    purpose = f"judge run, {connector} {split}, {model} {variant}"
    _guard_split(split, final_run, seeds, purpose)
    span = store.split_seeds(split)
    outside = [s for s in seeds if s not in span]
    if outside:
        raise ValueError(f"{len(outside)} seeds are not in the {split} split, for example {outside[:5]}")
    if router is None:
        router = Router(session=session, approved=approved, phase="zoo_judges")
    entry = router.models.get(model)
    summary: dict = {"connector": connector, "split": split, "model": model, "variant": variant,
                     "session": router.session, "path": None, "states": 0, "calls": 0, "cached": 0, "json_ok": 0,
                     "usd": 0.0, "estimate_usd": None, "placeholder": entry.adapter == "fake", "crop_rule": None,
                     "refused": None, "stopped": None}

    def refuse(reason: str) -> dict:
        summary["refused"] = reason
        say(f"judge run refused. {reason} No call was made.")
        return summary

    if entry.paid:
        blocked = skip_reason(entry)
        if blocked:
            return refuse(f"{model} cannot be called now: {blocked}.")
        if router.session is None:
            return refuse(f"{model} is a paid model and no spend session was named. The 60-state dev run uses "
                          "session S3.")
        if router.caps.needs_approval(router.session) and not router.approved:
            return refuse(f"Spend session {router.session} needs Kevin's OK before any paid call. Pass "
                          "approved=True only after he has given it.")

    table = store.load(connector, split, "states", root)
    rows = {int(r["seed"]): r for r in table.to_dict("records")}
    missing = [s for s in seeds if s not in rows]
    if missing:
        raise ValueError(f"{len(missing)} seeds have no row in the states table of {connector} {split}, "
                         f"for example {missing[:5]}")
    if len(seeds) > max_states:
        say(f"{len(seeds)} seeds were given and max_states is {max_states}. Running the leading {max_states}.")
        seeds = seeds[:max_states]
    if variant == "crop_ref" and reference is None:
        reference = reference_image(connector, profile=store.profile_of_split(split))
    settings = {"model": model, "variant": variant, "router": router, "reference": reference}
    allowance = judge.output_allowance(model, router.models)

    # the estimate, before any call: what the router would reserve for each state's own request
    per_call = []
    rules = set()
    for seed in seeds:
        claim, params = state_inputs(rows[seed], connector, split, root)
        request = judge.build_request(claim, {**params, "judge": settings})
        rules.add(_crop_rule(request))
        if request.problem is None:
            per_call.append(float(router.estimate(model, request.text, image_count=len(request.images),
                                                  json_schema=judge.ANSWER_SCHEMA, max_tokens=allowance)))
    worst = float(sum(per_call))
    summary["estimate_usd"] = worst
    summary["crop_rule"] = "/".join(sorted(rules)) if rules else None
    say(f"estimate for {connector} {split}, {model}, {variant}: {len(per_call)} calls, worst case "
        f"${max(per_call, default=0.0):.6f} per call and ${worst:.4f} in total. A cached answer costs 0.")
    if "center" in rules:
        say(f"The states table of {connector} {split} holds no camera calibration, so the crops are the central "
            "square of each frame. Generate the split again with the current `avail zoo` to crop around the port.")
    if entry.paid or worst > 0:
        if router.session is None:
            return refuse(f"{model} costs money and no spend session was named.")
        room, cap = router.remaining(), router.cap_usd
        if worst > room + 1e-9:
            return refuse(f"The worst-case estimate ${worst:.4f} is above what session {router.session} has left "
                          f"(${room:.4f} of its ${cap:.2f} cap). Run fewer states, or a cheaper model or variant.")

    path = judges_path(connector, split, model, variant, root)
    out: list[dict] = []
    failures = 0
    t0 = time.perf_counter()
    for i, seed in enumerate(seeds, 1):
        claim, params = state_inputs(rows[seed], connector, split, root)
        params["judge"] = settings
        try:
            result = judge.judge(claim, params)
        except ModelCallError as exc:
            failures += 1
            out.append(_row(seed, model, variant, claim, None, summary["placeholder"],
                            error=f"{type(exc).__name__}: {exc}"[:300]))
            if failures >= MAX_CALL_FAILURES:
                summary["stopped"] = f"{failures} calls in a row failed. The last error: {exc}"
                break
            continue
        except (RouterError, MissingCredentialError) as exc:      # a cap, a key, a session: nothing more can be sent
            summary["stopped"] = f"{type(exc).__name__}: {exc}"
            break
        failures = 0
        out.append(_row(seed, model, variant, claim, result, summary["placeholder"]))
        if i % 20 == 0:
            _write(out, path)
            say(f"  {i}/{len(seeds)} states, {time.perf_counter() - t0:.0f} s")
    _write(out, path)
    answered = [r for r in out if r["request_hash"] is not None]
    summary.update(path=str(path), states=len(out), calls=sum(1 for r in answered if not r["cached"]),
                   cached=sum(1 for r in answered if r["cached"]), json_ok=sum(1 for r in out if r["json_ok"]),
                   usd=float(sum(r["usd"] for r in out)))
    say(f"judge run {connector} {split}, {model}, {variant}: {summary['states']} states, {summary['calls']} answered "
        f"by the model and {summary['cached']} from the cache, {summary['json_ok']} valid answers, "
        f"${summary['usd']:.6f} charged.")
    if summary["placeholder"]:
        say("These answers are placeholders from the fake model. They measure nothing about a judge.")
    if summary["stopped"]:
        say(f"The run stopped early. {summary['stopped']}")
    say(f"table: {path}")
    return summary


# ----------------------------------------------------------------------------- look at what a state sends
def save_example(connector: str, split: str, seed: int, variant: str, out_dir=None, *, root=None,
                 reference: np.ndarray | None = None, final_run: str | None = None) -> dict:
    """Write the exact text and images 1 stored state would send, so a person can look before paying.

    Returns the paths: ``prompt`` and ``images``. Nothing is sent to a model.
    """
    from PIL import Image

    _guard_split(split, final_run, [int(seed)], f"judge example, {connector} {split}")
    table = store.load(connector, split, "states", root)
    match = table[table["seed"] == int(seed)]
    if len(match) != 1:
        raise ValueError(f"seed {seed} has no row in the states table of {connector} {split}")
    claim, params = state_inputs(match.to_dict("records")[0], connector, split, root)
    if variant == "crop_ref" and reference is None:
        reference = reference_image(connector, profile=store.profile_of_split(split))
    params["judge"] = {"model": "fake", "variant": variant, "reference": reference}
    request = judge.build_request(claim, params)
    folder = Path(out_dir) if out_dir is not None else runs_dir() / "judge_examples"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{connector}_{split}_{int(seed)}_{variant}"
    prompt = folder / f"{stem}_prompt.txt"
    prompt.write_text(request.text if request.problem is None else f"nothing would be sent: {request.problem}\n",
                      encoding="utf-8", newline="\n")
    images = []
    for i, image in enumerate(request.images, 1):
        path = folder / f"{stem}_image{i}.png"
        Image.fromarray(image).save(path, format="PNG")
        images.append(str(path))
    return {"prompt": str(prompt), "images": images, "crop_rule": _crop_rule(request), "problem": request.problem}


# ----------------------------------------------------------------------------- command line
def _list(text: str | None) -> list[str] | None:
    return [part.strip() for part in text.split(",") if part.strip()] if text else None


def _seed_range(text: str) -> list[int]:
    lo, hi = text.split(":")
    return list(range(int(lo), int(hi)))


def main(argv: list[str] | None = None) -> int:
    """``avail zoo judges``: ``estimate``, ``run`` and ``show``. Returns 0, or 2 when a run was refused or stopped."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="avail zoo judges", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_est = sub.add_parser("estimate", help="print the worst-case cost of a judge run. Calls nothing")
    p_est.add_argument("--connector", default=None, help="default: every modeled connector")
    p_est.add_argument("--n", type=int, default=None, help="states per connector (default: the config's judge subsample)")
    p_est.add_argument("--models", default=None, help="comma-separated model keys (default: e1_minimum plus fable)")
    p_est.add_argument("--variants", default=None, help="comma-separated: full, crop, crop_ref (default: all 3)")
    p_est.add_argument("--views", type=int, default=2, help="camera views per call (default 2)")

    splits = ["dev", "fit", "calibration", "test", "dev_hard", "fit_hard", "calibration_hard", "test_hard",
              "dev_hard2", "fit_hard2", "calibration_hard2", "test_hard2"]
    p_run = sub.add_parser("run", help="ask 1 model about a stratified sample of stored states")
    p_run.add_argument("--connector", required=True)
    p_run.add_argument("--split", default="dev", choices=splits)
    p_run.add_argument("--classes", default=None, help="comma-separated injected classes to sample from (S4b: N1,N2,N5)")
    p_run.add_argument("--model", required=True, help="a key of config/models.yaml, for example fake")
    p_run.add_argument("--variant", default="all", help="full, crop, crop_ref, a comma-separated list, or all")
    p_run.add_argument("--n", type=int, default=DEFAULT_STATES, help=f"states to run (default {DEFAULT_STATES})")
    p_run.add_argument("--seeds", type=_seed_range, default=None,
                       help="low:high, high excluded: run these seeds in place of a stratified sample")
    p_run.add_argument("--sample-seed", type=int, default=0, help="the seed of the stratified sample")
    p_run.add_argument("--allocation", default="proportional", choices=list(ALLOCATIONS))
    p_run.add_argument("--session", default=None, help="the spend session of a paid run, for example S3")
    p_run.add_argument("--approved", action="store_true", help="Kevin's OK for a session that needs one")
    p_run.add_argument("--final-run", default=None, help="E1E2 opens the splits other than dev, once")

    p_show = sub.add_parser("show", help="write the text and images 1 state would send. Calls nothing")
    p_show.add_argument("--connector", required=True)
    p_show.add_argument("--split", default="dev", choices=splits)
    p_show.add_argument("--seed", type=int, required=True)
    p_show.add_argument("--variant", default="all")
    p_show.add_argument("--out", default=None, help="folder (default AVAIL_DATA/runs/judge_examples)")
    p_show.add_argument("--final-run", default=None)

    args = parser.parse_args(argv)
    exp = load_experiment()
    if args.command == "estimate":
        connectors = [args.connector] if args.connector else [c for c, spec in exp.connectors.items() if spec.modeled]
        n = args.n if args.n is not None else exp.zoo.judge_subsample_per_connector
        total = 0.0
        for connector in connectors:
            exp.connector(connector)
            total += print_estimate(connector, n, _list(args.models), _list(args.variants), views=args.views)["total_usd"]
            print(flush=True)
        if len(connectors) > 1:
            print(f"All {len(connectors)} connectors: ${total:.2f} worst case.", flush=True)
        return 0

    variants = list(judge.VARIANTS) if args.variant == "all" else _list(args.variant)
    if args.command == "show":
        for variant in variants:
            shown = save_example(args.connector, args.split, args.seed, variant, args.out, final_run=args.final_run)
            print(json.dumps(shown, indent=1), flush=True)
        return 0

    if args.seeds is not None:
        seeds = list(args.seeds)
    else:
        seeds = subsample(args.connector, args.split, args.n, seed=args.sample_seed, allocation=args.allocation,
                          final_run=args.final_run, classes=_list(args.classes))
    code = 0
    for variant in variants:
        summary = run(args.connector, args.split, seeds, args.model, variant, session=args.session,
                      max_states=max(args.n, 1), approved=args.approved, final_run=args.final_run)
        if summary["refused"] or summary["stopped"]:
            code = 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())
