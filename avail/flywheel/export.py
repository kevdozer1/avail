"""The frame log of a cycle, and the export of a run's interventions as a LeRobot-style dataset.

Skill notes and regression cases (docs/06) come in S6.

The frame log. The cell writes 1 parquet file per cycle, ``AVAIL_DATA/runs/<run_id>/frames/<cycle_id>.parquet``,
with 1 row per logged control frame:

- ``frame_index``: the frame's index in the cycle's log, from 0
- ``t``: the sim clock, in seconds
- ``item`` and ``step``: what the robot was doing
- ``tag``: ``autonomous``, or ``assisted`` while a person had control (``avail.fork.handoff.HandoffLog``)
- ``state_j0`` to ``state_j5`` and ``state_grip``: the joint readings in rad, and the gripper from 0 closed to 1 open
- ``action_j0`` to ``action_j5`` and ``action_grip``: the joint targets the controller sent, and the gripper target

Other columns ride along untouched, for example what a replay needs. :func:`write_frames` checks a
frame log and writes it.

The export. :func:`export_interventions` reads a run's ledger and writes::

    <out_dir>/data/chunk-000/episode_000000.parquet   1 per takeover, every frame from 2 s before the handoff
                                                      to the handback, each tagged autonomous or assisted
    <out_dir>/meta/episodes.jsonl                     1 line per episode: its intervention, cycle, item, step,
                                                      question, outcome, handoff and handback frames, length
    <out_dir>/meta/tasks.jsonl                        the task text of the episodes: their step
    <out_dir>/meta/info.json                          fps, features, counts and sim_or_real
    <out_dir>/labels.jsonl                            the calibration label of every intervention that has one

An episode's rows carry LeRobot's columns: ``observation.state`` and ``action`` as 7 floats (6
joints, then the gripper), ``timestamp`` from the episode's first frame, ``frame_index`` within the
episode, ``episode_index``, ``index`` across the export and ``task_index``. They also carry AVAIL's:
``tag``, ``is_handoff``, ``is_handback`` and ``cycle_frame_index``, the frame's index in the cycle's
log. Sim and real are never pooled: a run is 1 of them and the export says which.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import ValidationError

from ..schemas import Intervention
from ..schemas.kinds import FRAME_TAGS
from .intervention import PRE_ROLL_S, as_label, demo_window, is_takeover

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "ACTION_COLUMNS", "CHUNK_SIZE", "DATA_PATH", "FRAME_COLUMNS", "FRAMES_DIR", "STATE_COLUMNS", "check_frames",
    "export_interventions", "frames_relpath", "read_frames", "write_frames",
]

JOINTS = 6
STATE_COLUMNS: tuple[str, ...] = (*(f"state_j{i}" for i in range(JOINTS)), "state_grip")
ACTION_COLUMNS: tuple[str, ...] = (*(f"action_j{i}" for i in range(JOINTS)), "action_grip")
FRAME_COLUMNS: tuple[str, ...] = ("frame_index", "t", "item", "step", "tag", *STATE_COLUMNS, *ACTION_COLUMNS)
"""The columns every frame log has, in this order. Others may follow."""

FRAMES_DIR = "frames"
CHUNK_SIZE = 1000
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
CODEBASE_VERSION = "v2.1"
SIM_OR_REAL = ("sim", "real")
_VECTOR_NAMES = [*(f"j{i}" for i in range(JOINTS)), "grip"]
_CYCLE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


# ------------------------------------------------------------------ the frame log
def frames_relpath(cycle_id: str) -> str:
    """Where a cycle's frame log sits in its run folder, ``frames/<cycle_id>.parquet``: an Intervention's frames_path."""
    if not isinstance(cycle_id, str) or not _CYCLE_ID.fullmatch(cycle_id) or cycle_id.endswith("."):
        raise ValueError(f"cycle id {cycle_id!r} is not usable as a file name. Fix: use letters, digits, '.', '_' "
                         "or '-', starting with a letter or a digit")
    return f"{FRAMES_DIR}/{cycle_id}.parquet"


def check_frames(frame: pd.DataFrame) -> pd.DataFrame:
    """``frame`` when it is a frame log, else ValueError naming what is wrong."""
    import pandas as pd

    missing = [column for column in FRAME_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"the frame log has no column {', '.join(missing)}. Fix: log every frame with "
                         f"{', '.join(FRAME_COLUMNS)}")
    if frame.empty:
        return frame
    index = frame["frame_index"]
    if not pd.api.types.is_integer_dtype(index) or bool((index < 0).any()):
        raise ValueError("frame_index holds something that is not a whole number of 0 or more. Fix: count the "
                         "frames of the cycle from 0")
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("frame_index repeats or goes back. Fix: log each frame once, in order")
    for column in ("t", *STATE_COLUMNS, *ACTION_COLUMNS):
        values = frame[column]
        if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values) or \
                not bool(np.isfinite(values.to_numpy(dtype=float)).all()):
            raise ValueError(f"{column} holds something that is not a finite number. Fix: log the reading as a float")
    tags = sorted(set(map(str, frame["tag"].unique())) - set(FRAME_TAGS))
    if tags:
        raise ValueError(f"tag holds {', '.join(map(repr, tags))}. Fix: tag each frame {' or '.join(FRAME_TAGS)}")
    return frame


def write_frames(run_dir: str | os.PathLike[str], cycle_id: str,
                 rows: pd.DataFrame | Iterable[Mapping[str, Any]]) -> Path:
    """Check a cycle's frame log and write it to ``<run_dir>/frames/<cycle_id>.parquet``. Returns the path.

    ``rows`` is a DataFrame or 1 dict per frame. The write goes through a temporary file, so a crash
    leaves the old file or the new one, never half of one.
    """
    import pandas as pd

    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    if frame.empty and len(frame.columns) == 0:
        frame = pd.DataFrame({column: [] for column in FRAME_COLUMNS})
    check_frames(frame)
    path = Path(run_dir) / frames_relpath(cycle_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return path


def read_frames(path: str | os.PathLike[str]) -> pd.DataFrame:
    """A frame log, read back."""
    import pandas as pd

    return pd.read_parquet(Path(path))


# ------------------------------------------------------------------ the export
def _frame_log(run_dir: Path, iv: Intervention, cache: dict[Path, tuple[Any, str]]) -> tuple[Any, str]:
    """(the cycle's frame log, "") or (None, why it cannot be read)."""
    import pandas as pd

    try:
        rel = iv.frames_path or frames_relpath(iv.cycle_id)
    except ValueError as exc:
        return None, str(exc)
    path = run_dir / rel
    if not path.resolve().is_relative_to(run_dir.resolve()):
        return None, f"the frame log {rel} is outside the run folder"
    if path not in cache:
        if not path.is_file():
            cache[path] = (None, f"the frame log {rel} does not exist")
        else:
            try:
                cache[path] = (pd.read_parquet(path), "")
            except (OSError, ValueError) as exc:
                cache[path] = (None, f"the frame log {rel} cannot be read ({type(exc).__name__})")
    return cache[path]


def _features_meta() -> dict[str, Any]:
    vector = {"shape": [len(_VECTOR_NAMES)], "names": list(_VECTOR_NAMES)}
    scalar = {"shape": [1], "names": None}
    return {
        "observation.state": {"dtype": "float32", **vector},
        "action": {"dtype": "float32", **vector},
        "timestamp": {"dtype": "float32", **scalar},
        "frame_index": {"dtype": "int64", **scalar},
        "episode_index": {"dtype": "int64", **scalar},
        "index": {"dtype": "int64", **scalar},
        "task_index": {"dtype": "int64", **scalar},
        "tag": {"dtype": "string", **scalar},
        "is_handoff": {"dtype": "bool", **scalar},
        "is_handback": {"dtype": "bool", **scalar},
        "cycle_frame_index": {"dtype": "int64", **scalar},
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def run_dirs_of(path: str | os.PathLike[str]) -> list[Path]:
    """The runs at ``path``: ``path`` itself when it holds a ledger, else each folder just below it that holds
    one. A shift folder (``avail.shift.runner``) holds 1 run per block of cycles."""
    path = Path(path)
    if (path / "ledger.jsonl").is_file():
        return [path]
    return sorted(ledger.parent for ledger in path.glob("*/ledger.jsonl"))


def export_interventions(run_dir: str | os.PathLike[str], out_dir: str | os.PathLike[str], *,
                         fps: float | None = None, sim_or_real: str = "sim", robot_type: str | None = None,
                         pre_roll_s: float = PRE_ROLL_S) -> dict[str, Any]:
    """Write a run's interventions as a LeRobot-style tree in ``out_dir`` and return what was written.

    ``run_dir`` is the run's folder (it holds ``ledger.jsonl`` and ``frames/``), or a shift's folder,
    whose runs then go into 1 tree with their episodes numbered on. ``out_dir`` must be new or
    empty, so an export never mixes with an older one. ``fps`` is measured from the frame logs when
    it is not given. A takeover whose frames cannot be found is skipped and listed with its reason
    under ``skipped``. Ledger lines that do not read back are listed under ``problems``.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from ..ledger.reader import load_ledger

    if sim_or_real not in SIM_OR_REAL:
        raise ValueError(f"sim_or_real is {sim_or_real!r}. Fix: say sim or real. The 2 are never pooled")
    if fps is not None and (isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps)
                            or fps <= 0):
        raise ValueError(f"fps is {fps!r}. Fix: give the frame log's rate in frames per second, or leave it out")
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise FileExistsError(f"{out_dir} is not an empty folder. Fix: export into a new folder, so 2 exports "
                              "never mix")
    runs = run_dirs_of(run_dir)
    if not runs:
        raise FileNotFoundError(f"{run_dir} holds no ledger.jsonl, and no folder just below it does. Fix: give a "
                                "run's folder or a shift's")

    problems: list[str] = []
    run_ids: list[str] = []
    interventions: list[tuple[Intervention, Path]] = []
    claim_predicate: dict[str, str] = {}
    verdict_version: dict[str, str] = {}
    for rdir in runs:
        tables = load_ledger(rdir)
        prefix = "" if len(runs) == 1 else f"{rdir.name}: "         # a shift's problems name their run
        problems += [f"{prefix}{p}" for p in tables.problems]
        run_ids.append(str(tables.events["run_id"].iloc[0]) if len(tables.events) else rdir.name)
        for kind, payload in zip(tables.events["kind"], tables.events["payload"]):
            if not isinstance(payload, Mapping):
                continue
            if kind == "intervention":
                try:
                    interventions.append((Intervention.model_validate(payload), rdir))
                except ValidationError as exc:
                    first = exc.errors(include_url=False)[0]
                    where = ".".join(str(step) for step in first.get("loc", ())) or "the record"
                    problems.append(f"{prefix}an intervention event does not read back ({where}: {first['msg']})")
            elif kind == "claim":
                predicate = payload.get("predicate")
                name = predicate.get("name") if isinstance(predicate, Mapping) else None
                if isinstance(payload.get("claim_id"), str) and isinstance(name, str):
                    claim_predicate[payload["claim_id"]] = name
            elif kind == "verdict":
                if isinstance(payload.get("verdict_id"), str) and isinstance(payload.get("witness_version"), str):
                    verdict_version[payload["verdict_id"]] = payload["witness_version"]
    run_id = run_ids[0] if len(runs) == 1 else run_dir.name

    labels: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    tasks: dict[str, int] = {}
    cache: dict[Path, tuple[Any, str]] = {}
    steps: list[np.ndarray] = []
    total = 0
    takeovers = 0
    for iv, rdir in interventions:
        label = as_label(iv, predicate=iv.predicate_name or claim_predicate.get(iv.claim_id or ""),
                         witness_version=verdict_version.get(iv.verdict_id or ""))
        if label is not None:
            labels.append(label)
        if not is_takeover(iv):
            continue
        takeovers += 1
        frames, why = _frame_log(rdir, iv, cache)
        demo = None
        if frames is not None:
            demo, why = demo_window(iv, frames, pre_roll_s)
        if demo is not None:
            absent = [column for column in (*STATE_COLUMNS, *ACTION_COLUMNS, "tag") if column not in demo.columns]
            if absent:
                demo, why = None, f"the frame log has no column {', '.join(absent)}"
        if demo is not None:
            state = demo[list(STATE_COLUMNS)].to_numpy(dtype=np.float32)
            action = demo[list(ACTION_COLUMNS)].to_numpy(dtype=np.float32)
            t = demo["t"].to_numpy(dtype=float)
            if not (np.isfinite(state).all() and np.isfinite(action).all() and np.isfinite(t).all()):
                demo, why = None, "the frame log holds a reading that is not a finite number"
        if demo is None:
            skipped.append({"intervention_id": iv.intervention_id, "cycle_id": iv.cycle_id, "reason": why})
            continue
        episode_index, n = len(episodes), len(demo)
        task_index = tasks.setdefault(iv.step, len(tasks))
        table = pa.table({
            "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32())),
            "action": pa.array(action.tolist(), type=pa.list_(pa.float32())),
            "timestamp": pa.array((t - t[0]).astype(np.float32)),
            "frame_index": pa.array(np.arange(n, dtype=np.int64)),
            "episode_index": pa.array(np.full(n, episode_index, dtype=np.int64)),
            "index": pa.array(np.arange(total, total + n, dtype=np.int64)),
            "task_index": pa.array(np.full(n, task_index, dtype=np.int64)),
            "tag": pa.array([str(tag) for tag in demo["tag"]], type=pa.string()),
            "is_handoff": pa.array(demo["is_handoff"].to_numpy(dtype=bool)),
            "is_handback": pa.array(demo["is_handback"].to_numpy(dtype=bool)),
            "cycle_frame_index": pa.array(demo["frame_index"].to_numpy(dtype=np.int64)),
        })
        rel = DATA_PATH.format(episode_chunk=episode_index // CHUNK_SIZE, episode_index=episode_index)
        (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out_dir / rel)
        is_handoff, is_handback = demo["is_handoff"].to_numpy(dtype=bool), demo["is_handback"].to_numpy(dtype=bool)
        episodes.append({
            "episode_index": episode_index, "tasks": [iv.step], "length": n,
            "intervention_id": iv.intervention_id, "cycle_id": iv.cycle_id, "item": iv.item, "step": iv.step,
            "question": iv.question, "trigger": iv.trigger, "outcome": iv.outcome,
            "handoff_frame": iv.handoff_frame, "handback_frame": iv.handback_frame,
            "handoff_episode_frame": int(np.flatnonzero(is_handoff)[0]),
            "handback_episode_frame": int(np.flatnonzero(is_handback)[0]),
            "assisted_frames": int(sum(str(tag) == "assisted" for tag in demo["tag"])),
            "data_path": rel,
        })
        total += n
        gaps = np.diff(t)
        steps.append(gaps[gaps > 0])

    if fps is None:                    # the mean gap: a log on a fixed grid of control ticks alternates 2 gap lengths
        gaps = np.concatenate(steps) if steps else np.array([])
        fps = round(1.0 / float(np.mean(gaps))) if gaps.size else None
    counts = {"interventions": len(interventions), "takeovers": takeovers, "episodes": len(episodes),
              "frames": total, "labels": len(labels), "skipped": len(skipped)}
    info = {
        "codebase_version": CODEBASE_VERSION, "robot_type": robot_type, "fps": fps,
        "total_episodes": len(episodes), "total_frames": total, "total_tasks": len(tasks), "total_videos": 0,
        "total_chunks": (len(episodes) - 1) // CHUNK_SIZE + 1 if episodes else 0, "chunks_size": CHUNK_SIZE,
        "splits": {"train": f"0:{len(episodes)}"} if episodes else {}, "data_path": DATA_PATH, "video_path": None,
        "features": _features_meta(), "sim_or_real": sim_or_real, "source_run": run_id,
        "pre_roll_s": float(pre_roll_s), "counts": counts,
    }
    if len(runs) > 1:
        info["source_runs"] = run_ids
    files = {
        "info": out_dir / "meta" / "info.json",
        "episodes": _write_jsonl(out_dir / "meta" / "episodes.jsonl", episodes),
        "tasks": _write_jsonl(out_dir / "meta" / "tasks.jsonl",
                              [{"task_index": index, "task": task} for task, index in tasks.items()]),
        "labels": _write_jsonl(out_dir / "labels.jsonl", labels),
    }
    files["info"].write_text(json.dumps(info, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                             encoding="utf-8", newline="\n")
    return {"out_dir": str(out_dir), "run_id": run_id, "sim_or_real": sim_or_real, "fps": fps, **counts,
            "skipped": skipped, "problems": problems, "files": {name: str(path) for name, path in files.items()}}
