"""The frame log and the intervention export (avail/flywheel/export.py, avail/flywheel/intervention.py).

Every run here is synthetic and lives in tmp_path: a ledger written with LedgerWriter and frame logs
written with write_frames, at 20 Hz, with takeovers marked by a HandoffLog.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from avail.flywheel import export as fx
from avail.flywheel.intervention import as_demo, as_label, as_metric, demo_window, is_takeover
from avail.fork import HandoffLog
from avail.ledger import LedgerWriter
from avail.schemas import Claim, ClaimContext, Cost, Intervention, Predicate, Verdict

DT = 0.05                                                   # the frame log's rate: 20 Hz


def frame_rows(n: int, handoff: int | None = None, handback: int | None = None, step: str = "insert") -> list[dict]:
    """``n`` frames of 1 cycle, tagged through a HandoffLog the way a cell would tag them."""
    log, rows = HandoffLog(), []
    for i in range(n):
        if i == handoff:
            log.begin(i, i * DT)
        rows.append({"frame_index": i, "t": round(i * DT, 6), "item": i // 100, "step": step, "tag": log.tag(i),
                     **{f"state_j{j}": 0.1 * j + 0.001 * i for j in range(6)}, "state_grip": 0.5,
                     **{f"action_j{j}": 0.1 * j + 0.001 * (i + 1) for j in range(6)}, "action_grip": 0.0,
                     "qpos": [0.0, float(i)]})                                    # a column that rides along
        if i == handback:
            log.end(i, i * DT)
    return rows


def iv(**changes) -> Intervention:
    fields = {
        "intervention_id": "iv_1", "run_id": "run_x", "cycle_id": "cycle_1", "item": 0, "step": "insert",
        "claim_id": "claim_1", "verdict_id": "verdict_1", "question": "takeover", "trigger": "reject after 2 attempts",
        "answer": "taken over", "truth_label": True, "asked_ts": 100.0, "answered_ts": 104.0, "seconds_waiting": 4.0,
        "seconds_handling": 3.0, "handoff_ts": 104.0, "handback_ts": 107.0, "handoff_frame": 100,
        "handback_frame": 160, "frames_path": "frames/cycle_1.parquet", "outcome": "taken_over_done",
        "features": {"probe.tug/disp_mm": 0.4, "passive.click/clicked": 0.0},
    }
    fields.update(changes)
    return Intervention(**fields)


def make_run(root) -> dict[str, Intervention]:
    """A run folder: 3 cycles, 5 interventions and 1 line that does not read back as one."""
    run = root / "run_x"
    fx.write_frames(run, "cycle_1", frame_rows(200, handoff=100, handback=160))
    fx.write_frames(run, "cycle_2", frame_rows(100, handoff=10, handback=30))
    made = {
        "takeover": iv(),                                                         # predicate from its claim
        "confirm_no": iv(intervention_id="iv_2", item=1, step="release", predicate_name="released_stable",
                         claim_id=None, verdict_id=None, question="confirm", answer="no", truth_label=False,
                         outcome="retried", handoff_ts=None, handback_ts=None, handoff_frame=None,
                         handback_frame=None, frames_path=None, trigger="abstain at stage 1"),
        "choose": iv(intervention_id="iv_3", question="choose", answer="retry", truth_label=None, outcome="retried",
                     handoff_ts=None, handback_ts=None, handoff_frame=None, handback_frame=None, frames_path=None),
        "early_takeover": iv(intervention_id="iv_4", cycle_id="cycle_2", item=2, frames_path=None, truth_label=None,
                             handoff_frame=10, handback_frame=30, outcome="taken_over_failed"),
        "no_frames": iv(intervention_id="iv_5", cycle_id="cycle_3", frames_path=None),
    }
    claim = Claim(claim_id="claim_1", run_id="run_x", cycle_id="cycle_1", step_id="insert",
                  predicate=Predicate(name="seated", target={"item": 0, "port": 4}), raised_by="skill",
                  context=ClaimContext(task="cable_loop", stage="insert", attempt=1))
    verdict = Verdict(verdict_id="verdict_1", claim_id="claim_1", decision="reject", stage_reached=3,
                      witness_version="witness-v3", cost_total=Cost.zero())
    with LedgerWriter("run_x", folder=run) as ledger:
        ledger.append("cycle_start", {"cycle_id": "cycle_1", "items": [0, 1]})
        ledger.append("claim", claim)
        ledger.append("verdict", verdict)
        for record in made.values():
            ledger.append("intervention", record)
        ledger.append("intervention", {"intervention_id": "iv_bad", "question": "takeover"})
    return made


# ------------------------------------------------------------------ the export
def test_export_writes_a_lerobot_style_tree(tmp_path):
    made = make_run(tmp_path)
    out = tmp_path / "export"
    result = fx.export_interventions(tmp_path / "run_x", out)
    # iv_5 lost its frame log: no episode, and its label still counts
    assert (result["interventions"], result["takeovers"], result["episodes"], result["labels"]) == (5, 3, 2, 3)
    assert result["frames"] == 101 + 31 and result["fps"] == 20 and result["sim_or_real"] == "sim"
    assert result["run_id"] == "run_x"
    assert result["skipped"] == [{"intervention_id": "iv_5", "cycle_id": "cycle_3",
                                  "reason": "the frame log frames/cycle_3.parquet does not exist"}]
    assert any("an intervention event does not read back" in problem for problem in result["problems"])
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()) == [
        "data/chunk-000/episode_000000.parquet", "data/chunk-000/episode_000001.parquet", "labels.jsonl",
        "meta/episodes.jsonl", "meta/info.json", "meta/tasks.jsonl"]

    first = pd.read_parquet(out / "data/chunk-000/episode_000000.parquet")
    assert list(first.columns) == ["observation.state", "action", "timestamp", "frame_index", "episode_index", "index",
                                   "task_index", "tag", "is_handoff", "is_handback", "cycle_frame_index"]
    assert len(first) == 101                                   # 2 s before the handoff (40 frames) to the handback
    assert first["cycle_frame_index"].tolist() == list(range(60, 161))
    assert first["frame_index"].tolist() == list(range(101)) and first["index"].tolist() == list(range(101))
    assert set(first["episode_index"]) == {0} and set(first["task_index"]) == {0}
    assert first["timestamp"].iloc[0] == 0.0 and first["timestamp"].iloc[-1] == pytest.approx(5.0, abs=1e-5)
    assert first["tag"].tolist() == ["autonomous"] * 40 + ["assisted"] * 61       # every frame tagged
    assert np.flatnonzero(first["is_handoff"]).tolist() == [40] and np.flatnonzero(first["is_handback"]).tolist() == [100]
    state, action = first["observation.state"].iloc[40], first["action"].iloc[40]
    assert len(state) == len(action) == 7 and state.dtype == np.float32
    assert state.tolist() == pytest.approx([0.1 * j + 0.1 for j in range(6)] + [0.5], abs=1e-6)    # frame 100
    assert action.tolist() == pytest.approx([0.1 * j + 0.101 for j in range(6)] + [0.0], abs=1e-6)

    second = pd.read_parquet(out / "data/chunk-000/episode_000001.parquet")
    assert second["cycle_frame_index"].tolist() == list(range(0, 31))            # the log starts 0.5 s before it
    assert second["index"].tolist() == list(range(101, 132)) and set(second["episode_index"]) == {1}

    episodes = [json.loads(line) for line in (out / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [e["intervention_id"] for e in episodes] == ["iv_1", "iv_4"]
    assert episodes[0] == {
        "episode_index": 0, "tasks": ["insert"], "length": 101, "intervention_id": "iv_1", "cycle_id": "cycle_1",
        "item": "0", "step": "insert", "question": "takeover", "trigger": "reject after 2 attempts",
        "outcome": "taken_over_done", "handoff_frame": 100, "handback_frame": 160, "handoff_episode_frame": 40,
        "handback_episode_frame": 100, "assisted_frames": 61, "data_path": "data/chunk-000/episode_000000.parquet"}
    assert [json.loads(line) for line in (out / "meta/tasks.jsonl").read_text(encoding="utf-8").splitlines()] == [
        {"task_index": 0, "task": "insert"}]

    info = json.loads((out / "meta/info.json").read_text(encoding="utf-8"))
    assert (info["fps"], info["total_episodes"], info["total_frames"], info["total_tasks"]) == (20, 2, 132, 1)
    assert info["sim_or_real"] == "sim" and info["source_run"] == "run_x" and info["splits"] == {"train": "0:2"}
    assert info["data_path"] == "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    assert info["features"]["observation.state"] == {"dtype": "float32", "shape": [7],
                                                     "names": ["j0", "j1", "j2", "j3", "j4", "j5", "grip"]}
    assert set(info["features"]) == set(first.columns)
    assert info["counts"] == {"interventions": 5, "takeovers": 3, "episodes": 2, "frames": 132, "labels": 3,
                              "skipped": 1}

    labels = [json.loads(line) for line in (out / "labels.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(row["intervention_id"], row["predicate"], row["label"]) for row in labels] == [
        ("iv_1", "seated", True), ("iv_2", "released_stable", False), ("iv_5", "seated", True)]
    assert labels[0]["witness_version"] == "witness-v3" and labels[1]["witness_version"] is None
    assert all(row["source"] == "assist" and row["sampling_probability"] == 1.0 for row in labels)
    assert labels[0]["features"] == made["takeover"].features


def test_an_export_never_mixes_with_an_older_one(tmp_path):
    make_run(tmp_path)
    out = tmp_path / "export"
    fx.export_interventions(tmp_path / "run_x", out)
    with pytest.raises(FileExistsError, match="export into a new folder"):
        fx.export_interventions(tmp_path / "run_x", out)
    (tmp_path / "empty").mkdir()
    assert fx.export_interventions(tmp_path / "run_x", tmp_path / "empty", fps=30)["fps"] == 30
    with pytest.raises(ValueError, match="say sim or real"):
        fx.export_interventions(tmp_path / "run_x", tmp_path / "e2", sim_or_real="both")
    with pytest.raises(ValueError, match="frames per second"):
        fx.export_interventions(tmp_path / "run_x", tmp_path / "e3", fps=0)
    real = fx.export_interventions(tmp_path / "run_x", tmp_path / "e4", sim_or_real="real")
    assert json.loads((tmp_path / "e4" / "meta" / "info.json").read_text(encoding="utf-8"))["sim_or_real"] == "real"
    assert real["sim_or_real"] == "real"


def test_a_shift_folder_exports_its_runs_into_1_tree(tmp_path):
    """A shift folder holds 1 run per block of cycles. Its export is 1 tree, its episodes numbered on."""
    shift = tmp_path / "shift"
    make_run(shift)                                                               # shift/run_x: 2 episodes
    run_y = shift / "run_y"
    fx.write_frames(run_y, "cycle_9", frame_rows(80, handoff=40, handback=60))
    with LedgerWriter("run_y", folder=run_y) as ledger:
        ledger.append("intervention", iv(intervention_id="iv_9", run_id="run_y", cycle_id="cycle_9",
                                         frames_path="frames/cycle_9.parquet", handoff_frame=40, handback_frame=60))
    result = fx.export_interventions(shift, tmp_path / "out")
    assert (result["run_id"], result["takeovers"], result["episodes"], result["labels"]) == ("shift", 4, 3, 4)
    episodes = [json.loads(line) for line in
                (tmp_path / "out" / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [e["episode_index"] for e in episodes] == [0, 1, 2] and episodes[2]["intervention_id"] == "iv_9"
    info = json.loads((tmp_path / "out" / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["source_runs"] == ["run_x", "run_y"] and info["total_episodes"] == 3
    last = pd.read_parquet(tmp_path / "out" / "data/chunk-000/episode_000002.parquet")
    assert last["index"].iloc[0] == 132 and set(last["episode_index"]) == {2}
    assert last["tag"].iloc[-1] == "assisted" and last["is_handback"].iloc[-1]
    assert any(p.startswith("run_x: an intervention event does not read back") for p in result["problems"])
    with pytest.raises(FileNotFoundError, match="holds no ledger"):
        fx.export_interventions(tmp_path / "nothing_here", tmp_path / "out2")


def test_an_export_of_a_run_with_no_interventions_is_empty_and_says_so(tmp_path):
    run = tmp_path / "quiet"
    with LedgerWriter("quiet", folder=run) as ledger:
        ledger.append("cycle_start", {"cycle_id": "cycle_1", "items": [0]})
    result = fx.export_interventions(run, tmp_path / "out")
    assert (result["interventions"], result["episodes"], result["labels"], result["fps"]) == (0, 0, 0, None)
    info = json.loads((tmp_path / "out" / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["total_episodes"] == 0 and info["total_chunks"] == 0 and info["splits"] == {}
    assert (tmp_path / "out" / "labels.jsonl").read_text(encoding="utf-8") == ""


def test_a_frame_log_outside_the_run_folder_is_never_read(tmp_path):
    run = tmp_path / "run_x"
    fx.write_frames(tmp_path / "elsewhere", "cycle_1", frame_rows(200, handoff=100, handback=160))
    with LedgerWriter("run_x", folder=run) as ledger:
        ledger.append("intervention", iv(frames_path="../elsewhere/frames/cycle_1.parquet"))
    result = fx.export_interventions(run, tmp_path / "out")
    assert result["episodes"] == 0
    assert result["skipped"][0]["reason"] == "the frame log ../elsewhere/frames/cycle_1.parquet is outside the run folder"


# ------------------------------------------------------------------ 1 record, 3 uses
def test_as_label_is_the_calibration_row_of_a_labeled_answer():
    row = as_label(iv(predicate_name="seated"))
    assert row["predicate"] == "seated" and row["label"] is True and row["source"] == "assist"
    assert row["sampling_probability"] == 1.0 and row["features"] == {"probe.tug/disp_mm": 0.4, "passive.click/clicked": 0.0}
    assert (row["cycle_id"], row["item"], row["step"], row["ts"]) == ("cycle_1", "0", "insert", 104.0)
    assert as_label(iv(), predicate="latched", witness_version="v7")["predicate"] == "latched"
    assert as_label(iv())["predicate"] is None                              # the record does not say
    assert as_label(iv(truth_label=None, outcome="taken_over_failed")) is None
    assert as_label(iv().model_dump(mode="json"))["intervention_id"] == "iv_1"   # a payload dict works too
    with pytest.raises(TypeError, match="pass an Intervention record"):
        as_label("iv_1")


def test_as_metric_is_what_the_shift_card_counts():
    metric = as_metric(iv())
    assert metric == {"intervention_id": "iv_1", "cycle_id": "cycle_1", "item": "0", "step": "insert",
                      "question": "takeover", "trigger": "reject after 2 attempts", "outcome": "taken_over_done",
                      "seconds_waiting": 4.0, "seconds_handling": 3.0, "takeover": True, "labeled": True}
    confirm = iv(question="confirm", answer="yes", outcome="signed_off", handoff_ts=None, handback_ts=None,
                 handoff_frame=None, handback_frame=None)
    assert as_metric(confirm)["takeover"] is False and is_takeover(confirm) is False
    assert is_takeover(iv(question="choose", answer="taken over"))          # the outcome and the frames say so


def test_as_demo_is_the_takeover_with_its_2_ends_marked():
    frames = pd.DataFrame(frame_rows(200, handoff=100, handback=160))
    demo = as_demo(iv(), frames)
    assert demo["frame_index"].tolist() == list(range(60, 161))
    assert demo.loc[demo["is_handoff"], "frame_index"].tolist() == [100]
    assert demo.loc[demo["is_handback"], "frame_index"].tolist() == [160]
    assert "qpos" in demo.columns and len(frames) == 200 and "is_handoff" not in frames.columns   # input untouched
    assert as_demo(iv(), frames, pre_roll_s=0.0)["frame_index"].tolist() == list(range(100, 161))
    shuffled = frames.sample(frac=1.0, random_state=3)
    assert as_demo(iv(), shuffled)["frame_index"].tolist() == list(range(60, 161))      # in frame order
    confirm = iv(question="confirm", answer="yes", outcome="signed_off", handoff_ts=None, handback_ts=None,
                 handoff_frame=None, handback_frame=None)
    assert demo_window(confirm, frames) == (None, "not a takeover")
    assert demo_window(iv(handback_ts=None, handback_frame=None, outcome="taken_over_failed"), frames)[1] == (
        "the takeover has no handoff frame and handback frame")
    assert demo_window(iv(handoff_frame=300, handback_frame=310), frames)[1] == "the handoff frame 300 is not in the frame log"
    assert demo_window(iv(handback_frame=250), frames)[1] == "the handback frame 250 is not in the frame log"
    assert demo_window(iv(), frames.drop(columns=["t"]))[1] == "the frame log has no t column"


# ------------------------------------------------------------------ the frame log
def test_write_frames_puts_1_file_per_cycle_in_the_run_folder(tmp_path):
    path = fx.write_frames(tmp_path / "run", "cycle_0003", frame_rows(30))
    assert path == tmp_path / "run" / "frames" / "cycle_0003.parquet"
    assert fx.frames_relpath("cycle_0003") == "frames/cycle_0003.parquet"
    assert [p.name for p in path.parent.iterdir()] == ["cycle_0003.parquet"]          # no temporary file left
    back = fx.read_frames(path)
    assert list(back.columns[: len(fx.FRAME_COLUMNS)]) == list(fx.FRAME_COLUMNS) and "qpos" in back.columns
    assert len(back) == 30 and back["tag"].tolist() == ["autonomous"] * 30
    assert fx.read_frames(fx.write_frames(tmp_path / "run", "cycle_empty", [])).empty
    assert fx.write_frames(tmp_path / "run", "cycle_df", pd.DataFrame(frame_rows(3))).is_file()
    with pytest.raises(ValueError, match="is not usable as a file name"):
        fx.frames_relpath("../cycle")


@pytest.mark.parametrize("change, words", [
    (lambda rows: [{k: v for k, v in row.items() if k != "action_grip"} for row in rows], "no column action_grip"),
    (lambda rows: [{**row, "tag": "teleop"} for row in rows], "tag holds 'teleop'"),
    (lambda rows: rows + rows[-1:], "frame_index repeats or goes back"),
    (lambda rows: rows[::-1], "frame_index repeats or goes back"),
    (lambda rows: [{**row, "frame_index": row["frame_index"] - 5} for row in rows], "is not a whole number of 0 or more"),
    (lambda rows: [{**row, "state_j2": float("nan")} for row in rows], "state_j2 holds something"),
    (lambda rows: [{**row, "t": "soon"} for row in rows], "t holds something"),
])
def test_write_frames_refuses_a_frame_log_it_cannot_vouch_for(tmp_path, change, words):
    with pytest.raises(ValueError, match="Fix: ") as caught:
        fx.write_frames(tmp_path / "run", "cycle_1", change(frame_rows(10)))
    assert words in str(caught.value)
    assert not (tmp_path / "run" / "frames" / "cycle_1.parquet").exists()
