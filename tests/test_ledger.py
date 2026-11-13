"""The ledger: append-only JSONL with one locked writer per run, and typed pandas tables.

Ported from loopkit's tests/test_traces.py where they apply (crash safety, resume, locking, odd
payloads), plus the AVAIL contract: seq is gapless and continues across a reopen, a second writer
is refused by name, appends from several threads stay gapless, Ctrl+C and a killed process leave
every completed append readable, an unknown kind is refused with the valid kinds listed, and
load_ledger returns one typed table per event kind.

Every test runs with AVAIL_DATA pointed at pytest's tmp_path. Nothing reaches the real data drive.
"""

from __future__ import annotations

import collections
import dataclasses
import datetime as dt
import enum
import gc
import io
import itertools
import json
import os
import random
import subprocess
import sys
import threading
import time
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from avail import paths
from avail.ledger import (
    BASE_COLUMNS, FIELDS, LEDGER_FILE, LOCK_FILE, MAX_SEQ, TABLE_NAMES, LedgerLockedError, LedgerTables,
    LedgerWarning, LedgerWriter, SeqBreak, TornLine, check_ledger_run_id, iter_events, ledger_path, list_runs,
    load_ledger, parse_line, to_jsonable,
)
from avail.ledger import jsonsafe as jsonsafe_module
from avail.ledger import lines as lines_module
from avail.ledger import writer as writer_module
from avail.schemas.kinds import EVENT_KINDS

FFFD = "\N{REPLACEMENT CHARACTER}"  # what a lone surrogate becomes


@pytest.fixture(autouse=True)
def data_root(tmp_path, monkeypatch):
    """AVAIL_DATA is a temp folder for every test in this file."""
    root = tmp_path / "data"
    monkeypatch.setenv("AVAIL_DATA", str(root))
    assert paths.data_root() == root  # the override is in force before a test body runs
    yield root
    assert paths.data_root() == root  # and no test undid it on the way
    left_open = list(writer_module._OPEN_RUNS.values())
    for claim in left_open:  # give them back, so that one test's leak cannot fail the tests after it
        claim.release()
    assert not left_open, f"runs left open by this test: {[claim.run_id for claim in left_open]}"


# --------------------------------------------------------------------------- helpers
def _events(source, **kwargs):
    return list(iter_events(source, **kwargs))


def _seqs(source):
    return [event["seq"] for event in iter_events(source)]


def _line(seq, run_id="f", kind="tool_call", payload=None, ts=1.0):
    """One ledger line as another tool would write it (plain json.dumps)."""
    body = {"n": seq} if payload is None else payload
    return json.dumps({"run_id": run_id, "seq": seq, "ts": ts, "kind": kind, "payload": body})


def _raw_ledger(folder, *lines):
    """A run folder holding ``lines`` as its ledger, as another tool or a repair script left it."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / LEDGER_FILE).write_bytes(b"".join(line.encode("utf-8") + b"\n" for line in lines))
    return folder


def _child(script, *args, **kwargs):
    """``script`` in a fresh interpreter from the repo root. Every caller reaps it with _finish or kill."""
    return subprocess.Popen([sys.executable, "-c", script, *map(str, args)], cwd=paths.REPO_ROOT, text=True,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)


def _finish(child, timeout, input=None):
    """(stdout, stderr) of a child. A child that hangs (a deadlock) is killed and fails the test."""
    try:
        return child.communicate(input, timeout=timeout)
    except subprocess.TimeoutExpired:
        child.kill()
        out, err = child.communicate()
        pytest.fail(f"child still running after {timeout} s (deadlocked?)\n{out}\n{err}")


def _with_deadline(fn, seconds=30):
    """``fn()`` on a daemon thread, so a deadlock fails the test instead of hanging the suite."""
    outcome = {}

    def target():
        try:
            outcome["value"] = fn()
        except BaseException as exc:  # handed back to the test thread
            outcome["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        pytest.fail(f"still running after {seconds} s: deadlocked")
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")


def _retry_permission(fn, *args, attempts=10, step=0.02):
    """``fn(*args)``, retried while Windows reports the file in use for a moment (a scanner, an indexer)."""
    for attempt in range(attempts):
        try:
            return fn(*args)
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(step * (attempt + 1))


def _fill(run_id):
    """A small run with the shapes the record types have: 1 claim, 2 evidence items, 2 verdicts, 2 cycle ends."""
    cost = {"seconds": 1.2, "usd": 0.0, "moved_robot": True}
    with LedgerWriter(run_id) as w:
        w.append("claim", {
            "claim_id": "c1", "run_id": run_id, "cycle_id": 0, "step_id": "insert",
            "predicate": {"name": "seated", "target": {"plug": "p1", "port": 7}}, "raised_by": "skill", "ts": 100.0,
            "context": {"task": "plug", "stage": 1, "attempt": 0},
        }, ts=100.5)
        w.append("evidence", {
            "evidence_id": "e1", "claim_id": "c1", "source_id": "probe.push_seat", "kind": "active_probe",
            "features": {"peak_n": 5.5, "travel_mm": 0.4}, "score": 0.91, "cost": cost, "artifacts": [],
            "side_effect": "none", "ts": 101.0,
        }, ts=101.5)
        w.append("evidence", {
            "evidence_id": "e2", "claim_id": "c1", "source_id": "judge.vlm", "kind": "judge", "features": {},
            "score": None, "cost": {"seconds": 3.0, "usd": 0.002, "moved_robot": False},
            "artifacts": ["a.jpg", "b.jpg"], "side_effect": "none", "ts": 102.0,
        }, ts=102.5)
        w.append("verdict", {
            "verdict_id": "v1", "claim_id": "c1", "decision": "accept", "score": 0.91, "stage_reached": 1,
            "witness_version": 3, "evidence_ids": ["e1", "e2"], "cost_total": {"seconds": 4.2, "usd": 0.002},
        }, ts=103.0)
        w.append("verdict", {
            "verdict_id": "v2", "claim_id": "c2", "decision": "abstain", "score": 0.5, "stage_reached": 2,
            "witness_version": 3, "evidence_ids": [], "cost_total": {"seconds": 0.0, "usd": 0.0},
        }, ts=104.0)
        w.append("cycle_end", {
            "cycle_id": 0, "declared_success": True, "true_success": False, "wall_s": 12,
            "predicates": [{"name": "seated", "verdict_id": "v1"}],
        }, ts=105.0)
        w.append("cycle_end", {"cycle_id": 1, "declared_success": False, "wall_s": 9.5, "predicates": []}, ts=106.0)
    return w


# --------------------------------------------------------------------------- writing
def test_every_kind_round_trips():
    with LedgerWriter("all_kinds") as w:
        seqs = [w.append(kind, {"n": i, "note": f"a {kind}"}) for i, kind in enumerate(EVENT_KINDS)]
        last = w.append("audit", {"truth_label": True}, ts=1_700_000_000.5)
    assert seqs == list(range(len(EVENT_KINDS))) and last == len(EVENT_KINDS)
    torn = []
    events = _events("all_kinds", torn=torn)
    assert torn == []
    assert [e["kind"] for e in events] == [*EVENT_KINDS, "audit"]
    assert [e["seq"] for e in events] == list(range(len(EVENT_KINDS) + 1))
    assert {e["run_id"] for e in events} == {"all_kinds"}
    assert [e["payload"] for e in events[:-1]] == [{"n": i, "note": f"a {kind}"} for i, kind in enumerate(EVENT_KINDS)]
    assert events[-1]["ts"] == 1_700_000_000.5 and events[-1]["payload"] == {"truth_label": True}
    assert all(tuple(e) == FIELDS == ("run_id", "seq", "ts", "kind", "payload") for e in events)


def test_seq_starts_at_0_and_lines_are_compact_ascii_json():
    with LedgerWriter("seq") as w:
        assert w.next_seq == 0 and not w.resumed and w.torn_tail is None
        seqs = [w.append("tool_call", {"step": i}) for i in range(5)]
        assert w.next_seq == 5
    assert seqs == [0, 1, 2, 3, 4]
    raw = w.path.read_bytes()
    lines = raw.split(b"\n")
    assert raw.isascii() and lines[-1] == b"" and len(lines) == 6  # one line per event, newline terminated
    for i, line in enumerate(lines[:-1]):
        obj = json.loads(line)
        assert list(obj) == ["run_id", "seq", "ts", "kind", "payload"]  # the 5 keys, in this order
        assert (obj["run_id"], obj["seq"], obj["kind"], obj["payload"]) == ("seq", i, "tool_call", {"step": i})
        assert type(obj["seq"]) is int and type(obj["ts"]) is float
        assert line.decode("ascii") == json.dumps(obj, separators=(",", ":"))
    stamps = [e["ts"] for e in iter_events("seq")]
    assert stamps == sorted(stamps) and abs(stamps[0] - time.time()) < 60  # stamped now, in seq order


def test_default_location_is_avail_data_runs(data_root):
    with LedgerWriter("unit_1") as w:
        w.append("claim", {"claim_id": "c1"})
    assert w.run_id == "unit_1" and w.dir == data_root / "runs" / "unit_1"
    assert w.path == data_root / "runs" / "unit_1" / "ledger.jsonl"
    assert w.path == paths.run_dir("unit_1", create=False) / LEDGER_FILE
    assert w.lock_path == w.dir / LOCK_FILE
    # the lock file stays after close: deleting it would let 2 writers lock 2 different files
    assert sorted(p.name for p in w.dir.iterdir()) == [".ledger.lock", "ledger.jsonl"]
    assert [e["payload"] for e in iter_events("unit_1")] == [{"claim_id": "c1"}]


def test_a_writer_can_be_given_its_folder(tmp_path, data_root):
    folder = tmp_path / "elsewhere" / "zoo_block_3"
    with LedgerWriter("zoo_3", folder=folder) as w:
        w.append("evidence", {"evidence_id": "e1"})
    assert w.dir == folder and w.path == folder / "ledger.jsonl" and w.path.is_file()
    assert not data_root.exists()  # nothing went to AVAIL_DATA
    for source in (folder, str(folder), folder / "ledger.jsonl", str(folder / "ledger.jsonl")):
        assert [(e["run_id"], e["payload"]) for e in iter_events(source)] == [("zoo_3", {"evidence_id": "e1"})]


def test_seq_continues_across_reopens():
    with LedgerWriter("resume") as w:
        assert [w.append("claim", {"n": "a"}), w.append("claim", {"n": "b"})] == [0, 1]
    with LedgerWriter("resume") as w:
        assert w.resumed and w.next_seq == 2 and w.torn_tail is None
        assert w.append("verdict", {"n": "c"}) == 2
    with LedgerWriter("resume") as w:
        assert w.resumed and w.next_seq == 3
        assert w.append("fork", {"n": "d"}) == 3
    assert [(e["seq"], e["payload"]["n"]) for e in iter_events("resume")] == [(0, "a"), (1, "b"), (2, "c"), (3, "d")]
    tables = load_ledger("resume")
    assert tables.events["seq"].tolist() == [0, 1, 2, 3] and tables.problems == []


def test_resume_reads_a_long_ledger_from_its_end(tmp_path):
    folder = tmp_path / "long"
    with LedgerWriter("long", folder=folder) as w:
        for i in range(3000):  # about 200 KB, several 64 KiB blocks
            w.append("tool_call", {"i": i, "pad": "x" * 30})
    with LedgerWriter("long", folder=folder) as w:
        assert w.next_seq == 3000
        assert w.append("evidence", {"image": "y" * 300_000}) == 3000  # one line longer than 4 blocks
    with LedgerWriter("long", folder=folder) as w:
        assert w.next_seq == 3001
    with open(folder / LEDGER_FILE, "ab") as fh:
        fh.write(b'{"run_id":"long","seq":3001,')
    with pytest.warns(LedgerWarning, match=r"line 3002, "):
        with LedgerWriter("long", folder=folder) as w:
            assert w.next_seq == 3001 and w.torn_tail.number == 3002 and w.torn_tail.length == 28
    assert _seqs(folder) == list(range(3001))


def test_the_tail_scan_reads_only_the_end_of_a_big_file():
    line = _line(0).encode() + b"\n"
    data = line * 100_000  # about 6 MB

    class Counting(io.BytesIO):
        bytes_read = 0

        def read(self, size=-1):
            chunk = super().read(size)
            Counting.bytes_read += len(chunk)
            return chunk

    scan = lines_module._lines_from_end(Counting(data), len(data))
    assert next(scan) == (len(data), b"", False)  # the empty segment after the last newline
    offset, last, terminated = next(scan)
    assert (offset, last + b"\n", terminated) == (len(data) - len(line), line, True)
    assert 0 < Counting.bytes_read <= 2 * (1 << 16)


def test_lines_from_end_matches_a_forward_split_at_every_block_size():
    rng = random.Random(7)
    for trial in range(300):
        lengths = [rng.choice([0, 0, 1, 2, 5, 9, 17]) for _ in range(rng.randint(1, 8))]
        segments = [bytes(rng.choice(b"ab{}:,") for _ in range(length)) for length in lengths]
        data = b"\n".join(segments)
        expected, offset = [], 0
        for i, segment in enumerate(segments):
            expected.append((offset, segment, i < len(segments) - 1))
            offset += len(segment) + 1
        for block in (1, 2, 3, 5, 64):
            got = list(lines_module._lines_from_end(io.BytesIO(data), len(data), block=block))
            assert got == expected[::-1], (trial, block, data)


def test_unknown_kind_is_refused_with_the_valid_kinds_listed():
    with LedgerWriter("kinds") as w:
        for bad in ("claims", "Verdict", "", " claim", None, 7, b"claim"):
            with pytest.raises(ValueError) as err:
                w.append(bad, {"x": 1})
            message = str(err.value)
            assert message == f"unknown ledger event kind {bad!r}. Valid kinds: {', '.join(EVENT_KINDS)}"
            assert all(kind in message for kind in ("claim", "evidence", "verdict", "fork", "assist", "audit",
                                                    "tool_call", "model_call", "calibration_update", "drift_alarm",
                                                    "cycle_end"))
        assert w.next_seq == 0 and w.path.read_bytes() == b""  # nothing written, no seq used
        assert w.append("claim", {"x": 1}) == 0
    assert _seqs("kinds") == [0]


def test_invalid_payloads_and_timestamps_raise_value_error_and_write_nothing():
    with LedgerWriter("bad") as w:
        with pytest.raises(ValueError, match=r"claim payload for run 'bad' must be a dict or a record.*got list"):
            w.append("claim", ["not", "a", "dict"])
        with pytest.raises(ValueError, match="must be a dict or a record.*got str"):
            w.append("claim", "the plug is seated")
        with pytest.raises(ValueError, match="claim payload for run 'bad' is None"):
            w.append("claim", None)
        for bad_ts in (float("nan"), float("inf"), "1700000000", True, dt.datetime(2026, 11, 11), 10**400):
            with pytest.raises(ValueError, match="ts must be"):
                w.append("claim", {"x": 1}, ts=bad_ts)
        with pytest.raises(ValueError, match=r"^ts must be a finite number, got <int>$"):
            w.append("claim", {"x": 1}, ts=10**5000)  # too many digits even to print
        loop: dict = {}
        loop["self"] = loop
        with pytest.raises(ValueError, match="claim payload for run 'bad' is nested too deeply or contains itself"):
            w.append("claim", {"x": 1, "loop": loop})
        assert w.next_seq == 0 and w.path.read_bytes() == b""
        assert w.append("claim", {"ok": True}, ts=5) == 0  # a whole number of seconds is a fine ts
    assert [(e["seq"], e["ts"], e["payload"]) for e in iter_events("bad")] == [(0, 5.0, {"ok": True})]


def test_closed_writer_refuses_appends():
    w = LedgerWriter("closed")
    w.append("claim", {"n": 1})
    assert not w.closed
    w.close()
    w.close()  # safe to call twice
    assert w.closed
    with pytest.raises(ValueError, match="LedgerWriter for run 'closed' is closed"):
        w.append("claim", {"n": 2})
    with pytest.raises(ValueError, match="LedgerWriter for run 'closed' is closed"):
        w.sync()
    assert _seqs("closed") == [0]


def test_records_with_model_dump_are_written_in_json_mode():
    from pydantic import BaseModel

    class Cost(BaseModel):
        seconds: float
        usd: float
        moved_robot: bool

    class Item(BaseModel):
        evidence_id: str
        when: dt.datetime
        cost: Cost

    class Duck:  # not pydantic: any object with model_dump(mode="json") is a record
        def __init__(self):
            self.modes = []

        def model_dump(self, mode="python"):
            self.modes.append(mode)
            return {"shape": "duck", "mode": mode}

    item = Item(evidence_id="e1", when=dt.datetime(2026, 11, 11, 12, 0, tzinfo=dt.timezone.utc),
                cost=Cost(seconds=1.5, usd=0.0, moved_robot=True))
    duck = Duck()
    with LedgerWriter("records") as w:
        w.append("evidence", item)
        w.append("tool_call", duck)
        w.append("claim", {"nested": item})  # a record inside a dict
    events = _events("records")
    as_json = {"evidence_id": "e1", "when": "2026-11-11T12:00:00Z",
               "cost": {"seconds": 1.5, "usd": 0.0, "moved_robot": True}}
    assert item.model_dump(mode="json") == as_json  # JSON mode writes UTC as Z, Python mode would not
    assert events[0]["payload"] == as_json
    assert duck.modes == ["json"] and events[1]["payload"] == {"shape": "duck", "mode": "json"}
    assert events[2]["payload"] == {"nested": as_json}


def test_records_with_numpy_fields_serialize_and_a_broken_record_is_refused():
    from pydantic import BaseModel, ConfigDict, Field, field_serializer

    class Detection(BaseModel):  # what a perception source might return
        model_config = ConfigDict(arbitrary_types_allowed=True)
        label: str
        corners_px: np.ndarray
        score: np.float32
        debug: str = Field(default="internal", exclude=True)

    class Broken(BaseModel):
        value: int = 1

        @field_serializer("value")
        def _fail(self, value):
            raise RuntimeError("serializer failed")

    det = Detection(label="port", corners_px=np.array([[10, 20], [30, 40]], dtype=np.int32), score=np.float32(0.5))
    with LedgerWriter("models") as w:
        w.append("evidence", det)
        w.append("evidence", {"det": det, "depth_mm": np.float32(8.5), "gap": np.array([0.1, 0.2])})
        with pytest.raises(ValueError,
                           match=r"evidence payload for run 'models' is not JSON serializable.*serializer failed"):
            w.append("evidence", {"bad": Broken()})
        with pytest.raises(ValueError, match=r"evidence payload for run 'models' is not JSON serializable"):
            w.append("evidence", Broken())
        assert w.next_seq == 2
    expected = {"label": "port", "corners_px": [[10, 20], [30, 40]], "score": 0.5}  # the excluded field stays out
    assert [e["payload"] for e in iter_events("models")] == [
        expected, {"det": expected, "depth_mm": 8.5, "gap": [0.1, 0.2]}]


def test_what_the_writer_writes_is_what_the_schemas_call_a_ledger_event():
    from avail.schemas import Cost, LedgerEvent  # imported here only: the ledger itself imports no record type

    cost = Cost(seconds=1.5, usd=0.002, moved_robot=True)
    with LedgerWriter("typed") as w:
        for kind in EVENT_KINDS:
            if kind == "cycle_end":  # the 1 kind whose payload the schema checks
                w.append(kind, {"declared_success": True, "predicates": [], "wall_s": 12.5})
            else:
                w.append(kind, {"cost": cost, "note": kind})
        w.append("tool_call", cost)  # a record goes in whole, through model_dump(mode="json")
    events = _events("typed")
    assert len(events) == len(EVENT_KINDS) + 1
    for event in events:
        checked = LedgerEvent.model_validate(event)  # the 5 fields, by the schema's own rules
        assert checked.model_dump(mode="json") == event
    assert events[-1]["payload"] == cost.model_dump(mode="json") == {"seconds": 1.5, "usd": 0.002, "moved_robot": True}
    assert Cost.model_validate(events[-1]["payload"]) == cost  # and comes back as the same record
    assert events[0]["payload"]["cost"] == events[-1]["payload"]


def test_numpy_paths_datetimes_nan_and_inf_are_written_as_plain_json():
    @dataclasses.dataclass
    class Grip:
        width_m: float
        closed: bool

    payload = {
        "count": np.int64(4), "score": np.float32(0.5), "flag": np.bool_(True), "name": np.str_("find_port"),
        "pos": np.array([0.1, 0.2, 0.3]), "grid": np.arange(6, dtype=np.uint8).reshape(2, 3),
        "gaps": np.array([1.0, np.nan]), "missing": float("nan"), "far": np.float64("inf"), "near": float("-inf"),
        "image": Path("images") / "a.png", "pair": (1, 2), "tags": {"b", "a"},
        "grip": Grip(np.float64(0.04), np.bool_(False)), "day": np.datetime64("2026-11-11"),
        "when": dt.datetime(2026, 11, 11, 9, 30, 15), "date": dt.date(2026, 11, 11),
        "took": dt.timedelta(seconds=1.5), "error": RuntimeError("port not found"), "raw": b"\xff\x00",
        7: "int key",
    }
    expected = {
        "count": 4, "score": 0.5, "flag": True, "name": "find_port", "pos": [0.1, 0.2, 0.3],
        "grid": [[0, 1, 2], [3, 4, 5]], "gaps": [1.0, None], "missing": None, "far": None, "near": None,
        "image": str(Path("images") / "a.png"), "pair": [1, 2], "tags": ["a", "b"],
        "grip": {"width_m": 0.04, "closed": False}, "day": "2026-11-11", "when": "2026-11-11T09:30:15",
        "date": "2026-11-11", "took": 1.5, "error": "RuntimeError: port not found", "raw": "base64:/wA=",
        "7": "int key",
    }
    with LedgerWriter("np") as w:
        w.append("evidence", payload, ts=np.float64(12.5))
    line = w.path.read_text(encoding="utf-8").splitlines()[0]
    assert "NaN" not in line and "Infinity" not in line  # strict JSON
    assert json.loads(line)["payload"] == expected
    event = _events("np")[0]
    assert event["payload"] == expected and event["ts"] == 12.5
    assert type(event["payload"]["count"]) is int and type(event["payload"]["flag"]) is bool
    assert to_jsonable(payload) == expected


def test_other_odd_values_and_keys_become_plain_json():
    class Stage(enum.Enum):
        PUSH = "push"

    class Level(enum.IntEnum):
        HIGH = 3

    class Opaque:
        def __str__(self):
            return "an opaque thing"

    class ArrayLike:
        def tolist(self):
            return [1, np.float32(2.5)]

    class Record:
        def model_dump(self, mode="python"):
            return {"ok": True}

    assert to_jsonable({None: "none key", True: "bool key", 1.5: "float key", Stage.PUSH: Stage.PUSH,
                        np.int64(3): [Stage.PUSH, Level.HIGH], np.timedelta64(5, "s"): "a duration"}) == {
        "null": "none key", "true": "bool key", "1.5": "float key", "push": "push", "3": ["push", 3],
        "5 seconds": "a duration"}
    assert to_jsonable([range(3), collections.deque([1, 2]), b"plain", complex(1, 2), dt.time(9, 30),
                        np.timedelta64(5, "s"), np.array(["a", "b"]), np.array([np.datetime64("2026-11-11")]),
                        Opaque(), ArrayLike(), frozenset({1, "a"}), np.complex64(1 + 2j)]) == [
        [0, 1, 2], [1, 2], "plain", "(1+2j)", "09:30:00", "5 seconds", ["a", "b"], ["2026-11-11"],
        "an opaque thing", [1, 2.5], ["a", 1], "(1+2j)"]
    assert to_jsonable(types.MappingProxyType({"a": (1, 2)})) == {"a": [1, 2]}
    assert to_jsonable(Record()) == {"ok": True}
    assert to_jsonable(Record) == str(Record)  # the class itself is no record, though it has model_dump


def test_kinds_and_payloads_in_other_spellings_are_written_plainly():
    class Kind(str, enum.Enum):
        CLAIM = "claim"

    @dataclasses.dataclass
    class Note:
        text: str
        weight: float

    with LedgerWriter("spellings") as w:
        w.append(Kind.CLAIM, {"n": 0})  # a str enum member
        w.append(np.str_("verdict"), {"n": 1})  # a numpy string
        w.append("assist", Note("guide the arm", np.float64(0.5)))  # a dataclass
        w.append("audit", types.MappingProxyType({"truth_label": True}))  # a read-only mapping
        with pytest.raises(ValueError, match=r"^claim payload for run 'spellings' is None"):
            w.append(Kind.CLAIM, None)  # the message names the plain kind too
    events = _events("spellings")
    assert [e["kind"] for e in events] == ["claim", "verdict", "assist", "audit"]
    assert all(type(e["kind"]) is str for e in events)
    assert [e["payload"] for e in events] == [{"n": 0}, {"n": 1}, {"text": "guide the arm", "weight": 0.5},
                                              {"truth_label": True}]
    assert b'"kind":"claim"' in w.path.read_bytes() and b"Kind" not in w.path.read_bytes()


def test_non_ascii_text_round_trips_on_ascii_lines():
    odd = [chr(c) for c in (0xEF, 0x2013, 0xB0, 0x2028, 0x85, 0x4E2D, 0x1F600)]  # incl. U+2028, NEL and an emoji
    text = "yaw +2.6 " + " | ".join(odd) + " end"
    with LedgerWriter("text") as w:
        w.append("assist", {"answer": text})
    raw = w.path.read_bytes()
    assert raw.isascii() and raw.count(b"\n") == 1  # no character a line splitter could take for a line break
    assert _events("text")[0]["payload"]["answer"] == text
    assert load_ledger("text").assists["answer"].iloc[0] == text


def test_lone_surrogates_are_written_as_replacement_characters():
    cut_emoji = "model said: done \ud83d"  # a reply cut inside a surrogate pair
    undecodable = b"stderr: \x80".decode("utf-8", "surrogateescape")
    with LedgerWriter("surrogates") as w:
        w.append("model_call", {"usd": 0.25})
        w.append("model_call", {"usd": 0.40, "error": "ValueError: " + cut_emoji})
        w.append("tool_call", {"error": undecodable,
                               "outputs": {"key \udfff": "v", "split pair": chr(0xD83D) + chr(0xDE00)}})
    torn = []
    events = _events("surrogates", torn=torn)
    assert torn == [] and [e["seq"] for e in events] == [0, 1, 2]
    assert events[1]["payload"]["error"] == "ValueError: model said: done " + FFFD
    assert events[2]["payload"]["error"] == "stderr: " + FFFD
    assert events[2]["payload"]["outputs"] == {"key " + FFFD: "v", "split pair": "\U0001f600"}
    assert load_ledger("surrogates").model_calls["usd"].sum() == pytest.approx(0.65)
    assert to_jsonable({"a\ud83d": ["b\udc80", np.array(["c\udfff"])]}) == {"a" + FFFD: ["b" + FFFD, ["c" + FFFD]]}


def test_appends_from_several_threads_are_serialized_and_gapless():
    n_threads, per_thread = 8, 50
    total = n_threads * per_thread
    returned = [[] for _ in range(n_threads)]
    with LedgerWriter("threads") as w:
        start = threading.Barrier(n_threads)

        def work(i):
            start.wait()
            for j in range(per_thread):
                returned[i].append(w.append("tool_call", {"thread": i, "j": j}))

        threads = [threading.Thread(target=work, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert w.next_seq == total
    torn = []
    events = _events("threads", torn=torn)
    assert torn == []
    assert [e["seq"] for e in events] == list(range(total))  # file order is seq order: no gap, no repeat
    assert sorted(seq for seqs in returned for seq in seqs) == list(range(total))  # every append got a seq of its own
    for i, seqs in enumerate(returned):
        assert seqs == sorted(seqs)  # a thread's own appends keep their order
        # the seq append returned is the seq of the line that holds that thread's payload
        written = [(events[s]["payload"]["thread"], events[s]["payload"]["j"]) for s in seqs]
        assert written == [(i, j) for j in range(per_thread)]
    stamps = [e["ts"] for e in events]
    assert stamps == sorted(stamps)  # stamped under the lock, so default timestamps follow seq


BAD_IDS = ["", "a/b", "a\\b", "x:y", ".", "..", " padded", "dot.", "t2_dump.jsonl", "X.JSONL", "cut\ud83d",
           "nul", "NUL", "Nul.txt", "con", "aux.log", "prn", "COM1", "com9.tar.gz", "LPT1", "lpt9", 7, None]


@pytest.mark.parametrize("bad", BAD_IDS, ids=[repr(b).encode("ascii", "backslashreplace").decode() for b in BAD_IDS])
def test_bad_run_ids_are_rejected_before_anything_is_created(tmp_path, bad):
    with pytest.raises(ValueError):
        LedgerWriter(bad)
    with pytest.raises(ValueError):
        LedgerWriter(bad, folder=tmp_path / "explicit")
    with pytest.raises(ValueError):
        check_ledger_run_id(bad)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("good", ["console", "nul_run", "com10", "lpt", "auxiliary", "con-1", "run.json", "t2_m1_r0",
                                  "e3.arm-1_2026"])
def test_names_close_to_device_names_are_fine(good):
    assert check_ledger_run_id(good) == good
    with LedgerWriter(good) as w:
        w.append("claim", {"ok": True})
    assert [e["run_id"] for e in iter_events(good)] == [good]


@pytest.mark.skipif(os.name != "nt", reason="only Windows maps <folder>/nul to a device")
def test_a_folder_that_windows_maps_to_a_device_is_refused(tmp_path):
    with pytest.raises(ValueError, match="resolves to the Windows device"):
        LedgerWriter("fine_id", folder=tmp_path / "nul")
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- crash safety and resume
def test_a_torn_last_line_is_reported_and_never_merged_with_new_data():
    with LedgerWriter("crash") as w:
        for i in range(3):
            w.append("claim", {"before": i})
    good = w.path.read_bytes()
    partial = b'{"run_id":"crash","seq":3,"ts":1.0,"kind":"cla'
    with open(w.path, "ab") as fh:  # a crash mid-write leaves part of a line and no newline
        fh.write(partial)

    torn = []
    assert [e["seq"] for e in iter_events("crash", torn=torn)] == [0, 1, 2]
    assert len(torn) == 1 and isinstance(torn[0], TornLine)
    assert (torn[0].number, torn[0].offset, torn[0].length, torn[0].terminated) == (4, len(good), len(partial), False)
    assert "not valid JSON" in torn[0].reason and torn[0].text == partial.decode()

    with pytest.warns(LedgerWarning, match=r"run 'crash' ended in a torn line \(line 4, 46 bytes") as caught:
        w = LedgerWriter("crash")
    try:
        assert len(caught) == 1 and str(w.path) in str(caught[0].message)
        assert w.resumed and w.next_seq == 3
        assert w.torn_tail == torn[0]  # the same report the readers give
        assert w.path.read_bytes() == good + partial + b"\n"  # opening only ends the fragment
        assert w.append("claim", {"after": "crash"}) == 3  # the fragment's seq goes to the next event
        assert w.append("claim", {"and": "one more"}) == 4
    finally:
        w.close()
    data = w.path.read_bytes()
    assert data.startswith(good + partial + b"\n{") and data.endswith(b"}\n")  # nothing rewritten, nothing merged

    tables = load_ledger("crash")
    assert tables.events["seq"].tolist() == [0, 1, 2, 3, 4] and tables.seq_breaks == ()
    assert tables.claims["after"].tolist()[3] == "crash"
    assert [(t.number, t.terminated) for t in tables.torn_lines] == [(4, True)]  # now a line of its own, still skipped
    assert tables.torn_tail is None
    with warnings.catch_warnings():
        warnings.simplefilter("error", LedgerWarning)  # the fragment is ended: a later open has nothing to warn about
        with LedgerWriter("crash") as w:
            assert w.torn_tail is None and w.next_seq == 5


def test_a_complete_last_line_without_its_newline_is_kept():
    with LedgerWriter("no_newline") as w:
        w.append("claim", {"n": "a"})
        w.append("claim", {"n": "b"})
    w.path.write_bytes(w.path.read_bytes().rstrip(b"\n"))
    assert [e["payload"]["n"] for e in iter_events("no_newline")] == ["a", "b"]  # every byte of the event is there
    with warnings.catch_warnings():
        warnings.simplefilter("error", LedgerWarning)  # not torn: nothing to warn about
        with LedgerWriter("no_newline") as w:
            assert w.next_seq == 2 and w.torn_tail is None
            w.append("claim", {"n": "c"})
    torn = []
    assert [e["payload"]["n"] for e in iter_events("no_newline", torn=torn)] == ["a", "b", "c"] and torn == []
    assert w.path.read_bytes().count(b"\n") == 3


def test_a_failed_write_leaves_a_fragment_on_its_own_line_and_no_gap_in_seq(monkeypatch):
    real_write = writer_module._write_all
    calls = {"n": 0}

    def disk_full_on_second_write(fh, data):
        calls["n"] += 1
        if calls["n"] == 2:
            fh.write(data[: len(data) // 2])  # half a line reaches the file, then the disk fills
            raise OSError("disk full")
        real_write(fh, data)

    monkeypatch.setattr(writer_module, "_write_all", disk_full_on_second_write)
    with LedgerWriter("disk_full") as w:
        assert w.append("claim", {"n": "one"}) == 0
        with pytest.raises(OSError, match="disk full"):
            w.append("claim", {"n": "two"})
        assert w.path.read_bytes().endswith(b"\n")  # the fragment is ended before append hands back control
        assert "next_seq=1" in repr(w) and w.next_seq == 1  # the event never arrived, so its seq is still free
        assert w.append("claim", {"n": "three"}) == 1
    torn = []
    events = _events("disk_full", torn=torn)
    assert [(e["seq"], e["payload"]["n"]) for e in events] == [(0, "one"), (1, "three")]
    assert [(t.number, t.terminated) for t in torn] == [(2, True)]  # the half line, on a line of its own
    assert load_ledger("disk_full").seq_breaks == ()


def test_a_fragment_that_cannot_be_ended_yet_blocks_appends_until_it_is(monkeypatch):
    real_write = writer_module._write_all
    disk = {"full": False}

    def write(fh, data):
        if disk["full"]:
            if data != b"\n":
                fh.write(data[:10])  # part of the line arrives
            raise OSError("disk full")
        real_write(fh, data)

    monkeypatch.setattr(writer_module, "_write_all", write)
    with LedgerWriter("stuck") as w:
        w.append("claim", {"n": 0})
        disk["full"] = True
        with pytest.raises(OSError, match="disk full"):
            w.append("claim", {"n": 1})  # leaves a fragment, and its newline cannot be written either
        size = w.path.stat().st_size
        with pytest.raises(OSError, match="disk full"):
            w.append("claim", {"n": 2})  # refused before a byte is written: never merged into the fragment
        assert w.path.stat().st_size == size
        disk["full"] = False
        assert w.next_seq == 1
        assert w.append("claim", {"n": 3}) == 1
    torn = []
    assert [(e["seq"], e["payload"]["n"]) for e in iter_events("stuck", torn=torn)] == [(0, 0), (1, 3)]
    assert [(t.number, t.length, t.terminated) for t in torn] == [(2, 10, True)]

    with LedgerWriter("stuck_at_close") as w:
        w.append("claim", {"n": 0})
        disk["full"] = True
        with pytest.raises(OSError, match="disk full"):
            w.append("claim", {"n": 1})
        assert not w.path.read_bytes().endswith(b"\n")  # the fragment could not be ended
        disk["full"] = False
    assert w.path.read_bytes().endswith(b"\n")  # close() ended it
    with warnings.catch_warnings():
        warnings.simplefilter("error", LedgerWarning)  # so the next open finds no torn tail
        with LedgerWriter("stuck_at_close") as w:
            assert w.next_seq == 1


def test_an_interrupt_after_the_line_is_down_never_reuses_its_seq(monkeypatch):
    real_write = writer_module._write_all
    interrupt = {"armed": False}

    def ctrl_c_right_after_the_write(fh, data):
        real_write(fh, data)
        if interrupt["armed"]:
            interrupt["armed"] = False
            raise KeyboardInterrupt  # the line reached the file, append has not returned yet

    monkeypatch.setattr(writer_module, "_write_all", ctrl_c_right_after_the_write)
    with LedgerWriter("ctrl_c") as w:
        w.append("model_call", {"request_hash": "first"})
        interrupt["armed"] = True
        with pytest.raises(KeyboardInterrupt):
            w.append("model_call", {"request_hash": "interrupted"})
        assert w.next_seq == 2
        assert w.append("assist", {"answer": "stopped by operator"}) == 2
    torn = []
    events = _events("ctrl_c", torn=torn)
    assert [(e["seq"], e["kind"]) for e in events] == [(0, "model_call"), (1, "model_call"), (2, "assist")]
    assert events[1]["payload"]["request_hash"] == "interrupted" and torn == []


def test_an_interrupt_before_the_write_uses_no_seq_and_leaves_no_fragment(monkeypatch):
    real_write = writer_module._write_all
    interrupt = {"armed": False}

    def ctrl_c_before_the_write(fh, data):
        if interrupt["armed"]:
            interrupt["armed"] = False
            raise KeyboardInterrupt
        real_write(fh, data)

    monkeypatch.setattr(writer_module, "_write_all", ctrl_c_before_the_write)
    with LedgerWriter("early") as w:
        w.append("claim", {"n": "first"})
        before = w.path.read_bytes()
        interrupt["armed"] = True
        with pytest.raises(KeyboardInterrupt):
            w.append("claim", {"n": "interrupted"})
        assert w.next_seq == 1 and w.path.read_bytes() == before
        assert w.append("claim", {"n": "after"}) == 1
    torn = []
    assert [(e["seq"], e["payload"]["n"]) for e in iter_events("early", torn=torn)] == [(0, "first"), (1, "after")]
    assert torn == []


def test_a_write_cut_just_before_its_newline_is_completed_and_counts_once(monkeypatch):
    real_write = writer_module._write_all
    cut = {"armed": False}

    def all_but_the_newline(fh, data):
        if cut["armed"]:
            cut["armed"] = False
            real_write(fh, data[:-1])
            raise KeyboardInterrupt
        real_write(fh, data)

    monkeypatch.setattr(writer_module, "_write_all", all_but_the_newline)
    with LedgerWriter("one_short") as w:
        w.append("claim", {"n": 0})
        cut["armed"] = True
        with pytest.raises(KeyboardInterrupt):
            w.append("claim", {"n": 1})
        assert w.next_seq == 2  # every byte of the event is in the file, so it counts
        assert w.append("claim", {"n": 2}) == 2
    torn = []
    assert [(e["seq"], e["payload"]["n"]) for e in iter_events("one_short", torn=torn)] == [(0, 0), (1, 1), (2, 2)]
    assert torn == [] and w.path.read_bytes().count(b"\n") == 3


def test_a_second_fault_while_settling_is_repaired_at_the_next_look(monkeypatch):
    real_write, real_read = writer_module._write_all, writer_module._read_from
    state = {"interrupt": False, "read_fails": 0}

    def write(fh, data):
        real_write(fh, data)
        if state["interrupt"]:
            state["interrupt"] = False
            raise KeyboardInterrupt

    def read(path, start):
        if state["read_fails"]:
            state["read_fails"] -= 1
            raise OSError("file busy")
        return real_read(path, start)

    monkeypatch.setattr(writer_module, "_write_all", write)
    monkeypatch.setattr(writer_module, "_read_from", read)
    with LedgerWriter("double") as w:
        w.append("claim", {"n": 0})
        state.update(interrupt=True, read_fails=1)
        with pytest.raises(KeyboardInterrupt):
            w.append("claim", {"n": 1})  # the line is down, and the look at the file that follows fails once
        assert state["read_fails"] == 0  # that look was tried and failed
        assert w.next_seq == 2  # the next look settles it
        assert w.append("claim", {"n": 2}) == 2
    assert [(e["seq"], e["payload"]["n"]) for e in iter_events("double")] == [(0, 0), (1, 1), (2, 2)]


def test_keyboard_interrupt_inside_the_with_block_keeps_every_completed_append():
    n = 7
    with pytest.raises(KeyboardInterrupt):
        with LedgerWriter("interrupted") as w:
            for i in range(n):
                w.append("tool_call", {"i": i})
            raise KeyboardInterrupt
    assert w.closed  # the with block released the run on the way out
    tables = load_ledger("interrupted")
    assert len(tables.events) == n and tables.events["seq"].tolist() == list(range(n))
    assert tables.tool_calls["i"].tolist() == list(range(n))
    assert tables.problems == []
    with LedgerWriter("interrupted") as again:
        assert again.next_seq == n


_CTRL_C_LOOP = r"""
import _thread, random, sys, threading
from avail.ledger import LedgerWriter, iter_events
root, trials, broken, events = sys.argv[1], int(sys.argv[2]), 0, 0
random.seed(7)
for t in range(trials):
    folder = f"{root}/r{t}"
    w = LedgerWriter(f"r{t}", folder=folder)
    timer = threading.Timer(random.uniform(0.001, 0.004), _thread.interrupt_main)  # Ctrl+C, as the console sends it
    try:
        try:
            timer.start()
            while True:
                w.append("tool_call", {"tick": True})
        except KeyboardInterrupt:
            w.append("assist", {"answer": "stopped by operator"})
    except KeyboardInterrupt:
        pass
    finally:
        timer.cancel()
        w.close()
    seqs = [e["seq"] for e in iter_events(folder)]
    broken += seqs != list(range(len(seqs)))
    events += len(seqs)
print("runs with a gap or a repeat in seq:", broken, "of", trials, "events:", events, flush=True)
"""


def test_real_ctrl_c_during_append_never_skips_or_repeats_a_seq(tmp_path):
    out, err = _finish(_child(_CTRL_C_LOOP, tmp_path, 20), 120)
    assert "runs with a gap or a repeat in seq: 0 of 20" in out, out + err
    assert int(out.split("events:")[1]) > 20, out + err  # the loops did append before each Ctrl+C


_HOLD_RUN = r"""
import os, sys
from avail.ledger import LedgerWriter
w = LedgerWriter(sys.argv[2], folder=sys.argv[1])
for i in range(int(sys.argv[3])):
    w.append("tool_call", {"i": i, "from": "the child"})
print("ready", os.getpid(), flush=True)
if sys.stdin.readline().strip() == "close":
    w.close()
    print("closed", flush=True)
"""


def _hold_run(folder, run_id, appends=1):
    """A child process that opens the run, appends, and then holds it open. Returns (child, its pid)."""
    child = _child(_HOLD_RUN, folder, run_id, appends)
    words = child.stdout.readline().split()
    if words[:1] != ["ready"]:
        child.kill()
        raise AssertionError(f"child did not open the run: {child.communicate(timeout=60)[1]}")
    return child, int(words[1])


def _open_once_free(run_id, folder, timeout=20.0):
    """A LedgerWriter as soon as a dead holder's lock is gone (the OS drops it as the process ends)."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            return LedgerWriter(run_id, folder=folder)
        except LedgerLockedError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.05)


def test_a_killed_process_leaves_a_readable_ledger_with_every_completed_append(tmp_path):
    folder = tmp_path / "killed"
    child, _ = _hold_run(folder, "killed", appends=25)
    try:
        # read while the other process still holds the run: readers take no lock
        assert [e["payload"]["i"] for e in iter_events(folder)] == list(range(25))
    finally:
        child.kill()  # no close(), no fsync, no atexit handler
        _finish(child, 60)
    tables = load_ledger(folder)
    assert tables.tool_calls["i"].tolist() == list(range(25)) and tables.problems == []
    with _open_once_free("killed", folder) as w:  # the OS dropped the dead holder's lock
        assert w.next_seq == 25 and w.torn_tail is None


_APPEND_FOREVER = r"""
import sys
from avail.ledger import LedgerWriter
w = LedgerWriter(sys.argv[2], folder=sys.argv[1])
i = 0
while True:
    seq = w.append("tool_call", {"i": i, "pad": "x" * 200})
    sys.stdout.write(f"{seq}\n")
    sys.stdout.flush()
    i += 1
"""


def test_a_process_killed_in_the_middle_of_a_run_loses_no_append_it_reported(tmp_path):
    folder = tmp_path / "midrun"
    child = _child(_APPEND_FOREVER, folder, "midrun")
    reported = []
    try:
        for _ in range(300):  # let it get going
            line = child.stdout.readline()
            if not line.strip().isdigit():
                child.kill()
                pytest.fail(f"child stopped early: {line!r}\n{child.communicate(timeout=60)[1]}")
            reported.append(int(line))
        child.kill()  # in the middle of its loop
        out, _ = _finish(child, 60)
    finally:
        child.kill()
    reported += [int(text) for text in out.split("\n")[:-1] if text.strip().isdigit()]  # whole lines only
    assert reported == list(range(len(reported))) and len(reported) >= 300

    torn = []
    seqs = [e["seq"] for e in iter_events(folder, torn=torn)]
    assert seqs == list(range(len(seqs)))  # gapless and in order
    assert len(seqs) >= len(reported)  # every append that returned is in the file
    assert len(torn) <= 1 and all(not t.terminated for t in torn)  # at most the one write the kill cut short
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LedgerWarning)
        with _open_once_free("midrun", folder) as w:
            assert w.next_seq == len(seqs)
            assert w.append("cycle_end", {"declared_success": False}) == len(seqs)
    assert _seqs(folder) == list(range(len(seqs) + 1))


def test_resume_and_the_readers_agree_on_which_lines_are_events(tmp_path):
    lines = [
        _line(0),  # line 1: an event
        _line(1.0),  # seq as a float
        _line("2"),  # seq as text
        _line(True),  # seq as a boolean
        _line(1, kind="bogus"),  # a kind that is not in EVENT_KINDS
        json.dumps({"run_id": "f", "seq": 1, "ts": 1.0, "kind": "claim"}),  # no payload
        json.dumps({"run_id": "f", "seq": 1, "ts": 1.0, "kind": "claim", "payload": {}, "extra": 1}),
        json.dumps({"run_id": "f", "seq": 1, "ts": "noon", "kind": "claim", "payload": {}}),
        json.dumps({"run_id": "f", "seq": 1, "ts": 1.0, "kind": "claim", "payload": [1, 2]}),
        json.dumps({"run_id": "", "seq": 1, "ts": 1.0, "kind": "claim", "payload": {}}),
        json.dumps({"run_id": "f", "seq": -1, "ts": 1.0, "kind": "claim", "payload": {}}),
        "[1, 2, 3]",
        "not json at all",
        "",  # line 14: a blank line is no line at all
        "  \t ",  # line 15: nor is a line of spaces
        _line(1),  # line 16: an event
    ]
    folder = _raw_ledger(tmp_path / "f", *lines)
    torn = []
    assert [e["seq"] for e in iter_events(folder, torn=torn)] == [0, 1]
    assert [t.number for t in torn] == list(range(2, 14)) and all(t.terminated for t in torn)
    reasons = {t.number: t.reason for t in torn}
    for number in (2, 3, 4, 11):
        assert "seq must be a whole number from 0 to 2**63 - 1" in reasons[number]
    assert "unknown event kind 'bogus'" in reasons[5] and "Valid kinds: claim, evidence" in reasons[5]
    assert "missing payload" in reasons[6] and "unknown 'extra'" in reasons[7]
    assert "ts must be Unix time in seconds" in reasons[8] and "payload must be a JSON object" in reasons[9]
    assert "run_id must be a non-empty string" in reasons[10]
    assert "a JSON array where an object" in reasons[12] and "not valid JSON" in reasons[13]
    for bad in lines[1:13]:
        with pytest.raises(ValueError):
            parse_line(bad)
    with LedgerWriter("f", folder=folder) as w:  # the writer resumes where the readers stop
        assert w.next_seq == 2 and w.torn_tail is None
        assert w.append("claim", {"n": "appended"}) == 2
    assert _seqs(folder) == [0, 1, 2]


def test_resume_walks_back_over_bad_lines_to_the_last_complete_event(tmp_path):
    folder = _raw_ledger(tmp_path / "g", _line(0, run_id="g"), _line(1, run_id="g"), _line(7, run_id="g", kind="bogus"),
                         "{half a line", "[" * 100_000 + "]" * 100_000)
    with warnings.catch_warnings():
        warnings.simplefilter("error", LedgerWarning)  # every bad line has its newline: no torn tail to warn about
        with LedgerWriter("g", folder=folder) as w:
            assert w.next_seq == 2 and w.torn_tail is None  # the bogus line's seq 7 is no event's seq
            assert w.append("claim", {"n": 2}) == 2
    torn = []
    assert [e["seq"] for e in iter_events(folder, torn=torn)] == [0, 1, 2]
    assert [t.number for t in torn] == [3, 4, 5] and "nested too deeply" in torn[2].reason
    assert len(torn[2].text) == 120  # a long line is clipped in the report


def test_a_seq_beyond_int64_is_not_an_event_and_the_last_seq_is_never_passed(tmp_path):
    folder = _raw_ledger(tmp_path / "big", _line(0, run_id="big"), _line(10**20, run_id="big"))
    torn = []
    assert [e["seq"] for e in iter_events(folder, torn=torn)] == [0] and [t.number for t in torn] == [2]
    tables = load_ledger(folder)
    assert tables.events["seq"].tolist() == [0] and len(tables.torn_lines) == 1
    with LedgerWriter("big", folder=folder) as w:
        assert w.next_seq == 1

    last = _raw_ledger(tmp_path / "last", _line(MAX_SEQ, run_id="last"))
    with pytest.warns(LedgerWarning, match=rf"line 1 has seq {MAX_SEQ} where seq 0 belongs"):
        assert load_ledger(last).events["seq"].tolist() == [MAX_SEQ] and MAX_SEQ == 2**63 - 1
    with LedgerWriter("last", folder=last) as w:  # the writer never hands out a seq the readers would drop
        with pytest.raises(ValueError, match="run 'last' has no seq left"):
            w.append("claim", {"one": "too many"})
    assert _seqs(last) == [MAX_SEQ]


def test_a_ledger_of_another_run_is_never_appended_to(tmp_path):
    folder = tmp_path / "shared_folder"
    with LedgerWriter("run_a", folder=folder) as w:
        w.append("claim", {"n": 1})
    before = (folder / LEDGER_FILE).read_bytes()
    with pytest.raises(ValueError,
                       match=r"holds events of run 'run_a' and cannot take events of run 'run_b'") as refused:
        LedgerWriter("run_b", folder=folder)
    assert (folder / LEDGER_FILE).read_bytes() == before
    # The refused open gave the lock back by itself. The error is still held here, and its traceback
    # keeps the half-built writer alive, so nothing was left to a finalizer.
    with LedgerWriter("run_a", folder=folder) as w:
        assert w.next_seq == 1
    assert refused.value.__traceback__ is not None


def test_fsync_runs_at_sync_and_at_close_and_never_per_append(monkeypatch):
    real_fsync = os.fsync
    synced_sizes = []

    def counting_fsync(fd):
        synced_sizes.append(os.fstat(fd).st_size)
        return real_fsync(fd)

    monkeypatch.setattr(writer_module.os, "fsync", counting_fsync)
    w = LedgerWriter("policy")
    for i in range(5):
        w.append("tool_call", {"i": i})
        assert len(_events("policy")) == i + 1  # with the OS as soon as append returns: nothing sits in a buffer
    assert synced_sizes == []  # no fsync per append
    w.sync()
    assert synced_sizes == [w.path.stat().st_size]  # the ledger file itself, all 5 lines
    w.append("tool_call", {"i": 5})
    w.close()
    assert synced_sizes == [synced_sizes[0], w.path.stat().st_size] and synced_sizes[1] > synced_sizes[0]


def test_a_ledger_rewritten_with_a_byte_order_mark_and_crlf_still_loads(tmp_path):
    folder = tmp_path / "bom"
    folder.mkdir()
    body = "\r\n".join([_line(0, run_id="bom"), _line(1, run_id="bom")]) + "\r\n"  # what PowerShell 5.1 writes
    (folder / LEDGER_FILE).write_bytes(b"\xef\xbb\xbf" + body.encode())
    torn = []
    assert [e["seq"] for e in iter_events(folder, torn=torn)] == [0, 1] and torn == []
    with LedgerWriter("bom", folder=folder) as w:
        assert w.next_seq == 2

    single = tmp_path / "bom1"
    single.mkdir()
    (single / LEDGER_FILE).write_bytes(b"\xef\xbb\xbf" + _line(0, run_id="bom1").encode() + b"\r\n")
    with LedgerWriter("bom1", folder=single) as w:  # the tail scan reaches the mark at byte 0
        assert w.next_seq == 1


# --------------------------------------------------------------------------- one writer per run
def test_a_second_writer_in_this_process_is_refused_and_changes_nothing(monkeypatch):
    first = LedgerWriter("shared")
    try:
        first.append("claim", {"n": "a"})
        before = first.path.read_bytes()
        with pytest.raises(LedgerLockedError, match=r"run 'shared' already has an open LedgerWriter in this process"):
            LedgerWriter("shared")
        monkeypatch.chdir(first.dir.parent)
        with pytest.raises(LedgerLockedError, match="'shared'"):  # the same folder reached through a relative path
            LedgerWriter("shared", folder="shared")
        assert first.path.read_bytes() == before
        assert first.append("claim", {"n": "b"}) == 1  # the open writer is untouched
    finally:
        first.close()
    assert (first.dir / LOCK_FILE).is_file()
    with LedgerWriter("shared") as again:  # close() released the run
        assert again.next_seq == 2


def test_the_os_lock_alone_refuses_a_second_writer_the_registry_does_not_see(monkeypatch):
    first = LedgerWriter("aliased")
    try:
        # 2 spellings of one folder that this process cannot tell are the same (a junction, a short name)
        # get past the in-process registry. The OS lock still refuses the second writer.
        with monkeypatch.context() as patch:
            patch.setattr(writer_module, "_OPEN_RUNS", {})
            with pytest.raises(LedgerLockedError,
                               match=r"run 'aliased' is already open in another LedgerWriter in this process"):
                LedgerWriter("aliased")
        assert first.append("claim", {"n": 1}) == 0
    finally:
        first.close()
    with LedgerWriter("aliased") as again:
        assert again.next_seq == 1


def test_a_writer_dropped_without_close_frees_its_run():
    w = LedgerWriter("dropped")
    w.append("claim", {"n": "a"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)  # the unclosed ledger file says so, as it should
        del w
        gc.collect()
    with LedgerWriter("dropped") as again:
        assert again.next_seq == 1


def test_a_writer_in_another_process_holds_the_run_until_it_closes_or_dies(tmp_path):
    folder = tmp_path / "unit_7"
    child, pid = _hold_run(folder, "unit_7")
    try:
        with pytest.raises(LedgerLockedError,
                           match=rf"run 'unit_7' is already open in another LedgerWriter \(process {pid}\)"):
            LedgerWriter("unit_7", folder=folder)
        out, err = _finish(child, 60, input="close\n")
    finally:
        child.kill()
    assert child.returncode == 0 and "closed" in out, err
    with LedgerWriter("unit_7", folder=folder) as w:  # closed by its owner: free
        assert w.next_seq == 1
        w.append("tool_call", {"from": "the parent"})

    child, pid = _hold_run(folder, "unit_7")
    try:
        with pytest.raises(LedgerLockedError, match=rf"'unit_7'.*\(process {pid}\)"):
            LedgerWriter("unit_7", folder=folder)
    finally:
        child.kill()  # the interpreter dies holding the lock, with no close()
        _finish(child, 60)
    with _open_once_free("unit_7", folder) as w:
        assert w.next_seq == 3
    assert [e["payload"]["from"] for e in iter_events(folder)] == ["the child", "the parent", "the child"]


_TAKE_TURNS = r"""
import sys, time
from avail.ledger import LedgerLockedError, LedgerWriter
folder, run_id, name, rounds, per_round = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
refused = 0
for r in range(rounds):
    while True:
        try:
            w = LedgerWriter(run_id, folder=folder)
            break
        except LedgerLockedError:
            refused += 1
            time.sleep(0.001)
    with w:
        first = w.next_seq
        for i in range(per_round):
            seq = w.append("tool_call", {"by": name, "round": r, "i": i})
            assert seq == first + i, (seq, first, i)  # nobody else wrote while this process held the run
print("done", name, "refused", refused, flush=True)
"""


def test_processes_that_take_turns_on_one_run_never_break_seq(tmp_path):
    folder = tmp_path / "contended"
    names, rounds, per_round = ["a", "b", "c"], 20, 3
    children = [_child(_TAKE_TURNS, folder, "contended", name, rounds, per_round) for name in names]
    try:
        results = [_finish(child, 120) for child in children]
    finally:
        for child in children:
            child.kill()
    for child, (out, err) in zip(children, results):
        assert child.returncode == 0 and out.startswith("done"), out + err
    torn = []
    events = _events(folder, torn=torn)
    assert torn == [] and [e["seq"] for e in events] == list(range(len(names) * rounds * per_round))
    for name in names:  # every event of every process is there, in the order that process wrote it
        mine = [(e["payload"]["round"], e["payload"]["i"]) for e in events if e["payload"]["by"] == name]
        assert mine == [(r, i) for r in range(rounds) for i in range(per_round)]
    # the events of one turn sit together: a process had the run to itself for the whole turn
    turns = [key for key, _ in itertools.groupby((e["payload"]["by"], e["payload"]["round"]) for e in events)]
    assert len(turns) == len(names) * rounds


_APPEND_QUIETLY = r"""
import sys, time
from avail.ledger import LedgerWriter
w = LedgerWriter(sys.argv[2], folder=sys.argv[1])
w.append("tool_call", {"i": 0})
print("ready", flush=True)
i = 1
while True:
    w.append("tool_call", {"i": i, "pad": "x" * 100})
    i += 1
    if i % 10 == 0:
        time.sleep(0.002)
"""


def test_a_reader_in_another_process_sees_a_consistent_ledger_while_it_grows(tmp_path):
    folder = tmp_path / "growing"
    child = _child(_APPEND_QUIETLY, folder, "growing")
    counts = []
    try:
        if child.stdout.readline().strip() != "ready":
            child.kill()
            pytest.fail(f"child did not start: {child.communicate(timeout=60)[1]}")
        deadline = time.monotonic() + 90
        while len(counts) < 15 or counts[-1] == counts[0]:  # 15 reads, and until the file has grown between reads
            assert time.monotonic() < deadline, f"the ledger never grew: {counts}"
            torn = []
            seqs = [e["seq"] for e in iter_events(folder, torn=torn)]
            assert seqs == list(range(len(seqs)))  # every read is a gapless run of whole events
            assert len(torn) <= 1 and all(not t.terminated for t in torn)  # at most the append in flight
            counts.append(len(seqs))
        tables = load_ledger(folder)  # the pandas loader reads a live ledger too
        assert tables.events["seq"].tolist() == list(range(len(tables.events))) and tables.seq_breaks == ()
    finally:
        child.kill()
        _finish(child, 60)
    assert counts == sorted(counts) and counts[0] >= 1 and counts[-1] > counts[0]  # it only ever grows


# --------------------------------------------------------------------------- re-entry from a signal handler
def test_reentrant_append_sync_and_close_raise_instead_of_deadlocking(monkeypatch):
    real_write = writer_module._write_all
    handler = {"call": None}

    def a_signal_handler_runs_inside_the_write(fh, data):
        if handler["call"] is not None:
            call, handler["call"] = handler["call"], None
            call()
        real_write(fh, data)

    monkeypatch.setattr(writer_module, "_write_all", a_signal_handler_runs_inside_the_write)

    def scenario():  # one thread throughout: the handler's call re-enters the writer it interrupted
        with LedgerWriter("reenter") as w:
            for action, call, instead in (
                ("append", lambda: w.append("assist", {"from": "the handler"}), "append"),
                ("sync", w.sync, "sync"),
                ("close", w.close, "close the writer"),
            ):
                handler["call"] = call
                pattern = (rf"{action} called re-entrantly on run 'reenter'.*"
                           rf"Set a flag in the handler and {instead} from the main loop")
                with pytest.raises(RuntimeError, match=pattern):
                    w.append("tool_call", {"from": "the main loop"})
            assert not w.closed
            assert w.next_seq == 0  # a refused handler costs the event it interrupted, and no seq
            assert w.append("tool_call", {"n": "after"}) == 0

    _with_deadline(scenario)
    torn = []
    assert [(e["seq"], e["payload"]["n"]) for e in iter_events("reenter", torn=torn)] == [(0, "after")]
    assert torn == []


_SIGINT_INSIDE_APPEND = r"""
import _thread, signal, sys
from avail.ledger import LedgerWriter, writer

w = LedgerWriter("s", folder=sys.argv[1])
real_write = writer._write_all
armed = {"on": False}

def write_with_ctrl_c(fh, data):
    if armed["on"]:
        armed["on"] = False
        _thread.interrupt_main()  # SIGINT for this (main) thread
        for _ in range(3):        # the handler runs at the next check, here, inside append
            pass
    real_write(fh, data)

def on_sigint(signum, frame):     # the pattern the docs warn against: appending from the handler
    w.append("assist", {"answer": "logged from the handler"})

writer._write_all = write_with_ctrl_c
signal.signal(signal.SIGINT, on_sigint)
w.append("model_call", {"usd": 0.001})
armed["on"] = True
try:
    w.append("model_call", {"usd": 0.001})
except RuntimeError as exc:
    print("refused:", exc, flush=True)
w.append("assist", {"answer": "logged from the main loop"})
w.close()
print("finished", flush=True)
"""


def test_a_sigint_handler_that_appends_inside_append_gets_an_error_not_a_hang(tmp_path):
    out, err = _finish(_child(_SIGINT_INSIDE_APPEND, tmp_path / "s"), 60)  # a deadlock is killed and fails here
    assert "refused: append called re-entrantly on run 's'" in out, out + err
    assert "Set a flag in the handler and append from the main loop" in out and "finished" in out, out + err
    events = _events(tmp_path / "s")
    assert [(e["seq"], e["kind"]) for e in events] == [(0, "model_call"), (1, "assist")]
    assert events[1]["payload"]["answer"] == "logged from the main loop"


def test_a_reentrant_call_during_serialization_raises_the_runtime_error_unwrapped(monkeypatch):
    real_dumps = writer_module.dumps
    handler = {"call": None}

    def a_signal_handler_runs_inside_dumps(obj, *args, **kwargs):
        if handler["call"] is not None:
            call, handler["call"] = handler["call"], None
            call()
        return real_dumps(obj, *args, **kwargs)

    monkeypatch.setattr(writer_module, "dumps", a_signal_handler_runs_inside_dumps)

    def scenario():  # must not come out as "ValueError: ... is not JSON serializable: RuntimeError: ..."
        with LedgerWriter("in_dumps") as w:
            for action, call in (
                ("append", lambda: w.append("assist", {"from": "the handler"})),
                ("sync", w.sync),
                ("close", w.close),
            ):
                handler["call"] = call
                with pytest.raises(RuntimeError, match=rf"{action} called re-entrantly on run 'in_dumps'") as err:
                    w.append("tool_call", {"from": "the main loop"})
                assert type(err.value) is RuntimeError and err.value.__cause__ is None  # as raised, unwrapped
            assert w.append("tool_call", {"n": "after"}) == 0  # the refused events were never written

    _with_deadline(scenario)
    assert [(e["seq"], e["payload"]["n"]) for e in iter_events("in_dumps")] == [(0, "after")]


# --------------------------------------------------------------------------- Ctrl+C while a writer opens or closes
_LEDGER_FILES = {writer_module.__file__, lines_module.__file__, jsonsafe_module.__file__}


def _ctrl_c_point(frame, event, arg):
    """Where a Ctrl+C could surface for one profile event, or None for an event outside the ledger code.

    CPython delivers Ctrl+C as a Python function starts or right after a call returns. The points
    are: a ledger function, or a Python function the ledger calls, starting or returning, and a
    C function the ledger calls returning.
    """
    if event in ("call", "return"):
        files = {frame.f_code.co_filename, frame.f_back.f_code.co_filename if frame.f_back else None}
        if files & _LEDGER_FILES:
            return event, frame.f_code.co_qualname
    elif event == "c_return" and frame.f_code.co_filename in _LEDGER_FILES:
        return event, frame.f_code.co_qualname, getattr(arg, "__name__", type(arg).__name__)
    return None


class _CtrlC:
    """A profile hook that raises KeyboardInterrupt at Ctrl+C point number ``at`` (never, when None),
    which surfaces in the writer exactly as a Ctrl+C arriving during that call would. Deterministic:
    no timer, no second thread. ``points`` lists the points passed, ``hit`` the one interrupted."""

    def __init__(self, at=None):
        self.at, self.points, self.hit = at, [], None

    def __call__(self, frame, event, arg):
        point = _ctrl_c_point(frame, event, arg)
        if point is None:
            return
        self.points.append(point)
        if len(self.points) - 1 == self.at:
            sys.setprofile(None)
            self.hit = point
            raise KeyboardInterrupt

    def run(self, fn):
        sys.setprofile(self)
        try:
            fn()
        finally:
            sys.setprofile(None)


def _assert_run_free(folder, rid):
    """Nothing in this process still holds the run: no registry entry, no OS lock, no open lock file."""
    run_dir = Path(os.path.abspath(folder))
    assert os.path.normcase(str(run_dir)) not in writer_module._OPEN_RUNS, f"run {rid!r} is still registered"
    LedgerWriter(rid, folder=folder).close()  # LedgerLockedError while any descriptor in this process holds the lock
    if os.name == "nt":  # Windows refuses to delete a file that a forgotten descriptor still has open
        _retry_permission(os.unlink, run_dir / LOCK_FILE)


# Points of the open and the close that the sweep must interrupt.
_NAMED_POINTS = {
    "the holder note is written under the lock": ("c_return", "_lock_file", "write"),
    "acquire() leaves its registry block": ("c_return", "_RunLock.acquire", "__exit__"),
    "acquire() returns into LedgerWriter.__init__": ("return", "_RunLock.acquire"),
    "the tail of the ledger has been read": ("return", "read_tail"),
    "the ledger file is open for appending": ("c_return", "LedgerWriter.__init__", "open"),
    "__init__ returns, the caller never gets the writer": ("return", "LedgerWriter.__init__"),
    "close() has synced the ledger file": ("c_return", "LedgerWriter.close", "fsync"),
    "release() starts, called by close()": ("call", "_RunLock.release"),
    "release() closed the lock file, the registry entry is next": ("c_return", "_RunLock.release", "close"),
    "_unlock_file() starts": ("call", "_unlock_file"),
}
_RESUME_POINTS = {
    "the torn tail got its newline": ("return", "_write_all"),
}

_OPEN_EACH = r"""
import sys, warnings
from avail.ledger import LedgerLockedError, LedgerWriter
warnings.simplefilter("ignore")
refused = []
for rid in sys.stdin.read().split():
    try:
        LedgerWriter(rid, folder=f"{sys.argv[1]}/{rid}").close()
    except LedgerLockedError:
        refused.append(rid)
print("refused:", refused, flush=True)
"""


@pytest.mark.parametrize("resume", [False, True], ids=["new_run", "resume_with_torn_tail"])
def test_ctrl_c_at_any_point_of_open_or_close_leaves_the_run_free(tmp_path, resume):
    def prepare(rid):
        if resume:  # one event and a crash's partial line: the open scans the tail, warns and ends the line
            _raw_ledger(tmp_path / rid, _line(0, run_id=rid))
            with open(tmp_path / rid / LEDGER_FILE, "ab") as fh:
                fh.write(b'{"run_id": "' + rid.encode())

    def open_and_close(rid, box):
        box.append(LedgerWriter(rid, folder=tmp_path / rid))
        box[0].close()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LedgerWarning)
        prepare("warm_up")
        LedgerWriter("warm_up", folder=tmp_path / "warm_up").close()  # one-time work stays out of the sweep
    hit, rids = set(), []
    gc.collect()
    gc.disable()  # what the interrupt drops must be freed at once, by reference counting alone
    try:
        for n in range(5000):  # point 0, 1, 2, ... until an open and close runs to the end uninterrupted
            rid = f"r{n:05d}"
            rids.append(rid)
            prepare(rid)
            box, inject = [], _CtrlC(at=n)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)  # a file object the interrupt dropped closes itself
                warnings.simplefilter("ignore", LedgerWarning)
                try:
                    inject.run(lambda: open_and_close(rid, box))
                    raised = False
                except KeyboardInterrupt:
                    raised = True
                assert raised == (inject.hit is not None), f"the Ctrl+C at {inject.hit} never reached the caller"
                if inject.hit is not None:
                    hit.add(inject.hit)
                if box and not box[0].closed:  # what a caller does: close the writer object it has, if any
                    box[0].close()
                _assert_run_free(tmp_path / rid, rid)
            if resume:
                assert _seqs(tmp_path / rid) == [0]  # nothing lost, nothing added
            if inject.hit is None:  # every point has had its turn
                break
    finally:
        gc.enable()
    assert n > 60 and inject.hit is None, f"{n} points interrupted"
    wanted = {**_NAMED_POINTS, **(_RESUME_POINTS if resume else {})}
    missing = {label: point for label, point in wanted.items() if point not in hit}
    assert not missing, f"the sweep never interrupted these points: {missing}"
    out, err = _finish(_child(_OPEN_EACH, tmp_path), 120, input="\n".join(rids))
    assert "refused: []" in out, out + err  # free for other processes too


def test_ctrl_c_at_any_point_of_an_append_keeps_seq_gapless_and_the_writer_usable(tmp_path):
    folder = tmp_path / "sweep"
    hit, outcomes = set(), collections.Counter()
    with LedgerWriter("sweep", folder=folder) as w:
        w.append("claim", {"n": "warm up"})
        for n in range(2000):  # point 0, 1, 2, ... until an append runs to the end uninterrupted
            before = len(_seqs(folder))
            inject = _CtrlC(at=n)
            try:
                inject.run(lambda: w.append("tool_call", {"n": n}))
                raised = False
            except KeyboardInterrupt:
                raised = True
            assert raised == (inject.hit is not None), f"the Ctrl+C at {inject.hit} never reached the caller"
            torn = []
            seqs = [e["seq"] for e in iter_events(folder, torn=torn)]
            assert seqs == list(range(len(seqs))) and torn == []  # no gap, no repeat, no fragment
            assert w.next_seq == len(seqs)  # and the writer knows where it is
            if inject.hit is None:
                assert len(seqs) == before + 1
                break
            hit.add(inject.hit)
            assert len(seqs) - before in (0, 1)  # the interrupted event is in the ledger whole, or absent
            outcomes["in the ledger" if len(seqs) > before else "absent"] += 1
            assert w.append("tool_call", {"after": n}) == len(seqs)  # the writer still works
        final = _events(folder)
    assert n > 20 and inject.hit is None, f"{n} points interrupted"
    assert outcomes["in the ledger"] > 0 and outcomes["absent"] > 0, outcomes  # both sides of the write were hit
    for point in (("call", "LedgerWriter.append"), ("call", "_write_all"), ("return", "_write_all"),
                  ("c_return", "_write_all", "write"), ("c_return", "LedgerWriter.append", "flush")):
        assert point in hit, f"the sweep never interrupted {point}"
    assert [e["seq"] for e in final] == list(range(len(final)))
    # an event whose append was interrupted is never in the ledger twice
    interrupted = [e["payload"]["n"] for e in final if "n" in e["payload"] and e["payload"]["n"] != "warm up"]
    assert len(interrupted) == len(set(interrupted))


@pytest.mark.parametrize("first_fault", ["nothing written", "half a line", "all but the newline", "the whole line"])
def test_a_second_ctrl_c_at_any_point_of_the_settling_still_ends_gapless(tmp_path, monkeypatch, first_fault):
    """The first Ctrl+C stops the write in one of 4 states. As it passes through append, the writer
    looks at the file to settle what was left, and a second Ctrl+C lands at each point of that look."""
    real_write = writer_module._write_all
    real_settle = writer_module.LedgerWriter._settle_quietly
    fault = {"armed": False}
    second = {"hook": None}

    def write(fh, data):
        if fault["armed"] and data != b"\n":
            fault["armed"] = False
            part = {"nothing written": b"", "half a line": data[: len(data) // 2], "all but the newline": data[:-1],
                    "the whole line": data}[first_fault]
            if part:
                real_write(fh, part)
            raise KeyboardInterrupt
        real_write(fh, data)

    def settle_quietly(self, fh):
        hook, second["hook"] = second["hook"], None
        if hook is None:
            return real_settle(self, fh)
        sys.setprofile(hook)  # the second Ctrl+C: armed only for the settle that follows the first
        try:
            return real_settle(self, fh)
        finally:
            sys.setprofile(None)

    monkeypatch.setattr(writer_module, "_write_all", write)
    monkeypatch.setattr(writer_module.LedgerWriter, "_settle_quietly", settle_quietly)
    arrives = first_fault in ("all but the newline", "the whole line")  # every byte of the event reached the file
    folder = tmp_path / "run"
    hit = set()
    with LedgerWriter("run", folder=folder) as w:
        w.append("claim", {"n": "first"})
        for n in range(500):
            before = len(_seqs(folder))
            inject = _CtrlC(at=n)
            fault["armed"], second["hook"] = True, inject
            with pytest.raises(KeyboardInterrupt):
                w.append("tool_call", {"interrupted": n})
            assert second["hook"] is None  # the settle ran, with the second Ctrl+C armed
            if inject.hit is not None:
                hit.add(inject.hit)
            assert w.next_seq == before + arrives  # whatever the second Ctrl+C left, the next look settles
            torn = []
            seqs = [e["seq"] for e in iter_events(folder, torn=torn)]
            assert seqs == list(range(before + arrives))
            assert all(t.terminated for t in torn)  # a fragment has its newline by now
            assert w.append("tool_call", {"after": n}) == len(seqs)  # and the next event starts a fresh line
            if inject.hit is None:
                break
    assert n >= 8 and inject.hit is None, f"{n} points interrupted"
    for point in (("call", "LedgerWriter._settle"), ("return", "_read_from"), ("return", "LedgerWriter._settle")):
        assert point in hit, f"the sweep never interrupted {point}"
    if first_fault in ("half a line", "all but the newline"):
        assert ("return", "_write_all") in hit  # right after the newline that ends the fragment or completes the line
    torn = []
    events = _events(folder, torn=torn)
    assert [e["seq"] for e in events] == list(range(len(events)))
    assert len(torn) == (n + 1 if first_fault == "half a line" else 0)
    interrupted = [e["payload"]["interrupted"] for e in events if "interrupted" in e["payload"]]
    assert interrupted == (list(range(n + 1)) if arrives else [])  # each in the ledger exactly once, or not at all


def test_ctrl_c_before_close_starts_leaves_the_writer_open_and_close_finishes_later():
    w = LedgerWriter("late")
    inject = _CtrlC(at=0)
    with pytest.raises(KeyboardInterrupt):
        inject.run(w.close)
    assert inject.hit == ("call", "LedgerWriter.close")
    assert not w.closed  # nothing was let go: the writer still works and still holds the run
    assert w.append("claim", {"still": "open"}) == 0
    with pytest.raises(LedgerLockedError, match="already has an open LedgerWriter in this process"):
        LedgerWriter("late")
    w.close()
    assert w.closed
    _assert_run_free(w.dir, "late")
    assert [e["payload"] for e in iter_events("late")] == [{"still": "open"}]


def test_close_stopped_by_two_ctrl_c_says_not_closed_until_the_run_is_released(monkeypatch):
    real_release = writer_module._RunLock.release
    shots = {"left": 0}

    def stopped_at_its_first_line(self):  # a Ctrl+C that lands as release() starts
        if shots["left"]:
            shots["left"] -= 1
            raise KeyboardInterrupt
        real_release(self)

    monkeypatch.setattr(writer_module._RunLock, "release", stopped_at_its_first_line)
    for finish in ("close again", "drop the writer"):
        w = LedgerWriter("twice")
        folder = w.dir
        shots["left"] = 2  # one for each of close()'s 2 release() calls
        with pytest.raises(KeyboardInterrupt):
            w.close()
        assert not w.closed  # the ledger file is closed, the run is not released yet
        with pytest.raises(ValueError, match="closed"):
            w.append("claim", {"too": "late"})
        with pytest.raises(LedgerLockedError, match="already has an open LedgerWriter in this process"):
            LedgerWriter("twice")
        if finish == "close again":
            w.close()
            assert w.closed
        else:  # the finalizer is still armed: dropping the writer frees the run
            del w
            gc.collect()
        _assert_run_free(folder, "twice")


def test_a_release_reentered_from_a_handler_leaves_a_new_writer_locked(tmp_path, monkeypatch):
    """A handler that runs inside a release, after it read the lock file's descriptor and before it
    unlocks, then closes the writer and opens and keeps another one, must not have the outer release
    unlock that new writer through the reused descriptor number."""
    W = writer_module
    root = tmp_path / "runs"
    w = W.LedgerWriter("run_a", folder=root / "run_a")
    kept, fired = [], []

    def handler():
        fired.append(True)
        w.close()
        kept.append(W.LedgerWriter("run_b", folder=root / "run_b"))

    # the outer release's first OS call on the descriptor it read: the seek to the lock byte on Windows,
    # the unlock itself elsewhere. The handler runs just before it, as a signal handler could.
    with monkeypatch.context() as patch:  # undone right after the close, and nothing else is undone with it
        if os.name == "nt":
            real_lseek = W.os.lseek

            def hooked_lseek(fd, pos, how):
                if pos == W._LOCK_OFFSET and not fired:
                    handler()
                return real_lseek(fd, pos, how)

            patch.setattr(W.os, "lseek", hooked_lseek)
        else:
            real_flock = W.fcntl.flock

            def hooked_flock(fd, op):
                if op == W.fcntl.LOCK_UN and not fired:
                    handler()
                return real_flock(fd, op)

            patch.setattr(W.fcntl, "flock", hooked_flock)
        w.close()
    assert fired and w.closed
    probe = ("import sys\nfrom avail.ledger import LedgerWriter, LedgerLockedError\n"
             "try:\n    LedgerWriter(sys.argv[2], folder=sys.argv[1]).close()\n    print('opened')\n"
             "except LedgerLockedError:\n    print('refused')\n")

    def probe_run(run):
        done = subprocess.run([sys.executable, "-c", probe, str(root / run), run], capture_output=True, text=True,
                              timeout=60, cwd=paths.REPO_ROOT)
        return done.stdout.strip(), done.stdout + done.stderr

    answer, detail = probe_run("run_b")
    assert answer == "refused", detail
    kept[0].close()
    for run in ("run_a", "run_b"):  # both runs are free again, here and in another process
        W.LedgerWriter(run, folder=root / run).close()
        answer, detail = probe_run(run)
        assert answer == "opened", detail


# --------------------------------------------------------------------------- reading
def test_load_ledger_gives_typed_tables(data_root):
    w = _fill("tables")
    t = load_ledger("tables")
    assert isinstance(t, LedgerTables) and t.path == w.path and t.problems == []

    ev = t.events
    assert list(ev.columns) == ["run_id", "seq", "ts", "kind", "payload"]
    assert [str(d) for d in ev.dtypes] == ["str", "int64", "float64", "str", "object"]
    assert ev["run_id"].tolist() == ["tables"] * 7 and ev["seq"].tolist() == list(range(7))
    assert ev["ts"].tolist() == [100.5, 101.5, 102.5, 103.0, 104.0, 105.0, 106.0]
    assert ev["kind"].tolist() == ["claim", "evidence", "evidence", "verdict", "verdict", "cycle_end", "cycle_end"]
    assert type(ev["payload"].iloc[0]) is dict and ev["payload"].iloc[3]["evidence_ids"] == ["e1", "e2"]
    assert list(ev.index) == list(range(7))

    v = t.verdicts
    assert list(v.columns) == ["run_id", "seq", "ts", "kind", "verdict_id", "claim_id", "decision", "score",
                               "stage_reached", "witness_version", "evidence_ids", "cost_total"]
    assert [str(d) for d in v.dtypes] == ["str", "int64", "float64", "str", "str", "str", "str", "float64", "Int64",
                                          "Int64", "object", "object"]
    assert v["seq"].tolist() == [3, 4] and v["ts"].tolist() == [103.0, 104.0] and v["kind"].tolist() == ["verdict"] * 2
    assert v["decision"].tolist() == ["accept", "abstain"] and v["score"].tolist() == [0.91, 0.5]
    assert v["stage_reached"].tolist() == [1, 2] and int(v["witness_version"].sum()) == 6
    assert v["evidence_ids"].tolist() == [["e1", "e2"], []]  # a list stays a list
    assert v["cost_total"].iloc[0] == {"seconds": 4.2, "usd": 0.002}  # a nested dict stays a dict
    assert list(v.index) == [0, 1] and v.attrs["renamed_payload_keys"] == {}
    assert t.table("verdict") is v

    e = t.evidence
    assert e["kind"].tolist() == ["evidence", "evidence"] and e["ts"].tolist() == [101.5, 102.5]  # the event's own
    assert e["payload_kind"].tolist() == ["active_probe", "judge"] and e["payload_ts"].tolist() == [101.0, 102.0]
    assert e.attrs["renamed_payload_keys"] == {"kind": "payload_kind", "ts": "payload_ts"}
    assert str(e["payload_kind"].dtype) == "str" and e["payload_ts"].dtype == np.float64
    assert e["score"].dtype == np.float64 and e["score"].isna().tolist() == [False, True]  # a null score is NA
    assert e["features"].tolist() == [{"peak_n": 5.5, "travel_mm": 0.4}, {}] and e["features"].dtype == object
    assert e["artifacts"].tolist() == [[], ["a.jpg", "b.jpg"]]
    assert e["cost"].iloc[1]["usd"] == 0.002

    c = t.claims
    assert c["run_id"].tolist() == ["tables"] and c["payload_run_id"].tolist() == ["tables"]
    assert c["ts"].tolist() == [100.5] and c["payload_ts"].tolist() == [100.0]
    assert str(c["cycle_id"].dtype) == "Int64" and str(c["raised_by"].dtype) == "str"
    assert c["predicate"].iloc[0] == {"name": "seated", "target": {"plug": "p1", "port": 7}}

    ce = t.cycle_ends
    assert str(ce["declared_success"].dtype) == "boolean" and ce["declared_success"].tolist() == [True, False]
    assert str(ce["true_success"].dtype) == "boolean" and ce["true_success"].isna().tolist() == [False, True]
    assert bool(ce["true_success"].iloc[0]) is False  # a missing key is NA, a false stays false
    assert ce["wall_s"].dtype == np.float64 and ce["wall_s"].tolist() == [12.0, 9.5]  # 12 and 9.5 together
    assert int((ce["declared_success"] & ~ce["true_success"].fillna(True)).sum()) == 1  # one uncaught failure

    for kind in ("fork", "assist", "audit", "tool_call", "model_call", "calibration_update", "drift_alarm"):
        frame = t.table(kind)  # a kind with no events: an empty frame with the base columns, typed
        assert frame.empty and list(frame.columns) == ["run_id", "seq", "ts", "kind"] == list(BASE_COLUMNS)
        assert [str(d) for d in frame.dtypes] == ["str", "int64", "float64", "str"]
    with pytest.raises(ValueError, match="unknown ledger event kind 'verdicts'. Valid kinds: claim, evidence"):
        t.table("verdicts")
    assert "7 events" in repr(t) and "verdicts 2" in repr(t)


def test_there_is_one_table_per_event_kind_under_the_documented_name(tmp_path):
    documented = {
        "claim": "claims", "evidence": "evidence", "verdict": "verdicts", "fork": "forks", "assist": "assists",
        "audit": "audits", "tool_call": "tool_calls", "model_call": "model_calls",
        "calibration_update": "calibration_updates", "drift_alarm": "drift_alarms", "cycle_end": "cycle_ends",
    }
    assert {kind: TABLE_NAMES[kind] for kind in documented} == documented
    assert tuple(TABLE_NAMES) == tuple(EVENT_KINDS)  # every kind the schemas know has a table
    with LedgerWriter("each", folder=tmp_path / "each") as w:
        for i, kind in enumerate(EVENT_KINDS):
            for _ in range(i + 1):
                w.append(kind, {"of": kind})
    tables = load_ledger(tmp_path / "each")
    for i, (kind, name) in enumerate(TABLE_NAMES.items()):
        frame = getattr(tables, name)
        assert frame is tables.table(kind) and len(frame) == i + 1
        assert set(frame["kind"]) == {kind} and set(frame["of"]) == {kind}
    assert sum(len(tables.table(kind)) for kind in EVENT_KINDS) == len(tables.events)


def test_empty_and_missing_ledgers(tmp_path, data_root):
    with LedgerWriter("idle"):
        pass  # a run that has not logged anything yet
    (tmp_path / "bare").mkdir()  # a run folder with no ledger file
    for source in ("idle", tmp_path / "bare"):
        t = load_ledger(source)
        assert t.events.empty and list(t.events.columns) == ["run_id", "seq", "ts", "kind", "payload"]
        assert [str(d) for d in t.events.dtypes] == ["str", "int64", "float64", "str", "object"]
        assert t.events["seq"].sum() == 0 and t.events["ts"].sum() == 0.0
        for kind in EVENT_KINDS:
            frame = t.table(kind)
            assert frame.empty and list(frame.columns) == list(BASE_COLUMNS)
        assert t.problems == [] and t.torn_lines == () and t.seq_breaks == () and t.torn_tail is None
        assert _events(source) == []
    assert load_ledger(tmp_path / "bare").path == tmp_path / "bare" / LEDGER_FILE

    with pytest.raises(FileNotFoundError, match=r"no ledger for run 'no_such_run': .*pass a pathlib\.Path"):
        load_ledger("no_such_run")
    with pytest.raises(FileNotFoundError, match="no ledger for run 'no_such_run'"):
        iter_events("no_such_run")  # at the call, before any iteration
    with pytest.raises(FileNotFoundError):
        load_ledger(tmp_path / "nowhere" / "ledger.jsonl")
    with pytest.raises(FileNotFoundError):
        load_ledger(tmp_path / "nowhere")
    assert ledger_path(tmp_path / "nowhere") == tmp_path / "nowhere" / LEDGER_FILE  # no suffix: a run folder
    assert ledger_path(tmp_path / "nowhere" / "copy.jsonl") == tmp_path / "nowhere" / "copy.jsonl"  # .jsonl: the file
    with pytest.raises(FileNotFoundError, match=r"^no ledger at .*copy\.jsonl does not exist$"):
        load_ledger(tmp_path / "nowhere" / "copy.jsonl")
    # reading creates nothing
    assert not (data_root / "runs" / "no_such_run").exists() and not (tmp_path / "nowhere").exists()


def test_readers_accept_a_run_id_a_folder_or_a_file(tmp_path, monkeypatch):
    with LedgerWriter("r1") as w:
        w.append("claim", {"x": 1})
    for source in ("r1", w.dir, str(w.dir), w.path, str(w.path)):
        assert ledger_path(source) == w.path
        assert [e["payload"] for e in iter_events(source)] == [{"x": 1}]
        tables = load_ledger(source)
        assert tables.path == w.path and tables.claims["x"].tolist() == [1]

    # a str that is a usable run id is always a run, never a folder of that name in the working directory
    cwd = tmp_path / "cwd"
    _raw_ledger(cwd / "r1", _line(0, run_id="other"), _line(1, run_id="other"))
    (cwd / "copy.jsonl").write_bytes(w.path.read_bytes())
    monkeypatch.chdir(cwd)
    assert [e["run_id"] for e in iter_events("r1")] == ["r1"]
    assert [e["run_id"] for e in iter_events(Path("r1"))] == ["other", "other"]  # a Path is always a path
    assert [e["run_id"] for e in iter_events("r1/ledger.jsonl")] == ["other", "other"]
    assert [e["run_id"] for e in iter_events("copy.jsonl")] == ["r1"]  # a .jsonl name is a file, never a run id
    assert load_ledger("./r1").events["run_id"].tolist() == ["other", "other"]


def test_list_runs_names_the_runs_that_have_a_ledger(data_root):
    assert list_runs() == [] and not data_root.exists()  # no data root yet, and asking creates nothing
    for run_id in ("run_b", "run_a"):
        with LedgerWriter(run_id) as w:
            w.append("claim", {"n": 1})
    (data_root / "runs" / "no_ledger").mkdir()  # a run folder with no ledger file yet
    (data_root / "runs" / "notes.txt").write_text("not a run", encoding="utf-8")
    _raw_ledger(data_root / "runs" / "odd.jsonl", _line(0))  # a name the readers take for a file, never a run id
    assert list_runs() == ["run_a", "run_b"]
    assert [load_ledger(run_id).events["run_id"].tolist() for run_id in list_runs()] == [["run_a"], ["run_b"]]


def test_iter_events_streams_and_filters_by_kind():
    w = _fill("filter")
    stream = iter_events("filter")
    assert iter(stream) is stream and not isinstance(stream, (list, tuple))  # one event at a time
    assert next(stream)["seq"] == 0 and next(stream)["seq"] == 1
    stream.close()
    moved = w.path.with_name("moved.jsonl")
    os.replace(w.path, moved)  # a stream that is closed has let go of the file (Windows refuses this otherwise)
    os.replace(moved, w.path)
    kept = [e["kind"] for e in iter_events("filter", kinds=["verdict", "cycle_end"])]
    assert kept == ["verdict", "verdict", "cycle_end", "cycle_end"]
    assert [e["seq"] for e in iter_events("filter", kinds="evidence")] == [1, 2]
    assert _events("filter", kinds=[]) == []
    with pytest.raises(ValueError, match=r"unknown ledger event kind\(s\) 'verdicts'. Valid kinds: claim, evidence"):
        iter_events("filter", kinds=["verdict", "verdicts"])


_NO_PANDAS = r"""
import sys
import avail.ledger as ledger
events = list(ledger.iter_events(sys.argv[1]))
with ledger.LedgerWriter("np", folder=sys.argv[1]) as w:
    w.append("claim", {"n": len(events)})
print(len(events), len(list(ledger.iter_events(sys.argv[1]))), "pandas" in sys.modules, flush=True)
"""


def test_writing_and_streaming_never_import_pandas(tmp_path):
    folder = tmp_path / "np"
    with LedgerWriter("np", folder=folder) as w:
        for i in range(3):
            w.append("claim", {"n": i})
    out, err = _finish(_child(_NO_PANDAS, folder), 60)
    assert out.split() == ["3", "4", "False"], out + err


def test_a_ledger_can_be_read_while_it_is_being_written():
    with LedgerWriter("live") as w:
        for i in range(5):
            w.append("tool_call", {"i": i})
            assert [e["payload"]["i"] for e in iter_events("live")] == list(range(i + 1))
        tables = load_ledger("live")  # no lock needed, and the writer is still open
        assert tables.tool_calls["i"].tolist() == [0, 1, 2, 3, 4] and tables.problems == []
        assert w.append("tool_call", {"i": 5}) == 5
    assert len(load_ledger("live").events) == 6


def test_payload_keys_never_shadow_the_base_columns():
    with LedgerWriter("odd") as w:
        w.append("assist", {"seq": "not the seq", "kind": "not the kind", "run_id": "not the run", "ts": "not the ts",
                            "notes": {"a": [1, 2]}})
    frame = load_ledger("odd").assists
    assert list(frame.columns) == ["run_id", "seq", "ts", "kind", "payload_seq", "payload_kind", "payload_run_id",
                                   "payload_ts", "notes"]
    row = frame.iloc[0]
    assert row["kind"] == "assist" and row["seq"] == 0 and row["run_id"] == "odd" and isinstance(row["ts"], float)
    assert row["payload_seq"] == "not the seq" and row["payload_kind"] == "not the kind"
    assert row["payload_run_id"] == "not the run" and row["payload_ts"] == "not the ts"
    assert row["notes"] == {"a": [1, 2]}
    assert frame.attrs["renamed_payload_keys"] == {"seq": "payload_seq", "kind": "payload_kind",
                                                   "run_id": "payload_run_id", "ts": "payload_ts"}
    assert load_ledger("odd").events["payload"].iloc[0]["kind"] == "not the kind"  # the payload itself is untouched


def test_a_payload_key_equal_to_a_prefixed_name_gets_a_numeric_suffix():
    with LedgerWriter("simple") as w:
        w.append("assist", {"seq": "from key seq", "payload_seq": "from key payload_seq"})
    frame = load_ledger("simple").assists
    row = frame.iloc[0]
    assert row["seq"] == 0 and row["payload_seq"] == "from key seq" and row["payload_seq_2"] == "from key payload_seq"
    assert frame.attrs["renamed_payload_keys"] == {"seq": "payload_seq", "payload_seq": "payload_seq_2"}

    with LedgerWriter("crowded") as w:
        w.append("assist", {"seq": "key seq", "payload_seq": "key payload_seq", "payload_seq_2": "key payload_seq_2"})
        w.append("assist", {"payload_seq": "literal only"})  # one column means one key in every row
    frame = load_ledger("crowded").assists
    assert frame.attrs["renamed_payload_keys"] == {"seq": "payload_seq", "payload_seq": "payload_seq_3"}
    assert frame["payload_seq"].tolist()[0] == "key seq" and pd.isna(frame["payload_seq"].iloc[1])
    assert frame["payload_seq_2"].tolist()[0] == "key payload_seq_2"
    assert frame["payload_seq_3"].tolist() == ["key payload_seq", "literal only"]


def test_columns_are_typed_from_their_values():
    with LedgerWriter("mixed") as w:
        w.append("tool_call", {"n": 1, "ratio": 1, "label": "a", "flag": True, "mixed": 1, "big": 2**70,
                               "nothing": None, "args": {"x": 1}, "flag_or_count": True})
        w.append("tool_call", {"n": 2, "ratio": 0.5, "label": "b", "flag": False, "mixed": "two", "big": 1,
                               "nothing": None, "args": [1, 2], "flag_or_count": 2})
        w.append("tool_call", {"n": None})
    frame = load_ledger("mixed").tool_calls
    assert {col: str(frame[col].dtype) for col in frame.columns[4:]} == {
        "n": "Int64", "ratio": "float64", "label": "str", "flag": "boolean", "mixed": "object", "big": "object",
        "nothing": "object", "args": "object", "flag_or_count": "object",
    }
    assert frame["n"].tolist()[:2] == [1, 2] and frame["n"].isna().tolist() == [False, False, True]
    assert frame["ratio"].tolist()[:2] == [1.0, 0.5] and np.isnan(frame["ratio"].iloc[2])
    assert frame["label"].isna().tolist() == [False, False, True]
    assert frame["flag"].isna().tolist() == [False, False, True]
    assert frame["mixed"].tolist()[:2] == [1, "two"]  # a column that mixes types keeps the raw values
    assert frame["big"].iloc[0] == 2**70 and frame["big"].iloc[1] == 1  # too big for int64: kept as it is
    assert frame["args"].tolist()[:2] == [{"x": 1}, [1, 2]]
    assert frame["flag_or_count"].tolist()[:2] == [True, 2] and frame["flag_or_count"].iloc[0] is True


def test_a_ledger_with_a_torn_tail_still_loads_and_reports_the_torn_line():
    with LedgerWriter("torn") as w:
        for i in range(4):
            w.append("verdict", {"verdict_id": f"v{i}", "decision": "accept"})
    good = w.path.read_bytes()
    fragment = b'{"run_id":"torn","seq":4,"ts":17940'
    with open(w.path, "ab") as fh:
        fh.write(fragment)
    t = load_ledger("torn")
    assert t.events["seq"].tolist() == [0, 1, 2, 3]  # the complete lines load
    assert t.verdicts["verdict_id"].tolist() == ["v0", "v1", "v2", "v3"]
    assert len(t.torn_lines) == 1
    line = t.torn_lines[0]
    assert (line.number, line.offset, line.length, line.terminated) == (5, len(good), len(fragment), False)
    assert line.text == fragment.decode() and t.torn_tail is line and t.seq_breaks == ()
    assert t.problems == [line.describe()]
    said = f"line 5 is not a complete event (35 bytes at byte {len(good)}, no newline after it): "
    assert t.problems[0].startswith(said)
    assert "1 torn" in repr(t)


def test_breaks_in_seq_are_reported_and_nothing_is_dropped(tmp_path):
    folder = _raw_ledger(tmp_path / "gappy", _line(0, run_id="g"), _line(1, run_id="g"), _line(3, run_id="g"),
                         _line(3, run_id="g"), _line(4, run_id="h"))
    said = r"ledger\.jsonl is not a whole ledger: line 3 has seq 3 where seq 2 belongs \(and 2 more\)\. Events are"
    with pytest.warns(LedgerWarning, match=said) as caught:
        t = load_ledger(folder)
    assert len(caught) == 1
    assert t.events["seq"].tolist() == [0, 1, 3, 3, 4]  # file order, as written: nothing dropped, nothing moved
    assert t.seq_breaks == (SeqBreak(3, 2, 3), SeqBreak(4, 4, 3))
    assert t.problems == ["line 3 has seq 3 where seq 2 belongs", "line 4 has seq 3 where seq 4 belongs",
                          "the file holds events of 2 runs: 'g', 'h'"]

    merged = _raw_ledger(tmp_path / "merged", _line(0, run_id="g"), _line(1, run_id="h"))  # gapless, but 2 runs
    with pytest.warns(LedgerWarning, match=r"is not a whole ledger: the file holds events of 2 runs: 'g', 'h'\. "):
        t = load_ledger(merged)
    assert t.seq_breaks == () and t.problems == ["the file holds events of 2 runs: 'g', 'h'"]

    with warnings.catch_warnings():
        warnings.simplefilter("error", LedgerWarning)  # a torn line alone is no reason to warn a reader
        torn_only = _raw_ledger(tmp_path / "torn_only", _line(0, run_id="g"), "{half a line", _line(1, run_id="g"))
        t = load_ledger(torn_only)
    assert t.seq_breaks == () and len(t.problems) == 1 and t.problems[0].startswith("line 2 is not a complete event")


def test_lines_from_another_tool_with_lone_surrogate_escapes_or_nan_still_load(tmp_path):
    texts = ["fine", "cut \ud83d", "low \udc80"]
    calls = [_line(i, run_id="old", kind="model_call", payload={"usd": 0.1, "error": text})
             for i, text in enumerate(texts)]
    folder = _raw_ledger(
        tmp_path / "old", *calls,
        '{"run_id":"old","seq":3,"ts":1.0,"kind":"evidence","payload":{"score":NaN,"far":Infinity,"near":-Infinity}}',
    )
    assert b"\\ud83d" in (folder / LEDGER_FILE).read_bytes()  # what plain json.dumps wrote
    torn = []
    events = _events(folder, torn=torn)
    assert torn == []
    assert [e["payload"]["error"] for e in events[:3]] == ["fine", "cut " + FFFD, "low " + FFFD]
    assert events[3]["payload"] == {"score": None, "far": None, "near": None}  # as the writer would have stored them
    tables = load_ledger(folder)
    assert len(tables.model_calls) == 3 and tables.model_calls["usd"].sum() == pytest.approx(0.3)
    assert str(tables.model_calls["error"].dtype) == "str"
    with LedgerWriter("old", folder=folder) as w:
        assert w.next_seq == 4 and w.torn_tail is None


def test_parse_line_takes_text_or_bytes_and_returns_the_5_fields_in_order():
    text = '{"payload":{"a":[1,2]},"kind":"claim","ts":3,"seq":2,"run_id":"r"}'
    for line in (text, text.encode(), bytearray(text.encode())):
        event = parse_line(line)
        assert list(event) == ["run_id", "seq", "ts", "kind", "payload"]
        assert event == {"run_id": "r", "seq": 2, "ts": 3.0, "kind": "claim", "payload": {"a": [1, 2]}}
        assert type(event["ts"]) is float
    with pytest.raises(ValueError, match="not UTF-8 text"):
        parse_line(b'{"run_id":"r\xff"}')
    with pytest.raises(ValueError, match="ts must be a finite number"):
        parse_line('{"run_id":"r","seq":0,"ts":1e999,"kind":"claim","payload":{}}')
