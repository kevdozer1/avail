"""Reading a ledger back: a stream of events, or one typed pandas table per event kind.

``load_ledger`` is what reports build from. It returns a :class:`LedgerTables`
holding ``events`` (every event, with its payload as a dict) and one table per
kind (``claims``, ``evidence``, ``verdicts``, ...) where the payload's top-level
keys are columns. ``iter_events`` gives the same events one at a time and never
imports pandas, for the code that only needs to walk a run.

Both skip a line that is not a complete event and say so: ``load_ledger`` lists
such lines in ``LedgerTables.torn_lines`` and ``iter_events`` adds them to the
``torn`` list it is given. ``load_ledger`` also checks that seq counts up by 1
from 0 and warns when it does not. Neither takes the writer's lock, so a ledger
can be read while a run is still writing it.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from .. import paths
from ..schemas.kinds import EVENT_KINDS
from .lines import LEDGER_FILE, LedgerWarning, TornLine, is_run_id, parse_line, scan_lines, strip_line, torn_line

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "BASE_COLUMNS", "TABLE_NAMES", "LedgerTables", "SeqBreak", "iter_events", "ledger_path", "list_runs",
    "load_ledger",
]

BASE_COLUMNS = ("run_id", "seq", "ts", "kind")
_BASE_DTYPES = {"run_id": "str", "seq": "int64", "ts": "float64", "kind": "str"}
_PAYLOAD_PREFIX = "payload_"


def _table_name(kind: str) -> str:
    return kind if kind == "evidence" else kind + "s"


#: Event kind -> the LedgerTables attribute that holds its table (``claim`` -> ``claims``).
TABLE_NAMES: dict[str, str] = {kind: _table_name(kind) for kind in EVENT_KINDS}


class SeqBreak(NamedTuple):
    """A place where seq does not count up by 1: the event on line ``number`` has seq ``found`` where
    ``expected`` belongs. A gap means an event is missing and a repeat means one was written twice."""

    number: int
    expected: int
    found: int

    def describe(self) -> str:
        """One sentence for a report."""
        return f"line {self.number} has seq {self.found} where seq {self.expected} belongs"


# ------------------------------------------------------------------------ finding the file
def ledger_path(source: str | os.PathLike[str]) -> Path:
    """The ledger file for a run id, a run folder or a ledger file.

    A ``str`` that is a usable run id names the run under ``AVAIL_DATA/runs``.
    Any other ``str``, and any ``Path``, is a path: a folder stands for the
    ``ledger.jsonl`` inside it and anything else is the ledger file itself. So
    ``"e3_arm1"`` is a run, ``"out/e3_arm1"`` and ``Path("e3_arm1")`` are folders
    and ``"copy.jsonl"`` is a file. Nothing is created.
    """
    if isinstance(source, str) and is_run_id(source):
        return paths.run_dir(source, create=False) / LEDGER_FILE
    target = Path(source)
    if target.is_dir():
        return target / LEDGER_FILE
    if target.is_file() or target.suffix.lower() == ".jsonl":
        return target
    return target / LEDGER_FILE


def list_runs() -> list[str]:
    """The run ids under ``AVAIL_DATA/runs`` that have a ledger file, sorted.

    Each one can be handed to :func:`load_ledger` or :func:`iter_events`. A data
    root that does not exist yet gives an empty list, and nothing is created.
    """
    base = paths.runs_dir(create=False)
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if is_run_id(p.name) and (p / LEDGER_FILE).is_file())


def _existing(source: str | os.PathLike[str]) -> Path | None:
    """The ledger file to read, or None for a run folder that has no ledger yet. FileNotFoundError otherwise."""
    path = ledger_path(source)
    if path.is_file():
        return path
    if path.name == LEDGER_FILE and path.parent.is_dir():
        return None
    if isinstance(source, str) and is_run_id(source):
        raise FileNotFoundError(
            f"no ledger for run {source!r}: {path} does not exist. A plain name is read as a run id under "
            "AVAIL_DATA/runs. To read a folder or a file, pass a pathlib.Path."
        )
    raise FileNotFoundError(f"no ledger at {os.fspath(source)!r}: {path} does not exist")


# ------------------------------------------------------------------------ streaming
def _numbered(path: Path | None, torn: list[TornLine] | None) -> Iterator[tuple[int, dict[str, Any]]]:
    """``(line number, event)`` for each complete event of the file, in file order."""
    if path is None:
        return
    with open(path, "rb") as fh:
        for number, offset, raw, terminated in scan_lines(fh):
            line = strip_line(raw, offset)
            if not line:
                continue
            try:
                event = parse_line(line)
            except ValueError as exc:
                if torn is not None:
                    torn.append(torn_line(number, offset, raw, terminated, str(exc)))
                continue
            yield number, event


def _kind_filter(kinds: Iterable[str] | str | None) -> frozenset[str] | None:
    if kinds is None:
        return None
    wanted = [kinds] if isinstance(kinds, str) else list(kinds)
    unknown = [k for k in wanted if not isinstance(k, str) or k not in TABLE_NAMES]
    if unknown:
        raise ValueError(
            f"unknown ledger event kind(s) {', '.join(map(repr, unknown))}. Valid kinds: {', '.join(EVENT_KINDS)}"
        )
    return frozenset(wanted)


def _events(path: Path | None, wanted: frozenset[str] | None, torn: list[TornLine] | None) -> Iterator[dict[str, Any]]:
    for _, event in _numbered(path, torn):
        if wanted is None or event["kind"] in wanted:
            yield event


def iter_events(source: str | os.PathLike[str], kinds: Iterable[str] | str | None = None,
                torn: list[TornLine] | None = None) -> Iterator[dict[str, Any]]:
    """Each complete event of a ledger as a dict, in file order, read one line at a time.

    ``source`` is a run id, a run folder or a ledger file (see :func:`ledger_path`).
    Every event is ``{"run_id", "seq", "ts", "kind", "payload"}``. ``kinds`` keeps
    only the named kinds. Lines that are not complete events are skipped, and when
    ``torn`` is a list each one is added to it as a :class:`TornLine` as the
    stream passes it.

    A run folder with no ledger file yet gives no events. A run that does not
    exist raises FileNotFoundError, and an unknown kind raises ValueError, both at
    the call. pandas is not imported.
    """
    return _events(_existing(source), _kind_filter(kinds), torn)


# ------------------------------------------------------------------------ tables
class LedgerTables:
    """One run's ledger as pandas tables. What :func:`load_ledger` returns.

    ``events`` has one row per event in file order with the columns ``run_id``,
    ``seq``, ``ts``, ``kind`` and ``payload`` (the payload as a dict). Each event
    kind has its own table under the name in ``TABLE_NAMES``: ``claims``,
    ``evidence``, ``verdicts``, ``forks``, ``assists``, ``audits``,
    ``tool_calls``, ``model_calls``, ``calibration_updates``, ``drift_alarms``
    and ``cycle_ends``. ``table(kind)`` returns the same frame by kind.

    ``torn_lines`` lists the lines that are not complete events and
    ``seq_breaks`` the places where seq does not count up by 1. ``problems``
    puts both into sentences for a report. ``path`` is the file that was read.
    """

    claims: "pd.DataFrame"
    evidence: "pd.DataFrame"
    verdicts: "pd.DataFrame"
    forks: "pd.DataFrame"
    assists: "pd.DataFrame"
    audits: "pd.DataFrame"
    tool_calls: "pd.DataFrame"
    model_calls: "pd.DataFrame"
    calibration_updates: "pd.DataFrame"
    drift_alarms: "pd.DataFrame"
    cycle_ends: "pd.DataFrame"

    def __init__(self, path: Path, events: "pd.DataFrame", tables: dict[str, "pd.DataFrame"],
                 torn_lines: Iterable[TornLine] = (), seq_breaks: Iterable[SeqBreak] = ()):
        self.path = path
        self.events = events
        self.torn_lines: tuple[TornLine, ...] = tuple(torn_lines)
        self.seq_breaks: tuple[SeqBreak, ...] = tuple(seq_breaks)
        self._tables = dict(tables)
        for kind, frame in self._tables.items():
            setattr(self, TABLE_NAMES[kind], frame)

    def table(self, kind: str) -> "pd.DataFrame":
        """The table for one event kind, such as ``table("verdict")``."""
        if kind not in self._tables:
            raise ValueError(f"unknown ledger event kind {kind!r}. Valid kinds: {', '.join(EVENT_KINDS)}")
        return self._tables[kind]

    @property
    def torn_tail(self) -> TornLine | None:
        """The file's last line when it is torn and has no newline after it, else None."""
        last = self.torn_lines[-1] if self.torn_lines else None
        return last if last is not None and not last.terminated else None

    @property
    def problems(self) -> list[str]:
        """What a report should say about this file's integrity. Empty for a clean ledger.

        A torn line is an event that never finished being written, so nothing that
        was reported as logged is missing. A break in seq or a second run id means
        events are missing, repeated or mixed, and ``load_ledger`` also warns about
        those.
        """
        torn = [line.describe() for line in self.torn_lines]
        return torn + _damage(self.seq_breaks, self.events["run_id"].tolist())

    def __repr__(self) -> str:
        counts = ", ".join(f"{TABLE_NAMES[kind]} {len(frame)}" for kind, frame in self._tables.items() if len(frame))
        torn = f", {len(self.torn_lines)} torn" if self.torn_lines else ""
        return f"LedgerTables({str(self.path)!r}: {len(self.events)} events{', ' + counts if counts else ''}{torn})"


def load_ledger(source: str | os.PathLike[str]) -> LedgerTables:
    """One run's ledger as typed pandas tables.

    ``source`` is a run id, a run folder or a ledger file (see :func:`ledger_path`).

    ``events`` holds every event: ``run_id`` (str), ``seq`` (int64), ``ts``
    (float64, Unix seconds), ``kind`` (str) and ``payload`` (the dict, as an
    object). Each kind's table starts with the 4 base columns ``run_id``,
    ``seq``, ``ts`` and ``kind``, followed by one column per top-level payload
    key in the order the keys first appear. A kind with no events gives an empty
    table with the 4 base columns.

    Payload columns are typed from their values: all booleans give ``boolean``,
    whole numbers ``Int64``, numbers with a fraction ``float64``, text ``str``.
    A missing key or a null is NA. A dict or a list stays a Python object, and so
    does a column that mixes types.

    A payload key with the name of a base column gets a ``payload_`` prefix, so
    an evidence item's own ``kind`` is the column ``payload_kind`` and a claim's
    own ``ts`` is ``payload_ts``. A key that equals a name already taken gets a
    numeric suffix (``payload_ts_2``). ``frame.attrs["renamed_payload_keys"]``
    maps each renamed key to its column.

    A line that is not a complete event (a torn last line after a crash) does not
    stop the load. The complete lines load and the line is listed in
    ``torn_lines``. A break in seq, or events of a second run in the file, is
    listed in ``seq_breaks`` and ``problems``, and the load warns about it with a
    LedgerWarning, because numbers built from such a file may be wrong. A run
    folder with no ledger file gives empty tables. A run that does not exist
    raises FileNotFoundError.
    """
    import pandas as pd

    path = _existing(source)
    torn: list[TornLine] = []
    breaks: list[SeqBreak] = []
    run_ids: list[str] = []
    seqs: list[int] = []
    stamps: list[float] = []
    kinds: list[str] = []
    payloads: list[dict[str, Any]] = []
    rows: dict[str, list[int]] = {kind: [] for kind in EVENT_KINDS}
    expected = 0
    for number, event in _numbered(path, torn):
        seq = event["seq"]
        if seq != expected:
            breaks.append(SeqBreak(number, expected, seq))
        expected = seq + 1
        rows[event["kind"]].append(len(seqs))
        run_ids.append(event["run_id"])
        seqs.append(seq)
        stamps.append(event["ts"])
        kinds.append(event["kind"])
        payloads.append(event["payload"])

    events = pd.DataFrame({
        "run_id": pd.Series(run_ids, dtype="str"),
        "seq": pd.Series(seqs, dtype="int64"),
        "ts": pd.Series(stamps, dtype="float64"),
        "kind": pd.Series(kinds, dtype="str"),
        "payload": pd.Series(payloads, dtype=object),
    })
    tables: dict[str, pd.DataFrame] = {}
    for kind, at in rows.items():
        base = {"run_id": [run_ids[i] for i in at], "seq": [seqs[i] for i in at], "ts": [stamps[i] for i in at],
                "kind": [kind] * len(at)}
        tables[kind] = _kind_table(base, [payloads[i] for i in at])
    damage = _damage(breaks, run_ids)
    if damage:
        more = f" (and {len(damage) - 1} more)" if len(damage) > 1 else ""
        warnings.warn(
            f"{path} is not a whole ledger: {damage[0]}{more}. Events are missing, repeated or from another run, "
            "so a number built from this file may be wrong.",
            LedgerWarning, stacklevel=2,
        )
    return LedgerTables(ledger_path(source) if path is None else path, events, tables, torn, breaks)


def _damage(breaks: Iterable[SeqBreak], run_ids: Iterable[str]) -> list[str]:
    """Sentences for what makes a file less than one whole run: breaks in seq, and events of more than 1 run."""
    out = [gap.describe() for gap in breaks]
    runs = list(dict.fromkeys(run_ids))
    if len(runs) > 1:
        out.append(f"the file holds events of {len(runs)} runs: {', '.join(map(repr, runs))}")
    return out


def _kind_table(base: dict[str, list[Any]], payloads: list[dict[str, Any]]) -> "pd.DataFrame":
    import pandas as pd

    keys: dict[str, None] = {}
    for payload in payloads:
        for key in payload:
            keys.setdefault(key)
    columns, renamed = _payload_columns(keys)
    data: dict[str, pd.Series] = {col: pd.Series(base[col], dtype=_BASE_DTYPES[col]) for col in BASE_COLUMNS}
    for key, col in columns.items():
        data[col] = _column([payload.get(key) for payload in payloads])
    frame = pd.DataFrame(data)
    frame.attrs["renamed_payload_keys"] = renamed
    return frame


def _payload_columns(keys: Iterable[str]) -> tuple[dict[str, str], dict[str, str]]:
    """The column for each payload key, and the keys stored under another name.

    A key that shadows a base column gets the ``payload_`` prefix first, since that
    name is the documented one. Every other key keeps its own name when it is free.
    A key whose name is already taken (``payload_seq`` next to ``seq``) gets the
    first free ``_2``, ``_3``, ... suffix, so no 2 keys ever share a column. The
    result keeps the order of ``keys``.
    """
    keys = list(keys)
    chosen = {key: _PAYLOAD_PREFIX + key for key in keys if key in BASE_COLUMNS}
    taken = set(BASE_COLUMNS) | set(chosen.values())
    for key in keys:
        if key not in chosen and key not in taken:
            chosen[key] = key
            taken.add(key)
    for key in keys:
        if key not in chosen:
            n = 2
            while f"{key}_{n}" in taken:
                n += 1
            chosen[key] = f"{key}_{n}"
            taken.add(chosen[key])
    columns = {key: chosen[key] for key in keys}
    renamed = {key: col for key, col in columns.items() if col != key}
    return columns, renamed


def _value_types(values: list[Any]) -> set[type]:
    seen: set[type] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, bool):
            seen.add(bool)
        elif isinstance(value, int):
            seen.add(int)
        elif isinstance(value, float):
            seen.add(float)
        elif isinstance(value, str):
            seen.add(str)
        else:
            seen.add(object)
    return seen


def _inferred_dtype(seen: set[type]) -> Any:
    if not seen or object in seen:
        return object
    if seen == {bool}:
        return "boolean"
    if seen == {int}:
        return "Int64"
    if seen <= {int, float}:
        return "float64"
    if seen == {str}:
        return "str"
    return object


def _column(values: list[Any]) -> "pd.Series":
    """A column typed from its values. A value pandas cannot hold in that type makes it an object column."""
    import pandas as pd

    dtype = _inferred_dtype(_value_types(values))
    try:
        return pd.Series(values, dtype=dtype)
    except (TypeError, ValueError, OverflowError):  # a whole number beyond int64, text Arrow cannot encode
        return pd.Series(values, dtype=object)
