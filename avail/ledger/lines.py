"""The ledger's line format, and the one rule for what a complete event is.

A ledger file is text with one event per line. Each line is a compact JSON
object with exactly 5 keys, written in this order::

    {"run_id":"e3_arm1","seq":12,"ts":1794000000.25,"kind":"verdict","payload":{...}}

``run_id`` names the run, ``seq`` counts the run's events from 0 with no gaps,
``ts`` is Unix time in seconds, ``kind`` is one of
``avail.schemas.kinds.EVENT_KINDS`` and ``payload`` is a JSON object.

:func:`parse_line` is the single test for "this line is a complete event". The
writer uses it to find where a run left off and both readers use it to decide
what to load, so a writer and a reader never disagree about a line. A line that
fails it is a torn line. It stays in the file, the readers skip it, and they
report it as a :class:`TornLine`.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, BinaryIO

from .. import paths
from ..schemas.kinds import EVENT_KINDS
from .jsonsafe import to_jsonable

__all__ = [
    "FIELDS", "LEDGER_FILE", "LOCK_FILE", "MAX_SEQ", "LedgerWarning", "Tail", "TornLine", "check_ledger_run_id",
    "is_run_id", "parse_line", "read_tail", "scan_lines", "short", "strip_line", "torn_line",
]


LEDGER_FILE = "ledger.jsonl"
LOCK_FILE = ".ledger.lock"
FIELDS = ("run_id", "seq", "ts", "kind", "payload")
MAX_SEQ = 2**63 - 1  # load_ledger keeps seq in an int64 column, so a line with a larger seq is not an event

_FIELD_SET = frozenset(FIELDS)
_KIND_SET = frozenset(EVENT_KINDS)
_UTF8_BOM = b"\xef\xbb\xbf"
_PREVIEW_BYTES = 120
_SURROGATE_ESCAPE = re.compile(r"\\u[dD][89a-fA-F]")  # \ud800 to \udfff as json.dumps writes them
# Names Windows can map to a device in any folder, in any case, with or without an extension.
_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"} | {f"{port}{n}" for port in ("COM", "LPT") for n in "0123456789"}
)


class LedgerWarning(UserWarning):
    """Something in a ledger file that a person should know about.

    A writer warns when the ledger it opens ends in a torn line. ``load_ledger``
    warns when seq does not count up by 1 or the file holds events of several runs.
    """


# ------------------------------------------------------------------------ run ids
def check_ledger_run_id(run_id: str) -> str:
    """``run_id`` when a ledger can be kept under it, else ValueError.

    The rule is ``avail.paths.check_run_id`` plus 2 refusals that only matter to
    the ledger. A name ending in ``.jsonl`` is refused because the readers take
    such a string for the path of a ledger file. A Windows device name (NUL, CON,
    PRN, AUX, COM0 to COM9, LPT0 to LPT9, with or without an extension) is refused
    because Windows can map it to a device, and events written there are lost.
    """
    paths.check_run_id(run_id)
    if run_id.lower().endswith(".jsonl"):
        raise ValueError(
            f"run id {run_id!r} ends in .jsonl, which the ledger readers take for the path of a ledger file. "
            "Drop the suffix."
        )
    stem = run_id.split(".", 1)[0].upper()
    if stem in _DEVICE_NAMES:
        raise ValueError(
            f"run id {run_id!r} is the Windows device name {stem}, which Windows will not use as a folder. "
            "Pick another name."
        )
    return run_id


def is_run_id(value: str) -> bool:
    """True when ``value`` passes :func:`check_ledger_run_id`."""
    try:
        check_ledger_run_id(value)
    except ValueError:
        return False
    return True


# ------------------------------------------------------------------------ one line
@dataclass(frozen=True)
class TornLine:
    """A line of a ledger file that is not a complete event.

    A write cut short by a crash leaves one at the end of the file. The next
    writer ends it with a newline and starts its event on a fresh line, so the
    fragment stays where it is and nothing is ever merged into it.

    ``number`` is the 1-based line number, ``offset`` the byte offset of the
    line's first byte and ``length`` its size in bytes without the newline.
    ``reason`` says why no reader takes it for an event. ``terminated`` is False
    when the file ends inside this line, which is what a crash in the middle of a
    write (or an append still in flight) looks like. ``text`` holds the first 120
    bytes as ASCII, for a report.
    """

    number: int
    offset: int
    length: int
    reason: str
    terminated: bool
    text: str

    def describe(self) -> str:
        """One sentence for a report."""
        ending = "" if self.terminated else ", no newline after it"
        return (
            f"line {self.number} is not a complete event ({self.length} bytes at byte {self.offset}{ending}): "
            f"{self.reason}"
        )


def torn_line(number: int, offset: int, raw: bytes, terminated: bool, reason: str) -> TornLine:
    """The :class:`TornLine` for ``raw``, the bytes of line ``number`` without its newline."""
    text = raw.strip()[:_PREVIEW_BYTES].decode("ascii", "backslashreplace")
    return TornLine(number=number, offset=offset, length=len(raw), reason=reason, terminated=terminated, text=text)


def strip_line(raw: bytes, offset: int) -> bytes:
    """``raw`` without surrounding whitespace, and without a UTF-8 byte order mark at the start of the file.

    A file rewritten by PowerShell 5.1 can start with the mark and end its lines
    with a carriage return. Neither is part of an event.
    """
    line = raw.strip()
    if offset == 0 and line.startswith(_UTF8_BOM):
        line = line[len(_UTF8_BOM):].strip()
    return line


def _null(_name: str) -> None:
    """NaN, Infinity and -Infinity are not JSON. They load as None, which is how the writer stores them."""
    return None


def _json_type(value: Any) -> str:
    names = {list: "array", str: "string", bool: "boolean", int: "number", float: "number", type(None): "null"}
    return names.get(type(value), type(value).__name__)


def parse_line(line: bytes | str) -> dict[str, Any]:
    """One ledger line as an event dict: the rule every ledger reader applies.

    The result has the keys ``run_id``, ``seq``, ``ts``, ``kind`` and ``payload``
    in that order, with ``ts`` as a float. Raises ValueError saying what is wrong
    when the line is not UTF-8 JSON, is not an object with exactly those 5 keys,
    or holds a value of the wrong type: ``run_id`` a non-empty string, ``seq`` a
    whole number from 0 to 2**63 - 1, ``ts`` a finite number, ``kind`` one of the
    event kinds, ``payload`` an object.

    NaN and Infinity tokens load as None. A lone surrogate escape such as
    ``\\ud83d``, which strict parsers reject, loads as U+FFFD.
    """
    if isinstance(line, str):
        text = line
    else:
        try:
            text = bytes(line).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"not UTF-8 text ({exc.reason} at byte {exc.start})") from None
    try:
        obj = json.loads(text, parse_constant=_null)
    except RecursionError:
        raise ValueError("nested too deeply to parse") from None
    except ValueError as exc:  # json.JSONDecodeError, or a number with too many digits
        raise ValueError(f"not valid JSON ({exc})") from None
    if type(obj) is not dict:
        raise ValueError(f"a JSON {_json_type(obj)} where an object with the keys {', '.join(FIELDS)} belongs")
    if obj.keys() != _FIELD_SET:
        missing = [key for key in FIELDS if key not in obj]
        extra = sorted(key for key in obj if key not in _FIELD_SET)
        parts = ([f"missing {', '.join(missing)}"] if missing else []) + (
            [f"unknown {', '.join(map(repr, extra))}"] if extra else [])
        raise ValueError(f"an event has exactly the keys {', '.join(FIELDS)} ({' and '.join(parts)})")
    if _SURROGATE_ESCAPE.search(text):  # an escape is the only way a surrogate gets into a JSON string
        try:
            obj = to_jsonable(obj)
        except RecursionError:
            raise ValueError("nested too deeply to parse") from None
    run_id, seq, ts, kind, payload = (obj[key] for key in FIELDS)
    if type(run_id) is not str or not run_id:
        raise ValueError(f"run_id must be a non-empty string, got {short(run_id)}")
    if type(seq) is not int or not 0 <= seq <= MAX_SEQ:
        raise ValueError(f"seq must be a whole number from 0 to 2**63 - 1, got {short(seq)}")
    if type(ts) not in (int, float):
        raise ValueError(f"ts must be Unix time in seconds, got {short(ts)}")
    try:
        when = float(ts)
    except OverflowError:
        when = math.inf
    if not math.isfinite(when):
        raise ValueError(f"ts must be a finite number, got {short(ts)}")
    if type(kind) is not str or kind not in _KIND_SET:
        raise ValueError(f"unknown event kind {short(kind)}. Valid kinds: {', '.join(EVENT_KINDS)}")
    if type(payload) is not dict:
        raise ValueError(f"payload must be a JSON object, got a JSON {_json_type(payload)}")
    return {"run_id": run_id, "seq": seq, "ts": when, "kind": kind, "payload": payload}


def short(value: Any) -> str:
    """``repr(value)``, clipped, for an error message."""
    try:
        text = repr(value)
    except Exception:  # an int with too many digits to print
        text = f"<{type(value).__name__}>"
    return text if len(text) <= 60 else text[:57] + "..."


# ------------------------------------------------------------------------ whole files
def scan_lines(fh: BinaryIO) -> Iterator[tuple[int, int, bytes, bool]]:
    """``(number, offset, line, terminated)`` for each line of an open ledger file.

    Lines are split on ``\\n`` only. ``line`` comes without its newline, ``number``
    counts from 1 and ``offset`` is the byte offset of the line's first byte.
    ``terminated`` is False for a last line with no newline after it.
    """
    offset = 0
    for number, raw in enumerate(fh, 1):
        terminated = raw.endswith(b"\n")
        yield number, offset, (raw[:-1] if terminated else raw), terminated
        offset += len(raw)


@dataclass(frozen=True)
class Tail:
    """The end of a ledger file, as a writer needs it to resume.

    ``last`` is the last complete event (None when the file holds none), ``torn``
    the file's unterminated last line when it is not a complete event, and
    ``needs_newline`` says the file does not end with a newline.
    """

    last: dict[str, Any] | None = None
    torn: TornLine | None = None
    needs_newline: bool = False


def read_tail(path: str | os.PathLike[str]) -> Tail:
    """Where the ledger at ``path`` left off. A missing or empty file gives an empty :class:`Tail`.

    Reads the file from its end, one block at a time, back to the last complete
    event, so resuming a long run does not read the whole ledger. A torn last
    line is reported with its line number, which costs one pass that counts
    newlines and happens only after a crash.
    """
    try:
        fh = open(path, "rb")
    except FileNotFoundError:
        return Tail()
    with fh:
        size = fh.seek(0, os.SEEK_END)
        if size == 0:
            return Tail()
        fh.seek(size - 1)
        needs_newline = fh.read(1) != b"\n"
        last: dict[str, Any] | None = None
        fragment: tuple[int, bytes, str] | None = None
        lines = _lines_from_end(fh, size)
        try:
            for offset, raw, terminated in lines:
                line = strip_line(raw, offset)
                if not line:
                    continue
                try:
                    last = parse_line(line)
                    break
                except ValueError as exc:
                    if not terminated:  # the file ends inside this line
                        fragment = (offset, raw, str(exc))
        finally:
            lines.close()  # here, where a Ctrl+C during the close reaches the caller instead of being swallowed
        torn = None
        if fragment is not None:
            offset, raw, reason = fragment
            torn = torn_line(1 + _count_newlines(fh, offset), offset, raw, False, reason)
    return Tail(last=last, torn=torn, needs_newline=needs_newline)


def _lines_from_end(fh: BinaryIO, size: int, block: int = 1 << 16) -> Iterator[tuple[int, bytes, bool]]:
    """``(offset, line, terminated)`` for each line of a file of ``size`` bytes, last line first.

    Only the segment after the file's last newline comes with ``terminated``
    False. It is empty when the file ends with a newline.
    """
    pos = size
    pieces: list[bytes] = []  # the rest of the line being put together, in the order read (latest in the file first)
    terminated = False
    while pos > 0:
        step = min(block, pos)
        pos -= step
        fh.seek(pos)
        chunk = fh.read(step)
        parts = chunk.split(b"\n")
        if len(parts) == 1:  # no newline in this block: all of it belongs to the line being put together
            pieces.append(chunk)
            continue
        start = pos + len(chunk) - len(parts[-1])
        yield start, parts[-1] + b"".join(reversed(pieces)), terminated
        terminated = True
        newline = start - 1  # the newline that ends the part before
        for part in reversed(parts[1:-1]):
            yield newline - len(part), part, True
            newline -= len(part) + 1
        pieces = [parts[0]]
    yield 0, b"".join(reversed(pieces)), terminated


def _count_newlines(fh: BinaryIO, upto: int, block: int = 1 << 20) -> int:
    """The number of newlines in the first ``upto`` bytes of an open file."""
    fh.seek(0)
    count, left = 0, upto
    while left > 0:
        chunk = fh.read(min(block, left))
        if not chunk:
            break
        count += chunk.count(b"\n")
        left -= len(chunk)
    return count
