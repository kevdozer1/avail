"""LedgerWriter: one run's append-only event log.

A run's ledger lives in ``AVAIL_DATA/runs/<run_id>/`` (``avail.paths.run_dir``),
or in the folder handed to the writer::

    <run folder>/
      ledger.jsonl   one compact JSON line per event: {run_id, seq, ts, kind, payload}
      .ledger.lock   held locked by the run's open LedgerWriter

seq. The first event of a run has seq 0 and every later event has the seq of
the event before it plus 1, with no gaps, across every reopen. A writer that
opens an existing ledger reads its last complete line and carries on from there.

One writer per run. A LedgerWriter holds an OS lock on ``.ledger.lock``
(``msvcrt.locking`` on Windows, ``fcntl.flock`` elsewhere) from the moment it
opens until ``close()``, and the OS drops the lock when the process dies. A
second LedgerWriter for the same run, in another process or in this one, raises
LedgerLockedError naming the run. Within a process, one writer is safe to share
between threads: appends are serialized and seq stays gapless. Readers take no
lock and can read a ledger while it is being written.

Flush and fsync policy (the one loopkit's trace writer uses). The ledger file
is opened unbuffered and each event is one ``write`` of one whole line, so
nothing ever waits in a Python buffer. When ``append`` returns, the line is in
the hands of the operating system. It survives Ctrl+C, an exception and a
killed process. ``fsync`` runs at ``close()`` and at ``sync()`` and never per
append, so a power cut or an OS crash can lose the last moments of a run that
was still open.

Torn lines. A crash in the middle of a write can leave part of a line at the
end of the file. The next writer finds it, reports it (``torn_tail`` and a
LedgerWarning), ends it with a newline and starts its own event on a fresh
line. Nothing written earlier is rewritten and nothing is merged into the
fragment. It stays in the file as a line that no reader takes for an event, and
the next event takes the seq it would have had, so complete lines stay gapless.

An append that raises. A full disk or a Ctrl+C can stop ``append`` partway. The
writer then looks at the file itself to learn what became of the line. When all
of it arrived, the event is in the ledger and its seq is used. When part of it
arrived, the fragment is ended as above. When none of it arrived, nothing
changed. In each case the next event gets the right seq, with no gap and no
duplicate.

Ctrl+C while a writer opens or closes never leaves the run locked with no
writer to close. An open that it interrupts gives back everything it took, and
a Ctrl+C that lands inside close() still releases the run. ``closed`` turns
True only once the run is released. If it is still False after a
KeyboardInterrupt, close() never got started, and calling it again finishes the
job.

append is not re-entrant. A signal handler that appends while the same thread
is inside append gets RuntimeError instead of a deadlock. In a handler, set a
flag and append from the main loop.

The lock, the Ctrl+C handling and the re-entry guard are ported from loopkit's
``traces/writer.py``, with its regression tests.
"""

from __future__ import annotations

import errno
import io
import math
import os
import threading
import time
import warnings
import weakref
from pathlib import Path
from typing import Any

import numpy as np

from .. import paths
from ..schemas.kinds import EVENT_KINDS
from .jsonsafe import dumps, to_jsonable
from .lines import LEDGER_FILE, LOCK_FILE, MAX_SEQ, LedgerWarning, TornLine, check_ledger_run_id, read_tail, short

if os.name == "nt":
    import msvcrt
else:
    import fcntl

__all__ = ["LedgerLockedError", "LedgerWriter"]

_KINDS = {kind: kind for kind in EVENT_KINDS}
# Windows byte-range locks are mandatory, so the lock sits past the holder note (the
# file's first line, which a refused writer reads to name the holder).
_LOCK_OFFSET = 1 << 20
_LOCK_HELD_ERRNOS = {errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK, errno.EDEADLK}
_REENTRANT_MARK = "_avail_reentrant"


class LedgerLockedError(RuntimeError):
    """Another LedgerWriter has this run open. One writer per run."""


# ------------------------------------------------------------------------ the run lock
# Ctrl+C reaches Python code right after a C call returns or as a Python function
# starts, so it can land between any 2 steps of taking or giving back a run. Each
# step below leaves a state that _RunLock.release() undoes, and a release() that Ctrl+C
# cut short is finished by calling it again.
_OPEN_RUNS: dict[str, "_RunLock"] = {}
# Re-entrant: release() also runs from a finalizer or a signal handler, possibly on
# the thread that holds this lock.
_OPEN_RUNS_LOCK = threading.RLock()


class _RunLock:
    """This process's claim on one run: its registry entry plus the OS lock on ``.ledger.lock``.

    Built holding nothing, so its owner has it in hand before acquire() takes anything.
    release() gives back whatever part of the claim is still held.
    """

    def __init__(self, run_dir: Path, run_id: str):
        self.key = os.path.normcase(str(run_dir))
        self.run_dir = run_dir
        self.path = run_dir / LOCK_FILE
        self.run_id = run_id
        self._fh: io.FileIO | None = None  # the open lock file: closing it drops the OS lock
        self._releasing = False  # a release() is under way in this thread (see release)

    @property
    def held(self) -> bool:
        """True while the lock file is open or the registry lists this claim."""
        fh = self._fh
        return (fh is not None and not fh.closed) or _OPEN_RUNS.get(self.key) is self

    def acquire(self) -> None:
        """Take the OS lock, then the registry entry. LedgerLockedError when another writer has the run.

        When this raises, Ctrl+C included, whatever it took is recorded here for the
        owner's release() to give back.
        """
        with _OPEN_RUNS_LOCK:
            if self.key in _OPEN_RUNS:
                raise LedgerLockedError(
                    f"run {self.run_id!r} already has an open LedgerWriter in this process ({self.run_dir}). "
                    "One writer per run: share that writer, or close it before opening another."
                )
            self._fh = _lock_file(self.path, self.run_id)
            _OPEN_RUNS[self.key] = self

    def release(self) -> None:
        """Give back what this claim holds: close the lock file, then drop the registry entry.

        Idempotent, and safe to repeat after a call that Ctrl+C cut short. Each step
        looks at what is still held, and neither can be left half done. FileIO.close
        marks the file closed in the same C call that closes the descriptor, so a
        repeat never closes a descriptor twice, and the registry check runs no Python
        code, so no signal handler can run between it and the pop. The 2 steps sit
        in separate finally blocks, so an interrupt in one still runs the other.

        A call that re-enters a release already under way (a signal handler that closes
        the writer) returns at once and leaves the work to the outer call. Otherwise the
        handler could close the lock file, open another writer on the freed descriptor
        number, and the outer call would then unlock that new writer's lock.
        """
        if self._releasing:
            return
        try:
            self._releasing = True
            with _OPEN_RUNS_LOCK:
                try:
                    fh = self._fh
                    if fh is not None and not fh.closed:
                        try:
                            _unlock_file(fh)
                        finally:
                            fh.close()
                finally:
                    if self.key in _OPEN_RUNS and _OPEN_RUNS[self.key] is self:
                        _OPEN_RUNS.pop(self.key, None)
        finally:
            self._releasing = False


def _open_lock_file(path: Path) -> io.FileIO:
    """``path`` open for reading and writing, created when missing, never truncated.

    io.FileIO opens the file and takes the descriptor in one C call, so no Ctrl+C can
    fall between the 2 steps and leave a descriptor open that nothing will close.
    """
    try:
        return io.FileIO(path, "r+")
    except FileNotFoundError:
        pass
    try:
        return io.FileIO(path, "x+")
    except FileExistsError:  # another writer created it in between
        return io.FileIO(path, "r+")


def _lock_file(path: Path, run_id: str) -> io.FileIO:
    """``path`` open, holding its exclusive OS lock. LedgerLockedError when someone else holds it.

    Every way out but the return closes the file first, Ctrl+C included, so the lock is
    never left held by a descriptor that nothing owns.
    """
    fh = _open_lock_file(path)
    try:
        fd = fh.fileno()
        if os.name == "nt":
            os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        fh.close()
        if exc.errno not in _LOCK_HELD_ERRNOS:
            raise
        raise LedgerLockedError(
            f"run {run_id!r} is already open in another LedgerWriter{_holder(path)}: {path} is locked. "
            "One writer per run: wait for that writer to close, or stop it."
        ) from None
    except BaseException:
        fh.close()
        raise
    try:  # the holder note a refused writer reads
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, f"pid {os.getpid()}".ljust(31).encode("ascii") + b"\n")
    except OSError:
        pass
    except BaseException:  # Ctrl+C: give the lock back before passing the interrupt on
        fh.close()
        raise
    return fh


def _unlock_file(fh: io.FileIO) -> None:
    """Drop the OS lock ``fh`` holds. Best effort: closing ``fh`` drops it as well."""
    try:
        fd = fh.fileno()
        if os.name == "nt":
            os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)
    except (OSError, ValueError):  # not locked, or fh closed meanwhile (ValueError)
        pass


def _holder(path: Path) -> str:
    """Who holds the lock, from the lock file's note: ' in this process', ' (process <pid>)' or ''."""
    try:
        words = path.read_bytes()[:64].decode("ascii", "replace").split()
    except OSError:
        return ""
    if len(words) >= 2 and words[0] == "pid" and words[1].isdigit():
        pid = int(words[1])
        return " in this process" if pid == os.getpid() else f" (process {pid})"
    return ""


# ------------------------------------------------------------------------ helpers
def _run_folder(run_id: str, folder: str | os.PathLike[str] | None) -> Path:
    """The run's folder as an absolute path, created.

    Without ``folder`` it is ``avail.paths.run_dir(run_id)``. A folder that Windows
    maps to a device (``<anything>/nul``) is refused before anything is created: a
    ledger there reports success while its events go nowhere.
    """
    path = paths.run_dir(run_id, create=False) if folder is None else Path(folder)
    resolved = os.path.abspath(path)
    if os.name == "nt" and resolved.startswith("\\\\.\\"):
        raise ValueError(
            f"ledger folder {str(path)!r} resolves to the Windows device {resolved}. Events written there are "
            "lost. Pick another folder."
        )
    if folder is None:
        paths.run_dir(run_id)  # created the way avail.paths creates a run folder
    else:
        Path(resolved).mkdir(parents=True, exist_ok=True)
    return Path(resolved)


def _payload_dict(value: Any, what: str) -> dict[str, Any]:
    """``value`` (a dict, or a record with ``model_dump``) as a JSON-safe dict. ValueError otherwise."""
    if value is None:
        raise ValueError(f"{what} is None. Pass a dict, or a record with model_dump(mode='json').")
    try:
        out = to_jsonable(value)
    except RecursionError:
        raise ValueError(f"{what} is nested too deeply or contains itself") from None
    except Exception as exc:  # a pydantic serializer, a failing __str__, ...
        raise ValueError(f"{what} is not JSON serializable: {type(exc).__name__}: {exc}") from exc
    if not isinstance(out, dict):
        raise ValueError(
            f"{what} must be a dict or a record with model_dump(mode='json'), got {type(value).__name__}"
        )
    return out


def _timestamp(ts: Any) -> float:
    if isinstance(ts, (bool, np.bool_)) or not isinstance(ts, (int, float, np.integer, np.floating)):
        raise ValueError(f"ts must be Unix time in seconds, got {short(ts)}")
    try:
        value = float(ts)
    except OverflowError:
        value = math.inf
    if not math.isfinite(value):
        raise ValueError(f"ts must be a finite number, got {short(ts)}")
    return value


def _write_all(fh: io.FileIO, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = fh.write(view)
        if not written:
            raise OSError(f"no bytes written to {fh.name}")
        view = view[written:]


def _read_from(path: Path, start: int) -> bytes:
    """The bytes of ``path`` from byte ``start`` to its end."""
    with open(path, "rb") as fh:
        fh.seek(start)
        return fh.read()


def _reentered(action: str, run_id: str, instead: str) -> RuntimeError:
    """The RuntimeError for a call a signal handler made while this thread was inside the writer,
    marked so that append passes it on unchanged wherever it surfaces."""
    error = RuntimeError(
        f"{action} called re-entrantly on run {run_id!r}: a signal handler ran while this thread was inside "
        f"the writer. Set a flag in the handler and {instead} from the main loop."
    )
    setattr(error, _REENTRANT_MARK, True)
    return error


# ---------------------------------------------------------------------- the writer
class LedgerWriter:
    """Append-only JSONL events for one run.

    ``LedgerWriter(run_id)`` writes ``AVAIL_DATA/runs/<run_id>/ledger.jsonl``.
    ``LedgerWriter(run_id, folder=path)`` writes ``<path>/ledger.jsonl``. Use it as
    a context manager, or call ``close()``. Raises LedgerLockedError when another
    LedgerWriter, in any process, has the run open, and ValueError when the
    ledger in the folder belongs to another run id.

    Attributes: ``run_id``, ``dir`` (the run folder), ``path`` (ledger.jsonl),
    ``lock_path``, ``next_seq`` (the seq the next event gets), ``resumed`` (the
    file already held events), ``closed`` and ``torn_tail`` (the torn last line
    found on open, as a TornLine, else None).
    """

    def __init__(self, run_id: str, folder: str | os.PathLike[str] | None = None):
        self.run_id = check_ledger_run_id(run_id)  # checked before anything is created
        self.dir = _run_folder(run_id, folder)
        self.path = self.dir / LEDGER_FILE
        self.lock_path = self.dir / LOCK_FILE
        # An RLock only so that a signal handler's append on the thread inside append gets
        # past the lock and is refused by the _busy check instead of deadlocking.
        self._lock = threading.RLock()
        self._busy = False
        self._fh: io.FileIO | None = None
        self._next_seq = 0
        self._end = 0  # the file's size after the last line known to be whole
        self._pending: tuple[int, int, bytes] | None = None  # (seq, offset, line) of a write not yet known to be whole
        self.resumed = False
        self.torn_tail: TornLine | None = None
        self._run_lock = _RunLock(self.dir, self.run_id)  # holds nothing yet: acquire() takes the run
        try:
            self._release_run = weakref.finalize(self, self._run_lock.release)  # frees a writer dropped unclosed
            self._run_lock.acquire()  # one writer per run, held until close()
            tail = read_tail(self.path)
            if tail.last is not None and tail.last["run_id"] != self.run_id:
                raise ValueError(
                    f"{self.path} holds events of run {tail.last['run_id']!r} and cannot take events of run "
                    f"{self.run_id!r}. Give each run its own folder."
                )
            # Unbuffered: each event is one write straight to the OS, so a failed write
            # never leaves bytes in a Python buffer to be flushed later.
            self._fh = open(self.path, "ab", buffering=0)  # held open until close()
            if tail.needs_newline:  # end a crash's partial last line so the next event starts on its own line
                _write_all(self._fh, b"\n")
            self._end = os.fstat(self._fh.fileno()).st_size
            if tail.last is not None:
                self._next_seq = tail.last["seq"] + 1
                self.resumed = True
            self.torn_tail = tail.torn
            if tail.torn is not None:
                warnings.warn(
                    f"the ledger of run {self.run_id!r} ended in a torn line (line {tail.torn.number}, "
                    f"{tail.torn.length} bytes, left by a write that was cut short). It stays in the file, the "
                    f"readers skip and report it, and the next event starts on a fresh line: {self.path}",
                    LedgerWarning, stacklevel=2,
                )
        except BaseException:
            try:
                if self._fh is not None:
                    self._fh.close()
                    self._fh = None
            finally:
                # Twice: a Ctrl+C that stops the first call, even at its first line, cannot stop both.
                try:
                    self._run_lock.release()  # gives back whatever acquire() took, however far it got
                finally:
                    self._run_lock.release()
            raise

    # ------------------------------------------------------------------ state
    @property
    def next_seq(self) -> int:
        """The seq the next event gets."""
        if self._pending is not None:  # an append was cut short and what it left is not settled yet
            with self._lock:
                if not self._busy:
                    try:
                        self._busy = True
                        self._settle_quietly(self._fh)
                    finally:
                        self._busy = False
        return self._next_seq

    @property
    def closed(self) -> bool:
        """True once close() has finished: the ledger file is closed and the run released."""
        return self._fh is None and not self._run_lock.held

    def __repr__(self) -> str:
        return f"LedgerWriter(run_id={self.run_id!r}, next_seq={self._next_seq}, dir={str(self.dir)!r})"

    # ------------------------------------------------------------------ events
    def append(self, kind: str, payload: Any, ts: float | None = None) -> int:
        """Append one event as a JSON line and return its seq.

        ``kind`` is one of ``avail.schemas.kinds.EVENT_KINDS``. ``payload`` is a
        dict, or a record with ``model_dump(mode='json')`` such as the pydantic
        types in ``avail.schemas``. ``ts`` is Unix time in seconds and defaults to
        now.

        Raises ValueError for an unknown ``kind`` (the message lists the valid
        ones), a payload that is not a dict or cannot be made JSON safe, a ``ts``
        that is not a finite number, or a closed writer. Such an event writes
        nothing and uses no seq. When the write itself fails or is interrupted (a
        full disk, Ctrl+C), the writer works out from the file whether the line
        arrived, so the next event still gets the right seq.

        Not re-entrant: from a signal handler that interrupted append on the same
        thread, append raises RuntimeError instead of deadlocking. Set a flag in
        the handler and append from the main loop.
        """
        name = _KINDS.get(str.__str__(kind)) if isinstance(kind, str) else None  # the plain str of a str enum
        if name is None:
            raise ValueError(f"unknown ledger event kind {kind!r}. Valid kinds: {', '.join(EVENT_KINDS)}")
        when = None if ts is None else _timestamp(ts)
        what = f"{name} payload for run {self.run_id!r}"
        body = _payload_dict(payload, what)
        with self._lock:
            if self._busy:  # only the append (or close, sync) this thread is already inside sets it
                raise _reentered("append", self.run_id, "append")
            try:
                self._busy = True
                fh = self._fh
                if fh is None:
                    raise ValueError(f"LedgerWriter for run {self.run_id!r} is closed")
                if self._pending is not None:  # an earlier append was cut short: settle it before writing more
                    self._settle(fh)
                if when is None:  # stamped under the lock, so default timestamps follow seq order across threads
                    when = time.time()
                seq = self._next_seq
                if seq > MAX_SEQ:  # readers drop such a line, so never write one
                    raise ValueError(f"run {self.run_id!r} has no seq left: the next would be {seq}, beyond int64")
                try:
                    event = {"run_id": self.run_id, "seq": seq, "ts": when, "kind": name, "payload": body}
                    line = dumps(event).encode("ascii") + b"\n"
                except Exception as exc:
                    if getattr(exc, _REENTRANT_MARK, False):  # a signal handler's refused call: the payload is fine
                        raise
                    raise ValueError(f"{what} is not JSON serializable: {type(exc).__name__}: {exc}") from exc
                start = self._end
                end = start + len(line)
                self._pending = (seq, start, line)  # until it is cleared, only the file knows whether the line is whole
                _write_all(fh, line)
                # The line is whole. Each of the 3 assignments is absolute, so a Ctrl+C
                # between any 2 of them leaves a state that _settle() puts right.
                self._next_seq = seq + 1
                self._end = end
                self._pending = None
                fh.flush()
            finally:
                try:
                    if self._pending is not None:  # the write failed or was interrupted
                        self._settle_quietly(self._fh)
                finally:
                    self._busy = False
        return seq

    def _settle(self, fh: io.FileIO) -> None:
        """Find out what an interrupted write left in the file, and set the writer's state to match.

        ``_pending`` holds the seq, the offset and the bytes of a line whose write
        did not finish cleanly. The bytes in the file from that offset on decide.
        The whole line: the event is in the ledger and its seq is used. The line
        without its newline: the newline is written now, which completes it.
        Anything else: a fragment, which gets a newline so that nothing is merged
        into it, and its seq goes to the next event. Nothing at all: nothing
        happened.

        Every step is absolute, so a call that Ctrl+C cuts short is repaired by
        running it again. ``_pending`` is cleared last.
        """
        pending = self._pending
        if pending is None:
            return
        seq, start, line = pending
        tail = _read_from(self.path, start)
        landed = tail == line or tail == line[:-1]
        if tail and not tail.endswith(b"\n"):
            _write_all(fh, b"\n")
        self._next_seq = seq + 1 if landed else seq
        self._end = os.fstat(fh.fileno()).st_size
        self._pending = None

    def _settle_quietly(self, fh: io.FileIO | None) -> None:
        """_settle() where an error must not replace the one already on its way out.

        Whatever it cannot settle stays pending, and the next append settles it or
        raises the reason.
        """
        if fh is None or fh.closed:
            return
        try:
            self._settle(fh)
        except Exception:
            pass

    def sync(self) -> None:
        """Ask the OS to put every appended line on the disk now (``fsync``).

        ``append`` hands each line to the OS, which survives the death of the
        process. This call is for the moments that should also survive a power
        cut, such as the end of a cycle. ``close()`` does the same once.
        """
        with self._lock:
            if self._busy:
                raise _reentered("sync", self.run_id, "sync")
            try:
                self._busy = True
                fh = self._fh
                if fh is None:
                    raise ValueError(f"LedgerWriter for run {self.run_id!r} is closed")
                fh.flush()
                os.fsync(fh.fileno())
            finally:
                self._busy = False

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        """Flush, fsync and close the ledger file and release the run lock. Safe to call twice.

        A Ctrl+C that lands inside close() still releases the run. One that stops it
        before it starts leaves the writer open (``closed`` stays False), and calling
        close() again finishes the job.
        """
        with self._lock:
            if self._busy:
                raise _reentered("close", self.run_id, "close the writer")
            try:
                self._busy = True
                fh, self._fh = self._fh, None
                if fh is not None:
                    try:
                        if self._pending is not None:  # end a fragment the last append left
                            self._settle_quietly(fh)
                        fh.flush()
                        os.fsync(fh.fileno())
                    except OSError:
                        pass
                    finally:
                        fh.close()
            finally:
                self._busy = False
                if self._fh is None:  # the ledger file is let go: let go of the run as well
                    # Called directly, because the finalizer is spent once it is called, whether or not
                    # it finished. Twice: a Ctrl+C that stops the first call, even at its first line,
                    # cannot stop both.
                    try:
                        self._run_lock.release()
                    finally:
                        self._run_lock.release()
                    self._release_run.detach()  # only now: until the run is released, dropping the writer frees it

    def __enter__(self) -> "LedgerWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
