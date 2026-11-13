"""The spend ledger and its caps.

The ledger is append-only JSONL at ``AVAIL_DATA/spend_ledger.jsonl``, shared by
every session and every process::

    {"type": "reserve", "id", "ts", "session", "model_key", "usd": projected, "request_hash", "tags"}
    {"type": "settle",  "id", "ts", "usd": actual, "usd_source"}

Every HTTP attempt of a paid call reserves its projected worst case before it is
sent and is settled at the measured cost afterwards (the router does this per
attempt, so a retry or a JSON-mode fallback is reserved too). Spent is the sum
over reserves of the settled amount when there is one, else the reserved amount.
So the actual cost replaces the reservation once it is known, and an attempt whose
cost is never learned (a timeout, a dropped connection, a 5xx, a 200 without
usage) stays counted at its worst case. The ledger is AVAIL's own record. It never
seeds from a provider's usage page, because the key may be shared with other
projects.

Caps are config. They come from ``spend_caps_usd`` in ``config/experiment.yaml``
and no code path changes one at run time:

* ``build_total`` is the total cap. It bounds the sum over every session of the
  build.
* Every other key is a session with its own cap (``S1``, ``S3``,
  ``S4_without_ok``, ``full_judge_run``). A reservation names its session.
* ``full_judge_run`` is the full judge run that PLAN.md allows only after Kevin's
  OK. Its cap is above the build total, so it is counted on its own, outside the
  build total, and a paid call in it needs ``approved=True``.

A paid reservation is refused, before any network I/O, when it would cross its
session's cap, the caller's cap for that model, or the total cap, and also when
one of those is already reached, even for a $0 projection. With no session every
paid call is refused. A session name that is not in the config is an error. Only
the FakeModel at a $0 projection is free. The router marks every other call paid.

Records are written as ASCII JSON (``ensure_ascii``), 1 per line, and read back by
splitting the raw bytes on ``\\n`` only, so no character in a label can split a
record. Session names, model keys and phase labels must be single-line printable
text. A settle with no matching reserve is counted and reported (``summary()``
and a warning), never silently ignored.

The check and the append happen under 1 cross-process lock: an OS lock on a
persistent lock file (``msvcrt.locking`` on Windows, ``fcntl.flock`` elsewhere).
The OS releases it when the holding process dies, so there is no stale-lock
breaking and the lock file is never deleted.

Ported from loopkit's ``router/ledger.py``. loopkit kept its caps and its active
session in a file that any caller could rewrite. Here the caps are read from the
config and the session travels with each reservation.
"""

from __future__ import annotations

import json
import math
import os
import time
import uuid
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from .. import config, paths
from .errors import RouterError, SessionNotApproved, SpendCapExceeded, UnknownSessionError

#: the key of ``spend_caps_usd`` that holds the total cap. Every other key is a session
TOTAL_CAP_KEY = "build_total"
#: sessions counted outside the build total. A paid call in one needs Kevin's OK (``approved=True``)
APPROVAL_SESSIONS = ("full_judge_run", "S4b_judges")   # S4b_judges: Kevin's OK of 2026-11-11, up to $30
APPROVAL_PREFIXES = ("E3_vlm_arm_", "FIG_")   # FIG_: the figure run's judge arm, Kevin's $6 of 2026-11-13
# E3_vlm_arm_<variant>: the E3 final run's judge, 1 session per variant, each counted on its own. Their caps are 0
# until Kevin sets them, before he starts E3
LOCK_TIMEOUT_S = 120.0
_EPS = 1e-12
_ANY = object()
_ORPHANS_WARNED: dict[str, int] = {}  # ledger path -> orphan count already warned about by reserve()

if os.name == "nt":  # pragma: no cover - platform branch
    import msvcrt

    def _try_lock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)  # 1 byte at offset 0. Raises OSError while another holds it

    def _unlock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
else:  # pragma: no cover - platform branch
    import fcntl

    def _try_lock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


# --------------------------------------------------------------------------- labels
def is_label(value: Any) -> bool:
    """True for single-line printable text (no line or paragraph separators, no control characters)."""
    return isinstance(value, str) and value.isprintable()


def check_label(value: Any, what: str, *, allow_empty: bool = False) -> str:
    """``value`` when it is single-line printable text, else ValueError."""
    if not is_label(value) or (not allow_empty and not value.strip()):
        raise ValueError(f"{what} must be {'' if allow_empty else 'non-empty '}single-line printable text "
                         f"(no line breaks, separators or control characters), got {value!r}")
    return value


def _usd(value: Any, what: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{what} must be a number of US dollars, got {value!r}")
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a number of US dollars, got {value!r}") from None
    if not math.isfinite(v) or v < 0:
        raise ValueError(f"{what} must be a finite non-negative number of US dollars, got {value!r}")
    return v


def _fmt(usd: float) -> str:
    return f"${usd:.4f}"


# --------------------------------------------------------------------------- lock
class FileLock:
    """A cross-process lock: an OS lock on a persistent lock file (never deleted, no stale breaking).

    The OS drops the lock when the holder closes the file or dies. Threads of one
    process exclude each other too, because each acquire opens its own handle.
    """

    def __init__(self, path: str | Path, *, timeout_s: float = LOCK_TIMEOUT_S, poll_s: float = 0.02):
        self.path = Path(path)
        self.timeout_s = float(timeout_s)
        self.poll_s = float(poll_s)
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def acquire(self) -> None:
        if self._fd is not None:
            raise RouterError(f"lock {self.path} is already held by this FileLock")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout_s
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o666)
        try:
            while True:
                try:
                    _try_lock(fd)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise RouterError(
                            f"could not take the spend-ledger lock {self.path} within {self.timeout_s:.0f} s. "
                            "Another AVAIL process holds it. The OS releases it when that process exits.") from None
                    time.sleep(self.poll_s)
        except BaseException:
            os.close(fd)  # closing the handle also drops a lock taken just before an interrupt
            raise
        self._fd = fd

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            _unlock(fd)
        except OSError:
            pass  # closing the handle below releases it anyway
        finally:
            os.close(fd)

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()


# --------------------------------------------------------------------------- caps
@dataclass(frozen=True)
class SpendCaps:
    """The caps one ledger enforces: a total, and one cap per session.

    ``outside_total`` names the sessions that are counted on their own and that
    need ``approved=True`` for a paid call. Every value must be a finite
    non-negative number of US dollars, so a damaged config refuses calls.
    """

    total_usd: float | None
    sessions: Mapping[str, float]
    outside_total: frozenset[str] = frozenset()
    source: str = field(default="caps passed in code", compare=False)

    def __post_init__(self) -> None:
        where = self.source
        total = self.total_usd
        if total is not None:
            total = self._cap(total, f"{where}: the total cap")
        if not isinstance(self.sessions, Mapping):
            raise RouterError(f"{where}: session caps must map session names to US dollars")
        sessions: dict[str, float] = {}
        for name, cap in self.sessions.items():
            if not is_label(name) or not name.strip():
                raise RouterError(f"{where}: a session name must be non-empty single-line printable text, "
                                  f"got {name!r}")
            sessions[name] = self._cap(cap, f"{where}: the cap of session {name}")
        outside = frozenset(self.outside_total)
        unknown = sorted(outside - set(sessions))
        if unknown:
            raise RouterError(f"{where}: {', '.join(unknown)} is listed outside the total but has no session cap")
        object.__setattr__(self, "total_usd", total)
        object.__setattr__(self, "sessions", MappingProxyType(sessions))
        object.__setattr__(self, "outside_total", outside)

    @staticmethod
    def _cap(value: Any, what: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise RouterError(f"{what} must be a finite non-negative number of US dollars, got {value!r}. "
                              "Fix spend_caps_usd in config/experiment.yaml.")
        return float(value)

    def known(self) -> list[str]:
        return sorted(self.sessions)

    def session_cap(self, session: str) -> float:
        """The cap of ``session``. UnknownSessionError when the config has no such session."""
        if not isinstance(session, str) or session not in self.sessions:
            hint = (f" {TOTAL_CAP_KEY} is the total cap and cannot be used as a session."
                    if session == TOTAL_CAP_KEY else "")
            raise UnknownSessionError(
                f"unknown spend session {session!r}.{hint} Known sessions: {', '.join(self.known()) or 'none'} "
                f"(from {self.source}).")
        return self.sessions[session]

    def needs_approval(self, session: str) -> bool:
        return session in self.outside_total

    def check_session(self, session: str | None, *, paid: bool = True, approved: bool = False) -> None:
        """Raise unless ``session`` may be charged: it exists, and it has Kevin's OK when it needs one.

        ``session`` None is fine for a free call and refused for a paid one.
        """
        if session is None:
            if paid:
                raise SpendCapExceeded(
                    "No spend session is active, so the router refuses this paid call. Start the work with "
                    f"Router(session=...) using one of: {', '.join(self.known()) or 'none'}.",
                    scope="no_session", cap_usd=0.0, spent_usd=0.0, projected_usd=0.0, session=None)
            return
        cap = self.session_cap(session)
        if paid and self.needs_approval(session) and not approved:
            raise SessionNotApproved(
                f"spend session {session} (cap {_fmt(cap)}) is outside the build total and needs Kevin's OK "
                "before any paid call. Pass approved=True only after he has given it.")

    def as_dict(self) -> dict[str, Any]:
        return {"total_cap_usd": self.total_usd, "session_caps": dict(self.sessions),
                "outside_total": sorted(self.outside_total), "source": self.source}

    def __reduce__(self):  # the read-only mapping does not pickle, so rebuild from plain values
        return (SpendCaps, (self.total_usd, dict(self.sessions), self.outside_total, self.source))

    def __hash__(self) -> int:  # the read-only mapping does not hash either
        return hash((self.total_usd, tuple(sorted(self.sessions.items())), self.outside_total))


@lru_cache(maxsize=8)
def _raw_caps(path: str, mtime_ns: int, size: int) -> tuple[tuple[Any, Any], ...]:
    """``spend_caps_usd`` as the YAML holds it, cached per file version.

    The typed config turns ``true`` into 1.0 and the text ``"8"`` into 8.0. A cap
    written that way is a typo, so the values are taken as written and
    :class:`SpendCaps` refuses anything that is not a number.
    """
    block = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["spend_caps_usd"]
    if not isinstance(block, dict):
        raise TypeError("spend_caps_usd must map names to US dollars")
    return tuple(block.items())


def load_spend_caps(path: Path | None = None) -> SpendCaps:
    """The caps in ``spend_caps_usd`` of ``config/experiment.yaml`` (or ``path``).

    A config that cannot be read or validated raises RouterError, so a damaged
    file refuses calls. So does a cap that is not a finite non-negative number.
    """
    source = Path(path) if path is not None else Path(config.EXPERIMENT_YAML)
    try:
        config.load_experiment(source)  # the whole file must validate, as for every other reader of it
        st = source.stat()
        raw = dict(_raw_caps(str(source), st.st_mtime_ns, st.st_size))
    except Exception as exc:  # noqa: BLE001 - a missing file, bad YAML or a failed validation: fail closed
        raise RouterError(f"cannot read the spend caps from {source} ({type(exc).__name__}). Fix the file. "
                          "No paid call is allowed until it loads.") from exc
    if TOTAL_CAP_KEY not in raw:
        raise RouterError(f"{source}: spend_caps_usd has no {TOTAL_CAP_KEY}, the total cap. Add it.")
    sessions = {name: cap for name, cap in raw.items() if name != TOTAL_CAP_KEY}
    return SpendCaps(total_usd=raw[TOTAL_CAP_KEY], sessions=sessions,
                     outside_total=frozenset(name for name in sessions
                                             if name in APPROVAL_SESSIONS or name.startswith(APPROVAL_PREFIXES)),
                     source=str(source))


def _reached(spent: float, projected: float, cap: float) -> bool:
    """A paid reservation is refused when it would cross ``cap`` or the cap is already reached."""
    return spent + projected > cap + _EPS or spent >= cap - _EPS


# --------------------------------------------------------------------------- ledger
class SpendLedger:
    """The append-only spend ledger, with the caps it enforces.

    ``path`` None means ``AVAIL_DATA/spend_ledger.jsonl``. ``caps`` None means the
    caps in ``config/experiment.yaml``, read again at each reservation so an edit
    to the config takes effect at once.
    """

    def __init__(self, path: str | Path | None = None, caps: SpendCaps | None = None, *,
                 lock_timeout_s: float = LOCK_TIMEOUT_S):
        self._path = Path(path) if path is not None else None
        self._caps = caps
        self.lock_timeout_s = float(lock_timeout_s)

    @property
    def path(self) -> Path:
        """The ledger file. The default is resolved on first use, so making a ledger touches no disk."""
        if self._path is None:
            self._path = paths.spend_ledger_path()
        return self._path

    @property
    def lock_path(self) -> Path:
        return self.path.with_name(self.path.name + ".lock")

    @property
    def caps(self) -> SpendCaps:
        return self._caps if self._caps is not None else load_spend_caps()

    def _lock(self) -> FileLock:
        return FileLock(self.lock_path, timeout_s=self.lock_timeout_s)

    # ------------------------------------------------------------------ file
    def _scan(self) -> tuple[list[dict[str, Any]], int]:
        """(every parseable record, the number of lines that did not parse)."""
        try:
            raw = self.path.read_bytes()
        except FileNotFoundError:
            return [], 0
        out: list[dict[str, Any]] = []
        bad = 0
        for line in raw.split(b"\n"):  # bytes split on \n only: no other character ends a record
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line.decode("utf-8", errors="replace"))
            except (ValueError, RecursionError):
                bad += 1
                continue
            if isinstance(rec, dict):
                out.append(rec)
            else:
                bad += 1
        return out, bad

    def records(self) -> list[dict[str, Any]]:
        """Every parseable line. A truncated or corrupt line (a crash mid-write) is skipped."""
        return self._scan()[0]

    def _append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a+b") as f:
            f.seek(0, os.SEEK_END)
            prefix = b""
            if f.tell() > 0:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    prefix = b"\n"  # a crash left a truncated last line: start on a fresh one
                f.seek(0, os.SEEK_END)
            f.write(prefix + line.encode("ascii") + b"\n")
            f.flush()
            os.fsync(f.fileno())

    @staticmethod
    def _index(records: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        reserves: dict[str, dict[str, Any]] = {}
        settles: dict[str, dict[str, Any]] = {}
        for rec in records:
            rid = rec.get("id")
            if not isinstance(rid, str):
                continue
            if rec.get("type") == "reserve":
                reserves[rid] = rec
            elif rec.get("type") == "settle":
                settles[rid] = rec  # the last settle wins
        return reserves, settles

    @staticmethod
    def _orphans(reserves: dict[str, dict[str, Any]], settles: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        return [rec for rid, rec in settles.items() if rid not in reserves]

    def _warn_orphans(self, orphans: list[dict[str, Any]], *, always: bool = False) -> None:
        """Warn about settles with no reserve: always from summary(), once per new count from reserve()."""
        key = str(self.path)
        if not orphans or (not always and _ORPHANS_WARNED.get(key) == len(orphans)):
            return
        _ORPHANS_WARNED[key] = len(orphans)
        usd = sum(self._amount(rec, None) for rec in orphans)
        warnings.warn(f"spend ledger {self.path}: {len(orphans)} settle record(s) ({_fmt(usd)}) have no matching "
                      "reserve, so their calls are missing from every cap. The ledger file was damaged or edited.",
                      RuntimeWarning, stacklevel=3)

    @staticmethod
    def _amount(reserve: dict[str, Any], settle: dict[str, Any] | None) -> float:
        value = settle.get("usd") if settle is not None else reserve.get("usd")
        try:
            v = float(value)
        except (TypeError, ValueError):
            return 0.0
        return v if math.isfinite(v) and v > 0 else 0.0

    @classmethod
    def _spent(cls, reserves: dict[str, dict[str, Any]], settles: dict[str, dict[str, Any]], session: Any = _ANY,
               model_key: str | None = None, *, skip: frozenset[str] = frozenset()) -> float:
        total = 0.0
        for rid, rec in reserves.items():
            if session is not _ANY and rec.get("session") != session:
                continue
            if skip and rec.get("session") in skip:
                continue
            if model_key is not None and rec.get("model_key") != model_key:
                continue
            total += cls._amount(rec, settles.get(rid))
        return total

    # ------------------------------------------------------------------ reading
    def spent(self, session: str | None = None, model_key: str | None = None) -> float:
        """USD spent (settled amounts, and unsettled reservations at their projection). None means any."""
        reserves, settles = self._index(self.records())
        return self._spent(reserves, settles, _ANY if session is None else session, model_key)

    def spent_in_total(self) -> float:
        """USD counted against the total cap: every session except the ones outside it."""
        reserves, settles = self._index(self.records())
        return self._spent(reserves, settles, skip=self.caps.outside_total)

    def remaining(self, session: str) -> float:
        """USD ``session`` can still reserve: the smaller of its own headroom and the total's. Never negative."""
        caps = self.caps
        cap = caps.session_cap(session)
        reserves, settles = self._index(self.records())
        room = cap - self._spent(reserves, settles, session)
        if caps.total_usd is not None and session not in caps.outside_total:
            room = min(room, caps.total_usd - self._spent(reserves, settles, skip=caps.outside_total))
        return max(0.0, room)

    # ------------------------------------------------------------------ writing
    def reserve(self, projected: float, model_key: str, request_hash: str, *, session: str | None = None,
                tags: dict[str, Any] | None = None, paid: bool = True, model_cap_usd: float | None = None,
                approved: bool = False) -> str:
        """Reserve ``projected`` USD for one attempt in ``session``, or raise. Returns the reservation id.

        ``paid`` (the default) runs every check even at a $0 projection: a session
        is required and a reached cap refuses the call. The router passes
        ``paid=False`` only for the FakeModel at its zero prices. A positive
        projection is always paid. ``model_cap_usd`` is the caller's own cap for
        this model within the session. Raises SpendCapExceeded (a cap, or no
        session), UnknownSessionError or SessionNotApproved, in every case before
        anything is written.
        """
        projected = _usd(projected, "projected cost")
        check_label(model_key, "model key")
        if session is not None:
            check_label(session, "session name")
        tags = dict(tags or {})
        if tags.get("phase") is not None:
            check_label(tags["phase"], "phase", allow_empty=True)
        model_cap = _usd(model_cap_usd, f"the cap for model {model_key}") if model_cap_usd is not None else None
        paid = bool(paid) or projected > 0
        caps = self.caps
        try:
            caps.check_session(session, paid=paid, approved=approved)
        except SpendCapExceeded as exc:  # no session: say which call was refused
            exc.projected_usd = projected
            exc.args = (f"{exc.args[0]} The call to {model_key} projects {_fmt(projected)}.",)
            raise
        with self._lock():
            reserves, settles = self._index(self.records())
            self._warn_orphans(self._orphans(reserves, settles))
            if paid:
                session_cap = caps.session_cap(session)
                spent_session = self._spent(reserves, settles, session)
                if _reached(spent_session, projected, session_cap):
                    raise SpendCapExceeded(
                        f"Session {session} cap {_fmt(session_cap)} would be crossed or is reached: spent "
                        f"{_fmt(spent_session)}, this call to {model_key} projects {_fmt(projected)}.",
                        scope="session", cap_usd=session_cap, spent_usd=spent_session, projected_usd=projected,
                        session=session)
                if model_cap is not None:
                    spent_model = self._spent(reserves, settles, session, model_key)
                    if _reached(spent_model, projected, model_cap):
                        raise SpendCapExceeded(
                            f"Model {model_key} cap {_fmt(model_cap)} in session {session} would be crossed or is "
                            f"reached: spent {_fmt(spent_model)}, this call projects {_fmt(projected)}.",
                            scope="model", cap_usd=model_cap, spent_usd=spent_model, projected_usd=projected,
                            session=session)
                if caps.total_usd is not None and session not in caps.outside_total:
                    spent_total = self._spent(reserves, settles, skip=caps.outside_total)
                    if _reached(spent_total, projected, caps.total_usd):
                        raise SpendCapExceeded(
                            f"Total cap {_fmt(caps.total_usd)} would be crossed or is reached: spent "
                            f"{_fmt(spent_total)} over all sessions, this call to {model_key} in session {session} "
                            f"projects {_fmt(projected)}.",
                            scope="total", cap_usd=caps.total_usd, spent_usd=spent_total, projected_usd=projected,
                            session=session)
            rid = uuid.uuid4().hex
            self._append({"type": "reserve", "id": rid, "ts": time.time(), "session": session,
                          "model_key": model_key, "usd": projected, "request_hash": request_hash,
                          "tags": tags})
        return rid

    def settle(self, rid: str, usd: float, usd_source: str, note: str | None = None) -> None:
        """Record a reservation's actual cost (``usd_source`` is ``provider`` or ``computed``)."""
        usd = _usd(usd, "settled cost")
        if usd_source not in ("provider", "computed"):
            raise ValueError("usd_source must be 'provider' or 'computed'")
        rec: dict[str, Any] = {"type": "settle", "id": rid, "ts": time.time(), "usd": usd, "usd_source": usd_source}
        if note:
            rec["note"] = note
        with self._lock():
            self._append(rec)

    def summary(self) -> dict[str, Any]:
        """Spend per session, per model and per session and model, plus unsettled reservations.

        ``orphan_settles`` counts settle records with no matching reserve (a damaged
        or edited ledger). When there are any, a RuntimeWarning says so.
        ``corrupt_lines`` counts lines that did not parse. ``total_usd`` is
        everything in the file and ``in_total_usd`` is the part counted against
        the total cap.
        """
        records, corrupt = self._scan()
        reserves, settles = self._index(records)
        orphans = self._orphans(reserves, settles)
        self._warn_orphans(orphans, always=True)
        caps = self.caps
        by_session: dict[str, float] = {}
        by_model: dict[str, float] = {}
        by_session_model: dict[str, dict[str, float]] = {}
        unsettled_n = 0
        unsettled_usd = 0.0
        total = 0.0
        calls: set[str] = set()
        for rid, rec in reserves.items():
            tags = rec.get("tags") if isinstance(rec.get("tags"), dict) else {}
            calls.add(str(tags.get("call") or rid))  # one call may hold several attempt reservations
            amount = self._amount(rec, settles.get(rid))
            session = str(rec.get("session")) if rec.get("session") is not None else "(none)"
            model = str(rec.get("model_key"))
            total += amount
            by_session[session] = by_session.get(session, 0.0) + amount
            by_model[model] = by_model.get(model, 0.0) + amount
            by_session_model.setdefault(session, {})
            by_session_model[session][model] = by_session_model[session].get(model, 0.0) + amount
            if rid not in settles:
                unsettled_n += 1
                unsettled_usd += amount

        def r(d: dict[str, float]) -> dict[str, float]:
            return {k: round(v, 8) for k, v in sorted(d.items())}

        return {
            "ledger": str(self.path),
            "total_usd": round(total, 8),
            "in_total_usd": round(self._spent(reserves, settles, skip=caps.outside_total), 8),
            "calls": len(calls),
            "reservations": len(reserves),
            "by_session": r(by_session),
            "by_model": r(by_model),
            "by_session_model": {s: r(m) for s, m in sorted(by_session_model.items())},
            "unsettled": {"count": unsettled_n, "usd": round(unsettled_usd, 8)},
            "orphan_settles": {"count": len(orphans),
                               "usd": round(sum(self._amount(rec, None) for rec in orphans), 8)},
            "corrupt_lines": corrupt,
            "caps": caps.as_dict(),
        }
