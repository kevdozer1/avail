"""The ledger: one append-only JSONL event log per run, and typed pandas tables read back from it.

Every claim, evidence item, verdict, fork, assist and audit of a run is one
line in ``AVAIL_DATA/runs/<run_id>/ledger.jsonl``, and every number in a report
comes from these lines::

    from avail.ledger import LedgerWriter, list_runs, load_ledger

    with LedgerWriter("e3_arm1") as ledger:
        seq = ledger.append("claim", claim)              # a record with model_dump(mode="json")
        ledger.append("tool_call", {"tool": "observe"})  # or a plain dict

    tables = load_ledger("e3_arm1")
    tables.verdicts      # one row per verdict, payload keys as typed columns
    tables.events        # every event, payload as a dict
    tables.problems      # torn lines and seq breaks, as sentences for a report

    list_runs()          # the run ids under AVAIL_DATA/runs that have a ledger

Each line is ``{run_id, seq, ts, kind, payload}``. ``seq`` starts at 0 and
counts up by 1 with no gaps, across every reopen of the run. ``kind`` is one of
``avail.schemas.kinds.EVENT_KINDS``.

A run has one writer at a time. A second LedgerWriter on an open run, in any
process, raises LedgerLockedError. Every append is one unbuffered write that
reaches the operating system before ``append`` returns, so Ctrl+C or a killed
process leaves a readable ledger with every completed append in it. ``fsync``
runs at ``close()`` and ``sync()`` and never per append. A line torn by a crash
is reported and never merged with new data. ``iter_events`` streams a ledger
without pandas.

See ``writer.py`` for the lock, the flush and fsync policy and the Ctrl+C
rules, ``lines.py`` for the line format and ``reader.py`` for the tables.
"""

from .jsonsafe import dumps, to_jsonable
from .lines import FIELDS, LEDGER_FILE, LOCK_FILE, MAX_SEQ, LedgerWarning, TornLine, check_ledger_run_id, parse_line
from .reader import (
    BASE_COLUMNS, TABLE_NAMES, LedgerTables, SeqBreak, iter_events, ledger_path, list_runs, load_ledger,
)
from .writer import LedgerLockedError, LedgerWriter

__all__ = [
    "BASE_COLUMNS", "FIELDS", "LEDGER_FILE", "LOCK_FILE", "MAX_SEQ", "TABLE_NAMES", "LedgerLockedError",
    "LedgerTables", "LedgerWarning", "LedgerWriter", "SeqBreak", "TornLine", "check_ledger_run_id", "dumps",
    "iter_events", "ledger_path", "list_runs", "load_ledger", "parse_line", "to_jsonable",
]
