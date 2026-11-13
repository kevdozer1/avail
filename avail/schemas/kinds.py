"""The closed vocabularies shared by the schemas and the ledger.

One place for every literal list in ``docs/01_architecture.md``, so the record
types and the ledger cannot drift apart. ``accept`` is a verdict decision and is
deliberately absent from the fork options: nothing at a fork can choose it.

The predicates are data. Each task file (``tasks/<task>.yaml``) defines the
predicates its steps claim, each with a "done means" text a person can apply.
``PREDICATE_DEFINITIONS`` and ``PREDICATES`` are filled from those files the first
time either is used, so importing the schemas reads no file. 2 task files may word
the same predicate for their own items. :func:`load_predicate_definitions` says
which wording becomes the default, and a claim that knows its task should carry
its task's own text as the predicate's definition.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any, Literal, get_args

from ..paths import REPO_ROOT

Decision = Literal["accept", "reject", "abstain"]
RaisedBy = Literal["skill", "policy", "agent", "human"]
EvidenceKind = Literal["passive_proprio", "passive_vision", "active_probe", "judge", "instrumented"]
SideEffect = Literal["none", "repaired", "disturbed"]
ChosenBy = Literal["policy", "agent", "human"]
QuestionType = Literal["confirm", "choose", "guide_marks", "guide_text", "takeover"]
CorrectionKind = Literal["marks", "step_text", "pose_offset", "takeover"]
EventKind = Literal[
    "claim", "evidence", "verdict", "fork", "assist", "audit",
    "tool_call", "model_call", "calibration_update", "drift_alarm", "cycle_end",
    "cycle_start", "intervention",
]
InterventionOutcome = Literal["signed_off", "retried", "taken_over_done", "taken_over_failed", "unresolved"]
FrameTag = Literal["autonomous", "assisted"]

DECISIONS: tuple[str, ...] = get_args(Decision)
RAISED_BY: tuple[str, ...] = get_args(RaisedBy)
EVIDENCE_KINDS: tuple[str, ...] = get_args(EvidenceKind)
SIDE_EFFECTS: tuple[str, ...] = get_args(SideEffect)
CHOSEN_BY: tuple[str, ...] = get_args(ChosenBy)
QUESTION_TYPES: tuple[str, ...] = get_args(QuestionType)
CORRECTION_KINDS: tuple[str, ...] = get_args(CorrectionKind)
EVENT_KINDS: tuple[str, ...] = get_args(EventKind)
INTERVENTION_OUTCOMES: tuple[str, ...] = get_args(InterventionOutcome)
FRAME_TAGS: tuple[str, ...] = get_args(FrameTag)

CONFIRM_ANSWERS: tuple[str, ...] = ("yes", "no", "cannot_tell")
"""What a person answers to a confirm question. yes and no label the claim, cannot_tell does not."""

CONFIRM_LABELS: dict[str, bool | None] = {"yes": True, "no": False, "cannot_tell": None}
"""The truth label each confirm answer carries."""

# A fork option is one of the bare words or a prefixed form such as "repair:<probe>".
FORK_OPTION_WORDS: tuple[str, ...] = ("retry", "ask_human", "abort")
FORK_OPTION_PREFIXES: tuple[str, ...] = ("repair:", "recover:")


# ------------------------------------------------------------------ the predicates, from the task files
TASKS_DIR = REPO_ROOT / "tasks"
"""Where the task files live: 1 YAML file per task."""

LAZY_NAMES: tuple[str, ...] = ("PREDICATE_DEFINITIONS", "PREDICATES")
"""The 2 names filled from the task files the first time they are used."""

# Declared for readers and type checkers. The values are set by __getattr__ below on first use.
PREDICATE_DEFINITIONS: dict[str, str]
PREDICATES: tuple[str, ...]

_PREDICATE_NAME = re.compile(r"[a-z][a-z0-9_]*")
_FILL_LOCK = threading.Lock()


class PredicateDefinitionsError(RuntimeError):
    """A task file the predicate definitions cannot be read from.

    It is not a ValueError on purpose: raised inside a record's validator, it stops the
    program with its own message instead of turning into a refusal of the record.
    """


def task_files(folder: str | Path | None = None) -> list[Path]:
    """The task files in ``folder`` (default: the repo's ``tasks``), sorted by name."""
    root = Path(folder) if folder is not None else TASKS_DIR
    if not root.is_dir():
        return []
    return sorted(path for path in root.glob("*.yaml") if path.is_file())


def _definitions_in(path: Path) -> dict[str, str]:
    """``predicates.<name>.done_means`` of 1 task file, in file order, each as 1 line of text."""
    import yaml

    fix = "give each predicate under predicates a done_means text a person can apply"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise PredicateDefinitionsError(
            f"the task file {path} cannot be read ({type(exc).__name__}: {exc}). Fix: {fix}") from exc
    predicates = data.get("predicates") if isinstance(data, dict) else None
    if not isinstance(predicates, dict) or not predicates:
        raise PredicateDefinitionsError(f"the task file {path} defines no predicates. Fix: {fix}")
    found: dict[str, str] = {}
    for name, spec in predicates.items():
        if not isinstance(name, str) or not _PREDICATE_NAME.fullmatch(name):
            raise PredicateDefinitionsError(
                f"the task file {path} names a predicate {name!r}, which is not a lowercase identifier. "
                "Fix: use lowercase letters, digits and underscores, starting with a letter")
        text = spec.get("done_means") if isinstance(spec, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise PredicateDefinitionsError(
                f"the predicate {name} in the task file {path} has no done_means text. Fix: {fix}")
        found[name] = " ".join(text.split())
    return found


def load_predicate_definitions(folder: str | Path | None = None) -> dict[str, str]:
    """Every predicate defined in the task files of ``folder`` (default: the repo's ``tasks``), with its text.

    The files are read widest first: the file that defines the most predicates, then the
    others, ties in name order. The first file to define a predicate gives its text and its
    place in the order. A later file that words the same predicate for its own items does
    not change the default. Its text belongs on that task's claims.

    Raises :class:`PredicateDefinitionsError` when the folder holds no task file or a task file
    has no usable ``done_means`` text.
    """
    files = task_files(folder)
    if not files:
        where = Path(folder) if folder is not None else TASKS_DIR
        raise PredicateDefinitionsError(f"no task file (*.yaml) in {where}, so no predicate has a definition. "
                                        "Fix: put the task files in that folder, 1 YAML file per task")
    per_file = [(path, _definitions_in(path)) for path in files]
    per_file.sort(key=lambda pair: (-len(pair[1]), pair[0].name))
    merged: dict[str, str] = {}
    for _path, definitions in per_file:
        for name, text in definitions.items():
            merged.setdefault(name, text)
    return merged


def __getattr__(name: str) -> Any:
    """Fill ``PREDICATE_DEFINITIONS`` and ``PREDICATES`` from the task files on first use."""
    if name not in LAZY_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    with _FILL_LOCK:
        if name not in globals():
            definitions = load_predicate_definitions()
            globals().update(PREDICATE_DEFINITIONS=definitions, PREDICATES=tuple(definitions))
    return globals()[name]


def __dir__() -> list[str]:
    return sorted({*globals(), *LAZY_NAMES})
