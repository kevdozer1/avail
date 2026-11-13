"""Record types: Predicate, Claim, Evidence, Witness, Verdict, Fork, Assist, Audit, Intervention, LedgerEvent, and Cost.

Spec: ``docs/01_architecture.md``, section "Schemas". Built in session S1.

Every record forbids unknown fields, refuses NaN and infinity, and does not change
after it is built. A refused value raises pydantic's ``ValidationError``.
:func:`explain` turns one into lines that name the field and say how to fix it, and
:func:`validate_record` raises :class:`SchemaError` carrying those lines.

2 rules here carry the done gate (CLAUDE.md rule 11): a :class:`Verdict` that accepts
must list evidence ids, and a :class:`Fork` never holds ``accept``. The static half of
the gate is in :mod:`avail.schemas.gate_rules`.

The known predicates and their definitions come from the task files (``tasks/*.yaml``)
the first time ``PREDICATE_DEFINITIONS`` or ``PREDICATES`` is used (:mod:`avail.schemas.kinds`).

``avail schemas`` writes the JSON Schema of every record type (:mod:`avail.schemas.export`).

Example::

    from avail.schemas import Claim, ClaimContext, Predicate, new_id, validate_record

    claim = Claim(
        claim_id=new_id("claim"), run_id="run_a", cycle_id="cycle_17", step_id="insert",
        predicate=Predicate(name="seated", target={"item": "p1", "port": 7}),
        raised_by="skill", context=ClaimContext(task="plug_insert", stage="insert", attempt=0),
    )
    row = claim.model_dump(mode="json")        # plain JSON, ready for the ledger
    same = validate_record("claim", row)       # the record again, or SchemaError with 1 fix line per problem
"""

from typing import Any

from . import kinds
from .base import Record, new_id
from .errors import SchemaError, explain, record_model, validate_record
from .kinds import (
    CHOSEN_BY,
    CONFIRM_ANSWERS,
    CORRECTION_KINDS,
    DECISIONS,
    EVENT_KINDS,
    EVIDENCE_KINDS,
    FORK_OPTION_PREFIXES,
    FORK_OPTION_WORDS,
    FRAME_TAGS,
    INTERVENTION_OUTCOMES,
    QUESTION_TYPES,
    RAISED_BY,
    SIDE_EFFECTS,
    ChosenBy,
    CorrectionKind,
    Decision,
    EventKind,
    EvidenceKind,
    FrameTag,
    InterventionOutcome,
    QuestionType,
    RaisedBy,
    SideEffect,
)
from .records import (
    FORK_OPTION_FORMS,
    PART_MODELS,
    PAYLOAD_MODELS,
    RECORD_MODELS,
    Assist,
    Audit,
    Calibration,
    Claim,
    ClaimContext,
    Correction,
    Cost,
    CycleEnd,
    CyclePredicate,
    Evidence,
    Fork,
    Intervention,
    LedgerEvent,
    Predicate,
    Shadow,
    Stage,
    Verdict,
    Witness,
    WitnessScope,
    is_fork_option,
)

__all__ = [
    # records
    "Assist", "Audit", "Claim", "Cost", "Evidence", "Fork", "Intervention", "LedgerEvent", "Predicate", "Verdict",
    "Witness",
    # their parts
    "Calibration", "ClaimContext", "Correction", "CycleEnd", "CyclePredicate", "Shadow", "Stage", "WitnessScope",
    # the literal lists, from kinds
    "CHOSEN_BY", "CONFIRM_ANSWERS", "CORRECTION_KINDS", "DECISIONS", "EVENT_KINDS", "EVIDENCE_KINDS",
    "FORK_OPTION_PREFIXES", "FORK_OPTION_WORDS", "FRAME_TAGS", "INTERVENTION_OUTCOMES", "PREDICATES",
    "PREDICATE_DEFINITIONS", "QUESTION_TYPES", "RAISED_BY", "SIDE_EFFECTS",
    "ChosenBy", "CorrectionKind", "Decision", "EventKind", "EvidenceKind", "FrameTag", "InterventionOutcome",
    "QuestionType", "RaisedBy", "SideEffect",
    # helpers
    "FORK_OPTION_FORMS", "PART_MODELS", "PAYLOAD_MODELS", "RECORD_MODELS", "Record", "SchemaError", "explain",
    "is_fork_option", "new_id", "record_model", "validate_record",
]


def __getattr__(name: str) -> Any:
    """``PREDICATE_DEFINITIONS`` and ``PREDICATES`` come from :mod:`avail.schemas.kinds`, which reads the task files."""
    if name in kinds.LAZY_NAMES:
        value = getattr(kinds, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *kinds.LAZY_NAMES})
