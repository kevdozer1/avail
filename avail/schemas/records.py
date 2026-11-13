"""The record types of ``docs/01_architecture.md``.

A claim comes in, evidence is gathered, a witness gives a verdict, a fork picks
the next move, a person may assist or audit, and a ledger event records each of
them. When a person is asked for help, an :class:`Intervention` records the whole
episode: the question, the answer, the seconds and, for a takeover, the handoff and
handback in the cycle's frame log. Every type here is a
:class:`~avail.schemas.base.Record`: unknown fields are refused, NaN and infinity are
refused, and a record does not change after it is built. :class:`CycleEnd` is the 1
type that keeps unknown fields, because it describes the payload of a ledger event,
which is a free dict.

2 rules in this file carry the done gate (CLAUDE.md rule 11):

* a :class:`Verdict` whose decision is ``accept`` must list evidence ids
* a :class:`Fork` never holds ``accept``: in its options, in its choice or in its
  shadow choice

Field descriptions are short phrases that say what belongs in the field.
:func:`avail.schemas.explain` quotes them after a fix, and the JSON Schema carries them.
"""

from __future__ import annotations

import difflib
import math
import re
import time
import unicodedata
from typing import Annotated, Any

from pydantic import BeforeValidator, ConfigDict, Field, ValidationInfo, WithJsonSchema, field_validator, model_validator

from .base import (
    Count,
    Flag,
    Id,
    IsoDate,
    NonNegative,
    Number,
    Record,
    Share,
    Timestamp,
    Version,
    path_text,
    plain,
    problem,
    refuse,
    short,
)
from . import kinds as _kinds
from .kinds import (
    CHOSEN_BY,
    CONFIRM_LABELS,
    FORK_OPTION_PREFIXES,
    FORK_OPTION_WORDS,
    ChosenBy,
    CorrectionKind,
    Decision,
    EventKind,
    EvidenceKind,
    InterventionOutcome,
    QuestionType,
    RaisedBy,
    SideEffect,
)

__all__ = [
    "Assist", "Audit", "Calibration", "Claim", "ClaimContext", "Correction", "Cost", "CycleEnd", "CyclePredicate",
    "Evidence", "FORK_OPTION_FORMS", "Fork", "Intervention", "LedgerEvent", "PART_MODELS", "PAYLOAD_MODELS",
    "Predicate", "RECORD_MODELS", "SUM_TOLERANCE", "Shadow", "Stage", "Verdict", "Witness", "WitnessScope",
    "is_fork_option",
]

SUM_TOLERANCE = 1e-9
"""How far the stage shares may exceed a witness's budget before it is refused."""

_DONE_RULE = "Only evidence or a person marks a step done"


# ------------------------------------------------------------------ cost
class Cost(Record):
    """What a measurement or a verdict cost. Costs add with ``+``: seconds and dollars sum, and
    ``moved_robot`` is true when any part moved the robot."""

    seconds: NonNegative = Field(description="the wall time spent, in seconds")
    usd: NonNegative = Field(description="the money spent, in US dollars")
    moved_robot: Flag = Field(description="true when this moved the robot")

    @classmethod
    def zero(cls) -> Cost:
        """The cost of nothing: 0 seconds, 0 dollars, no motion."""
        return cls(seconds=0.0, usd=0.0, moved_robot=False)

    def __add__(self, other: object) -> Cost:
        if not isinstance(other, Cost):
            return NotImplemented
        return Cost(
            seconds=self.seconds + other.seconds,
            usd=self.usd + other.usd,
            moved_robot=self.moved_robot or other.moved_robot,
        )

    def __radd__(self, other: object) -> Cost:
        # sum() starts from the int 0, so sum(costs) works without a start value
        if isinstance(other, int) and not isinstance(other, bool) and other == 0:
            return self
        return NotImplemented


# ------------------------------------------------------------------ predicate
_PREDICATE_NAME = re.compile(r"[a-z][a-z0-9_]*")
_TARGET_EXAMPLE = "{'plug': 'p1', 'port': 7}"


def _predicate_name(value: Any) -> str:
    if not isinstance(value, str) or not _PREDICATE_NAME.fullmatch(value):
        raise problem("predicate_name", f"{short(value)} is not a lowercase identifier",
                      "use lowercase letters, digits and underscores, starting with a letter, "
                      "for example seated or released_stable")
    return value


def _target_key(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise problem("target_key", f"the key {short(value)} is not a non-empty string",
                      f"name each part of the target with text, for example {_TARGET_EXAMPLE}")
    return value


def _target_value(value: Any) -> str | int:
    value = plain(value)
    if isinstance(value, bool) or not isinstance(value, (str, int)) or (isinstance(value, str) and not value.strip()):
        raise problem("target_value", f"{short(value)} is not text or a whole number",
                      f"use a non-empty string or an integer, for example {_TARGET_EXAMPLE}")
    return value


PredicateName = Annotated[str, Field(pattern=f"^{_PREDICATE_NAME.pattern}$"), BeforeValidator(_predicate_name)]
TargetKey = Annotated[str, Field(min_length=1), BeforeValidator(_target_key)]
TargetValue = Annotated[str | int, BeforeValidator(_target_value)]


class Predicate(Record):
    """A physical condition with a definition a human auditor can apply."""

    name: PredicateName = Field(description="the predicate's name, a lowercase identifier such as seated")
    target: dict[TargetKey, TargetValue] = Field(
        description=f"what the condition is about, for example {_TARGET_EXAMPLE}")
    definition: str = Field(
        default="", validate_default=True,
        description="plain text a human auditor can apply. Left empty, it is filled in for a known predicate")

    @field_validator("definition", mode="before")
    @classmethod
    def _null_is_empty(cls, value: Any) -> Any:
        return "" if value is None else value

    @field_validator("definition")
    @classmethod
    def _fill_definition(cls, value: str, info: ValidationInfo) -> str:
        if value.strip():
            return value
        name = info.data.get("name")
        if name is None:  # the name itself was refused, and that error is enough
            return value
        definitions = _kinds.PREDICATE_DEFINITIONS  # read from the task files the first time it is needed
        if name in definitions:
            return definitions[name]
        raise problem("predicate_definition", f"{name} is not a known predicate, so it needs its own definition",
                      "add a definition a human auditor can apply, or use a known predicate: "
                      + ", ".join(_kinds.PREDICATES))


# ------------------------------------------------------------------ claim
class ClaimContext(Record):
    """Where in a task a claim came from."""

    task: str = Field(description="the task the claim belongs to, for example plug_cycle")
    stage: str = Field(description="the stage of the task that raised the claim, for example insert")
    attempt: Count = Field(description="the try at this step, counting from 0")


class Claim(Record):
    """An assertion that a predicate holds for a target, raised by a skill, a policy, an agent or a person."""

    claim_id: Id = Field(description="this claim's id. new_id('claim') makes a fresh one")
    run_id: Id = Field(description="the run that holds the claim")
    cycle_id: Id = Field(description="the cycle the claim belongs to")
    step_id: Id = Field(description="the step the claim says is done")
    predicate: Predicate = Field(description="the condition claimed to hold")
    raised_by: RaisedBy = Field(description="who raised the claim")
    ts: Timestamp = Field(default_factory=time.time, description="when the claim was raised, in Unix seconds")
    context: ClaimContext = Field(description="where in the task the claim came from")


# ------------------------------------------------------------------ evidence
_LEAVE_OUT = "leave this key out of features when the source has no reading"


def _feature_key(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise problem("feature_key", f"the key {short(value)} is not a non-empty string",
                      "name each feature with text, for example travel_mm")
    return value


def _feature_value(value: Any) -> float:
    value = plain(value)
    if isinstance(value, (bool, int, float)):
        try:
            number = float(value)
        except OverflowError:
            number = math.inf
        if math.isfinite(number):
            return number
        raise problem("feature_not_finite", f"{short(value)} is not a finite number", _LEAVE_OUT)
    if value is None:
        raise problem("feature_missing", "null is not a number", _LEAVE_OUT)
    if isinstance(value, (dict, list, tuple, set, frozenset)) or hasattr(value, "shape"):
        raise problem("feature_not_flat", f"{short(value)} is not a single number",
                      "features is flat, so give each number its own key, for example force_x_n and force_y_n")
    raise problem("feature_not_a_number", f"{short(value)} is not a number",
                  "give a number, or true or false. Text is refused")


FeatureKey = Annotated[str, Field(min_length=1), BeforeValidator(_feature_key)]
FeatureValue = Annotated[float, BeforeValidator(_feature_value)]
ArtifactPath = Annotated[str, Field(min_length=1), BeforeValidator(path_text)]


class Evidence(Record):
    """One measurement that bears on a claim, with its cost."""

    evidence_id: Id = Field(description="this evidence item's id. new_id('ev') makes a fresh one")
    claim_id: Id = Field(description="the claim this evidence bears on")
    source_id: Id = Field(description="the evidence source that measured it, for example probe.push_seat")
    kind: EvidenceKind = Field(description="the kind of source")
    features: dict[FeatureKey, FeatureValue] = Field(
        default_factory=dict,
        description="a flat dict of finite numbers measured by the source. true, false and whole numbers become "
                    "floats, and a source with no reading for a key leaves the key out")
    score: Number | None = Field(
        default=None,
        description="1 number where higher means the predicate is more likely true, or null when the source gives none")
    cost: Cost = Field(description="what gathering this cost")
    artifacts: list[ArtifactPath] = Field(
        default_factory=list, description="the paths of the images or files this evidence saved")
    side_effect: SideEffect = Field(default="none", description="what gathering this did to the scene")
    ts: Timestamp = Field(default_factory=time.time, description="when the evidence was gathered, in Unix seconds")


# ------------------------------------------------------------------ witness
class WitnessScope(Record):
    """What a witness's calibration covers."""

    station_id: Id = Field(description="the station the witness was calibrated on")
    connector: Id = Field(description="the connector type the witness was calibrated for")


class Stage(Record):
    """One stage of a witness: the evidence it collects, its scorer, its 2 thresholds and its share of the
    error budget. A score above ``tau_accept`` accepts and a score below ``tau_reject`` rejects. A threshold
    of null never fires, because JSON has no infinity."""

    source_ids: list[Id] = Field(description="the evidence sources this stage collects")
    scorer_id: Id = Field(description="the scorer that turns this stage's evidence into 1 score")
    tau_reject: Number | None = Field(
        description="the reject threshold: a score below it rejects, and null means this stage never rejects")
    tau_accept: Number | None = Field(
        description="the accept threshold: a score above it accepts, and null means this stage never accepts")
    alpha_k: Share = Field(description="this stage's share of the false-accept budget alpha")
    beta_k: Share = Field(description="this stage's share of the false-reject budget beta")


class Calibration(Record):
    """What a witness's thresholds were calibrated on, and what was measured there."""

    n_pos: Count = Field(description="how many calibration states had the predicate true")
    n_neg: Count = Field(description="how many calibration states had the predicate false")
    split_id: Id = Field(description="the split the calibration states came from, for example zoo_calibration")
    far_point: Share | None = Field(
        default=None, description="the measured false-accept rate, or null when it was not measured")
    far_ucb95: Share | None = Field(
        default=None,
        description="the one-sided 95% upper bound on the false-accept rate, or null when it was not measured")
    frr_point: Share | None = Field(
        default=None, description="the measured false-reject rate, or null when it was not measured")
    date: IsoDate = Field(description="the calibration date in ISO form, for example 2026-11-11")


class Witness(Record):
    """A calibrated rule from evidence to accept, reject or abstain, with a stated false-accept bound.

    The stage shares must fit the budget: ``sum(alpha_k) <= alpha`` and ``sum(beta_k) <= beta``, within 1e-9.
    ``docs/04_calibration_math.md`` section 4 builds the cascade's false-accept bound on that sum, under the
    2 conditions it lists. This type checks the sum and nothing else about the calibration.
    """

    witness_id: Id = Field(description="this witness's id. new_id('witness') makes a fresh one")
    version: Version = Field(description="the label of this fit of the witness, for example v3")
    predicate_name: PredicateName = Field(description="the predicate this witness judges")
    scope: WitnessScope = Field(description="the station and connector type the calibration covers")
    stages: list[Stage] = Field(description="the stages in the order they run, cheapest first")
    alpha: Share = Field(description="the false-accept budget of the whole cascade")
    beta: Share = Field(description="the false-reject budget of the whole cascade")
    calibration: Calibration = Field(description="what the thresholds were calibrated on")

    @model_validator(mode="after")
    def _shares_fit_the_budget(self) -> Witness:
        found = []
        for budget, share, words in (("alpha", "alpha_k", "false-accept"), ("beta", "beta_k", "false-reject")):
            limit = getattr(self, budget)
            total = math.fsum(getattr(stage, share) for stage in self.stages)
            if total > limit + SUM_TOLERANCE:
                found.append((budget, problem(
                    "budget_overspent",
                    f"the stages spend {total:.12g} of {words} budget (the sum of {share}), "
                    f"more than {budget} = {limit:.12g}",
                    f"lower the {share} values until they add up to at most {budget}, or raise {budget}",
                ), limit))
        if found:
            refuse("Witness", found)
        return self


# ------------------------------------------------------------------ verdict
def _accept_rule_schema(schema: dict[str, Any], _model: type) -> None:
    """Say the accept rule in the JSON Schema as well."""
    schema["allOf"] = [{
        "if": {"properties": {"decision": {"const": "accept"}}, "required": ["decision"]},
        "then": {"properties": {"evidence_ids": {"minItems": 1}}, "required": ["evidence_ids"]},
    }]


class Verdict(Record):
    """The witness's answer on one claim.

    A verdict whose decision is ``accept`` must list the ids of the evidence behind it
    (CLAUDE.md rule 11). Only ``avail/witness`` builds verdicts. Everything else reads
    them back with ``Verdict.model_validate``.
    """

    model_config = ConfigDict(json_schema_extra=_accept_rule_schema)

    verdict_id: Id = Field(description="this verdict's id. new_id('verdict') makes a fresh one")
    claim_id: Id = Field(description="the claim this verdict answers")
    decision: Decision = Field(description="the witness's answer")
    score: Number | None = Field(
        default=None, description="the score of the last stage that ran, or null when no stage gave one")
    stage_reached: Count = Field(
        description="how many stages ran: 1 means the first stage gave the answer and 0 means no stage ran")
    witness_version: Version = Field(description="the version of the witness that gave this verdict")
    evidence_ids: list[Id] = Field(
        default_factory=list,
        description="the ids of the evidence the witness used. An accept must list at least 1")
    cost_total: Cost = Field(description="the summed cost of the evidence gathered for this verdict")

    @model_validator(mode="after")
    def _accept_needs_evidence(self) -> Verdict:
        if self.decision == "accept" and not self.evidence_ids:
            refuse("Verdict", [("evidence_ids", problem(
                "accept_needs_evidence",
                "an accept needs evidence, and this verdict lists no evidence ids",
                "list the ids of the Evidence records the witness used, or use the decision abstain. "
                + _DONE_RULE,
            ), self.evidence_ids)])
        return self


# ------------------------------------------------------------------ fork
_OPTION_NAME = r"[A-Za-z0-9][A-Za-z0-9_.-]*"
_FORK_OPTION = re.compile(
    "(?:" + "|".join(re.escape(word) for word in FORK_OPTION_WORDS)
    + "|(?:" + "|".join(re.escape(prefix) for prefix in FORK_OPTION_PREFIXES) + ")" + _OPTION_NAME + ")"
)
_PLACEHOLDERS = {"repair:": "<probe>", "recover:": "<skill>"}
FORK_OPTION_FORMS = ", ".join(
    [*FORK_OPTION_WORDS, *(prefix + _PLACEHOLDERS.get(prefix, "<name>") for prefix in FORK_OPTION_PREFIXES)]
)
"""The fork options in words: ``retry, ask_human, abort, repair:<probe>, recover:<skill>``."""


def is_fork_option(value: Any) -> bool:
    """True when ``value`` is a well-formed fork option. ``accept`` never is, in any casing or with any prefix."""
    return isinstance(value, str) and not _asks_to_accept(value) and _FORK_OPTION.fullmatch(value) is not None


def _asks_to_accept(text: str) -> bool:
    return "accept" in unicodedata.normalize("NFKC", text).casefold()


def _closest_option(text: str) -> str | None:
    """The fork option ``text`` was probably meant to be, or None."""
    for candidate in (text.strip(), text.strip().casefold()):
        if _FORK_OPTION.fullmatch(candidate):
            return candidate
    near = difflib.get_close_matches(text.strip().casefold(), FORK_OPTION_WORDS, n=1, cutoff=0.6)
    return near[0] if near else None


def _fork_option(value: Any) -> str:
    if not isinstance(value, str):
        raise problem("fork_option", f"{short(value)} is not text", f"use one of {FORK_OPTION_FORMS}")
    if _asks_to_accept(value):
        raise problem("fork_accept", f"{short(value)} asks a fork to accept, and accept is never a fork option",
                      f"use one of {FORK_OPTION_FORMS}. {_DONE_RULE}, so no decider at a fork can accept")
    if not _FORK_OPTION.fullmatch(value):
        bare = value.strip().casefold().rstrip(":")
        if bare + ":" in FORK_OPTION_PREFIXES:
            hint = f". {bare} needs a name after the colon, for example {bare}:{_PLACEHOLDERS.get(bare + ':', '<name>')}"
        else:
            closest = _closest_option(value)
            hint = f". Did you mean {closest}?" if closest else ""
        raise problem("fork_option", f"{short(value)} is not a fork option", f"use one of {FORK_OPTION_FORMS}{hint}")
    return value


ForkOption = Annotated[
    str,
    BeforeValidator(_fork_option),
    WithJsonSchema({
        "type": "string",
        "pattern": f"^{_FORK_OPTION.pattern}$",
        "not": {"pattern": "[Aa][Cc][Cc][Ee][Pp][Tt]"},
    }),
]


class Shadow(Record):
    """What a small decision model would have picked at a fork. It is logged with its latency and cost and is
    never used."""

    decider: Id = Field(description="the shadow model's id")
    choice: ForkOption | None = Field(
        description="the fork option the shadow model picked, or null when it gave no usable answer. "
                    "accept is refused here too")
    latency_s: NonNegative = Field(description="how long the shadow model took, in seconds")
    usd: NonNegative = Field(description="what the shadow call cost, in US dollars")


class Fork(Record):
    """The choice after a verdict that is not accept.

    ``options`` is the contract: a decider picks from it and from nothing else. ``accept`` is refused in
    ``options``, in ``chosen`` and in ``shadow.choice``, in any casing and with any prefix.

    ``chosen`` stays null while the fork is open. ``chosen_by`` may be named when the fork opens and must be
    named once an option is chosen. To close a fork, use
    ``fork.model_copy(update={"chosen": ..., "chosen_by": ...})``, which validates the result.
    """

    fork_id: Id = Field(description="this fork's id. new_id('fork') makes a fresh one")
    verdict_id: Id = Field(description="the reject or abstain verdict that opened this fork")
    options: list[ForkOption] = Field(
        description=f"the menu a decider may pick from, drawn from {FORK_OPTION_FORMS}. accept is never on it",
        json_schema_extra={"minItems": 1, "uniqueItems": True})
    chosen: ForkOption | None = Field(
        default=None, description="the option that was picked, or null while the fork is open")
    chosen_by: ChosenBy | None = Field(
        default=None,
        description="who picks the option. It may be named when the fork opens and must be named once an option "
                    "is chosen")
    shadow: Shadow | None = Field(
        default=None, description="what a small decision model would have picked. It is logged and never used")
    reason: str = Field(default="", description="why this option was picked, in plain words")

    @field_validator("options")
    @classmethod
    def _a_real_menu(cls, value: list[str]) -> list[str]:
        if not value:
            raise problem("fork_options_empty", "a fork with no options leaves nothing to choose",
                          f"give at least 1 of {FORK_OPTION_FORMS}")
        repeated = sorted({option for option in value if value.count(option) > 1})
        if repeated:
            raise problem("fork_option_repeated", f"{', '.join(repeated)} is listed more than once",
                          "list each option once")
        return value

    @model_validator(mode="after")
    def _choice_fits_the_menu(self) -> Fork:
        if self.chosen is None:
            return self
        found = []
        if self.chosen not in self.options:
            found.append(("chosen", problem(
                "fork_choice", f"{short(self.chosen)} is not one of this fork's options ({', '.join(self.options)})",
                "pick one of the options, or leave chosen as null while the fork is open",
            ), self.chosen))
        if self.chosen_by is None:
            found.append(("chosen_by", problem(
                "fork_chooser", "this fork has a choice and does not say who made it",
                f"set chosen_by to one of {', '.join(CHOSEN_BY)}",
            ), self.chosen_by))
        if found:
            refuse("Fork", found)
        return self


# ------------------------------------------------------------------ assist and audit
class Correction(Record):
    """A correction a person gave with an answer."""

    kind: CorrectionKind = Field(description="the form of the correction")
    payload: dict[str, Any] = Field(
        description="the correction itself, for example the marked points or the step text")


class Assist(Record):
    """A person's answer at a fork, with an optional correction."""

    assist_id: Id = Field(description="this assist's id. new_id('assist') makes a fresh one")
    fork_id: Id = Field(description="the fork the person answered")
    question: QuestionType = Field(description="the type of question the person was asked")
    options: list[str] = Field(default_factory=list, description="the answers offered to the person")
    answer: str = Field(description="what the person answered")
    truth_label: Flag | None = Field(
        default=None,
        description="the person's yes or no on the claim as a real boolean, or null when the answer is not a label")
    correction: Correction | None = Field(default=None, description="the correction the person gave, if any")
    seconds_to_answer: NonNegative = Field(description="the seconds from the question appearing to the answer")
    seconds_handling: NonNegative = Field(description="the seconds the person spent handling the case")
    assistant_id: Id = Field(description="who answered")


class Audit(Record):
    """A person's label on a sampled claim."""

    audit_id: Id = Field(description="this audit's id. new_id('audit') makes a fresh one")
    claim_id: Id = Field(description="the claim the person labeled")
    truth_label: Flag = Field(description="the person's label as a real boolean: true when the predicate holds")
    auditor: Id = Field(description="who gave the label")
    seconds: NonNegative = Field(description="the seconds the audit took")


# ------------------------------------------------------------------ intervention
_DRIVE = re.compile(r"^[A-Za-z]:")
_CONFIRM_FIX = "A confirm answer is yes with truth_label true, no with truth_label false, or cannot_tell with null"


def _item_id(value: Any) -> str:
    value = plain(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if not isinstance(value, str) or not value.strip():
        raise problem("item_id", f"{short(value)} is not an item id",
                      "give the item's id as text, for example cable_3. A whole number such as 3 is turned into text")
    return value


def _run_relative_path(value: Any) -> str:
    text = path_text(value)
    if text.startswith(("/", "\\")) or _DRIVE.match(text):
        raise problem("path_not_relative", f"{short(text)} is an absolute path",
                      "give the path relative to the run folder, for example frames/cycle_0003.parquet")
    return text


ItemId = Annotated[str, Field(min_length=1), BeforeValidator(_item_id)]
RunPath = Annotated[str, Field(min_length=1), BeforeValidator(_run_relative_path)]


class Intervention(Record):
    """One time a person was asked for help: what opened it, what they answered or did, and what it cost.

    1 record serves 3 uses: a label for calibration (``truth_label`` on the ``features`` at the
    fork), a correction demo for policy training (a takeover's handoff and handback frames in the
    cycle's frame log) and a shift metric (the question and the seconds).

    Times run forward: the answer is not before the question, and a handback is not before its
    handoff, in seconds and in frames. A handback needs a handoff. The answer to a confirm question
    is yes, no or cannot_tell, and its truth label agrees with it.
    """

    intervention_id: Id = Field(description="this intervention's id. new_id('iv') makes a fresh one")
    run_id: Id = Field(description="the run that holds the intervention")
    cycle_id: Id = Field(description="the cycle the intervention happened in")
    item: ItemId = Field(description="the item the step was about, for example cable_3. A whole number becomes text")
    step: Id = Field(description="the step of the task the person helped with, for example insert")
    predicate_name: PredicateName | None = Field(
        default=None, description="the predicate the claim at the fork named, for example seated, or null when unknown")
    claim_id: Id | None = Field(default=None, description="the claim at the fork, or null when there was none")
    verdict_id: Id | None = Field(default=None, description="the verdict that opened the fork, or null when there was none")
    fork_id: Id | None = Field(default=None, description="the fork that called the person, or null when there was none")
    assist_id: Id | None = Field(
        default=None, description="the Assist record of the person's answer, or null when none was logged")
    question: QuestionType = Field(description="the type of question the person was asked")
    trigger: Id = Field(
        description="what called the person, in plain words, for example abstain at stage 2 or reject after 2 attempts")
    answer: Id = Field(description="the person's answer: yes, no, cannot_tell, an option, marks or taken over")
    truth_label: Flag | None = Field(
        default=None,
        description="the person's label on the claim as a real boolean, or null when the answer is not a label. For a "
                    "takeover it is what the person saw as they took the arm, the state the features describe")
    asked_ts: Timestamp = Field(description="when the person was asked, in Unix seconds")
    answered_ts: Timestamp = Field(description="when the person answered, in Unix seconds")
    seconds_waiting: NonNegative = Field(description="the seconds the robot waited for the person")
    seconds_handling: NonNegative = Field(description="the seconds the person spent handling the case")
    handoff_ts: Timestamp | None = Field(
        default=None, description="a takeover only: when the person took control, in Unix seconds")
    handback_ts: Timestamp | None = Field(
        default=None, description="a takeover only: when the person gave control back, in Unix seconds")
    handoff_frame: Count | None = Field(
        default=None, description="a takeover only: the first frame of the cycle's frame log under the person's control")
    handback_frame: Count | None = Field(
        default=None, description="a takeover only: the last frame of the cycle's frame log under the person's control")
    frames_path: RunPath | None = Field(
        default=None,
        description="the cycle's frame log, relative to the run folder, for example frames/cycle_0003.parquet")
    outcome: InterventionOutcome = Field(description="how the intervention ended")
    features: dict[FeatureKey, FeatureValue] = Field(
        default_factory=dict,
        description="the evidence features at the fork, a flat dict of finite numbers, for the calibration label")

    @model_validator(mode="after")
    def _times_run_forward_and_labels_agree(self) -> Intervention:
        found = []
        if self.answered_ts < self.asked_ts:
            found.append(("answered_ts", problem(
                "answered_before_asked",
                f"the answer at {self.answered_ts:.3f} comes before the question at {self.asked_ts:.3f}",
                "take both times from the same clock, the answer at or after the question",
            ), self.answered_ts))
        for start, end, unit in (("handoff_ts", "handback_ts", "time"), ("handoff_frame", "handback_frame", "frame")):
            first, last = getattr(self, start), getattr(self, end)
            if last is not None and first is None:
                found.append((end, problem(
                    "handback_without_handoff", f"a handback {unit} is given with no handoff {unit}",
                    f"set {start} as well, or leave {end} null",
                ), last))
            elif first is not None and last is not None and last < first:
                shown = (f"{last:.3f}", f"{first:.3f}") if unit == "time" else (str(last), str(first))
                found.append((end, problem(
                    "handback_before_handoff", f"the handback {unit} {shown[0]} comes before the handoff {unit} {shown[1]}",
                    f"give {end} at or after {start}",
                ), last))
        if self.question == "confirm":
            word = self.answer.strip().casefold().replace(" ", "_")
            if word not in CONFIRM_LABELS:
                found.append(("answer", problem(
                    "confirm_answer", f"{short(self.answer)} is not an answer to a confirm question",
                    f"use yes, no or cannot_tell. {_CONFIRM_FIX}",
                ), self.answer))
            elif CONFIRM_LABELS[word] is not self.truth_label:
                found.append(("truth_label", problem(
                    "confirm_label",
                    f"the answer {short(self.answer)} carries the truth label {short(CONFIRM_LABELS[word])}, "
                    f"and this record says {short(self.truth_label)}",
                    f"set truth_label to match the answer. {_CONFIRM_FIX}",
                ), self.truth_label))
        if found:
            refuse("Intervention", found)
        return self


# ------------------------------------------------------------------ ledger
class CyclePredicate(Record):
    """One predicate a task required, with what settled it. Both ids are null when nothing did."""

    predicate: Predicate = Field(
        description="a predicate the task required, with its target, for example "
                    "{'name': 'seated', 'target': {'plug': 'p1', 'port': 7}}")
    verdict_id: Id | None = Field(default=None, description="the verdict on this predicate, if there was one")
    signoff_assist_id: Id | None = Field(
        default=None, description="the assist in which a person signed this predicate off, if there was one")


class CycleEnd(Record):
    """What the payload of a ``cycle_end`` event must carry.

    The payload of a ledger event is a free dict, so this is the 1 type here that keeps keys it does
    not know: a cycle id, a seed or an arm name ride along untouched and come back out of
    ``model_dump``. The 4 fields below are checked.
    """

    model_config = ConfigDict(extra="allow")

    declared_success: Flag = Field(description="whether the cycle was declared a success, as a real boolean")
    predicates: list[CyclePredicate] = Field(
        description="the predicates the task required, each as {predicate, verdict_id or signoff_assist_id}")
    wall_s: NonNegative = Field(description="the cycle's wall time in seconds")
    true_success: Flag | None = Field(
        default=None,
        description="whether the cycle really succeeded. Sim only, written by the evaluator, and left out otherwise")


_CYCLE_END_FIX = (
    "add it to the payload. A cycle_end payload carries declared_success (true or false), predicates "
    "(a list of {predicate, verdict_id or signoff_assist_id}) and wall_s (seconds), and may carry true_success"
)


def _inlined(schema: dict[str, Any]) -> dict[str, Any]:
    """``schema`` with each local ``$ref`` replaced by what it points to, and no ``$defs``."""
    defs = schema.get("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            rest = {key: walk(item) for key, item in node.items() if key not in ("$ref", "$defs")}
            if "$ref" in node:
                return {**walk(defs[node["$ref"].rsplit("/", 1)[-1]]), **rest}
            return rest
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node

    return walk(schema)


def _cycle_end_rule_schema(schema: dict[str, Any], _model: type) -> None:
    """Say in the JSON Schema what a cycle_end payload must carry. Other keys may ride along."""
    schema["allOf"] = [{
        "if": {"properties": {"kind": {"const": "cycle_end"}}, "required": ["kind"]},
        "then": {"properties": {"payload": _inlined(CycleEnd.model_json_schema())}},
    }]


class LedgerEvent(Record):
    """One line of a run's ledger.

    ``payload`` is a free dict and is stored as given. For the kind ``cycle_end`` it must carry
    ``declared_success``, ``predicates`` and ``wall_s``, and may carry ``true_success`` (see
    ``CycleEnd``). Other keys in a cycle_end payload are left alone. The payloads of the other
    kinds are not checked here: ``validate_record(event.kind, event.payload)`` gives the typed
    record for the kinds in ``PAYLOAD_MODELS``.
    """

    model_config = ConfigDict(json_schema_extra=_cycle_end_rule_schema)

    run_id: Id = Field(description="the run this event belongs to")
    seq: Count = Field(description="the event's position in its run, counting from 0")
    ts: Timestamp = Field(default_factory=time.time, description="when the event was logged, in Unix seconds")
    kind: EventKind = Field(description="the kind of event")
    payload: dict[str, Any] = Field(
        description="the event's content. A cycle_end payload must carry declared_success, predicates and wall_s")

    @field_validator("payload")
    @classmethod
    def _cycle_end_carries_its_fields(cls, value: dict[str, Any], info: ValidationInfo) -> dict[str, Any]:
        if info.data.get("kind") != "cycle_end":
            return value
        missing = [name for name, field in CycleEnd.model_fields.items() if field.is_required() and name not in value]
        if missing:
            refuse("CycleEnd", [
                (name, problem("cycle_end_field", f"a cycle_end event must carry {name}", _CYCLE_END_FIX), value)
                for name in missing
            ])
        CycleEnd.model_validate(value)
        return value


# ------------------------------------------------------------------ registries
RECORD_MODELS: dict[str, type[Record]] = {
    model.__name__: model
    for model in (Cost, Predicate, Claim, Evidence, Witness, Verdict, Fork, Assist, Audit, LedgerEvent, Intervention)
}
"""Every record type by name. ``avail schemas`` writes 1 JSON Schema file per entry."""

PART_MODELS: dict[str, type[Record]] = {
    model.__name__: model
    for model in (ClaimContext, WitnessScope, Stage, Calibration, Shadow, Correction, CyclePredicate, CycleEnd)
}
"""The types that only appear inside a record."""

PAYLOAD_MODELS: dict[str, type[Record]] = {
    "claim": Claim,
    "evidence": Evidence,
    "verdict": Verdict,
    "fork": Fork,
    "assist": Assist,
    "audit": Audit,
    "cycle_end": CycleEnd,
    "intervention": Intervention,
}
"""The record type behind each ledger event kind that has one. ``cycle_start`` and the call logs are free-form."""
