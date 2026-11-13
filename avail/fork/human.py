"""The Human protocol: how a fork asks a person, and what comes back.

A :class:`HumanRequest` carries the claim, the question type, the options and the camera frames.
A :class:`HumanAnswer` carries what the person said or did, the seconds it took and, for a
takeover, the handoff. ``Human.ask`` is the 1 method a person needs. The assist page, the sim's
``OracleHuman`` (testbed code under ``avail/station``, the only Human that may read truth) and
:class:`ScriptedHuman` for tests all have it.

A person's answer never becomes a verdict. A yes to a confirm question is a truth label on the
claim and a sign-off for the done gate (:mod:`avail.fork.gate`), and that is all it can be.
:func:`to_assist` turns a request and its answer into the ``Assist`` record the ledger keeps.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ..schemas import Assist, Claim, Correction, new_id
from ..schemas.base import plain
from ..schemas.kinds import CONFIRM_ANSWERS, CONFIRM_LABELS, CORRECTION_KINDS, QUESTION_TYPES

if TYPE_CHECKING:
    import numpy as np

__all__ = ["Human", "HumanAnswer", "HumanRequest", "ScriptedHuman", "answer_for", "check_answer", "to_assist"]


def _confirm_word(text: str) -> str:
    """``cannot tell`` and ``Cannot_Tell`` are the confirm answer ``cannot_tell``."""
    return text.strip().casefold().replace(" ", "_")


def _seconds(value: Any, name: str) -> float:
    value = plain(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} is {value!r}. Fix: give a finite number of seconds, 0 or more")
    return float(value)


@dataclass(frozen=True)
class HumanRequest:
    """What a person is shown at a fork.

    ``question`` is a question type (confirm, choose, guide_marks, guide_text or takeover).
    ``options`` are the answers offered: a confirm offers yes, no and cannot_tell when none are
    given, and a choose must offer its options. No option may ask for accept: a person signs a
    step off by answering yes to a confirm. ``frames`` maps a camera name to its image.
    """

    fork_id: str
    claim: Claim
    question: str
    options: list[str] = field(default_factory=list)
    frames: dict[str, np.ndarray] | None = None
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.fork_id, str) or not self.fork_id.strip():
            raise ValueError(f"fork_id is {self.fork_id!r}. Fix: give the id of the fork that asks")
        if not isinstance(self.claim, Claim):
            raise TypeError(f"claim is a {type(self.claim).__name__}. Fix: pass the Claim record the fork is about")
        if self.question not in QUESTION_TYPES:
            raise ValueError(f"{self.question!r} is not a question type. Fix: use one of {', '.join(QUESTION_TYPES)}")
        if isinstance(self.options, (str, bytes)) or not hasattr(self.options, "__iter__"):
            raise TypeError(f"options is {self.options!r}. Fix: give a list of the answers offered")
        options = list(self.options) or (list(CONFIRM_ANSWERS) if self.question == "confirm" else [])
        bad = [option for option in options if not isinstance(option, str) or not option.strip()]
        if bad:
            raise ValueError(f"the options {options!r} hold an empty or non-text answer. Fix: give each as text")
        sneaky = [option for option in options if "accept" in option.casefold()]
        if sneaky:
            raise ValueError(f"the option {sneaky[0]!r} asks a person to accept. Fix: ask a confirm question. "
                             "A person signs a step off by answering yes")
        if self.question == "choose" and not options:
            raise ValueError("a choose question offers nothing to choose. Fix: pass the fork's options")
        object.__setattr__(self, "options", options)
        if self.frames is not None and not isinstance(self.frames, Mapping):
            raise TypeError(f"frames is a {type(self.frames).__name__}. Fix: give a dict of camera name to image")
        if not isinstance(self.context, Mapping):
            raise TypeError(f"context is a {type(self.context).__name__}. Fix: give a dict")


@dataclass(frozen=True)
class HumanAnswer:
    """What a person answered, or did, and what it cost.

    ``truth_label`` is the person's label on the claim as a real boolean, or None when the answer
    is not a label. ``correction_kind`` (marks, step_text, pose_offset or takeover) names the form
    of ``correction``. ``handoff`` is set for a takeover: ``handoff_ts``, ``handback_ts``,
    ``handoff_frame`` and ``handback_frame``, as the ``Intervention`` record takes them.
    """

    answer: str
    truth_label: bool | None = None
    correction_kind: str | None = None
    correction: dict[str, Any] = field(default_factory=dict)
    seconds_to_answer: float = 0.0
    seconds_handling: float = 0.0
    handoff: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.answer, str) or not self.answer.strip():
            raise ValueError(f"answer is {self.answer!r}. Fix: give the person's answer as text")
        label = plain(self.truth_label)
        if label is not None and not isinstance(label, bool):
            raise ValueError(f"truth_label is {self.truth_label!r}. Fix: give True, False or None. Text such as "
                             "'yes' and numbers are refused on purpose")
        object.__setattr__(self, "truth_label", label)
        if self.correction_kind is not None and self.correction_kind not in CORRECTION_KINDS:
            raise ValueError(f"{self.correction_kind!r} is not a correction kind. Fix: use one of "
                             f"{', '.join(CORRECTION_KINDS)}")
        if not isinstance(self.correction, Mapping):
            raise TypeError(f"correction is a {type(self.correction).__name__}. Fix: give a dict")
        if self.correction and self.correction_kind is None:
            raise ValueError("a correction is given with no kind. Fix: set correction_kind")
        object.__setattr__(self, "correction", dict(self.correction))
        object.__setattr__(self, "seconds_to_answer", _seconds(self.seconds_to_answer, "seconds_to_answer"))
        object.__setattr__(self, "seconds_handling", _seconds(self.seconds_handling, "seconds_handling"))
        if self.handoff is not None and not isinstance(self.handoff, Mapping):
            raise TypeError(f"handoff is a {type(self.handoff).__name__}. Fix: give a dict, or None")


@runtime_checkable
class Human(Protocol):
    """A person who can be asked at a fork."""

    def ask(self, request: HumanRequest) -> HumanAnswer:
        """Show the request, wait for the person, return the answer."""
        ...


def answer_for(request: HumanRequest, word: str, **more: Any) -> HumanAnswer:
    """A plain answer to a request. The answer to a confirm question carries its own truth label.

    ``more`` takes the other fields of :class:`HumanAnswer`, such as ``seconds_to_answer``.
    """
    given_label = more.pop("truth_label", None)
    label = CONFIRM_LABELS.get(_confirm_word(word)) if request.question == "confirm" else given_label
    return HumanAnswer(answer=word, truth_label=label, **more)


def check_answer(request: HumanRequest, answer: HumanAnswer) -> HumanAnswer:
    """``answer`` when it fits ``request``, else ValueError.

    A confirm is answered yes, no or cannot_tell, with the truth label that answer carries. A choose
    is answered with 1 of its options.
    """
    if not isinstance(answer, HumanAnswer):
        raise TypeError(f"the answer is a {type(answer).__name__}. Fix: return a HumanAnswer")
    if request.question == "confirm":
        word = _confirm_word(answer.answer)
        if word not in CONFIRM_LABELS:
            raise ValueError(f"{answer.answer!r} is not an answer to a confirm question. Fix: answer yes, no or "
                             "cannot_tell")
        if CONFIRM_LABELS[word] is not answer.truth_label:
            raise ValueError(f"the answer {answer.answer!r} carries the truth label {CONFIRM_LABELS[word]}, and the "
                             f"answer says {answer.truth_label}. Fix: give the label the answer carries")
    elif request.question == "choose" and answer.answer not in request.options:
        raise ValueError(f"{answer.answer!r} is not one of the options {request.options}. Fix: answer with 1 of them")
    return answer


class ScriptedHuman:
    """A person for tests and dry runs: ``script(request)`` gives each answer.

    The script returns a ``HumanAnswer``, a dict of its fields, or 1 word (see :func:`answer_for`).
    Every answer is checked against its request, and every request is kept in ``asked``.
    """

    def __init__(self, script: Callable[[HumanRequest], HumanAnswer | Mapping[str, Any] | str],
                 assistant_id: str = "scripted") -> None:
        if not callable(script):
            raise TypeError("script is not callable. Fix: pass a function of the request")
        self.script = script
        self.assistant_id = assistant_id
        self.asked: list[HumanRequest] = []

    def ask(self, request: HumanRequest) -> HumanAnswer:
        if not isinstance(request, HumanRequest):
            raise TypeError(f"the request is a {type(request).__name__}. Fix: pass a HumanRequest")
        self.asked.append(request)
        given = self.script(request)
        if isinstance(given, str):
            given = answer_for(request, given)
        elif isinstance(given, Mapping):
            given = HumanAnswer(**given)
        return check_answer(request, given)


def to_assist(request: HumanRequest, answer: HumanAnswer, assistant_id: str, *,
              assist_id: str | None = None) -> Assist:
    """The ``Assist`` record of 1 answer, ready for the ledger."""
    correction = None
    if answer.correction_kind is not None:
        correction = Correction(kind=answer.correction_kind, payload=dict(answer.correction))
    return Assist(assist_id=assist_id or new_id("assist"), fork_id=request.fork_id, question=request.question,
                  options=list(request.options), answer=answer.answer, truth_label=answer.truth_label,
                  correction=correction, seconds_to_answer=answer.seconds_to_answer,
                  seconds_handling=answer.seconds_handling, assistant_id=assistant_id)
