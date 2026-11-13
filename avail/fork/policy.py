"""The default fork policy and the allowed-options contract (docs/05).

A fork opens when a verdict is reject or abstain. The policy reads the verdict's decision and
2 counts, and nothing else: no evidence, no frames, no text from a model. This file is under the
truth wall (avail/walls.py).

| Verdict | A person is there | Nobody is there |
|---|---|---|
| reject, tries left | retry | retry |
| reject on the last try | ask_human with the question ``reject_last`` (default choose) | abort, unresolved |
| abstain | ask_human with the question ``abstain`` (default confirm) | abort, unresolved |

``accept`` opens no fork. No plan can hold it: :class:`ForkPlan` refuses it in its options and
in its pick, and so does the ``Fork`` record :func:`fork_record` builds from a plan.

``ask`` is the task file's ``fork.ask`` block, for example ``{abstain: confirm, reject_last: choose,
at_port_reject: guide_marks}``. A key ``<predicate>_reject`` sets the question for the last try of
that predicate in place of ``reject_last``. It applies when :func:`decide` is told the predicate.

Example::

    plan = decide(verdict.decision, claim.context.attempt, task.fork["max_attempts"], human=True,
                  ask=task.fork.get("ask", {}), predicate=claim.predicate.name)
    fork = fork_record(verdict.verdict_id, plan)       # chosen_by policy, logged as a fork event
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..schemas import Fork, is_fork_option, new_id
from ..schemas.kinds import CHOSEN_BY, DECISIONS, QUESTION_TYPES, ChosenBy, Decision

__all__ = ["DEFAULT_QUESTIONS", "ForkPlan", "choose", "decide", "fork_record"]

DEFAULT_QUESTIONS: dict[str, str] = {"abstain": "confirm", "reject_last": "choose"}
"""The question a person gets when the task file's ``fork.ask`` says nothing."""

_ASK_KEY = re.compile(r"[a-z][a-z0-9_]*")       # what the task loader accepts. The policy reads 3 forms of key
_MENU_FIX = "use options from retry, ask_human, abort, repair:<probe> and recover:<skill>. accept is never one"


@dataclass(frozen=True)
class ForkPlan:
    """What the policy offers at a fork and what it picks.

    ``options`` is the menu: whoever decides picks from it and from nothing else. ``chosen`` is the
    policy's pick, or None to leave the pick to an agent or a person. ``question`` is the question
    type for ask_human and is set when the pick is ask_human. ``reason`` says why, in plain words.
    """

    options: list[str]
    chosen: str | None
    question: str | None
    reason: str

    def __post_init__(self) -> None:
        if isinstance(self.options, (str, bytes)) or not hasattr(self.options, "__iter__"):
            raise TypeError(f"options is {self.options!r}. Fix: give a list of fork options")
        options = list(self.options)
        object.__setattr__(self, "options", options)        # a list the plan owns, never the caller's
        if not options:
            raise ValueError(f"a fork plan with no options leaves nothing to choose. Fix: {_MENU_FIX}")
        bad = [option for option in options if not is_fork_option(option)]
        if bad:
            raise ValueError(f"{', '.join(map(repr, bad))} is not a fork option. Fix: {_MENU_FIX}")
        if len(set(options)) != len(options):
            raise ValueError(f"the options {options} list an option more than once. Fix: list each option once")
        if self.chosen is not None and self.chosen not in options:
            raise ValueError(f"the pick {self.chosen!r} is not on the menu {options}. Fix: pick one of the options")
        if self.question is not None:
            if self.question not in QUESTION_TYPES:
                raise ValueError(f"{self.question!r} is not a question type. Fix: use one of {', '.join(QUESTION_TYPES)}")
            if "ask_human" not in options:
                raise ValueError("a plan with a question must offer ask_human. Fix: add ask_human to the options, "
                                 "or drop the question")
        elif self.chosen == "ask_human":
            raise ValueError(f"a plan that picks ask_human must say what to ask. Fix: give a question, one of "
                             f"{', '.join(QUESTION_TYPES)}")
        if not isinstance(self.reason, str):
            raise TypeError(f"reason is {self.reason!r}. Fix: say why in plain words")


def _whole(value: Any, name: str, least: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < least:
        raise ValueError(f"{name} is {value!r}. Fix: give a whole number of {least} or more")
    return value


def _questions(ask: Mapping[str, Any] | None) -> dict[str, str]:
    """The task file's ``fork.ask`` over the defaults, every key and question checked."""
    if ask is None:
        return dict(DEFAULT_QUESTIONS)
    if not isinstance(ask, Mapping):
        raise ValueError(f"ask is {ask!r}. Fix: give the task file's fork.ask block, for example "
                         "{'abstain': 'confirm', 'reject_last': 'choose'}")
    for key, question in ask.items():
        if not isinstance(key, str) or not _ASK_KEY.fullmatch(key):
            raise ValueError(f"fork.ask has the key {key!r}. Fix: name what opens the fork, as abstain, reject_last "
                             "or <predicate>_reject")
        if question not in QUESTION_TYPES:
            raise ValueError(f"fork.ask gives {key} the question {question!r}. Fix: use one of "
                             f"{', '.join(QUESTION_TYPES)}")
    return {**DEFAULT_QUESTIONS, **ask}


def _tries(count: int) -> str:
    return f"{count} {'try' if count == 1 else 'tries'}"


def decide(verdict_decision: Decision, attempt: int, max_attempts: int, human: bool,
           ask: Mapping[str, Any] | None = None, *, predicate: str | None = None) -> ForkPlan:
    """The default policy's plan for a fork.

    ``verdict_decision`` is the verdict's decision, reject or abstain. ``attempt`` is the try that
    just ended, counting from 0 as ``ClaimContext.attempt`` does, and ``max_attempts`` is how many
    tries the step gets in all (the task file's ``fork.max_attempts``). ``human`` says whether a
    person can be asked. ``ask`` is the task file's ``fork.ask`` block. ``predicate`` picks a
    ``<predicate>_reject`` question for the last try.

    An accept opens no fork and raises ValueError, as does anything that is not a decision.
    """
    if verdict_decision not in DECISIONS:
        raise ValueError(f"{verdict_decision!r} is not a verdict decision. Fix: pass verdict.decision, which is "
                         "reject or abstain at a fork")
    if verdict_decision == "accept":
        raise ValueError("an accept opens no fork: the step is done. Fix: call decide only for a reject or an abstain")
    attempt = _whole(attempt, "attempt", 0)
    max_attempts = _whole(max_attempts, "max_attempts", 1)
    if not isinstance(human, bool):
        raise ValueError(f"human is {human!r}. Fix: say with True or False whether a person can be asked")
    questions = _questions(ask)
    if predicate is not None and (not isinstance(predicate, str) or not predicate.strip()):
        raise ValueError(f"predicate is {predicate!r}. Fix: give the claim's predicate name, or leave it out")
    left = max_attempts - attempt - 1
    if verdict_decision == "reject" and left > 0:
        menu = ["retry", "ask_human", "abort"] if human else ["retry", "abort"]
        return ForkPlan(menu, "retry", None,
                        f"reject on try {attempt + 1} of {max_attempts}, {_tries(left)} left: retry")
    if verdict_decision == "reject":
        if human:
            question = questions["reject_last"]
            if predicate is not None:
                question = questions.get(f"{predicate}_reject", question)
            return ForkPlan(["ask_human", "abort"], "ask_human", question,
                            f"reject on the last of {_tries(max_attempts)}: ask a person ({question})")
        return ForkPlan(["abort"], "abort", None,
                        f"reject on the last of {_tries(max_attempts)} and nobody to ask: abort, the step stays unresolved")
    if human:
        question = questions["abstain"]
        return ForkPlan(["ask_human", "abort"], "ask_human", question, f"abstain: ask a person ({question})")
    return ForkPlan(["abort"], "abort", None, "abstain and nobody to ask: abort, the step stays unresolved")


def fork_record(verdict_id: str, plan: ForkPlan, *, fork_id: str | None = None,
                chosen_by: ChosenBy = "policy") -> Fork:
    """The ``Fork`` record of a plan: its menu, the policy's pick and why. The record checks it all again.

    A plan with no pick gives an open fork, with ``chosen_by`` left null.
    """
    if not isinstance(plan, ForkPlan):
        raise TypeError(f"plan is a {type(plan).__name__}. Fix: pass the ForkPlan that decide returned")
    if chosen_by not in CHOSEN_BY:
        raise ValueError(f"chosen_by is {chosen_by!r}. Fix: use one of {', '.join(CHOSEN_BY)}")
    return Fork(fork_id=fork_id or new_id("fork"), verdict_id=verdict_id, options=list(plan.options),
                chosen=plan.chosen, chosen_by=chosen_by if plan.chosen is not None else None, reason=plan.reason)


def choose(fork: Fork, option: str, chosen_by: ChosenBy, reason: str | None = None) -> Fork:
    """A copy of ``fork`` with ``option`` picked by an agent or a person. The record refuses an option off its menu."""
    update: dict[str, Any] = {"chosen": option, "chosen_by": chosen_by}
    if reason is not None:
        update["reason"] = reason
    return fork.model_copy(update=update)
