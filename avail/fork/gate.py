"""The done gate behind ``avail_done``: ok only when every required predicate has an accept or a person's yes.

For each item of a cycle the task file requires 1 predicate per step (``TaskSpec.required(variant)``,
for example insert needs seated and release needs released_stable). :class:`DoneGate` keeps, for each
item and predicate, what settled it. 2 things can, and nothing else:

- :meth:`DoneGate.record_accept`: a ``Verdict`` record whose decision is accept. The record itself
  guarantees that it lists its evidence (CLAUDE.md rule 11). Text, a dict and a reject are refused.
- :meth:`DoneGate.record_signoff`: the ``Assist`` or ``Intervention`` record of a person's yes, a
  truth label of true. An answer that is not a yes is refused, and so is a bare id, which cannot
  show what the person said.

:meth:`DoneGate.done` and :meth:`DoneGate.cycle_done` say ok or list what is unresolved.
:meth:`DoneGate.from_ledger` rebuilds the same state from a run's ledger events, so ``avail_done``
can be answered from the ledger alone::

    gate = DoneGate(load_task("cable_loop"), variant="latch").from_ledger(iter_events(run_dir), "cycle_0003")
    ok, unresolved = gate.cycle_done()       # the items come from the cycle's cycle_start event
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from ..schemas import Assist, Claim, Fork, Intervention, Verdict
from ..schemas.base import plain

if TYPE_CHECKING:
    from ..core.task import TaskSpec

__all__ = ["DoneGate"]

_DONE_RULE = "Only evidence or a person marks a step done"


def _item_key(item: Any) -> str:
    """The key of an item: its id as text. The whole number 3 and the text 3 are the same item."""
    item = plain(item)
    if isinstance(item, int) and not isinstance(item, bool):
        return str(item)
    if isinstance(item, str) and item.strip():
        return item
    raise ValueError(f"item is {item!r}. Fix: give the item's id as text or a whole number")


def _items_kind(task: Any) -> str | None:
    """The task's item kind, such as cable: the target key that names the item in an older claim."""
    items = getattr(task, "items", None)
    kind = items.get("kind") if isinstance(items, Mapping) else getattr(items, "kind", None)
    return kind if isinstance(kind, str) and kind else None


def _event_rows(events: Any) -> Iterator[tuple[Any, Any]]:
    """(kind, payload) for each ledger event, from a run folder, a run id, a ledger table or a list of events."""
    if isinstance(events, (str, os.PathLike)):
        from ..ledger.reader import iter_events

        events = iter_events(events)
    table = getattr(events, "events", None)                 # what load_ledger returns
    if table is not None and hasattr(table, "columns"):
        events = table
    if hasattr(events, "columns") and hasattr(events, "itertuples"):   # a pandas frame of events
        if "kind" not in events.columns or "payload" not in events.columns:
            raise ValueError("the events table has no kind and payload columns. Fix: pass load_ledger(run).events")
        yield from zip(events["kind"], events["payload"])
        return
    for event in events:
        if isinstance(event, Mapping):
            yield event.get("kind"), event.get("payload")
        elif isinstance(event, BaseModel) and hasattr(event, "kind") and hasattr(event, "payload"):
            yield event.kind, event.payload
        else:
            raise TypeError(f"a ledger event is a {type(event).__name__}. Fix: pass the events iter_events gives")


class DoneGate:
    """What settled each required predicate of each item in 1 cycle. Nothing marks a step done but evidence or a person.

    ``task`` is a task spec (``avail.core.task.TaskSpec``): the gate uses ``required(variant)``,
    ``predicates`` and ``items``. ``variant`` is the cell's variant, for the plug station the
    connector type, which picks each step's predicate.
    """

    def __init__(self, task: TaskSpec, variant: str) -> None:
        if not callable(getattr(task, "required", None)):
            raise TypeError(f"task is a {type(task).__name__} with no required(variant). Fix: pass a TaskSpec, "
                            "for example load_task('cable_loop')")
        if not isinstance(variant, str) or not variant.strip():
            raise ValueError(f"variant is {variant!r}. Fix: give the cell's variant, for example its connector type")
        required = tuple((str(step), str(predicate)) for step, predicate in task.required(variant))
        if not required:
            raise ValueError(f"the task requires no predicate for the variant {variant}, so the gate could never "
                             "say done. Fix: check the task file's steps")
        self.task, self.variant = task, variant
        self.required: tuple[tuple[str, str], ...] = required
        """(step, predicate) for each step of an item, in order."""
        known = getattr(task, "predicates", None)
        self.known: frozenset[str] = frozenset(known) if known else frozenset(p for _, p in required)
        self.items: list[Any] = []
        """The items of the cycle, as ``from_ledger`` found them in the cycle's ``cycle_start`` event."""
        self.problems: list[str] = []
        """Ledger events ``from_ledger`` could not use, as sentences for a report."""
        self._item_kind = _items_kind(task)
        self._accepts: dict[tuple[str, str], str] = {}       # (item, predicate) -> verdict id
        self._signoffs: dict[tuple[str, str], str] = {}      # (item, predicate) -> assist or intervention id

    # ---------------------------------------------------------------- what may settle a predicate
    def _predicate(self, predicate: Any) -> str:
        if not isinstance(predicate, str) or predicate not in self.known:
            name = getattr(self.task, "task", "the task")
            raise ValueError(f"{name} defines no predicate {predicate!r}. Fix: use one of {', '.join(sorted(self.known))}")
        return predicate

    def record_accept(self, item: Any, predicate: str, verdict: Verdict) -> None:
        """Mark ``predicate`` of ``item`` settled by a witness's accept. Anything but an accepting Verdict is refused."""
        if not isinstance(verdict, Verdict):
            raise TypeError(f"record_accept takes the Verdict record a witness built, and got a "
                            f"{type(verdict).__name__}. Fix: pass the verdict that verify returned. {_DONE_RULE}")
        if verdict.decision != "accept":
            raise ValueError(f"the verdict {verdict.verdict_id} is {verdict.decision}, and only an accept settles a "
                             f"predicate. Fix: follow the fork it opens. {_DONE_RULE}")
        self._accepts.setdefault((_item_key(item), self._predicate(predicate)), verdict.verdict_id)

    def record_signoff(self, item: Any, predicate: str, assist: Assist | Intervention) -> None:
        """Mark ``predicate`` of ``item`` settled by a person's yes: an Assist or Intervention with truth label true.

        A bare assist id is refused because it cannot show what the person said.
        """
        if not isinstance(assist, (Assist, Intervention)):
            raise TypeError(f"record_signoff takes the Assist or Intervention record of the person's answer, and got a "
                            f"{type(assist).__name__}. Fix: pass the record, whose truth_label is true for a yes. A "
                            f"bare id or a word cannot show what the person said. {_DONE_RULE}")
        if assist.truth_label is not True:
            raise ValueError(f"the person's answer {assist.answer!r} is not a yes on the claim (truth_label "
                             f"{assist.truth_label}), so it signs nothing off. Fix: follow the answer instead. "
                             f"{_DONE_RULE}")
        ident = assist.assist_id if isinstance(assist, Assist) else (assist.assist_id or assist.intervention_id)
        self._signoffs.setdefault((_item_key(item), self._predicate(predicate)), ident)

    # ---------------------------------------------------------------- the gate
    def settled(self, item: Any, predicate: str) -> dict[str, str | None] | None:
        """What settled ``predicate`` of ``item``: ``{verdict_id, signoff_assist_id}``, or None when nothing has."""
        key = (_item_key(item), predicate)
        if key not in self._accepts and key not in self._signoffs:
            return None
        return {"verdict_id": self._accepts.get(key), "signoff_assist_id": self._signoffs.get(key)}

    def done(self, item: Any) -> tuple[bool, list[str]]:
        """(ok, the required predicates of ``item`` that nothing settled yet, in step order)."""
        unresolved: list[str] = []
        for _step, predicate in self.required:
            if self.settled(item, predicate) is None and predicate not in unresolved:
                unresolved.append(predicate)
        return not unresolved, unresolved

    def cycle_done(self, items: Iterable[Any] | None = None) -> tuple[bool, list[dict[str, Any]]]:
        """(ok, ``{item, step, predicate}`` for each required predicate not settled), over every item of the cycle.

        ``items`` defaults to the items ``from_ledger`` found in the cycle's ``cycle_start`` event. A
        cycle with no items is refused: it has nothing that could be done.
        """
        if items is None:
            items = self.items
        if isinstance(items, (str, bytes, int)):
            raise TypeError(f"items is {items!r}. Fix: give a list of the cycle's items, for example [0, 1, 2, 3, 4]")
        items = list(items)
        if not items:
            raise ValueError("no items were given, so nothing could be done. Fix: pass every item of the cycle, or "
                             "rebuild the gate from a ledger whose cycle_start event lists them under items")
        unresolved: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in items:
            key = _item_key(item)
            if key in seen:
                continue
            seen.add(key)
            for step, predicate in self.required:
                if self.settled(key, predicate) is None:
                    unresolved.append({"item": item, "step": step, "predicate": predicate})
        return not unresolved, unresolved

    def entries(self, item: Any) -> list[dict[str, Any]]:
        """``{step, predicate, verdict_id, signoff_assist_id}`` for each required predicate, for a cycle_end payload."""
        out = []
        for step, predicate in self.required:
            got = self.settled(item, predicate) or {"verdict_id": None, "signoff_assist_id": None}
            out.append({"step": step, "predicate": predicate, **got})
        return out

    # ---------------------------------------------------------------- from the ledger
    def _read(self, model: type[BaseModel], kind: str, payload: Any) -> Any:
        """A logged record read back, or None (and a problem) when it does not validate."""
        try:
            return model.model_validate(payload)
        except ValidationError as exc:
            first = exc.errors(include_url=False)[0]
            where = ".".join(str(step) for step in first.get("loc", ())) or kind
            self.problems.append(f"a {kind} event does not read back as a {model.__name__} ({where}: {first['msg']})")
            return None

    def _item_of(self, claim: Claim) -> str | None:
        target = claim.predicate.target
        for key in ("item", self._item_kind):
            if key and key in target:
                return _item_key(target[key])
        return None

    def from_ledger(self, events: Any, cycle_id: str) -> DoneGate:
        """Rebuild the gate for ``cycle_id`` from ledger events and return it. What was recorded before is dropped.

        ``events`` is a run folder or run id, ``load_ledger(...)``, its ``events`` table, or a list of
        event dicts as ``iter_events`` gives them. An accept counts when its claim belongs to the
        cycle. A person's yes counts through its fork, verdict and claim, or as an ``intervention``
        event of the cycle. The item of a claim is ``target["item"]``, or the target key named by the
        task's item kind. Events that do not read back are listed in ``problems`` and never count.
        """
        if not isinstance(cycle_id, str) or not cycle_id.strip():
            raise ValueError(f"cycle_id is {cycle_id!r}. Fix: give the id of the cycle to check")
        self._accepts.clear()
        self._signoffs.clear()
        self.items, self.problems = [], []
        claims: dict[str, tuple[str, str]] = {}          # this cycle's claims: claim id -> (item, predicate)
        verdicts: list[Verdict] = []
        claim_of_verdict: dict[str, str] = {}
        verdict_of_fork: dict[str, str] = {}
        assists: list[Assist] = []
        interventions: list[Intervention] = []
        for kind, payload in _event_rows(events):
            if not isinstance(payload, Mapping):
                self.problems.append(f"a {kind} event has no payload dict")
                continue
            if kind == "cycle_start":
                if payload.get("cycle_id") == cycle_id and isinstance(payload.get("items"), list):
                    self.items = list(payload["items"])
            elif kind == "claim":
                claim = self._read(Claim, kind, payload)
                if claim is None or claim.cycle_id != cycle_id:
                    continue
                item = self._item_of(claim)
                if item is None:
                    self.problems.append(f"the claim {claim.claim_id} names no item in its target "
                                         f"{claim.predicate.target}, so it cannot settle anything")
                    continue
                claims[claim.claim_id] = (item, claim.predicate.name)
            elif kind == "verdict":
                verdict = self._read(Verdict, kind, payload)
                if verdict is not None:
                    verdicts.append(verdict)
                    claim_of_verdict[verdict.verdict_id] = verdict.claim_id
            elif kind == "fork":
                fork = self._read(Fork, kind, payload)
                if fork is not None:
                    verdict_of_fork[fork.fork_id] = fork.verdict_id
            elif kind == "assist":
                assist = self._read(Assist, kind, payload)
                if assist is not None:
                    assists.append(assist)
            elif kind == "intervention":
                record = self._read(Intervention, kind, payload)
                if record is not None and record.cycle_id == cycle_id:
                    interventions.append(record)
        for verdict in verdicts:
            if verdict.decision == "accept" and verdict.claim_id in claims:
                item, predicate = claims[verdict.claim_id]
                if predicate in self.known:
                    self._accepts.setdefault((item, predicate), verdict.verdict_id)
        for assist in assists:
            claim_id = claim_of_verdict.get(verdict_of_fork.get(assist.fork_id, ""))
            if assist.truth_label is True and claim_id in claims:
                item, predicate = claims[claim_id]
                if predicate in self.known:
                    self._signoffs.setdefault((item, predicate), assist.assist_id)
        step_predicate = dict(self.required)
        for record in interventions:
            if record.truth_label is not True:
                continue
            predicate = claims[record.claim_id][1] if record.claim_id in claims else (
                record.predicate_name or step_predicate.get(record.step))
            if predicate in self.known:
                self._signoffs.setdefault((_item_key(record.item), predicate),
                                          record.assist_id or record.intervention_id)
        return self

    def __repr__(self) -> str:
        return (f"DoneGate({getattr(self.task, 'task', type(self.task).__name__)!r}, {self.variant!r}: "
                f"{len(self._accepts)} accepts, {len(self._signoffs)} sign-offs)")
