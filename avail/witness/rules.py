"""The rule witness: thresholds from a task file on named features, for a predicate with no calibrated cascade.

A task file decides such a predicate with a rule::

    decide:
      kind: rule
      accept_all: [{feature: passive.jaw_hold/in_band, above: 0.5}, {feature: vision.cable_tag/match, above: 0.5}]
      reject_any: [{feature: passive.jaw_hold/closed_empty, above: 0.5}]

:class:`RuleWitness` collects every check named for the predicate, in order, through the evidence
registry, and names each feature ``<check>/<feature>``, for example ``vision.gap@now/plug_found``.
A check written ``source@view`` runs the source with that view, as in the cascade. Then:

- reject when any reject condition holds, so a sign of failure always blocks an accept
- accept when every accept condition holds and there is at least 1 accept condition
- abstain otherwise, which sends the claim to a person

``above`` and ``below`` are strict, as the cascade's thresholds are: a value equal to a threshold
does not meet it. A condition on a feature that no check reported does not hold, so missing
evidence never accepts and never rejects.

The thresholds come from the task file and nothing here calibrates them. Every verdict names the rule's version.
Like the cascade, this module builds ``Verdict`` records (CLAUDE.md rule 11). Each verdict lists the
evidence it stands on, and the record refuses an accept with none.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..schemas import Cost, Evidence, Verdict, new_id

ACCEPT, REJECT, ABSTAIN = "accept", "reject", "abstain"
_CONDITION_KEYS = frozenset({"feature", "above", "below"})
_DECIDE_KEYS = frozenset({"kind", "accept_all", "reject_any", "witness"})
_EXAMPLE = "{feature: passive.jaw_hold/in_band, above: 0.5}"


def _plain(spec: Any) -> Any:
    """A pydantic model or a plain object as a dict. A mapping or anything else comes back unchanged."""
    if hasattr(spec, "model_dump") and callable(spec.model_dump):
        return spec.model_dump()
    if not isinstance(spec, Mapping) and hasattr(spec, "__dict__") and not isinstance(spec, type):
        return {key: value for key, value in vars(spec).items() if not key.startswith("_")}
    return spec


def _number(value: Any, where: str, key: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, str, bytes)):
        raise ValueError(f"{where}: {key} is {value!r}, which is not a number. Fix: give a finite number, "
                         f"for example {_EXAMPLE}")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{where}: {key} is {value!r}, which is not a number. Fix: give a finite number") from None
    if not math.isfinite(number):
        raise ValueError(f"{where}: {key} is {value!r}. Fix: give a finite number")
    return number


def _shown(value: float) -> str:
    return f"{value:g}"


@dataclass(frozen=True)
class RuleCondition:
    """1 threshold on 1 named feature. ``above`` and ``below`` are strict. Both together make a band."""

    feature: str
    above: float | None = None
    below: float | None = None

    @property
    def check(self) -> str:
        """The check that reports the feature: ``vision.gap@now`` for ``vision.gap@now/plug_found``."""
        return self.feature.partition("/")[0]

    def holds(self, features: Mapping[str, float]) -> bool:
        """True when the feature was measured and meets every bound."""
        value = features.get(self.feature)
        if value is None or not math.isfinite(value):
            return False
        if self.above is not None and not value > self.above:
            return False
        if self.below is not None and not value < self.below:
            return False
        return True

    def bounds(self) -> str:
        """The bounds in words: ``above 0.5``, ``below 0.8`` or ``above 0.5 and below 2``."""
        parts = [f"above {_shown(self.above)}" if self.above is not None else "",
                 f"below {_shown(self.below)}" if self.below is not None else ""]
        return " and ".join(part for part in parts if part)

    def says(self, features: Mapping[str, float]) -> str:
        """What was measured against the condition: ``passive.jaw_hold/in_band is 0 and needs to be above 0.5``."""
        value = features.get(self.feature)
        if value is None:
            return f"{self.feature} was not measured"
        if self.holds(features):
            return f"{self.feature} is {_shown(value)}, {self.bounds()}"
        return f"{self.feature} is {_shown(value)} and needs to be {self.bounds()}"

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"feature": self.feature}
        if self.above is not None:
            out["above"] = self.above
        if self.below is not None:
            out["below"] = self.below
        return out

    @classmethod
    def parse(cls, raw: Any, where: str) -> RuleCondition:
        """A condition from a dict such as ``{feature: ..., above: 0.5}`` or a model with those fields."""
        raw = _plain(raw)
        if not isinstance(raw, Mapping):
            raise ValueError(f"{where} is {raw!r}, which is not a condition. Fix: write it as {_EXAMPLE}")
        unknown = sorted(str(key) for key in raw if key not in _CONDITION_KEYS)
        if unknown:
            raise ValueError(f"{where} has the key {', '.join(unknown)}, which a condition does not take. "
                             "Fix: use feature, above and below")
        feature = raw.get("feature")
        check, _, name = feature.partition("/") if isinstance(feature, str) else ("", "", "")
        if not check.strip() or not name.strip():
            raise ValueError(f"{where} names the feature {feature!r}. Fix: name it <check>/<feature>, "
                             "for example passive.jaw_hold/in_band")
        above, below = _number(raw.get("above"), where, "above"), _number(raw.get("below"), where, "below")
        if above is None and below is None:
            raise ValueError(f"{where} has no threshold. Fix: give above, below or both, for example {_EXAMPLE}")
        if above is not None and below is not None and not above < below:
            raise ValueError(f"{where} asks for a value above {_shown(above)} and below {_shown(below)}, which no "
                             "value meets. Fix: give a band whose above is less than its below")
        return cls(feature, above, below)


@dataclass(frozen=True)
class RuleJudgement:
    """What a rule made of 1 set of features, condition by condition."""

    decision: str
    held: tuple[RuleCondition, ...]          # accept conditions that hold
    unmet: tuple[RuleCondition, ...]         # accept conditions that do not, a missing feature included
    fired: tuple[RuleCondition, ...]         # reject conditions that hold
    missing: tuple[str, ...]                 # features a condition names that no check reported
    values: tuple[tuple[str, float], ...]    # each named feature that was measured, with its value

    @property
    def reason(self) -> str:
        """1 line in plain words, for a fork's reason or an evidence summary."""
        measured = dict(self.values)
        if self.decision == REJECT:
            return "reject: " + ", ".join(condition.says(measured) for condition in self.fired)
        if self.decision == ACCEPT:
            return f"accept: all {len(self.held)} accept conditions hold"
        if not self.held and not self.unmet:
            return "abstain: no reject condition holds, and this rule has no accept condition"
        return "abstain: " + ", ".join(condition.says(measured) for condition in self.unmet)


class RuleWitness:
    """The task file's rule for 1 predicate. ``verify`` returns what ``Cascade.verify`` returns.

    ``spec`` is the predicate's ``decide`` block (a dict or a model with ``accept_all`` and
    ``reject_any``), or the whole predicate entry with ``checks`` and ``decide``. ``checks`` are the
    evidence sources to collect, in order. When they are not given they come from the predicate
    entry, and failing that from the conditions, in the order the conditions name them.

    A bad spec raises ValueError when the witness is built, so a task file with a typo fails at
    start and never during a cycle.
    """

    kind = "rule"

    def __init__(self, predicate: str, spec: Any, version: str, checks: Sequence[str] | None = None):
        if not isinstance(predicate, str) or not predicate.strip():
            raise ValueError(f"predicate is {predicate!r}. Fix: give the predicate's name, for example held")
        if not isinstance(version, str) or not version.strip():
            raise ValueError(f"version is {version!r}. Fix: give the rule a version label, for example rule-v1")
        where = f"the rule for {predicate}"
        decide = _plain(spec)
        if isinstance(decide, Mapping) and "decide" in decide:          # the whole predicate entry
            if checks is None:
                checks = decide.get("checks")
            decide = _plain(decide["decide"])
        if not isinstance(decide, Mapping):
            raise ValueError(f"{where} is {decide!r}, which is not a decide block. Fix: give a dict with kind: rule, "
                             "accept_all and reject_any")
        kind = decide.get("kind", "rule")
        if kind != "rule":
            raise ValueError(f"{where}: the predicate is decided by {kind!r}. Fix: a RuleWitness takes kind: rule. "
                             "A cascade comes from the witness store")
        unknown = sorted(str(key) for key in decide if key not in _DECIDE_KEYS)
        if unknown:
            raise ValueError(f"{where} has the key {', '.join(unknown)}, which a rule does not take. "
                             "Fix: use kind, accept_all and reject_any")
        if decide.get("witness") is not None:
            raise ValueError(f"{where} names the witness {decide['witness']!r}. Fix: a rule takes no witness name")
        self.predicate, self.version = predicate, version
        self.accept_all = tuple(RuleCondition.parse(raw, f"{where}, accept_all item {i + 1}")
                                for i, raw in enumerate(decide.get("accept_all") or ()))
        self.reject_any = tuple(RuleCondition.parse(raw, f"{where}, reject_any item {i + 1}")
                                for i, raw in enumerate(decide.get("reject_any") or ()))
        if not self.accept_all and not self.reject_any:
            raise ValueError(f"{where} has no condition, so it would only ever abstain. "
                             f"Fix: give accept_all and reject_any conditions, for example {_EXAMPLE}")
        named = tuple(dict.fromkeys(condition.check for condition in (*self.accept_all, *self.reject_any)))
        if checks is None:
            checks = named
        checks = tuple(checks)
        bad = [check for check in checks if not isinstance(check, str) or not check.strip()]
        if bad or len(set(checks)) != len(checks):
            raise ValueError(f"{where} collects the checks {list(checks)!r}. Fix: list each check once, by its id")
        absent = [check for check in named if check not in checks]
        if absent:
            raise ValueError(f"{where} has a condition on {', '.join(absent)}, which is not among the checks it "
                             f"collects ({', '.join(checks) or 'none'}). Fix: add the check to the predicate's "
                             "checks, or fix the feature's name")
        self.checks: tuple[str, ...] = checks

    @classmethod
    def from_predicate(cls, spec: Any, version: str) -> RuleWitness:
        """The witness for a task file's predicate entry: a model or a dict with ``name``, ``checks`` and ``decide``."""
        entry = _plain(spec)
        if not isinstance(entry, Mapping) or "name" not in entry or "decide" not in entry:
            raise ValueError(f"{spec!r} is not a predicate entry. Fix: give one with name, checks and decide")
        return cls(entry["name"], entry["decide"], version, checks=entry.get("checks"))

    # ---------------------------------------------------------------- the rule
    def judge(self, features: Mapping[str, float]) -> RuleJudgement:
        """The rule applied to a feature dict keyed ``<check>/<feature>``."""
        held = tuple(condition for condition in self.accept_all if condition.holds(features))
        unmet = tuple(condition for condition in self.accept_all if not condition.holds(features))
        fired = tuple(condition for condition in self.reject_any if condition.holds(features))
        named = tuple(dict.fromkeys(condition.feature for condition in (*self.accept_all, *self.reject_any)))
        missing = tuple(name for name in named if name not in features)
        values = tuple((name, float(features[name])) for name in named if name in features)
        if fired:
            decision = REJECT
        elif self.accept_all and not unmet:
            decision = ACCEPT
        else:
            decision = ABSTAIN
        return RuleJudgement(decision, held, unmet, fired, missing, values)

    def features(self, evidence: Sequence[Evidence]) -> dict[str, float]:
        """The feature dict ``verify`` decided on, rebuilt from the evidence it returned.

        ``verify`` returns 1 evidence item per check, in the order of ``checks``. The source id of an
        item does not say which view ran, so the keys come from the checks.
        """
        if len(evidence) != len(self.checks):
            raise ValueError(f"got {len(evidence)} evidence items for {len(self.checks)} checks. Fix: pass the "
                             "evidence list that verify returned, unchanged")
        out: dict[str, float] = {}
        for check, item in zip(self.checks, evidence):
            if item.source_id != check.partition("@")[0]:
                raise ValueError(f"the evidence from {item.source_id} sits where {check} belongs. Fix: pass the "
                                 "evidence list that verify returned, unchanged")
            out.update({f"{check}/{name}": float(value) for name, value in item.features.items()})
        return out

    # ---------------------------------------------------------------- live
    def verify(self, view, claim, params: dict | None) -> tuple[Verdict, list[Evidence]]:
        """Collect every check, apply the rule, and return (verdict, evidence list)."""
        from ..evidence import base

        if claim.predicate.name != self.predicate:
            raise ValueError(f"this rule judges {self.predicate}, and the claim {claim.claim_id} is about "
                             f"{claim.predicate.name}. Fix: verify the claim with its own predicate's witness")
        features: dict[str, float] = {}
        evidence: list[Evidence] = []
        total = Cost.zero()
        for key in self.checks:
            source_id, _, viewname = key.partition("@")
            call = dict(params or {})
            if viewname:
                call["view"] = viewname
            item = base.get(source_id).collect(view, claim, call)
            evidence.append(item)
            total = total + item.cost
            for name, value in item.features.items():
                features[f"{key}/{name}"] = float(value)
        judgement = self.judge(features)
        verdict = Verdict(verdict_id=new_id("verdict"), claim_id=claim.claim_id, decision=judgement.decision,
                          score=None, stage_reached=1, witness_version=self.version,
                          evidence_ids=[item.evidence_id for item in evidence], cost_total=total)
        return verdict, evidence

    # ---------------------------------------------------------------- records
    def to_dict(self) -> dict[str, Any]:
        """The rule as plain data, for a ledger line or a report. :meth:`from_dict` reads it back."""
        return {"kind": self.kind, "predicate": self.predicate, "version": self.version, "checks": list(self.checks),
                "accept_all": [condition.as_dict() for condition in self.accept_all],
                "reject_any": [condition.as_dict() for condition in self.reject_any]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RuleWitness:
        """The witness :meth:`to_dict` wrote."""
        decide = {key: data[key] for key in ("kind", "accept_all", "reject_any") if key in data}
        return cls(data["predicate"], decide, data["version"], checks=data.get("checks"))

    def __repr__(self) -> str:
        return (f"RuleWitness({self.predicate!r}, {self.version!r}, checks={list(self.checks)!r}, "
                f"{len(self.accept_all)} accept and {len(self.reject_any)} reject conditions)")
