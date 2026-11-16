"""Tasks as data: a task file lists the steps, the predicate each step claims, and the checks that serve it.

A task file is ``tasks/<task>.yaml``. It names what a cycle works on, the steps each item goes
through, and for each predicate a definition in plain words (what done means, so a person can
apply it), the checks that serve it and how they decide::

    task: cable_loop                      # the task's id, which is also the file's name
    title: one line
    items: {kind: cable, count: 5}        # a cycle is every step for every item, then a reset
    steps:                                # run in this order for each item
      - {name: pick, claim: held}
      - {name: insert, claim: seated, claim_by_variant: {latch: latched}}
    predicates:
      held:
        done_means: The gripper holds the named cable in its jaws, and no other cable.
        checks: [passive.jaw_hold, vision.cable_tag]       # evidence source ids, run in this order
        decide:                                            # a rule: thresholds on named features
          kind: rule
          accept_all: [{feature: passive.jaw_hold/in_band, above: 0.5}]
          reject_any: [{feature: passive.jaw_hold/closed_empty, above: 0.5}]
      seated:
        done_means: ...
        checks: [...]
        decide: {kind: witness, witness: "{variant}"}       # a calibrated cascade from the witness store
    fork:
      max_attempts: 2
      ask: {abstain: confirm, reject_last: choose}

A variant is what a cell is set up for (for the plug station, the connector type). A step may
claim another predicate on some variants, and a witness name may hold the variant. A new check
is 1 file under ``avail/evidence``. A new task is 1 file here, plus any new checks.

Loading refuses what would only fail later: a key the format does not have, a key written twice,
a step that claims a predicate the file does not define, a rule with no accept condition (every
condition of an empty list holds), a condition on a feature of a check the predicate does not run,
and a file whose task id is not its file name. Every problem in a file is reported at once.
"""

from __future__ import annotations

import difflib
import math
import re
import string
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from ..paths import REPO_ROOT
from ..schemas.kinds import QUESTION_TYPES

__all__ = [
    "TASKS_DIR", "TASK_SUFFIX", "TaskFileError", "RuleCondition", "DecideSpec", "PredicateSpec", "StepSpec",
    "ItemsSpec", "TaskSpec", "load_task", "task_names", "task_path", "predicate_definitions",
]

TASKS_DIR = REPO_ROOT / "tasks"
TASK_SUFFIX = ".yaml"

NAME_PATTERN = r"[a-z][a-z0-9_]*"                                        # a task, step, predicate or item kind
VARIANT_PATTERN = r"[A-Za-z0-9][A-Za-z0-9_.-]*"
CHECK_PATTERN = r"[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+(?:@[a-z0-9_]+)?"       # an evidence source id, maybe with @view
FEATURE_PATTERN = rf"{CHECK_PATTERN}/[A-Za-z0-9_][A-Za-z0-9_.-]*"         # <check>/<feature>, as a rule reads it
FORK_KEYS = ("max_attempts", "ask", "ladder", "ladder_after")
LADDER_AFTER = ("abstain", "reject")                   # what a predicate's ladder may follow: an abstain, a last reject
WITNESS_FIELDS = ("variant",)                                           # the fields a witness name may hold
MIN_DEFINITION_WORDS = 5

_NAME = re.compile(NAME_PATTERN)
_VARIANT = re.compile(VARIANT_PATTERN)
_CHECK = re.compile(CHECK_PATTERN)


class TaskFileError(ValueError):
    """A task file that cannot be used. The message names the file and every problem in it."""


def _did_you_mean(word: str, options) -> str:
    close = difflib.get_close_matches(str(word), [str(o) for o in options], n=1)
    return f" Did you mean {close[0]}?" if close else ""


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ----------------------------------------------------------------------------- the decision
class RuleCondition(_Spec):
    """A threshold on 1 named feature: ``<check>/<feature>`` above a number, below one, or between 2."""

    feature: str = Field(pattern=f"^{FEATURE_PATTERN}$",
                         description="the check and the feature it reports, for example passive.jaw_hold/in_band")
    above: float | None = Field(default=None, strict=True, allow_inf_nan=False,
                                description="the condition holds only when the feature is greater than this")
    below: float | None = Field(default=None, strict=True, allow_inf_nan=False,
                                description="the condition holds only when the feature is less than this")

    @model_validator(mode="after")
    def _has_a_bound(self) -> RuleCondition:
        if self.above is None and self.below is None:
            raise ValueError(f"the condition on {self.feature} gives no bound. Give above, below or both")
        if self.above is not None and self.below is not None and not self.above < self.below:
            raise ValueError(f"the condition on {self.feature} asks for more than {self.above} and less than "
                             f"{self.below} at once, so it never holds. Swap the 2 numbers or drop 1")
        return self

    @property
    def check(self) -> str:
        """The check whose feature this condition reads: ``passive.jaw_hold`` for ``passive.jaw_hold/in_band``."""
        return self.feature.partition("/")[0]

    def holds(self, value: Any) -> bool:
        """True when ``value`` is a measured number strictly above ``above`` and strictly below ``below``.

        A feature that was not measured (None, NaN, absent) never holds a condition, so on its own
        it can neither accept nor reject.
        """
        if value is None or isinstance(value, (str, bytes)):
            return False
        try:
            x = float(value)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(x):
            return False
        return (self.above is None or x > self.above) and (self.below is None or x < self.below)


def _template_fields(template: str) -> list[str]:
    """The fields a witness name holds: ``["variant"]`` for ``"{variant}"``. ValueError for anything else."""
    try:
        parts = list(string.Formatter().parse(template))
    except ValueError as exc:
        raise ValueError(f"the witness name {template!r} cannot be read ({exc}). Write a field as {{variant}}") from None
    fields = []
    for _literal, field_name, format_spec, conversion in parts:
        if field_name is None:
            continue
        if field_name not in WITNESS_FIELDS or format_spec or conversion:
            raise ValueError(f"the witness name {template!r} holds {{{field_name}}}. A witness name may hold only "
                             + ", ".join("{" + f + "}" for f in WITNESS_FIELDS))
        fields.append(field_name)
    return fields


class DecideSpec(_Spec):
    """How a predicate's checks decide: a rule on named features, or a calibrated cascade from the witness store.

    A rule accepts when every condition in ``accept_all`` holds and rejects when any condition in
    ``reject_any`` holds. ``witness`` names a stored cascade, and may hold ``{variant}``.
    """

    kind: Literal["rule", "witness"] = Field(description="rule or witness")
    accept_all: list[RuleCondition] = Field(default_factory=list,
                                            description="a rule: the conditions that must all hold to accept")
    reject_any: list[RuleCondition] = Field(default_factory=list,
                                            description="a rule: the conditions any 1 of which rejects")
    witness: str | None = Field(default=None, description="a witness: the stored cascade's name, for example {variant}")

    @field_validator("witness", mode="before")
    @classmethod
    def _quoted(cls, value: Any) -> Any:
        if isinstance(value, dict) and len(value) == 1 and next(iter(value.values())) is None:
            name = next(iter(value))
            raise ValueError(f"the witness name reads as a mapping. Put it in quotes: witness: \"{{{name}}}\"")
        return value

    @model_validator(mode="after")
    def _fits_its_kind(self) -> DecideSpec:
        if self.kind == "rule":
            if self.witness is not None:
                raise ValueError("a rule names no witness. Drop witness, or use kind: witness")
            if not self.accept_all:
                raise ValueError("a rule needs at least 1 condition in accept_all. Every condition of an empty list "
                                 "holds, so a rule with none could accept a claim on no evidence")
            return self
        if self.accept_all or self.reject_any:
            raise ValueError("a witness takes no conditions. Drop accept_all and reject_any, or use kind: rule")
        if self.witness is None or not self.witness.strip():
            raise ValueError("a witness decision names its witness, for example witness: \"{variant}\"")
        _template_fields(self.witness)
        return self

    def witness_for(self, variant: str) -> str:
        """The witness name on a cell of ``variant``: ``"{variant}"`` gives the variant itself."""
        if self.kind != "witness" or self.witness is None:
            raise ValueError("this predicate is decided by a rule, which has no witness")
        return self.witness.format(variant=variant)

    def conditions(self) -> list[RuleCondition]:
        """Every condition of a rule, the accept conditions first."""
        return [*self.accept_all, *self.reject_any]


# ----------------------------------------------------------------------------- predicates and steps
class PredicateSpec(_Spec):
    """A predicate of a task: what done means in plain words, the checks that serve it, and how they decide."""

    name: str = Field(pattern=f"^{NAME_PATTERN}$", description="the predicate's name, its key in predicates")
    done_means: str = Field(description="a definition a person can apply, in plain words")
    checks: list[str] = Field(default_factory=list, description="evidence source ids, run in this order")
    decide: DecideSpec = Field(description="how the checks decide")

    @field_validator("done_means")
    @classmethod
    def _plain_words(cls, value: str) -> str:
        text = value.strip()
        if len(text.split()) < MIN_DEFINITION_WORDS or not text.endswith("."):
            raise ValueError(f"done_means is a definition a person can apply: at least {MIN_DEFINITION_WORDS} words, "
                             "ending with a period")
        return text

    @field_validator("checks")
    @classmethod
    def _check_ids(cls, value: list[str]) -> list[str]:
        bad = [c for c in value if not _CHECK.fullmatch(c)]
        if bad:
            raise ValueError(f"{', '.join(repr(c) for c in bad)} is not a check id. A check id is an evidence source "
                             "id such as kind.name, with an optional @view")
        twice = sorted({c for c in value if value.count(c) > 1})
        if twice:
            raise ValueError(f"{', '.join(twice)} is listed more than once")
        return value

    @model_validator(mode="after")
    def _rule_reads_its_checks(self) -> PredicateSpec:
        if self.decide.kind == "rule":
            missing = sorted({c.check for c in self.decide.conditions()} - set(self.checks))
            if missing:
                raise ValueError(f"the rule reads features of {', '.join(missing)}, which checks does not list "
                                 f"({', '.join(self.checks) or 'no check'}). A rule reads only its own checks")
        return self


class StepSpec(_Spec):
    """A step of a task and the predicate it claims, on every variant or on 1 of them."""

    name: str = Field(pattern=f"^{NAME_PATTERN}$", description="the step's name, for example pick")
    claim: str = Field(pattern=f"^{NAME_PATTERN}$", description="the predicate the step claims")
    claim_by_variant: dict[str, str] = Field(default_factory=dict,
                                             description="variant to the predicate the step claims on it instead")
    judge: bool = Field(default=False, strict=True,
                        description="the vlm arm's judge decides this step. On the other steps that arm trusts the "
                                    "skill's own flag")

    @field_validator("claim_by_variant")
    @classmethod
    def _names(cls, value: dict[str, str]) -> dict[str, str]:
        bad = [k for k in value if not _VARIANT.fullmatch(k)] + [v for v in value.values() if not _NAME.fullmatch(v)]
        if bad:
            raise ValueError(f"{', '.join(repr(b) for b in bad)} is not a name. A variant and a predicate are names "
                             "such as abc or abc_2")
        return value

    def claim_for(self, variant: str) -> str:
        """The predicate this step claims on a cell of ``variant``."""
        return self.claim_by_variant.get(variant, self.claim)


class ItemsSpec(_Spec):
    """What a cycle works on: ``count`` items of 1 ``kind``. A cycle is every step for every item, then a reset."""

    kind: str = Field(pattern=f"^{NAME_PATTERN}$", description="the kind of item, for example cable")
    count: int = Field(ge=1, strict=True, description="how many items a cycle works on")


# ----------------------------------------------------------------------------- the task
class TaskSpec(_Spec):
    """A task: its items, its steps in order, its predicates, and how the fork asks for help."""

    task: str = Field(pattern=f"^{NAME_PATTERN}$", description="the task's id, which is also its file's name")
    title: str = Field(description="1 line that says what the task is")
    items: ItemsSpec = Field(description="what a cycle works on")
    steps: list[StepSpec] = Field(min_length=1, description="the steps each item goes through, in order")
    predicates: dict[str, PredicateSpec] = Field(min_length=1, description="every predicate a step may claim")
    fork: dict[str, Any] = Field(default_factory=dict,
                                 description="max_attempts, and ask: what to ask a person, by what opened the fork")
    testbed: str | None = Field(default=None, pattern=r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$",
                                description="the module that runs the task's cycles: its cells, witnesses, person and "
                                            "evaluator. The shift runner imports it by this name, so the core names no testbed")

    @model_validator(mode="before")
    @classmethod
    def _name_the_predicates(cls, data: Any) -> Any:
        """A predicate takes its name from its key under ``predicates``."""
        if isinstance(data, dict) and isinstance(data.get("predicates"), dict):
            named = {}
            for key, spec in data["predicates"].items():
                named[key] = {**spec, "name": key} if isinstance(spec, dict) and "name" not in spec else spec
            data = {**data, "predicates": named}
        return data

    @field_validator("title")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("the title is empty. Say in 1 line what the task is")
        return value.strip()

    @field_validator("fork")
    @classmethod
    def _fork(cls, value: dict[str, Any]) -> dict[str, Any]:
        problems = [f"fork does not take {key}.{_did_you_mean(key, FORK_KEYS)} It takes {', '.join(FORK_KEYS)}"
                    for key in value if key not in FORK_KEYS]
        if "max_attempts" in value:
            n = value["max_attempts"]
            if isinstance(n, bool) or not isinstance(n, int) or n < 1:
                problems.append(f"fork.max_attempts is {n!r}. Give a whole number, 1 or more")
        ask = value.get("ask", {})
        if not isinstance(ask, dict):
            problems.append("fork.ask maps what opened a fork to the question for a person, "
                            "for example {abstain: confirm}")
            ask = {}
        for trigger, question in ask.items():
            if not isinstance(trigger, str) or not _NAME.fullmatch(trigger):
                problems.append(f"fork.ask has the key {trigger!r}. Name what opens the fork, for example abstain")
            if question not in QUESTION_TYPES:
                problems.append(f"fork.ask.{trigger} is {question!r}, which is not a question.{_did_you_mean(question, QUESTION_TYPES)} "
                                f"A question is one of {', '.join(QUESTION_TYPES)}")
        # 2026-11-16 (docs/07, step 2): the escalation ladder, per predicate the cell's recovery actions to try in order
        # before a call to the operator, each followed by the predicate's own witness on a fresh claim
        ladder = value.get("ladder", {})
        if not isinstance(ladder, dict):
            problems.append("fork.ladder maps a predicate to the rungs to try before a call, for example {at_port: [regrip]}")
            ladder = {}
        for predicate, rungs in ladder.items():
            if not isinstance(predicate, str) or not _NAME.fullmatch(predicate):
                problems.append(f"fork.ladder has the key {predicate!r}. Name a predicate of the task")
            if not isinstance(rungs, list) or not all(isinstance(r, str) and _NAME.fullmatch(r) for r in rungs):
                problems.append(f"fork.ladder.{predicate} is {rungs!r}. Give a list of rung names, for example [regrip]")
        # what each predicate's ladder follows (default both): a 2nd look at a state no rung changed is an extra chance
        # for a false accept, so a look from elsewhere runs after an abstain only
        after = value.get("ladder_after", {})
        if not isinstance(after, dict):
            problems.append("fork.ladder_after maps a predicate to what its ladder follows, for example {held: [abstain]}")
            after = {}
        for predicate, when in after.items():
            if predicate not in ladder:
                problems.append(f"fork.ladder_after names {predicate!r}, which fork.ladder does not")
            if not isinstance(when, list) or not when or any(w not in LADDER_AFTER for w in when):
                problems.append(f"fork.ladder_after.{predicate} is {when!r}. Give a list from {', '.join(LADDER_AFTER)}")
        if problems:
            raise ValueError(". ".join(p.rstrip(".") for p in problems))
        out = dict(value)
        if "ask" in out:
            out["ask"] = dict(out["ask"])
        if "ladder" in out:
            out["ladder"] = {k: list(v) for k, v in out["ladder"].items()}
        if "ladder_after" in out:
            out["ladder_after"] = {k: list(v) for k, v in out["ladder_after"].items()}
        return out

    @model_validator(mode="after")
    def _steps_claim_known_predicates(self) -> TaskSpec:
        problems = []
        for key, spec in self.predicates.items():
            if spec.name != key:
                problems.append(f"the predicate under {key} is named {spec.name}. Its name is its key")
        names = [step.name for step in self.steps]
        twice = sorted({n for n in names if names.count(n) > 1})
        if twice:
            problems.append(f"the step {', '.join(twice)} is listed more than once. Step names are unique")
        for step in self.steps:
            for variant, claim in [(None, step.claim), *step.claim_by_variant.items()]:
                if claim not in self.predicates:
                    on = "" if variant is None else f" on {variant}"
                    problems.append(f"the step {step.name} claims {claim}{on}, which predicates does not define."
                                    f"{_did_you_mean(claim, self.predicates)} Defined: {', '.join(self.predicates)}")
        for predicate in dict(self.fork.get("ladder", {})):
            if predicate not in self.predicates:
                problems.append(f"fork.ladder names {predicate}, which predicates does not define."
                                f"{_did_you_mean(predicate, self.predicates)} Defined: {', '.join(self.predicates)}")
        if problems:
            raise ValueError(". ".join(p.rstrip(".") for p in problems))
        return self

    # ---------------------------------------------------------------- reading a task
    def required(self, variant: str) -> list[tuple[str, str]]:
        """(step name, predicate) for every step, in order, on a cell of ``variant``. What done needs per item."""
        return [(step.name, step.claim_for(variant)) for step in self.steps]

    def cycle(self, variant: str) -> list[tuple[int, str, str]]:
        """(item index, step name, predicate) for a whole cycle: every step of item 0, then of item 1, and on."""
        return [(item, step, predicate) for item in range(self.items.count) for step, predicate in self.required(variant)]

    def step(self, name: str) -> StepSpec:
        """The step called ``name``. KeyError, with the task's steps, when there is none."""
        for step in self.steps:
            if step.name == name:
                return step
        raise KeyError(f"task {self.task} has no step {name!r}.{_did_you_mean(name, [s.name for s in self.steps])} "
                       f"Its steps: {', '.join(s.name for s in self.steps)}")

    def predicate_for(self, step: str, variant: str) -> PredicateSpec:
        """The predicate ``step`` claims on a cell of ``variant``, with its definition, its checks and its decision."""
        return self.predicates[self.step(step).claim_for(variant)]

    def variants(self) -> list[str]:
        """The variants on which some step claims another predicate."""
        return sorted({variant for step in self.steps for variant in step.claim_by_variant})


# ----------------------------------------------------------------------------- reading files
class _TaskLoader(yaml.SafeLoader):
    """YAML's safe loader with 1 change: a key written twice in 1 mapping is an error. The plain loader keeps the last."""


def _mapping_without_repeats(loader: _TaskLoader, node: yaml.MappingNode) -> dict:
    seen = set()
    for key_node, _value_node in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            continue
        key = loader.construct_object(key_node, deep=True)
        try:
            repeated = key in seen
        except TypeError:                         # an unhashable key: construct_mapping reports it
            continue
        if repeated:
            raise yaml.constructor.ConstructorError("while reading a mapping", node.start_mark,
                                                    f"found the key {key!r} twice", key_node.start_mark)
        seen.add(key)
    return loader.construct_mapping(node, deep=True)


_TaskLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping_without_repeats)


def _read(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise TaskFileError(f"{path}: cannot be read ({type(exc).__name__}: {exc})") from None
    try:
        data = yaml.load(text, Loader=_TaskLoader)
    except yaml.YAMLError as exc:
        raise TaskFileError(f"{path}: is not valid YAML. {exc}") from None
    if not isinstance(data, dict):
        held = "nothing" if data is None else f"a {type(data).__name__}"
        raise TaskFileError(f"{path}: holds {held}. A task file is a mapping with task, title, items, steps, "
                            "predicates and fork")
    return data


def _message(error: dict) -> str:
    kind, text = error.get("type", ""), str(error.get("msg", ""))
    if kind == "extra_forbidden":
        return "is not a key the task format has. Check the spelling, or drop it"
    if kind == "missing":
        return "is missing"
    return text.removeprefix("Value error, ")


def _explain(path: Path, exc: ValidationError) -> str:
    n = exc.error_count()
    lines = [f"{path}: {n} problem{'' if n == 1 else 's'}"]
    for error in exc.errors(include_url=False):
        where = ".".join(str(part) for part in error.get("loc", ())) or "the file"
        lines.append(f"  {where}: {_message(error)}")
    return "\n".join(lines)


def task_names(tasks_dir: str | Path | None = None) -> list[str]:
    """The task ids that have a file in the tasks folder, sorted."""
    folder = Path(tasks_dir) if tasks_dir is not None else TASKS_DIR
    if not folder.is_dir():
        return []
    return sorted(p.stem for p in folder.glob(f"*{TASK_SUFFIX}") if p.is_file())


def task_path(name_or_path: str | Path, tasks_dir: str | Path | None = None) -> Path:
    """The file of a task: ``tasks/<name>.yaml`` for a task id, or the path itself.

    A relative path that is not found from the working folder is looked for from the repo root.
    """
    folder = Path(tasks_dir) if tasks_dir is not None else TASKS_DIR
    if isinstance(name_or_path, str) and _NAME.fullmatch(name_or_path):
        path = folder / f"{name_or_path}{TASK_SUFFIX}"
        if not path.is_file():
            known = task_names(folder)
            raise FileNotFoundError(f"no task {name_or_path!r} in {folder}.{_did_you_mean(name_or_path, known)} "
                                    f"Known: {', '.join(known) or 'none'}")
        return path
    path = Path(name_or_path)
    if not path.is_absolute() and not path.is_file() and (REPO_ROOT / path).is_file():
        path = REPO_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"no task file at {path}")
    return path


def load_task(name_or_path: str | Path, tasks_dir: str | Path | None = None) -> TaskSpec:
    """A task by its id (``tasks/<name>.yaml``) or by the path of its file. TaskFileError lists every problem."""
    path = task_path(name_or_path, tasks_dir)
    data = _read(path)
    try:
        spec = TaskSpec.model_validate(data)
    except ValidationError as exc:
        raise TaskFileError(_explain(path, exc)) from None
    if spec.task != path.stem:
        raise TaskFileError(f"{path}: the file holds task {spec.task}. A task file is named after its task: "
                            f"name the file {spec.task}{TASK_SUFFIX}, or set task: {path.stem}")
    return spec


def predicate_definitions(tasks: str | list[str] | tuple[str, ...] | None = None,
                          tasks_dir: str | Path | None = None) -> dict[str, str]:
    """Every ``done_means`` in the task files, by predicate.

    ``tasks`` names the task files to read, in order of precedence. A predicate that more than 1 of
    them defines keeps the words of the first. The default reads every task file, the one that
    defines the most predicates first and ties in name order, which is the order
    ``avail.schemas.kinds`` fills ``PREDICATE_DEFINITIONS`` in. A task's own words are
    ``load_task(task).predicates[name].done_means``.
    """
    if tasks is None:
        specs = sorted((load_task(name, tasks_dir) for name in task_names(tasks_dir)),
                       key=lambda spec: (-len(spec.predicates), spec.task))
    else:
        specs = [load_task(name, tasks_dir) for name in ([tasks] if isinstance(tasks, str) else tasks)]
    out: dict[str, str] = {}
    for spec in specs:
        for predicate, definition in spec.predicates.items():
            out.setdefault(predicate, definition.done_means)
    return out
