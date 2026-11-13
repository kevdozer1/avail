"""Validation errors a person can act on.

:func:`explain` turns a pydantic ``ValidationError`` into 1 line per problem. Each
line names the field path and says how to fix it::

    evidence_ids: an accept needs evidence, and this verdict lists no evidence ids. Fix: list the ids of ...
    stages.1.alpha_k: 1.5 is above 1. Fix: give a value of 1 or less (alpha_k: this stage's share of ...)

:func:`validate_record` validates data against a record type and raises
:class:`SchemaError` carrying those lines.
"""

from __future__ import annotations

import difflib
import typing
from dataclasses import dataclass
from typing import Any, TypeVar, overload

from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo

from .base import SCALAR_CODES, Record, short
from .records import PART_MODELS, PAYLOAD_MODELS, RECORD_MODELS, CycleEnd, LedgerEvent

__all__ = ["SchemaError", "explain", "record_model", "validate_record"]

_M = TypeVar("_M", bound=BaseModel)
_FIX_MARK = ". Fix: "
_KEY_STEP = "[key]"  # how pydantic marks a problem with a dict key

_MODELS: dict[str, type[BaseModel]] = {**RECORD_MODELS, **PART_MODELS}

# A free dict that holds a typed record in some cases: where to look for its fields.
_TYPED_DICTS: dict[tuple[type[BaseModel], str], type[BaseModel]] = {(LedgerEvent, "payload"): CycleEnd}

_NAMES: dict[str, type[Record]] = {
    **{name.lower(): model for name, model in RECORD_MODELS.items()},
    **{name.lower(): model for name, model in PART_MODELS.items()},
    **PAYLOAD_MODELS,
    "ledger_event": LedgerEvent,
    "event": LedgerEvent,
}

# pydantic's own error types, as (what is wrong, how to fix it). {got} is the refused value.
_TYPE_ERRORS = {
    "string_type": ("{got} is not text", "give a string"),
    "dict_type": ("{got} is not a dict", "give a dict of names to values, for example {{}} when there is nothing to say"),
    "list_type": ("{got} is not a list", "give a list, for example [] when there is nothing to list"),
    "bool_type": ("{got} is not true or false", "give a real boolean"),
    "int_type": ("{got} is not a whole number", "give an integer"),
    "float_type": ("{got} is not a number", "give a number"),
    "finite_number": ("{got} is not a finite number", "give a finite number"),
    "frozen_instance": (
        "a record does not change after it is built",
        "build a changed copy with model_copy(update={{...}}), which validates the result",
    ),
}
_BOUNDS = {
    "greater_than_equal": ("ge", "is below {bound}", "give a value of {bound} or more"),
    "greater_than": ("gt", "is not above {bound}", "give a value above {bound}"),
    "less_than_equal": ("le", "is above {bound}", "give a value of {bound} or less"),
    "less_than": ("lt", "is not below {bound}", "give a value below {bound}"),
}


class SchemaError(ValueError):
    """Data that does not fit a record type.

    ``record`` is the record type's name. ``problems`` holds 1 line per problem, each
    naming the field path and saying how to fix it.
    """

    def __init__(self, record: str, problems: list[str]):
        self.record = record
        self.problems = list(problems)
        count = len(self.problems)
        head = f"{record} is not valid ({count} problem{'' if count == 1 else 's'}):"
        super().__init__("\n".join([head, *(f"  {line}" for line in self.problems)]))

    def __reduce__(self) -> tuple[Any, ...]:
        # so the error survives pickling, for example on its way out of a worker process
        return type(self), (self.record, self.problems)


# ------------------------------------------------------------------ where an error sits
@dataclass
class _Spot:
    """The last named step of an error's path: the model it was looked up in and its field, when known."""

    owner: type[BaseModel] | None = None
    name: str | None = None
    field: FieldInfo | None = None


def _model_in(annotation: Any) -> type[BaseModel] | None:
    """The record type inside an annotation such as ``list[Stage]`` or ``Shadow | None``."""
    try:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return annotation
    except TypeError:
        return None
    for inner in typing.get_args(annotation):
        found = _model_in(inner)
        if found is not None:
            return found
    return None


def _spot(model: type[BaseModel] | None, loc: tuple[Any, ...]) -> _Spot:
    spot = _Spot()
    current = model
    for step in loc:
        if isinstance(step, int):
            continue  # an index into a list of the same type
        if current is None or step == _KEY_STEP:
            break  # inside a free dict such as features: the dict's own field is the spot
        field = current.model_fields.get(step)
        spot = _Spot(current, step, field)
        current = _TYPED_DICTS.get((current, step)) or (_model_in(field.annotation) if field is not None else None)
    return spot


def _path(loc: tuple[Any, ...], title: str) -> str:
    """``stages.1.alpha_k`` for a field inside a list, and the record's name for a problem with the whole input."""
    steps = [str(step) if step != "" else '""' for step in loc if step != _KEY_STEP]
    return ".".join(steps) or title


# ------------------------------------------------------------------ what and fix
def _unknown_field(spot: _Spot) -> tuple[str, str]:
    if spot.owner is None or spot.name is None:
        return "this field is not known here", "remove it"
    fields = list(spot.owner.model_fields)
    close = difflib.get_close_matches(spot.name, fields, n=1)
    hint = f"Did you mean {close[0]}? " if close else ""
    return (
        f"{spot.owner.__name__} has no field named {spot.name}",
        f"remove it or rename it. {hint}{spot.owner.__name__} has: {', '.join(fields)}",
    )


def _count(number: Any, unit: str) -> str:
    return f"{number} {unit}" if number == 1 else f"{number} {unit}s"


def _what_and_fix(item: dict[str, Any], spot: _Spot) -> tuple[str, str]:
    """(what is wrong, how to fix it) for 1 of pydantic's own errors."""
    kind = item.get("type", "")
    ctx = item.get("ctx") or {}
    got = short(item.get("input"))
    if kind == "missing":
        return "this required field is missing", f"add {spot.name or 'it'}"
    if kind == "extra_forbidden":
        return _unknown_field(spot)
    if kind == "literal_error":
        return f"{got} is not an allowed value", f"use {ctx.get('expected', 'one of the allowed values')}"
    if kind == "model_type":
        name = ctx.get("class_name", "the record")
        return f"{got} is not a dict or an instance of {name}", f"give a dict with the fields of {name}"
    if kind in _BOUNDS:
        key, what, fix = _BOUNDS[kind]
        bound = f"{ctx.get(key):g}" if isinstance(ctx.get(key), (int, float)) else str(ctx.get(key))
        return f"{got} {what.format(bound=bound)}", fix.format(bound=bound)
    if kind in ("too_short", "string_too_short"):
        unit = "character" if kind == "string_too_short" else "item"
        have, need = ctx.get("actual_length"), ctx.get("min_length", 1)
        have = len(item.get("input") or "") if have is None else have
        return f"this has {_count(have, unit)}", f"give at least {_count(need, unit)}"
    if kind == "json_invalid":
        return f"this is not valid JSON ({ctx.get('error', 'it could not be parsed')})", "give 1 JSON object"
    if kind in _TYPE_ERRORS:
        what, fix = _TYPE_ERRORS[kind]
        return what.format(got=got), fix.format()
    message = str(item.get("msg", "this value is not valid"))
    for prefix in ("Value error, ", "Assertion failed, "):
        message = message.removeprefix(prefix)
    return message.rstrip("."), "change the value to fit the record type. 'avail schemas' writes its JSON Schema"


def explain(error: ValidationError) -> list[str]:
    """1 line per problem in ``error``. Each line names the field path and says how to fix it.

    Works for any pydantic ``ValidationError``. For a record type of this package the
    line also quotes the field's description, so the reader sees what belongs there.
    """
    model = _MODELS.get(error.title)
    lines = []
    for item in error.errors(include_url=False):
        loc = tuple(item.get("loc", ()))
        spot = _spot(model, loc)
        note = f" ({spot.name}: {spot.field.description})" if spot.field is not None and spot.field.description else ""
        message = str(item.get("msg", ""))
        if _FIX_MARK in message:  # one of this package's own refusals: it carries its fix
            if item.get("type") not in SCALAR_CODES:
                note = ""  # a rule's message is complete. A scalar's fix is general, so say what the field is for
        else:
            what, fix = _what_and_fix(item, spot)
            message = f"{what}{_FIX_MARK}{fix}"
        lines.append(f"{_path(loc, error.title)}: {message}{note}")
    return lines


# ------------------------------------------------------------------ validate
def record_model(kind_or_model: str | type[BaseModel]) -> type[BaseModel]:
    """The record type for a name, a ledger event kind or a model class.

    Names are matched without regard to case: ``"Verdict"``, ``"verdict"`` and the ledger
    kind ``"verdict"`` all give :class:`Verdict`, and ``"cycle_end"`` gives :class:`CycleEnd`.
    """
    if isinstance(kind_or_model, type) and issubclass(kind_or_model, BaseModel):
        return kind_or_model
    if isinstance(kind_or_model, str):
        model = _NAMES.get(kind_or_model.strip().lower())
        if model is not None:
            return model
    raise ValueError(
        f"no record type is named {kind_or_model!r}. Fix: pass a record class, a record name "
        f"({', '.join(RECORD_MODELS)}) or a ledger kind that has a record type ({', '.join(PAYLOAD_MODELS)}). "
        "The other ledger kinds have a free-form payload."
    )


@overload
def validate_record(kind_or_model: type[_M], data: Any) -> _M: ...
@overload
def validate_record(kind_or_model: str, data: Any) -> BaseModel: ...


def validate_record(kind_or_model: str | type[BaseModel], data: Any) -> Any:
    """``data`` as a validated record, or :class:`SchemaError` with 1 fix line per problem.

    ``kind_or_model`` is a record class, a record name or a ledger event kind (see
    :func:`record_model`). ``data`` is a dict, a record, or JSON text.
    """
    model = record_model(kind_or_model)
    try:
        if isinstance(data, (str, bytes, bytearray)):
            return model.model_validate_json(data)
        return model.model_validate(data)
    except ValidationError as exc:
        raise SchemaError(model.__name__, explain(exc)) from exc
