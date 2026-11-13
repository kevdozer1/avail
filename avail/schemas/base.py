"""What every record type shares: the base class, the scalar field types and ``new_id``.

A record is a logged fact. It forbids unknown fields, refuses NaN and infinity,
and does not change after it is built. The scalar types below refuse the quiet
conversions pydantic would otherwise make, such as the text ``"yes"`` becoming
``True`` or ``"0.9"`` becoming a score, because a model's text must never turn
into a label or a number by accident.

Every refusal raised here carries its own fix: the message reads
``<what is wrong>. Fix: <what to do>``.
"""

from __future__ import annotations

import copy as _copy
import datetime as _dt
import json
import math
import os
import re
import secrets
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, NoReturn, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError
from pydantic_core import InitErrorDetails, PydanticCustomError

__all__ = [
    "SCALAR_CODES", "Count", "Flag", "Id", "IsoDate", "NonNegative", "Number", "Record", "Share", "Timestamp",
    "Version", "new_id", "path_text", "plain", "problem", "refuse", "short",
]

_ID_HEX_CHARS = 12
_ID_PREFIX = re.compile(r"[A-Za-z][A-Za-z0-9_.-]*")

SCALAR_CODES = frozenset({
    "not_a_number", "not_finite", "not_a_whole_number", "not_a_boolean", "not_text", "blank", "version", "iso_date",
    "not_a_path",
})
"""The error codes of the scalar types below. Their fix is general, so ``explain`` adds what the field is for."""


# ------------------------------------------------------------------ messages
def short(value: Any, limit: int = 60) -> str:
    """``value`` as compact text for a message, cut to ``limit`` characters."""
    if isinstance(value, float) and not math.isfinite(value):
        text = repr(value)  # nan, inf, -inf
    else:
        try:
            text = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError, OverflowError, RecursionError):
            try:
                text = repr(value)
            except Exception:  # a failing __repr__ must not hide the real problem
                text = f"<{type(value).__name__}>"
    return text if len(text) <= limit else text[: limit - 3] + "..."


def problem(code: str, what: str, fix: str) -> PydanticCustomError:
    """A validation error whose message says what is wrong and how to fix it."""
    return PydanticCustomError(code, f"{what}. Fix: {fix}")  # type: ignore[arg-type]


def refuse(record: str, found: Sequence[tuple[str, PydanticCustomError, Any]]) -> NoReturn:
    """Raise one validation error that puts each problem on the field it belongs to.

    ``found`` holds ``(field name, problem, the refused value)``. Used by rules that
    read more than 1 field, so the error still names a field.
    """
    raise ValidationError.from_exception_data(
        record,
        [InitErrorDetails(type=error, loc=(field,), input=value) for field, error, value in found],
    )


# ------------------------------------------------------------------ scalars
def plain(value: Any) -> Any:
    """A numpy scalar or 0-d array as the plain Python value. Anything else comes back unchanged."""
    if getattr(value, "ndim", None) == 0 and hasattr(value, "dtype") and callable(getattr(value, "item", None)):
        try:
            return value.item()
        except Exception:  # an exotic array type: let the caller's type check refuse it
            return value
    return value


def _number(value: Any) -> float:
    value = plain(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise problem("not_a_number", f"{short(value)} is not a number",
                      "give a number such as 0.5. Text and true or false are refused")
    try:
        number = float(value)
    except OverflowError:
        number = math.inf
    if not math.isfinite(number):
        raise problem("not_finite", f"{short(value)} is not a finite number", "give a finite number")
    return number


def _whole(value: Any) -> int:
    value = plain(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise problem("not_a_whole_number", f"{short(value)} is not a whole number",
                      "give an integer such as 0, 1 or 2")
    return value


def _flag(value: Any) -> bool:
    value = plain(value)
    if not isinstance(value, bool):
        raise problem("not_a_boolean", f"{short(value)} is not true or false",
                      "give a real boolean. Text such as 'yes' and numbers such as 1 are refused on purpose")
    return value


def _id(value: Any) -> str:
    if not isinstance(value, str):
        raise problem("not_text", f"{short(value)} is not text", "give a non-empty string")
    if not value.strip():
        raise problem("blank", "this is empty", "give a non-empty string")
    return value


def _version(value: Any) -> str:
    value = plain(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if not isinstance(value, str) or not value.strip():
        raise problem("version", f"{short(value)} is not a version label",
                      "give the witness version as text, for example 'v3'. A whole number is turned into text")
    return value


def _iso_date(value: Any) -> str:
    if isinstance(value, (_dt.date, _dt.datetime)):
        return value.isoformat()
    if isinstance(value, str):
        for parse in (_dt.date.fromisoformat, _dt.datetime.fromisoformat):
            try:
                parse(value)
            except ValueError:
                continue
            return value
    raise problem("iso_date", f"{short(value)} is not an ISO date", "give a date such as 2026-11-11")


Id = Annotated[str, Field(min_length=1), BeforeValidator(_id)]
"""A non-empty string: an id or the name of a thing."""

Flag = Annotated[bool, BeforeValidator(_flag)]
"""A real boolean. Text and numbers are refused."""

Count = Annotated[int, Field(ge=0), BeforeValidator(_whole)]
"""A whole number, 0 or more."""

Number = Annotated[float, BeforeValidator(_number)]
"""A finite number."""

NonNegative = Annotated[float, Field(ge=0), BeforeValidator(_number)]
"""A finite number, 0 or more."""

Share = Annotated[float, Field(ge=0, le=1), BeforeValidator(_number)]
"""A finite number from 0 to 1."""

Timestamp = Annotated[float, BeforeValidator(_number)]
"""Unix time in seconds."""

Version = Annotated[str, Field(min_length=1), BeforeValidator(_version)]
"""A witness version label such as ``v3``."""

IsoDate = Annotated[str, BeforeValidator(_iso_date)]
"""An ISO date or datetime, kept as text so a dumped record is plain JSON."""


def path_text(value: Any) -> str:
    """A file path as text. Accepts ``str`` and path objects."""
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str) or not value.strip():
        raise problem("not_a_path", f"{short(value)} is not a file path", "give the path as text, for example str(path)")
    return value


# ------------------------------------------------------------------ ids
def new_id(prefix: str) -> str:
    """A fresh id: ``prefix``, an underscore and 12 hex characters, for example ``claim_3f9a1c0b7d2e``.

    The hex part comes from the operating system's random source, so making an id
    never touches a seeded random stream.
    """
    if not isinstance(prefix, str) or not _ID_PREFIX.fullmatch(prefix):
        raise ValueError(
            f"new_id prefix {prefix!r} is not usable. Fix: give a short word of letters, digits, '_', '.' or '-' "
            "that starts with a letter, for example 'claim'."
        )
    return f"{prefix}_{secrets.token_hex(_ID_HEX_CHARS // 2)}"


# ------------------------------------------------------------------ the base class
class Record(BaseModel):
    """Base of every record type.

    * Unknown fields are refused. (``CycleEnd``, the payload of a ledger event, is the 1 type that keeps them.)
    * NaN and infinity are refused.
    * A record does not change after it is built. ``model_copy(update=...)`` gives a
      changed copy and validates it, which pydantic's own ``model_copy`` does not.
    * ``model_construct`` is refused, so a record that exists has passed its checks.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        """A copy of this record. With ``update``, the changed copy is validated like a new record."""
        if not update:
            return super().model_copy(deep=deep)
        data = {**(self.model_extra or {}), **{name: getattr(self, name) for name in type(self).model_fields}}
        if deep:
            data = _copy.deepcopy(data)
        data.update(update)
        return type(self).model_validate(data)

    def copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False, **unsupported: Any) -> Self:
        """The old pydantic name for :meth:`model_copy`. It validates too."""
        if unsupported:
            raise TypeError(
                f"{type(self).__name__}.copy does not take {', '.join(sorted(unsupported))}. "
                "Fix: use model_copy(update=...) for a changed copy, or model_dump(include=..., exclude=...) for a part."
            )
        return self.model_copy(update=update, deep=deep)

    @classmethod
    def model_construct(cls, _fields_set: set[str] | None = None, **values: Any) -> NoReturn:
        """Refused. pydantic's ``model_construct`` skips validation."""
        raise TypeError(
            f"{cls.__name__}.model_construct skips validation, and a record is never built unvalidated. "
            f"Fix: use {cls.__name__}(...) or {cls.__name__}.model_validate(data)."
        )
