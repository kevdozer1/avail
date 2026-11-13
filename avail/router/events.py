"""The ``model_call`` event: what the router hands to the run's ledger once per call.

``chat()`` emits exactly 1 ``model_call`` event per call that reached the cache
lookup: a cache hit, an answered call, or a call that failed after an attempt was
reserved. The payload is a flat dict of the fields below. :class:`ModelCallPayload`
is that contract as a pydantic model, so the ledger reader and the tests can
validate a payload and a report can rely on every field being there.

Accounting fields: ``usd`` is the answer's cost on success, and the spend ledger's
count for the call on failure. ``ledger_usd`` is everything the spend ledger
counts for the call, including lost attempts still reserved at their projection
(``unreconciled_attempts``, ``unreconciled_reserved_usd``). A cache hit has
``cache_hit`` true, ``usd`` 0 and ``attempts`` 0, and it carries what the answer
cost and how long it took when it was computed (``original_usd``,
``original_latency_ms``), so a report built from a rerun can still state the cost
of a check.

No field ever holds a key, a request header or image data.

The sink is the run's ``avail.ledger.LedgerWriter`` (its ``append(kind, payload)``
is called), or any object with an ``emit(kind, payload)`` method, or a plain
callable taking ``(kind, payload)``.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .models import Role

EVENT_KIND = "model_call"
_CONTAINERS = (list, tuple, dict, set, frozenset, bytearray, deque, str, bytes)


class ModelCallPayload(BaseModel):
    """The payload of one ``model_call`` event."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    role: Role
    session: str | None
    input_tokens: int | None = Field(ge=0)
    output_tokens: int | None = Field(ge=0)
    reasoning_tokens: int | None = Field(ge=0)
    cached_tokens: int | None = Field(ge=0)
    usd: float = Field(ge=0)
    usd_source: Literal["provider", "computed"]
    latency_ms: float = Field(ge=0)
    cache_hit: bool
    request_hash: str = Field(min_length=64, max_length=64)
    model_key: str
    adapter: str
    ok: bool
    error: str | None
    finish_reason: str | None
    reasoning_setting: Any
    image_count: int = Field(ge=0)
    est_input_tokens: int = Field(ge=0)
    projected_usd: float = Field(ge=0)
    phase: str | None
    json_mode: str | None
    http_status: int | None
    attempts: int | None
    max_tokens: int = Field(ge=1)
    json_error: str | None
    model_requested: str
    upstream_provider: str | None
    route: Literal["primary", "alt_route"]
    ledger_usd: float = Field(ge=0)
    reserved_attempts: int = Field(ge=0)
    unreconciled_attempts: int = Field(ge=0)
    unreconciled_reserved_usd: float = Field(ge=0)
    timeout_s: float | None
    reasoning_budget: int = Field(ge=0)
    #: only on a cache hit: what the answer cost and how long it took when it was computed
    original_usd: float | None = Field(ge=0)
    original_latency_ms: float | None = Field(ge=0)


def _target(sink: Any) -> Any:
    """The callable of ``sink`` that takes ``(kind, payload)``, or None when it has none."""
    if sink is None or isinstance(sink, _CONTAINERS):
        return None
    for name in ("emit", "append"):
        method = getattr(sink, name, None)
        if callable(method):
            return method
    return sink if callable(sink) else None


def check_sink(sink: Any) -> None:
    """TypeError unless ``sink`` is None or can take a ``model_call`` event (see the module docstring)."""
    if sink is None or _target(sink) is not None:
        return
    raise TypeError("events must be None, the run's LedgerWriter, an object with an emit(kind, payload) method, "
                    f"or a callable taking (kind, payload), got {type(sink).__name__}")


def emit(sink: Any, payload: dict[str, Any]) -> None:
    """Hand ``payload`` to ``sink`` as a ``model_call`` event. ``sink`` None does nothing."""
    if sink is None:
        return
    target = _target(sink)
    if target is None:
        check_sink(sink)
        return
    target(EVENT_KIND, payload)
