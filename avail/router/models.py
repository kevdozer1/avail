"""The model table: ``config/models.yaml`` as typed entries.

The router owns this table because it owns prices and spend. An entry says which
adapter reaches the model, its id, its prices in USD per million tokens, whether
it takes images, and what role it plays in AVAIL:

* ``judge``: asked whether a predicate holds, from 1 to 3 images
* ``decider``: a small typed-decision model run in shadow at forks
* ``agent``: the coding agent attached to the tool server
* ``test``: the offline FakeModel, for tests and dry runs

``verified: true`` means the id and the prices were read from a live catalogue or
from the provider's own pages, and ``source`` and ``accessed`` say where and when.
``verified: false`` means no paid call may use the entry yet. The router refuses
such a call before it spends anything.

Modeled on ``ModelEntry`` and ``ModelsConfig`` in loopkit's ``config.py``.
"""

from __future__ import annotations

import datetime as dt
import math
import re
import threading
from pathlib import Path
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .. import config

#: the adapters the router implements (``avail.router.adapters``)
ADAPTERS = ("openrouter", "elorian", "jev", "fake")
Role = Literal["judge", "decider", "agent", "test"]
ROLES: tuple[str, ...] = get_args(Role)
#: reasoning efforts from least to most, as the OpenRouter catalogue names them
EFFORT_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _iso_date(value: Any, field_name: str) -> Any:
    """``value`` as ISO date text. A ``datetime.date`` (unquoted in YAML) becomes its text."""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, str):
        try:
            if not _DATE_RE.fullmatch(value):
                raise ValueError(value)
            dt.date.fromisoformat(value)
        except ValueError:
            raise ValueError(f"{field_name} must be a date written as YYYY-MM-DD, got {value!r}") from None
    return value


class DirectRoute(BaseModel):
    """Another way to reach the same model: an adapter name and that adapter's id for it."""

    model_config = ConfigDict(extra="forbid")

    adapter: str
    id: str | None


class ModelEntry(BaseModel):
    """One row of the ``models`` block of ``config/models.yaml``.

    Fields without a default must be present (they may be null where the type
    allows it), so a row cannot silently lack its price or its source.
    """

    model_config = ConfigDict(extra="forbid")

    key: str = ""
    display: str
    adapter: str
    id: str | None
    price_in: float | None
    price_out: float | None
    vision: bool
    roles: list[Role]
    verified: bool
    source: str | None
    accessed: str | None
    # optional, adapter-specific
    cached_in: float | None = None
    direct: DirectRoute | None = None
    alt_route: DirectRoute | None = None
    fallback_id: str | None = None
    base_url: str | None = None
    endpoint: str | None = None
    env_key: str | None = None
    json_mode: str | None = None
    reasoning: dict[str, Any] | None = None
    image_tokens: int | None = None
    notes: str | None = None
    #: the catalogue says the model always reasons, so ``disabled`` is sent as its lowest effort
    reasoning_mandatory: bool = False
    #: the reasoning efforts the catalogue lists for the model
    reasoning_efforts: list[str] | None = None
    #: False when the catalogue does not list ``temperature`` for the model. None means unknown
    supports_temperature: bool | None = None

    @field_validator("adapter")
    @classmethod
    def _known_adapter(cls, value: str) -> str:
        if value not in ADAPTERS:
            raise ValueError(f"adapter {value!r} is not one of {', '.join(ADAPTERS)}")
        return value

    @field_validator("price_in", "price_out", "cached_in")
    @classmethod
    def _non_negative(cls, value: float | None) -> float | None:
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError("prices are finite non-negative USD per million tokens (no NaN or infinity)")
        return value

    @field_validator("accessed", mode="before")
    @classmethod
    def _date_text(cls, value: Any) -> Any:
        return _iso_date(value, "accessed")

    @field_validator("reasoning_efforts")
    @classmethod
    def _effort_words(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and (not value or any(not isinstance(e, str) or not e.strip() for e in value)):
            raise ValueError("reasoning_efforts must list at least 1 effort name, or be left out")
        return value

    @model_validator(mode="after")
    def _zero_prices_marked_free(self) -> "ModelEntry":
        # a paid adapter at 0 / 0 is usually a typo or a promotion. The router still treats it as paid
        if self.adapter != "fake" and self.price_in == 0 and self.price_out == 0 \
                and "free" not in (self.notes or "").lower():
            raise ValueError("price_in and price_out are both 0 for a non-fake adapter. If the model really is "
                             "free, say so with notes: free (and why). Otherwise set its prices")
        return self

    @model_validator(mode="after")
    def _verified_is_complete(self) -> "ModelEntry":
        if not self.roles:
            raise ValueError(f"roles must list at least one of {', '.join(ROLES)}")
        if not self.verified:
            return self
        if self.id is None or self.price_in is None or self.price_out is None:
            raise ValueError("verified: true needs id, price_in and price_out")
        if not (self.source or "").strip():
            raise ValueError("verified: true needs a source that says where the id and the prices were read")
        if self.adapter != "fake" and not (self.accessed or "").strip():
            raise ValueError("verified: true needs accessed, the date the id and the prices were read")
        return self

    @property
    def priced(self) -> bool:
        return self.price_in is not None and self.price_out is not None

    @property
    def runnable(self) -> bool:
        """True when the entry has a model id to call."""
        return self.adapter == "fake" or self.id is not None

    @property
    def paid(self) -> bool:
        """True for every adapter that reaches a provider. Only the FakeModel is free."""
        return self.adapter != "fake"

    @property
    def lowest_effort(self) -> str:
        """The least reasoning the model supports (``low`` when the table lists no efforts)."""
        listed = self.reasoning_efforts or []
        known = [effort for effort in EFFORT_ORDER if effort in listed]
        return known[0] if known else "low"


class PendingEntry(BaseModel):
    """One row of the ``to_verify`` block: a model to look up before it can be called.

    ``model_key`` names the row under ``models`` once the id and the prices were
    read and the entry was added there.
    """

    model_config = ConfigDict(extra="forbid")

    display: str
    why: str
    adapter: str
    id: str | None
    roles: list[Role]
    verified: bool
    model_key: str | None = None
    source: str | None = None
    accessed: str | None = None
    notes: str | None = None

    @field_validator("accessed", mode="before")
    @classmethod
    def _date_text(cls, value: Any) -> Any:
        return _iso_date(value, "accessed")

    @model_validator(mode="after")
    def _verified_has_an_id(self) -> "PendingEntry":
        if self.verified and not self.id:
            raise ValueError("verified: true needs the id that was read")
        return self


class ModelsConfig(BaseModel):
    """``config/models.yaml``: the callable ``models`` and the blocks that describe them."""

    model_config = ConfigDict(extra="forbid")

    version: str
    currency: Literal["USD"]
    unit: Literal["usd_per_million_tokens"]
    roles: dict[Role, str] = Field(default_factory=dict)
    carry_over_from_loopkit: dict[Role, list[str]] = Field(default_factory=dict)
    to_verify: dict[str, PendingEntry] = Field(default_factory=dict)
    e1_minimum: list[str] = Field(default_factory=list)
    models: dict[str, ModelEntry]

    @field_validator("version", mode="before")
    @classmethod
    def _version_text(cls, value: Any) -> Any:
        return value.isoformat() if hasattr(value, "isoformat") else value

    @model_validator(mode="after")
    def _consistent(self) -> "ModelsConfig":
        for key, entry in self.models.items():
            entry.key = key
        missing = [key for key in self.e1_minimum if key not in self.models]
        if missing:
            raise ValueError(f"e1_minimum names {', '.join(missing)}, which the models block does not have")
        if self.roles:
            used = {role for entry in self.models.values() for role in entry.roles}
            used |= {role for entry in self.to_verify.values() for role in entry.roles}
            undescribed = sorted(used - set(self.roles))
            if undescribed:
                raise ValueError(f"the roles block has no line for {', '.join(undescribed)}")
        for role, keys in self.carry_over_from_loopkit.items():
            for key in keys:
                if key not in self.models or role not in self.models[key].roles:
                    raise ValueError(f"carry_over_from_loopkit lists {key} as a {role}, but models.{key} is "
                                     "missing or does not have that role")
        for name, pending in self.to_verify.items():
            if pending.model_key is None:
                continue
            target = self.models.get(pending.model_key)
            if target is None or target.id != pending.id:
                raise ValueError(f"to_verify.{name} points at models.{pending.model_key}, which is missing or "
                                 "has another id")
        return self

    def get(self, key: str) -> ModelEntry:
        if key not in self.models:
            hint = (" It is listed under to_verify: read its id and its prices from the live catalogue, then add "
                    "it to models." if key in self.to_verify else "")
            raise KeyError(f"unknown model key {key!r}.{hint} Known: {', '.join(sorted(self.models))}")
        return self.models[key]

    def by_role(self, role: str) -> list[str]:
        """The keys of the entries that have ``role``, in file order."""
        if role not in ROLES:
            raise ValueError(f"role must be one of {', '.join(ROLES)}, got {role!r}")
        return [key for key, entry in self.models.items() if role in entry.roles]


def load_models(path: Path | None = None) -> ModelsConfig:
    """``config/models.yaml`` (or ``path``), validated."""
    return ModelsConfig.model_validate(config.load_models_raw(path))


_MODELS_LOCK = threading.Lock()
_MODELS_CACHE: dict[str, Any] = {"stamp": None, "cfg": None}


def default_models() -> ModelsConfig:
    """``config/models.yaml``, reloaded when the file changes."""
    path = Path(config.MODELS_YAML)
    try:
        st = path.stat()
        stamp: tuple[Any, ...] | None = (str(path), st.st_mtime_ns, st.st_size)
    except OSError:
        stamp = None
    with _MODELS_LOCK:
        if stamp is not None and _MODELS_CACHE["stamp"] == stamp and _MODELS_CACHE["cfg"] is not None:
            return _MODELS_CACHE["cfg"]
        cfg = load_models(path)
        _MODELS_CACHE.update(stamp=stamp, cfg=cfg)
        return cfg
