"""AVAIL's model router: every model call, its cache and its spend ledger.

Every paid call in AVAIL goes through this package (CLAUDE.md rule 6). Start a
piece of work by opening a :class:`Router` for its spend session::

    from avail.router import Router

    router = Router(session="S3", events=run_ledger)   # the S3 cap in config/experiment.yaml is now active
    worst = router.estimate("opus", prompt, image_count=2, json_schema=SCHEMA, max_tokens=600) * n_calls
    answer = router.chat("opus", prompt, images=[top, wrist], json_schema=SCHEMA, max_tokens=600)
    answer.json, answer.json_error, answer.usd, answer.cached, answer.original_usd

    Router().chat("fake", prompt, images=[top], json_schema=SCHEMA)   # no key, no session, $0, a placeholder

``chat`` looks the model up in ``config/models.yaml``, serves a cached answer for
$0 when it has one, projects the worst-case cost and reserves it in the spend
ledger (refusing, before any network I/O, a call that would cross the session cap
or the total cap), sends the call through the entry's adapter, replaces the
reservation with the measured cost, validates a JSON answer and emits one
``model_call`` event to the run's ledger.

What a caller should know:

* Sessions and caps come from ``spend_caps_usd`` in ``config/experiment.yaml``:
  ``S1``, ``S3``, ``S4_without_ok`` and ``full_judge_run``, with ``build_total`` as
  the total. Any other name is an UnknownSessionError. ``full_judge_run`` needs
  ``approved=True``, which is for Kevin's OK only.
* The defaults fit a judge's small JSON answer: role ``judge``, reasoning
  disabled, 1024 output tokens, ``usage`` included. A model the table marks
  ``reasoning_mandatory`` cannot turn reasoning off. It gets its lowest effort,
  ``answer.raw["reasoning_note"]`` says so, and it needs a larger ``max_tokens``.
* An answer that fails its schema is returned with ``json_error`` set. Nothing is
  repaired and nothing raises for it. Check ``answer.json_ok``.
* A cached answer has ``cached`` true and ``usd`` 0. ``original_usd`` and
  ``original_latency_ms`` keep what it cost when it was computed.
* The FakeModel (``fake``) is offline. Its answers are placeholders, so a run
  that used it must report its judges as unmeasured.

Adapters: ``openrouter`` (default), ``elorian``, ``jev`` and ``fake`` (offline, $0,
no key). Keys come from ``.env`` through ``avail.config`` and never reach a log,
an error, the cache or an event. ``avail models`` (``smoke.py``) prints the model
table and, with a key, 1 measured row per model.

Ported from loopkit's ``router/`` with its review fixes and regression tests.
"""

from ..config import MissingCredentialError
from .adapters import available_adapters, get_adapter, register_adapter
from .adapters.fake import FakeRequest, clear_fake_script, fake_calls, fake_script, from_schema, set_fake_script
from .cache import DiskCache
from .core import (
    DEFAULT_MAX_TOKENS,
    Response,
    Router,
    canonical_json,
    chat,
    estimate_usd,
    parse_json_text,
    request_hash,
    resolve_route,
    validate_json,
)
from .errors import (
    ModelCallError,
    ModelNotConfigured,
    RouterError,
    SessionNotApproved,
    SpendCapExceeded,
    UnknownSessionError,
)
from .events import ModelCallPayload
from .http import HttpClient, HttpResult, default_client, set_default_client
from .media import ImageData, normalize_image
from .models import ROLES, ModelEntry, ModelsConfig, default_models, load_models
from .spend import FileLock, SpendCaps, SpendLedger, load_spend_caps

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DiskCache",
    "FakeRequest",
    "FileLock",
    "HttpClient",
    "HttpResult",
    "ImageData",
    "MissingCredentialError",
    "ModelCallError",
    "ModelCallPayload",
    "ModelEntry",
    "ModelNotConfigured",
    "ModelsConfig",
    "ROLES",
    "Response",
    "Router",
    "RouterError",
    "SessionNotApproved",
    "SpendCapExceeded",
    "SpendCaps",
    "SpendLedger",
    "UnknownSessionError",
    "available_adapters",
    "canonical_json",
    "chat",
    "clear_fake_script",
    "default_client",
    "default_models",
    "estimate_usd",
    "fake_calls",
    "fake_script",
    "from_schema",
    "get_adapter",
    "load_models",
    "load_spend_caps",
    "normalize_image",
    "parse_json_text",
    "register_adapter",
    "request_hash",
    "resolve_route",
    "set_default_client",
    "set_fake_script",
    "validate_json",
]
