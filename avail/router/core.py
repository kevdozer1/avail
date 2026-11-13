"""``chat()``: the one door every model call goes through. :class:`Router` opens it for one spend session.

    router = Router(session="S3")                      # the S3 cap from config/experiment.yaml is now active
    answer = router.chat("opus", prompt, images=[top, wrist], json_schema=SCHEMA, max_tokens=600)
    answer.json, answer.usd, answer.cached

Order of operations for one call:

1. Look up the models.yaml entry. A non-fake entry without an id is not callable
   yet (ModelNotConfigured). So is an elorian entry without a base_url, and an
   entry whose base_url or endpoint is not usable. The URL is checked as written
   (leading or trailing whitespace, any space or control character is refused) and
   as the exact URL the adapter will send (httpx must parse it, http or https,
   with a real host and a port from 1 to 65535). A session name the config does
   not have is an UnknownSessionError here, before anything else happens.
2. Normalize messages and images. ``json_schema``, ``extra`` and the reasoning
   setting become plain JSON (numpy scalars and arrays turn into numbers and
   lists), so the request body always serializes. Hash the canonical request:
   adapter, model id, destination URL, messages, the sha256 of each image's bytes
   in order, the schema as order-preserving text (structured output follows
   property order), the JSON modes tried, schema name, max_tokens, the reasoning
   setting actually sent, temperature, cache_salt, extra. That hash keys the
   cache, so it covers the image bytes, the prompt, the model id and the
   generation settings.
3. Cache: a hit returns at once with ``cache_hit`` true and $0 (the original cost
   is kept as ``raw["original_usd"]``). No credential, no spend-ledger write, no
   network.
4. An entry marked ``verified: false`` may not make a paid call
   (ModelNotConfigured). Credentials come through the adapter
   (MissingCredentialError names the variable). A key holding a character that
   cannot go in an HTTP header (a smart quote, a zero-width space) is refused the
   same way, by name and never by value.
5. Projection: ``ceil(text chars / 3.5) + images * image_tokens`` input tokens plus
   ``max_tokens`` of output, plus the reasoning budget when the setting carries one
   (``max_tokens`` or ``budget_tokens``), at the entry's prices (Astra prices when a
   price is unknown).
6. Every HTTP attempt is reserved at the projection, in the call's spend session,
   before it is sent: the first here, each retry and JSON-mode fallback through
   ``AdapterRequest.attempt_hooks`` from the HTTP layer. ``SpendLedger.reserve``
   refuses an attempt that would cross the session cap, the caller's cap for the
   model or the total cap (SpendCapExceeded), before any network I/O for it.
   Every call except the FakeModel at a $0 projection is paid: it needs a
   session, and a reached cap refuses it even at $0.
7. The adapter sends it. Per attempt: a refusal before generation (HTTP 400, 401,
   402, 403, 404 or 429 with no generation id) settles at $0. So does an attempt
   that could not connect (``httpx.ConnectError``, including a DNS failure or a
   refused connection, ``ConnectTimeout`` or ``PoolTimeout``), because nothing
   reached the provider. A lost attempt (read timeout, deadline, a connection
   dropped after the request was sent, read error, 5xx, a 200 without usage) stays
   reserved at its projection, because the provider may have generated and billed
   it. The attempt that answered settles at its cost, which replaces its
   reservation. A cost read from a non-2xx body never settles anything: only an
   answered (2xx) attempt, or the FakeModel's one unsent reservation, settles at a
   reported cost.
8. The answer's cost is the provider's ``usage.cost`` (plus
   ``cost_details.upstream_inference_cost`` on a BYOK account) when given
   (``usd_source: provider``), unless that is below half the cost of the token
   counts it reported at the models.yaml prices (a count it left out counts as 0
   there), in which case the computed cost is used instead (``computed``, with a
   note in ``raw``). Without a provider cost it is computed from tokens
   (``computed``), adding reasoning tokens unless the adapter counts them inside
   the output, and taking a count that was not reported from the projection. A
   token count that is not a whole number from 0 to below 10**12 (a negative or
   fractional number, a string, a bool, a list, NaN, ``10**400``) is treated as
   not reported, with a note naming it in ``raw["usd_note"]`` and the spend
   ledger. When one was and the provider gave no usable cost, the call settles at
   its projection (the cautious choice, since the counts left are not trusted to
   price it). A cost that is not a finite number, or that cannot be computed at
   all, settles at the projection.
9. With a JSON schema, the answer is parsed (code fences and leading prose dropped,
   in linear time whatever the text) and validated. Any parse or validation failure
   becomes ``Response.json_error``, never an exception. No repair here.
10. A successful answer is cached. A generation that failed upstream
    (finish_reason "error") raises ModelCallError and is never cached. One
    ``model_call`` event is emitted per call, also when something fails after the
    provider was paid. The spend ledger is settled (conservatively) before any
    exception is re-raised. Once the provider has answered, chat() raises nothing
    but a RouterError (or an interrupt such as KeyboardInterrupt): an answer the
    adapter cannot read is a ModelCallError, and any other failure after the call
    (a cache write, the event sink) is re-raised as a RouterError that names it.

Key safety: every string that comes back from the provider (the answer, its model
name, finish reason, generation id and upstream provider, the attempt log) has the
key cut in each of its forms (as written and JSON-escaped) before it reaches the
Response, the event, the cache or an error. An answer returned as data (the jev
adapter's answers) has its text rebuilt as ``json.dumps`` of the scrubbed value.
JSON text can spell the key with other escapes too (``\\u0076`` for ``v``), which
parsing decodes. So a parsed answer has the key cut from every string (dict keys
included) before it is validated, ``json_error`` has it cut as well, and when the
parsed value held the key the text is rebuilt the same way, so the cache never
keeps a spelling that a cache hit would decode again. Error text is redacted
before it is cut to length, so a key across the cut cannot survive as a prefix.

Event and Response accounting: ``usd`` is the answer's cost on success (the spend
ledger's count for the call on failure). ``ledger_usd`` is everything the spend
ledger counts for the call, including lost attempts still reserved at their
projection (``unreconciled_attempts``, ``unreconciled_reserved_usd``).

Ported from loopkit's ``router/core.py`` with its review fixes. New here: the
AVAIL roles, the spend session that travels with each call, the refusal of
unverified entries, the defaults for a small JSON answer (reasoning disabled,
1024 output tokens), :func:`estimate_usd` and :class:`Router`.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .. import config
from ..config import MissingCredentialError, load_secret
from . import events as router_events
from .adapters import get_adapter
from .adapters.base import (
    MAX_TOKEN_COUNT,
    Adapter,
    AdapterRequest,
    AdapterResult,
    as_cost,
    reasoning_level,
)
from .cache import DiskCache
from .errors import ModelCallError, ModelNotConfigured, RouterError
from .http import HttpClient, default_client
from .jsonschema_lite import check
from .media import normalize_images
from .models import ROLES, ModelEntry, ModelsConfig, default_models
from .redact import cut_secret, redact, scrub, strip_secret
from .spend import SpendCaps, SpendLedger, _usd, check_label, load_spend_caps

MESSAGE_ROLES = ("system", "user", "assistant")
CHARS_PER_TOKEN = 3.5
DEFAULT_IMAGE_TOKENS = 1600
ASTRA_DEFAULT_PRICES = (10.00, 50.00)
#: the output allowance when the caller gives none: room for a small JSON answer
DEFAULT_MAX_TOKENS = 1024
#: small JSON answers truncate when a model reasons before it answers, so reasoning is off unless asked for
DEFAULT_REASONING = "disabled"
#: per-attempt deadline when the caller gives none. An agent turn can be a long generation
AGENT_TIMEOUT_S = 900.0
DEFAULT_TIMEOUT_S = 300.0
#: a provider total below this fraction of the token-computed cost is not trusted
PROVIDER_COST_FLOOR = 0.5
_JSON_ERRORS_SHOWN = 12
_NOTE_SEP = " | "


@dataclass
class Response:
    """One answer. ``usd`` is what this call cost: 0 for a cached answer and for the FakeModel."""

    text: str
    json: Any | None
    json_error: str | None
    model_key: str
    provider: str
    model: str
    role: str
    session: str | None
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    cached_tokens: int | None
    usd: float
    usd_source: str
    latency_ms: float
    cache_hit: bool
    request_hash: str
    finish_reason: str | None
    reasoning_setting: Any
    json_mode: str | None
    image_count: int
    est_input_tokens: int
    projected_usd: float
    raw: dict  # scrubbed provider response (no key, no base64)

    @property
    def cached(self) -> bool:
        """True when the answer came from the disk cache. It then cost $0 and touched no network."""
        return self.cache_hit

    @property
    def original_usd(self) -> float | None:
        """What the answer cost when it was computed: ``usd`` for a fresh answer, the first call's cost for a
        cached one (None when the cache record does not say). Use it for the cost of a check in a report."""
        if not self.cache_hit:
            return self.usd
        return as_cost(self.raw.get("original_usd"))

    @property
    def original_latency_ms(self) -> float | None:
        """How long the answer took when it was computed: ``latency_ms`` for a fresh answer, the first call's
        latency for a cached one (None when the cache record does not say)."""
        if not self.cache_hit:
            return self.latency_ms
        return as_cost(self.raw.get("original_latency_ms"))

    @property
    def json_ok(self) -> bool:
        """True when a JSON answer parsed and validated."""
        return self.json is not None and self.json_error is None

    @property
    def truncated(self) -> bool:
        """True when the answer was cut at ``max_tokens`` (finish_reason ``length``).

        A model that reasons before it answers can spend the whole allowance on
        reasoning and leave half a JSON object. A truncated answer is cached like
        any other, so ask again with a larger ``max_tokens``, which is a new request.
        """
        return self.finish_reason == "length"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- helpers
def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=repr)
    item = getattr(obj, "item", None)
    if callable(item):  # numpy scalar
        try:
            return item()
        except (TypeError, ValueError):
            pass
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):  # numpy array
        return tolist()
    raise TypeError(f"{type(obj).__name__} is not JSON serializable (the request must be plain JSON data)")


def canonical_json(obj: Any) -> str:
    """Sorted keys, compact separators: the same request always gives the same text."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
                      default=_json_default)


def request_hash(doc: dict[str, Any]) -> str:
    """The sha256 of the canonical request. It keys the cache."""
    return hashlib.sha256(canonical_json(doc).encode("utf-8")).hexdigest()


def plain_json(value: Any, what: str) -> Any:
    """``value`` as plain JSON data (numpy scalars and arrays become numbers and lists), key order kept.

    Raises ValueError or TypeError (a bad argument, before anything is spent).
    """
    if value is None:
        return None
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False, default=_json_default))
    except RecursionError:
        raise ValueError(f"{what} is nested too deeply to send") from None
    except TypeError as exc:
        raise TypeError(f"{what} must be plain JSON data: {exc}") from None
    except ValueError as exc:
        raise ValueError(f"{what} must be plain JSON data (no NaN or infinity): {exc}") from None


def _ordered_json(value: Any) -> str | None:
    """Compact JSON text that keeps key order (a JSON schema's property order changes the request)."""
    if value is None:
        return None
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def normalize_messages(messages: list[dict[str, Any]] | str) -> list[dict[str, str]]:
    """A string becomes 1 user message. A list must hold {role, content} dicts with string content."""
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    if not isinstance(messages, (list, tuple)) or not messages:
        raise ValueError("messages must be a string or a non-empty list of {'role', 'content'} dicts")
    out: list[dict[str, str]] = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(f"messages[{i}] must be a dict with 'role' and 'content'")
        role = msg.get("role")
        content = msg.get("content")
        if role not in MESSAGE_ROLES:
            raise ValueError(f"messages[{i}].role must be one of {', '.join(MESSAGE_ROLES)}, got {role!r}")
        if not isinstance(content, str):
            raise ValueError(f"messages[{i}].content must be a string (attach images with images=, never inside "
                             "content)")
        out.append({"role": role, "content": content})
    return out


def resolve_route(entry: ModelEntry) -> tuple[ModelEntry, str | None]:
    """The entry to call: itself, or its models.yaml ``alt_route`` when only the alternate key is set.

    The alternate route is taken only when the primary adapter's key is missing and
    the alternate adapter's key is present (for example glimpse through OpenRouter
    when ``ELORIAN_API_KEY`` is empty). The returned copy keeps the key, prices and
    reasoning of the entry and swaps in the alternate adapter and id. The second
    value says why the route changed (None when it did not).
    """
    alt = getattr(entry, "alt_route", None)
    if alt is None or not getattr(alt, "id", None) or not getattr(alt, "adapter", None):
        return entry, None
    try:
        primary = get_adapter(entry.adapter)
        names = primary.env_vars(entry)
        if not names or config.has_secret(names):
            return entry, None
        alt_adapter = get_adapter(alt.adapter)
    except ModelNotConfigured:
        return entry, None
    routed = entry.model_copy(update={"adapter": alt.adapter, "id": alt.id, "base_url": None, "endpoint": None,
                                      "env_key": None, "alt_route": None})
    alt_names = alt_adapter.env_vars(routed)
    if alt_names and not config.has_secret(alt_names):
        return entry, None
    return routed, (f"{names[0]} is not set, so {entry.key} goes through its alt_route "
                    f"({alt.adapter}, {alt.id})")


def prices(entry: ModelEntry, models: ModelsConfig) -> tuple[float, float, float | None, str | None]:
    """(price_in, price_out, cached_in, fallback) in USD per million tokens.

    When either of the entry's prices is unknown, both come from the models.yaml
    ``astra`` entry (or 10.00 / 50.00 if that is missing), and ``fallback`` says so.
    """
    if entry.price_in is not None and entry.price_out is not None:
        return float(entry.price_in), float(entry.price_out), entry.cached_in, None
    astra = models.models.get("astra")
    if astra is not None and astra.price_in is not None and astra.price_out is not None:
        return float(astra.price_in), float(astra.price_out), None, "astra"
    return ASTRA_DEFAULT_PRICES[0], ASTRA_DEFAULT_PRICES[1], None, "astra_default"


def _token_count(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return None
    if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
        return int(math.ceil(value))
    return None


def reasoning_budget(setting: Any) -> int:
    """Reasoning tokens a setting allows beyond max_tokens: its ``max_tokens`` or ``budget_tokens``, else 0.

    When both are given the larger counts. Only a dict setting carries a budget.
    """
    if not isinstance(setting, dict):
        return 0
    found = [_token_count(setting.get(name)) for name in ("max_tokens", "budget_tokens")]
    return max((n for n in found if n is not None), default=0)


def project(entry: ModelEntry, models: ModelsConfig, text_chars: int, image_count: int,
            max_tokens: int) -> tuple[int, float, str | None]:
    """(estimated input tokens, worst-case USD, price fallback) for one attempt.

    ``max_tokens`` is the output allowance: the call's max_tokens plus any reasoning budget.
    """
    image_tokens = entry.image_tokens or DEFAULT_IMAGE_TOKENS
    est_in = math.ceil(text_chars / CHARS_PER_TOKEN) + image_count * image_tokens
    pin, pout, _cached, fallback = prices(entry, models)
    return est_in, est_in * pin / 1e6 + max_tokens * pout / 1e6, fallback


TOKEN_COUNT_FIELDS = ("input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens")


def _describe(value: Any) -> str:
    """A provider value for a note: a number as it is (a huge one by its size), anything else by its type only.

    A string is never quoted: it could hold the key.
    """
    if value is None or isinstance(value, (bool, float)):
        return repr(value)
    if isinstance(value, int):
        if abs(value) < 10 ** 15:
            return repr(value)
        return f"an integer of about {int(abs(value).bit_length() * 0.30103) + 1} digits"
    return f"a value of type {type(value).__name__}"


def clean_token_counts(result: Any) -> list[str]:
    """Set each token count of ``result`` that is not a count to None (not reported), with a note for each.

    A count is a whole number from 0 to below 10**12 (an integral float, or a numpy
    integer, is taken as its int). Anything else would break or distort the cost, so
    it is dropped: ``10**400``, ``1e300``, a negative or fractional number, NaN,
    infinity, a bool, a string, a list or an object. The adapters pass the counts as
    the provider reported them, so every one of these gets its note. Returns the
    notes: 1 per count dropped, none when every count was a count or missing.
    """
    notes: list[str] = []
    for name in TOKEN_COUNT_FIELDS:
        value = getattr(result, name, None)
        if value is None:
            continue
        count: int | None = None
        if isinstance(value, float):
            if math.isfinite(value) and value.is_integer():
                count = int(value)
        elif not isinstance(value, bool):
            try:
                count = operator.index(value)  # int, numpy integers
            except Exception:  # noqa: BLE001 - a str, a list, a dict, an object whose __index__ fails: not a count
                count = None
        if count is not None and 0 <= count < MAX_TOKEN_COUNT:
            if type(value) is not int:
                setattr(result, name, int(count))
            continue
        setattr(result, name, None)
        notes.append(f"the provider's {name} ({_describe(value)}) is not a token count (a whole number from 0 to "
                     "below 10**12), so it was treated as not reported")
    return notes


def _token_cost(result: AdapterResult, entry: ModelEntry, models: ModelsConfig, est_in: int,
                max_tokens: int, *, fill_missing: bool = True) -> tuple[float | None, list[str]]:
    """The cost computed from the result's token counts at models.yaml prices (None with no counts).

    With ``fill_missing`` a count the provider did not report comes from the
    projection (the input estimate, the output allowance), with a note. Without it a
    missing count is 0: the cost of what was reported, the floor a provider's own
    figure is checked against. Never raises: counts too large to price cost infinity.
    """
    n_in, n_out = result.input_tokens, result.output_tokens
    if n_in is None and n_out is None:
        return None, []
    notes: list[str] = []
    pin, pout, cached_in, fallback = prices(entry, models)
    if fallback:
        notes.append(f"models.yaml has no price for {entry.key}: cost computed at Astra prices "
                     f"({pin:.2f} / {pout:.2f} per million tokens)")
    if n_in is None:
        n_in = est_in if fill_missing else 0
        notes.append("input tokens not reported: the projection's estimate was used" if fill_missing
                     else "input tokens not reported: counted as 0")
    if n_out is None:
        n_out = max_tokens if fill_missing else 0
        notes.append("output tokens not reported: the output allowance (max_tokens plus any reasoning budget) "
                     "was used" if fill_missing else "output tokens not reported: counted as 0")
    try:
        reasoning = int(result.reasoning_tokens or 0)
        inside = getattr(result, "reasoning_in_output", True)
        if reasoning > 0 and inside is not True:
            n_out += reasoning
            if inside is False:
                notes.append("reasoning tokens are reported separately and were added to the output tokens")
            elif result.output_tokens is not None and reasoning > result.output_tokens:
                notes.append("reasoning tokens exceed the output tokens, so they are reported separately and were "
                             "added")
            else:
                notes.append("whether this provider's output tokens include reasoning tokens is unknown, so they "
                             "were added (conservative)")
        n_cached = min(max(int(result.cached_tokens or 0), 0), int(n_in))
        rate_cached = cached_in if (cached_in is not None and fallback is None) else pin
        usd = ((n_in - n_cached) * pin + n_cached * rate_cached + n_out * pout) / 1e6
    except (ArithmeticError, TypeError, ValueError):  # e.g. OverflowError for a count past float range
        notes.append("the token counts cannot be priced (too large or not numbers)")
        usd = float("inf")
    return usd, notes


def cost_known(result: AdapterResult) -> bool:
    """False when the result carries neither a cost nor a token count (a 200 without usage)."""
    return (as_cost(result.usd) is not None or result.input_tokens is not None
            or result.output_tokens is not None)


def settled_cost(result: AdapterResult, entry: ModelEntry, models: ModelsConfig, projected: float,
                 est_in: int, max_tokens: int, *, dropped: bool = False) -> tuple[float, str, list[str]]:
    """(usd, usd_source, notes) for a call that returned. ``max_tokens`` is the output allowance.

    A provider cost below half of what the token counts it reported cost (a count it
    left out counts as 0) is not trusted: the call is settled at the computed cost,
    which takes a missing count from the projection. ``dropped`` says the provider
    sent a token count that was not a count (:func:`clean_token_counts` dropped it).
    With no usable cost the call is then settled at its projection, the cautious
    choice, since the counts left are not trusted to price it. A usable cost is still
    taken, checked against the counts that were valid.
    """
    notes: list[str] = []
    computed, computed_notes = _token_cost(result, entry, models, est_in, max_tokens)
    reported = as_cost(result.usd)
    if result.usd is not None and reported is None:
        notes.append(f"the provider's cost ({_describe(result.usd)}) is not a finite non-negative number and was "
                     "ignored")
    if reported is None and dropped:
        notes.append("a token count the provider sent is not a count and it reported no usable cost, so the call is "
                     "settled at its projection (the cautious choice)")
        return float(projected), "computed", notes
    if reported is not None:
        floor, _floor_notes = _token_cost(result, entry, models, est_in, max_tokens, fill_missing=False)
        if floor is not None and math.isfinite(floor) and reported < PROVIDER_COST_FLOOR * floor:
            partial = result.input_tokens is None or result.output_tokens is None
            notes.append(f"the provider reported ${reported:.6f}, below half the ${floor:.6f} computed from its "
                         "token counts at models.yaml prices"
                         + (" (the count it did not report counted as 0)" if partial else "")
                         + ", so the call is settled at the computed cost")
            notes.extend(computed_notes)
            usd, source = computed, "computed"
        else:
            return float(reported), "provider", notes
    elif computed is None:
        notes.append("the provider reported no token counts, so the call is settled at its projected worst case")
        usd, source = float(projected), "computed"
    else:
        notes.extend(computed_notes)
        usd, source = computed, "computed"
    if not math.isfinite(usd) or usd < 0:
        notes.append(f"the computed cost {usd!r} is not a finite non-negative number, so the call is settled at its "
                     "projection")
        usd, source = float(projected), "computed"
    return float(usd), source, notes


# --------------------------------------------------------------------------- JSON answers
_FENCE = "```"
#: what may follow an opening fence: spaces or tabs, a language tag, spaces or tabs, then \r and \n
_FENCE_INFO_RE = re.compile(r"[ \t]*[A-Za-z0-9_+-]*[ \t]*\r?\n?")


def _fenced_blocks(text: str) -> list[str]:
    r"""The bodies of ``text``'s fenced code blocks, in order, in linear time.

    They are the bodies ``re.finditer(r"```[ \t]*[A-Za-z0-9_+-]*[ \t]*\r?\n?(.*?)```", text, re.DOTALL)``
    gives, the pattern loopkit used before its review. That pattern backtracks
    cubically: for each way of splitting a run of spaces between its two ``[ \t]*``
    it scans to the end of the text for a closing fence, so a fence and 2,000 spaces
    with none took seconds. Here each fence is found with str.find, its info part is
    matched where it stands (nothing follows it in the pattern, so it never
    backtracks) and the body runs to the next fence. The first fence with no fence
    after its info part ends the search: the info part holds no backtick, so a later
    fence's body cannot start any earlier, and no fence follows.
    """
    bodies: list[str] = []
    start = text.find(_FENCE)
    while start >= 0:
        body = _FENCE_INFO_RE.match(text, start + len(_FENCE)).end()
        close = text.find(_FENCE, body)
        if close < 0:
            break
        bodies.append(text[body:close])
        start = text.find(_FENCE, close + len(_FENCE))
    return bodies


def _clip(text: str, n: int = 160) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 3] + "..."


def _decode_error(exc: BaseException) -> str:
    if isinstance(exc, json.JSONDecodeError):
        return f"not JSON: {exc.msg} at line {exc.lineno} column {exc.colno}"
    if isinstance(exc, RecursionError):
        return "not JSON: nested too deeply to parse"
    if isinstance(exc, ValueError):  # for example an integer literal longer than Python reads
        return f"not JSON: {_clip(str(exc))}"
    return f"not JSON: {type(exc).__name__}: {_clip(str(exc))}"


def _parse_candidate(text: str) -> tuple[Any, str | None]:
    s = text.strip()
    if not s:
        return None, "empty answer"
    try:
        return json.loads(s), None
    except Exception as exc:  # noqa: BLE001 - JSONDecodeError, ValueError, RecursionError: all a json_error
        first: BaseException = exc
    starts = sorted(i for i in (s.find("{"), s.find("[")) if i >= 0)
    if not starts:
        return None, _decode_error(first)
    last: BaseException = first
    decoder = json.JSONDecoder()
    for start in starts:
        body = s[start:]
        try:
            value, _end = decoder.raw_decode(body)
            return value, None
        except Exception as exc:  # noqa: BLE001
            last = exc
        end = max(body.rfind("}"), body.rfind("]"))
        if end > 0:
            try:
                return json.loads(body[: end + 1]), None
            except Exception:  # noqa: BLE001
                pass
    return None, _decode_error(last)


def parse_json_text(text: str | None) -> tuple[Any, str | None]:
    """A model's JSON answer: (value, None), or (None, why it is not JSON). Never raises.

    Code fences are stripped, and prose before the first ``{`` or ``[`` is dropped.
    """
    try:
        if text is None or not str(text).strip():
            return None, "empty answer"
        s = str(text).strip()
        candidates = _fenced_blocks(s)
        if s.startswith(_FENCE) and not candidates:  # an opening fence the answer never closed (truncated)
            body = s[3:]
            newline = body.find("\n")
            candidates.append(body[newline + 1:] if newline >= 0 else body)
        candidates.append(s)
        error = "not JSON"
        for candidate in candidates:
            value, err = _parse_candidate(candidate)
            if err is None:
                return value, None
            error = err
        return None, error
    except Exception as exc:  # noqa: BLE001 - a degenerate answer is a json_error, never an exception
        return None, _decode_error(exc)


def validate_json(value: Any, schema: dict[str, Any]) -> str | None:
    """None when ``value`` satisfies ``schema``, else the errors joined into 1 line. Never raises."""
    try:
        errors = check(value, schema)
    except RecursionError:
        return "schema check failed: the answer is nested too deeply to check"
    except Exception as exc:  # noqa: BLE001 - e.g. OverflowError for a 400-digit number under "type": "number"
        return f"schema check failed: {type(exc).__name__}: {_clip(str(exc))}"
    if not errors:
        return None
    shown = [str(e) for e in errors[:_JSON_ERRORS_SHOWN]]
    more = f" (and {len(errors) - len(shown)} more)" if len(errors) > len(shown) else ""
    return _NOTE_SEP.join(shown) + more


def _json_result(text: str, json_value: Any, json_schema: dict[str, Any] | None) -> tuple[Any, str | None]:
    """(value, json_error) for a cached answer (its text holds no form of the key: see :func:`_fresh_json`)."""
    try:
        if json_schema is not None:
            value, err = parse_json_text(text)
            if err is not None:
                return None, err
            return value, validate_json(value, json_schema)
        if json_value is not None:
            return json_value, None
        return None, None
    except Exception as exc:  # noqa: BLE001 - never raise after a paid call
        return None, f"could not read the answer as JSON: {type(exc).__name__}: {_clip(str(exc))}"


def _rebuilt_text(value: Any) -> str:
    """``json.dumps`` of a parsed answer that held the key, or "" when it cannot be written (never cached).

    Never the answer's own text, which spells the key in a form a parse decodes.
    """
    try:
        return json.dumps(value)
    except (TypeError, ValueError, RecursionError):
        return ""


_KEPT_RAW_NOTES = ("usd_note", "route_note", "price_fallback", "reasoning_budget_projected", "reasoning_note",
                   "parameters_left_out")


def _without_answer(raw: dict[str, Any]) -> dict[str, Any]:
    """``raw`` with the provider's body left out: the router's own notes and the numeric usage counts only.

    For an answer that held the key in a spelling parsing decodes: the body keeps the answer as
    written, so anyone parsing ``raw`` (or a cache hit's ``raw``) would get the key back.
    """
    kept = {k: raw[k] for k in _KEPT_RAW_NOTES if k in raw}
    usage = raw.get("usage")
    if isinstance(usage, dict):
        kept["usage"] = {k: v for k, v in usage.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    kept["provider_body"] = "left out: the answer held the API key in an escaped spelling"
    return kept


def _fresh_json(text: str, json_value: Any, json_schema: dict[str, Any] | None,
                key: str) -> tuple[str, Any, str | None]:
    """(text, value, json_error) for an answer just received, with the key in none of them. Never raises.

    The text arrives with the key cut as written and JSON-escaped, but JSON can spell
    it with other escapes (``\\u0076`` for ``v``, ``\\u0022`` for a quote) that parsing
    decodes. So the parsed value has the key cut from every string, dict keys
    included, before it is validated (a validation message quotes a value cut to 60
    characters, which could keep part of the key), and ``json_error`` has it cut too.
    When the value held the key, the text becomes ``json.dumps`` of the scrubbed
    value, so neither the Response nor the cache keeps a spelling that a later parse
    (every cache hit) would decode again.
    """
    if json_schema is None:  # no schema, so no value to return, but a parse could still decode the key later
        try:
            parsed, error = parse_json_text(text)
            if error is None and cut_secret(parsed, key)[1]:
                return _rebuilt_text(cut_secret(parsed, key)[0]), json_value, None
        except Exception:  # noqa: BLE001 - never raise after a paid call
            pass
        return text, json_value, None
    try:
        value, error = parse_json_text(text)
        if error is None:
            value, found = cut_secret(value, key)
            if found:
                text = _rebuilt_text(value)
            error = validate_json(value, json_schema)
    except Exception as exc:  # noqa: BLE001 - never raise after a paid call
        value, error = None, (f"could not read the answer as JSON: {type(exc).__name__}: "
                              f"{_clip(strip_secret(str(exc), key))}")
    return text, value, strip_secret(error, key)


def _text_chars(msgs: list[dict[str, str]], json_schema: dict[str, Any] | None, extra: dict[str, Any]) -> int:
    chars = sum(len(m["content"]) for m in msgs)
    if json_schema is not None:
        chars += len(canonical_json(json_schema))
    if extra:
        chars += len(canonical_json(extra))
    return chars


def _projection(entry: ModelEntry, models: ModelsConfig, msgs: list[dict[str, str]],
                json_schema: dict[str, Any] | None, extra: dict[str, Any], image_count: int, max_tokens: int,
                reasoning_setting: Any) -> tuple[int, float, str | None, int, int]:
    """(est input tokens, worst-case USD, price fallback, reasoning budget, output allowance) of one attempt."""
    budget = reasoning_budget(reasoning_setting)
    max_out = max_tokens + budget
    est_in, projected, fallback = project(entry, models, _text_chars(msgs, json_schema, extra), image_count, max_out)
    return est_in, projected, fallback, budget, max_out


def _check_max_tokens(max_tokens: Any) -> int:
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError(f"max_tokens must be a positive integer, got {max_tokens!r}")
    return max_tokens


def _check_key(key: str, names: list[str], model_key: str, adapter: str) -> None:
    """A key must be printable ASCII without spaces: it goes into an HTTP header. Names the variable only."""
    if not key:
        return
    if key.isascii() and all(32 < ord(c) < 127 for c in key):
        return
    var = next((n for n in names if config.env(n)), names[0] if names else "the API key variable")
    what = ("a character that is not ASCII (a smart quote or a zero-width space, for example)" if not key.isascii()
            else "a space or a control character")
    raw = os.environ.get(var)                       # where config.env took it from: the environment wins over .env
    if raw is not None and raw.strip() != "":
        where = ("The value comes from the environment, which wins over .env. Correct the key in the .env beside the "
                 "env.ps1 you dot-sourced, using plain ASCII characters only, then dot-source that env.ps1 again.")
    else:
        where = f"Re-enter the key in {config.ENV_FILE} using plain ASCII characters only."
    raise MissingCredentialError(
        f"{model_key} ({adapter}): {var} holds {what}, so it cannot be sent in an HTTP header. {where}") from None


# --------------------------------------------------------------------------- spend per attempt
class _CallSpend:
    """The spend-ledger side of one chat() call: 1 reservation per HTTP attempt.

    The first attempt is reserved by chat() before the adapter runs. The HTTP layer
    calls :meth:`before_attempt` and :meth:`after_attempt` around every POST
    (``avail.router.http.AttemptHooks``), so each retry and JSON-mode fallback is
    reserved before it is sent and refused when a cap would be crossed.
    """

    def __init__(self, ledger: SpendLedger, projected: float, model_key: str, rhash: str, tags: dict[str, Any],
                 paid: bool, *, session: str | None, approved: bool, model_cap_usd: float | None):
        self.ledger = ledger
        self.projected = float(projected)
        self.model_key = model_key
        self.rhash = rhash
        self.tags = dict(tags)
        self.paid = bool(paid)
        self.session = session
        self.approved = bool(approved)
        self.model_cap_usd = model_cap_usd
        self.attempts: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ reservations
    def _reserve(self) -> dict[str, Any]:
        n = len(self.attempts) + 1
        rid = self.ledger.reserve(self.projected, self.model_key, self.rhash, session=self.session,
                                  tags={**self.tags, "attempt": n}, paid=self.paid,
                                  model_cap_usd=self.model_cap_usd, approved=self.approved)
        rec = {"rid": rid, "n": n, "state": "reserved", "status": None, "usd": self.projected, "settled": False}
        self.attempts.append(rec)
        return rec

    def reserve_first(self) -> None:
        self._reserve()

    def before_attempt(self, *, url: str | None = None, attempt: int | None = None) -> str:
        last = self.attempts[-1] if self.attempts else None
        rec = last if last is not None and last["state"] == "reserved" else self._reserve()
        rec["state"] = "sent"
        return rec["rid"]

    def after_attempt(self, token: Any, *, outcome: str, status: int | None = None,
                      generation_id: str | None = None) -> None:
        rec = next((a for a in self.attempts if a["rid"] == token), None)
        if rec is None:
            return
        rec["state"], rec["status"] = outcome, status
        if outcome == "rejected":
            self._settle(rec, 0.0, "computed", f"HTTP {status} before generation, recorded as 0", quiet=True)
        elif outcome == "not_sent":  # still an attempt (and retried), but nothing reached the provider
            self._settle(rec, 0.0, "computed", "no connection could be made, so nothing reached the provider. "
                         "Recorded as 0", quiet=True)

    # ------------------------------------------------------------------ settling
    def _settle(self, rec: dict[str, Any], usd: float, source: str, note: str | None, *, quiet: bool) -> None:
        try:
            self.ledger.settle(rec["rid"], usd, source, note=note)
        except Exception:
            if quiet:  # unsettled is the conservative outcome: it stays counted at the projection
                return
            raise
        rec["usd"], rec["settled"] = float(usd), True

    def _answer_record(self) -> dict[str, Any] | None:
        """The attempt a reported cost may settle, or None.

        The last attempt that answered (a 2xx), or, for a call that sends no HTTP (the
        FakeModel), its one reservation, which was never sent. Never a lost or refused
        attempt: a cost read from a non-2xx body settles nothing.
        """
        for rec in reversed(self.attempts):
            if rec["state"] == "answered":
                return rec
        if len(self.attempts) == 1 and self.attempts[0]["state"] == "reserved":
            return self.attempts[0]
        return None

    @property
    def answered(self) -> bool:
        """True once an attempt got a 2xx answer: the provider answered."""
        return any(a["state"] == "answered" for a in self.attempts)

    @property
    def answered_status(self) -> int | None:
        return next((a["status"] for a in reversed(self.attempts) if a["state"] == "answered"), None)

    def settle_answer(self, usd: float, source: str, note: str | None, *, known: bool) -> None:
        """Settle the attempt that answered at ``usd``. With ``known`` false it stays reserved."""
        rec = self._answer_record()
        if rec is None:
            return
        if rec["state"] == "reserved":  # a call that sends no HTTP (the FakeModel): its one reservation
            rec["state"] = "answered"
        if known:
            self._settle(rec, usd, source, note, quiet=False)

    def settle_failure(self, exc: BaseException, cost_of: Any) -> str:
        """Settle what a failed call's error tells. Everything else stays reserved. Returns the usd_source.

        A cost on the error (``exc.usd``) settles only the attempt that answered (a 2xx
        whose generation failed), or the FakeModel's one unsent reservation. When no
        attempt answered it settles nothing, so every lost attempt stays counted at its
        projection.
        """
        if not isinstance(exc, ModelCallError):
            return "computed"
        if exc.rejected_before_generation:
            for rec in self.attempts:  # sent attempts were settled as they were refused
                if rec["state"] == "reserved" and not rec["settled"]:
                    self._settle(rec, 0.0, "computed", "nothing was sent for this reservation (the request was "
                                 "refused before generation or could not be built), recorded as 0", quiet=True)
            return "computed"
        if exc.usd is not None:
            rec = self._answer_record()
            if rec is None:
                return "computed"
            usd, source, notes = cost_of(exc)
            if rec["state"] == "reserved":
                rec["state"] = "answered"
            self._settle(rec, usd, source,
                         "failed call: " + (_NOTE_SEP.join(notes) or "cost reported with the error"), quiet=True)
            return source if rec["settled"] else "computed"
        return "computed"

    # ------------------------------------------------------------------ accounting
    @property
    def sent(self) -> int:
        return sum(1 for a in self.attempts if a["state"] != "reserved")

    def counted(self) -> float:
        """What the spend ledger counts for this call: settled amounts, and projections for the rest."""
        return sum(a["usd"] for a in self.attempts)

    def fields(self) -> dict[str, Any]:
        open_ = [a for a in self.attempts if not a["settled"]]
        return {"ledger_usd": round(self.counted(), 10), "reserved_attempts": len(self.attempts),
                "unreconciled_attempts": len(open_),
                "unreconciled_reserved_usd": round(sum(a["usd"] for a in open_), 10)}


def _failed_call_cost(exc: ModelCallError, adapter: Adapter, entry: ModelEntry, models: ModelsConfig,
                      projected: float, est_in: int, max_out: int) -> tuple[float, str, list[str]]:
    """The cost of a failed call that reported one, checked against its token counts like any other."""
    usage = exc.raw.get("usage") if isinstance(exc.raw, dict) and isinstance(exc.raw.get("usage"), dict) else {}
    probe = AdapterResult(text="", usd=exc.usd, reasoning_in_output=adapter.reasoning_in_output,
                          **{name: usage.get(name) for name in TOKEN_COUNT_FIELDS})  # as reported, like an answer's
    notes = clean_token_counts(probe)
    usd, source, cost_notes = settled_cost(probe, entry, models, projected, est_in, max_out, dropped=bool(notes))
    return usd, source, notes + cost_notes


def _provider_text(value: Any, key: str) -> str | None:
    """A string the provider sent (a model name, finish reason, generation id) without the key in any form."""
    if value is None:
        return None
    return strip_secret(value if isinstance(value, str) else str(value), key)


def _answer_text(value: Any, fallback: str) -> str:
    """The text of an answer returned as data (the jev adapter's answers): ``json.dumps`` of the scrubbed value."""
    try:
        return json.dumps(value)
    except (TypeError, ValueError, RecursionError):
        return fallback


def _scrub_error(exc: ModelCallError, key: str) -> None:
    """A failed call's message, generation id and raw details (the attempt log among them) without the key.

    The adapters scrub their messages already. This is the one place that makes sure, for any adapter.
    Never raises.
    """
    try:
        if exc.args and isinstance(exc.args[0], str):
            exc.args = (redact(exc.args[0], key, media=False), *exc.args[1:])
        if exc.generation_id is not None:
            exc.generation_id = _provider_text(exc.generation_id, key)
        if exc.raw:
            exc.raw = scrub(exc.raw, key, media=False)
    except Exception:  # noqa: BLE001 - the call's own error matters more
        pass


def _after_answer(exc: BaseException, model_key: str, model_id: str, label: str, key: str) -> BaseException:
    """``exc``, raised after the provider answered, as a RouterError: nothing else leaves chat() then.

    A RouterError, and an interrupt (KeyboardInterrupt, SystemExit), come back unchanged.
    """
    if isinstance(exc, RouterError) or not isinstance(exc, Exception):
        return exc
    # redacted before it is cut to length: a key across the cut would otherwise survive as a prefix
    return RouterError(redact(f"{model_key} ({model_id}) via {label}: the provider answered, but the router failed "
                              f"after the call ({type(exc).__name__}: {_clip(redact(str(exc), key), 300)})", key))


# --------------------------------------------------------------------------- the model_call event
def _payload(*, provider: str, model: str, role: str, session: str | None, input_tokens: int | None,
             output_tokens: int | None, reasoning_tokens: int | None, cached_tokens: int | None, usd: float,
             usd_source: str, latency_ms: float, cache_hit: bool, request_hash: str, model_key: str, adapter: str,
             ok: bool, error: str | None, finish_reason: str | None, reasoning_setting: Any, image_count: int,
             est_input_tokens: int, projected_usd: float, phase: str | None, json_mode: str | None,
             http_status: int | None, attempts: int | None, max_tokens: int, json_error: str | None,
             model_requested: str, upstream_provider: str | None, route: str = "primary",
             ledger_usd: float = 0.0, reserved_attempts: int = 0, unreconciled_attempts: int = 0,
             unreconciled_reserved_usd: float = 0.0, timeout_s: float | None = None,
             reasoning_budget: int = 0, original_usd: Any = None,
             original_latency_ms: Any = None) -> dict[str, Any]:
    """A ``model_call`` payload: every field of ``avail.router.events.ModelCallPayload``."""
    return {
        "provider": provider, "model": model, "role": role, "session": session,
        "input_tokens": input_tokens, "output_tokens": output_tokens, "reasoning_tokens": reasoning_tokens,
        "usd": round(float(usd), 10), "usd_source": usd_source, "latency_ms": round(float(latency_ms), 3),
        "cache_hit": bool(cache_hit), "request_hash": request_hash,
        "model_key": model_key, "adapter": adapter, "ok": bool(ok), "error": error,
        "finish_reason": finish_reason, "reasoning_setting": reasoning_setting, "image_count": int(image_count),
        "est_input_tokens": int(est_input_tokens), "projected_usd": round(float(projected_usd), 10),
        "phase": phase, "json_mode": json_mode, "http_status": http_status, "attempts": attempts,
        "cached_tokens": cached_tokens, "max_tokens": int(max_tokens), "json_error": json_error,
        "model_requested": model_requested, "upstream_provider": upstream_provider, "route": route,
        "ledger_usd": round(float(ledger_usd), 10), "reserved_attempts": int(reserved_attempts),
        "unreconciled_attempts": int(unreconciled_attempts),
        "unreconciled_reserved_usd": round(float(unreconciled_reserved_usd), 10), "timeout_s": timeout_s,
        "reasoning_budget": int(reasoning_budget),
        # a number that is finite and not negative, or None. Only a cache hit has them
        "original_usd": as_cost(original_usd), "original_latency_ms": as_cost(original_latency_ms),
    }


def _emit(sink: Any, payload: dict[str, Any]) -> None:
    router_events.emit(sink, payload)


# --------------------------------------------------------------------------- estimate
def estimate_usd(model_key: str, messages: list[dict] | str, *, image_count: int | None = None,
                 images: list | None = None, json_schema: dict | None = None, max_tokens: int = DEFAULT_MAX_TOKENS,
                 reasoning: str | dict | None = DEFAULT_REASONING, extra: dict | None = None,
                 models: ModelsConfig | None = None) -> float:
    """The worst-case USD of 1 attempt of this call, without calling anything. $0 for the FakeModel.

    It is the amount chat() would reserve for the same arguments, so multiply it
    by the number of calls for the estimate to print before a paid run. A retry
    or a JSON-mode fallback reserves the same amount again, and the measured cost
    is usually far below it. Give the images themselves or only their count.
    """
    if not isinstance(model_key, str):
        raise TypeError(f"model_key must be a string, got {type(model_key).__name__}")
    cfg = models if models is not None else default_models()
    entry, _route_note = resolve_route(cfg.get(model_key))
    adapter = get_adapter(entry.adapter)
    _check_max_tokens(max_tokens)
    if images is not None and image_count is not None and image_count != len(images):
        raise ValueError(f"image_count is {image_count} but {len(images)} images were given. Pass one of the two")
    n_images = len(images) if images is not None else (image_count or 0)
    if isinstance(n_images, bool) or not isinstance(n_images, int) or n_images < 0:
        raise ValueError(f"image_count must be a whole number from 0 up, got {image_count!r}")
    if n_images and not entry.vision:
        raise RouterError(f"model {model_key!r} is text-only in config/models.yaml (vision: false), "
                          f"so it cannot take the {n_images} image(s) given")
    msgs = normalize_messages(messages)
    schema_d = plain_json(json_schema, "json_schema")
    extra_d = plain_json(extra, "extra") or {}
    setting = plain_json(adapter.normalize_reasoning(reasoning, entry), "reasoning")
    return _projection(entry, cfg, msgs, schema_d, extra_d, n_images, max_tokens, setting)[1]


# --------------------------------------------------------------------------- chat
def chat(model_key: str, messages: list[dict] | str, images: list | None = None, json_schema: dict | None = None,
         role: str = "judge", max_tokens: int = DEFAULT_MAX_TOKENS,
         reasoning: str | dict | None = DEFAULT_REASONING, *, session: str | None = None, approved: bool = False,
         model_cap_usd: float | None = None, temperature: float | None = None, schema_name: str = "response",
         use_cache: bool = True, cache_salt: str | None = None, events: Any = None, phase: str | None = None,
         timeout_s: float | None = None, models: ModelsConfig | None = None, ledger: SpendLedger | None = None,
         extra: dict | None = None, cache: DiskCache | None = None, http: HttpClient | None = None) -> Response:
    """Call ``model_key`` (a key in config/models.yaml) through its adapter. See the module docstring.

    ``session`` is the spend session the call is charged to (a key of
    ``spend_caps_usd`` in config/experiment.yaml). Most callers use
    :class:`Router`, which passes it. ``role`` is ``judge``, ``decider``, ``agent``
    or ``test`` and is recorded with the call. ``images`` attach to the last user
    message (a judge call sends 1 to 3). ``reasoning`` defaults to ``disabled``,
    which is what a small JSON answer needs. ``events`` receives 1 ``model_call``
    event per call (the run's LedgerWriter, see ``avail.router.events``).
    ``timeout_s`` is the per-attempt deadline. None means 900 s for an agent call
    and 300 s otherwise. ``cache``, ``ledger`` and ``http`` replace the default
    disk cache, spend ledger and HTTP client.

    Raises ModelNotConfigured, MissingCredentialError, UnknownSessionError,
    SessionNotApproved, SpendCapExceeded, ModelCallError or RouterError. A bad
    argument raises ValueError or TypeError before anything is spent.
    """
    # 1. the entry (and its alt_route when only the alternate key is set), and the session
    if not isinstance(model_key, str):
        raise TypeError(f"model_key must be a string, got {type(model_key).__name__}")
    check_label(model_key, "model_key")
    router_events.check_sink(events)
    if session is not None:
        if not isinstance(session, str):
            raise TypeError(f"session must be a string or None, got {type(session).__name__}")
        check_label(session, "session name")
        (ledger.caps if ledger is not None else load_spend_caps()).session_cap(session)  # UnknownSessionError
    if model_cap_usd is not None:
        model_cap_usd = _usd(model_cap_usd, f"the cap for model {model_key}")
    cfg = models if models is not None else default_models()
    configured = cfg.get(model_key)
    entry, route_note = resolve_route(configured)
    adapter: Adapter = get_adapter(entry.adapter)
    if entry.adapter != "fake" and not entry.id:
        raise ModelNotConfigured(
            f"model {model_key!r} has no id in config/models.yaml: the entry is marked unverified "
            f"(verified: {str(entry.verified).lower()}). Look up its id and price, set them, then call it.")
    adapter.check_entry(entry)
    if role not in ROLES:
        raise ValueError(f"role must be one of {', '.join(ROLES)}, got {role!r}")
    _check_max_tokens(max_tokens)
    if timeout_s is None:
        timeout_s = AGENT_TIMEOUT_S if role == "agent" else DEFAULT_TIMEOUT_S
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) \
            or timeout_s <= 0:
        raise ValueError(f"timeout_s must be a positive number of seconds, got {timeout_s!r}")
    timeout = float(timeout_s)
    if json_schema is not None and not isinstance(json_schema, dict):
        raise TypeError("json_schema must be a dict (a JSON Schema) or None")
    if extra is not None and not isinstance(extra, dict):
        raise TypeError("extra must be a dict or None")
    if phase is not None:
        if not isinstance(phase, str):
            raise TypeError(f"phase must be a string or None, got {type(phase).__name__}")
        check_label(phase, "phase", allow_empty=True)
    if cache_salt is not None and not isinstance(cache_salt, str):
        cache_salt = str(cache_salt)
    reasoning_level(reasoning)  # a bad setting fails here for every adapter, even one that ignores it

    # 2. normalize and hash (plain JSON before anything else, so the body always serializes once money is reserved)
    msgs = normalize_messages(messages)
    imgs = normalize_images(images)
    target: int | None = None
    if imgs:
        if not entry.vision:
            raise RouterError(f"model {model_key!r} is text-only in config/models.yaml (vision: false), "
                              f"so it cannot take the {len(imgs)} image(s) given")
        target = next((i for i in range(len(msgs) - 1, -1, -1) if msgs[i]["role"] == "user"), None)
        if target is None:
            raise ValueError("images attach to the last user message, but messages has no user message")
    schema_d: dict[str, Any] | None = plain_json(json_schema, "json_schema")
    extra_d: dict[str, Any] = plain_json(extra, "extra") or {}
    reasoning_setting = plain_json(adapter.normalize_reasoning(reasoning, entry), "reasoning")
    reasoning_note = adapter.reasoning_note(reasoning, entry)
    temperature = plain_json(temperature, "temperature")
    model_id = entry.id or "fake"
    request = AdapterRequest(
        model_key=model_key, model_id=model_id, entry=entry, role=role, messages=msgs, images=imgs,
        image_message_index=target, json_schema=schema_d, schema_name=schema_name, max_tokens=max_tokens,
        reasoning=reasoning_setting, reasoning_requested=reasoning, temperature=temperature,
        timeout_s=timeout, extra=extra_d)
    adapter.check_request(request)
    doc: dict[str, Any] = {
        "adapter": entry.adapter, "model": model_id, "url": adapter.url(entry), "messages": msgs,
        "images": [i.sha256 for i in imgs],  # the sha256 of the bytes that are sent, in order
        "json_schema": _ordered_json(schema_d),  # property order is part of the request
        "json_modes": adapter.modes(request) if schema_d is not None else None,
        "schema_name": schema_name, "max_tokens": max_tokens, "reasoning": reasoning_setting,
        "temperature": temperature, "cache_salt": cache_salt, "extra": extra_d,
    }
    token = adapter.cache_token(request)
    if token:
        doc["fake_script"] = token
    rhash = request_hash(doc)
    request.request_hash = rhash
    est_in, projected, price_fallback, budget, max_out = _projection(
        entry, cfg, msgs, schema_d, extra_d, len(imgs), max_tokens, reasoning_setting)

    def payload(**kw: Any) -> dict[str, Any]:
        base = dict(provider=adapter.name, model=model_id, role=role, session=session, input_tokens=None,
                    output_tokens=None, reasoning_tokens=None, cached_tokens=None, usd=0.0, usd_source="computed",
                    latency_ms=0.0, cache_hit=False, request_hash=rhash, model_key=model_key,
                    adapter=entry.adapter, ok=True, error=None, finish_reason=None,
                    reasoning_setting=reasoning_setting, image_count=len(imgs), est_input_tokens=est_in,
                    projected_usd=projected, phase=phase, json_mode=None, http_status=None, attempts=0,
                    max_tokens=max_tokens, json_error=None, model_requested=model_id, upstream_provider=None,
                    route="alt_route" if route_note else "primary", timeout_s=timeout, reasoning_budget=budget)
        base.update(kw)
        return _payload(**base)

    # 3. cache
    store = cache if cache is not None else DiskCache()
    if use_cache:
        t0 = time.perf_counter()
        hit = store.get(rhash)
        if hit is not None and hit.get("finish_reason") == "error":
            hit = None  # a generation that failed upstream is never an answer
        if hit is not None:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            value, json_error = _json_result(hit["text"], hit.get("json_value"), schema_d)
            raw = dict(hit.get("raw") or {})
            raw.update(original_usd=hit.get("usd"), original_usd_source=hit.get("usd_source"),
                       original_latency_ms=hit.get("latency_ms"), cached_at=hit.get("created"))
            if hit.get("ledger_usd") is not None:
                raw["original_ledger_usd"] = hit.get("ledger_usd")
            response = Response(
                text=hit["text"], json=value, json_error=json_error, model_key=model_key, provider=adapter.name,
                model=hit.get("model") or model_id, role=role, session=session,
                input_tokens=hit.get("input_tokens"), output_tokens=hit.get("output_tokens"),
                reasoning_tokens=hit.get("reasoning_tokens"), cached_tokens=hit.get("cached_tokens"), usd=0.0,
                usd_source="computed", latency_ms=latency_ms, cache_hit=True, request_hash=rhash,
                finish_reason=hit.get("finish_reason"),
                reasoning_setting=hit.get("reasoning_setting", reasoning_setting), json_mode=hit.get("json_mode"),
                image_count=len(imgs), est_input_tokens=est_in, projected_usd=projected, raw=raw)
            _emit(events, payload(model=response.model, input_tokens=response.input_tokens,
                                  output_tokens=response.output_tokens, reasoning_tokens=response.reasoning_tokens,
                                  cached_tokens=response.cached_tokens, latency_ms=latency_ms, cache_hit=True,
                                  finish_reason=response.finish_reason, json_mode=response.json_mode,
                                  reasoning_setting=response.reasoning_setting, json_error=json_error,
                                  upstream_provider=hit.get("upstream_provider"),
                                  original_usd=hit.get("usd"), original_latency_ms=hit.get("latency_ms")))
            return response

    # 4. the paid path starts here: the entry must be verified, and the key must be there
    if entry.adapter != "fake" and not entry.verified:
        raise ModelNotConfigured(
            f"model {model_key!r} is marked verified: false in config/models.yaml, so no paid call may use it yet. "
            "Read its id and its prices from the live catalogue, set verified: true with source and accessed, "
            "then call it.")
    names = adapter.env_vars(entry)
    key = ""
    if names:
        try:
            key = load_secret(names, adapter.label)
        except MissingCredentialError as exc:
            hint = ""
            alt = getattr(configured, "alt_route", None)
            if alt is not None and getattr(alt, "id", None):
                try:
                    alt_vars = get_adapter(alt.adapter).default_env_vars
                except ModelNotConfigured:
                    alt_vars = ()
                if alt_vars:
                    hint = f" Or set {alt_vars[0]} to call it through its alt_route ({alt.adapter}, {alt.id})."
            raise MissingCredentialError(f"{model_key} ({entry.adapter}): {exc}{hint}") from None
        _check_key(key, names, model_key, entry.adapter)

    # 5-6. projection (above) and the first attempt's reservation. Later attempts reserve through the hooks
    led = ledger if ledger is not None else SpendLedger()
    tags = {"role": role, "phase": phase, "adapter": entry.adapter, "model": model_id,
            "call": uuid.uuid4().hex[:16]}
    if approved and session is not None and led.caps.needs_approval(session):
        tags["approved"] = True
    spend = _CallSpend(led, projected, model_key, rhash, tags=tags,
                       paid=entry.adapter != "fake" or projected > 0, session=session, approved=approved,
                       model_cap_usd=model_cap_usd)
    spend.reserve_first()  # a refusal here means nothing was attempted, so nothing is sent and no event is written
    request.attempt_hooks = spend

    # 7. send
    client = http if http is not None else default_client()
    t_send = time.perf_counter()
    sent_error: BaseException | None = None
    try:
        result = adapter.send(request, key, client)
    except BaseException as exc:  # noqa: BLE001 - settled, recorded and re-raised just below
        sent_error = exc
    if sent_error is not None:
        latency_ms = (time.perf_counter() - t_send) * 1000.0
        error = sent_error
        if spend.answered and isinstance(error, Exception) and not isinstance(error, RouterError):
            # the provider answered but the adapter could not read the answer: a ModelCallError, never anything
            # else. The answered attempt stays counted at its projection, because its cost was never read. The
            # error text is redacted before it is cut to length, so a key across the cut cannot leave a prefix.
            error = ModelCallError(
                redact(f"{model_key} ({model_id}) via {adapter.label}: the provider answered (HTTP "
                       f"{spend.answered_status}), but the answer could not be read ({type(sent_error).__name__}: "
                       f"{_clip(redact(str(sent_error), key), 300)})", key),
                status=spend.answered_status, attempts=spend.sent, raw={"error_type": type(sent_error).__name__})
        if isinstance(error, ModelCallError):
            _scrub_error(error, key)
        source = "computed"
        try:
            source = spend.settle_failure(error, lambda e: _failed_call_cost(e, adapter, entry, cfg, projected,
                                                                               est_in, max_out))
        except Exception:  # noqa: BLE001 - unsettled is the conservative outcome. The call's error matters more
            pass
        if events is not None:
            try:
                raw_err = getattr(error, "raw", None)
                attempts = getattr(error, "attempts", None)
                finish = raw_err.get("finish_reason") if isinstance(raw_err, dict) else None
                _emit(events, payload(ok=False, error=redact(f"{type(error).__name__}: {error}", key),
                                      usd=spend.counted(), usd_source=source, latency_ms=latency_ms,
                                      http_status=getattr(error, "status", None),
                                      attempts=attempts if attempts is not None else (spend.sent or None),
                                      finish_reason=_provider_text(finish, key), **spend.fields()))
            except Exception:  # noqa: BLE001 - the call's own error matters more
                pass
        if error is sent_error:
            raise sent_error
        raise error from None
    latency_ms = (time.perf_counter() - t_send) * 1000.0

    # 8. cost and settle. The provider has answered: from here on nothing escapes before the spend ledger is
    # settled and the event is written, a cost that cannot be computed is the projection (the paid answer is
    # kept), and only a RouterError (or an interrupt) is raised.
    failure: BaseException | None = None
    usd, usd_source, notes, known = float(projected), "computed", [], False
    try:
        notes = clean_token_counts(result)  # 10**400, -5 or "600" tokens is not a count: dropped, with a note
        usd, usd_source, cost_notes = settled_cost(result, entry, cfg, projected, est_in, max_out,
                                                   dropped=bool(notes))
        known = cost_known(result) or bool(notes)  # a dropped count without a usable cost settles at the projection
        notes = notes + cost_notes
    except Exception as exc:  # noqa: BLE001 - the cost is unknown: the reservation stays counted at the projection
        usd, usd_source, known = float(projected), "computed", False
        notes = notes + [f"the cost could not be computed ({type(exc).__name__}: "
                         f"{_clip(redact(str(exc), key, media=False))}), so the call stays counted at its projection"]
    except BaseException as exc:  # noqa: BLE001 - an interrupt: re-raised once the event is written
        failure = exc
    note = redact(_NOTE_SEP.join(notes), key, media=False) if notes else None
    try:
        spend.settle_answer(usd, usd_source, note, known=known and failure is None)
    except BaseException as exc:  # noqa: BLE001
        failure = failure or exc
    spent_fields = spend.fields()

    # every string from the provider, without the key in any of its forms, before the Response, event or cache
    model_name, finish_reason, generation_id, upstream = model_id, None, None, None
    try:
        model_name = _provider_text(getattr(result, "model_returned", None), key) or model_id
        finish_reason = _provider_text(getattr(result, "finish_reason", None), key)
        generation_id = _provider_text(getattr(result, "generation_id", None), key)
        upstream = _provider_text(getattr(result, "upstream_provider", None), key)
    except Exception:  # noqa: BLE001 - a field that cannot be read as text is left out
        pass

    # 9-10. JSON, the response and the cache
    text, json_error, response = "", None, None
    if failure is None:
        try:
            raw = scrub(result.raw or {}, key)
            if note:
                raw["usd_note"] = note
            if route_note:
                raw["route_note"] = route_note
            if reasoning_note:
                raw["reasoning_note"] = reasoning_note
            if price_fallback and usd_source == "computed":
                raw["price_fallback"] = price_fallback
            if budget:
                raw["reasoning_budget_projected"] = budget
            cached_raw = dict(raw)  # this call's spend figures belong to this call, never to a later cache hit
            if spent_fields["unreconciled_attempts"]:
                raw["unreconciled_attempts"] = spent_fields["unreconciled_attempts"]
                raw["unreconciled_reserved_usd"] = spent_fields["unreconciled_reserved_usd"]
            raw["ledger_usd"] = spent_fields["ledger_usd"]
            # only the key is cut, as written and JSON-escaped. The answer is otherwise kept as written
            text = strip_secret(result.text or "", key)
            json_value = strip_secret(result.json_value, key)
            if result.json_value is not None:  # an answer returned as data (jev): its text is the scrubbed value's
                text = _answer_text(json_value, text)
            # parsing decodes any other spelling of the key: it is cut from the parsed value and json_error too, and
            # an answer whose value held it is kept as json.dumps of the scrubbed value
            as_written = text
            text, value, json_error = _fresh_json(text, json_value, schema_d, key)
            if text != as_written:  # the provider body carries the answer as written, that spelling included
                ours = {k: raw[k] for k in ("unreconciled_attempts", "unreconciled_reserved_usd", "ledger_usd")
                        if k in raw}
                raw = {**_without_answer(raw), **ours}
                cached_raw = _without_answer(cached_raw)
            response = Response(
                text=text, json=value, json_error=json_error, model_key=model_key, provider=adapter.name,
                model=model_name, role=role, session=session, input_tokens=result.input_tokens,
                output_tokens=result.output_tokens, reasoning_tokens=result.reasoning_tokens,
                cached_tokens=result.cached_tokens, usd=usd, usd_source=usd_source, latency_ms=latency_ms,
                cache_hit=False, request_hash=rhash, finish_reason=finish_reason,
                reasoning_setting=reasoning_setting, json_mode=result.json_mode, image_count=len(imgs),
                est_input_tokens=est_in, projected_usd=projected, raw=raw)
            if use_cache and text.strip() and finish_reason != "error":
                record = {
                    "request_hash": rhash, "created": time.time(), "model_key": model_key, "provider": adapter.name,
                    "adapter": entry.adapter, "model": model_name, "model_requested": model_id, "text": text,
                    "json_value": json_value, "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens, "reasoning_tokens": result.reasoning_tokens,
                    "cached_tokens": result.cached_tokens, "usd": usd, "usd_source": usd_source,
                    "latency_ms": round(latency_ms, 3), "finish_reason": finish_reason,
                    "json_mode": result.json_mode, "reasoning_setting": reasoning_setting,
                    "http_status": result.http_status, "attempts": result.attempts,
                    "generation_id": generation_id, "upstream_provider": upstream,
                    "ledger_usd": spent_fields["ledger_usd"], "session": session, "raw": cached_raw,
                }
                store.put(rhash, record)
        except BaseException as exc:  # noqa: BLE001
            failure = exc

    # the model_call event, then the result or the error
    event = payload(model=model_name, input_tokens=getattr(result, "input_tokens", None),
                    output_tokens=getattr(result, "output_tokens", None),
                    reasoning_tokens=getattr(result, "reasoning_tokens", None),
                    cached_tokens=getattr(result, "cached_tokens", None),
                    usd=usd if failure is None else spend.counted(), usd_source=usd_source, latency_ms=latency_ms,
                    finish_reason=finish_reason, json_mode=getattr(result, "json_mode", None),
                    http_status=getattr(result, "http_status", None), attempts=getattr(result, "attempts", None),
                    json_error=json_error, upstream_provider=upstream, ok=failure is None,
                    error=None if failure is None else redact(f"{type(failure).__name__}: {failure}", key),
                    **spent_fields)
    if failure is None:
        try:
            _emit(events, event)
        except BaseException as exc:  # noqa: BLE001 - the event sink failed after a paid call
            failure = exc
        else:
            return response
    else:
        try:
            _emit(events, event)
        except Exception:  # noqa: BLE001 - the original error matters more
            pass
    error = _after_answer(failure, model_key, model_id, adapter.label, key)
    if error is failure:
        raise failure
    raise error from None


# --------------------------------------------------------------------------- Router
class Router:
    """Model calls for one spend session.

    ``Router(session="S3")`` makes the S3 cap from ``spend_caps_usd`` in
    config/experiment.yaml active for every call made through it, next to the
    total cap. A session name the config does not have raises UnknownSessionError
    here. ``Router()`` has no session: the FakeModel and cached answers work, and
    every paid call is refused. ``approved=True`` is for the sessions that need
    Kevin's OK (``full_judge_run``) and must be passed only after he gave it.
    ``model_caps`` maps model keys to a cap in USD inside the session, so 1 model
    cannot use up the session.

    Several Router objects, in several processes, can share a session: the spend
    ledger is 1 file with an OS lock, and each reservation is checked against
    everything already in it.

    ``events`` and ``phase`` are the defaults for every call (the run's
    LedgerWriter, and a label such as ``zoo``). ``models``, ``ledger``, ``cache``
    and ``http`` replace the model table, the spend ledger, the disk cache and
    the HTTP client. ``caps`` replaces the caps of the config and is for tests
    and dry runs only: real work takes its caps from config/experiment.yaml.
    """

    def __init__(self, session: str | None = None, *, approved: bool = False,
                 model_caps: dict[str, float] | None = None, models: ModelsConfig | None = None,
                 ledger: SpendLedger | None = None, caps: SpendCaps | None = None, cache: DiskCache | None = None,
                 http: HttpClient | None = None, events: Any = None, phase: str | None = None):
        if ledger is not None and caps is not None:
            raise ValueError("pass either ledger or caps. A SpendLedger already carries its caps")
        if session is not None and not isinstance(session, str):
            raise TypeError(f"session must be a string or None, got {type(session).__name__}")
        self.session = session
        self.approved = bool(approved)
        self._models = models
        self._ledger = ledger
        self._caps = caps
        self.cache = cache
        self.http = http
        self.events = events
        self.phase = phase
        router_events.check_sink(events)
        self.model_caps = {check_label(str(k), "model key"): _usd(v, f"the cap for model {k}")
                           for k, v in (model_caps or {}).items()}
        if session is not None:
            check_label(session, "session name")
            self.caps.session_cap(session)  # UnknownSessionError for a name the config does not have

    # ------------------------------------------------------------------ state
    @property
    def caps(self) -> SpendCaps:
        """The caps in force: the ledger's, the ones passed in, or the config's."""
        if self._ledger is not None:
            return self._ledger.caps
        return self._caps if self._caps is not None else load_spend_caps()

    @property
    def ledger(self) -> SpendLedger:
        """The spend ledger (``AVAIL_DATA/spend_ledger.jsonl`` unless one was passed in)."""
        if self._ledger is None:
            self._ledger = SpendLedger(caps=self._caps)
        return self._ledger

    @property
    def models(self) -> ModelsConfig:
        return self._models if self._models is not None else default_models()

    @property
    def cap_usd(self) -> float | None:
        """This session's cap, or None without a session."""
        return None if self.session is None else self.caps.session_cap(self.session)

    def spent(self) -> float:
        """USD the spend ledger counts for this session (0 without a session)."""
        return 0.0 if self.session is None else self.ledger.spent(self.session)

    def remaining(self) -> float:
        """USD this session can still reserve under its own cap and the total cap (0 without a session)."""
        return 0.0 if self.session is None else self.ledger.remaining(self.session)

    def summary(self) -> dict[str, Any]:
        """The spend ledger's summary: spend per session and per model, and the caps."""
        return self.ledger.summary()

    def __repr__(self) -> str:
        return f"Router(session={self.session!r}, approved={self.approved})"

    # ------------------------------------------------------------------ calls
    def chat(self, model_key: str, messages: list[dict] | str, images: list | None = None,
             json_schema: dict | None = None, role: str = "judge", max_tokens: int = DEFAULT_MAX_TOKENS,
             reasoning: str | dict | None = DEFAULT_REASONING, **kwargs: Any) -> Response:
        """:func:`chat` in this router's session, with its ledger, cache, HTTP client and event sink.

        Keyword arguments of :func:`chat` (``temperature``, ``schema_name``,
        ``use_cache``, ``cache_salt``, ``phase``, ``timeout_s``, ``extra``,
        ``events``) pass through and override the router's own defaults.
        """
        for name in ("session", "approved", "ledger"):
            if name in kwargs:
                raise TypeError(f"{name} belongs to the Router. Make another Router to change it")
        kwargs.setdefault("models", self._models)
        kwargs.setdefault("cache", self.cache)
        kwargs.setdefault("http", self.http)
        kwargs.setdefault("events", self.events)
        kwargs.setdefault("phase", self.phase)
        kwargs.setdefault("model_cap_usd", self.model_caps.get(model_key))
        return chat(model_key, messages, images=images, json_schema=json_schema, role=role, max_tokens=max_tokens,
                    reasoning=reasoning, session=self.session, approved=self.approved, ledger=self.ledger, **kwargs)

    def estimate(self, model_key: str, messages: list[dict] | str, **kwargs: Any) -> float:
        """:func:`estimate_usd` with this router's model table."""
        kwargs.setdefault("models", self._models)
        return estimate_usd(model_key, messages, **kwargs)
