"""Router tests. All offline.

HTTP goes through httpx.MockTransport. The default client is replaced by one that
fails the test if anything tries the network, sockets refuse to connect, no key is
read from the environment or .env, and AVAIL_DATA points at a temp folder, so
nothing touches the real data drive.

The first part is loopkit's router suite, ported: its tests, then 1 regression
test group per review decision (R1 to R15), per round 2 finding (G1 to G6, O1)
and per round 3 finding (H1 to H6). The last part is new for AVAIL: spend
sessions and caps from config/experiment.yaml, refusal before network I/O, the
model table, the judge call shape and the ``avail models`` command.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import io
import json
import math
import os
import pickle
import random
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
import yaml
from PIL import Image
from pydantic import ValidationError

from avail import config
from avail.router import (
    DEFAULT_MAX_TOKENS,
    DiskCache,
    FileLock,
    HttpClient,
    MissingCredentialError,
    ModelCallError,
    ModelCallPayload,
    ModelEntry,
    ModelNotConfigured,
    ModelsConfig,
    Router,
    RouterError,
    SessionNotApproved,
    SpendCapExceeded,
    SpendCaps,
    SpendLedger,
    UnknownSessionError,
    canonical_json,
    chat,
    clear_fake_script,
    estimate_usd,
    fake_calls,
    fake_script,
    from_schema,
    load_models,
    load_spend_caps,
    normalize_image,
    parse_json_text,
    set_fake_script,
)
from avail.router import core as router_core
from avail.router import http as router_http
from avail.router.adapters import get_adapter
from avail.router.adapters.base import AdapterRequest
from avail.router.adapters.elorian import build_elorian_payload, normalize_elorian_reasoning
from avail.router.adapters.jev import DEFAULT_URL as JEV_URL
from avail.router.adapters.openrouter import API_URL as OPENROUTER_URL
from avail.router.adapters.openrouter import OpenRouterAdapter, build_payload, normalize_reasoning
from avail.router.core import normalize_messages
from avail.router.jsonschema_lite import check
from avail.router.media import normalize_images

REPO = Path(config.REPO_ROOT)
# every key variable .env.example documents, plus the ones these tests use
KEY_VARS = tuple(sorted({k for k in config.dotenv_values(REPO / ".env.example") if k.endswith("_KEY")}
                        | {"OPENROUTER_API_KEY", "ELORIAN_API_KEY", "TYPESAFE_JEV_API_KEY", "TYPESAFE_API_KEY",
                           "LIQUID_API_KEY", "AV_TEST_SYS1_KEY"}))
# test keys, built at runtime so no literal looks like a credential. Neither is a real key
PATTERN_KEY = "sk-or-" + "test-" + "k" * 40  # also matches the redaction patterns
PLAIN_KEY = "av-test-" + "Q7" * 20  # matches no pattern: only exact-value redaction catches it
COUNT_SCHEMA = {"type": "object", "properties": {"plugged": {"type": "integer", "minimum": 0, "maximum": 8}},
                "required": ["plugged"], "additionalProperties": False}
# the judge answer of docs/03_evidence_sources.md
JUDGE_SCHEMA = {"type": "object",
                "properties": {"holds": {"type": "boolean"},
                               "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                               "reason": {"type": "string"}},
                "required": ["holds", "confidence", "reason"], "additionalProperties": False}
HAS_ALT_ROUTE = "alt_route" in ModelEntry.model_fields
# characters that must not sit in a source file as themselves: they are invisible or break lines in editors
LSEP, PSEP = chr(0x2028), chr(0x2029)  # the line and the paragraph separator
LQUOTE, RQUOTE = chr(0x201C), chr(0x201D)  # the smart quotes a text editor puts around a pasted key
ZWSP = chr(0x200B)  # a zero-width space
EM_DASH = chr(0x2014)
E_ACUTE = chr(0xE9)


def _entry(**kw):
    base = dict(display="test entry", adapter="openrouter", id="vendor/test-model", price_in=1.0, price_out=2.0,
                vision=True, roles=["judge", "decider", "agent", "test"], verified=True, source="test",
                accessed="2026-11-11")
    base.update(kw)
    return base


def _models() -> ModelsConfig:
    models = {
        "fake": _entry(adapter="fake", id="fake", price_in=0.0, price_out=0.0),
        "paid_fake": _entry(adapter="fake", id="fake-paid", price_in=10.0, price_out=50.0),
        "orm": _entry(id="vendor/test-model", price_in=2.0, price_out=10.0),
        "orm_hi": _entry(id="vendor/test-model-hi", price_in=2.0, price_out=10.0, reasoning={"effort": "xhigh"}),
        "orm_must": _entry(id="vendor/always-reasons", price_in=2.0, price_out=10.0, reasoning_mandatory=True,
                           reasoning_efforts=["max", "xhigh", "high", "medium", "low"]),
        "orm_no_temp": _entry(id="vendor/no-temperature", price_in=2.0, price_out=10.0, supports_temperature=False),
        "unset": _entry(id=None, price_in=None, price_out=None, verified=False),
        "unpriced": _entry(id="vendor/unpriced", price_in=None, price_out=None, verified=False),
        "unverified": _entry(id="vendor/unverified", price_in=2.0, price_out=10.0, verified=False),
        "astra": _entry(id="vendor/astra", price_in=10.0, price_out=50.0),
        "el": _entry(adapter="elorian", id="elorian-test", base_url="https://elorian.invalid/v1",
                     env_key="ELORIAN_API_KEY", price_in=0.6, price_out=3.0, reasoning={"effort": "xhigh"}),
        "el_nourl": _entry(adapter="elorian", id="elorian-test", base_url=None, price_in=0.6, price_out=3.0),
        "jev": _entry(adapter="jev", id="jev-latest", env_key="TYPESAFE_JEV_API_KEY", price_in=0.042,
                      price_out=0.0, vision=False, roles=["decider"]),
        "sys1_vision": _entry(adapter="jev", id="sys1-vision", endpoint="https://decisions.invalid/systemone",
                              env_key="AV_TEST_SYS1_KEY", price_in=0.04, price_out=0.0, vision=True,
                              roles=["decider"]),
        "free_or": _entry(id="vendor/free-tier", price_in=0.0, price_out=0.0, notes="free tier (rate limited)"),
        "orm_cache": _entry(id="vendor/cache-model", price_in=2.0, price_out=10.0, cached_in=0.2),
    }
    if HAS_ALT_ROUTE:
        models["el_alt"] = _entry(adapter="elorian", id="elorian-test", base_url="https://elorian.invalid/v1",
                                  env_key="ELORIAN_API_KEY", price_in=0.6, price_out=3.0,
                                  alt_route={"adapter": "openrouter", "id": "vendor/elorian-alt"})
    return ModelsConfig.model_validate({"version": "test", "currency": "USD", "unit": "usd_per_million_tokens",
                                        "models": models})


MODELS = _models()


def caps(total=None, outside=(), **sessions) -> SpendCaps:
    """Caps for a test ledger. The real ones come from config/experiment.yaml (see the AVAIL tests below)."""
    return SpendCaps(total_usd=total, sessions=sessions or {"T": 1.0}, outside_total=frozenset(outside))


def recap(ledger: SpendLedger, total=None, outside=(), **sessions) -> SpendLedger:
    """The same ledger file under other caps. Caps cannot be changed, so another cap is another SpendLedger."""
    return SpendLedger(ledger.path, caps(total, outside, **sessions))


def _refuse_network(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"a test tried the network: {request.method} {request.url}")


def _refuse_socket(self, address, *args, **kwargs):
    raise AssertionError(f"a test tried to open a socket to {address!r}")


def refusing_client() -> HttpClient:
    """An HTTP client whose transport fails the test if it is ever called."""
    return HttpClient(transport=httpx.MockTransport(_refuse_network), sleep=lambda s: None)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """No real data root, no keys from the environment or .env, no network, no leftover fake script."""
    monkeypatch.setenv("AVAIL_DATA", str(tmp_path / "data"))
    for name in KEY_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "_DOTENV", {})
    monkeypatch.setattr(router_http, "_DEFAULT_CLIENT", refusing_client())
    monkeypatch.setattr(socket.socket, "connect", _refuse_socket)
    clear_fake_script()
    yield
    clear_fake_script()


@pytest.fixture
def ledger(tmp_path) -> SpendLedger:
    return SpendLedger(tmp_path / "ledger" / "spend_ledger.jsonl", caps(T=1.0))


@pytest.fixture
def cache(tmp_path) -> DiskCache:
    return DiskCache(tmp_path / "cache")


@pytest.fixture
def ask(ledger, cache):
    def _ask(model_key, messages="hello world", **kw):
        kw.setdefault("models", MODELS)
        kw.setdefault("ledger", ledger)
        kw.setdefault("cache", cache)
        kw.setdefault("session", "T")
        return chat(model_key, messages, **kw)

    return _ask


class Recorder:
    """An event sink with emit(kind, payload), the shape loopkit's trace writer had."""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def emit(self, kind, payload):
        self.events.append((kind, payload))


class AppendOnly:
    """An event sink shaped like avail.ledger.LedgerWriter: append(kind, payload, ts=None) -> seq."""

    def __init__(self):
        self.rows: list[tuple[str, dict]] = []

    def append(self, kind, payload, ts=None):
        self.rows.append((kind, payload))
        return len(self.rows) - 1


class Server:
    """A canned HTTP server: each item is (status, body[, headers]), "timeout" (a read timeout), "drop" (the
    connection died after the request was sent), "connect" (refused: nothing was sent) or a callable(request)."""

    def __init__(self, *items):
        self.items = list(items)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.items.pop(0) if len(self.items) > 1 else self.items[0]
        if item == "timeout":
            raise httpx.ReadTimeout("timed out", request=request)
        if item == "drop":
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.", request=request)
        if item == "connect":
            raise httpx.ConnectError("connection refused", request=request)
        if callable(item):
            return item(request)
        status, body, *rest = item
        headers = rest[0] if rest else {}
        if isinstance(body, (dict, list)):
            return httpx.Response(status, json=body, headers=headers)
        return httpx.Response(status, content=str(body).encode("utf-8"), headers=headers)

    def client(self, sleeps: list | None = None) -> HttpClient:
        return HttpClient(transport=httpx.MockTransport(self),
                          sleep=sleeps.append if sleeps is not None else (lambda s: None))

    def bodies(self) -> list[dict]:
        return [json.loads(r.content) for r in self.requests]


def or_response(content='{"plugged": 3}', *, cost=0.00123, prompt=600, completion=3, reasoning=0, cached=0,
                gen_id="gen-test-1", model="vendor/test-model", finish="stop", **extra):
    # the default tokens cost exactly the default $0.00123 at the orm prices (2.00 / 10.00), so the router's
    # check of a provider cost against its token counts keeps it
    usage = {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion,
             "completion_tokens_details": {"reasoning_tokens": reasoning},
             "prompt_tokens_details": {"cached_tokens": cached}}
    if cost is not None:
        usage["cost"] = cost
    body = {"id": gen_id, "model": model, "provider": "TestUpstream", "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": content}}],
            "usage": usage}
    body.update(extra)
    return body


def make_request(model_key, messages="Count the plugged ports.", *, images=None, json_schema=None,
                 reasoning="high", max_tokens=300, temperature=None, extra=None) -> AdapterRequest:
    entry = MODELS.get(model_key)
    adapter = get_adapter(entry.adapter)
    msgs = normalize_messages(messages)
    imgs = normalize_images(images)
    target = max(i for i, m in enumerate(msgs) if m["role"] == "user") if imgs else None
    return AdapterRequest(model_key=model_key, model_id=entry.id, entry=entry, role="judge", messages=msgs,
                          images=imgs, image_message_index=target, json_schema=json_schema, schema_name="answer",
                          max_tokens=max_tokens, reasoning=adapter.normalize_reasoning(reasoning, entry),
                          reasoning_requested=reasoning, temperature=temperature, extra=extra or {})


def board_array() -> np.ndarray:
    arr = np.full((24, 48, 3), 40, dtype=np.uint8)
    arr[8:16, 4:12] = (240, 150, 20)
    arr[8:16, 20:28] = (18, 18, 20)
    return arr


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _reserves(ledger):
    return [r for r in ledger.records() if r["type"] == "reserve"]


def _settles(ledger):
    return [r for r in ledger.records() if r["type"] == "settle"]


# =========================================================================== ported from loopkit
# --------------------------------------------------------------------------- FakeModel and cache
def test_fake_round_trip_tokens_and_zero_cost(ask, ledger):
    r = ask("fake", "hello world", role="test")
    assert r.text == "fake response"
    assert r.input_tokens == math.ceil(len("hello world") / 4)
    assert r.output_tokens == math.ceil(len("fake response") / 4)
    assert r.usd == 0.0 and r.usd_source == "computed" and r.projected_usd == 0.0
    assert r.provider == "fake" and r.model == "fake" and r.model_key == "fake" and r.role == "test"
    assert r.cache_hit is False and r.cached is False and len(r.request_hash) == 64 and r.session == "T"
    assert r.reasoning_setting == {"enabled": False}  # the default: reasoning off
    assert r.json is None and r.json_error is None and r.image_count == 0
    with_image = ask("fake", "hello world", images=[board_array()])
    assert with_image.input_tokens == math.ceil(len("hello world") / 4) + 765
    assert with_image.image_count == 1 and with_image.est_input_tokens == math.ceil(11 / 3.5) + 1600
    kinds = [rec["type"] for rec in ledger.records()]
    assert kinds == ["reserve", "settle", "reserve", "settle"]  # the fake still goes through the spend ledger
    assert ledger.spent() == 0.0


def test_repeat_is_a_cache_hit_and_salt_is_a_miss_all_recorded(ask, ledger):
    rec = Recorder()
    r1 = ask("fake", "same request", events=rec, phase="setup")
    r2 = ask("fake", "same request", events=rec, phase="setup")
    assert r2.cache_hit and r2.usd == 0.0 and r2.usd_source == "computed"
    assert r2.text == r1.text and r2.request_hash == r1.request_hash
    assert r2.raw["original_usd"] == 0.0
    r3 = ask("fake", "same request", events=rec, cache_salt="repeat-1")
    assert not r3.cache_hit and r3.request_hash != r1.request_hash
    assert [rec_["type"] for rec_ in ledger.records()].count("reserve") == 2  # the hit wrote nothing
    assert [k for k, _ in rec.events] == ["model_call"] * 3
    payloads = [p for _, p in rec.events]
    assert [p["cache_hit"] for p in payloads] == [False, True, False]
    for p in payloads:
        ModelCallPayload.model_validate(p)
        assert set(p) == set(ModelCallPayload.model_fields)  # every field of the contract, and nothing else
    assert payloads[0]["phase"] == "setup" and payloads[0]["ok"] is True and payloads[1]["usd"] == 0.0
    assert payloads[1]["attempts"] == 0 and payloads[0]["session"] == "T" and payloads[0]["role"] == "judge"
    assert [p["original_usd"] for p in payloads] == [None, 0.0, None]  # only the hit names an earlier cost


def test_use_cache_false_neither_reads_nor_writes(ask, cache):
    r1 = ask("fake", "no cache please", use_cache=False)
    assert cache.get(r1.request_hash) is None
    ask("fake", "no cache please")
    r3 = ask("fake", "no cache please", use_cache=False)
    assert not r3.cache_hit


def test_paid_cache_hit_keeps_original_cost_in_raw(ask, ledger):
    r1 = ask("paid_fake", "price me", max_tokens=100)
    assert r1.usd > 0 and r1.usd_source == "computed"
    assert r1.usd == pytest.approx((r1.input_tokens * 10.0 + r1.output_tokens * 50.0) / 1e6)
    assert r1.usd < r1.projected_usd
    r2 = ask("paid_fake", "price me", max_tokens=100)
    assert r2.cache_hit and r2.usd == 0.0 and r2.raw["original_usd"] == pytest.approx(r1.usd)
    assert ledger.spent() == pytest.approx(r1.usd)


# --------------------------------------------------------------------------- spend ledger and caps
def test_session_cap_refuses_before_any_reservation(ask, ledger):
    tight = recap(ledger, T=0.001)
    with pytest.raises(SpendCapExceeded) as err:
        ask("paid_fake", "x" * 100, max_tokens=1000, ledger=tight)
    e = err.value
    assert e.scope == "session" and e.cap_usd == 0.001 and e.spent_usd == 0.0 and e.session == "T"
    assert e.projected_usd == pytest.approx(math.ceil(100 / 3.5) * 10.0 / 1e6 + 1000 * 50.0 / 1e6)
    assert ledger.records() == []


def test_model_cap_refuses(ask, ledger, cache):
    with pytest.raises(SpendCapExceeded) as err:
        ask("paid_fake", "hello", max_tokens=1000, model_cap_usd=0.01)
    assert err.value.scope == "model" and err.value.cap_usd == 0.01
    ask("fake", "hello")  # another model in the same session is not affected
    router = Router(session="T", ledger=ledger, cache=cache, models=MODELS, model_caps={"paid_fake": 0.01})
    with pytest.raises(SpendCapExceeded) as err:
        router.chat("paid_fake", "hello again", max_tokens=1000)
    assert err.value.scope == "model" and err.value.session == "T"
    assert router.chat("fake", "hello again").usd == 0.0
    assert _reserves(ledger) and all(r["model_key"] == "fake" for r in _reserves(ledger))
    with pytest.raises(ValueError):
        Router(session="T", ledger=ledger, model_caps={"paid_fake": -1.0})


def test_total_cap_counts_every_session(ask, tmp_path):
    led = SpendLedger(tmp_path / "two" / "spend_ledger.jsonl", caps(total=0.06, A=1.0, B=1.0))
    with fake_script(["x" * 4000]):  # 1000 output tokens at $50 per million: about $0.05 spent in session A
        first = ask("paid_fake", "first", max_tokens=1000, ledger=led, session="A")
    assert first.usd == pytest.approx((2 * 10.0 + 1000 * 50.0) / 1e6)
    with pytest.raises(SpendCapExceeded) as err:
        # projects about $0.02: fine alone and inside B's own cap, over the total with A's spend
        ask("paid_fake", "second", max_tokens=400, ledger=led, session="B")
    assert err.value.scope == "total" and err.value.spent_usd == pytest.approx(first.usd)
    assert err.value.projected_usd < 0.06 and err.value.session == "B" and err.value.cap_usd == 0.06
    summary = led.summary()
    assert summary["by_session"] == {"A": pytest.approx(first.usd)}
    assert summary["by_model"]["paid_fake"] == pytest.approx(first.usd)
    assert summary["caps"]["total_cap_usd"] == 0.06 and summary["caps"]["session_caps"] == {"A": 1.0, "B": 1.0}
    assert summary["in_total_usd"] == pytest.approx(first.usd)


def test_no_session_refuses_paid_but_allows_fake(ask, tmp_path):
    bare = SpendLedger(tmp_path / "bare" / "spend_ledger.jsonl", caps(T=1.0))
    with pytest.raises(SpendCapExceeded) as err:
        ask("paid_fake", "hello", ledger=bare, session=None)
    assert err.value.scope == "no_session" and "Router(session=" in str(err.value)
    assert err.value.projected_usd > 0 and "paid_fake" in str(err.value) and err.value.session is None
    assert bare.records() == []
    r = ask("fake", "hello", ledger=bare, session=None)
    assert r.usd == 0.0 and r.session is None
    assert [rec["session"] for rec in bare.records() if rec["type"] == "reserve"] == [None]


def test_unsettled_reservation_counts_as_spent(ask, ledger):
    rec = Recorder()
    with fake_script([RuntimeError("upstream vanished")]):
        with pytest.raises(RuntimeError, match="upstream vanished"):
            ask("paid_fake", "will fail", max_tokens=200, events=rec)
    reserve = [r for r in ledger.records() if r["type"] == "reserve"]
    assert len(reserve) == 1 and not [r for r in ledger.records() if r["type"] == "settle"]
    assert ledger.spent() == pytest.approx(reserve[0]["usd"]) and reserve[0]["usd"] > 0
    assert ledger.summary()["unsettled"]["count"] == 1
    (kind, payload), = rec.events
    assert kind == "model_call" and payload["ok"] is False and "upstream vanished" in payload["error"]
    assert payload["usd"] == pytest.approx(reserve[0]["usd"])
    ModelCallPayload.model_validate(payload)


def test_rejected_before_generation_settles_at_zero(ask, ledger, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    server = Server((400, {"error": {"code": 400, "message": "vendor/test-model is not a valid model ID"}}))
    rec = Recorder()
    with pytest.raises(ModelCallError) as err:
        ask("orm", "hi", http=server.client(), events=rec)
    assert err.value.status == 400 and err.value.rejected_before_generation and err.value.attempts == 1
    settles = [r for r in ledger.records() if r["type"] == "settle"]
    assert len(settles) == 1 and settles[0]["usd"] == 0.0
    assert ledger.spent() == 0.0
    assert rec.events[0][1]["usd"] == 0.0 and rec.events[0][1]["http_status"] == 400


def test_5xx_after_retries_stays_reserved(ask, ledger, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    sleeps: list[float] = []
    server = Server((502, {"error": {"message": "bad gateway"}}))
    with pytest.raises(ModelCallError) as err:
        ask("orm", "hi", http=server.client(sleeps))
    assert err.value.status == 502 and err.value.attempts == 3 and not err.value.rejected_before_generation
    assert len(sleeps) == 2 and len(server.requests) == 3
    reserve = [r for r in ledger.records() if r["type"] == "reserve"]
    # every attempt was reserved before it was sent, and each lost one stays counted at the projection
    assert len(reserve) == 3 and reserve[0]["usd"] > 0 and not [r for r in ledger.records() if r["type"] == "settle"]
    assert ledger.spent() == pytest.approx(3 * reserve[0]["usd"])


def test_spend_caps_are_validated_and_cannot_be_changed():
    c = SpendCaps(total_usd=21, sessions={"S1": 1, "S4": 5.0})
    assert c.total_usd == 21.0 and dict(c.sessions) == {"S1": 1.0, "S4": 5.0} and c.known() == ["S1", "S4"]
    assert c.session_cap("S4") == 5.0 and not c.needs_approval("S4")
    with pytest.raises(TypeError):
        c.sessions["S1"] = 100.0  # no code path raises a cap at run time
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.total_usd = 100.0
    for bad in (-1.0, float("nan"), float("inf"), float("-inf"), True, "1.0", None, [1.0]):
        with pytest.raises(RouterError):
            SpendCaps(total_usd=None, sessions={"S1": bad})
    for bad_total in (-1.0, float("nan"), float("inf"), True, "15"):
        with pytest.raises(RouterError):
            SpendCaps(total_usd=bad_total, sessions={"S1": 1.0})
    with pytest.raises(RouterError):
        SpendCaps(total_usd=None, sessions={"S1": 1.0}, outside_total=frozenset({"S9"}))
    with pytest.raises(RouterError):
        SpendCaps(total_usd=None, sessions=[("S1", 1.0)])
    assert pickle.loads(pickle.dumps(c)) == c
    same = SpendCaps(total_usd=21.0, sessions={"S4": 5, "S1": 1.0})
    assert same == c and hash(same) == hash(c) and {c: "found"}[same] == "found"
    assert c != SpendCaps(total_usd=21.0, sessions={"S1": 1.0, "S4": 5.5})
    assert c.as_dict() == {"total_cap_usd": 21.0, "session_caps": {"S1": 1.0, "S4": 5.0}, "outside_total": [],
                           "source": "caps passed in code"}


def test_ledger_tolerates_a_truncated_last_line(ledger):
    rid = ledger.reserve(0.25, "orm", "a" * 64, session="T")
    with open(ledger.path, "ab") as f:
        f.write(b'{"type": "reserve", "id": "dead", "usd": 9')  # a crash mid-write
    assert ledger.spent() == pytest.approx(0.25)
    ledger.settle(rid, 0.10, "provider")
    assert ledger.spent() == pytest.approx(0.10)
    assert [r["type"] for r in ledger.records()] == ["reserve", "settle"]
    summary = ledger.summary()
    assert summary["total_usd"] == pytest.approx(0.10) and summary["unsettled"]["count"] == 0
    assert summary["by_session_model"] == {"T": {"orm": pytest.approx(0.10)}}
    assert summary["corrupt_lines"] == 1


def test_ledger_lock_is_an_os_lock_on_a_persistent_file(ledger):
    # R5: a lock file left on disk (by any earlier process, however old) blocks nobody. Only a held OS lock does
    ledger.lock_path.parent.mkdir(parents=True, exist_ok=True)
    ledger.lock_path.write_text("12345 0\n", encoding="ascii")
    old = time.time() - 3600
    os.utime(ledger.lock_path, (old, old))
    ledger.reserve(0.0, "fake", "b" * 64, session="T")
    assert ledger.lock_path.exists()  # never deleted, so no holder can lose its lock to someone's release()
    holder = FileLock(ledger.lock_path)
    holder.acquire()
    try:
        old = time.time() - 3600  # however old the holder's lock looks, a waiter never breaks it
        os.utime(ledger.lock_path, (old, old))
        impatient = SpendLedger(ledger.path, ledger.caps, lock_timeout_s=0.2)
        with pytest.raises(RouterError, match="lock"):
            impatient.reserve(0.0, "fake", "c" * 64, session="T")
        assert ledger.lock_path.exists() and holder.held
    finally:
        holder.release()
    impatient.reserve(0.0, "fake", "c" * 64, session="T")
    assert [r["request_hash"][0] for r in ledger.records()] == ["b", "c"]


def test_ledger_lock_is_released_when_its_holder_dies(ledger):
    code = ("import os, sys\nfrom avail.router.spend import FileLock\nFileLock(sys.argv[1]).acquire()\n"
            "print('held', flush=True)\nos._exit(3)\n")  # dies holding the lock: no release, no cleanup
    proc = subprocess.run([sys.executable, "-c", code, str(ledger.lock_path)], capture_output=True, text=True,
                          timeout=120, cwd=REPO)
    assert proc.returncode == 3 and proc.stdout.strip() == "held", proc.stderr[-500:]
    t0 = time.monotonic()
    SpendLedger(ledger.path, ledger.caps, lock_timeout_s=5.0).reserve(0.0, "fake", "d" * 64, session="T")
    assert time.monotonic() - t0 < 5.0 and ledger.lock_path.exists()


def test_ledger_lock_excludes_threads_of_one_process(tmp_path):
    path = tmp_path / "locks" / "x.lock"
    state = {"inside": 0, "overlaps": 0, "entries": 0}

    def work():
        for _ in range(25):
            with FileLock(path, timeout_s=30):
                state["inside"] += 1
                state["entries"] += 1
                if state["inside"] > 1:
                    state["overlaps"] += 1
                time.sleep(0.001)
                state["inside"] -= 1

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert state == {"inside": 0, "overlaps": 0, "entries": 100}
    lock = FileLock(path)
    lock.acquire()
    with pytest.raises(RouterError, match="already held"):
        lock.acquire()
    lock.release()
    lock.release()  # a second release is harmless


# --------------------------------------------------------------------------- configuration errors
def test_missing_key_names_openrouter_variable_before_any_spend(ask, ledger):
    with pytest.raises(MissingCredentialError) as err:
        ask("orm", "hi")
    assert "OPENROUTER_API_KEY" in str(err.value)
    assert ledger.records() == []
    with pytest.raises(MissingCredentialError) as err:
        ask("jev", "state", extra={"questions": {"q": {"type": "noul", "instructions": "x"}}})
    assert "TYPESAFE_JEV_API_KEY" in str(err.value)


def test_model_without_id_is_not_configured(ask, ledger):
    with pytest.raises(ModelNotConfigured) as err:
        ask("unset", "hi")
    assert "unset" in str(err.value) and "unverified" in str(err.value)
    with pytest.raises(ModelNotConfigured, match="base_url"):
        ask("el_nourl", "hi")
    with pytest.raises(KeyError, match="unknown model key"):
        ask("no_such_model", "hi")
    assert ledger.records() == []


def test_bad_arguments_fail_before_spend(ask, ledger, monkeypatch):
    with pytest.raises(ValueError):
        ask("fake", "hi", role="critic")
    for old_role in ("writer", "reader", "system1"):  # loopkit's roles are not AVAIL's
        with pytest.raises(ValueError, match="judge, decider, agent, test"):
            ask("fake", "hi", role=old_role)
    with pytest.raises(ValueError):
        ask("fake", [{"role": "robot", "content": "hi"}])
    with pytest.raises(ValueError):
        ask("fake", "hi", reasoning="maximum")
    for bad_tokens in (0, -5, 2.5, True, "300"):
        with pytest.raises(ValueError):
            ask("fake", "hi", max_tokens=bad_tokens)
    monkeypatch.setenv("TYPESAFE_JEV_API_KEY", PLAIN_KEY)
    with pytest.raises(ValueError):  # the jev adapter ignores reasoning, but a bad value still fails
        ask("jev", "state", reasoning="maximum", extra={"questions": {"q": {"type": "noul"}}})
    with pytest.raises(RouterError, match="text-only"):
        ask("jev", "state", images=[board_array()], extra={"questions": {"q": {"type": "noul"}}})
    bad_url = _models()
    bad_url.models["el"].base_url = "elorian.invalid/v1"
    with pytest.raises(ModelNotConfigured, match="http"):
        ask("el", "hi", models=bad_url)
    assert ledger.records() == []


def test_errors_keep_their_fields_through_pickling():
    cap = pickle.loads(pickle.dumps(SpendCapExceeded("over", scope="model", cap_usd=2.0, spent_usd=1.5,
                                                     projected_usd=0.7, session="S3")))
    assert (cap.scope, cap.cap_usd, cap.spent_usd, cap.projected_usd, cap.session, str(cap)) == (
        "model", 2.0, 1.5, 0.7, "S3", "over")
    call = pickle.loads(pickle.dumps(ModelCallError("failed", status=402, attempts=1,
                                                    rejected_before_generation=True)))
    assert (call.status, call.attempts, call.rejected_before_generation) == (402, 1, True)
    assert isinstance(call, RouterError) and isinstance(call, RuntimeError)
    for kind in (UnknownSessionError, SessionNotApproved, ModelNotConfigured):
        again = pickle.loads(pickle.dumps(kind("no such session")))
        assert type(again) is kind and str(again) == "no such session" and isinstance(again, RouterError)


# --------------------------------------------------------------------------- OpenRouter adapter
def test_openrouter_reasoning_mapping():
    plain, hi = MODELS.get("orm"), MODELS.get("orm_hi")
    assert normalize_reasoning("disabled", plain) == {"enabled": False}
    assert normalize_reasoning(None, plain) == {"enabled": False}
    assert normalize_reasoning(False, plain) == {"enabled": False}
    assert normalize_reasoning("low", plain) == {"effort": "low"}
    assert normalize_reasoning("high", plain) == {"effort": "high"}
    assert normalize_reasoning("highest", plain) == {"effort": "high"}
    assert normalize_reasoning("highest", hi) == {"effort": "xhigh"}
    assert normalize_reasoning({"max_tokens": 2000}, plain) == {"max_tokens": 2000}
    with pytest.raises(ValueError):
        normalize_reasoning("extreme", plain)


def test_openrouter_payload_shape():
    arr = board_array()
    req = make_request("orm", [{"role": "system", "content": "You read boards."},
                               {"role": "user", "content": "Count the plugged ports."}],
                       images=[arr], json_schema=COUNT_SCHEMA, reasoning="disabled")
    body = build_payload(req, "json_schema")
    assert body["model"] == "vendor/test-model" and body["max_tokens"] == 300
    assert body["usage"] == {"include": True}
    assert body["reasoning"] == {"enabled": False}
    assert "temperature" not in body
    assert body["messages"][0] == {"role": "system", "content": "You read boards."}
    parts = body["messages"][1]["content"]
    assert parts[0] == {"type": "text", "text": "Count the plugged ports."}
    assert parts[1]["type"] == "image_url"
    url = parts[1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    decoded = base64.b64decode(url.split(",", 1)[1])
    assert decoded.startswith(b"\x89PNG") and np.array_equal(np.asarray(Image.open(io.BytesIO(decoded))), arr)
    assert body["response_format"] == {"type": "json_schema", "json_schema": {
        "name": "answer", "strict": False, "schema": COUNT_SCHEMA}}
    assert body["provider"] == {"require_parameters": True}

    obj = build_payload(req, "json_object_schema_in_prompt")
    assert obj["response_format"] == {"type": "json_object"}
    assert obj["messages"][1]["content"][-1]["text"].startswith("Return only JSON that matches this JSON Schema")
    plain = build_payload(req, "plain_schema_in_prompt")
    assert "response_format" not in plain and "provider" not in plain
    assert "JSON Schema" in plain["messages"][1]["content"][-1]["text"]
    no_schema = build_payload(make_request("orm", "hi", temperature=0.2, reasoning="highest"), None)
    assert no_schema["temperature"] == 0.2 and no_schema["reasoning"] == {"effort": "high"}
    assert "response_format" not in no_schema and no_schema["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert OpenRouterAdapter().headers() == {"X-Title": "avail"}


def test_openrouter_send_with_injected_transport_reports_provider_cost(ask, ledger, monkeypatch):
    server = Server((200, or_response(reasoning=3, cached=200)))
    req = make_request("orm", images=[board_array()], json_schema=COUNT_SCHEMA, reasoning="disabled")
    result = OpenRouterAdapter().send(req, PLAIN_KEY, server.client())
    assert result.usd == 0.00123 and result.text == '{"plugged": 3}' and result.json_mode == "json_schema"
    assert (result.input_tokens, result.output_tokens, result.reasoning_tokens, result.cached_tokens) == (
        600, 3, 3, 200)
    assert result.generation_id == "gen-test-1" and result.upstream_provider == "TestUpstream"
    sent = server.requests[0]
    assert str(sent.url) == OPENROUTER_URL
    assert sent.headers["authorization"] == "Bearer " + PLAIN_KEY and sent.headers["x-title"] == "avail"
    assert PLAIN_KEY.encode() not in sent.content

    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    server2 = Server((200, or_response()))
    r = ask("orm", "Count the plugged ports.", images=[board_array()], json_schema=COUNT_SCHEMA,
            reasoning="disabled", role="judge", http=server2.client())
    assert r.usd == 0.00123 and r.usd_source == "provider" and r.json == {"plugged": 3} and r.json_error is None
    assert r.reasoning_setting == {"enabled": False} and r.json_mode == "json_schema" and r.model == "vendor/test-model"
    settle = [x for x in ledger.records() if x["type"] == "settle"][0]
    assert settle["usd"] == 0.00123 and settle["usd_source"] == "provider"


def test_openrouter_json_mode_falls_back(ask, ledger, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    refuse = (400, {"error": {"message": "This model does not support response_format json_schema"}})
    server = Server(refuse, (200, or_response(content='{"plugged": 4}')))
    r = ask("orm", "count", json_schema=COUNT_SCHEMA, http=server.client())
    assert r.json_mode == "json_object_schema_in_prompt" and r.json == {"plugged": 4}
    first, second = server.bodies()
    assert first["response_format"]["type"] == "json_schema"
    assert second["response_format"] == {"type": "json_object"}
    assert "JSON Schema" in second["messages"][-1]["content"][-1]["text"]
    assert len(r.raw["modes_tried"]) == 1
    server3 = Server(refuse, (404, {"error": {"message": "No endpoints found that can handle the requested "
                                                         "parameters"}}), (200, or_response()))
    r3 = ask("orm", "count again", json_schema=COUNT_SCHEMA, http=server3.client())
    assert r3.json_mode == "plain_schema_in_prompt" and "response_format" not in server3.bodies()[2]
    # R1: every POST in every mode was reserved before it was sent, and the refused modes settled at $0
    assert len([x for x in ledger.records() if x["type"] == "reserve"]) == 5
    assert [x["usd"] for x in ledger.records() if x["type"] == "settle"] == [0.0, 0.00123, 0.0, 0.0, 0.00123]
    assert ledger.spent() == pytest.approx(2 * 0.00123)


def test_a_missing_price_projects_at_astra_prices(ask, ledger, monkeypatch):
    # loopkit called an unverified entry without prices here. AVAIL refuses unverified entries (see
    # test_an_unverified_entry_makes_no_paid_call), so the fallback is reached with a price removed after validation
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    models = _models()
    models.models["orm"].price_in = None  # past validation, as a hand-built entry could be
    models.models["orm"].price_out = None
    server = Server((200, or_response(cost=None, prompt=100, completion=10)))
    r = ask("orm", "x" * 35, max_tokens=1000, models=models, http=server.client())
    assert r.est_input_tokens == 10
    assert r.projected_usd == pytest.approx(10 * 10.0 / 1e6 + 1000 * 50.0 / 1e6)
    assert r.usd_source == "computed" and r.usd == pytest.approx((100 * 10.0 + 10 * 50.0) / 1e6)
    assert r.raw["price_fallback"] == "astra" and "Astra prices" in r.raw["usd_note"]


def test_200_without_usage_stays_reserved_at_projection(ask, ledger, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    body = or_response()
    del body["usage"]
    r = ask("orm", "hi", http=Server((200, body)).client())
    assert r.usd == pytest.approx(r.projected_usd) and r.usd_source == "computed" and "usd_note" in r.raw
    assert ledger.spent() == pytest.approx(r.projected_usd)
    assert ledger.summary()["unsettled"] == {"count": 1, "usd": pytest.approx(r.projected_usd)}


# --------------------------------------------------------------------------- HTTP retries
def test_retry_on_429_then_success_without_sleeping():
    sleeps: list[float] = []
    server = Server((429, {"error": {"message": "rate limited"}}, {"Retry-After": "2"}), (200, {"ok": True}))
    res = server.client(sleeps).post_json("https://api.invalid/x", {"a": 1}, key=PLAIN_KEY, timeout_s=5)
    assert res.ok and res.status == 200 and res.data == {"ok": True} and res.attempts == 2
    assert sleeps == [2.0]
    assert [a["status"] for a in res.attempt_log] == [429, 200]


def test_retry_policy_backoff_caps_and_no_retry_on_400():
    sleeps: list[float] = []
    server = Server((503, "busy"), (520, "edge"), (200, {"ok": 1}))
    res = server.client(sleeps).post_json("https://api.invalid/x", {}, timeout_s=5)
    assert res.ok and res.attempts == 3 and len(sleeps) == 2
    assert 1.5 * 0.7 <= sleeps[0] <= 1.5 * 1.3 and 3.0 * 0.7 <= sleeps[1] <= 3.0 * 1.3
    sleeps.clear()
    res = Server((429, "", {"Retry-After": "120"}), (200, {})).client(sleeps).post_json("https://a.invalid", {})
    assert sleeps == [20.0]
    sleeps.clear()
    res = Server((400, {"error": {"message": "bad"}})).client(sleeps).post_json("https://a.invalid", {})
    assert res.status == 400 and res.attempts == 1 and sleeps == []
    res = Server("timeout", (200, {"ok": 2})).client(sleeps).post_json("https://a.invalid", {})
    assert res.ok and res.attempts == 2 and len(sleeps) == 1
    res = Server("connect").client().post_json("https://a.invalid", {})
    assert res.status is None and res.attempts == 3 and "connection" in res.error


def test_per_call_deadline_covers_a_trickling_body():
    def trickle(request):
        def chunks():
            for _ in range(5):
                time.sleep(0.03)
                yield b" "
            yield b"{}"
        return httpx.Response(200, content=chunks())

    res = Server(trickle).client().post_json("https://a.invalid", {}, timeout_s=0.05)
    assert res.status is None and "timeout" in res.error and res.attempts == 3


# --------------------------------------------------------------------------- secrets
def test_key_never_in_errors_events_cache_or_ledger(ask, ledger, cache, monkeypatch):
    for key in (PLAIN_KEY, PATTERN_KEY):
        monkeypatch.setenv("OPENROUTER_API_KEY", key)
        echo = {"error": {"message": f"key {key} refused, request was data:image/png;base64,{'A' * 300}"}}
        rec = Recorder()
        with pytest.raises(ModelCallError) as err:
            ask("orm", "hi", images=[board_array()], http=Server((401, echo)).client(), events=rec)
        dumped = json.dumps(err.value.raw)
        for text in (str(err.value), repr(err.value), dumped, json.dumps(rec.events)):
            assert key not in text
            assert "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" not in text
        assert err.value.rejected_before_generation

        leaky = or_response(echo_header=f"Bearer {key}", note=f"served for {key}")
        r = ask("orm", f"hello {len(key)}", http=Server((200, leaky)).client(), events=rec)
        record_text = cache.path_for(r.request_hash).read_text(encoding="utf-8")
        assert key not in record_text and key not in json.dumps(r.raw) and key not in json.dumps(rec.events)
        assert key not in ledger.path.read_text(encoding="utf-8")
        assert r.text == '{"plugged": 3}'

        # an answer in plain prose that repeats the key: only the key is cut, the rest is kept as written
        prose = or_response(content=f"I will not judge that. The key I was given is {key}.")
        r = ask("orm", f"prose {len(key)}", http=Server((200, prose)).client(), events=rec)
        assert r.text == "I will not judge that. The key I was given is [redacted]."
        cached = json.loads(cache.path_for(r.request_hash).read_text(encoding="utf-8"))
        assert cached["text"] == r.text and key not in json.dumps(cached) and key not in json.dumps(rec.events)
        assert ask("orm", f"prose {len(key)}").text == r.text  # and the cache hit gives the same text back


def test_an_answer_whose_finish_reason_is_error_is_never_cached_whatever_the_adapter(ask, cache, monkeypatch):
    from avail.router.adapters.fake import FakeAdapter

    real_send = FakeAdapter.send

    def failing(self, request, key, http):  # an adapter that returns a failed generation instead of raising
        result = real_send(self, request, key, http)
        result.finish_reason = "error"
        return result

    monkeypatch.setattr(FakeAdapter, "send", failing)
    r = ask("fake", "a failed generation")
    assert r.finish_reason == "error" and not list(cache.root.rglob("*.json"))
    assert not ask("fake", "a failed generation").cached
    monkeypatch.setattr(FakeAdapter, "send", real_send)
    assert not ask("fake", "a failed generation").cached and ask("fake", "a failed generation").cached


def test_cache_hit_needs_no_key(ask, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    r1 = ask("orm", "cache me", http=Server((200, or_response())).client())
    monkeypatch.delenv("OPENROUTER_API_KEY")
    r2 = ask("orm", "cache me")
    assert r2.cache_hit and r2.text == r1.text and r2.usd == 0.0 and r2.raw["original_usd"] == 0.00123


# --------------------------------------------------------------------------- JSON answers
def test_json_parsing_strips_fences_and_prose():
    assert parse_json_text('```json\n{"a": 1}\n```') == ({"a": 1}, None)
    assert parse_json_text("```\n[1, 2]\n```") == ([1, 2], None)
    assert parse_json_text('Here is the answer:\n{"a": {"b": [1]}}\nHope that helps.') == ({"a": {"b": [1]}}, None)
    assert parse_json_text('```json\n{"a": 1}') == ({"a": 1}, None)  # unclosed fence (truncated)
    assert parse_json_text("I see 3 [orange] plugs: {\"plugged\": 3}") == ({"plugged": 3}, None)
    value, err = parse_json_text("no json here")
    assert value is None and err.startswith("not JSON")
    assert parse_json_text("") == (None, "empty answer")


def test_json_schema_violation_reported_not_repaired(ask):
    with fake_script(['```json\n{"plugged": "three"}\n```', 'Sure! {"plugged": 3}', "nothing useful",
                      '{"plugged": 9, "extra": 1}']):
        bad = ask("fake", "one", json_schema=COUNT_SCHEMA)
        good = ask("fake", "two", json_schema=COUNT_SCHEMA)
        junk = ask("fake", "three", json_schema=COUNT_SCHEMA)
        two = ask("fake", "four", json_schema=COUNT_SCHEMA)
    assert bad.json == {"plugged": "three"} and "plugged" in bad.json_error and "integer" in bad.json_error
    assert good.json == {"plugged": 3} and good.json_error is None and good.json_ok
    assert junk.json is None and junk.json_error.startswith("not JSON") and not junk.json_ok
    assert "above the maximum 8" in two.json_error and "unknown field 'extra'" in two.json_error
    assert ";" not in two.json_error  # errors are joined into 1 line without semicolons


# --------------------------------------------------------------------------- typed-decision models and elorian
def test_jev_payload_and_cost_from_input_tokens(ask, ledger, monkeypatch):
    monkeypatch.setenv("TYPESAFE_JEV_API_KEY", PLAIN_KEY)
    questions = {"plugged": {"type": "choice", "instructions": "How many ports hold a plug?",
                             "criteria": {str(n): f"{n} plugged" for n in range(9)}}}
    answers = {"plugged": {"type": "choice", "choice": "3", "probabilities": {"3": 0.9, "4": 0.1},
                           "confidence": 0.8}}
    server = Server((200, {"id": "sys1-req-1", "model": "jev-latest", "answers": answers,
                           "usage": {"input_tokens": 1000, "output_tokens": 3}}))
    r = ask("jev", [{"role": "system", "content": "ignored"}, {"role": "user", "content": "board: 8 ports"}],
            role="decider", extra={"questions": questions}, http=server.client())
    assert str(server.requests[0].url) == JEV_URL
    assert server.bodies()[0] == {"model": "jev-latest", "state": "board: 8 ports", "questions": questions}
    assert server.requests[0].headers["authorization"] == "Bearer " + PLAIN_KEY
    assert r.usd == pytest.approx(1000 * 0.042 / 1e6) and r.usd_source == "computed"
    assert r.text == json.dumps(answers) and r.json == answers and r.json_error is None
    assert r.reasoning_setting is None and r.provider == "jev" and r.json_mode == "typed_questions"
    assert r.role == "decider"


def test_jev_needs_typed_questions_before_spend(ask, ledger, monkeypatch):
    monkeypatch.setenv("TYPESAFE_JEV_API_KEY", PLAIN_KEY)
    with pytest.raises(RouterError, match="typed questions"):
        ask("jev", "board state")
    assert ledger.records() == []


def test_jev_fallback_variable_and_typed_decision_images(ask, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", PLAIN_KEY)  # the fallback name
    ok = {"answers": {"q": {"type": "noul", "noul": 0.7}}, "usage": {"input_tokens": 10, "output_tokens": 1}}
    r = ask("jev", "state", extra={"questions": {"q": {"type": "noul", "instructions": "x"}}},
            http=Server((200, ok)).client())
    assert r.json == ok["answers"]
    monkeypatch.setenv("AV_TEST_SYS1_KEY", PLAIN_KEY)
    server = Server((200, ok))
    ask("sys1_vision", "board in the image", images=[board_array()],
        extra={"questions": {"q": {"type": "noul", "instructions": "x"}}}, http=server.client())
    body = server.bodies()[0]
    assert str(server.requests[0].url) == "https://decisions.invalid/systemone"
    assert body["model"] == "sys1-vision" and body["images"][0].startswith("data:image/png;base64,")
    assert body["state"] == "board in the image"  # plain text, never nested inside another JSON string


def test_elorian_payload_and_computed_cost(ask, ledger, monkeypatch):
    entry = MODELS.get("el")
    assert normalize_elorian_reasoning("disabled", entry) == "low"
    assert normalize_elorian_reasoning("medium", entry) == "medium"
    assert normalize_elorian_reasoning("high", entry) == "xhigh"
    assert normalize_elorian_reasoning("highest", entry) == "xhigh"
    assert normalize_elorian_reasoning({"reasoning_effort": "low"}, entry) == "low"
    body = build_elorian_payload(make_request("el", images=[board_array()], json_schema=COUNT_SCHEMA), "json_schema")
    assert "usage" not in body and "provider" not in body and "max_tokens" not in body
    assert body["max_completion_tokens"] == 300 and body["reasoning_effort"] == "xhigh"
    assert body["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert body["response_format"]["json_schema"]["strict"] is False

    monkeypatch.setenv("ELORIAN_API_KEY", PLAIN_KEY)
    server = Server((200, or_response(cost=None, prompt=2000, completion=100, model="elorian-test")))
    r = ask("el", "count", json_schema=COUNT_SCHEMA, reasoning="disabled", http=server.client())
    assert str(server.requests[0].url) == "https://elorian.invalid/v1/chat/completions"
    assert r.usd_source == "computed" and r.usd == pytest.approx((2000 * 0.6 + 100 * 3.0) / 1e6)
    assert r.reasoning_setting == "low" and server.bodies()[0]["reasoning_effort"] == "low"
    assert "no off switch" in r.raw["reasoning_note"]  # disabled could not be sent as asked, and the answer says so


@pytest.mark.skipif(not HAS_ALT_ROUTE, reason="the model table has no alt_route field")
def test_alt_route_used_only_when_the_primary_key_is_missing(ask, monkeypatch):
    rec = Recorder()
    with pytest.raises(MissingCredentialError) as err:
        ask("el_alt", "hi")
    assert "ELORIAN_API_KEY" in str(err.value) and "OPENROUTER_API_KEY" in str(err.value)
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    server = Server((200, or_response(model="vendor/elorian-alt")))
    r = ask("el_alt", "hi", http=server.client(), events=rec)
    assert str(server.requests[0].url) == OPENROUTER_URL and server.bodies()[0]["model"] == "vendor/elorian-alt"
    assert r.provider == "openrouter" and r.usd_source == "provider" and "route_note" in r.raw
    assert rec.events[0][1]["route"] == "alt_route" and rec.events[0][1]["model_key"] == "el_alt"
    monkeypatch.setenv("ELORIAN_API_KEY", PLAIN_KEY)
    server2 = Server((200, or_response(cost=None, model="elorian-test")))
    r2 = ask("el_alt", "hi", http=server2.client())
    assert str(server2.requests[0].url).startswith("https://elorian.invalid/") and r2.provider == "elorian"


def test_has_alt_route_field():
    assert HAS_ALT_ROUTE  # the test above must never be skipped: glimpse relies on its alt_route


# --------------------------------------------------------------------------- images, cache files, schemas, scripts
def test_images_normalize_and_hash(tmp_path):
    arr = board_array()
    a = normalize_image(arr)
    assert a.mime == "image/png" and a.data.startswith(b"\x89PNG") and normalize_image(arr.copy()).sha256 == a.sha256
    assert a.sha256 == hashlib.sha256(a.data).hexdigest()  # the hash is over the bytes that are sent
    assert normalize_image(Image.fromarray(arr)).sha256 == a.sha256
    raw_png = png_bytes(arr)
    assert normalize_image(raw_png).data == raw_png
    jpeg = io.BytesIO()
    Image.fromarray(arr).save(jpeg, format="JPEG")
    j = normalize_image(jpeg.getvalue())
    assert j.mime == "image/jpeg" and j.data == jpeg.getvalue()
    path = tmp_path / "board.png"
    path.write_bytes(raw_png)
    assert normalize_image(path).sha256 == normalize_image(str(path)).sha256
    with pytest.raises(ValueError):
        normalize_image(arr.astype(np.float32))
    with pytest.raises(ValueError):
        normalize_image(tmp_path / "missing.png")
    with pytest.raises(TypeError):
        normalize_images(arr)  # one array is not a list of images
    assert "bytes=" in repr(a) and "base64" not in repr(a)


def test_image_bytes_change_the_request_hash(ask):
    other = board_array()
    other[0, 0] = (1, 2, 3)
    r1 = ask("fake", "look", images=[board_array()])
    r2 = ask("fake", "look", images=[other])
    r3 = ask("fake", "look", images=[board_array()])
    assert r1.request_hash != r2.request_hash and r3.cache_hit


def test_disk_cache_layout_corruption_and_atomic_write(tmp_path):
    cache = DiskCache(tmp_path / "c")
    h = "ab" + "0" * 62
    path = cache.put(h, {"text": "hello", "usd": 0.5})
    assert path == tmp_path / "c" / "ab" / f"{h}.json"
    assert cache.get(h)["text"] == "hello" and h in cache
    assert not list((tmp_path / "c" / "ab").glob("*.tmp"))
    path.write_text("{ not json", encoding="utf-8")
    assert cache.get(h) is None
    path.write_text('["a list"]', encoding="utf-8")
    assert cache.get(h) is None
    assert cache.get("cd" + "1" * 62) is None
    with pytest.raises(ValueError):
        cache.get("../../escape")
    with pytest.warns(RuntimeWarning):
        assert cache.put("ef" + "2" * 62, {"text": "x", "raw": {"score": float("nan")}}) is None


def test_default_cache_and_ledger_live_under_the_data_root(tmp_path):
    r = chat("fake", "where do I go", models=MODELS)
    data = tmp_path / "data"
    assert (data / "cache" / "router" / r.request_hash[:2] / f"{r.request_hash}.json").is_file()
    assert (data / "spend_ledger.jsonl").is_file()
    r2 = Router(models=MODELS).chat("fake", "where do I go")  # a Router with no arguments uses the same places
    assert r2.cached and len((data / "spend_ledger.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_from_schema_gives_valid_instances():
    for schema in (
        JUDGE_SCHEMA,
        COUNT_SCHEMA,
        {"type": "object", "properties": {"n": {"type": "integer", "minimum": 3, "maximum": 9},
                                          "x": {"type": "number", "exclusiveMinimum": 1.0},
                                          "s": {"type": "string", "minLength": 6},
                                          "port": {"enum": ["seated", "short"]},
                                          "opt": {"anyOf": [{"type": "null"}, {"type": "boolean"}]},
                                          "items": {"type": "array", "items": {"type": "integer"}, "minItems": 2}}},
        {"$defs": {"P": {"type": "object", "properties": {"k": {"const": 7}}, "required": ["k"]}},
         "type": "array", "items": {"$ref": "#/$defs/P"}},
        {"$defs": {"Verdict": {"type": "object",
                               "properties": {"decision": {"enum": ["reject", "abstain"]},
                                              "score": {"type": "number", "minimum": 0.25, "maximum": 0.75}},
                               "required": ["decision", "score"]}},
         "type": "object", "properties": {"verdict": {"$ref": "#/$defs/Verdict"}}, "required": ["verdict"]},
    ):
        value = from_schema(schema)
        assert check(value, schema) == [], (schema, value)
    assert from_schema(COUNT_SCHEMA) == {"plugged": 0}
    assert from_schema(JUDGE_SCHEMA) == {"holds": True, "confidence": 0.5, "reason": "fake"}
    assert check({"holds": "yes", "confidence": 2}, JUDGE_SCHEMA) != []  # the checker can fail


def test_fake_script_list_callable_and_exhaustion(ask):
    seen = []

    def script(req):
        seen.append(req)
        return {"echo": req.text, "images": req.image_count}

    set_fake_script(script)
    r = ask("fake", "first", images=[board_array()], json_schema={"type": "object"}, extra={"k": 1}, max_tokens=99)
    assert json.loads(r.text) == {"echo": "first", "images": 1}
    req = seen[0]
    assert req.call_index == 0 and req.model_key == "fake" and req.max_tokens == 99 and req.extra == {"k": 1}
    assert len(req.images) == 1 and len(req.images[0]) == 64 and req.json_schema == {"type": "object"}
    set_fake_script(["one", {"two": 2}])
    assert ask("fake", "a").text == "one"
    assert ask("fake", "b").text == '{"two": 2}'
    assert fake_calls() == 2
    with pytest.raises(RouterError, match="exhausted"):
        ask("fake", "c")
    clear_fake_script()
    assert ask("fake", "d").text == "fake response"


def test_fake_script_scopes_the_cache(ask):
    with fake_script(["from script A"]):
        assert ask("fake", "identical").text == "from script A"
        assert ask("fake", "identical").cache_hit  # same script: a hit
    with fake_script(["from script B"]):
        r = ask("fake", "identical")
        assert r.text == "from script B" and not r.cache_hit  # another script never sees A's reply
    with fake_script(["outer"]):
        with fake_script(["inner"]):
            assert ask("fake", "nested").text == "inner"
        assert ask("fake", "nested 2").text == "outer"


# --------------------------------------------------------------------------- live smoke, offline
def test_smoke_runs_offline_and_charges_its_session(tmp_path, monkeypatch):
    from avail.router import smoke

    models = _models()
    del models.models["paid_fake"]  # keep the first run free, so it needs no spend session
    led = SpendLedger(tmp_path / "smoke" / "spend_ledger.jsonl", caps(total=15.0, S1=1.0))
    out = smoke.run_smoke(include_fake=True, models=models, router=Router(models=models, ledger=led),
                          out_dir=tmp_path / "s1", echo=False)
    assert [r["model_key"] for r in out["rows"]] == ["fake"]
    assert out["rows"][0]["ok"] and out["session"] is None and out["spend_usd_this_run"] == 0.0
    reasons = {s["model_key"]: s["reason"] for s in out["skipped"]}
    assert "OPENROUTER_API_KEY" in reasons["orm"] and "verified: false" in reasons["unset"]
    assert "no paid call" in reasons["unverified"]
    assert "base_url" in reasons["el_nourl"] and "AV_TEST_SYS1_KEY" in reasons["sys1_vision"]
    assert (tmp_path / "s1" / "board.png").is_file() and (tmp_path / "s1" / "live_table.json").is_file()

    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    server = Server((200, or_response(content='{"plugged": 3}')))
    paid = Router(session="S1", models=models, ledger=led, http=server.client())
    out = smoke.run_smoke(only=["orm"], models=models, router=paid, out_dir=tmp_path / "s1", echo=False)
    (row,) = out["rows"]
    assert row["answer"] == smoke.EXPECTED_PLUGGED == 3 and row["correct"] and row["usd_source"] == "provider"
    assert row["usd"] == 0.00123 and row["latency_ms"] >= 0 and out["spend_usd_this_run"] == pytest.approx(0.00123)
    assert out["session"] == "S1" and [r["session"] for r in _reserves(led)] == [None, "S1"]
    body = server.bodies()[0]
    assert body["reasoning"] == {"enabled": False} and body["max_tokens"] == 300
    assert body["usage"] == {"include": True}
    assert body["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert PLAIN_KEY not in (tmp_path / "s1" / "live_table.json").read_text(encoding="utf-8")
    assert PLAIN_KEY not in smoke.format_smoke_table(out["rows"])
    # a model that cannot turn reasoning off gets the larger allowance, and the row says why
    out = smoke.run_smoke(only=["orm_must"], models=models, router=paid, out_dir=tmp_path / "s1", echo=False)
    assert server.bodies()[1]["max_tokens"] == smoke.MAX_TOKENS_REASONING == 1000
    assert server.bodies()[1]["reasoning"] == {"effort": "low"} and "mandatory" in out["rows"][0]["note"]


# =========================================================================== loopkit S1 fixes: R1 to R15
# --------------------------------------------------------------------------- R1: every HTTP attempt is reserved
def test_r1_lost_attempt_stays_reserved_when_a_retry_answers(ask, ledger, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    # O1: a refused connection ("connect") is not lost, it settles at $0 (test_o1_*). A connection that dropped
    # after the request was sent is lost
    for first in ("timeout", "drop", (524, "origin timed out")):
        before = ledger.spent()
        rec = Recorder()
        server = Server(first, (200, or_response()))
        r = ask("orm", f"judge the frame {first!r}", max_tokens=8000, http=server.client(), events=rec)
        assert len(server.requests) == 2 and r.usd == 0.00123 and r.usd_source == "provider"
        reserves = _reserves(ledger)[-2:]
        assert [x["usd"] for x in reserves] == [pytest.approx(r.projected_usd)] * 2
        assert [x["tags"]["attempt"] for x in reserves] == [1, 2] and len({x["tags"]["call"] for x in reserves}) == 1
        assert _settles(ledger)[-1]["id"] == reserves[1]["id"] and _settles(ledger)[-1]["usd"] == 0.00123
        # the lost attempt may have been generated and billed: it stays counted at its projection
        assert ledger.spent() - before == pytest.approx(r.projected_usd + 0.00123)
        p = rec.events[0][1]
        assert p["usd"] == 0.00123 and p["ledger_usd"] == pytest.approx(r.projected_usd + 0.00123)
        assert (p["reserved_attempts"], p["unreconciled_attempts"]) == (2, 1)
        assert p["unreconciled_reserved_usd"] == pytest.approx(r.projected_usd) and r.raw["unreconciled_attempts"] == 1
    summary = ledger.summary()
    assert summary["calls"] == 3 and summary["reservations"] == 6 and summary["unsettled"]["count"] == 3
    hit = ask("orm", "judge the frame 'timeout'", max_tokens=8000)  # a hit is not charged for the lost attempt
    assert hit.cache_hit and hit.usd == 0.0 and "ledger_usd" not in hit.raw and "unreconciled_attempts" not in hit.raw
    assert hit.raw["original_ledger_usd"] == pytest.approx(hit.projected_usd + 0.00123)


def test_r1_each_retry_is_cap_checked_before_it_is_sent(ask, ledger, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    tight = recap(ledger, T=0.70)  # orm: 35 chars and 30000 max_tokens project $0.30002 per attempt
    rec = Recorder()
    server = Server("timeout")
    with pytest.raises(SpendCapExceeded) as err:
        ask("orm", "x" * 35, max_tokens=30000, http=server.client(), events=rec, ledger=tight)
    assert len(server.requests) == 2 and err.value.scope == "session"  # the third POST was never sent
    assert err.value.spent_usd == pytest.approx(2 * 0.30002) and ledger.spent() == pytest.approx(2 * 0.30002)
    (kind, p), = rec.events
    assert kind == "model_call" and p["ok"] is False and "SpendCapExceeded" in p["error"] and p["attempts"] == 2
    assert p["usd"] == pytest.approx(2 * 0.30002) and p["unreconciled_attempts"] == 2
    ModelCallPayload.model_validate(p)


def test_r1_rate_limit_refusals_settle_at_zero_and_the_retry_is_reserved(ask, ledger, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    sleeps: list[float] = []
    server = Server((429, {"error": {"message": "rate limited"}}, {"Retry-After": "1"}), (200, or_response()))
    r = ask("orm", "rate limited once", http=server.client(sleeps))
    assert len(server.requests) == 2 and sleeps == [1.0] and r.usd == 0.00123
    assert [x["usd"] for x in _settles(ledger)] == [0.0, 0.00123] and len(_reserves(ledger)) == 2
    assert ledger.spent() == pytest.approx(0.00123)


def test_r1_agent_calls_default_to_a_900_s_deadline(ask, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    seen: list[float] = []

    def capture(request):
        seen.append(request.extensions["timeout"]["read"])
        return httpx.Response(200, json=or_response())

    rec = Recorder()
    for role, timeout_s in (("agent", None), ("judge", None), ("decider", None), ("test", None), ("agent", 42.0)):
        ask("orm", f"{role} {timeout_s}", role=role, timeout_s=timeout_s, http=Server(capture).client(), events=rec)
    assert seen == [900.0, 300.0, 300.0, 300.0, 42.0]
    assert [p["timeout_s"] for _, p in rec.events] == [900.0, 300.0, 300.0, 300.0, 42.0]
    with pytest.raises(ValueError):
        ask("orm", "bad", timeout_s=float("inf"))


def test_r1_deadline_keeps_a_body_that_already_arrived(ask, ledger, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    body = json.dumps(or_response()).encode()

    def late(request):  # the whole answer is in the last chunk, which lands just after the deadline
        def chunks():
            yield body[:40]
            time.sleep(0.6)
            yield body[40:]
        return httpx.Response(200, content=chunks())

    server = Server(late)
    r = ask("orm", "late but complete", timeout_s=0.4, http=server.client())
    assert len(server.requests) == 1 and r.text == '{"plugged": 3}'
    assert [a["status"] for a in r.raw["attempt_log"]] == [200] and ledger.spent() == pytest.approx(0.00123)

    def sized(complete):  # not JSON, so only its Content-Length can say it is whole
        def handler(request):
            def chunks():
                yield b"plain text "
                time.sleep(0.6)
                yield b"answer"
            n = len(b"plain text answer") + (0 if complete else 10)
            return httpx.Response(200, content=chunks(), headers={"Content-Length": str(n)})
        return handler

    def one_try(handler):
        return HttpClient(transport=httpx.MockTransport(handler), sleep=lambda s: None, max_attempts=1)

    res = one_try(sized(True)).post_json("https://a.invalid/x", {}, timeout_s=0.4)
    assert res.ok and res.text == "plain text answer" and res.attempts == 1
    res = one_try(sized(False)).post_json("https://a.invalid/x", {}, timeout_s=0.4)
    assert res.status is None and "timeout" in res.error and res.attempt_log[0]["outcome"] == "lost"


# --------------------------------------------------------------------------- R2: rejected only when nothing was lost
def test_r2_refusal_after_a_lost_attempt_is_not_rejected_before_generation(ask, ledger, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    # O1: "connect" then a refusal is rejected before generation (nothing reached the provider): test_o1_*
    for first in ("timeout", "drop"):
        before = ledger.spent()
        rec = Recorder()
        server = Server(first, (402, {"error": {"code": 402, "message": "Insufficient credits"}}))
        with pytest.raises(ModelCallError) as err:
            ask("orm", f"hi {first}", http=server.client(), events=rec)
        assert err.value.status == 402 and err.value.attempts == 2 and not err.value.rejected_before_generation
        lost = _reserves(ledger)[-2]
        assert _settles(ledger)[-1]["usd"] == 0.0 and _settles(ledger)[-1]["id"] == _reserves(ledger)[-1]["id"]
        assert ledger.spent() - before == pytest.approx(lost["usd"]) and rec.events[0][1]["usd"] == pytest.approx(
            lost["usd"])
    # across JSON modes: a lost 503 in the first mode, then refusals, is still not "rejected before generation"
    no_route = (503, {"error": {"message": "No available providers support the requested parameters"}})
    refuse_mode = (400, {"error": {"message": "response_format is not supported"}})
    refuse = (400, {"error": {"message": "invalid request"}})
    with pytest.raises(ModelCallError) as err:
        ask("orm", "modes", json_schema=COUNT_SCHEMA, http=Server(no_route, refuse_mode, refuse).client())
    assert err.value.attempts == 3 and not err.value.rejected_before_generation
    assert [entry["outcome"] for entry in err.value.raw["attempt_log"]] == ["lost", "rejected", "rejected"]
    # every attempt refused, in every mode: rejected, and nothing is counted
    before = ledger.spent()
    with pytest.raises(ModelCallError) as err:
        ask("orm", "modes refused", json_schema=COUNT_SCHEMA, http=Server(refuse_mode, refuse).client())
    assert err.value.rejected_before_generation and err.value.attempts == 2 and ledger.spent() == pytest.approx(before)


# --------------------------------------------------------------------------- R3: spend-ledger I/O
def test_r3_ledger_records_are_ascii_and_split_on_newline_only(ledger):
    odd = "rehearse round" + LSEP + "1\x85end " + E_ACUTE
    rid = ledger.reserve(0.30, "orm", "d" * 64, session="T", tags={"note": odd})
    ledger.settle(rid, 0.25, "provider")
    raw = ledger.path.read_bytes()
    assert raw.isascii() and raw.count(b"\n") == 2
    assert [r["type"] for r in ledger.records()] == ["reserve", "settle"] and ledger.records()[0]["tags"]["note"] == odd
    assert ledger.spent() == pytest.approx(0.25)
    # a record an older writer left with raw separators inside a string still reads back whole
    old = {"type": "reserve", "id": "old1", "usd": 0.5, "session": "T", "model_key": "orm", "tags": {"phase": odd}}
    with open(ledger.path, "ab") as f:
        f.write(json.dumps(old, ensure_ascii=False).encode("utf-8") + b"\r\n")
    assert ledger.spent() == pytest.approx(0.75) and ledger.summary()["corrupt_lines"] == 0


def test_r3_session_names_model_keys_and_phases_must_be_single_line(ask, ledger):
    for bad in ("rehearse round" + LSEP + "1", "a\nb", "x\x85y", "tab\there", "p" + PSEP + "q"):
        with pytest.raises(ValueError):
            ask("fake", "hi", phase=bad)
        with pytest.raises(ValueError):
            ask("fake", "hi", session=bad)
        with pytest.raises(RouterError):
            SpendCaps(total_usd=None, sessions={bad: 1.0})
        with pytest.raises(ValueError):
            ledger.reserve(0.1, bad, "e" * 64, session="T")
        with pytest.raises(ValueError):
            ledger.reserve(0.1, "orm", "e" * 64, session=bad)
        with pytest.raises(ValueError):
            ledger.reserve(0.1, "orm", "e" * 64, session="T", tags={"phase": bad})
        with pytest.raises(ValueError):
            chat(bad, "hi", models=MODELS, ledger=ledger)
        with pytest.raises(ValueError):
            Router(session=bad, ledger=ledger)
    assert ledger.records() == []
    assert ask("fake", "hi", phase="rehearse round 1").text == "fake response"  # plain spaces are fine
    with pytest.raises(ValueError):  # a cache hit never reaches the spend ledger, and is refused all the same
        ask("fake", "hi", phase="rehearse round" + LSEP + "1")


def test_r3_orphan_settles_are_counted_and_warned(ledger):
    rid = ledger.reserve(0.2, "orm", "f" * 64, session="T")
    ledger.settle(rid, 0.1, "provider")
    ledger.settle("no-such-reserve", 0.4, "provider")
    with pytest.warns(RuntimeWarning, match="no matching reserve"):
        ledger.reserve(0.1, "orm", "g" * 64, session="T")  # the next reservation says so
    with pytest.warns(RuntimeWarning, match="no matching reserve"):
        summary = ledger.summary()  # and so does every summary
    assert summary["orphan_settles"] == {"count": 1, "usd": 0.4}
    assert summary["total_usd"] == pytest.approx(0.2) and summary["calls"] == 2


# --------------------------------------------------------------------------- R4: BYOK and cost sanity
def test_r4_byok_upstream_cost_is_added(ask, ledger, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    body = or_response(cost=0.0005, prompt=5000, completion=15000)  # orm: 5000 * 2 + 15000 * 10 = $0.16
    body["usage"].update(is_byok=True, cost_details={"upstream_inference_cost": 0.16})
    r = ask("orm", "byok", max_tokens=16000, http=Server((200, body)).client())
    assert r.usd == pytest.approx(0.1605) and r.usd_source == "provider" and ledger.spent() == pytest.approx(0.1605)
    assert r.raw["cost_breakdown"] == {"usage_cost": 0.0005, "upstream_inference_cost": 0.16, "is_byok": True,
                                       "cost": pytest.approx(0.1605)}


def test_r4_provider_cost_far_below_token_cost_settles_at_computed(ask, ledger, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    tokens = dict(prompt=5000, completion=15000)
    r = ask("orm", "zero", max_tokens=16000, http=Server((200, or_response(cost=0.0, **tokens))).client())
    assert r.usd == pytest.approx(0.16) and r.usd_source == "computed" and "below half" in r.raw["usd_note"]
    fee_only = or_response(cost=0.008, **tokens)
    fee_only["usage"]["is_byok"] = True  # BYOK without the upstream amount: the fee alone is not the cost
    r2 = ask("orm", "fee only", max_tokens=16000, http=Server((200, fee_only)).client())
    assert r2.usd == pytest.approx(0.16) and r2.usd_source == "computed" and "note" in r2.raw["cost_breakdown"]
    r3 = ask("orm", "just over half", max_tokens=16000, http=Server((200, or_response(cost=0.081, **tokens))).client())
    assert r3.usd == 0.081 and r3.usd_source == "provider"
    assert [s["usd"] for s in _settles(ledger)] == [pytest.approx(0.16), pytest.approx(0.16), 0.081]
    # cached tokens count at cached_in: 1000 * 2 + 9000 * 0.2 + 100 * 10 = $0.0048 is a plausible provider cost
    cached = or_response(cost=0.0048, prompt=10000, completion=100, cached=9000, model="vendor/cache-model")
    r4 = ask("orm_cache", "cached", http=Server((200, cached)).client())
    assert r4.usd == 0.0048 and r4.usd_source == "provider"


# --------------------------------------------------------------------------- R5: see test_ledger_lock_* above


# --------------------------------------------------------------------------- R6: reasoning tokens per adapter
def test_r6_reasoning_tokens_follow_the_adapter_property(ask, monkeypatch):
    assert get_adapter("openrouter").reasoning_in_output is True and get_adapter("elorian").reasoning_in_output is None
    monkeypatch.setenv("ELORIAN_API_KEY", PLAIN_KEY)
    el = Server((200, or_response(cost=None, prompt=2000, completion=120, reasoning=6000, model="elorian-test")))
    r = ask("el", "count", max_tokens=8000, http=el.client())
    assert r.usd == pytest.approx((2000 * 0.6 + (120 + 6000) * 3.0) / 1e6) and "separately" in r.raw["usd_note"]
    el2 = Server((200, or_response(cost=None, prompt=2000, completion=120, reasoning=50, model="elorian-test")))
    r2 = ask("el", "count again", max_tokens=8000, http=el2.client())
    assert r2.usd == pytest.approx((2000 * 0.6 + 170 * 3.0) / 1e6) and "unknown" in r2.raw["usd_note"]
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)  # OpenRouter counts them inside completion_tokens
    orx = Server((200, or_response(cost=None, prompt=2000, completion=120, reasoning=100)))
    r3 = ask("orm", "count", http=orx.client())
    assert r3.usd == pytest.approx((2000 * 2.0 + 120 * 10.0) / 1e6)


# --------------------------------------------------------------------------- R7: nothing raises after a paid call
def test_r7_degenerate_json_is_a_json_error_never_an_exception(ask, ledger, monkeypatch):
    rec = Recorder()
    num_schema = {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]}
    replies = ['{"x": 1' + "0" * 400 + "}", '{"x": ' + "5" * 5000 + "}", "[" * 3000,
               '{"x": ' + "[" * 1500 + "]" * 1500 + "}"]
    with fake_script(replies):
        answers = [ask("paid_fake", f"q{i}", json_schema=num_schema, max_tokens=50, events=rec) for i in range(4)]
    assert all(a.json_error for a in answers[1:]) and answers[1].json_error.startswith("not JSON")
    assert len(rec.events) == 4 and all(p["ok"] for _, p in rec.events)
    assert [x["type"] for x in ledger.records()] == ["reserve", "settle"] * 4
    # a validator that blows up (OverflowError for a huge number, RecursionError for deep nesting) is a json_error
    for boom in (OverflowError("int too large to convert to float"), RecursionError("maximum recursion depth")):
        def raising_check(value, schema, _boom=boom):
            raise _boom
        monkeypatch.setattr(router_core, "check", raising_check)
        with fake_script(['{"x": 1}']):
            r = ask("paid_fake", f"boom {type(boom).__name__}", json_schema=num_schema, max_tokens=50)
        assert r.json == {"x": 1} and r.json_error.startswith("schema check failed")
    for text in ('{"a": ' + "9" * 5000 + "}", "[" * 100000 + "]" * 100000, None, "```json\n" + "{" * 5000):
        value, err = parse_json_text(text)
        assert value is None and err.startswith(("not JSON", "empty answer"))
    # a degenerate first block does not stop the parser from reading a good one after it
    two_blocks = "```\n" + "[" * 3000 + "\n```\nThe corrected answer:\n```json\n{\"x\": 1}\n```"
    assert parse_json_text(two_blocks) == ({"x": 1}, None)


def test_r7_failure_after_the_paid_call_settles_and_records_before_raising(ask, ledger, cache, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)

    def broken_put(*a, **k):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(cache, "put", broken_put)
    rec = Recorder()
    with pytest.raises(RuntimeError, match="disk on fire"):
        ask("orm", "paid then broken", http=Server((200, or_response())).client(), events=rec)
    (kind, p), = rec.events
    assert p["ok"] is False and "disk on fire" in p["error"] and p["usd"] == pytest.approx(0.00123)
    assert [s["usd"] for s in _settles(ledger)] == [0.00123]
    ModelCallPayload.model_validate(p)

    def broken_cost(*a, **k):
        raise ArithmeticError("cost model broke")

    # G6: a cost that cannot be computed after the paid call used to raise ArithmeticError here (a non-RouterError,
    # and the paid answer lost). Now the answer is returned at the projection, which stays counted unsettled.
    monkeypatch.setattr(router_core, "settled_cost", broken_cost)
    rec = Recorder()
    r = ask("orm", "cost broke", use_cache=False, http=Server((200, or_response())).client(), events=rec)
    (kind, p), = rec.events
    last = _reserves(ledger)[-1]
    assert r.text == '{"plugged": 3}' and r.usd == pytest.approx(last["usd"]) and r.usd_source == "computed"
    assert "could not be computed" in r.raw["usd_note"] and "ArithmeticError" in r.raw["usd_note"]
    assert p["ok"] is True and p["usd"] == pytest.approx(last["usd"]) and p["unreconciled_attempts"] == 1
    assert last["id"] not in {s["id"] for s in _settles(ledger)}  # unknown cost: stays counted at the projection
    ModelCallPayload.model_validate(p)


# --------------------------------------------------------------------------- R8: reasoning budget in the projection
def test_r8_a_reasoning_token_budget_is_projected(ask, ledger, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    tight = recap(ledger, T=0.05)
    server = Server((200, or_response()))
    with pytest.raises(SpendCapExceeded) as err:
        ask("orm", "x" * 35, max_tokens=1000, reasoning={"max_tokens": 32000}, http=server.client(), ledger=tight)
    assert err.value.projected_usd == pytest.approx((10 * 2.0 + 33000 * 10.0) / 1e6) and not server.requests
    budget_models = _models()
    budget_models.models["orm_hi"].reasoning = {"max_tokens": 32000}
    with pytest.raises(SpendCapExceeded) as err:
        ask("orm_hi", "x" * 35, max_tokens=1000, reasoning="highest", models=budget_models, http=server.client(),
            ledger=tight)
    assert err.value.projected_usd == pytest.approx((10 * 2.0 + 33000 * 10.0) / 1e6) and not server.requests
    rec = Recorder()
    r = ask("orm", "x" * 35, max_tokens=1000, reasoning={"type": "enabled", "budget_tokens": 5000},
            http=server.client(), events=rec)
    assert r.projected_usd == pytest.approx((10 * 2.0 + 6000 * 10.0) / 1e6)
    assert rec.events[0][1]["reasoning_budget"] == 5000
    assert server.bodies()[0]["reasoning"] == {"type": "enabled", "budget_tokens": 5000}


# --------------------------------------------------------------------------- R9: every non-fake adapter is paid
def test_r9_a_zero_priced_paid_model_still_needs_a_session_and_respects_caps(tmp_path, cache, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    bare = SpendLedger(tmp_path / "bare" / "spend_ledger.jsonl", caps(total=0.10, Z=0.10))

    def call(text, session, cost=0.25):
        return chat("free_or", text, max_tokens=16000, models=MODELS, ledger=bare, cache=cache, use_cache=False,
                    session=session, http=Server((200, or_response(cost=cost, model="vendor/free-tier"))).client())

    with pytest.raises(SpendCapExceeded) as err:
        call("no session", None)
    assert err.value.scope == "no_session" and bare.records() == []
    r = call("first", "Z")  # projects $0, admitted, and the provider's $0.25 is settled
    assert r.projected_usd == 0.0 and r.usd == 0.25 and bare.spent() == 0.25
    with pytest.raises(SpendCapExceeded) as err:  # the session is over its cap, so even a $0 projection is refused
        call("second", "Z")
    assert err.value.scope == "session" and len(_reserves(bare)) == 1
    at_cap = SpendLedger(tmp_path / "at_cap" / "spend_ledger.jsonl", caps(A=0.25))
    # no model or total cap: only "at the cap" can refuse the next call
    assert chat("free_or", "fills the cap", models=MODELS, ledger=at_cap, cache=cache, use_cache=False, session="A",
                http=Server((200, or_response(cost=0.25))).client()).usd == 0.25
    with pytest.raises(SpendCapExceeded) as err:  # exactly at the cap counts as reached
        chat("free_or", "one more", models=MODELS, ledger=at_cap, cache=cache, use_cache=False, session="A",
             http=Server((200, or_response(cost=0.0))).client())
    assert err.value.scope == "session" and err.value.spent_usd == pytest.approx(0.25)
    assert chat("fake", "the free fake needs nothing", models=MODELS,
                ledger=SpendLedger(tmp_path / "b2" / "l.jsonl", caps(T=1.0)), cache=cache).usd == 0.0


def test_r9_model_entries_reject_non_finite_and_unmarked_zero_prices(tmp_path):
    good = _entry(id="vendor/x", price_in=1.0, price_out=2.0)
    for field, bad in (("price_in", float("nan")), ("price_out", float("inf")), ("cached_in", float("nan")),
                       ("price_in", -1.0)):
        with pytest.raises(ValidationError):
            ModelEntry.model_validate({**good, field: bad})
    with pytest.raises(ValidationError, match="free"):
        ModelEntry.model_validate({**good, "price_in": 0.0, "price_out": 0.0})
    for adapter in ("openrouter", "elorian", "jev"):
        with pytest.raises(ValidationError, match="free"):
            ModelEntry.model_validate({**good, "adapter": adapter, "price_in": 0, "price_out": 0})
    ModelEntry.model_validate({**good, "price_in": 0.0, "price_out": 0.0, "notes": "Free during the preview"})
    ModelEntry.model_validate({**good, "adapter": "fake", "price_in": 0.0, "price_out": 0.0})
    ModelEntry.model_validate({**good, "price_in": 0.04, "price_out": 0.0})  # input-only billing is not 0 / 0
    yaml_text = ("version: t\ncurrency: USD\nunit: usd_per_million_tokens\nmodels:\n  el:\n"
                 "    display: e\n    adapter: elorian\n    id: e-1\n    base_url: https://e.invalid/v1\n"
                 "    price_in: 0.6\n    price_out: 3.0\n    cached_in: CACHED\n    vision: true\n"
                 "    roles: [judge]\n    verified: true\n    source: t\n    accessed: 2026-11-11\n")
    yaml_path = tmp_path / "models.yaml"
    yaml_path.write_text(yaml_text.replace("CACHED", "0.06"), encoding="utf-8")
    assert load_models(yaml_path).get("el").cached_in == 0.06  # the same file with a number loads
    assert load_models(yaml_path).get("el").accessed == "2026-11-11"  # an unquoted YAML date becomes its text
    yaml_path.write_text(yaml_text.replace("CACHED", ".nan"), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_models(yaml_path)
    load_models()  # the committed models.yaml stays valid


# --------------------------------------------------------------------------- R10: NaN
def test_r10_spend_caps_in_the_config_reject_nan_infinity_and_negative_values(tmp_path, cache, monkeypatch):
    raw = yaml.safe_load(config.EXPERIMENT_YAML.read_text(encoding="utf-8"))
    # true and "8" are here because the typed config would read them as 1.0 and 8.0: a typo must not become a cap
    for n, (name, bad) in enumerate((("S1", float("nan")), ("build_total", float("inf")), ("S3", -1.0),
                                     ("build_total", float("nan")), ("full_judge_run", float("-inf")),
                                     ("S1", True), ("S3", "8"), ("build_total", "15"))):
        broken = json.loads(json.dumps(raw["spend_caps_usd"]))
        broken[name] = bad
        path = tmp_path / f"experiment_{n}.yaml"
        path.write_text(yaml.safe_dump({**raw, "spend_caps_usd": broken}), encoding="utf-8")
        with pytest.raises(RouterError):
            load_spend_caps(path)
    no_total = {k: v for k, v in raw["spend_caps_usd"].items() if k != "build_total"}
    path = tmp_path / "experiment_no_total.yaml"
    path.write_text(yaml.safe_dump({**raw, "spend_caps_usd": no_total}), encoding="utf-8")
    with pytest.raises(RouterError, match="build_total"):
        load_spend_caps(path)
    with pytest.raises(RouterError, match="cannot read the spend caps"):
        load_spend_caps(tmp_path / "missing.yaml")
    # a damaged config refuses calls instead of admitting them: nothing is reserved, sent or written
    broken_path = tmp_path / "experiment_0.yaml"
    monkeypatch.setattr(config, "EXPERIMENT_YAML", broken_path)
    led = SpendLedger(tmp_path / "broken" / "spend_ledger.jsonl")  # no caps passed: they come from the config
    for attempt in (lambda: chat("paid_fake", "hi", models=MODELS, ledger=led, cache=cache, session="S1"),
                    lambda: led.reserve(0.5, "orm", "i" * 64, session="S1"),
                    lambda: Router(session="S1")):
        with pytest.raises(RouterError):
            attempt()
    assert led.records() == []


def test_r10_non_finite_computed_cost_settles_at_the_projection(ask, ledger, monkeypatch):
    monkeypatch.setenv("ELORIAN_API_KEY", PLAIN_KEY)
    models = _models()
    models.models["el"].cached_in = float("nan")  # past validation, as a hand-built entry could be
    server = Server((200, or_response(cost=None, prompt=50000, completion=8000, model="elorian-test")))
    r = ask("el", "count", models=models, http=server.client())
    assert r.usd == pytest.approx(r.projected_usd) and r.usd > 0 and "not a finite" in r.raw["usd_note"]
    assert ledger.spent() == pytest.approx(r.projected_usd)


# --------------------------------------------------------------------------- R11: failed generations
def test_r11_a_failed_generation_raises_and_is_never_cached(ask, ledger, cache, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    partial = '{"holds": true, "confidence": 0.9, "reason": "the shoulder is flu'
    failed = or_response(content=partial, finish="error", cost=0.0004, prompt=100, completion=30)
    rec = Recorder()
    with pytest.raises(ModelCallError, match="failed upstream") as err:
        ask("orm", "judge the port", http=Server((200, failed)).client(), events=rec)
    assert err.value.status == 200 and err.value.usd == 0.0004 and not err.value.rejected_before_generation
    assert not list(cache.root.rglob("*.json")) and ledger.spent() == pytest.approx(0.0004)
    assert rec.events[0][1]["ok"] is False and rec.events[0][1]["finish_reason"] == "error"
    server = Server((200, or_response(content='{"holds": false}')))
    r = ask("orm", "judge the port", http=server.client())  # the identical request goes to the provider again
    assert not r.cache_hit and len(server.requests) == 1 and r.text == '{"holds": false}'
    no_cost = or_response(content="partial")
    no_cost["choices"][0]["error"] = {"message": "upstream timed out", "code": 502}
    del no_cost["usage"]["cost"]
    before = ledger.spent()
    with pytest.raises(ModelCallError, match="upstream timed out"):
        ask("orm", "error object", http=Server((200, no_cost)).client())
    assert ledger.spent() - before == pytest.approx(_reserves(ledger)[-1]["usd"])  # unknown cost: the projection
    # a failed generation left in the cache by an older router is not served either
    hit_hash = r.request_hash
    record = json.loads(cache.path_for(hit_hash).read_text(encoding="utf-8"))
    record["finish_reason"] = "error"
    cache.path_for(hit_hash).write_text(json.dumps(record), encoding="utf-8")
    again = Server((200, or_response(content='{"holds": true}')))
    assert not ask("orm", "judge the port", http=again.client()).cache_hit and len(again.requests) == 1


# --------------------------------------------------------------------------- R12: pre-send failures
def test_r12_numpy_values_in_schema_and_extra_are_sent_as_plain_json(ask, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    schema = {"type": "object", "properties": {"plugged": {"type": "integer", "minimum": 0,
                                                           "maximum": np.int64(8)}}, "required": ["plugged"]}
    server = Server((200, or_response()))
    r = ask("orm", "count", json_schema=schema, http=server.client())
    assert r.json == {"plugged": 3} and r.json_error is None
    sent = server.bodies()[0]["response_format"]["json_schema"]["schema"]
    assert sent["properties"]["plugged"]["maximum"] == 8 and type(sent["properties"]["plugged"]["maximum"]) is int
    server2 = Server((200, or_response()))
    ask("orm", "warm", temperature=np.float32(0.5), reasoning={"max_tokens": np.int64(100)}, http=server2.client())
    assert server2.bodies()[0]["temperature"] == 0.5 and server2.bodies()[0]["reasoning"] == {"max_tokens": 100}
    with pytest.raises(ValueError):
        ask("orm", "nan", json_schema={"type": "number", "maximum": float("nan")})
    with pytest.raises(ValueError):
        ask("orm", "nan", temperature=float("nan"))


def test_r12_unparseable_url_is_not_configured_before_any_spend(ask, ledger, monkeypatch):
    monkeypatch.setenv("ELORIAN_API_KEY", PLAIN_KEY)
    for bad in ("https://api.elorian.invalid:443v1", "https://[::1/v1"):
        models = _models()
        models.models["el"].base_url = bad
        with pytest.raises(ModelNotConfigured, match="not a valid URL"):
            ask("el", "hi", models=models)
    assert ledger.records() == []


def test_r12_a_request_that_cannot_be_built_is_rejected_and_settled_at_zero(ask, ledger, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    server = Server((200, or_response()))
    with monkeypatch.context() as m:
        m.setattr(OpenRouterAdapter, "headers", lambda self: {"X-Title": "avail " + LQUOTE + "beta" + RQUOTE})
        with pytest.raises(ModelCallError) as err:
            ask("orm", "bad header", http=server.client())
    assert err.value.rejected_before_generation and err.value.status is None and "nothing was sent" in str(err.value)

    def bad_body(self, request, mode=None):
        raise TypeError("cannot build this")

    with monkeypatch.context() as m:
        m.setattr(OpenRouterAdapter, "build_payload", bad_body)
        with pytest.raises(ModelCallError, match="could not be built") as err:
            ask("orm", "bad body", http=server.client())
    assert err.value.rejected_before_generation and not server.requests and "nothing was sent" in str(err.value)
    assert len(_reserves(ledger)) == 2 and [s["usd"] for s in _settles(ledger)] == [0.0, 0.0]
    assert ledger.spent() == 0.0
    res = refusing_client().post_json("https://a.invalid/x", {"x": float("nan")})
    assert not res.sent and res.attempts == 0 and "not plain JSON" in res.error
    res = refusing_client().post_json("https://a.invalid:443v1/x", {})
    assert not res.sent and "URL is not valid" in res.error


def test_r12_smoke_survives_any_per_model_error(tmp_path, monkeypatch):
    from avail.router import smoke

    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    server = Server((200, or_response(content='{"plugged": 3}')))

    class Flaky(Router):
        def chat(self, model_key, *args, **kwargs):
            if model_key == "orm":
                raise KeyError("not a router error")
            return super().chat(model_key, *args, **kwargs)

    led = SpendLedger(tmp_path / "smoke" / "spend_ledger.jsonl", caps(S1=1.0))
    router = Flaky(session="S1", models=MODELS, ledger=led, http=server.client())
    out = smoke.run_smoke(only=["orm", "orm_hi"], models=MODELS, router=router, out_dir=tmp_path / "s1", echo=False)
    rows = {r["model_key"]: r for r in out["rows"]}
    assert not rows["orm"]["ok"] and "KeyError" in rows["orm"]["error"] and rows["orm_hi"]["ok"]
    assert rows["orm"]["input"] == "image + question"  # the failed row keeps what was asked
    assert (tmp_path / "s1" / "live_table.json").is_file()
    monkeypatch.setattr(smoke, "_call_one", lambda *a, **k: 1 / 0)  # even a failure outside chat() is 1 row
    out = smoke.run_smoke(only=["orm"], models=MODELS, router=router, out_dir=tmp_path / "s1", echo=False)
    assert out["rows"][0]["ok"] is False and "ZeroDivisionError" in out["rows"][0]["error"]


# --------------------------------------------------------------------------- R13: the cache key
def test_r13_schema_property_order_and_destination_change_the_request_hash(ask, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    first = {"type": "object", "properties": {"plugged": {"type": "integer"}, "reasoning": {"type": "string"}},
             "required": ["plugged", "reasoning"]}
    second = {"type": "object", "properties": {"reasoning": {"type": "string"}, "plugged": {"type": "integer"}},
              "required": ["plugged", "reasoning"]}
    r1 = ask("orm", "count", json_schema=first, http=Server((200, or_response(content='{"plugged": 4, '
                                                                                   '"reasoning": "4"}'))).client())
    s2 = Server((200, or_response(content='{"reasoning": "three", "plugged": 3}')))
    r2 = ask("orm", "count", json_schema=second, http=s2.client())
    assert r1.request_hash != r2.request_hash and not r2.cache_hit and len(s2.requests) == 1
    assert ask("orm", "count", json_schema=json.loads(json.dumps(first))).cache_hit  # the same schema is a hit
    elsewhere = _models()
    elsewhere.models["orm"].base_url = "https://staging.router.invalid/api/v1"
    s3 = Server((200, or_response()))
    r3 = ask("orm", "count", json_schema=first, models=elsewhere, http=s3.client())
    assert not r3.cache_hit and r3.request_hash != r1.request_hash
    assert str(s3.requests[0].url) == "https://staging.router.invalid/api/v1/chat/completions"


# --------------------------------------------------------------------------- R14: keys that cannot go in a header
def test_r14_a_non_ascii_key_is_refused_by_name_before_any_reservation(ask, ledger, monkeypatch):
    core_secret = "sk-or-v1-" + "a1" * 32
    server = Server((200, or_response()))
    for bad in (LQUOTE + core_secret + RQUOTE, core_secret[:9] + ZWSP + core_secret[9:],
                core_secret[:9] + " " + core_secret[9:]):
        monkeypatch.setenv("OPENROUTER_API_KEY", bad)
        with pytest.raises(MissingCredentialError) as err:
            ask("orm", "hi", http=server.client())
        for text in (str(err.value), repr(err.value), repr(err.value.args)):
            assert core_secret not in text and core_secret[9:30] not in text
        assert "OPENROUTER_API_KEY" in str(err.value) and err.value.__cause__ is None and err.value.__suppress_context__
    assert not server.requests and ledger.records() == []


def test_r14_header_build_errors_are_redacted_and_not_chained(ask, monkeypatch):
    secret = "av-test-" + LQUOTE + "Q7" * 20
    res = Server((200, {})).client().post_json("https://a.invalid/x", {"a": 1}, key=secret)
    assert not res.sent and res.attempts == 0 and "ASCII" in res.error and "Q7Q7" not in res.error
    with pytest.raises(ModelCallError) as err:
        OpenRouterAdapter().send(make_request("orm"), secret, Server((200, or_response())).client())
    assert err.value.rejected_before_generation and err.value.__cause__ is None and err.value.__context__ is None
    for text in (str(err.value), repr(err.value), json.dumps(err.value.raw)):
        assert "Q7Q7" not in text


# --------------------------------------------------------------------------- R15: the jev path scrubs the parsed answer
def test_r15_a_key_echoed_in_a_jev_answer_is_scrubbed_from_json_and_cache(ask, cache, monkeypatch):
    monkeypatch.setenv("TYPESAFE_JEV_API_KEY", PLAIN_KEY)
    questions = {"q": {"type": "choice", "instructions": "How many plugged ports?", "criteria": {"3": "three"}}}
    answers = {"q": {"type": "choice", "choice": "3", "debug": f"authorized as Bearer {PLAIN_KEY}",
                     PLAIN_KEY: ["as a key", {"deeper": PLAIN_KEY}]}}
    server = Server((200, {"id": "r1", "model": "jev-latest", "answers": answers,
                           "usage": {"input_tokens": 10, "output_tokens": 1}}))
    r = ask("jev", "board", role="decider", extra={"questions": questions}, http=server.client())
    assert PLAIN_KEY not in json.dumps(r.json) and PLAIN_KEY not in r.text and PLAIN_KEY not in json.dumps(r.raw)
    assert r.json["q"]["choice"] == "3" and r.json["q"]["debug"] == "authorized as Bearer [redacted]"
    assert r.json["q"]["[redacted]"] == ["as a key", {"deeper": "[redacted]"}]
    assert PLAIN_KEY not in cache.path_for(r.request_hash).read_text(encoding="utf-8")
    hit = ask("jev", "board", role="decider", extra={"questions": questions})
    assert hit.cache_hit and PLAIN_KEY not in json.dumps(hit.json) and hit.json == r.json


# =========================================================================== loopkit round 2: G1 to G6, O1
# --------------------------------------------------------------------------- G1: a non-2xx cost settles nothing
def test_g1_a_cost_in_a_failed_body_never_settles_a_lost_attempt(ask, ledger, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    rec = Recorder()
    server = Server("timeout", "timeout", (500, {"error": {"message": "upstream exploded"}, "usage": {"cost": 0}}))
    with pytest.raises(ModelCallError) as err:
        ask("orm", "lost, lost, 500", http=server.client(), events=rec)
    assert len(server.requests) == 3 and err.value.status == 500 and err.value.usd is None
    reserves = _reserves(ledger)
    assert len(reserves) == 3 and _settles(ledger) == []  # the 500's "cost 0" settled nothing
    assert ledger.spent() == pytest.approx(3 * reserves[0]["usd"])
    p = rec.events[0][1]
    assert p["ok"] is False and p["unreconciled_attempts"] == 3 and p["usd"] == pytest.approx(3 * reserves[0]["usd"])
    # a refusal that reports a cost after a lost attempt: the refusal is $0, the lost attempt stays at its projection
    before = ledger.spent()
    refused = (402, {"error": {"message": "Insufficient credits"}, "usage": {"cost": 0.5}})
    with pytest.raises(ModelCallError) as err:
        ask("orm", "lost then refused", http=Server("timeout", refused).client())
    assert err.value.usd is None and _settles(ledger)[-1]["usd"] == 0.0
    assert ledger.spent() - before == pytest.approx(_reserves(ledger)[-2]["usd"])
    # a 200 whose generation failed still settles its own attempt at the cost it reported (that attempt answered)
    before = ledger.spent()
    failed = or_response(finish="error", cost=0.0004, prompt=100, completion=30)
    with pytest.raises(ModelCallError, match="failed upstream"):
        ask("orm", "lost then failed generation", http=Server("timeout", (200, failed)).client())
    assert _settles(ledger)[-1]["id"] == _reserves(ledger)[-1]["id"] and _settles(ledger)[-1]["usd"] == 0.0004
    assert ledger.spent() - before == pytest.approx(_reserves(ledger)[-2]["usd"] + 0.0004)


# --------------------------------------------------------------------------- G2: the floor uses the counts reported
def test_g2_the_cost_floor_uses_whatever_token_counts_were_reported(ask, ledger, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    only_prompt = or_response(cost=0.0, prompt=200000)
    del only_prompt["usage"]["completion_tokens"]
    r = ask("orm", "x" * 35, max_tokens=1000, http=Server((200, only_prompt)).client())
    # orm prices 2 / 10: the 200000 input tokens it reported cost $0.40 alone, so its $0 is not trusted. The
    # output it did not report is taken at the output allowance
    assert r.usd_source == "computed" and r.usd == pytest.approx((200000 * 2.0 + 1000 * 10.0) / 1e6)
    assert "below half" in r.raw["usd_note"] and "counted as 0" in r.raw["usd_note"]
    assert _settles(ledger)[-1]["usd"] == pytest.approx(r.usd)
    only_completion = or_response(cost=0.0, completion=15000)
    del only_completion["usage"]["prompt_tokens"]
    r2 = ask("orm", "x" * 35, max_tokens=16000, http=Server((200, only_completion)).client())
    assert r2.usd_source == "computed" and r2.usd == pytest.approx((10 * 2.0 + 15000 * 10.0) / 1e6)
    fair = or_response(cost=0.25, prompt=200000)  # at least half of what its reported counts cost: trusted
    del fair["usage"]["completion_tokens"]
    r3 = ask("orm", "y" * 35, max_tokens=1000, http=Server((200, fair)).client())
    assert r3.usd == 0.25 and r3.usd_source == "provider"


# --------------------------------------------------------------------------- G3: the URL sent is the URL checked
def test_g3_a_url_is_checked_exactly_as_it_will_be_sent(ask, ledger, monkeypatch):
    monkeypatch.setenv("ELORIAN_API_KEY", PLAIN_KEY)
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    monkeypatch.setenv("AV_TEST_SYS1_KEY", PLAIN_KEY)
    server = Server((200, or_response()))
    cases = (("el", "base_url", "https://elorian.invalid/v1\n", "leading or trailing whitespace"),  # a YAML '|' block
             ("el", "base_url", " https://elorian.invalid/v1", "leading or trailing whitespace"),
             ("orm", "base_url", "https://router.invalid/api/v1\t", "leading or trailing whitespace"),
             ("sys1_vision", "endpoint", "https://decisions.invalid/systemone\r\n", "leading or trailing whitespace"),
             ("orm", "base_url", "https://router.invalid/api\x07/v1", "control character"),
             ("orm", "base_url", "https://router.invalid/api/v1" + ZWSP, "control character"),
             ("orm", "base_url", "https://[api.invalid/v1", "not a valid URL"))  # httpx would send it to %5Bapi.invalid
    for key, field, url, why in cases:
        models = _models()
        setattr(models.models[key], field, url)
        kw = {"role": "decider", "extra": {"questions": {"q": {"type": "noul", "instructions": "x"}}}} \
            if key == "sys1_vision" else {}
        with pytest.raises(ModelNotConfigured, match=why) as err:
            ask(key, "hi", models=models, http=server.client(), **kw)
        assert field in str(err.value) and repr(url) in str(err.value)
    assert not server.requests and ledger.records() == []  # refused before any reservation
    models = _models()
    models.models["el"].base_url = "https://elorian.invalid/v1/"
    ask("el", "hi", models=models, http=server.client())
    assert str(server.requests[0].url) == "https://elorian.invalid/v1/chat/completions"


# --------------------------------------------------------------------------- G4: the key's JSON-escaped form is cut too
def test_g4_a_key_with_a_quote_or_a_backslash_is_cut_in_its_json_escaped_form(ask, cache, monkeypatch):
    questions = {"q": {"type": "choice", "instructions": "How many plugged ports?", "criteria": {"3": "three"}}}
    for odd in ('av-test-"quote"-' + "Q7" * 12, "av-test-back\\slash-" + "Q7" * 12):
        escaped = json.dumps(odd)[1:-1]
        assert escaped != odd
        monkeypatch.setenv("TYPESAFE_JEV_API_KEY", odd)
        answers = {"q": {"choice": "3", "echo": odd, "note": f"seen {odd}"}}
        server = Server((200, {"id": "r1", "model": "jev-latest", "answers": answers, "usage": {"input_tokens": 10}}))
        r = ask("jev", f"board {odd[8:12]}", role="decider", extra={"questions": questions}, http=server.client())
        record = json.loads(cache.path_for(r.request_hash).read_text(encoding="utf-8"))
        for text in (r.text, json.dumps(r.json), json.dumps(r.raw), record["text"], json.dumps(record)):
            assert odd not in text and escaped not in text
        assert odd not in json.dumps(json.loads(r.text), ensure_ascii=False)  # not even once the text is decoded
        assert r.text == json.dumps(r.json) and r.json["q"] == {"choice": "3", "echo": "[redacted]",
                                                                 "note": "seen [redacted]"}
        hit = ask("jev", f"board {odd[8:12]}", role="decider", extra={"questions": questions})
        assert hit.cache_hit and json.loads(hit.text)["q"]["echo"] == "[redacted]"
    # an answer quoting the key JSON-escaped (a model writing JSON) loses it as well, as do the scrubbing helpers
    monkeypatch.setenv("OPENROUTER_API_KEY", odd)
    r = ask("orm", "quote it", http=Server((200, or_response(content='{"echo": "' + escaped + '"}'))).client())
    assert r.text == '{"echo": "[redacted]"}'
    from avail.router.redact import has_secret, redact, strip_secret

    assert redact(f"x {escaped} y", odd) == "x [redacted] y" and strip_secret([escaped], odd) == ["[redacted]"]
    assert has_secret({"k": escaped}, odd)


# --------------------------------------------------------------------------- G5: no key in response metadata
def test_g5_a_key_echoed_in_response_metadata_is_cut_everywhere(ask, cache, monkeypatch):
    for key in (PLAIN_KEY, 'av-test-"meta"-' + "Q7" * 12):
        monkeypatch.setenv("OPENROUTER_API_KEY", key)
        escaped = json.dumps(key)[1:-1]
        rec = Recorder()
        body = or_response(gen_id=f"gen-{key}", model=f"vendor/{key}", provider=f"up-{key}", finish=f"stop {escaped}")
        r = ask("orm", f"meta {key[8:12]}", http=Server((200, body)).client(), events=rec)
        record = json.loads(cache.path_for(r.request_hash).read_text(encoding="utf-8"))
        assert r.model == "vendor/[redacted]" and r.finish_reason == "stop [redacted]"
        assert record["generation_id"] == "gen-[redacted]" and record["upstream_provider"] == "up-[redacted]"
        p = rec.events[0][1]
        assert (p["model"], p["finish_reason"], p["upstream_provider"]) == ("vendor/[redacted]", "stop [redacted]",
                                                                            "up-[redacted]")
        for text in (r.model, r.finish_reason, json.dumps(r.raw), json.dumps(rec.events), json.dumps(record)):
            assert key not in text and escaped not in text
        # a generation that failed: the error's generation id and details, and its event
        rec = Recorder()
        failed = or_response(finish=f"error {key}", gen_id=f"gen-{key}", cost=0.0004, prompt=100, completion=30)
        failed["choices"][0]["error"] = {"message": f"upstream said {key}"}
        with pytest.raises(ModelCallError) as err:
            ask("orm", f"fail {key[8:12]}", http=Server((200, failed)).client(), events=rec)
        assert err.value.generation_id == "gen-[redacted]"
        for text in (str(err.value), json.dumps(err.value.raw), json.dumps(rec.events)):
            assert key not in text and escaped not in text


# --------------------------------------------------------------------------- G6: absurd token counts
def test_g6_a_token_count_that_is_not_a_count_is_treated_as_missing(ask, ledger, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)

    def body_with(usage):  # the body as text, so a count can be any JSON number
        body = or_response(cost=None)
        body["usage"] = usage
        text = json.dumps(body)
        return lambda request: httpx.Response(200, content=text.encode())

    rec = Recorder()
    huge = {"prompt_tokens": 10 ** 400, "completion_tokens": 7, "cost": 0.001}
    r = ask("orm", "a huge prompt count", http=Server(body_with(huge)).client(), events=rec)
    assert r.text == '{"plugged": 3}' and r.input_tokens is None and r.output_tokens == 7
    assert r.usd == 0.001 and r.usd_source == "provider" and _settles(ledger)[-1]["usd"] == 0.001
    assert "input_tokens" in r.raw["usd_note"] and "not a token count" in r.raw["usd_note"]
    assert rec.events[0][1]["ok"] is True and rec.events[0][1]["input_tokens"] is None
    ModelCallPayload.model_validate(rec.events[0][1])
    # every count field. With no cost and nothing left to price, the call is counted at its projection. H6: a dropped
    # count and no usable cost settle it there (the cautious choice), with the note
    usage = {"prompt_tokens": 1e13, "completion_tokens": 10 ** 12,
             "completion_tokens_details": {"reasoning_tokens": 10 ** 30},
             "prompt_tokens_details": {"cached_tokens": 10 ** 400}}
    r2 = ask("orm", "every count huge", http=Server(body_with(usage)).client())
    assert (r2.input_tokens, r2.output_tokens, r2.reasoning_tokens, r2.cached_tokens) == (None, None, None, None)
    assert r2.usd == pytest.approx(r2.projected_usd) and r2.usd_source == "computed"
    settle = _settles(ledger)[-1]
    assert settle["id"] == _reserves(ledger)[-1]["id"] and settle["usd"] == pytest.approx(r2.projected_usd)
    assert "settled at its projection" in settle["note"] and "unreconciled_attempts" not in r2.raw
    # a cost too large for a float is no cost: the token counts price the call
    r3 = ask("orm", "a huge cost", http=Server(body_with({"prompt_tokens": 600, "completion_tokens": 3,
                                                          "cost": 10 ** 400})).client())
    assert r3.usd == pytest.approx(0.00123) and r3.usd_source == "computed"

    # after the provider answered, nothing but a RouterError leaves chat(): an answer the adapter cannot read
    # (its attempt stays counted at its projection), or a failure after the call such as the event sink's
    def unreadable(self, *a, **k):
        raise RecursionError("maximum recursion depth exceeded")

    with monkeypatch.context() as m:
        m.setattr(OpenRouterAdapter, "_success", unreadable)
        rec = Recorder()
        with pytest.raises(ModelCallError, match="could not be read") as err:
            ask("orm", "unreadable", http=Server((200, or_response())).client(), events=rec)
    assert err.value.status == 200 and not err.value.rejected_before_generation and err.value.__cause__ is None
    assert rec.events[0][1]["ok"] is False and rec.events[0][1]["unreconciled_attempts"] == 1
    assert _reserves(ledger)[-1]["id"] not in {s["id"] for s in _settles(ledger)}

    class BrokenSink:
        def emit(self, kind, payload):
            raise OSError("ledger disk full")

    with pytest.raises(RouterError, match="ledger disk full"):
        ask("orm", "sink broke", http=Server((200, or_response())).client(), events=BrokenSink())
    assert _settles(ledger)[-1]["usd"] == 0.00123  # settled before the event sink failed


# --------------------------------------------------------------------------- O1: no connection costs $0
def test_o1_an_attempt_that_cannot_connect_settles_at_zero(ask, ledger, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)

    def connect_timeout(request):
        raise httpx.ConnectTimeout("timed out while connecting", request=request)

    for failure in ("connect", connect_timeout):
        rec = Recorder()
        server = Server(failure)
        with pytest.raises(ModelCallError) as err:
            ask("orm", f"no network {failure!r}", max_tokens=16000, http=server.client(), events=rec)
        assert len(server.requests) == 3 and err.value.attempts == 3  # still attempts, still retried
        assert err.value.rejected_before_generation and "nothing reached the provider" in str(err.value)
        assert [e["outcome"] for e in err.value.raw["attempt_log"]] == ["not_sent"] * 3
        p = rec.events[0][1]
        assert (p["attempts"], p["reserved_attempts"], p["unreconciled_attempts"], p["usd"]) == (3, 3, 0, 0.0)
    assert len(_reserves(ledger)) == 6 and [s["usd"] for s in _settles(ledger)] == [0.0] * 6
    assert ledger.spent() == 0.0
    # a blip never fills the cap: each refused connection settles before the next attempt is reserved
    tight = recap(ledger, T=0.70)  # 0.30002 per attempt: two lost ones would leave no room for a third
    server = Server("connect")
    with pytest.raises(ModelCallError):
        ask("orm", "x" * 35, max_tokens=30000, http=server.client(), ledger=tight)
    assert len(server.requests) == 3 and ledger.spent() == 0.0
    server = Server("connect", (200, or_response()))
    r = ask("orm", "a blip, then the answer", http=server.client())
    assert len(server.requests) == 2 and r.usd == 0.00123 and "unreconciled_attempts" not in r.raw
    assert ledger.spent() == pytest.approx(0.00123)
    # could not connect, then refused: nothing was generated at all
    with pytest.raises(ModelCallError) as err:
        ask("orm", "a blip, then 402", http=Server("connect", (402, {"error": {"message": "no credits"}})).client())
    assert err.value.rejected_before_generation and ledger.spent() == pytest.approx(0.00123)
    # a read timeout or a connection dropped after the request was sent is still lost at its projection
    for lost in ("timeout", "drop"):
        before = ledger.spent()
        with pytest.raises(ModelCallError) as err:
            ask("orm", f"lost {lost}", http=Server(lost).client())
        assert not err.value.rejected_before_generation
        assert ledger.spent() - before == pytest.approx(3 * _reserves(ledger)[-1]["usd"])


# =========================================================================== loopkit round 3: H1 to H6
# the fence pattern parse_json_text used before H1 (it backtracks cubically), kept to check the scan that replaced it
OLD_FENCE_RE = re.compile(r"```[ \t]*[A-Za-z0-9_+-]*[ \t]*\r?\n?(.*?)```", re.DOTALL)
BACKSLASH = chr(92)


def _u_escaped(text: str, at: int = 0) -> str:
    """``text`` as JSON string content with the character at ``at`` written as a backslash-u escape."""
    return json.dumps(text[:at])[1:-1] + BACKSLASH + "u%04x" % ord(text[at]) + json.dumps(text[at + 1:])[1:-1]


def _with_key(key: str, *texts: str) -> list[str]:
    """The texts that hold ``key`` in any of the forms the router cuts from text (as written, JSON-escaped)."""
    from avail.router.redact import secret_forms

    return [t[:80] for t in texts if any(form in t for form in secret_forms(key))]


def _usage_body(usage: dict, **kw):
    """A handler whose answer reports ``usage`` byte for byte (so NaN and Infinity arrive as JSON tokens)."""
    body = or_response(**kw)
    body["usage"] = usage
    text = json.dumps(body)
    return lambda request: httpx.Response(200, content=text.encode())


# --------------------------------------------------------------------------- H1: the fence scan is linear
def test_h1_fenced_blocks_match_the_old_pattern_on_random_text():
    tick, cr, lf, tab = "`", "\r", "\n", "\t"
    alphabet = tick * 14 + (" " + tab) * 2 + "ajZ09" + (cr + lf) * 2 + "{}" + "\"'" + "_+-"
    pieces = [tick * 3, tick * 3, tick * 3 + "json", tick * 3 + "py ", tick * 3 + lf, tick * 2, tick, " ", tab,
              cr + lf, lf, cr, "{", "}", '"', "x", "1", "+", "json", " " + tab + " "]
    fixed = ["", tick * 3, tick * 4, tick * 5, tick * 6, tick * 7, tick * 3 + lf + tick * 3, "```json\r\n{}```",
             "```a b\n{}```", "``` \t py \t\r\n[1]```x```", "```\r```", "```\n```\n```", 'x```{"a": 1}```y```']
    rng = random.Random(20261017)
    texts = fixed + ["".join(rng.choice(alphabet) for _ in range(rng.randrange(40))) if i % 2 else
                     "".join(rng.choice(pieces) for _ in range(rng.randrange(20))) for i in range(5000)]
    counts = {"no block": 0, "one block": 0, "several blocks": 0}
    for text in texts:
        expected = [m.group(1) for m in OLD_FENCE_RE.finditer(text)]
        assert router_core._fenced_blocks(text) == expected, repr(text)
        counts["no block" if not expected else "one block" if len(expected) == 1 else "several blocks"] += 1
    assert min(counts.values()) > 500, counts  # every case was exercised, not only text without a block


def test_h1_a_fence_and_100000_spaces_or_tabs_parse_in_linear_time():
    # the old pattern took about 1.4 s for 1,000 spaces and 11.6 s for 2,000 (8 times longer per doubling). The
    # small size runs first, so a scan that is cubic again fails here in seconds instead of running for days
    for n, limit in ((1_000, 0.1), (100_000, 0.5)):
        for text in ("```" + " " * n + "x", "```" + "\t" * n + "x", "```" + " \t" * (n // 2) + "{",
                     "```json" + " " * n + "x", "```" + " " * (n // 2) + "json" + "\t" * (n // 2) + "[",
                     "``` " * (n // 4) + "x"):
            t0 = time.perf_counter()
            value, err = parse_json_text(text)
            elapsed = time.perf_counter() - t0
            assert value is None and err.startswith("not JSON") and elapsed < limit, (n, text[:10], elapsed)
    t0 = time.perf_counter()
    assert parse_json_text("```" + " " * n + "\n" + '{"a": 1}' + "\n```") == ({"a": 1}, None)
    assert time.perf_counter() - t0 < 0.5


# --------------------------------------------------------------------------- H2: a key in another JSON escape
def test_h2_a_key_in_another_json_escape_is_cut_from_the_parsed_answer(ask, cache, monkeypatch):
    long_key = "av-test-" + "Q7" * 36  # longer than the 60 characters a validation message quotes of a value
    quote_key = 'av-test-"quote"-' + "Q7" * 12
    cases = ((PLAIN_KEY, '{"plugged": 3, "' + _u_escaped(PLAIN_KEY) + '": "x"}'),  # as a field name
             (PLAIN_KEY, '{"plugged": "' + _u_escaped(PLAIN_KEY, 9) + '"}'),  # as a value of the wrong type
             (long_key, '{"plugged": "' + _u_escaped(long_key, 3) + '"}'),
             (quote_key, '{"plugged": 3, "echo": ["' + _u_escaped(quote_key, 8) + '"]}'))  # one quote escaped, one not
    for n, (key, content) in enumerate(cases):
        monkeypatch.setenv("OPENROUTER_API_KEY", key)
        assert not _with_key(key, content)  # no form the text scrub knows: only parsing turns it back into the key
        rec = Recorder()
        r = ask("orm", f"escaped {n}", json_schema=COUNT_SCHEMA, events=rec,
                http=Server((200, or_response(content=content))).client())
        record = cache.path_for(r.request_hash).read_text(encoding="utf-8")
        decoded = json.dumps(json.loads(r.text), ensure_ascii=False)
        assert not _with_key(key, r.text, decoded, json.dumps(r.json), r.json_error or "", json.dumps(rec.events),
                             record), n
        assert r.json_error and key[:16] not in r.json_error  # validated after the scrub: no half key either
        assert r.text == json.dumps(r.json) and "[redacted]" in r.text  # rebuilt from the scrubbed value
        hit = ask("orm", f"escaped {n}", json_schema=COUNT_SCHEMA)
        assert hit.cache_hit and (hit.text, hit.json, hit.json_error) == (r.text, r.json, r.json_error)
    assert r.json == {"plugged": 3, "echo": ["[redacted]"]}
    # an answer without the key is kept exactly as written
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    fenced = 'Sure:\n```json\n{"plugged": 3}\n```'
    r = ask("orm", "no key here", json_schema=COUNT_SCHEMA, http=Server((200, or_response(content=fenced))).client())
    assert r.text == fenced and r.json == {"plugged": 3} and r.json_error is None
    # a scrubbed value too deep to write back gives no text, never the original one, and nothing is cached
    deep: list = []
    for _ in range(100_000):
        deep = [deep]
    assert router_core._rebuilt_text(deep) == ""
    monkeypatch.setattr(router_core, "_rebuilt_text", lambda value: "")
    content = '{"plugged": "' + _u_escaped(PLAIN_KEY) + '"}'
    r = ask("orm", "cannot rebuild", json_schema=COUNT_SCHEMA,
            http=Server((200, or_response(content=content))).client())
    assert r.text == "" and r.json == {"plugged": "[redacted]"} and not cache.path_for(r.request_hash).exists()


def _decodes_to_key(key: str, value) -> bool:
    """True when ``value``, or any string inside it parsed as JSON, gives ``key`` back."""
    if isinstance(value, dict):
        return any(_decodes_to_key(key, k) or _decodes_to_key(key, v) for k, v in value.items())
    if isinstance(value, list):
        return any(_decodes_to_key(key, v) for v in value)
    if isinstance(value, str):
        if key in value:
            return True
        try:
            inner = json.loads(value)
        except (ValueError, RecursionError):
            return False
        return not isinstance(inner, str) and _decodes_to_key(key, inner)
    return False


def test_h2b_the_provider_body_is_left_out_when_the_answer_held_the_key(ask, cache, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    content = '{"plugged": 3, "echo": "' + _u_escaped(PLAIN_KEY, 4) + '"}'
    r = ask("orm", "escaped in the body", json_schema=COUNT_SCHEMA,
            http=Server((200, or_response(content=content))).client())
    record = json.loads(cache.path_for(r.request_hash).read_text(encoding="utf-8"))
    for where in (r.raw, record, r.text, r.json):
        assert not _decodes_to_key(PLAIN_KEY, where)
    assert "choices" not in r.raw and r.raw["provider_body"].startswith("left out")
    assert "choices" not in record["raw"] and r.raw["ledger_usd"] >= 0
    hit = ask("orm", "escaped in the body", json_schema=COUNT_SCHEMA)
    assert hit.cache_hit and not _decodes_to_key(PLAIN_KEY, hit.raw)
    # without a schema there is no parsed value to return, but the text must not decode to the key either
    r2 = ask("orm", "escaped, no schema", http=Server((200, or_response(content=content))).client())
    assert not _decodes_to_key(PLAIN_KEY, r2.text) and not _decodes_to_key(PLAIN_KEY, r2.raw)
    # an answer without the key keeps its provider body
    r3 = ask("orm", "no key", json_schema=COUNT_SCHEMA,
             http=Server((200, or_response(content='{"plugged": 3}'))).client())
    assert "provider_body" not in r3.raw and r3.text == '{"plugged": 3}'


# --------------------------------------------------------------------------- H3: redact, then cut to length
def test_h3_error_text_is_redacted_before_it_is_cut_to_length(ask, ledger, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    prefix = PLAIN_KEY[:12]  # what a cut through the key would have left
    words = "word " * 56  # 280 characters. Words, so the media scrub sees no base64 run to remove with the key

    def unreadable(self, *a, **k):  # the key straddles the 300-character cut
        raise RuntimeError(words + PLAIN_KEY + " and more")

    with monkeypatch.context() as m:
        m.setattr(OpenRouterAdapter, "_success", unreadable)
        rec = Recorder()
        with pytest.raises(ModelCallError, match="could not be read") as err:
            ask("orm", "unreadable", http=Server((200, or_response())).client(), events=rec)
    for text in (str(err.value), repr(err.value), json.dumps(rec.events)):
        assert prefix not in text
    assert "[redacted]" in str(err.value)

    class BrokenSink:
        def emit(self, kind, payload):
            raise OSError(words + PLAIN_KEY)

    with pytest.raises(RouterError, match="failed after the call") as err:
        ask("orm", "sink broke", http=Server((200, or_response())).client(), events=BrokenSink())
    assert prefix not in str(err.value) and "[redacted]" in str(err.value)

    def broken_cost(*a, **k):  # the key straddles the 160-character cut of a cost note
        raise ArithmeticError(words[:130] + PLAIN_KEY)

    monkeypatch.setattr(router_core, "settled_cost", broken_cost)
    r = ask("orm", "cost broke", http=Server((200, or_response())).client())
    assert "could not be computed" in r.raw["usd_note"] and prefix not in r.raw["usd_note"]


# --------------------------------------------------------------------------- H4: the port is 1 to 65535
def test_h4_a_port_outside_1_to_65535_is_not_configured_before_any_spend(ask, ledger, monkeypatch):
    for var in ("ELORIAN_API_KEY", "OPENROUTER_API_KEY", "AV_TEST_SYS1_KEY"):
        monkeypatch.setenv(var, PLAIN_KEY)
    jev_kw = {"role": "decider", "extra": {"questions": {"q": {"type": "noul", "instructions": "x"}}}}
    server = Server((200, or_response()))
    for key, field, url in (("el", "base_url", "https://elorian.invalid:{}/v1"),
                            ("orm", "base_url", "https://router.invalid:{}/api/v1"),
                            ("sys1_vision", "endpoint", "https://decisions.invalid:{}/systemone")):
        for port in ("0", "65536", "99999", "-1", "1" + "0" * 30):  # httpx parses each. 99999 resolves to 34463
            models = _models()
            setattr(models.models[key], field, url.format(port))
            with pytest.raises(ModelNotConfigured, match="port") as err:
                ask(key, "hi", models=models, http=server.client(), **(jev_kw if key == "sys1_vision" else {}))
            assert field in str(err.value) and "from 1 to 65535" in str(err.value)
    assert not server.requests and ledger.records() == []  # refused before any reservation
    for port in (1, 65535):  # the ends of the range are ports, and the request goes to the one written
        models = _models()
        models.models["el"].base_url = f"https://elorian.invalid:{port}/v1"
        ask("el", f"port {port}", models=models, http=server.client())
    assert [r.url.port for r in server.requests] == [1, 65535]


# --------------------------------------------------------------------------- H5: every dropped count is noted
def test_h5_every_token_count_that_is_not_a_count_gets_a_note(ask, ledger, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    junk = (-5, 2.5, -3.0, "600", True, [600], {"n": 600}, float("nan"), float("inf"), float("-inf"), PLAIN_KEY)
    for n, value in enumerate(junk):  # not cached: a raw body holding NaN is not plain JSON (the cache skips it)
        r = ask("orm", f"junk {n}", use_cache=False,
                http=Server(_usage_body({"prompt_tokens": value, "completion_tokens": 3, "cost": 0.00123})).client())
        assert r.input_tokens is None and r.output_tokens == 3, value
        assert r.usd == 0.00123 and r.usd_source == "provider"  # a usable cost is still taken
        for note in (r.raw["usd_note"], _settles(ledger)[-1]["note"]):
            assert "the provider's input_tokens (" in note and "not a token count" in note, (value, note)
            assert PLAIN_KEY not in note and "600" not in note  # a string or a list is described, never quoted
    for name, usage in (("output_tokens", {"completion_tokens": "3"}),
                        ("reasoning_tokens", {"completion_tokens": 3,
                                              "completion_tokens_details": {"reasoning_tokens": -1}}),
                        ("cached_tokens", {"completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": [9]}})):
        r = ask("orm", f"field {name}", http=Server(_usage_body({"prompt_tokens": 600, "cost": 0.00123,
                                                                 **usage})).client())
        assert getattr(r, name) is None and f"the provider's {name} (" in r.raw["usd_note"]
        assert f"the provider's {name} (" in _settles(ledger)[-1]["note"]
    # the jev adapter passes its counts as reported too
    monkeypatch.setenv("TYPESAFE_JEV_API_KEY", PLAIN_KEY)
    jev = Server((200, {"answers": {"q": {"type": "noul", "noul": 0.5}}, "usage": {"input_tokens": "1000"}}))
    r = ask("jev", "state", role="decider", extra={"questions": {"q": {"type": "noul", "instructions": "x"}}},
            http=jev.client())
    assert r.input_tokens is None and "the provider's input_tokens (a value of type str)" in r.raw["usd_note"]
    # and so does a generation that failed upstream: its settle note names the count it dropped
    failed = or_response(finish="error", cost=0.0004, prompt=-5, completion=30)
    with pytest.raises(ModelCallError, match="failed upstream"):
        ask("orm", "failed with a count that is not one", http=Server((200, failed)).client())
    settle = _settles(ledger)[-1]
    assert settle["usd"] == 0.0004 and "the provider's input_tokens (-5) is not a token count" in settle["note"]


# --------------------------------------------------------------------------- H6: a dropped count and no cost
def test_h6_a_dropped_count_without_a_usable_cost_settles_at_the_projection(ask, ledger, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    for n, usage in enumerate(({"prompt_tokens": -5, "completion_tokens": 3},  # no cost at all
                               {"prompt_tokens": 600, "completion_tokens": 2.5, "cost": "free"},  # a cost that is none
                               {"prompt_tokens": 600, "completion_tokens": 3,
                                "completion_tokens_details": {"reasoning_tokens": "many"}})):
        r = ask("orm", f"dropped {n}", http=Server(_usage_body(usage)).client())
        settle = _settles(ledger)[-1]
        assert r.usd == pytest.approx(r.projected_usd) and r.usd_source == "computed"
        assert settle["id"] == _reserves(ledger)[-1]["id"] and settle["usd"] == pytest.approx(r.projected_usd)
        for note in (r.raw["usd_note"], settle["note"]):
            assert "not a token count" in note and "settled at its projection" in note, note
        assert "unreconciled_attempts" not in r.raw  # settled: the call's cost is no longer open
    # a usable cost is taken as before, checked only against the counts that were valid
    priced = {"prompt_tokens": "600", "completion_tokens": 3, "cost": 0.001}
    r = ask("orm", "dropped but priced", http=Server(_usage_body(priced)).client())
    assert r.usd == 0.001 and r.usd_source == "provider" and "projection" not in r.raw["usd_note"]
    low = {"prompt_tokens": "600", "completion_tokens": 15000, "cost": 0.0}  # below half of what 15000 out cost
    r = ask("orm", "x" * 35, max_tokens=16000, http=Server(_usage_body(low)).client())
    assert r.usd_source == "computed" and r.usd == pytest.approx((10 * 2.0 + 15000 * 10.0) / 1e6)
    # a count that was not reported (null) is not a dropped one: the others still price the call
    r = ask("orm", "hi", http=Server(_usage_body({"prompt_tokens": 600, "completion_tokens": None})).client())
    assert r.usd == pytest.approx((600 * 2.0 + DEFAULT_MAX_TOKENS * 10.0) / 1e6)
    assert r.usd != pytest.approx(r.projected_usd)
    assert "not a token count" not in r.raw["usd_note"]


# =========================================================================== new for AVAIL
# --------------------------------------------------------------------------- spend sessions and caps from the config
def test_caps_come_from_spend_caps_usd_in_experiment_yaml():
    # The caps Kevin set on 2026-11-11. A cap is changed in the config by him, and here in the same commit:
    # this test is what notices a cap that moved by accident
    c = load_spend_caps()
    assert c.total_usd == 15.0  # build_total
    assert dict(c.sessions) == {"S1": 1.0, "S3": 8.0, "S4_without_ok": 5.0, "full_judge_run": 40.0, "S4b_judges": 30.0,
                                "S5_vlm_arm": 10.0, "E3_vlm_arm_latch": 0.0, "E3_vlm_arm_fric": 0.0}
    assert c.outside_total == frozenset({"full_judge_run", "S4b_judges", "E3_vlm_arm_latch", "E3_vlm_arm_fric"})
    assert c.needs_approval("full_judge_run") and not c.needs_approval("S5_vlm_arm")
    assert c.needs_approval("E3_vlm_arm_latch")        # each variant's E3 judge counted on its own: 0 until Kevin sets it
    assert c.needs_approval("S4b_judges")              # Kevin's OK of 2026-11-11 for the hard-mode judge run, up to $30
    assert c.source == str(config.EXPERIMENT_YAML)
    raw = yaml.safe_load(config.EXPERIMENT_YAML.read_text(encoding="utf-8"))["spend_caps_usd"]
    assert c.total_usd == raw["build_total"] and dict(c.sessions) == {k: v for k, v in raw.items()
                                                                      if k != "build_total"}


def test_router_session_makes_that_sessions_cap_active(tmp_path):
    for session, cap in (("S1", 1.0), ("S3", 8.0), ("S4_without_ok", 5.0), ("full_judge_run", 40.0)):
        router = Router(session=session)  # the caps of the real config
        assert router.session == session and router.cap_usd == cap and router.caps.total_usd == 15.0
        assert router.spent() == 0.0 and router.remaining() == pytest.approx(cap if session == "full_judge_run"
                                                                             else min(cap, 15.0))
    assert Router().session is None and Router().cap_usd is None
    # the cap that binds is the active session's: the same call is refused in S1 and admitted in S3
    led_path = tmp_path / "data" / "spend_ledger.jsonl"
    with pytest.raises(SpendCapExceeded) as err:
        Router(session="S1", models=MODELS).chat("paid_fake", "x" * 35, max_tokens=30000)  # projects $1.5001
    assert (err.value.scope, err.value.session, err.value.cap_usd) == ("session", "S1", 1.0)
    assert not led_path.exists() or SpendLedger().records() == []
    r = Router(session="S3", models=MODELS).chat("paid_fake", "x" * 35, max_tokens=30000)
    assert r.session == "S3" and r.projected_usd == pytest.approx((10 * 10.0 + 30000 * 50.0) / 1e6)
    assert [rec["session"] for rec in _reserves(SpendLedger())] == ["S3"] and led_path.is_file()
    assert Router(session="S3").spent() == pytest.approx(r.usd) and Router(session="S1").spent() == 0.0


def test_an_unknown_session_name_is_an_error(ask, ledger, cache, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    server = Server((200, or_response()))
    for bad in ("S9", "s3", "S3 ", "S4", "build_total"):
        with pytest.raises(UnknownSessionError) as err:
            Router(session=bad)  # the real config
        assert "S1" in str(err.value) and "full_judge_run" in str(err.value) and repr(bad) in str(err.value)
        assert isinstance(err.value, RouterError)
    with pytest.raises(UnknownSessionError, match="total cap"):
        Router(session="build_total")  # the total is a cap, never a session
    with pytest.raises(UnknownSessionError):
        ask("orm", "hi", session="nope", http=server.client())
    with pytest.raises(UnknownSessionError):
        ask("fake", "hi", session="nope")  # even a free call
    with pytest.raises(UnknownSessionError):
        Router(session="nope", ledger=ledger)
    with pytest.raises(UnknownSessionError):
        ledger.reserve(0.01, "orm", "a" * 64, session="nope")
    with pytest.raises(UnknownSessionError):
        ledger.reserve(0.0, "fake", "a" * 64, session="nope", paid=False)
    with pytest.raises(UnknownSessionError):
        ledger.remaining("nope")
    with pytest.raises(TypeError):
        Router(session=3)
    assert not server.requests and ledger.records() == [] and not list(cache.root.rglob("*.json"))


def test_session_cap_refuses_a_paid_call_before_any_network_io(cache, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    led = SpendLedger(tmp_path / "s" / "spend_ledger.jsonl", caps(total=15.0, S3=0.05))
    # the transport fails the test if the router calls it
    router = Router(session="S3", models=MODELS, ledger=led, cache=cache, http=refusing_client())
    rec = Recorder()
    with pytest.raises(SpendCapExceeded) as err:
        router.chat("orm", "x" * 35, images=[board_array()], json_schema=COUNT_SCHEMA, max_tokens=6000, events=rec)
    e = err.value
    assert (e.scope, e.session, e.cap_usd, e.spent_usd) == ("session", "S3", 0.05, 0.0)
    assert e.projected_usd > 0.05 and "S3" in str(e) and "$0.0500" in str(e)
    assert led.records() == [] and rec.events == [] and not list(cache.root.rglob("*.json"))
    assert router.spent() == 0.0 and router.remaining() == pytest.approx(0.05)
    # a call that fits is admitted by the same session, and the refusal left nothing behind to block it
    server = Server((200, or_response()))
    ok = Router(session="S3", models=MODELS, ledger=led, cache=cache, http=server.client())
    r = ok.chat("orm", "x" * 35, images=[board_array()], json_schema=COUNT_SCHEMA, max_tokens=600)
    assert r.usd == 0.00123 and len(server.requests) == 1 and ok.spent() == pytest.approx(0.00123)
    # once the session's spend reaches its cap, the next call is refused without network I/O as well
    full = SpendLedger(led.path, caps(total=15.0, S3=0.00123))
    with pytest.raises(SpendCapExceeded) as err:
        Router(session="S3", models=MODELS, ledger=full, cache=cache, http=refusing_client()).chat(
            "orm", "one more", max_tokens=10)
    assert err.value.scope == "session" and err.value.spent_usd == pytest.approx(0.00123)


def test_total_cap_refuses_a_paid_call_before_any_network_io(cache, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    led = SpendLedger(tmp_path / "t" / "spend_ledger.jsonl", caps(total=0.10, S1=0.08, S3=0.08))
    paid = Server((200, or_response(cost=0.07, prompt=20000, completion=3000)))  # orm: 20000 * 2 + 3000 * 10 = $0.07
    s1 = Router(session="S1", models=MODELS, ledger=led, cache=cache, http=paid.client())
    first = s1.chat("orm", "first", max_tokens=3000)
    assert first.usd == 0.07 and led.spent("S1") == pytest.approx(0.07)
    s3 = Router(session="S3", models=MODELS, ledger=led, cache=cache, http=refusing_client())
    rec = Recorder()
    with pytest.raises(SpendCapExceeded) as err:
        s3.chat("orm", "second", max_tokens=4000, events=rec)  # projects $0.04: inside S3's own $0.08, over the total
    e = err.value
    assert (e.scope, e.session, e.cap_usd) == ("total", "S3", 0.10) and e.spent_usd == pytest.approx(0.07)
    assert e.projected_usd == pytest.approx((2 * 2.0 + 4000 * 10.0) / 1e6) and "Total cap" in str(e)
    assert led.spent("S3") == 0.0 and len(_reserves(led)) == 1 and rec.events == []
    assert s3.remaining() == pytest.approx(0.03)  # the smaller of S3's own room (0.08) and the total's (0.03)
    small = Server((200, or_response()))
    r = s3.chat("orm", "small", max_tokens=1000, http=small.client())  # projects $0.01: fits both
    assert r.usd == 0.00123 and led.spent_in_total() == pytest.approx(0.07123)


def test_the_actual_cost_replaces_the_reservation(ask, ledger, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    seen = {}

    def answer(request):  # while the call is in flight the spend ledger counts its projection
        seen["spent"] = ledger.spent("T")
        seen["remaining"] = ledger.remaining("T")
        seen["records"] = [r["type"] for r in ledger.records()]
        return httpx.Response(200, json=or_response())

    r = ask("orm", "x" * 35, max_tokens=20000, http=Server(answer).client())
    projected = (10 * 2.0 + 20000 * 10.0) / 1e6  # $0.20002
    assert r.projected_usd == pytest.approx(projected) and seen["spent"] == pytest.approx(projected)
    assert seen["remaining"] == pytest.approx(1.0 - projected) and seen["records"] == ["reserve"]
    # after the call the measured cost stands in its place
    assert r.usd == 0.00123 and ledger.spent("T") == pytest.approx(0.00123)
    assert ledger.remaining("T") == pytest.approx(1.0 - 0.00123)
    reserve, settle = ledger.records()
    assert (reserve["type"], reserve["session"], reserve["model_key"]) == ("reserve", "T", "orm")
    assert reserve["usd"] == pytest.approx(projected) and reserve["request_hash"] == r.request_hash
    assert (settle["type"], settle["id"], settle["usd"], settle["usd_source"]) == (
        "settle", reserve["id"], 0.00123, "provider")
    assert ledger.summary()["unsettled"] == {"count": 0, "usd": 0.0}


def test_a_cached_answer_costs_zero_and_is_marked_cached(ask, ledger, cache, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    kw = dict(images=[board_array()], json_schema=COUNT_SCHEMA, max_tokens=300)
    first = ask("orm", "Is port 7 seated?", http=Server((200, or_response())).client(), **kw)
    assert not first.cached and first.usd == 0.00123
    records_before = ledger.path.read_bytes()
    monkeypatch.delenv("OPENROUTER_API_KEY")  # no key from here on
    rec = Recorder()
    hit = ask("orm", "Is port 7 seated?", http=refusing_client(), events=rec, **kw)  # and no network
    assert hit.cached is True and hit.cache_hit is True and hit.usd == 0.0 and hit.usd_source == "computed"
    assert (hit.text, hit.json, hit.request_hash) == (first.text, first.json, first.request_hash)
    assert hit.raw["original_usd"] == 0.00123 and hit.raw["original_usd_source"] == "provider"
    assert ledger.path.read_bytes() == records_before and ledger.spent() == pytest.approx(0.00123)
    (kind, p), = rec.events
    assert (kind, p["cache_hit"], p["usd"], p["attempts"], p["ledger_usd"]) == ("model_call", True, 0.0, 0, 0.0)
    ModelCallPayload.model_validate(p)
    # what the answer cost and how long it took when it was computed stays on the hit, for an honest report
    assert hit.original_usd == 0.00123 and first.original_usd == first.usd == 0.00123
    assert hit.original_latency_ms == pytest.approx(first.latency_ms, abs=0.001) and hit.latency_ms != first.latency_ms
    assert first.original_latency_ms == first.latency_ms
    assert p["original_usd"] == 0.00123 and p["original_latency_ms"] == pytest.approx(first.latency_ms, abs=0.001)
    # a hit is served in any session and without one, even when the session's cap is used up
    assert ask("orm", "Is port 7 seated?", session=None, http=refusing_client(), **kw).cached
    used_up = recap(ledger, T=0.001)
    assert ask("orm", "Is port 7 seated?", ledger=used_up, http=refusing_client(), **kw).usd == 0.0
    # the miss next to it is still a paid call: no key, so it is refused, and nothing was spent on it
    with pytest.raises(MissingCredentialError):
        ask("orm", "Is port 8 seated?", http=refusing_client(), **kw)
    assert ledger.path.read_bytes() == records_before


def test_the_fake_model_costs_zero_and_needs_no_key_session_or_network(tmp_path, cache):
    assert not any(config.env(name) for name in KEY_VARS)  # no key anywhere
    led = SpendLedger(tmp_path / "fake" / "spend_ledger.jsonl", caps(total=15.0, S3=8.0))
    router = Router(models=MODELS, ledger=led, cache=cache, http=refusing_client())  # no session
    rec = Recorder()
    r = router.chat("fake", "Is the plug seated?", images=[board_array(), board_array()], json_schema=JUDGE_SCHEMA,
                    role="judge", events=rec)
    assert r.usd == 0.0 and r.projected_usd == 0.0 and not r.cached and r.provider == "fake" and r.session is None
    assert r.json == {"holds": True, "confidence": 0.5, "reason": "fake"} and r.json_ok and r.image_count == 2
    assert r.raw["fake"] is True and "placeholder" in r.raw["warning"]
    assert estimate_usd("fake", "Is the plug seated?", image_count=2, json_schema=JUDGE_SCHEMA, models=MODELS) == 0.0
    assert led.spent() == 0.0 and [x["usd"] for x in led.records()] == [0.0, 0.0]
    ModelCallPayload.model_validate(rec.events[0][1])
    # a script makes it answer as a judge would, still for $0
    with fake_script(lambda req: {"holds": req.image_count == 2, "confidence": 0.9, "reason": "scripted"}):
        scripted = router.chat("fake", "Is the plug seated?", images=[board_array(), board_array()],
                               json_schema=JUDGE_SCHEMA)
    assert scripted.json == {"holds": True, "confidence": 0.9, "reason": "scripted"} and scripted.usd == 0.0
    # the same router refuses every paid call: it has no session
    with pytest.raises(SpendCapExceeded) as err:
        router.chat("paid_fake", "hello")
    assert err.value.scope == "no_session"
    # inside a session the fake is free too, and the session's spend stays 0
    in_session = Router(session="S3", models=MODELS, ledger=led, cache=cache, http=refusing_client())
    assert in_session.chat("fake", "again", use_cache=False).usd == 0.0 and in_session.spent() == 0.0


def test_full_judge_run_needs_kevins_ok_and_is_counted_outside_the_build_total(ask, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    led = SpendLedger(tmp_path / "fj" / "spend_ledger.jsonl",
                      caps(total=0.05, outside=("full_judge_run",), S3=0.04, full_judge_run=0.20))
    server = Server((200, or_response(cost=0.03, prompt=10000, completion=1000)))  # orm: 10000 * 2 + 1000 * 10
    run = dict(max_tokens=1000, ledger=led, session="full_judge_run")
    # without the OK: refused before any network I/O, and nothing is written
    with pytest.raises(SessionNotApproved) as err:
        ask("orm", "judge it", http=refusing_client(), **run)
    assert "approved=True" in str(err.value) and "Kevin" in str(err.value) and led.records() == []
    assert isinstance(err.value, RouterError)
    # with it: charged to its own cap, and its spend may pass the build total
    for n in (1, 2):
        ask("orm", f"judge {n}", http=server.client(), approved=True, **run)
    assert led.spent("full_judge_run") == pytest.approx(0.06) and led.spent_in_total() == 0.0
    assert led.remaining("full_judge_run") == pytest.approx(0.14)
    assert all(rec["tags"]["approved"] is True for rec in _reserves(led))
    # a build session is not charged for it: S3 still has its whole room under the build total
    assert led.remaining("S3") == pytest.approx(0.04)
    r = ask("orm", "a build call", max_tokens=1000, ledger=led, session="S3", http=server.client())
    assert r.usd == 0.03 and led.spent_in_total() == pytest.approx(0.03)
    assert "approved" not in _reserves(led)[-1]["tags"]
    assert led.summary()["total_usd"] == pytest.approx(0.09) and led.summary()["in_total_usd"] == pytest.approx(0.03)
    # and the build's spend is not charged to it: $0.03 of build spend plus this $0.03 projection is over the
    # build total of $0.05, and the call is admitted all the same
    big = ask("orm", "judge 3", http=server.client(), approved=True, **dict(run, max_tokens=3000))
    assert big.projected_usd == pytest.approx((2 * 2.0 + 3000 * 10.0) / 1e6) and big.usd == 0.03
    assert led.spent("full_judge_run") == pytest.approx(0.09) and led.spent_in_total() == pytest.approx(0.03)
    # its own cap still binds
    tighter = SpendLedger(led.path, caps(total=0.05, outside=("full_judge_run",), S3=0.04, full_judge_run=0.095))
    with pytest.raises(SpendCapExceeded) as err:
        ask("orm", "judge 4", http=refusing_client(), approved=True, max_tokens=1000, ledger=tighter,
            session="full_judge_run")
    assert (err.value.scope, err.value.session) == ("session", "full_judge_run")
    # a free call and a cached answer need no OK
    assert ask("fake", "dry run", ledger=led, session="full_judge_run").usd == 0.0
    assert ask("orm", "judge 1", http=refusing_client(), **run).cached
    # the real config: the session exists, opens without the OK, and a paid call in it is refused
    router = Router(session="full_judge_run", models=MODELS, http=refusing_client())
    assert router.cap_usd == 40.0 and router.approved is False
    with pytest.raises(SessionNotApproved):
        router.chat("orm", "the full run", max_tokens=100)
    assert not (tmp_path / "data" / "spend_ledger.jsonl").exists()


def test_router_passes_its_session_and_refuses_overrides(ledger, cache):
    router = Router(session="T", models=MODELS, ledger=ledger, cache=cache, phase="zoo")
    rec = Recorder()
    r = router.chat("paid_fake", "hello", max_tokens=100, events=rec)
    assert r.session == "T" and rec.events[0][1]["session"] == "T" and rec.events[0][1]["phase"] == "zoo"
    assert _reserves(ledger)[0]["session"] == "T" and _reserves(ledger)[0]["tags"]["phase"] == "zoo"
    assert router.spent() == pytest.approx(r.usd) and router.cap_usd == 1.0
    assert router.remaining() == pytest.approx(1.0 - r.usd)
    assert router.summary()["by_session"] == {"T": pytest.approx(r.usd)}
    for name in ("session", "approved", "ledger"):
        with pytest.raises(TypeError, match="belongs to the Router"):
            router.chat("fake", "hi", **{name: None})
    with pytest.raises(ValueError):
        Router(session="T", ledger=ledger, caps=caps(T=5.0))
    assert repr(router) == "Router(session='T', approved=False)"
    none = Router(models=MODELS, ledger=ledger, cache=cache)
    assert none.session is None and none.cap_usd is None and none.spent() == 0.0 and none.remaining() == 0.0
    # caps passed to a Router (a test or a dry run) are the ones its own ledger enforces
    own = Router(session="A", caps=caps(A=0.001), models=MODELS, cache=cache)
    assert own.cap_usd == 0.001 and own.ledger.caps.session_cap("A") == 0.001
    assert own.chat("paid_fake", "hello", max_tokens=100).cached  # the answer above, from the cache: free
    with pytest.raises(SpendCapExceeded) as err:
        own.chat("paid_fake", "hello, a new request", max_tokens=100)
    assert (err.value.scope, err.value.cap_usd, err.value.session) == ("session", 0.001, "A")


CONCURRENT_WORKER = """
import json, os, sys, time
from avail.router.errors import SpendCapExceeded
from avail.router.spend import SpendCaps, SpendLedger

path, session, attempts, go, ready = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
ledger = SpendLedger(path, SpendCaps(total_usd=1.0, sessions={"A": 0.75, "B": 0.75}))
open(ready, "w").close()
deadline = time.monotonic() + 60
while not os.path.exists(go):
    if time.monotonic() > deadline:
        sys.exit(9)
    time.sleep(0.002)
admitted, refused = 0, {}
for i in range(attempts):
    try:
        rid = ledger.reserve(0.0625, "orm", "c" * 64, session=session, tags={"pid": os.getpid()})
        admitted += 1
        if i % 3 == 0:
            ledger.settle(rid, 0.0625, "provider")
    except SpendCapExceeded as exc:
        refused[exc.scope] = refused.get(exc.scope, 0) + 1
print(json.dumps({"admitted": admitted, "refused": refused}))
"""


def test_concurrent_processes_never_cross_a_cap(tmp_path):
    """4 processes race for $1.00 in $0.0625 steps: exactly 16 reservations fit, whatever the interleaving."""
    path = tmp_path / "race" / "spend_ledger.jsonl"
    path.parent.mkdir(parents=True)
    go = tmp_path / "race" / "go"
    procs = []
    for n, session in enumerate(("A", "A", "B", "B")):
        ready = tmp_path / "race" / f"ready_{n}"
        procs.append((ready, subprocess.Popen(
            [sys.executable, "-c", CONCURRENT_WORKER, str(path), session, "14", str(go), str(ready)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=REPO)))
    deadline = time.monotonic() + 60
    while not all(ready.exists() for ready, _ in procs):
        assert time.monotonic() < deadline, "the workers did not start"
        assert all(proc.poll() is None for _, proc in procs), [proc.stderr.read()[-400:] for _, proc in procs]
        time.sleep(0.01)
    go.write_text("go", encoding="ascii")  # all 4 start reserving at once
    results = []
    for _, proc in procs:
        out, err = proc.communicate(timeout=120)
        assert proc.returncode == 0, err[-800:]
        results.append(json.loads(out.strip().splitlines()[-1]))
    ledger = SpendLedger(path, caps(total=1.0, A=0.75, B=0.75))
    reserves = _reserves(ledger)
    assert sum(r["admitted"] for r in results) == 16 == len(reserves)  # 16 * 0.0625 = the total cap, never 17
    assert ledger.spent() == 1.0 and ledger.spent_in_total() == 1.0
    assert ledger.spent("A") <= 0.75 and ledger.spent("B") <= 0.75 and ledger.spent("A") + ledger.spent("B") == 1.0
    assert sum(sum(r["refused"].values()) for r in results) == 4 * 14 - 16
    assert {scope for r in results for scope in r["refused"]} <= {"session", "total"}
    summary = ledger.summary()
    assert summary["corrupt_lines"] == 0 and summary["orphan_settles"]["count"] == 0  # no torn or lost line
    assert len(ledger.path.read_bytes().splitlines()) == len(ledger.records())
    assert len({r["id"] for r in reserves}) == 16 and len({r["tags"]["pid"] for r in reserves}) >= 1
    with pytest.raises(SpendCapExceeded):  # and the cap stays reached for the next process
        ledger.reserve(0.0625, "orm", "c" * 64, session="A")


# --------------------------------------------------------------------------- the judge call shape
def test_a_judge_call_sends_1_to_3_images_in_order(ask, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    views = [board_array(), board_array(), board_array()]
    views[1][0, 0] = (9, 9, 9)
    views[2][1, 1] = (7, 7, 7)
    hashes = []
    for n in (1, 2, 3):
        server = Server((200, or_response(content='{"holds": true, "confidence": 0.9, "reason": "flush"}')))
        r = ask("orm", [{"role": "system", "content": "You judge connectors."},
                        {"role": "user", "content": "Is the plug seated in port 7?"}],
                images=views[:n], json_schema=JUDGE_SCHEMA, role="judge", max_tokens=300, http=server.client())
        parts = server.bodies()[0]["messages"][1]["content"]
        sent = [p for p in parts if p["type"] == "image_url"]
        assert len(sent) == n and r.image_count == n
        assert parts[0] == {"type": "text", "text": "Is the plug seated in port 7?"}  # the text, then the views
        for part, view in zip(sent, views):
            decoded = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
            assert np.array_equal(np.asarray(Image.open(io.BytesIO(decoded))), view)
        assert r.json == {"holds": True, "confidence": 0.9, "reason": "flush"} and r.json_ok and r.role == "judge"
        assert r.est_input_tokens >= n * 1600  # every image is in the projection
        hashes.append(r.request_hash)
    assert len(set(hashes)) == 3  # 1, 2 and 3 views are 3 different requests
    with pytest.raises(ValueError, match="no user message"):
        ask("orm", [{"role": "system", "content": "only a system message"}], images=views[:1])


def test_request_hash_covers_image_bytes_prompt_model_id_and_generation_settings(ask):
    a, b = board_array(), board_array()
    b[0, 0] = (1, 2, 3)  # 1 pixel differs
    prompt = "Is the plug seated?"
    base = dict(images=[a], json_schema=COUNT_SCHEMA, max_tokens=300, temperature=0.0, reasoning="disabled")
    h0 = ask("fake", prompt, **base).request_hash
    renamed = _models()
    renamed.models["fake"].id = "fake-2"
    variants = {
        "image bytes": ask("fake", prompt, **dict(base, images=[b])),
        "one more image": ask("fake", prompt, **dict(base, images=[a, b])),
        "image order": ask("fake", prompt, **dict(base, images=[b, a])),
        "no image": ask("fake", prompt, **dict(base, images=None)),
        "prompt": ask("fake", "Is the plug seated ?", **base),
        "system prompt": ask("fake", [{"role": "system", "content": "You judge connectors."},
                                      {"role": "user", "content": prompt}], **base),
        "model id": ask("fake", prompt, models=renamed, **base),
        "max_tokens": ask("fake", prompt, **dict(base, max_tokens=301)),
        "temperature": ask("fake", prompt, **dict(base, temperature=0.2)),
        "no temperature": ask("fake", prompt, **dict(base, temperature=None)),
        "reasoning": ask("fake", prompt, **dict(base, reasoning="low")),
        "schema": ask("fake", prompt, **dict(base, json_schema=JUDGE_SCHEMA)),
        "schema name": ask("fake", prompt, schema_name="verdict", **base),
        "cache salt": ask("fake", prompt, cache_salt="repeat-1", **base),
        "extra": ask("fake", prompt, extra={"body": {"top_logprobs": 5}}, **base),
    }
    assert not any(r.cached for r in variants.values())
    all_hashes = [h0] + [r.request_hash for r in variants.values()]
    assert len(set(all_hashes)) == len(all_hashes), [name for name, r in variants.items() if r.request_hash == h0]
    # the same request again is the same hash and a cache hit, whatever form the same pixels come in
    for same in (a.copy(), Image.fromarray(a), normalize_image(a)):
        again = ask("fake", prompt, **dict(base, images=[same]))
        assert again.request_hash == h0 and again.cached and again.usd == 0.0
    # the role and the session are recorded with the call and are not part of the request
    assert ask("fake", prompt, role="test", session=None, **base).request_hash == h0


def test_json_answers_are_requested_with_reasoning_disabled_and_usage_included(ask, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    answer = '{"holds": false, "confidence": 0.8, "reason": "a gap is visible"}'
    server = Server((200, or_response(content=answer)))
    r = ask("orm", "Is the plug seated?", images=[board_array()], json_schema=JUDGE_SCHEMA, http=server.client())
    body = server.bodies()[0]  # every argument at its default
    assert body["reasoning"] == {"enabled": False} and body["usage"] == {"include": True}
    assert body["max_tokens"] == DEFAULT_MAX_TOKENS == 1024
    assert r.reasoning_setting == {"enabled": False} and r.role == "judge" and "reasoning_note" not in r.raw
    assert r.json == {"holds": False, "confidence": 0.8, "reason": "a gap is visible"} and r.usd_source == "provider"
    # every JSON mode of the fallback chain keeps both
    refuse_schema = (400, {"error": {"message": "response_format json_schema is not supported"}})
    refuse_object = (400, {"error": {"message": "json_object mode is not supported"}})
    server = Server(refuse_schema, refuse_object, (200, or_response(content=answer)))
    r = ask("orm", "Is the plug seated? Again.", json_schema=JUDGE_SCHEMA, http=server.client())
    assert r.json_mode == "plain_schema_in_prompt" and len(server.requests) == 3
    for body in server.bodies():
        assert body["reasoning"] == {"enabled": False} and body["usage"] == {"include": True}
    # asking for reasoning is still possible, and is sent as asked
    server = Server((200, or_response(content=answer)))
    ask("orm", "Think first.", json_schema=JUDGE_SCHEMA, reasoning="high", http=server.client())
    assert server.bodies()[0]["reasoning"] == {"effort": "high"}


def test_a_model_that_always_reasons_gets_its_lowest_effort_when_reasoning_is_disabled(ask, monkeypatch):
    must = MODELS.get("orm_must")
    assert must.reasoning_mandatory and must.lowest_effort == "low"
    assert normalize_reasoning("disabled", must) == {"effort": "low"}
    assert normalize_reasoning(None, must) == {"effort": "low"}
    assert normalize_reasoning("high", must) == {"effort": "high"}
    for efforts, lowest in ((["high", "medium", "none"], "none"), (["medium", "high"], "medium"), (None, "low"),
                            (["max", "xhigh"], "xhigh"), (["turbo"], "low")):
        entry = ModelEntry.model_validate(_entry(reasoning_mandatory=True, reasoning_efforts=efforts))
        assert entry.lowest_effort == lowest and normalize_reasoning("disabled", entry) == {"effort": lowest}
    with pytest.raises(ValidationError):
        ModelEntry.model_validate(_entry(reasoning_efforts=[]))
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    server = Server((200, or_response()))
    r = ask("orm_must", "count", json_schema=COUNT_SCHEMA, http=server.client())
    assert server.bodies()[0]["reasoning"] == {"effort": "low"} and r.reasoning_setting == {"effort": "low"}
    assert "mandatory" in r.raw["reasoning_note"] and "effort low" in r.raw["reasoning_note"]
    hit = ask("orm_must", "count", json_schema=COUNT_SCHEMA)
    assert hit.cached and hit.raw["reasoning_note"] == r.raw["reasoning_note"]
    plain = ask("orm", "count", json_schema=COUNT_SCHEMA, http=server.client())
    assert server.bodies()[1]["reasoning"] == {"enabled": False} and "reasoning_note" not in plain.raw


def test_an_answer_cut_at_max_tokens_is_marked_truncated_and_never_repaired(ask, ledger, tmp_path, monkeypatch):
    from avail.router import smoke

    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    # the statebench failure: the model reasons its allowance away and the JSON stops halfway
    cut = or_response(content='{"holds": true, "confid', finish="length", prompt=600, completion=300, reasoning=290,
                      cost=0.0042)
    r = ask("orm_must", "Is the plug seated?", json_schema=JUDGE_SCHEMA, max_tokens=300,
            http=Server((200, cut)).client())
    assert r.truncated and r.finish_reason == "length" and r.json is None and not r.json_ok
    assert r.json_error.startswith("not JSON") and r.text == '{"holds": true, "confid'  # kept as written
    assert r.usd == 0.0042 and r.reasoning_tokens == 290 and ledger.spent() == pytest.approx(0.0042)  # still paid
    # the same request again is the cached, truncated answer. A larger allowance is a new request
    again = ask("orm_must", "Is the plug seated?", json_schema=JUDGE_SCHEMA, max_tokens=300)
    assert again.cached and again.truncated and again.json_error == r.json_error
    answered = or_response(content='{"holds": true, "confidence": 0.9, "reason": "flush"}', completion=400,
                           reasoning=350)
    more = ask("orm_must", "Is the plug seated?", json_schema=JUDGE_SCHEMA, max_tokens=2000,
               http=Server((200, answered)).client())
    assert not more.cached and not more.truncated and more.json_ok and more.request_hash != r.request_hash
    whole = ask("orm", "Is the plug seated?", json_schema=JUDGE_SCHEMA, http=Server((200, or_response(
        content='{"holds": true, "confidence": 0.9, "reason": "flush"}'))).client())
    assert not whole.truncated and whole.json_ok
    # the smoke row says so, and counts the answer as wrong
    led = SpendLedger(tmp_path / "smoke" / "spend_ledger.jsonl", caps(S1=1.0))
    router = Router(session="S1", models=MODELS, ledger=led,
                    http=Server((200, or_response(content='{"plugg', finish="length"))).client())
    out = smoke.run_smoke(only=["orm"], models=MODELS, router=router, out_dir=tmp_path / "s1", echo=False)
    (row,) = out["rows"]
    assert row["ok"] and not row["correct"] and row["answer"] is None and "cut at max_tokens (300)" in row["note"]
    assert "cut at max_tokens" in smoke.format_smoke_table(out["rows"])


def test_temperature_is_left_out_for_a_model_that_does_not_take_one(ask, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    server = Server((200, or_response()))
    r = ask("orm_no_temp", "count", json_schema=COUNT_SCHEMA, temperature=0.0, http=server.client())
    assert "temperature" not in server.bodies()[0] and r.raw["parameters_left_out"] == ["temperature"]
    assert server.bodies()[0]["provider"] == {"require_parameters": True}  # the reason it must not be sent
    r2 = ask("orm", "count", json_schema=COUNT_SCHEMA, temperature=0.0, http=server.client())
    assert server.bodies()[1]["temperature"] == 0.0 and "parameters_left_out" not in r2.raw
    r3 = ask("orm_no_temp", "count", json_schema=COUNT_SCHEMA, http=server.client())  # none given: nothing left out
    assert "temperature" not in server.bodies()[2] and "parameters_left_out" not in r3.raw
    assert r3.request_hash != r.request_hash  # the temperature the caller gave still keys the cache


def test_extra_body_fields_are_sent_and_cannot_override_the_routers_own(ask, ledger, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    body = or_response()
    body["choices"][0]["logprobs"] = {"content": [{"token": "3", "logprob": -0.02}]}
    server = Server((200, body))
    r = ask("orm", "judge", json_schema=COUNT_SCHEMA, extra={"body": {"logprobs": True, "top_logprobs": 5}},
            http=server.client())
    sent = server.bodies()[0]
    assert sent["logprobs"] is True and sent["top_logprobs"] == 5 and sent["usage"] == {"include": True}
    assert r.raw["choices"][0]["logprobs"] == {"content": [{"token": "3", "logprob": -0.02}]}
    plain = ask("orm", "judge", json_schema=COUNT_SCHEMA, http=server.client())
    assert "logprobs" not in server.bodies()[1] and plain.request_hash != r.request_hash
    for bad in ({"model": "vendor/other"}, {"max_tokens": 10 ** 6}, {"stream": True}, {"usage": {"include": False}},
                {"reasoning": {"effort": "max"}}, {"temperature": 2}, {"messages": []}, {"provider": {}}):
        with pytest.raises(RouterError, match="may not set"):
            ask("orm", "judge", extra={"body": bad}, http=server.client())
    with pytest.raises(RouterError, match="must be a dict"):
        ask("orm", "judge", extra={"body": ["logprobs"]}, http=server.client())
    assert len(server.requests) == 2 and len(_reserves(ledger)) == 2  # the refusals sent and reserved nothing
    monkeypatch.setenv("ELORIAN_API_KEY", PLAIN_KEY)
    el = Server((200, or_response(cost=None, model="elorian-test")))
    ask("el", "judge", extra={"body": {"top_logprobs": 3}}, http=el.client())
    assert el.bodies()[0]["top_logprobs"] == 3


def test_an_unverified_entry_makes_no_paid_call(ask, ledger, cache, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    for key in ("unverified", "unpriced"):  # an id is there, but verified is false
        rec = Recorder()
        with pytest.raises(ModelNotConfigured, match="verified: false") as err:
            ask(key, "hi", http=refusing_client(), events=rec)
        assert key in str(err.value) and "no paid call" in str(err.value) and rec.events == []
    assert ledger.records() == [] and not list(cache.root.rglob("*.json"))
    # a verified entry is called, and a cached answer of an entry that is unverified now is still served for $0
    models = _models()
    r = ask("orm", "hi", models=models, http=Server((200, or_response())).client())
    models.models["orm"].verified = False
    assert ask("orm", "hi", models=models, http=refusing_client()).cached
    with pytest.raises(ModelNotConfigured, match="verified: false"):
        ask("orm", "something new", models=models, http=refusing_client())
    assert len(_reserves(ledger)) == 1 and ledger.spent() == pytest.approx(r.usd)
    assert ask("fake", "the fake is never a paid call").usd == 0.0


def test_estimate_is_the_amount_a_call_reserves(ask, ledger, cache, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    prompt = "Is the plug seated in port 7?"
    kw = dict(json_schema=JUDGE_SCHEMA, max_tokens=600)
    est = estimate_usd("orm", prompt, image_count=2, models=MODELS, **kw)
    chars = len(prompt) + len(canonical_json(JUDGE_SCHEMA))
    assert est == pytest.approx((math.ceil(chars / 3.5) + 2 * 1600) * 2.0 / 1e6 + 600 * 10.0 / 1e6)
    assert ledger.records() == [] and not list(cache.root.rglob("*.json"))  # estimating spends and writes nothing
    r = ask("orm", prompt, images=[board_array(), board_array()], http=Server((200, or_response())).client(), **kw)
    assert est == pytest.approx(r.projected_usd) and _reserves(ledger)[0]["usd"] == pytest.approx(est)
    assert estimate_usd("orm", prompt, images=[board_array()], models=MODELS, **kw) == pytest.approx(
        est - 1600 * 2.0 / 1e6)
    assert Router(models=MODELS).estimate("orm", prompt, image_count=2, **kw) == est
    assert estimate_usd("orm", prompt, image_count=2, models=MODELS, reasoning={"max_tokens": 4000}, **kw) == (
        pytest.approx(est + 4000 * 10.0 / 1e6))  # a reasoning budget is part of the worst case
    assert estimate_usd("fake", prompt, image_count=3, models=MODELS) == 0.0
    with pytest.raises(RouterError, match="text-only"):
        estimate_usd("jev", "state", image_count=1, models=MODELS)
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError):
            estimate_usd("orm", prompt, image_count=bad, models=MODELS)
    with pytest.raises(ValueError):
        estimate_usd("orm", prompt, image_count=1, images=[board_array(), board_array()], models=MODELS)
    with pytest.raises(KeyError):
        estimate_usd("no_such_model", prompt, models=MODELS)
    with pytest.raises(ValueError):
        estimate_usd("orm", prompt, max_tokens=0, models=MODELS)
    # the committed table: a 3-view judge call to the dearest judge stays well under the S3 cap
    table = load_models()
    dearest = max(estimate_usd(k, prompt, image_count=3, models=table, **kw) for k in table.by_role("judge"))
    assert 0 < dearest < 0.5


# --------------------------------------------------------------------------- events
def test_model_call_events_reach_every_kind_of_sink(ask):
    writer, rec, calls = AppendOnly(), Recorder(), []
    ask("fake", "one", events=writer)
    ask("fake", "one", events=rec)
    ask("fake", "one", events=lambda kind, payload: calls.append((kind, payload)))
    assert [k for k, _ in writer.rows] == [k for k, _ in rec.events] == [k for k, _ in calls] == ["model_call"]
    assert [writer.rows[0][1]["cache_hit"], rec.events[0][1]["cache_hit"], calls[0][1]["cache_hit"]] == [
        False, True, True]
    for payload in (writer.rows[0][1], rec.events[0][1], calls[0][1]):
        ModelCallPayload.model_validate(payload)
        json.dumps(payload, allow_nan=False)  # plain JSON
    for bad in ([], {}, (), "ledger", 3, object()):
        with pytest.raises(TypeError, match="events"):
            ask("fake", "two", events=bad)
        with pytest.raises(TypeError, match="events"):
            Router(events=bad)
    with pytest.raises(ValidationError):  # the payload contract can fail: a field too many, a wrong role
        ModelCallPayload.model_validate({**rec.events[0][1], "surprise": 1})
    with pytest.raises(ValidationError):
        ModelCallPayload.model_validate({**rec.events[0][1], "role": "writer"})


def test_model_call_events_land_in_the_run_ledger(ledger, cache, tmp_path, monkeypatch):
    from avail.ledger import LedgerWriter, iter_events

    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    with LedgerWriter("router_events", folder=tmp_path / "run") as run:
        router = Router(session="T", models=MODELS, ledger=ledger, cache=cache, events=run, phase="zoo",
                        http=Server((200, or_response())).client())
        first = router.chat("orm", "Is port 7 seated?", images=[board_array()], json_schema=COUNT_SCHEMA)
        router.chat("orm", "Is port 7 seated?", images=[board_array()], json_schema=COUNT_SCHEMA)
    events = list(iter_events(tmp_path / "run", kinds="model_call"))
    assert [e["seq"] for e in events] == [0, 1] and {e["kind"] for e in events} == {"model_call"}
    assert [e["payload"]["cache_hit"] for e in events] == [False, True]
    assert [e["payload"]["usd"] for e in events] == [0.00123, 0.0]
    assert events[0]["payload"]["request_hash"] == first.request_hash and events[0]["payload"]["session"] == "T"
    for event in events:
        ModelCallPayload.model_validate(event["payload"])
        assert PLAIN_KEY not in json.dumps(event) and "base64" not in json.dumps(event)
    # a report can total the spend from the ledger alone, and a rerun's rows still say what the check cost
    from avail.ledger import load_ledger

    table = load_ledger(tmp_path / "run").model_calls
    assert list(table["cache_hit"]) == [False, True] and table["usd"].sum() == pytest.approx(0.00123)
    assert list(table["session"]) == ["T", "T"] and list(table["phase"]) == ["zoo", "zoo"]
    assert math.isnan(table["original_usd"][0]) and table["original_usd"][1] == 0.00123


# --------------------------------------------------------------------------- HTTP
def test_connecting_gets_at_most_30_s_of_the_deadline():
    seen = []

    def capture(request):
        seen.append(dict(request.extensions["timeout"]))
        return httpx.Response(200, json={})

    client = HttpClient(transport=httpx.MockTransport(capture), sleep=lambda s: None)
    client.post_json("https://a.invalid/x", {}, timeout_s=900.0)
    client.post_json("https://a.invalid/x", {}, timeout_s=12.0)
    assert seen[0] == {"connect": 30.0, "read": 900.0, "write": 900.0, "pool": 900.0}
    assert seen[1] == {"connect": 12.0, "read": 12.0, "write": 12.0, "pool": 12.0}
    assert router_http.CONNECT_TIMEOUT_S == 30.0


# --------------------------------------------------------------------------- redaction in errors and logs
def test_keys_are_redacted_in_errors_and_logs(ask, ledger, cache, tmp_path, monkeypatch, capsys, caplog, recwarn):
    from avail.router import smoke

    caplog.set_level(0)
    errors, events = [], Recorder()

    def refused(request):
        key = request.headers["authorization"].split(" ", 1)[1]
        raise httpx.ConnectError(f"cannot reach the host while holding {key}", request=request)

    for key in (PLAIN_KEY, PATTERN_KEY, 'av-test-"quoted"-' + "Q7" * 12):
        monkeypatch.setenv("OPENROUTER_API_KEY", key)
        echo = {"error": {"message": f"Invalid key {key}", "metadata": {"raw": f"Authorization: Bearer {key}"}}}
        for n, item in enumerate(((401, echo), (500, f"upstream failure while using {key}"), refused,
                                  (200, f"not json at all, but {key}"))):
            with pytest.raises(ModelCallError) as err:
                ask("orm", f"failure {n} {len(key)}", http=Server(item).client(), events=events)
            errors.append(err.value)
        ok = or_response(note=f"served for {key}", model=f"vendor/{key}")
        ask("orm", f"answer {len(key)}", http=Server((200, ok)).client(), events=events)
        with pytest.raises(SpendCapExceeded) as err:  # a refusal by the spend ledger
            ask("orm", f"too much {len(key)}", max_tokens=200000, http=refusing_client(), events=events)
        errors.append(err.value)
        # the smoke prints its rows: an error row that echoes the key is printed without it
        led = SpendLedger(tmp_path / f"smoke_{len(key)}" / "spend_ledger.jsonl", caps(S1=1.0))
        router = Router(session="S1", models=MODELS, ledger=led, cache=cache, http=Server((401, echo)).client())
        smoke.run_smoke(only=["orm"], models=MODELS, router=router, out_dir=tmp_path / f"out_{len(key)}", echo=True)
    printed = capsys.readouterr()
    logs = {
        "exceptions": "\n".join(f"{e} {e!r} {json.dumps(getattr(e, 'raw', {}))}" for e in errors),
        "events": json.dumps(events.events),
        "spend ledger": ledger.path.read_text(encoding="utf-8"),
        "cache": "\n".join(p.read_text(encoding="utf-8") for p in cache.root.rglob("*.json")),
        "stdout": printed.out,
        "stderr": printed.err,
        "logging": caplog.text,
        "warnings": "\n".join(str(w.message) for w in recwarn.list),
        "smoke tables": "\n".join(p.read_text(encoding="utf-8") for p in tmp_path.glob("out_*/live_table.json")),
    }
    for name in ("exceptions", "events", "spend ledger", "cache", "stdout", "smoke tables"):
        assert logs[name].strip(), f"{name} is empty, so this test would prove nothing about it"
    assert "[redacted]" in logs["exceptions"] and "[redacted]" in logs["stdout"] and "ERROR" in logs["stdout"]
    for key in (PLAIN_KEY, PATTERN_KEY, 'av-test-"quoted"-' + "Q7" * 12):
        escaped = json.dumps(key)[1:-1]
        for name, text in logs.items():
            assert key not in text and escaped not in text, f"the key is in the {name}"
    assert "Q7Q7Q7" not in "".join(logs.values()) and "kkkkkkkk" not in "".join(logs.values())
    assert len(events.events) == 3 * 5  # 4 failures and 1 answer per key (the cap refusal attempts nothing)


def test_credential_patterns_are_cut_even_when_the_key_is_not_the_loaded_one(ask, cache, monkeypatch):
    from avail.router.redact import redact, scrub

    other = "sk-or-v1-" + "9f" * 20  # shaped like a key, and not the one this process loaded
    assert redact(f"wrong key, did you mean {other}?") == "wrong key, did you mean [redacted]?"
    assert redact("token sk-" + "a" * 24 + " was refused") == "token [redacted] was refused"
    assert redact("sent Bearer " + "x" * 20 + " in the header") == "sent [redacted] in the header"
    assert redact("a short sk-abc and the word Bearer stay") == "a short sk-abc and the word Bearer stay"
    assert redact("see data:image/png;base64," + "A" * 200 + " here") == "see [data URL removed] here"
    assert redact("x" * 80) == "[base64 removed]" and redact("x" * 80, media=False) == "x" * 80
    cleaned = scrub({"Authorization": f"Bearer {other}", "headers": {"x-api-key": other}, "api_key": other,
                     "Set-Cookie": "s=1", "note": f"served for {other}", "blob": b"\x00\x01", "n": 3, other: "k"})
    assert cleaned == {"note": "served for [redacted]", "blob": "[2 bytes removed]", "n": 3, "[redacted]": "k"}
    # through chat(): another key in a refusal, and in the metadata of an answer that gets cached
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    with pytest.raises(ModelCallError) as err:
        ask("orm", "refused", http=Server((401, {"error": {"message": f"did you mean {other}?"}})).client())
    assert other not in str(err.value) and other not in json.dumps(err.value.raw) and "[redacted]" in str(err.value)
    r = ask("orm", "answered", http=Server((200, or_response(note=f"routed with {other}"))).client())
    record = cache.path_for(r.request_hash).read_text(encoding="utf-8")
    assert other not in json.dumps(r.raw) and other not in record and "routed with [redacted]" in record


def test_smoke_never_prints_a_loaded_key_even_from_the_model_table(tmp_path, monkeypatch, capsys):
    from avail.router import smoke

    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    leaky = _models()
    leaky.models["orm"].id = f"vendor/{PLAIN_KEY}"  # a key pasted into the wrong field of the table
    led = SpendLedger(tmp_path / "smoke" / "spend_ledger.jsonl", caps(S1=1.0))
    router = Router(session="S1", models=leaky, ledger=led,
                    http=Server((400, {"error": {"message": "no such model"}})).client())
    out = smoke.run_smoke(only=["orm"], models=leaky, router=router, out_dir=tmp_path / "s1", echo=True)
    printed = capsys.readouterr().out
    assert "calling orm (vendor/[redacted])" in printed and "ERROR ModelCallError" in printed
    assert PLAIN_KEY not in printed and PLAIN_KEY not in json.dumps(out)
    assert PLAIN_KEY not in (tmp_path / "s1" / "live_table.json").read_text(encoding="utf-8")


def test_smoke_output_survives_a_stream_that_cannot_encode_a_providers_text(tmp_path, monkeypatch):
    from avail.router import smoke

    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    odd = "bad " + chr(0x4E2D) + chr(0x6587) + "\nline 2"  # 2 characters cp1252 cannot write, and a line break
    led = SpendLedger(tmp_path / "smoke" / "spend_ledger.jsonl", caps(S1=1.0))
    router = Router(session="S1", models=MODELS, ledger=led,
                    http=Server((400, {"error": {"message": odd}})).client())
    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict", newline="\n")
    monkeypatch.setattr(sys, "stdout", stream)  # what a redirected stdout is on this machine
    out = smoke.run_smoke(only=["orm"], models=MODELS, router=router, out_dir=tmp_path / "s1", echo=True)
    stream.flush()
    printed = stream.buffer.getvalue().decode("cp1252")
    row = next(line for line in printed.splitlines() if line.startswith("orm "))
    assert "ERROR ModelCallError" in row and "\\u4e2d\\u6587 line 2" in row  # written as escapes, on 1 line
    assert out["rows"][0]["ok"] is False and chr(0x4E2D) in out["rows"][0]["error"]  # the row itself is unchanged
    saved = json.loads((tmp_path / "s1" / "live_table.json").read_text(encoding="utf-8"))
    assert chr(0x4E2D) in saved["rows"][0]["error"]


# --------------------------------------------------------------------------- the model table
LOOPKIT_IDS_AND_PRICES = {  # verbatim from loopkit's config/models.yaml (version 2026-10-17)
    "opus": ("openrouter", "anthropic/claude-opus-5.5", 4.00, 20.00),
    "astra": ("openrouter", "openai/gpt-6-astra", 10.00, 50.00),
    "sol": ("openrouter", "openai/gpt-6.1-sol", 2.00, 10.00),
    "argon": ("openrouter", "google/gemini-4-argon", 2.00, 10.00),
    "glimpse": ("elorian", "elorian-1", 0.60, 3.00),
    "cheap_open": ("openrouter", "nex-agi/nex-n2.5-mini", 0.025, 0.10),
    "luna": ("openrouter", "openai/gpt-6-luna", 0.10, 0.50),
    "d1": ("jev", "d1", 0.04, 0.0),
    "jev": ("jev", "jev-latest", 0.042, 0.0),
    "fake": ("fake", "fake", 0.0, 0.0),
}


def test_the_committed_model_table_carries_loopkits_entries_in_avail_roles():
    table = load_models()
    assert table.currency == "USD" and table.unit == "usd_per_million_tokens" and table.version == "2026-11-11"
    assert set(table.roles) == {"judge", "decider", "agent", "test"}
    assert table.carry_over_from_loopkit == {
        "judge": ["opus", "astra", "sol", "argon", "glimpse", "cheap_open", "luna"],
        "decider": ["jev", "d1"], "test": ["fake"]}
    for role, keys in table.carry_over_from_loopkit.items():
        for key in keys:
            assert table.models[key].roles == [role], key
    for key, (adapter, model_id, price_in, price_out) in LOOPKIT_IDS_AND_PRICES.items():
        entry = table.get(key)
        assert (entry.adapter, entry.id, entry.price_in, entry.price_out) == (adapter, model_id, price_in, price_out)
    assert set(table.models) == set(LOOPKIT_IDS_AND_PRICES) | {"fable"}
    assert table.by_role("judge") == ["opus", "fable", "astra", "sol", "argon", "glimpse", "cheap_open", "luna"]
    assert table.by_role("decider") == ["d1", "jev"] and table.by_role("test") == ["fake"]
    assert table.by_role("agent") == []
    # the blocks of the template are kept
    assert table.e1_minimum == ["opus", "astra", "argon", "luna", "cheap_open"]
    assert list(table.to_verify) == ["fable", "open_27b", "flora_rewarder"]
    assert table.to_verify["open_27b"].id is None and not table.to_verify["open_27b"].verified
    flora = table.to_verify["flora_rewarder"]
    assert (flora.id, flora.adapter, flora.verified) == ("nvidia/Flora-Rewarder", "local", False)
    # glimpse keeps its routes, and the fake is free
    glimpse = table.get("glimpse")
    assert glimpse.base_url == "https://api.elorian.ai/v1" and glimpse.env_key == "ELORIAN_API_KEY"
    assert (glimpse.alt_route.adapter, glimpse.alt_route.id) == ("openrouter", "elorian/glimpse")
    assert table.get("sol").fallback_id == "openai/gpt-6-sol"
    assert table.get("d1").endpoint == "https://api.liquid.ai/decisions/v1/systemone"
    assert table.get("jev").endpoint == "https://api.typesafe.ai/v1/systemone" and not table.get("jev").vision
    assert all(table.get(k).vision for k in table.by_role("judge"))
    with pytest.raises(KeyError, match="to_verify"):
        table.get("open_27b")  # an entry still to verify is not callable, and the error says why
    with pytest.raises(KeyError, match="unknown model key") as err:
        table.get("gpt")
    assert "to_verify" not in str(err.value)
    with pytest.raises(ValueError):
        table.by_role("writer")


def test_the_committed_model_table_records_what_was_verified_and_when():
    table = load_models()
    for key, entry in table.models.items():
        assert entry.verified and entry.source and entry.priced, key
    reread = {"opus", "fable", "astra", "sol", "argon", "cheap_open", "luna"}  # ids the 2026-11-11 catalogue lists
    for key in reread:
        entry = table.get(key)
        assert entry.accessed == "2026-11-11" and "2026-11-11T13:15Z" in entry.source, key
        assert entry.adapter == "openrouter" and "2026-11-11" in entry.notes
    # not re-read that day: they keep loopkit's dates, and their notes say so
    assert {k: table.get(k).accessed for k in ("glimpse", "d1", "jev", "fake")} == {
        "glimpse": "2026-10-17", "d1": "2026-10-17", "jev": "2026-09-26", "fake": None}
    for key in ("glimpse", "d1", "jev"):
        assert "not re-read" in table.get(key).notes.lower() and "2026-11-11" in table.get(key).notes
    # Claude Fable 5.2 was listed, so it has an entry and to_verify.fable points at it
    fable = table.get("fable")
    assert (fable.id, fable.price_in, fable.price_out, fable.cached_in) == (
        "anthropic/claude-fable-5.2", 8.00, 40.00, 0.20)
    assert fable.roles == ["judge"] and fable.vision and fable.display == "Claude Fable 5.2"
    pending = table.to_verify["fable"]
    assert (pending.id, pending.verified, pending.model_key, pending.accessed) == (
        "anthropic/claude-fable-5.2", True, "fable", "2026-11-11")
    # what the catalogue says about reasoning and temperature, which the router acts on
    assert {k for k, e in table.models.items() if e.reasoning_mandatory} == {"opus", "fable", "astra", "sol", "argon"}
    assert {k for k, e in table.models.items() if e.supports_temperature is False} == {
        "opus", "fable", "astra", "sol", "luna"}
    for key in ("opus", "fable", "astra", "sol", "argon"):
        assert normalize_reasoning("disabled", table.get(key)) == {"effort": "low"}, key
    for key in ("luna", "cheap_open"):
        assert normalize_reasoning("disabled", table.get(key)) == {"enabled": False}, key
    assert table.get("cheap_open").cached_in == 0.0025
    # the header says what was done and on which date, and the template wording is gone
    text = config.MODELS_YAML.read_text(encoding="utf-8")
    header = text.split("\nversion:", 1)[0]
    assert header.startswith("# AVAIL model table") and "What was done on 2026-11-11" in header
    assert "TEMPLATE" not in text and "models: {}" not in text and "2026-11-11-template" not in text
    assert EM_DASH not in text and "loopkit" in header and "13:15Z" in header


def test_model_table_validation_names_what_is_wrong(tmp_path):
    good = _entry(id="vendor/x")
    ModelEntry.model_validate(good)
    for change, message in (({"source": None}, "source"), ({"source": "  "}, "source"),
                            ({"accessed": None}, "accessed"), ({"accessed": "11/11/2026"}, "YYYY-MM-DD"),
                            ({"accessed": "20261111"}, "YYYY-MM-DD"), ({"accessed": "2026-13-45"}, "YYYY-MM-DD"),
                            ({"adapter": "anthropic"}, "not one of"), ({"roles": []}, "roles"),
                            ({"roles": ["writer"]}, "roles"), ({"id": None}, "verified: true needs id"),
                            ({"price_out": None}, "verified: true needs id"), ({"surprise": 1}, "surprise")):
        with pytest.raises(ValidationError, match=message):
            ModelEntry.model_validate({**good, **change})
    ModelEntry.model_validate({**good, "adapter": "fake", "accessed": None})  # the offline fake has no date
    ModelEntry.model_validate({**good, "verified": False, "source": None, "accessed": None, "price_in": None,
                               "price_out": None, "id": None})  # an entry still to look up may be empty
    base = {"version": "t", "currency": "USD", "unit": "usd_per_million_tokens", "models": {"m": good}}
    ModelsConfig.model_validate(base)
    pending = {"display": "d", "why": "w", "adapter": "openrouter", "id": "vendor/x", "roles": ["judge"],
               "verified": True, "model_key": "m"}
    ModelsConfig.model_validate({**base, "to_verify": {"p": pending}})
    for change, message in (({"e1_minimum": ["m", "ghost"]}, "ghost"),
                            ({"roles": {"decider": "x"}}, "roles block has no line"),
                            ({"carry_over_from_loopkit": {"judge": ["ghost"]}}, "ghost"),
                            ({"carry_over_from_loopkit": {"writer": ["m"]}}, "carry_over_from_loopkit"),
                            ({"to_verify": {"p": {**pending, "model_key": "ghost"}}}, "points at"),
                            ({"to_verify": {"p": {**pending, "id": "vendor/other"}}}, "points at"),
                            ({"to_verify": {"p": {**pending, "id": None}}}, "needs the id"),
                            ({"currency": "EUR"}, "currency"), ({"surprise": {}}, "surprise")):
        with pytest.raises(ValidationError, match=message):
            ModelsConfig.model_validate({**base, **change})
    path = tmp_path / "not_a_table.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="models"):
        load_models(path)


# --------------------------------------------------------------------------- the `avail models` command
def test_avail_models_without_a_key_prints_the_table_and_skips_the_live_smoke(capsys, tmp_path):
    from avail import cli

    assert cli.main(["models"]) == 0  # routed by avail/cli.py to avail.router.smoke.main
    out = capsys.readouterr().out
    table = load_models()
    assert "AVAIL model table" in out and "USD per million tokens" in out
    for key, entry in table.models.items():
        row = next(line for line in out.splitlines() if line.split(" | ")[0].strip() == key)
        cells = [cell.strip() for cell in row.split(" | ")]
        assert cells[:4] == [key, entry.id, entry.adapter, ", ".join(entry.roles)], row
        assert float(cells[4]) == entry.price_in and float(cells[5]) == entry.price_out, row
        assert cells[6].startswith("yes"), row
    assert next(line for line in out.splitlines() if line.startswith("opus ")).endswith("yes, 2026-11-11")
    assert "open_27b" in out and "flora_rewarder" in out  # named as not callable yet
    assert "No API key is set" in out and "live smoke was skipped" in out and "OPENROUTER_API_KEY" in out
    assert "calling" not in out
    assert not (tmp_path / "data").exists()  # the no-key path wrote nothing under AVAIL_DATA


def test_avail_models_with_a_key_adds_a_live_row_per_model_under_the_s1_cap(capsys, tmp_path, monkeypatch):
    from avail.router import smoke

    table = load_models()
    by_id = {e.id: e for e in table.models.values()} | {"elorian/glimpse": table.get("glimpse")}
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)

    def answer(request):
        body = json.loads(request.content)
        entry = by_id[body["model"]]
        cost = (1500 * entry.price_in + 40 * entry.price_out) / 1e6
        return httpx.Response(200, json=or_response(content='{"plugged": 3}', model=body["model"], cost=cost,
                                                    prompt=1500, completion=40))

    server = Server(answer)
    monkeypatch.setattr(router_http, "_DEFAULT_CLIENT", server.client())
    assert smoke.main([]) == 0
    out = capsys.readouterr().out
    # every entry the OpenRouter key reaches, in table order: the openrouter entries, and glimpse by its alt_route
    called = [k for k, e in table.models.items()
              if e.adapter == "openrouter" or (e.alt_route is not None and e.alt_route.adapter == "openrouter")]
    assert called[:3] == ["opus", "fable", "astra"] and "glimpse" in called and len(called) >= 8
    assert [json.loads(r.content)["model"] for r in server.requests] == [
        "elorian/glimpse" if k == "glimpse" else table.get(k).id for k in called]
    live = out.split("model key", 1)[1]
    for key in called:
        row = next(line for line in live.splitlines() if line.split(" | ")[0].strip() == key)
        cells = [cell.strip() for cell in row.split(" | ")]
        entry = table.get(key)
        assert float(cells[2]) >= 0 and cells[3:5] == ["1500", "40"], row  # latency, tokens in and out
        assert float(cells[5]) == pytest.approx((1500 * entry.price_in + 40 * entry.price_out) / 1e6, abs=5e-7), row
        assert cells[6] == "provider" and cells[7] == "3 (right)", row
    assert "d1: no key: set LIQUID_API_KEY" in out and "jev: no key: set TYPESAFE_JEV_API_KEY" in out
    assert "spend session S1: cap $1.00" in out and PLAIN_KEY not in out
    # reasoning is off where the model allows it, and at its lowest effort with more room where it does not
    for request in server.requests:
        body = json.loads(request.content)
        entry = by_id[body["model"]]
        assert body["usage"] == {"include": True}
        if entry.reasoning_mandatory:
            assert body["reasoning"] == {"effort": "low"} and body["max_tokens"] == 1000
        else:
            assert body["reasoning"] == {"enabled": False} and body["max_tokens"] == 300
    # every call was charged to S1 in the default spend ledger, and the session is under its $1 cap
    led = SpendLedger()
    assert led.path == tmp_path / "data" / "spend_ledger.jsonl"
    assert {r["session"] for r in _reserves(led)} == {"S1"}
    assert len(_reserves(led)) == len(_settles(led)) == len(called)
    expected = sum((1500 * table.get(k).price_in + 40 * table.get(k).price_out) / 1e6 for k in called)
    assert led.spent("S1") == pytest.approx(expected) and led.spent("S1") < led.caps.session_cap("S1") == 1.0
    saved = json.loads((tmp_path / "data" / "runs" / "s1_router_smoke" / "live_table.json").read_text("utf-8"))
    assert [r["model_key"] for r in saved["rows"]] == called and saved["session"] == "S1"
    assert PLAIN_KEY not in json.dumps(saved)
    # once S1 is nearly used up, a row is refused before any network I/O and the command returns 1
    led.reserve(0.95, "opus", "0" * 64, session="S1")
    before = len(server.requests)
    assert smoke.main(["--only", "astra"]) == 1
    out = capsys.readouterr().out
    assert "ERROR SpendCapExceeded" in out and len(server.requests) == before
    assert smoke.main(["--only", "no_such_model"]) == 2
    assert smoke.main(["--no-live"]) == 0 and len(server.requests) == before


def test_avail_models_fake_row_runs_without_a_key(capsys, tmp_path):
    from avail.router import smoke

    assert smoke.main(["--fake"]) == 0
    out = capsys.readouterr().out
    assert "No API key is set" in out and "live smoke was skipped" in out
    row = next(line for line in out.split("model key", 1)[1].splitlines() if line.startswith("fake "))
    cells = [cell.strip() for cell in row.split(" | ")]
    assert cells[5] == "0.000000" and cells[6] == "computed" and "0 (wrong)" in cells[7]  # a placeholder answer
    assert "Spend this run: $0.000000" in out
    assert SpendLedger().spent() == 0.0 and {r["session"] for r in _reserves(SpendLedger())} == {None}


# --------------------------------------------------------------------------- gaps found by the review's mutation run, 2026-11-11
# The review recorded 266 mutants of the router and 30 lived. These tests kill the ones that touch
# CLAUDE.md rule 5 (a key never reaches a log, an error or a file) and the cache's correctness.

def test_a_cached_record_written_for_another_request_is_a_miss(cache):
    """Mutant K01: the cache served a record whose own request hash is another one."""
    a, b = "a" * 64, "b" * 64
    cache.put(a, {"text": "answer for a", "request_hash": a})
    path_b = cache.path_for(b)
    path_b.parent.mkdir(parents=True, exist_ok=True)
    path_b.write_text(cache.path_for(a).read_text(encoding="utf-8"), encoding="utf-8")   # a's record under b's name
    assert cache.get(a)["text"] == "answer for a"
    assert cache.get(b) is None


def test_the_cache_cuts_credentials_on_the_way_in(cache):
    """Mutant K02: a record put straight into the cache kept a key in its metadata."""
    h = "c" * 64
    cache.put(h, {"text": "the answer", "note": f"served with {PATTERN_KEY}"})
    stored = cache.path_for(h).read_text(encoding="utf-8")
    assert PATTERN_KEY not in stored and "the answer" in stored


def test_the_failure_event_of_a_call_that_died_in_flight_holds_no_key(ask, monkeypatch):
    """Mutant C35: the event written for a call that failed before any answer carried the raw error text."""
    from avail.router.adapters.openrouter import OpenRouterAdapter

    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)

    def dies(self, request, key, client):
        raise RuntimeError(f"socket closed while sending with {key}")

    monkeypatch.setattr(OpenRouterAdapter, "send", dies)
    rec = Recorder()
    with pytest.raises(RuntimeError):
        ask("orm", "died in flight", events=rec)
    (kind, p), = rec.events
    assert p["ok"] is False and "socket closed" in p["error"]
    assert PLAIN_KEY not in json.dumps(rec.events)


def test_the_cost_note_holds_no_key(ask, cache, monkeypatch):
    """A provider that puts the key in a token count gets it into no output.

    The review's mutant C67 (the cost note not redacted) survives this test: the note does not
    quote the odd value, and later layers cut the key anyway. The end-to-end check stays.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    usage = {"prompt_tokens": PLAIN_KEY, "completion_tokens": 3, "total_tokens": 603, "cost": 0.00123}
    rec = Recorder()
    r = ask("orm", "odd usage", http=Server(_usage_body(usage)).client(), events=rec)
    texts = [json.dumps(r.raw), json.dumps(rec.events), cache.path_for(r.request_hash).read_text(encoding="utf-8")]
    assert all(PLAIN_KEY not in t for t in texts)
    assert r.raw.get("usd_note")                             # the odd count was noted, without the key
