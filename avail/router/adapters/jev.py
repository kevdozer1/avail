"""The ``jev`` adapter: small typed-decision models (Jev, d1) for the shadow decider at forks.

``POST <entry.endpoint or the default URL>`` with a Bearer key from
``TYPESAFE_JEV_API_KEY`` (fallback ``TYPESAFE_API_KEY``, or the entry's
``env_key``)::

    {"model": "jev-latest", "state": "<the last user message text>", "questions": {...}}

These models answer typed questions (``choice``, ``noul``, ``score``) and never
free text, so a caller passes them as ``extra={"questions": {...}}``. The state
goes out as plain text, never nested inside another JSON string
(``docs/11_machine_gotchas.md``). The response carries an ``answers`` dict and
``usage`` with ``input_tokens`` and ``output_tokens`` but no cost, so the router
computes the cost from tokens at the models.yaml prices (input at ``price_in``,
and output is free there). ``Response.text`` is ``json.dumps(answers)`` and
``Response.json`` is the answers dict. Reasoning does not apply and is not sent.

The same protocol serves other typed-decision models through the entry's
``endpoint`` and ``env_key`` (models.yaml routes d1 here). Such a model may take
images. For an entry with ``vision: true`` they are sent as ``"images": [data
URLs]``.

Ported from loopkit's ``router/adapters/jev.py``. No live call has been made
through it.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from ..errors import ModelCallError, RouterError
from ..redact import redact, scrub
from .base import Adapter, AdapterRequest, AdapterResult, as_cost, check_url

if TYPE_CHECKING:  # pragma: no cover
    from ..http import HttpClient
    from ..models import ModelEntry

DEFAULT_URL = "https://api.typesafe.ai/v1/systemone"


def build_jev_payload(request: AdapterRequest) -> dict[str, Any]:
    """The request body. Images (only for an entry with ``vision: true``) go out as data URLs."""
    body: dict[str, Any] = {
        "model": request.model_id,
        "state": request.last_user_text(),
        "questions": request.extra["questions"],
    }
    if request.images:
        body["images"] = [img.data_url() for img in request.images]
    return body


class JevAdapter(Adapter):
    name = "jev"
    label = "the typed-decision API"
    default_env_vars = ("TYPESAFE_JEV_API_KEY", "TYPESAFE_API_KEY")

    def url(self, entry: "ModelEntry") -> str:
        return entry.endpoint or DEFAULT_URL

    def check_entry(self, entry: "ModelEntry") -> None:
        if entry.endpoint is not None:  # sent exactly as written
            check_url(entry, "endpoint", entry.endpoint, sent=self.url(entry))

    def normalize_reasoning(self, reasoning: Any, entry: "ModelEntry") -> Any:
        return None  # this API has no reasoning setting

    def check_request(self, request: AdapterRequest) -> None:
        questions = request.extra.get("questions")
        if not isinstance(questions, dict) or not questions:
            raise RouterError(
                f"{request.model_key} uses the jev adapter, which answers typed questions and never free "
                "text. Pass extra={'questions': {name: {'type': 'choice', 'instructions': ..., 'criteria': "
                "{option: meaning}}}} (or 'noul' or 'score' questions). The last user message is sent as the state.")
        if request.last_user_index is None:
            raise RouterError(f"{request.model_key}: the jev adapter needs a user message to send as the state")

    def send(self, request: AdapterRequest, key: str, http: "HttpClient") -> AdapterResult:
        try:
            body = build_jev_payload(request)
        except Exception as exc:  # noqa: BLE001 - nothing was sent
            raise self.build_error(request, exc, key) from None
        res = http.post_json(self.url(request.entry), body, key=key, timeout_s=request.timeout_s,
                             hooks=request.attempt_hooks)
        if not res.ok:
            raise self.call_error(request, res, key, res.attempts)
        data = res.data
        answers = data.get("answers") if isinstance(data, dict) else None
        if not isinstance(answers, dict):
            raise ModelCallError(
                f"{request.model_key} ({request.model_id}) via {self.label}: HTTP {res.status} without an answers "
                f"object: {redact(res.text or '', key)[:200]}", status=res.status, attempts=res.attempts,
                raw={"attempt_log": res.attempt_log})
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        raw = scrub(data, key)
        raw["attempt_log"] = res.attempt_log
        return AdapterResult(
            text=json.dumps(answers),
            json_value=answers,
            input_tokens=usage.get("input_tokens"),  # as reported: chat() drops a value that is not a count
            output_tokens=usage.get("output_tokens"),
            usd=as_cost(usage.get("cost")),  # not in the API today. Used if it ever appears
            model_returned=data.get("model") if isinstance(data.get("model"), str) else None,
            raw=raw,
            http_status=res.status,
            attempts=res.attempts,
            json_mode="typed_questions",
            generation_id=data.get("id") if isinstance(data.get("id"), str) else None,
            reasoning_in_output=self.reasoning_in_output,
        )
