"""``avail models``: the model table, and a live smoke call per model when a key is set.

    uv run avail models                    # the table, then 1 small call per model that has a key
    uv run avail models --only opus,luna   # the live smoke for these keys only
    uv run avail models --fake             # also the offline FakeModel ($0), to check the path
    uv run avail models --no-live          # the table only

The table lists every entry of ``config/models.yaml``: key, id, adapter, roles,
prices in USD per million tokens and whether the entry is verified.

Without a key the command prints the table, says that no API key is set and that
the live smoke was skipped, and returns 0. It then reads the config and writes
nothing.

With a key it adds 1 measured row per model: latency, tokens, cost and the answer.
Vision models get 1 test image (a board with 8 ports, 3 of them holding a plug)
and "How many of the 8 ports have a plug in them?" with a JSON answer, reasoning
disabled and role judge. Models that cannot turn reasoning off (the table says
so) get a larger output allowance, because their reasoning tokens count against
it. Typed-decision models get the same board as 1 ``choice`` question with
options "0" to "8", and a text-only one gets the board as a short text state.

The live smoke runs in spend session S1 (cap $1 in config/experiment.yaml), so
every call is reserved before it is sent and refused once the cap would be
crossed. Calls bypass the cache so every number is measured live. Any error for
1 model (a router error or anything else) becomes that model's row, so 1 model
never stops the table. The rows are written to
``AVAIL_DATA/runs/s1_router_smoke/live_table.json``. Nothing a key could hide in
is printed: router errors are scrubbed, and the output and the returned rows are
checked against the loaded keys once more before they are printed, written or
returned.

Ported from loopkit's ``router/smoke.py``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

from .. import config, paths
from .adapters import available_adapters, get_adapter
from .core import Router, _check_key, resolve_route
from .errors import ModelCallError
from .http import HttpClient
from .models import ModelEntry, ModelsConfig, load_models
from .redact import redact, secret_forms, strip_secret

SESSION = "S1"
RUN_ID = "s1_router_smoke"
MAX_TOKENS = 300
#: for a model that cannot turn reasoning off: its reasoning tokens count against the allowance. Kept modest,
#: because an attempt that is lost stays reserved at its worst case and the S1 cap is $1
MAX_TOKENS_REASONING = 1000
# ports 1 to 8, left to right, top row then bottom row: 3 of them hold a plug
PLUGGED = (True, False, False, True, False, True, False, False)
PORTS = len(PLUGGED)
EXPECTED_PLUGGED = sum(1 for p in PLUGGED if p)
QUESTION = (f"The image shows a board with {PORTS} ports. How many of the {PORTS} ports have a plug in them? "
            'Answer with JSON {"plugged": <integer>}.')
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"plugged": {"type": "integer", "minimum": 0, "maximum": PORTS}},
    "required": ["plugged"],
    "additionalProperties": False,
}
TYPED_QUESTION_NAME = "plugged_ports"
IMAGE_STATE = (f"A board with {PORTS} ports in 2 rows of 4, shown in the image. A port either holds an orange plug "
               "or is empty.")


# --------------------------------------------------------------------------- the model table
def _price(value: float | None) -> str:
    if value is None:
        return "-"
    text = f"{value:.4f}".rstrip("0")
    whole, _, frac = text.partition(".")
    return f"{whole}.{frac.ljust(2, '0')}"


def table_rows(models: ModelsConfig) -> list[dict[str, str]]:
    """One row per entry of the ``models`` block: key, id, adapter, roles, prices, verified."""
    rows = []
    for key, entry in models.models.items():
        if not entry.verified:
            verified = "no"
        elif entry.accessed:
            verified = f"yes, {entry.accessed}"
        else:
            verified = f"yes, {entry.source or 'no date'}"
        rows.append({"key": key, "id": entry.id or "-", "adapter": entry.adapter, "roles": ", ".join(entry.roles),
                     "in": _price(entry.price_in), "out": _price(entry.price_out), "verified": verified})
    return rows


def _grid(head: tuple[str, ...], body: list[tuple[str, ...]]) -> str:
    widths = [max(len(h), *(len(b[i]) for b in body)) if body else len(h) for i, h in enumerate(head)]
    lines = [" | ".join(h.ljust(w) for h, w in zip(head, widths)).rstrip(),
             "-+-".join("-" * w for w in widths)]
    lines += [" | ".join(c.ljust(w) for c, w in zip(b, widths)).rstrip() for b in body]
    return "\n".join(lines)


def format_model_table(models: ModelsConfig) -> str:
    """The model table as text: key, id, adapter, roles, prices in USD per million tokens, verified."""
    head = ("key", "id", "adapter", "roles", "in", "out", "verified")
    body = [tuple(row[h] for h in head) for row in table_rows(models)]
    lines = [f"AVAIL model table (config/models.yaml, version {models.version}). "
             "Prices in USD per million tokens.", "", _grid(head, body)]
    pending = [name for name, entry in models.to_verify.items() if entry.model_key is None]
    if pending:
        lines += ["", f"Still to verify, so not callable yet: {', '.join(pending)}."]
    return "\n".join(lines)


# --------------------------------------------------------------------------- the smoke question
def smoke_dir() -> Path:
    return paths.run_dir(RUN_ID)


def make_board_image(path: Path) -> Path:
    """A dark board with 8 ports in 2 rows of 4. ``EXPECTED_PLUGGED`` of them hold an orange plug."""
    from PIL import Image, ImageDraw

    cols, pitch_x, pitch_y, margin = 4, 150, 170, 40
    width, height = margin * 2 + cols * pitch_x, margin * 2 + 2 * pitch_y
    img = Image.new("RGB", (width, height), (38, 44, 40))
    draw = ImageDraw.Draw(img)
    draw.rectangle([12, 12, width - 12, height - 12], fill=(52, 66, 56), outline=(90, 104, 94), width=3)
    for i, plugged in enumerate(PLUGGED):
        cx = margin + pitch_x // 2 + (i % cols) * pitch_x
        cy = margin + pitch_y // 2 + (i // cols) * pitch_y
        draw.rectangle([cx - 42, cy - 34, cx + 42, cy + 34], fill=(18, 18, 20), outline=(150, 156, 150), width=3)
        if plugged:
            draw.rectangle([cx - 34, cy - 26, cx + 34, cy + 26], fill=(240, 150, 20), outline=(120, 70, 0), width=3)
            draw.rectangle([cx - 10, cy - 60, cx + 10, cy - 26], fill=(30, 30, 34))  # the cable
        draw.text((cx - 4, cy + 44), str(i + 1), fill=(225, 225, 225))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    return path


def text_state() -> str:
    ports = ", ".join(f"port {i + 1} {'holds a plug' if plugged else 'is empty'}" for i, plugged in enumerate(PLUGGED))
    return f"A board with {PORTS} ports in 2 rows of 4. In order: {ports}."


def typed_questions() -> dict[str, Any]:
    criteria = {str(k): f"exactly {k} of the {PORTS} ports hold a plug" for k in range(PORTS + 1)}
    return {TYPED_QUESTION_NAME: {"type": "choice", "instructions": f"How many of the {PORTS} ports hold a plug?",
                                  "criteria": criteria}}


def skip_reason(entry: ModelEntry) -> str | None:
    """Why this entry cannot be called now, or None when it can."""
    routed, _note = resolve_route(entry)
    if routed.adapter not in available_adapters():
        return f"adapter {routed.adapter} is not implemented (route it through openrouter)"
    if routed.adapter != "fake" and not routed.id:
        return "no id in config/models.yaml (verified: false)"
    if routed.adapter != "fake" and not routed.verified:
        return "verified: false in config/models.yaml, so no paid call may use it yet"
    if routed.adapter == "elorian" and not routed.base_url:
        return "no base_url in config/models.yaml"
    names = get_adapter(routed.adapter).env_vars(routed)
    if names and not config.has_secret(names):
        others = f" (or {', '.join(names[1:])})" if len(names) > 1 else ""
        alt = getattr(entry, "alt_route", None)
        alt_hint = ""
        if alt is not None and getattr(alt, "id", None) and alt.adapter in available_adapters():
            alt_vars = get_adapter(alt.adapter).default_env_vars
            if alt_vars:
                alt_hint = f", or {alt_vars[0]} for its alt_route {alt.id}"
        return f"no key: set {names[0]}{others}{alt_hint} in .env"
    if names:                          # set, but a key the router cannot send in a header is refused at the call
        try:
            _check_key(config.load_secret(names, get_adapter(routed.adapter).label), names, entry.key, routed.adapter)
        except config.MissingCredentialError as exc:
            return str(exc)
    return None


def key_variables(models: ModelsConfig) -> list[str]:
    """Every variable an entry of the table could take its key from, in table order."""
    names: list[str] = []
    for entry in models.models.values():
        if entry.adapter not in available_adapters():
            continue
        candidates = list(get_adapter(entry.adapter).env_vars(entry))
        alt = getattr(entry, "alt_route", None)
        if alt is not None and alt.adapter in available_adapters():
            candidates += list(get_adapter(alt.adapter).default_env_vars)
        for name in candidates:
            if name not in names:
                names.append(name)
    return names


def _answer(adapter: str, response: Any) -> Any:
    value = response.json
    if adapter == "jev":
        block = value.get(TYPED_QUESTION_NAME) if isinstance(value, dict) else None
        choice = block.get("choice") if isinstance(block, dict) else None
        try:
            return int(choice)
        except (TypeError, ValueError):
            return choice
    if isinstance(value, dict) and "plugged" in value:
        return value["plugged"]
    return None


def call_one(key: str, entry: ModelEntry, image_path: Path, *, router: Router,
             max_tokens: int | None = None) -> dict[str, Any]:
    """One model's row. Never raises for that model: any error (router or other) becomes the row's ``error``."""
    try:
        return _call_one(key, entry, image_path, router=router, max_tokens=max_tokens)
    except Exception as exc:  # noqa: BLE001 - one model must never stop the table
        return {"model_key": key, "model_id": getattr(entry, "id", None), "real_id": getattr(entry, "id", None),
                "adapter": getattr(entry, "adapter", None), "expected": EXPECTED_PLUGGED, "ok": False,
                "error": redact(f"{type(exc).__name__}: {exc}")}


def _call_one(key: str, entry: ModelEntry, image_path: Path, *, router: Router,
              max_tokens: int | None = None) -> dict[str, Any]:
    routed, route_note = resolve_route(entry)
    row: dict[str, Any] = {"model_key": key, "model_id": routed.id, "real_id": routed.id,
                           "adapter": routed.adapter, "route": route_note, "expected": EXPECTED_PLUGGED, "ok": False}
    no_off_switch = get_adapter(routed.adapter).reasoning_note("disabled", routed) is not None
    allowance = max_tokens if max_tokens is not None else (MAX_TOKENS_REASONING if no_off_switch else MAX_TOKENS)
    kwargs: dict[str, Any] = dict(max_tokens=allowance, reasoning="disabled", use_cache=False, phase="smoke")
    if routed.adapter == "jev":  # the typed-question protocol (Jev, and d1 through its own endpoint)
        kwargs.update(role="decider", extra={"questions": typed_questions()})
        if entry.vision:
            kwargs.update(messages=IMAGE_STATE, images=[image_path])
            row["input"] = "image + typed choice question"
        else:
            kwargs.update(messages=text_state())
            row["input"] = "text state + typed choice question"
    elif entry.vision:
        role = "test" if routed.adapter == "fake" else "judge"
        kwargs.update(messages=QUESTION, images=[image_path], json_schema=ANSWER_SCHEMA, role=role)
        row["input"] = "image + question"
    else:
        kwargs.update(messages=f"{text_state()}\n\n{QUESTION}", json_schema=ANSWER_SCHEMA, role="judge")
        row["input"] = "text state + question"
    notes: list[str] = []
    try:
        try:
            response = router.chat(key, **kwargs)
        except ModelCallError as exc:
            # some reasoning models refuse reasoning disabled. That refusal costs nothing, so retry once with low
            if exc.status == 400 and "reason" in str(exc).lower():
                notes.append("the model refused reasoning disabled, retried with low")
                kwargs["reasoning"] = "low"
                response = router.chat(key, **kwargs)
            else:
                raise
    except Exception as exc:  # noqa: BLE001 - any error is this model's row, never the end of the table
        row.update(error=redact(f"{type(exc).__name__}: {exc}"), note=" | ".join(notes) or None)
        return row
    if response.raw.get("reasoning_note"):
        notes.append(str(response.raw["reasoning_note"]))
    if response.truncated:
        notes.append(f"the answer was cut at max_tokens ({allowance}). Run again with a larger --max-tokens")
    answer = _answer(routed.adapter, response)
    row.update(
        ok=True, real_id=response.model, latency_ms=round(response.latency_ms, 1),
        input_tokens=response.input_tokens, output_tokens=response.output_tokens,
        reasoning_tokens=response.reasoning_tokens, usd=response.usd, usd_source=response.usd_source,
        answer=answer, correct=answer == EXPECTED_PLUGGED, json_mode=response.json_mode,
        json_error=response.json_error, reasoning_setting=response.reasoning_setting,
        finish_reason=response.finish_reason, max_tokens=allowance, text=response.text[:300],
        note=" | ".join(notes) or None,
    )
    return row


def _fmt(value: Any, spec: str = "") -> str:
    if value is None:
        return "-"
    if spec and isinstance(value, (int, float)):
        return format(value, spec)
    return " ".join(str(value).split())  # one line, whatever a provider put in the text


def _say(text: str = "") -> None:
    """Print ``text``. A character the output stream cannot encode is written as an escape, never an error."""
    try:
        print(text, flush=True)
    except UnicodeEncodeError:  # a redirected Windows stream is cp1252, and a provider's error text may not be
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.encode(encoding, errors="backslashreplace").decode(encoding), flush=True)


def format_smoke_table(rows: list[dict[str, Any]]) -> str:
    """The live smoke rows as text: 1 row per model with measured latency and cost."""
    head = ("model key", "real id", "latency ms", "in tok", "out tok", "usd", "usd_source",
            f"answer (expected {EXPECTED_PLUGGED})")
    body = []
    for r in rows:
        if r.get("ok"):
            answer = f"{_fmt(r.get('answer'))} ({'right' if r.get('correct') else 'wrong'})"
            if r.get("finish_reason") == "length":
                answer += " cut at max_tokens"
            if r.get("json_error"):
                answer += f" json_error: {_fmt(r['json_error'])[:60]}"
        else:
            answer = f"ERROR {_fmt(r.get('error'))[:90]}"
        body.append((str(r["model_key"]), _fmt(r.get("real_id")), _fmt(r.get("latency_ms"), ".0f"),
                     _fmt(r.get("input_tokens")), _fmt(r.get("output_tokens")), _fmt(r.get("usd"), ".6f"),
                     _fmt(r.get("usd_source")), answer))
    return _grid(head, body)


def _free(entry: ModelEntry) -> bool:
    """A call that projects $0 (the FakeModel at its zero prices) needs no spend session."""
    return entry.adapter == "fake" and not (entry.price_in or entry.price_out)


def _loaded_keys(models: ModelsConfig) -> list[str]:
    return [v for v in (config.env(n) for n in key_variables(models)) if v and len(v) >= 6]


def _guard(text: str, secrets: list[str]) -> str:
    """``text`` without any loaded key, as written or JSON-escaped (the table is written as JSON)."""
    for secret in secrets:
        for form in secret_forms(secret):
            text = text.replace(form, "[redacted]")
    return text


def plan_smoke(models: ModelsConfig, *, include_fake: bool = False,
               only: list[str] | None = None) -> tuple[list[tuple[str, ModelEntry]], list[dict[str, str]]]:
    """(the entries the smoke would call, the entries it skips with the reason)."""
    plan: list[tuple[str, ModelEntry]] = []
    skipped: list[dict[str, str]] = []
    for key, entry in models.models.items():
        if entry.adapter == "fake" and not include_fake:
            continue
        if only and key not in only:
            continue
        try:
            reason = skip_reason(entry)
        except Exception as exc:  # noqa: BLE001 - one model must never stop the table
            reason = redact(f"could not check this entry: {type(exc).__name__}: {exc}")
        if reason:
            skipped.append({"model_key": key, "reason": reason})
        else:
            plan.append((key, entry))
    return plan, skipped


def run_smoke(*, include_fake: bool = False, only: list[str] | None = None, models: ModelsConfig | None = None,
              router: Router | None = None, http: HttpClient | None = None, out_dir: Path | None = None,
              max_tokens: int | None = None, echo: bool = True) -> dict[str, Any]:
    """Run the live smoke and return ``{"rows", "skipped", ...}`` (also written to live_table.json).

    ``router`` defaults to ``Router(session="S1")``, or a session-less Router when
    only the free FakeModel is called. ``http`` replaces the HTTP client (tests).
    """
    cfg = models if models is not None else load_models()
    plan, skipped = plan_smoke(cfg, include_fake=include_fake, only=only)
    secrets = _loaded_keys(cfg)

    def say(text: str = "") -> None:
        if echo:
            _say(_guard(text, secrets))

    paid = any(not _free(entry) for _key, entry in plan)
    if router is None:
        router = Router(session=SESSION if paid else None, models=cfg, http=http)
    session_note = None
    if paid and router.session is not None:
        session_note = (f"spend session {router.session}: cap ${router.cap_usd:.2f}, spent ${router.spent():.6f} "
                        f"so far, ${router.remaining():.6f} left")
        say(session_note)
    out = Path(out_dir) if out_dir is not None else smoke_dir()
    out.mkdir(parents=True, exist_ok=True)
    image_path = make_board_image(out / "board.png")
    spent_before = router.ledger.spent()

    rows: list[dict[str, Any]] = []
    for key, entry in plan:
        say(f"calling {key} ({entry.id}) ...")
        row = call_one(key, entry, image_path, router=router, max_tokens=max_tokens)
        for secret in secrets:  # no loaded key in a returned row either, wherever a server echoed it
            row = strip_secret(row, secret)
        rows.append(row)

    spent = router.ledger.spent() - spent_before
    result = {
        "date": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "question": QUESTION, "expected_plugged": EXPECTED_PLUGGED, "plugged": list(PLUGGED),
        "image": str(image_path), "reasoning": "disabled", "session": router.session,
        "rows": rows, "skipped": skipped, "spend_usd_this_run": round(spent, 8), "session_note": session_note,
    }
    table_path = out / "live_table.json"
    table_path.write_text(_guard(json.dumps(result, indent=2, default=str), secrets) + "\n", encoding="utf-8")

    say()
    say(format_smoke_table(rows) if rows else "No model was called (every entry was skipped).")
    if skipped:
        say()
        say("Skipped:")
        for s in skipped:
            say(f"  {s['model_key']}: {s['reason']}")
    say()
    say(f"Spend this run: ${spent:.6f}. Table: {table_path}")
    result["table_path"] = str(table_path)
    return result


def main(argv: list[str] | None = None) -> int:
    """The ``avail models`` command. Returns 0, or 1 when a live smoke row failed."""
    parser = argparse.ArgumentParser(
        prog="avail models",
        description="Print the model table. With an API key in .env, also make 1 small call per model and print "
                    "its measured latency and cost, under the S1 spend cap.")
    parser.add_argument("--only", default="", help="comma-separated model keys for the live smoke (default: all)")
    parser.add_argument("--fake", action="store_true", help="also call the offline FakeModel ($0, no key)")
    parser.add_argument("--no-live", action="store_true", help="print the table only")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help=f"output allowance per call (default: {MAX_TOKENS}, or {MAX_TOKENS_REASONING} for a "
                             "model that cannot turn reasoning off)")
    args = parser.parse_args(argv)
    models = load_models()
    _say(format_model_table(models))
    if args.no_live:
        return 0
    only = [k.strip() for k in args.only.split(",") if k.strip()] or None
    unknown = [k for k in (only or []) if k not in models.models]
    if unknown:
        print(f"\navail models: unknown model key(s) {', '.join(unknown)}. Known: {', '.join(models.models)}.",
              file=sys.stderr)
        return 2
    variables = key_variables(models)
    if not any(config.env(name) for name in variables):
        _say(f"\nNo API key is set ({', '.join(variables)} are all empty), so the live smoke was skipped. "
             "No paid call was made.\nPut a key in .env and run `avail models` again for 1 measured row per "
             "model, under the S1 cap.")
        if not args.fake:
            return 0
    _say()
    result = run_smoke(include_fake=args.fake, only=only, models=models, max_tokens=args.max_tokens)
    failed = [r for r in result["rows"] if not r.get("ok")]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
