"""The AVAIL tool server in station mode: MCP tools over a cell and the AVAIL core it is handed.

The robot tools keep ASPIRE's CAP names and argument shapes, and ``wam_act`` and ``stop_wam`` the names in GR00T 2's
figure. ``docs/aspire_tool_diff.md`` lists every difference from ASPIRE. Nothing here imports ``avail.station``.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
import sys
import time
import unicodedata
from collections.abc import Callable, Mapping
from typing import Any, BinaryIO

import numpy as np

from ..schemas.kinds import QUESTION_TYPES

__all__ = ["INSTRUCTIONS", "PROTOCOL_VERSION", "ToolServer", "UnknownTool"]

PROTOCOL_VERSION = "2025-06-18"
_PROTOCOLS = (PROTOCOL_VERSION, "2025-03-26", "2024-11-05")     # a client asking for 1 of these gets it back
_VERSION = "0.1.0"                                               # the package version in pyproject.toml
_ID = r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,127}$"                    # every name or id an argument carries
_SMALL = 64                                                      # arrays up to this many numbers go out in full
_LISTS = ("skills", "steps", "probes", "cameras", "arms")        # names a cell or core may list for the enums

INSTRUCTIONS = (                                                 # the skill note's 4 rules, sent at initialize
    "1. A success flag, a pose or a picture does not make a step done. Raise a claim with avail_verify. "
    "2. Follow the fork you get back. You may pick among its options. You may not skip it. "
    "3. If you are unsure which predicate applies, ask with avail_ask_human. "
    "4. Finish with avail_done. If it is not ok, the task is not finished."
)


class UnknownTool(LookupError):
    """A tool name this server does not have."""


def _obj(required: tuple[str, ...] = (), **properties: Any) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "additionalProperties": False,
            **({"required": list(required)} if required else {})}


def _name(names: list[str] | None, description: str) -> dict[str, Any]:
    return {"type": "string", "description": description, **({"enum": names} if names else {"pattern": _ID})}


_TYPES = {  # the JSON types, read on plain values: a boolean is no number, and NaN or infinity is none
    "number": lambda v: type(v) is int or (type(v) is float and math.isfinite(v)), "integer": lambda v: type(v) is int,
    "string": lambda v: type(v) is str, "array": lambda v: type(v) is list, "object": lambda v: type(v) is dict}


def _check(value: Any, schema: Mapping[str, Any], where: str) -> str | None:
    """The first way ``value`` breaks ``schema``, as a sentence, or None. Covers the subset these tools use."""
    kinds = [schema["type"]] if isinstance(schema.get("type"), str) else schema.get("type", [])
    if kinds and not any(_TYPES[kind](value) for kind in kinds):
        return f"{where} must be {' or '.join(kinds)}"
    if "enum" in schema and value not in schema["enum"]:
        return f"{where} must be 1 of {', '.join(map(str, schema['enum']))}"
    if isinstance(value, str) and not re.search(schema.get("pattern", ""), value):
        return f"{where} must be a name of up to 128 letters, digits and _ . : @ -, starting with a letter or digit"
    if _TYPES["number"](value) and not schema.get("minimum", -math.inf) <= value <= schema.get("maximum", math.inf):
        return f"{where} must be from {schema.get('minimum')} to {schema.get('maximum')}"
    if isinstance(value, list) and not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", math.inf):
        return f"{where} must hold from {schema.get('minItems', 0)} to {schema.get('maxItems')} items"
    known = schema.get("properties", {})
    if isinstance(value, dict) and (missing := [key for key in schema.get("required", []) if key not in value]):
        return f"{where} needs {', '.join(missing)}"
    extra = sorted(set(value) - set(known)) if isinstance(value, dict) and "properties" in schema else []
    if extra and schema.get("additionalProperties") is False:
        return f"{where} does not take {', '.join(extra)}. It takes {', '.join(known) or 'nothing'}"
    parts = enumerate(value) if isinstance(value, list) else value.items() if isinstance(value, dict) else ()
    for key, item in parts:
        inner = schema.get("items", {}) if isinstance(value, list) else known.get(key, {})
        if found := _check(item, inner, f"{where}[{key}]" if isinstance(value, list) else f"{where}.{key}"):
            return found
    if "oneOf" in schema and sum(_check(value, option, where) is None for option in schema["oneOf"]) != 1:
        return f"{where} needs exactly 1 of {', '.join(option['required'][0] for option in schema['oneOf'])}"
    return None


def _spells_accept(value: Any) -> str | None:
    """The first string in ``value`` that spells accept in any case or width, or None."""
    if isinstance(value, str):
        return value if "accept" in unicodedata.normalize("NFKC", value).casefold() else None
    parts = value.values() if isinstance(value, dict) else value if isinstance(value, list) else ()
    return next((hit for hit in map(_spells_accept, parts) if hit is not None), None)


def _plain(value: Any) -> Any:
    """``value`` as JSON types. An array of more than 64 numbers becomes its shape and dtype, so no image goes out."""
    value = value.item() if isinstance(value, np.generic) else value
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        small = value.size <= _SMALL and value.dtype.kind in "biuf"
        return _plain(value.tolist()) if small else {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(item) for item in value]
    if callable(getattr(value, "model_dump", None)):
        return _plain(value.model_dump(mode="json"))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _plain(getattr(value, field.name)) for field in dataclasses.fields(value)}
    return str(value)


def _unit_pose(pose: Mapping[str, Any]) -> dict[str, Any]:
    """``pose`` as ``{position, quaternion_wxyz, frame}`` with a unit quaternion, made from rpy_deg if that came."""
    if "rpy_deg" in pose:   # fixed-axis XYZ, R = Rz(yaw) Ry(pitch) Rx(roll), as ASPIRE reads it
        cr, sr, cp, sp, cy, sy = (f(math.radians(a) / 2) for a in pose["rpy_deg"] for f in (math.cos, math.sin))
        quat = [cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy, cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy]
    else:
        quat = [float(q) for q in pose["quaternion_wxyz"]]
    norm = math.sqrt(sum(q * q for q in quat))
    if abs(norm - 1.0) > 1e-3:
        raise ValueError(f"quaternion_wxyz has length {norm:.4f}. Send a unit quaternion [w, x, y, z]")
    return {"position": [float(v) for v in pose["position"]], "quaternion_wxyz": [q / norm for q in quat],
            "frame": pose["frame"]}


def _names(owner: Any, attr: str) -> list[str] | None:
    """``owner.<attr>`` as a list of names, or None when it is missing or holds something else."""
    value = getattr(owner, attr, None)
    value = value() if callable(value) else value
    names = [getattr(one, "name", one) for one in value] if isinstance(value, (list, tuple, dict)) else []
    return names if names and all(isinstance(name, str) for name in names) else None


class ToolServer:
    """MCP tools over a ``cell`` and the AVAIL ``core``. ``call_tool`` runs 1 in process for the scripted driver.
    ``serve_stdio`` speaks MCP 2025-06-18 over stdio, 1 JSON-RPC 2.0 message per line.

    The cell offers ``observe``, ``robot_state``, ``move_to_pose``, ``set_gripper``, ``run_skill`` and ``stop``,
    the core ``verify``, ``probe``, ``ask_human`` and ``done``. Each method gets its tool's arguments as keywords
    of the same names, None for an optional one left out. ``arm`` defaults to the first arm, and ``target_pose``
    arrives as ``{position, quaternion_wxyz, frame}`` with a unit quaternion. ``skills``, ``steps``, ``probes``,
    ``cameras`` or ``arms`` listed on either object (names, or objects with a ``name``) become the schemas' enums.
    ``describe()`` adds its dict to ``get_capability_graph``. No argument can carry a verdict: schemas refuse
    unknown keys, none names a decision, and a string that spells accept is refused. ``avail_done`` is ok only on
    the core's boolean ok with nothing unresolved. Every call goes to ``log("tool_call", payload)``, which fits
    ``LedgerWriter.append``. With ``strict``, refused arguments and failures raise after they are logged.
    """

    def __init__(self, cell: Any, core: Any, log: Callable[[str, dict[str, Any]], Any] | None = None, *,
                 strict: bool = False) -> None:
        self.cell, self.core, self.log, self.strict = cell, core, log, strict
        self.lists = {key: _names(cell, key) or _names(core, key) for key in _LISTS}
        c, k, first = cell, core, (self.lists["arms"] or [None])[0]
        item, arm = _name(None, "the item, for example a cable id"), _name(self.lists["arms"], "Default: the 1st arm")
        step = _name(self.lists["steps"], "the task step, for example pick, carry, insert or release")
        vector = {n: {"type": "array", "items": {"type": "number"}, "minItems": n, "maxItems": n} for n in (3, 4)}
        pose = {**_obj(("position", "frame"), position=vector[3], quaternion_wxyz=vector[4], rpy_deg=vector[3],
                       frame={"type": "string", "pattern": _ID}),
                "oneOf": [{"required": ["quaternion_wxyz"]}, {"required": ["rpy_deg"]}],
                "description": "position (m), a frame, and exactly 1 of quaternion_wxyz or rpy_deg (fixed-axis XYZ)"}
        rows = [
            ("get_capability_graph", "List the tools and the skills, steps, probes, cameras and arms of this cell.",
             _obj(), self._graph),
            ("observe", "Capture fresh camera frames: each camera's intrinsics, extrinsics, time and image size. "
             "Pixels are not sent.", _obj(cameras={"type": "array", "items": _name(self.lists["cameras"], "a camera")}),
             lambda a: c.observe(cameras=a.get("cameras"))),
            ("cap_get_robot_state", "Read each arm's joints, joint torque estimate, gripper and tool pose.", _obj(),
             lambda a: c.robot_state()),
            ("cap_move_to_pose", "Move an arm's tool to an absolute pose. The cell sets the speed and the force cap.",
             _obj(("target_pose",), target_pose=pose, arm=arm),
             lambda a: c.move_to_pose(target_pose=_unit_pose(a["target_pose"]), arm=a.get("arm", first))),
            ("cap_set_gripper", "Set a gripper from 0 closed to 1 open.",
             _obj(("position",), position={"type": "number", "minimum": 0, "maximum": 1}, arm=arm),
             lambda a: c.set_gripper(position=float(a["position"]), arm=a.get("arm", first))),
            ("wam_act", "Run 1 skill of the stand-in System 1 on an item. Returns its feedback and its own success "
             "flag. The flag does not make the step done.",
             _obj(("skill", "item"), skill=_name(self.lists["skills"], "the skill to run"), item=item,
                  port={"type": ["integer", "string"], "pattern": _ID, "description": "Default: the item's port"}),
             lambda a: c.run_skill(skill=a["skill"], item=a["item"], port=a.get("port"))),
            ("stop_wam", "Stop all motion and hold.", _obj(), lambda a: c.stop()),
            ("avail_verify", "Raise the claim of a step for an item and get the verdict from evidence: accept, "
             "reject or abstain, the evidence summary and, when it is not accept, the fork to follow.",
             _obj(("item", "step"), item=item, step=step), lambda a: k.verify(item=a["item"], step=a["step"])),
            ("avail_probe", "Run 1 named probe on the claim of a step and get its evidence. Evidence alone does "
             "not mark a step done.", _obj(("item", "step", "probe"), item=item, step=step,
                                           probe=_name(self.lists["probes"], "the probe, for example probe.tug")),
             lambda a: k.probe(item=a["item"], step=a["step"], probe=a["probe"])),
            ("avail_ask_human", "Ask a person about the claim of a step and wait for the answer.",
             _obj(("item", "step", "question"), item=item, step=step,
                  question={"type": "string", "enum": list(QUESTION_TYPES)},
                  options={"type": "array", "items": _name(None, "an answer to offer"), "maxItems": 16}),
             lambda a: k.ask_human(item=a["item"], step=a["step"], question=a["question"], options=a.get("options"))),
            ("avail_done", "The done gate for a cycle. ok only when every step the task requires has an accept "
             "verdict or a person's sign-off. Otherwise it returns the unresolved list and the cycle stays open.",
             _obj(("cycle_id",), cycle_id=_name(None, "the cycle to close")), lambda a: k.done(cycle_id=a["cycle_id"])),
        ]
        self._tools = {name: ({"name": name, "description": text, "inputSchema": schema}, run)
                       for name, text, schema, run in rows}

    def tools(self) -> list[dict[str, Any]]:      # what tools/list sends, in a fixed order
        return [spec for spec, _run in self._tools.values()]

    def call_tool(self, name: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Run 1 tool and get a dict whose ``ok`` is a real boolean. Refused arguments reach neither the cell nor
        the core. They and any exception come back as ``{"ok": false, "error": ...}``. An unknown tool raises."""
        start = time.perf_counter()
        args = {} if arguments is None else _plain(arguments)   # numpy scalars and tuples read as JSON would
        if not isinstance(name, str) or name not in self._tools:
            self._log(name, args, {"ok": False, "error": "unknown tool"}, start)
            raise UnknownTool(f"unknown tool {name!r}. The tools are {', '.join(self._tools)}")
        spec, run = self._tools[name]
        hit = _spells_accept(args)
        problem = _check(args, spec["inputSchema"], "arguments") if hit is None else (
            f"{hit!r} spells accept. Only evidence or a person marks a step done: raise a claim with avail_verify")
        value, failure = {"ok": False, "error": problem}, ValueError(problem)
        if problem is None:
            try:
                value, failure = self._envelope(name, run(args)), None
            except Exception as exc:  # a failing cell or core gives an error result, and the server goes on
                value, failure = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, exc
        self._log(name, args, value, start)
        if self.strict and failure is not None:
            raise failure
        return value

    def handle(self, message: Any) -> dict[str, Any] | None:
        """The reply to 1 JSON-RPC message, or None for a notification or a client's own reply."""
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return _reply(message.get("id") if isinstance(message, dict) else None,
                          error=(-32600, "Invalid Request: send 1 JSON-RPC 2.0 object per line"))
        if "method" not in message or "id" not in message:
            return None
        method, rid, params = message["method"], message["id"], message.get("params")
        params = params if isinstance(params, dict) else {}
        try:
            if method == "initialize":
                asked = params.get("protocolVersion")
                return _reply(rid, {"protocolVersion": asked if asked in _PROTOCOLS else PROTOCOL_VERSION,
                                    "capabilities": {"tools": {"listChanged": False}},
                                    "serverInfo": {"name": "avail", "version": _VERSION}, "instructions": INSTRUCTIONS})
            if method in ("ping", "tools/list"):
                return _reply(rid, {"tools": self.tools()} if method == "tools/list" else {})
            if method == "tools/call":
                value = self.call_tool(params.get("name"), params.get("arguments"))
                return _reply(rid, {"content": [{"type": "text", "text": json.dumps(value)}],
                                    "structuredContent": value, "isError": not value["ok"]})
            return _reply(rid, error=(-32601, f"Method not found: {method}"))
        except UnknownTool:
            return _reply(rid, error=(-32602, f"Unknown tool: {params.get('name')}"))
        except Exception as exc:  # a reply that cannot be built becomes an error reply, and the loop goes on
            return _reply(rid, error=(-32603, f"Internal error: {type(exc).__name__}: {exc}"))

    def serve_stdio(self, stdin: BinaryIO | None = None, stdout: BinaryIO | None = None) -> None:
        """Answer each JSON-RPC line of stdin with 1 line on the process's own stdout until stdin closes. Meanwhile
        ``sys.stdout`` points at stderr, so a stray print cannot break the protocol. Native code still can."""
        source, sink, saved = sys.stdin.buffer if stdin is None else stdin, stdout or sys.__stdout__.buffer, sys.stdout
        sys.stdout = sys.stderr if stdout is None else saved
        try:
            for line in (raw for raw in source if raw.strip()):
                try:
                    reply = self.handle(json.loads(line))
                except (ValueError, RecursionError):  # bad JSON, bad UTF-8 or nesting too deep
                    reply = _reply(None, error=(-32700, "Parse error: each line must be 1 JSON object"))
                if reply is not None:
                    sink.write(json.dumps(reply).encode("ascii") + b"\n")
                    sink.flush()
        finally:
            sys.stdout = saved

    def _graph(self, _args: dict[str, Any]) -> dict[str, Any]:
        graph = {"server": {"name": "avail", "version": _VERSION, "protocol": PROTOCOL_VERSION},
                 "tools": [{"name": spec["name"], "description": spec["description"]} for spec in self.tools()],
                 **{key: names for key, names in self.lists.items() if names}}
        return {**graph, **{key: owner.describe() for key, owner in (("cell", self.cell), ("core", self.core))
                            if callable(getattr(owner, "describe", None))}}

    @staticmethod
    def _envelope(name: str, raw: Any) -> dict[str, Any]:
        """A result as a dict with ``ok`` first. ``avail_done`` fails closed on anything but a clean ok."""
        plain = _plain(raw)
        value = plain if isinstance(plain, dict) else {} if plain is None else {"result": plain}
        clean = value.get("ok") is True and not value.get("unresolved")
        ok = clean if name == "avail_done" else bool(value.get("ok", True))
        return {"ok": ok, **{key: item for key, item in value.items() if key != "ok"}}

    def _log(self, name: Any, args: Any, value: dict[str, Any], start: float) -> None:
        if self.log is not None:
            error = {"error": value["error"]} if "error" in value and not value["ok"] else {}
            self.log("tool_call", {"tool": str(name), "arguments": _plain(args), "ok": value["ok"],
                                   "seconds": round(time.perf_counter() - start, 6), **error})


def _reply(rid: Any, result: dict[str, Any] | None = None, error: tuple[int, str] | None = None) -> dict[str, Any]:
    body = {"result": result} if error is None else {"error": {"code": error[0], "message": error[1]}}
    return {"jsonrpc": "2.0", "id": rid, **body}
