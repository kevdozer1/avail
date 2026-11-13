"""AVAIL in front of another MCP server: every upstream tool passes through unchanged, and 4 tools are added.

    uv run avail proxy --upstream-url http://127.0.0.1:8222/mcp --items 5 --steps insert    # ASPIRE's CAP server
    uv run avail proxy --upstream-cmd "python -m avail.agent.mock_aspire --latched" --items 1 --steps insert

The proxy is an MCP server over stdio for the agent and an MCP client of the upstream server. ``tools/list`` is the
upstream's list, each tool exactly as the upstream sent it, then ``avail_verify``, ``avail_probe``,
``avail_ask_human`` and ``avail_done``. A call to an upstream tool is sent on with its arguments untouched, and the
upstream's result comes back to the agent unchanged. Every call, of either kind, is written to the ledger.

The 4 added tools keep the claims of a cycle: (item, step) pairs the task lists. ``avail_probe`` runs a probe through
the upstream's own motion tools (``cap_get_robot_state``, ``cap_move_to_pose``): the tug commands a small lift with the
gripper closed, reads how far the tool really went and puts it back. ``avail_verify`` decides a claim from that
evidence with the thresholds it is given, as a rule witness does: held fast accepts, a free lift rejects, between them
it abstains and the fork asks a person. ``avail_done`` is ok only when every (item, step) has an accept verdict or a
person's sign-off. The agent has no argument that can mark a step done.

What this was tested against: a mock of ASPIRE's server (``avail.agent.mock_aspire``) with ASPIRE's real tool names
and schemas. ASPIRE's real server was not run. On a real station the tug's threshold must be measured (``avail
audit-check``), and ASPIRE's ``cap_move_to_pose`` has no force cap of its own, so the station must limit the pull.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable
from typing import Any

AVAIL_TOOLS = ("avail_verify", "avail_probe", "avail_ask_human", "avail_done")


# ----------------------------------------------------------------------------- the upstream, as an MCP client
class StdioUpstream:
    """An upstream MCP server run as a subprocess, 1 JSON-RPC message per line."""

    def __init__(self, command: str | list[str]):
        argv = shlex.split(command, posix=False) if isinstance(command, str) else list(command)
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8")
        self._id = 0

    def request(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self._id, "method": method,
                                          "params": params or {}}) + "\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait(timeout=10)


class HttpUpstream:
    """An upstream MCP server over streamable HTTP with JSON replies, as ASPIRE's CAP server is (stateless)."""

    def __init__(self, url: str, timeout_s: float = 600.0):
        self.url, self.timeout_s, self._id = url, timeout_s, 0

    def request(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        body = json.dumps({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}}).encode()
        req = urllib.request.Request(self.url, data=body, method="POST", headers={
            "Content-Type": "application/json", "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2025-06-18"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def close(self) -> None:
        pass


class InProcessUpstream:
    """An upstream object with ``handle(message) -> reply``, called in this process (tests)."""

    def __init__(self, server: Any):
        self.server, self._id = server, 0

    def request(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        return self.server.handle({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}})

    def close(self) -> None:
        pass


def _value(reply: dict) -> dict:
    """The structured value of a tools/call reply, read the way ASPIRE's server writes it."""
    result = reply.get("result") or {}
    if isinstance(result.get("structuredContent"), dict):
        return result["structuredContent"]
    for part in result.get("content", []):
        if part.get("type") == "text":
            try:
                return json.loads(part["text"])
            except ValueError:
                break
    return {"ok": False, "error": "no structured result"}


# ----------------------------------------------------------------------------- the probes, through upstream tools
def _pose(state_value: dict, arm: str) -> dict:
    """An arm's end-effector pose from ``cap_get_robot_state``'s value (ASPIRE's RobotState fields)."""
    state = state_value.get("result") or state_value
    poses = state["end_effector_poses"]
    pose = poses[arm] if arm in poses else next(iter(poses.values()))
    return {"position": [float(v) for v in pose["position"]], "quaternion_wxyz": [float(v) for v in pose["quaternion_wxyz"]],
            "frame": pose.get("frame") or state.get("base_frame", "base")}


def tug(call: Callable[[str, dict], dict], arm: str, lift_mm: float = 3.0) -> dict:
    """Command a lift of ``lift_mm`` with the gripper as it is, read how far the tool went, and put it back."""
    before = _pose(call("cap_get_robot_state", {}), arm)
    target = dict(before, position=[before["position"][0], before["position"][1], before["position"][2] + lift_mm / 1000])
    call("cap_move_to_pose", {"target_pose": target, "arm": arm})
    after = _pose(call("cap_get_robot_state", {}), arm)
    call("cap_move_to_pose", {"target_pose": before, "arm": arm})
    moved = (after["position"][2] - before["position"][2]) * 1000.0
    return {"probe": "probe.tug", "commanded_mm": lift_mm, "moved_mm": round(moved, 4)}


PROBES = {"probe.tug": tug}


# ----------------------------------------------------------------------------- the 4 added tools
class ProxyCore:
    """The claims of 1 cycle and the done gate. ``required`` is every (item, step) the task asks for. A claim is
    decided by the tug: held within ``accept_mm`` accepts, a lift of ``reject_mm`` or more rejects, between abstains.
    ``human(item, step, question, options)`` returns ``{"answer": ..., "signed_off": bool}``, or None when no person
    is attached, which leaves the claim open."""

    def __init__(self, call: Callable[[str, dict], dict], required: list[tuple[str, str]], *, arm: str = "left",
                 accept_mm: float = 0.5, reject_mm: float = 2.0, human: Callable[..., dict | None] | None = None):
        self.call, self.required, self.arm = call, [(str(i), str(s)) for i, s in required], arm
        self.accept_mm, self.reject_mm, self.human = accept_mm, reject_mm, human
        self.settled: dict[tuple[str, str], str] = {}          # (item, step) -> "verdict" or "person"
        self.evidence: dict[tuple[str, str], list[dict]] = {}

    def probe(self, item: str, step: str, probe: str) -> dict:
        if probe not in PROBES:
            raise ValueError(f"unknown probe {probe!r}. Known: {', '.join(PROBES)}")
        ev = PROBES[probe](self.call, self.arm)
        self.evidence.setdefault((str(item), str(step)), []).append(ev)
        return {"ok": True, "evidence": ev, "note": "evidence alone does not mark a step done"}

    def verify(self, item: str, step: str) -> dict:
        key = (str(item), str(step))
        if key not in self.required:
            raise ValueError(f"{item} {step} is not a step this task requires")
        ev = tug(self.call, self.arm)
        self.evidence.setdefault(key, []).append(ev)
        moved = ev["moved_mm"]
        decision = "accept" if moved <= self.accept_mm else "reject" if moved >= self.reject_mm else "abstain"
        if decision == "accept":
            self.settled[key] = "verdict"
        fork = None if decision == "accept" else {"options": ["retry", "ask_human"] if decision == "reject" else
                                                  ["ask_human"], "question": "confirm" if decision == "abstain" else "choose"}
        return {"ok": True, "decision": decision, "reason": f"tug moved {moved:.2f} mm of {ev['commanded_mm']:.0f} mm",
                "evidence": ev, "fork": fork}

    def ask_human(self, item: str, step: str, question: str, options: list | None = None) -> dict:
        key = (str(item), str(step))
        answer = self.human(item, step, question, options) if self.human else None
        if answer is None:
            return {"ok": False, "error": "no person is attached to this proxy. The claim stays open"}
        if answer.get("signed_off"):
            self.settled[key] = "person"
        return {"ok": True, **answer}

    def done(self, cycle_id: str) -> dict:
        unresolved = [f"{i}.{s}" for i, s in self.required if (i, s) not in self.settled]
        return {"ok": not unresolved, "cycle_id": cycle_id, "unresolved": unresolved}


def _avail_schemas() -> dict[str, dict]:
    def obj(required, **props):
        return {"type": "object", "properties": props, "required": list(required), "additionalProperties": False}

    name = {"type": "string", "pattern": r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,127}$"}
    return {
        "avail_verify": ("Raise the claim of a step for an item and get the verdict from evidence: accept, reject or "
                         "abstain, the decisive reason and, when it is not accept, the fork to follow.",
                         obj(("item", "step"), item=name, step=name)),
        "avail_probe": ("Run 1 named probe through the robot's own tools and get its evidence. Evidence alone does "
                        "not mark a step done.", obj(("item", "step", "probe"), item=name, step=name, probe=name)),
        "avail_ask_human": ("Ask a person about the claim of a step and wait for the answer.",
                            obj(("item", "step", "question"), item=name, step=name,
                                question={"type": "string", "enum": ["confirm", "choose", "guide_marks", "takeover"]},
                                options={"type": "array", "items": name, "maxItems": 16})),
        "avail_done": ("The done gate. ok only when every step the task requires has an accept verdict or a "
                       "person's sign-off. Otherwise it lists what is unresolved.", obj(("cycle_id",), cycle_id=name)),
    }


class AvailProxy:
    """The MCP server the agent talks to. ``handle`` answers 1 JSON-RPC message."""

    def __init__(self, upstream: Any, core_factory: Callable[[Callable[[str, dict], dict]], ProxyCore],
                 log: Callable[[str, dict], Any] | None = None):
        self.upstream, self.log = upstream, log
        self.core = core_factory(self._upstream_value)
        self._schemas = _avail_schemas()
        self._upstream_tools: list[dict] | None = None

    def _upstream_value(self, name: str, arguments: dict) -> dict:
        """An upstream tool called by the proxy itself (a probe), logged as such."""
        reply = self.upstream.request("tools/call", {"name": name, "arguments": arguments})
        self._log(name, arguments, reply, by="avail_probe")
        return _value(reply)

    def upstream_tools(self) -> list[dict]:
        if self._upstream_tools is None:
            self._upstream_tools = self.upstream.request("tools/list")["result"]["tools"]
            clash = sorted(set(AVAIL_TOOLS) & {t["name"] for t in self._upstream_tools})
            if clash:
                raise ValueError(f"the upstream already has {', '.join(clash)}")
        return self._upstream_tools

    def tools(self) -> list[dict]:
        added = [{"name": n, "description": d, "inputSchema": s} for n, (d, s) in self._schemas.items()]
        return list(self.upstream_tools()) + added

    def _log(self, name: str, arguments: Any, reply: dict, by: str = "agent") -> None:
        if self.log is not None:
            result = reply.get("result") or {}
            self.log("tool_call", {"tool": str(name), "arguments": arguments, "by": by,
                                   "ok": not result.get("isError", "error" in reply), "ts": time.time()})

    def handle(self, message: dict) -> dict | None:
        if not isinstance(message, dict) or "id" not in message or "method" not in message:
            return None
        rid, method, params = message["id"], message["method"], message.get("params") or {}
        if method == "initialize":
            info = self.upstream.request("initialize", params).get("result", {})
            return {"jsonrpc": "2.0", "id": rid, "result": {
                "protocolVersion": info.get("protocolVersion", "2025-06-18"), "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "avail-proxy", "version": "0.1.0", "upstream": info.get("serverInfo")},
                "instructions": "Raise a claim with avail_verify after each step. A tool's own success does not make a "
                                "step done. Follow the fork. Finish with avail_done: if it is not ok, the task is not finished."}}
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": rid, "result": {"tools": self.tools()}}
        if method == "tools/call":
            name, args = params.get("name"), params.get("arguments") or {}
            if name not in AVAIL_TOOLS:                       # passed through: the upstream's reply, unchanged
                reply = self.upstream.request("tools/call", params)
                self._log(name, args, reply)
                return {**reply, "id": rid}
            value = self._avail(name, args)
            reply = {"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": json.dumps(value)}],
                                                             "structuredContent": value, "isError": not value["ok"]}}
            self._log(name, args, reply)
            return reply
        return {**self.upstream.request(method, params), "id": rid}

    def _avail(self, name: str, args: dict) -> dict:
        schema = self._schemas[name][1]
        missing = [k for k in schema["required"] if k not in args]
        extra = sorted(set(args) - set(schema["properties"]))
        if missing or extra:
            return {"ok": False, "error": f"needs {', '.join(missing)}" if missing else f"does not take {', '.join(extra)}"}
        try:
            core = self.core
            if name == "avail_verify":
                return core.verify(args["item"], args["step"])
            if name == "avail_probe":
                return core.probe(args["item"], args["step"], args["probe"])
            if name == "avail_ask_human":
                return core.ask_human(args["item"], args["step"], args["question"], args.get("options"))
            out = core.done(args["cycle_id"])
            return {**out, "ok": out.get("ok") is True and not out.get("unresolved")}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def serve_stdio(self) -> None:
        for line in sys.stdin:
            if line.strip():
                reply = self.handle(json.loads(line))
                if reply is not None:
                    sys.stdout.write(json.dumps(reply) + "\n")
                    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="avail proxy", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--upstream-url", help="an MCP server over streamable HTTP, such as ASPIRE's CAP server")
    src.add_argument("--upstream-cmd", help="an MCP server to run as a subprocess over stdio")
    parser.add_argument("--items", type=int, default=1, help="items per cycle, named 0, 1, ...")
    parser.add_argument("--steps", default="insert", help="comma-separated steps each item must have accepted")
    parser.add_argument("--arm", default="left")
    parser.add_argument("--accept-mm", type=float, default=0.5)
    parser.add_argument("--reject-mm", type=float, default=2.0)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)
    from ..ledger.writer import LedgerWriter
    from ..paths import runs_dir

    run_id = args.run_id or time.strftime("proxy_%Y%m%dT%H%M%S")
    upstream = HttpUpstream(args.upstream_url) if args.upstream_url else StdioUpstream(args.upstream_cmd)
    required = [(str(i), s.strip()) for i in range(args.items) for s in args.steps.split(",") if s.strip()]
    with LedgerWriter(run_id, folder=runs_dir() / run_id) as ledger:
        proxy = AvailProxy(upstream, lambda call: ProxyCore(call, required, arm=args.arm, accept_mm=args.accept_mm,
                                                            reject_mm=args.reject_mm), log=ledger.append)
        try:
            proxy.serve_stdio()
        finally:
            upstream.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
