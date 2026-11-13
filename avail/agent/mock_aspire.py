"""A mock of ASPIRE's CAP MCP server, for testing the AVAIL proxy without a robot or ASPIRE's runtime.

Its tool list is ASPIRE's own: ``aspire_tools()`` builds it with ASPIRE's code (``cap_harness.agent.mcp.CapMcpBridge``)
over a registry of ``cap_harness.api.CapApi``'s real method signatures, at the pinned commit in ``third_party/ASPIRE``.
So every tool name and argument schema is the one ASPIRE's server would send for the YAM station. What the tools do is
a stand-in: 1 arm holding 1 plug over a port, the plug latched or not, a position-controlled move that stalls when the
plug is held fast. Results follow ASPIRE's envelopes (``ok``, ``result`` or ``job``, ``observation``) in shape only.

    uv run python -m avail.agent.mock_aspire --latched         # serve it over stdio, 1 JSON-RPC message per line

This is a mock. ASPIRE's real server (``python -m cap_harness.agent.server``) was not run against AVAIL.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from functools import lru_cache
from typing import Any

ARM = "left"
FRAME = "base"
HOLD_FAST_MM = 0.2                      # how far a stalled pull moves the tool when the plug is latched


def _aspire_src() -> None:
    from ..paths import third_party_dir

    src = str(third_party_dir() / "ASPIRE" / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


@lru_cache(maxsize=1)
def aspire_tools() -> tuple[dict, ...]:
    """ASPIRE's MCP tools for the YAM station, as its own bridge lists them with segmentation and IK configured."""
    _aspire_src()
    from cap_harness.agent.mcp import CapMcpBridge
    from cap_harness.api import CapApi
    from cap_harness.registry import YAM_REAL_PUBLIC_TOOL_NAMES, ToolRegistry
    from cap_harness.yam_real.adapter import YamRealAdapter

    api = object.__new__(CapApi)                       # signatures only: nothing is constructed or run
    registry = ToolRegistry(public_extension_allowlist=YAM_REAL_PUBLIC_TOOL_NAMES)
    for name in ("get_robot_state", "localize_object", "move_to_joints", "move_to_pose", "move_synchronized",
                 "set_gripper"):
        registry.register(getattr(api, name), name=name, layer="shared_high_level", capability="execution",
                          public=True)
    adapter = object.__new__(YamRealAdapter)
    registry.register(types.MethodType(YamRealAdapter.get_controller_metadata, adapter),
                      name="yam_real.get_controller_metadata", layer="extension", capability="metadata", public=True)
    try:                                               # the bridge uses the MCP SDK's Tool record only to hold fields
        import mcp.types  # noqa: F401
    except ImportError:
        stub = types.ModuleType("mcp.types")

        class Tool:
            def __init__(self, name, description=None, inputSchema=None):
                self.name, self.description, self.inputSchema = name, description, inputSchema

            def model_dump(self, mode="json", exclude_none=True):
                out = {"name": self.name, "description": self.description, "inputSchema": self.inputSchema}
                return {k: v for k, v in out.items() if v is not None or not exclude_none}

        stub.Tool = Tool
        sys.modules.setdefault("mcp", types.ModuleType("mcp"))
        sys.modules["mcp.types"] = stub
    session = types.SimpleNamespace(
        api=types.SimpleNamespace(configured_providers={"segmentation": ["sam3"], "ik": ["pyroki"],
                                                        "pose_planning": []}, default_camera="wrist"),
        registry=registry, program_catalog=types.SimpleNamespace(all=lambda: []))
    return tuple(t.model_dump(mode="json", exclude_none=True) for t in CapMcpBridge(session).tool_specs())


class MockAspire:
    """1 arm, 1 plug in a port, latched or not. ``handle`` answers 1 JSON-RPC message as ASPIRE's server would."""

    def __init__(self, latched: bool = True):
        self.latched = latched
        self.pos = [0.40, 0.00, 0.20]                  # the tool, m, base frame
        self.quat = [0.0, 1.0, 0.0, 0.0]
        self.gripper = 0.0                             # closed on the plug
        self.jobs = 0

    def _state(self) -> dict[str, Any]:
        return {"end_effector_poses": {ARM: {"position": list(self.pos), "quaternion_wxyz": list(self.quat),
                                             "frame": FRAME}},
                "gripper_positions": {ARM: self.gripper}, "base_frame": FRAME, "embodiment": "yam_real"}

    def _job(self, ok: bool = True) -> dict[str, Any]:
        self.jobs += 1
        return {"ok": ok, "job": {"job_id": f"job_{self.jobs}", "state": "succeeded", "error": None},
                "observation": {"state": self._state(), "stale": False}, "image_cameras": []}

    def _call(self, name: str, args: dict) -> dict[str, Any]:
        if name == "cap_get_robot_state":
            return {"ok": True, "result": self._state()}
        if name == "cap_set_gripper":
            self.gripper = float(args["position"])
            return self._job()
        if name == "cap_move_to_pose":
            target = [float(v) for v in args["target_pose"]["position"]]
            dz = target[2] - self.pos[2]
            if dz > 0 and self.gripper < 0.5 and self.latched:     # held fast: the move stalls
                target[2] = self.pos[2] + min(dz, HOLD_FAST_MM / 1000.0)
            self.pos = target
            return self._job()
        if name == "observe":
            return {"ok": True, "observation": {"state": self._state(), "stale": False}, "image_cameras": []}
        if name == "get_capability_graph":
            return {"ok": True, "tools": [{"name": t["name"], "description": t.get("description")} for t in aspire_tools()]}
        return {"ok": True, "result": None}

    def handle(self, message: dict) -> dict | None:
        if "id" not in message:
            return None
        rid, method, params = message["id"], message.get("method"), message.get("params") or {}
        if method == "initialize":
            return {"jsonrpc": "2.0", "id": rid, "result": {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                                                            "serverInfo": {"name": "cap", "version": "0.1.0"}}}
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": rid, "result": {"tools": list(aspire_tools())}}
        if method == "tools/call":
            name = params.get("name")
            if name not in {t["name"] for t in aspire_tools()}:
                value = {"ok": False, "error": f"ValueError: unknown tool {name!r}"}
            else:
                value = self._call(name, dict(params.get("arguments") or {}))
            return {"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": json.dumps(value)}],
                                                            "structuredContent": value, "isError": not value["ok"]}}
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"Method not found: {method}"}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m avail.agent.mock_aspire")
    parser.add_argument("--latched", action="store_true", help="the plug is held fast in its port")
    args = parser.parse_args(argv)
    mock = MockAspire(latched=args.latched)
    for line in sys.stdin:
        if line.strip():
            reply = mock.handle(json.loads(line))
            if reply is not None:
                sys.stdout.write(json.dumps(reply) + "\n")
                sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
