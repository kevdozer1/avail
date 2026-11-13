"""The AVAIL proxy in front of a mock of ASPIRE's CAP server (S5c). ASPIRE's real server is not run here.

The mock's tool names and schemas are ASPIRE's own, built by ASPIRE's bridge code at the pinned commit."""

from __future__ import annotations

import json
import sys

import pytest

from avail.agent.mock_aspire import MockAspire, aspire_tools
from avail.agent.proxy import AVAIL_TOOLS, AvailProxy, InProcessUpstream, ProxyCore, StdioUpstream
from avail.paths import aspire_dir

pytestmark = pytest.mark.skipif(not (aspire_dir() / "src" / "cap_harness" / "agent" / "mcp.py").is_file(),
                                reason="needs the ASPIRE clone: uv run python scripts/setup_aspire.py")


def _proxy(latched=True, items=2, human=None, log=None):
    mock = MockAspire(latched=latched)
    required = [(str(i), "insert") for i in range(items)]
    proxy = AvailProxy(InProcessUpstream(mock), lambda call: ProxyCore(call, required, human=human), log=log)
    return proxy, mock


def _call(server, name, arguments, rid=7):
    return server.handle({"jsonrpc": "2.0", "id": rid, "method": "tools/call",
                          "params": {"name": name, "arguments": arguments}})


def test_the_mock_lists_aspires_own_tools():
    names = [t["name"] for t in aspire_tools()]
    assert {"cap_get_robot_state", "cap_move_to_pose", "cap_set_gripper", "observe", "get_job"} <= set(names)
    pose = next(t for t in aspire_tools() if t["name"] == "cap_move_to_pose")["inputSchema"]
    assert pose["required"] == ["target_pose"] and "oneOf" in pose["properties"]["target_pose"]


def test_every_upstream_tool_passes_through_byte_for_byte():
    proxy, _ = _proxy()
    direct = MockAspire()
    listed = proxy.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]
    upstream = direct.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]
    assert json.dumps(listed[:len(upstream)]) == json.dumps(upstream)              # each tool exactly as sent
    assert [t["name"] for t in listed[len(upstream):]] == list(AVAIL_TOOLS)
    for name, args in (("cap_get_robot_state", {}), ("cap_set_gripper", {"position": 0.0}), ("observe", {}),
                       ("cap_move_to_pose", {"target_pose": {"position": [0.4, 0.0, 0.25], "rpy_deg": [180, 0, 0],
                                                            "frame": "base"}})):
        via = _call(proxy, name, args)
        straight = _call(direct, name, args)
        assert json.dumps(via) == json.dumps(straight)                             # the same reply, byte for byte


def test_the_avail_tools_work_through_the_proxy_and_every_call_is_logged():
    log = []
    proxy, mock = _proxy(latched=True, log=lambda kind, payload: log.append((kind, payload)))
    probe = _call(proxy, "avail_probe", {"item": "0", "step": "insert", "probe": "probe.tug"})["result"]["structuredContent"]
    assert probe["ok"] and probe["evidence"]["moved_mm"] <= 0.25                   # held fast by the latch
    verdict = _call(proxy, "avail_verify", {"item": "0", "step": "insert"})["result"]["structuredContent"]
    assert verdict["decision"] == "accept" and "tug moved" in verdict["reason"]
    assert mock.pos[2] == pytest.approx(0.20)                                     # the probe put the tool back
    tools = [p["tool"] for kind, p in log]
    assert "avail_verify" in tools and "cap_move_to_pose" in tools                # the probe's own upstream calls too
    loose, _ = _proxy(latched=False)
    assert _call(loose, "avail_verify", {"item": "0", "step": "insert"})["result"]["structuredContent"]["decision"] == "reject"


def test_avail_done_refuses_until_every_required_step_is_accepted():
    answers = []
    proxy, mock = _proxy(latched=True, items=2, human=lambda *a: answers.append(a) or {"answer": "yes", "signed_off": True})
    done = _call(proxy, "avail_done", {"cycle_id": "c1"})["result"]
    assert done["isError"] and done["structuredContent"]["unresolved"] == ["0.insert", "1.insert"]
    _call(proxy, "avail_verify", {"item": "0", "step": "insert"})
    assert _call(proxy, "avail_done", {"cycle_id": "c1"})["result"]["structuredContent"]["unresolved"] == ["1.insert"]
    mock.latched = False                                                         # item 1 is loose: rejected
    assert _call(proxy, "avail_verify", {"item": "1", "step": "insert"})["result"]["structuredContent"]["decision"] == "reject"
    assert _call(proxy, "avail_done", {"cycle_id": "c1"})["result"]["isError"]
    _call(proxy, "avail_ask_human", {"item": "1", "step": "insert", "question": "takeover"})   # a person signs it off
    final = _call(proxy, "avail_done", {"cycle_id": "c1"})["result"]
    assert not final["isError"] and final["structuredContent"]["ok"] is True and len(answers) == 1


def test_no_argument_can_mark_a_step_done():
    proxy, _ = _proxy()
    out = _call(proxy, "avail_done", {"cycle_id": "c1", "ok": True})["result"]["structuredContent"]
    assert out["ok"] is False and "does not take" in out["error"]
    out = _call(proxy, "avail_verify", {"item": "0", "step": "insert", "decision": "accept"})["result"]["structuredContent"]
    assert out["ok"] is False


def test_the_proxy_reaches_an_upstream_over_stdio():
    up = StdioUpstream([sys.executable, "-m", "avail.agent.mock_aspire", "--latched"])
    try:
        proxy = AvailProxy(up, lambda call: ProxyCore(call, [("0", "insert")]))
        assert _call(proxy, "avail_verify", {"item": "0", "step": "insert"})["result"]["structuredContent"]["decision"] == "accept"
        assert not _call(proxy, "avail_done", {"cycle_id": "c"})["result"]["isError"]
    finally:
        up.close()
