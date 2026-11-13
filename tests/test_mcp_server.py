"""The MCP tool server (avail/agent/mcp_server.py), driven with a fake cell and a fake core.

Nothing here builds a simulator. The fake core keeps a tiny done gate of its own, which is the contract the
real core keeps: a step is settled by an accept verdict or by a person's yes, and nothing else settles it.
"""

from __future__ import annotations

import ast
import io
import json
import math
import os
import re
import subprocess
import sys
import textwrap
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from avail import walls
from avail.agent import mcp_server
from avail.agent.mcp_server import INSTRUCTIONS, PROTOCOL_VERSION, ToolServer, UnknownTool
from avail.ledger import LedgerWriter, iter_events
from avail.paths import REPO_ROOT
from avail.schemas import gate_rules

TOOLS = ["get_capability_graph", "observe", "cap_get_robot_state", "cap_move_to_pose", "cap_set_gripper",
         "wam_act", "stop_wam", "avail_verify", "avail_probe", "avail_ask_human", "avail_done"]
STEPS = ("pick", "carry", "insert", "release")
SERVER_FILE = "avail/agent/mcp_server.py"
SKILL = REPO_ROOT / "avail" / "agent" / "skilldoc" / "SKILL.md"
DIFF = REPO_ROOT / "docs" / "aspire_tool_diff.md"
ASPIRE_MCP = REPO_ROOT / "third_party" / "ASPIRE" / "src" / "cap_harness" / "agent" / "mcp.py"


# ============================================================================ fakes
@dataclass
class Frame:
    """Shaped like a CameraFrame of the Station protocol."""

    name: str
    rgb: np.ndarray
    intrinsics: np.ndarray
    extrinsics: np.ndarray
    ts: float


@dataclass
class Obs:
    frames: dict
    ts: float


@dataclass
class Motion:
    """Shaped like a MotionResult."""

    ok: bool
    stopped_by: str
    travel_m: float = 0.0


class FakeCell:
    skills = STEPS
    cameras = ("top_camera_d405", "right_camera_d405")
    arms = ("right", "left")

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def observe(self, cameras=None):
        self.calls.append(("observe", {"cameras": cameras}))
        image = np.zeros((48, 64, 3), np.uint8)
        return Obs({name: Frame(name, image, np.eye(3), np.eye(4), 1.5) for name in (cameras or self.cameras)}, 1.5)

    def robot_state(self):
        self.calls.append(("robot_state", {}))
        return {"arms": {"right": {"joint_pos": np.arange(6, dtype=float), "gripper_pos": np.float64(0.25)}}, "ts": 2.0}

    def move_to_pose(self, target_pose, arm):
        self.calls.append(("move_to_pose", {"target_pose": target_pose, "arm": arm}))
        return Motion(ok=True, stopped_by="target", travel_m=0.01)

    def set_gripper(self, position, arm):
        self.calls.append(("set_gripper", {"position": position, "arm": arm}))
        return Motion(ok=True, stopped_by="target")

    def run_skill(self, skill, item, port):
        self.calls.append(("run_skill", {"skill": skill, "item": item, "port": port}))
        return {"flag": True, "feedback": f"{skill} done on {item}"}

    def stop(self):
        self.calls.append(("stop", {}))

    def describe(self):
        return {"station_id": "fake", "cycle_id": "cycle_0001"}


class FakeCore:
    """A core with a tiny gate. ``script`` maps (item, step) to the verdicts verify gives, in order."""

    steps = STEPS
    probes = ("probe.push_seat", "probe.tug")

    def __init__(self, items=("c1", "c2"), script=None):
        self.items, self.script = items, {key: list(words) for key, words in (script or {}).items()}
        self.settled: set[tuple[str, str]] = set()
        self.calls: list[tuple[str, dict]] = []

    def verify(self, item, step):
        self.calls.append(("verify", {"item": item, "step": step}))
        queue = self.script.get((item, step)) or ["accept"]
        word = queue.pop(0)
        verdict = {"verdict_id": f"verdict_{len(self.calls)}", "decision": word, "stage_reached": 1,
                   "evidence_ids": ["ev_1"] if word == "accept" else []}
        evidence = [{"source_id": "passive.jaw_hold", "score": np.float32(0.25 if word != "accept" else 0.97)}]
        if word == "accept":
            self.settled.add((item, step))
            return {"verdict": verdict, "evidence": evidence}
        fork = ({"fork_id": "fork_1", "options": ["retry", "ask_human", "abort"], "chosen": "retry"} if word == "reject"
                else {"fork_id": "fork_2", "options": ["ask_human", "abort"], "chosen": "ask_human",
                      "question": "confirm"})
        return {"verdict": verdict, "evidence": evidence, "fork": fork}

    def probe(self, item, step, probe):
        self.calls.append(("probe", {"item": item, "step": step, "probe": probe}))
        return {"evidence": {"source_id": probe, "features": {"disp_mm": np.float32(0.2)}, "score": 0.9}}

    def ask_human(self, item, step, question, options):
        self.calls.append(("ask_human", {"item": item, "step": step, "question": question, "options": options}))
        if question == "confirm":
            self.settled.add((item, step))        # a person's yes signs the step off
            return {"answer": "yes", "truth_label": True, "assist_id": "assist_1"}
        return {"answer": (options or ["abort"])[0], "truth_label": None, "assist_id": "assist_2"}

    def done(self, cycle_id):
        self.calls.append(("done", {"cycle_id": cycle_id}))
        unresolved = [f"{item}/{step}" for item in self.items for step in self.steps
                      if (item, step) not in self.settled]
        return {"ok": not unresolved, "unresolved": unresolved}


def make(**core_kw):
    """A server over a fresh fake cell and core, and the list its log fills."""
    events: list[tuple[str, dict]] = []
    cell, core = FakeCell(), FakeCore(**core_kw)
    return ToolServer(cell, core, log=lambda kind, payload: events.append((kind, payload))), cell, core, events


def rpc(rid, method, params=None):
    message = {"jsonrpc": "2.0", "id": rid, "method": method}
    return message if params is None else {**message, "params": params}


def lines(*messages) -> bytes:
    return b"".join((m if isinstance(m, bytes) else json.dumps(m).encode()) + b"\n" for m in messages)


def objects(schema):
    """Every object schema inside ``schema``, itself first."""
    if schema.get("type") == "object":
        yield schema
    inner = [*schema.get("properties", {}).values(), *schema.get("oneOf", [])]
    for part in inner + ([schema["items"]] if "items" in schema else []):
        yield from objects(part)


POSE = {"position": [0.3, 0.0, 0.2], "quaternion_wxyz": [1, 0, 0, 0], "frame": "world"}
VALID = {   # 1 call per tool that the server accepts
    "get_capability_graph": {}, "observe": {}, "cap_get_robot_state": {}, "cap_move_to_pose": {"target_pose": POSE},
    "cap_set_gripper": {"position": 0.5}, "wam_act": {"skill": "pick", "item": "c1"}, "stop_wam": {},
    "avail_verify": {"item": "c1", "step": "pick"},
    "avail_probe": {"item": "c1", "step": "insert", "probe": "probe.tug"},
    "avail_ask_human": {"item": "c1", "step": "insert", "question": "confirm"},
    "avail_done": {"cycle_id": "cycle_0001"},
}


# ============================================================================ the tool list
def test_tools_list_has_the_station_tools_in_a_fixed_order():
    server, *_ = make()
    assert [tool["name"] for tool in server.tools()] == TOOLS == list(VALID)
    for tool in server.tools():
        assert set(tool) == {"name", "description", "inputSchema"}
        assert tool["description"].endswith(".") and len(tool["description"]) > 20


def test_every_schema_refuses_unknown_keys_at_every_level():
    server, *_ = make()
    for tool in server.tools():
        found = list(objects(tool["inputSchema"]))
        assert found[0] is tool["inputSchema"], tool["name"]
        assert all(schema["additionalProperties"] is False for schema in found), tool["name"]
    move = server.tools()[3]["inputSchema"]
    assert any(schema is move["properties"]["target_pose"] for schema in objects(move))   # the nested pose too


def test_the_robot_tools_take_aspires_argument_shapes():
    server, *_ = make()
    schemas = {tool["name"]: tool["inputSchema"] for tool in server.tools()}
    for name in ("get_capability_graph", "cap_get_robot_state", "stop_wam"):
        assert schemas[name]["properties"] == {} and "required" not in schemas[name]
    assert set(schemas["observe"]["properties"]) == {"cameras"} and "required" not in schemas["observe"]
    assert schemas["observe"]["properties"]["cameras"]["type"] == "array"
    move = schemas["cap_move_to_pose"]
    assert move["required"] == ["target_pose"] and set(move["properties"]) == {"target_pose", "arm"}
    pose = move["properties"]["target_pose"]
    assert pose["required"] == ["position", "frame"]
    assert set(pose["properties"]) == {"position", "quaternion_wxyz", "rpy_deg", "frame"}
    assert pose["oneOf"] == [{"required": ["quaternion_wxyz"]}, {"required": ["rpy_deg"]}]
    sizes = {key: (spec["minItems"], spec["maxItems"]) for key, spec in pose["properties"].items() if key != "frame"}
    assert sizes == {"position": (3, 3), "quaternion_wxyz": (4, 4), "rpy_deg": (3, 3)}
    grip = schemas["cap_set_gripper"]
    assert grip["required"] == ["position"] and set(grip["properties"]) == {"position", "arm"}
    assert (grip["properties"]["position"]["minimum"], grip["properties"]["position"]["maximum"]) == (0, 1)


def test_the_wam_and_avail_tools_take_the_arguments_of_the_brief():
    server, *_ = make()
    schemas = {tool["name"]: tool["inputSchema"] for tool in server.tools()}
    expected = {
        "wam_act": (["skill", "item"], {"skill", "item", "port"}),
        "avail_verify": (["item", "step"], {"item", "step"}),
        "avail_probe": (["item", "step", "probe"], {"item", "step", "probe"}),
        "avail_ask_human": (["item", "step", "question"], {"item", "step", "question", "options"}),
        "avail_done": (["cycle_id"], {"cycle_id"}),
    }
    for name, (required, keys) in expected.items():
        assert (schemas[name]["required"], set(schemas[name]["properties"])) == (required, keys), name
    assert schemas["avail_ask_human"]["properties"]["question"]["enum"] == [
        "confirm", "choose", "guide_marks", "guide_text", "takeover"]
    assert schemas["wam_act"]["properties"]["port"]["type"] == ["integer", "string"]


def test_the_lists_of_the_cell_and_the_core_become_enums():
    server, *_ = make()
    props = {tool["name"]: tool["inputSchema"]["properties"] for tool in server.tools()}
    assert props["wam_act"]["skill"]["enum"] == list(STEPS)
    assert props["avail_verify"]["step"]["enum"] == props["avail_probe"]["step"]["enum"] == list(STEPS)
    assert props["avail_probe"]["probe"]["enum"] == ["probe.push_seat", "probe.tug"]
    assert props["observe"]["cameras"]["items"]["enum"] == list(FakeCell.cameras)
    assert props["cap_move_to_pose"]["arm"]["enum"] == props["cap_set_gripper"]["arm"]["enum"] == ["right", "left"]
    for name in ("avail_verify", "avail_probe", "avail_ask_human", "wam_act"):
        assert "enum" not in props[name]["item"] and props[name]["item"]["pattern"] == mcp_server._ID


def test_lists_may_be_objects_with_a_name_or_a_method_and_a_bare_cell_takes_plain_names():
    @dataclass
    class Step:
        name: str
        claim: str

    class Core:
        steps = [Step("pick", "held"), Step("insert", "seated")]

        def probes(self):
            return ["probe.tug"]

    server = ToolServer(object(), Core())
    props = {tool["name"]: tool["inputSchema"]["properties"] for tool in server.tools()}
    assert props["avail_verify"]["step"]["enum"] == ["pick", "insert"]
    assert props["avail_probe"]["probe"]["enum"] == ["probe.tug"]
    for name, key in (("wam_act", "skill"), ("cap_set_gripper", "arm")):
        assert props[name][key]["pattern"] == mcp_server._ID and "enum" not in props[name][key]
    assert "enum" not in props["observe"]["cameras"]["items"]


# ============================================================================ calls in process
def test_the_robot_tools_reach_the_cell_with_keywords_of_the_same_names():
    server, cell, core, events = make()
    observed = server.call_tool("observe", {"cameras": ["top_camera_d405"]})
    assert observed["ok"] is True and list(observed["frames"]) == ["top_camera_d405"]
    frame = observed["frames"]["top_camera_d405"]
    assert frame["rgb"] == {"shape": [48, 64, 3], "dtype": "uint8"}               # no pixels go out
    assert frame["intrinsics"] == np.eye(3).tolist() and frame["extrinsics"] == np.eye(4).tolist()
    state = server.call_tool("cap_get_robot_state")
    assert state["arms"]["right"] == {"joint_pos": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "gripper_pos": 0.25}
    assert server.call_tool("cap_move_to_pose", {"target_pose": POSE}) == {"ok": True, "stopped_by": "target",
                                                                          "travel_m": 0.01}
    assert server.call_tool("cap_set_gripper", {"position": 1, "arm": "left"})["ok"] is True
    assert server.call_tool("wam_act", {"skill": "carry", "item": "c1", "port": 3}) == {
        "ok": True, "flag": True, "feedback": "carry done on c1"}
    assert server.call_tool("stop_wam") == {"ok": True}
    server.call_tool("observe")
    assert cell.calls == [
        ("observe", {"cameras": ["top_camera_d405"]}),
        ("robot_state", {}),
        ("move_to_pose", {"target_pose": {"position": [0.3, 0.0, 0.2], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                                          "frame": "world"}, "arm": "right"}),
        ("set_gripper", {"position": 1.0, "arm": "left"}),
        ("run_skill", {"skill": "carry", "item": "c1", "port": 3}),
        ("stop", {}),
        ("observe", {"cameras": None}),
    ]
    assert core.calls == []
    assert [(kind, payload["tool"], payload["ok"]) for kind, payload in events] == [
        ("tool_call", name, True) for name in
        ("observe", "cap_get_robot_state", "cap_move_to_pose", "cap_set_gripper", "wam_act", "stop_wam", "observe")]


def test_an_optional_argument_left_out_arrives_as_none_and_the_arm_defaults_to_the_first():
    server, cell, *_ = make()
    server.call_tool("wam_act", {"skill": "pick", "item": "c2"})
    server.call_tool("cap_set_gripper", {"position": 0})
    assert cell.calls == [("run_skill", {"skill": "pick", "item": "c2", "port": None}),
                          ("set_gripper", {"position": 0.0, "arm": "right"})]

    class NoArms(FakeCell):
        arms = None

    bare = NoArms()
    ToolServer(bare, FakeCore()).call_tool("cap_set_gripper", {"position": 0.5})
    assert bare.calls == [("set_gripper", {"position": 0.5, "arm": None})]


def _matrix(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _rotation(axis, degrees):
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


@pytest.mark.parametrize("rpy", [(0, 0, 0), (0, 0, 90), (180, 0, 0), (0, 90, 0), (30, -20, 75), (-170, 45, 300)])
def test_rpy_deg_reaches_the_cell_as_the_unit_quaternion_of_aspires_convention(rpy):
    """ASPIRE reads rpy_deg as fixed-axis XYZ: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    server, cell, *_ = make()
    pose = {"position": [0.1, 0.2, 0.3], "rpy_deg": list(rpy), "frame": "world"}
    assert server.call_tool("cap_move_to_pose", {"target_pose": pose})["ok"] is True
    sent = cell.calls[-1][1]["target_pose"]
    assert set(sent) == {"position", "quaternion_wxyz", "frame"} and sent["position"] == [0.1, 0.2, 0.3]
    assert math.isclose(sum(q * q for q in sent["quaternion_wxyz"]), 1.0, abs_tol=1e-12)
    roll, pitch, yaw = rpy
    expected = _rotation("z", yaw) @ _rotation("y", pitch) @ _rotation("x", roll)
    assert np.allclose(_matrix(*sent["quaternion_wxyz"]), expected, atol=1e-12)


def test_a_nearly_unit_quaternion_is_rescaled_and_any_other_is_refused():
    server, cell, *_ = make()
    near = {**POSE, "quaternion_wxyz": [1.0005, 0, 0, 0]}
    assert server.call_tool("cap_move_to_pose", {"target_pose": near})["ok"] is True
    assert cell.calls[-1][1]["target_pose"]["quaternion_wxyz"] == [1.0, 0.0, 0.0, 0.0]
    for quat in ([1, 1, 0, 0], [0, 0, 0, 0]):
        result = server.call_tool("cap_move_to_pose", {"target_pose": {**POSE, "quaternion_wxyz": quat}})
        assert result["ok"] is False and "Send a unit quaternion" in result["error"]
    assert len(cell.calls) == 1                                  # the cell never saw the bad ones


@pytest.mark.parametrize("pose, words", [
    ({"position": [0, 0, 0], "frame": "world"}, "arguments.target_pose needs exactly 1 of quaternion_wxyz, rpy_deg"),
    ({**POSE, "rpy_deg": [0, 0, 0]}, "needs exactly 1 of quaternion_wxyz, rpy_deg"),
    ({**POSE, "position": [0, 0]}, "arguments.target_pose.position must hold from 3 to 3 items"),
    ({"position": [0, 0, 0], "rpy_deg": [0, 0, 0]}, "arguments.target_pose needs frame"),
    ({**POSE, "position": [0, 0, "x"]}, "arguments.target_pose.position[2] must be number"),
    ({**POSE, "position": [0, 0, float("nan")]}, "arguments.target_pose.position[2] must be number"),
    ({**POSE, "position": [0, 0, float("inf")]}, "must be number"),
    ({**POSE, "position": [0, 0, True]}, "must be number"),
    ({**POSE, "scale": 2}, "arguments.target_pose does not take scale"),
    ({**POSE, "frame": ""}, "arguments.target_pose.frame must be a name"),
    ("world", "arguments.target_pose must be object"),
])
def test_a_bad_pose_is_refused_before_the_cell_sees_it(pose, words):
    server, cell, *_ = make()
    result = server.call_tool("cap_move_to_pose", {"target_pose": pose})
    assert result["ok"] is False and words in result["error"], result
    assert cell.calls == []


@pytest.mark.parametrize("args, words", [
    ({"position": 1.5}, "arguments.position must be from 0 to 1"),
    ({"position": -0.1}, "arguments.position must be from 0 to 1"),
    ({"position": True}, "arguments.position must be number"),
    ({"position": "0.5"}, "arguments.position must be number"),
    ({}, "arguments needs position"),
    ({"position": 0.5, "arm": "middle"}, "arguments.arm must be 1 of right, left"),
    ({"position": 0.5, "max_torque": 1}, "arguments does not take max_torque. It takes position, arm"),
])
def test_a_bad_gripper_call_is_refused_before_the_cell_sees_it(args, words):
    server, cell, *_ = make()
    result = server.call_tool("cap_set_gripper", args)
    assert result == {"ok": False, "error": result["error"]} and words in result["error"], result
    assert cell.calls == []


@pytest.mark.parametrize("tool, args, words", [
    ("wam_act", {"skill": "dance", "item": "c1"}, "arguments.skill must be 1 of pick, carry, insert, release"),
    ("wam_act", {"skill": "pick", "item": "c 1"}, "arguments.item must be a name"),
    ("wam_act", {"skill": "pick", "item": "c1", "port": 2.5}, "arguments.port must be integer or string"),
    ("avail_verify", {"item": "c1"}, "arguments needs step"),
    ("avail_verify", {"item": "c1", "step": "dance"}, "arguments.step must be 1 of"),
    ("avail_probe", {"item": "c1", "step": "insert", "probe": "probe.shake"}, "arguments.probe must be 1 of"),
    ("avail_ask_human", {"item": "c1", "step": "insert", "question": "vote"}, "arguments.question must be 1 of"),
    ("avail_ask_human", {"item": "c1", "step": "insert", "question": "choose", "options": ["x"] * 17},
     "arguments.options must hold from 0 to 16 items"),
    ("avail_ask_human", {"item": "c1", "step": "insert", "question": "confirm", "options": ["cannot tell"]},
     "arguments.options[0] must be a name"),
    ("avail_done", {"cycle_id": "x" * 129}, "arguments.cycle_id must be a name"),
    ("observe", {"cameras": "top_camera_d405"}, "arguments.cameras must be array"),
    ("observe", {"cameras": ["wrist"]}, "arguments.cameras[0] must be 1 of top_camera_d405, right_camera_d405"),
    ("stop_wam", ["now"], "arguments must be object"),
])
def test_bad_arguments_are_refused_with_a_sentence_that_names_the_argument(tool, args, words):
    server, cell, core, events = make()
    result = server.call_tool(tool, args)
    assert result["ok"] is False and words in result["error"], result
    assert cell.calls == [] and core.calls == []
    assert events[-1][1]["error"] == result["error"]


def test_numpy_scalars_and_tuples_from_the_driver_read_as_json_would():
    server, cell, *_ = make()
    pose = {"position": np.array([0.1, 0.2, 0.3]), "quaternion_wxyz": (1, 0, 0, 0), "frame": "world"}
    assert server.call_tool("cap_move_to_pose", {"target_pose": pose, "arm": np.str_("left")})["ok"] is True
    assert server.call_tool("wam_act", {"skill": "insert", "item": "c1", "port": np.int64(5)})["ok"] is True
    assert server.call_tool("cap_set_gripper", {"position": np.float32(0.5)})["ok"] is True
    assert cell.calls[0][1] == {"target_pose": {"position": [0.1, 0.2, 0.3], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                                                "frame": "world"}, "arm": "left"}
    assert cell.calls[1][1]["port"] == 5 and type(cell.calls[1][1]["port"]) is int
    assert cell.calls[2][1] == {"position": 0.5, "arm": "right"}


def test_the_avail_tools_reach_the_core_with_keywords_of_the_same_names():
    server, cell, core, _ = make()
    verified = server.call_tool("avail_verify", {"item": "c1", "step": "pick"})
    assert verified["ok"] is True and verified["verdict"]["decision"] == "accept" and "fork" not in verified
    assert verified["evidence"] == [{"source_id": "passive.jaw_hold", "score": pytest.approx(0.97)}]
    probed = server.call_tool("avail_probe", {"item": "c1", "step": "insert", "probe": "probe.tug"})
    assert probed["evidence"]["features"] == {"disp_mm": pytest.approx(0.2)}
    assert type(probed["evidence"]["features"]["disp_mm"]) is float      # a numpy float came back as a float
    answered = server.call_tool("avail_ask_human", {"item": "c1", "step": "release", "question": "confirm"})
    assert answered == {"ok": True, "answer": "yes", "truth_label": True, "assist_id": "assist_1"}
    server.call_tool("avail_ask_human", {"item": "c1", "step": "insert", "question": "choose",
                                         "options": ["retry", "abort"]})
    assert core.calls == [
        ("verify", {"item": "c1", "step": "pick"}),
        ("probe", {"item": "c1", "step": "insert", "probe": "probe.tug"}),
        ("ask_human", {"item": "c1", "step": "release", "question": "confirm", "options": None}),
        ("ask_human", {"item": "c1", "step": "insert", "question": "choose", "options": ["retry", "abort"]}),
    ]
    assert cell.calls == []


def test_a_verdict_that_is_not_accept_comes_back_with_its_fork_and_is_no_error():
    server, *_ = make(script={("c1", "insert"): ["reject"], ("c1", "release"): ["abstain"]})
    reject = server.call_tool("avail_verify", {"item": "c1", "step": "insert"})
    assert reject["ok"] is True and reject["verdict"]["decision"] == "reject"
    assert reject["fork"]["chosen"] == "retry" and "accept" not in reject["fork"]["options"]
    call = {"name": "avail_verify", "arguments": {"item": "c1", "step": "release"}}
    abstain = server.handle(rpc(1, "tools/call", call))
    assert abstain["result"]["isError"] is False
    assert abstain["result"]["structuredContent"]["fork"]["question"] == "confirm"


def test_the_capability_graph_lists_the_tools_the_names_and_what_the_cell_describes():
    server, *_ = make()
    graph = server.call_tool("get_capability_graph")
    assert graph["ok"] is True
    assert graph["server"] == {"name": "avail", "version": "0.1.0", "protocol": PROTOCOL_VERSION}
    assert [tool["name"] for tool in graph["tools"]] == TOOLS
    assert all(set(tool) == {"name", "description"} for tool in graph["tools"])
    assert (graph["skills"], graph["steps"], graph["probes"]) == (list(STEPS), list(STEPS), ["probe.push_seat",
                                                                                              "probe.tug"])
    assert graph["cameras"] == list(FakeCell.cameras) and graph["arms"] == ["right", "left"]
    assert graph["cell"] == {"station_id": "fake", "cycle_id": "cycle_0001"} and "core" not in graph


def test_results_count_as_ok_unless_they_say_otherwise():
    class Cell(FakeCell):
        def run_skill(self, skill, item, port):
            return {"ok": False, "flag": False, "feedback": "the gripper jammed"}

        def robot_state(self):
            return [1, 2]

    server = ToolServer(Cell(), FakeCore())
    assert server.call_tool("stop_wam") == {"ok": True}
    assert server.call_tool("wam_act", {"skill": "pick", "item": "c1"}) == {
        "ok": False, "flag": False, "feedback": "the gripper jammed"}
    assert server.call_tool("cap_get_robot_state") == {"ok": True, "result": [1, 2]}


def test_a_failing_cell_gives_an_error_result_and_strict_mode_raises_after_logging():
    class Broken(FakeCell):
        def robot_state(self):
            raise RuntimeError("encoder lost")

    logged: list[dict] = []
    server = ToolServer(Broken(), FakeCore(), log=lambda kind, payload: logged.append(payload))
    assert server.call_tool("cap_get_robot_state") == {"ok": False, "error": "RuntimeError: encoder lost"}
    reply = server.handle(rpc(4, "tools/call", {"name": "cap_get_robot_state"}))["result"]
    assert reply["isError"] is True and "encoder lost" in reply["content"][0]["text"]
    strict = ToolServer(Broken(), FakeCore(), log=lambda kind, payload: logged.append(payload), strict=True)
    with pytest.raises(RuntimeError, match="encoder lost"):
        strict.call_tool("cap_get_robot_state")
    with pytest.raises(ValueError, match="arguments needs position"):
        strict.call_tool("cap_set_gripper", {})
    assert [(payload["tool"], payload["ok"]) for payload in logged] == [
        ("cap_get_robot_state", False)] * 3 + [("cap_set_gripper", False)]


# ============================================================================ the done gate
def test_avail_done_refuses_an_early_call_and_passes_once_every_step_is_settled():
    server, cell, core, events = make(items=("c1",), script={("c1", "release"): ["abstain"]})
    early = server.call_tool("avail_done", {"cycle_id": "cycle_0001"})
    assert early == {"ok": False, "unresolved": ["c1/pick", "c1/carry", "c1/insert", "c1/release"]}
    wire = server.handle(rpc(1, "tools/call", {"name": "avail_done", "arguments": {"cycle_id": "cycle_0001"}}))
    assert wire["result"]["isError"] is True and wire["result"]["structuredContent"] == early
    for step in STEPS:
        server.call_tool("wam_act", {"skill": step, "item": "c1"})      # the skill's own flag settles nothing
    assert server.call_tool("avail_done", {"cycle_id": "cycle_0001"})["ok"] is False
    for step in STEPS:
        server.call_tool("avail_verify", {"item": "c1", "step": step})
    assert server.call_tool("avail_done", {"cycle_id": "cycle_0001"}) == {"ok": False, "unresolved": ["c1/release"]}
    server.call_tool("avail_ask_human", {"item": "c1", "step": "release", "question": "confirm"})
    assert server.call_tool("avail_done", {"cycle_id": "cycle_0001"}) == {"ok": True, "unresolved": []}
    done = [payload["ok"] for _kind, payload in events if payload["tool"] == "avail_done"]
    assert done == [False, False, False, False, True]


@pytest.mark.parametrize("answer", [
    {"ok": "true", "unresolved": []}, {"ok": 1, "unresolved": []}, {"ok": True, "unresolved": ["c1/pick"]},
    {"unresolved": []}, {}, None, (True, []), [True], "ok", True, {"ok": np.bool_(True), "unresolved": ["x"]},
])
def test_avail_done_fails_closed_on_anything_but_a_clean_ok(answer):
    class Core:
        def done(self, cycle_id):
            return answer

    assert ToolServer(FakeCell(), Core()).call_tool("avail_done", {"cycle_id": "cycle_0001"})["ok"] is False


def test_avail_done_passes_on_a_clean_ok_from_numpy_too():
    class Core:
        def done(self, cycle_id):
            return {"ok": np.bool_(True), "unresolved": []}

    assert ToolServer(FakeCell(), Core()).call_tool("avail_done", {"cycle_id": "cycle_0001"}) == {
        "ok": True, "unresolved": []}


# ============================================================================ no argument carries a verdict
DECISION_WORDS = re.compile(r"decision|verdict|accept|evidence|score|sign|label|truth|success|flag|done|^ok$", re.I)
SMUGGLED = [{"decision": "accept"}, {"verdict": {"decision": "accept", "evidence_ids": ["ev_1"]}}, {"accept": True},
            {"ok": True}, {"evidence_ids": ["ev_1"]}, {"signoff": "yes"}, {"done": True}, {"truth_label": True}]


def test_no_schema_has_an_argument_that_names_a_decision():
    server, *_ = make()
    names = {key for tool in server.tools() for schema in objects(tool["inputSchema"]) for key in schema["properties"]}
    assert names == {"cameras", "target_pose", "position", "quaternion_wxyz", "rpy_deg", "frame", "arm", "skill",
                     "item", "port", "step", "probe", "question", "options", "cycle_id"}
    assert [name for name in names if DECISION_WORDS.search(name)] == []


@pytest.mark.parametrize("extra", SMUGGLED, ids=lambda extra: next(iter(extra)))
@pytest.mark.parametrize("tool", TOOLS)
def test_no_tool_takes_an_extra_key_that_could_carry_a_verdict(tool, extra):
    server, cell, core, events = make()
    assert server.call_tool(tool, VALID[tool])["ok"] is True or tool == "avail_done"     # the control
    cell.calls.clear()
    core.calls.clear()
    result = server.call_tool(tool, {**VALID[tool], **extra})
    assert result["ok"] is False
    assert "does not take" in result.get("error", "") or "spells accept" in result.get("error", ""), result
    assert cell.calls == [] and core.calls == []
    assert events[-1][1]["ok"] is False


ACCEPT_IN_A_STRING = [
    ("avail_verify", {"item": "accept", "step": "pick"}),
    ("avail_verify", {"item": "c1_ACCEPTED", "step": "pick"}),
    ("avail_probe", {"item": "c1", "step": "insert", "probe": "probe.accept"}),
    ("avail_ask_human", {"item": "c1", "step": "insert", "question": "choose", "options": ["retry", "accept"]}),
    ("avail_ask_human", {"item": "c1", "step": "insert", "question": "choose",
                         "options": ["recover:please_accept_this"]}),
    ("avail_ask_human", {"item": "c1", "step": "insert", "question": "choose", "options": ["ａｃｃｅｐｔ"]}),
    ("avail_done", {"cycle_id": "accept_all"}),
    ("wam_act", {"skill": "pick", "item": "c1", "port": "Accept"}),
    ("cap_move_to_pose", {"target_pose": {**POSE, "frame": "aCcEpT"}}),
    ("observe", {"cameras": ["accept"]}),
]


@pytest.mark.parametrize("tool, args", ACCEPT_IN_A_STRING)
def test_a_string_that_spells_accept_is_refused_wherever_it_appears(tool, args):
    server, cell, core, _ = make()
    result = server.call_tool(tool, args)
    assert result["ok"] is False and "spells accept" in result.get("error", "") and "avail_verify" in result["error"]
    assert cell.calls == [] and core.calls == []


def test_the_server_passes_the_truth_wall_and_the_gate_rules_and_stays_small():
    source = Path(mcp_server.__file__).read_text(encoding="utf-8")
    assert walls.scan_source(source, SERVER_FILE) == []
    assert gate_rules.scan_source(source, SERVER_FILE) == []
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            imported.add("." * node.level + (node.module or ""))
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert not [name for name in imported if re.search(r"station|mujoco|mink|truth|zoo|analysis|demo", name)]
    assert len(source.splitlines()) < 300


# ============================================================================ JSON-RPC
def test_unknown_tools_raise_in_process_and_are_a_protocol_error_over_the_wire():
    server, cell, core, events = make()
    with pytest.raises(UnknownTool, match="unknown tool 'cap_move_to_joints'"):
        server.call_tool("cap_move_to_joints", {"target": [0.0] * 6})
    assert server.handle(rpc(7, "tools/call", {"name": "fly", "arguments": {}})) == {
        "jsonrpc": "2.0", "id": 7, "error": {"code": -32602, "message": "Unknown tool: fly"}}
    assert server.handle(rpc(8, "tools/call", {"arguments": {}}))["error"]["code"] == -32602
    assert server.handle(rpc(9, "tools/call", {"name": ["observe"]}))["error"]["code"] == -32602
    assert [(payload["tool"], payload["ok"], payload["error"]) for _kind, payload in events] == [
        ("cap_move_to_joints", False, "unknown tool"), ("fly", False, "unknown tool"),
        ("None", False, "unknown tool"), ("['observe']", False, "unknown tool")]
    assert cell.calls == [] and core.calls == []


@pytest.mark.parametrize("asked, answered", [
    ("2025-06-18", "2025-06-18"), ("2025-03-26", "2025-03-26"), ("2024-11-05", "2024-11-05"),
    ("2099-01-01", "2025-06-18"), (None, "2025-06-18"),
])
def test_initialize_answers_the_version_asked_for_when_it_is_known(asked, answered):
    server, *_ = make()
    reply = server.handle(rpc(1, "initialize", {} if asked is None else {"protocolVersion": asked}))
    assert reply["result"]["protocolVersion"] == answered


def test_a_session_over_an_in_memory_pipe():
    server, cell, core, events = make()
    incoming = io.BytesIO(lines(
        rpc(1, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                              "clientInfo": {"name": "test", "version": "0"}}),
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        rpc(2, "tools/list"),
        rpc(3, "tools/call", {"name": "avail_verify", "arguments": {"item": "c1", "step": "pick"}}),
        rpc(4, "ping"),
        rpc("five", "tools/call", {"name": "avail_done", "arguments": {"cycle_id": "cycle_0001"}}),
        b"   ",
        b"{not json",
        b"\xff\xfe",
        rpc(6, "resources/list"),
        {"jsonrpc": "2.0", "id": 99, "result": {}},
        {"id": 7, "method": "ping"},
        [rpc(8, "ping")],
    ))
    outgoing = io.BytesIO()
    server.serve_stdio(incoming, outgoing)
    raw = outgoing.getvalue()
    assert raw.endswith(b"\n") and b"\r" not in raw and raw.isascii()
    replies = [json.loads(line) for line in raw.splitlines()]
    assert [reply["id"] for reply in replies] == [1, 2, 3, 4, "five", None, None, 6, 7, None]
    assert all(reply["jsonrpc"] == "2.0" for reply in replies)
    init = replies[0]["result"]
    assert init["protocolVersion"] == PROTOCOL_VERSION and init["capabilities"] == {"tools": {"listChanged": False}}
    assert init["serverInfo"] == {"name": "avail", "version": "0.1.0"} and init["instructions"] == INSTRUCTIONS
    assert [tool["name"] for tool in replies[1]["result"]["tools"]] == TOOLS
    call = replies[2]["result"]
    assert call["isError"] is False and call["structuredContent"]["verdict"]["decision"] == "accept"
    assert call["content"] == [{"type": "text", "text": json.dumps(call["structuredContent"])}]
    assert replies[3]["result"] == {}
    assert replies[4]["result"]["isError"] is True and replies[4]["result"]["structuredContent"]["ok"] is False
    assert replies[5]["error"]["code"] == replies[6]["error"]["code"] == -32700
    assert replies[7]["error"] == {"code": -32601, "message": "Method not found: resources/list"}
    assert replies[8]["error"]["code"] == replies[9]["error"]["code"] == -32600
    assert core.calls == [("verify", {"item": "c1", "step": "pick"}), ("done", {"cycle_id": "cycle_0001"})]
    assert [payload["tool"] for _kind, payload in events] == ["avail_verify", "avail_done"]


def test_each_line_is_answered_before_the_next_one_arrives():
    """A client waits for each reply before it writes again, so the loop must not wait for more input."""
    server, *_ = make()
    read_in, write_in = os.pipe()
    read_out, write_out = os.pipe()
    server_in, server_out = open(read_in, "rb"), open(write_out, "wb")
    client_in, client_out = open(read_out, "rb"), open(write_in, "wb")
    replies: list[dict] = []

    def serve():
        try:
            server.serve_stdio(server_in, server_out)
        finally:
            server_out.close()                       # the client sees the end even if the loop fails

    def talk():
        for rid, method in ((1, "ping"), (2, "tools/list"), (3, "ping")):
            client_out.write(json.dumps(rpc(rid, method)).encode() + b"\n")
            client_out.flush()
            replies.append(json.loads(client_in.readline()))

    loop, client = threading.Thread(target=serve, daemon=True), threading.Thread(target=talk, daemon=True)
    loop.start()
    client.start()
    client.join(timeout=20)
    assert not client.is_alive(), "the server did not answer a line before the next one was written"
    client_out.close()                               # the end of the input ends the loop
    loop.join(timeout=20)
    assert not loop.is_alive()
    client_in.close()
    server_in.close()
    assert [reply["id"] for reply in replies] == [1, 2, 3]


SERVE = textwrap.dedent('''
    import sys
    from avail.agent.mcp_server import ToolServer

    class Cell:
        def stop(self):
            print("a stray print from the cell")

    class Core:
        def done(self, cycle_id):
            return {"ok": False, "unresolved": ["c1/pick"]}

    ToolServer(Cell(), Core()).serve_stdio()
    print("the loop ended", file=sys.stderr)
''')


@pytest.mark.parametrize("launcher_moved_stdout", [False, True], ids=["plain", "launcher-moved-stdout"])
def test_the_real_stdio_carries_only_json_lines(tmp_path, launcher_moved_stdout):
    """A launcher that builds the cell may point sys.stdout at stderr first. The replies still go to stdout."""
    prelude = "import sys\nsys.stdout = sys.stderr\nprint('building the cell')\n" if launcher_moved_stdout else ""
    session = lines(rpc(1, "initialize", {"protocolVersion": "2025-06-18"}),
                    {"jsonrpc": "2.0", "method": "notifications/initialized"},
                    rpc(2, "tools/call", {"name": "stop_wam", "arguments": {}}),
                    rpc(3, "tools/call", {"name": "avail_done", "arguments": {"cycle_id": "cycle_0001"}}))
    env = {**os.environ, "AVAIL_DATA": str(tmp_path / "avail_data")}
    done = subprocess.run([sys.executable, "-c", prelude + SERVE], input=session, capture_output=True,
                          cwd=REPO_ROOT, env=env, timeout=120)
    assert done.returncode == 0, done.stderr.decode(errors="replace")
    replies = [json.loads(line) for line in done.stdout.splitlines()]          # every stdout line is a reply
    assert [reply["id"] for reply in replies] == [1, 2, 3] and b"\r" not in done.stdout
    assert replies[1]["result"]["structuredContent"] == {"ok": True}
    assert replies[2]["result"]["isError"] is True
    assert b"a stray print from the cell" in done.stderr and b"the loop ended" in done.stderr
    assert (b"building the cell" in done.stderr) is launcher_moved_stdout


# ============================================================================ the ledger
def test_every_call_is_logged_as_a_plain_tool_call(tmp_path):
    cell, core = FakeCell(), FakeCore()
    with LedgerWriter("mcp_server_test", folder=tmp_path) as ledger:
        server = ToolServer(cell, core, log=ledger.append)
        server.call_tool("wam_act", {"skill": "pick", "item": "c1", "port": np.int64(2)})
        server.call_tool("cap_set_gripper", {"position": 7})
        server.call_tool("avail_done", {"cycle_id": "cycle_0001"})
    events = list(iter_events(tmp_path))
    assert [event["kind"] for event in events] == ["tool_call"] * 3
    called, refused, gate = (event["payload"] for event in events)
    assert called["tool"] == "wam_act" and called["arguments"] == {"skill": "pick", "item": "c1", "port": 2}
    assert called["ok"] is True and "error" not in called and called["seconds"] >= 0
    assert refused == {"tool": "cap_set_gripper", "arguments": {"position": 7}, "ok": False,
                       "seconds": refused["seconds"], "error": "arguments.position must be from 0 to 1"}
    assert gate["tool"] == "avail_done" and gate["ok"] is False and "error" not in gate


# ============================================================================ the 2 documents
def test_the_skill_note_has_the_4_rules_and_names_exactly_the_tools_that_exist():
    text = SKILL.read_text(encoding="utf-8")
    front = text.split("---")[1]
    assert text.startswith("---\n") and re.search(r"^name: \S+$", front, re.M) and "\ndescription: " in front
    plain = text.replace("`", "")
    for rule in re.split(r"\s*\d\. ", INSTRUCTIONS)[1:]:
        assert rule.strip() in plain, rule
    named = set(re.findall(r"`((?:avail|cap|wam|stop)_\w+|observe|get_capability_graph)\b", text))
    assert named == set(TOOLS)


def test_the_diff_names_every_tool_of_both_servers_and_the_15_points():
    text = DIFF.read_text(encoding="utf-8")
    aspire = {"get_capability_graph", "observe", "get_job", "read_operator_messages", "cap_get_robot_state",
              "cap_localize_object", "cap_move_to_pose", "cap_move_to_joints", "cap_move_synchronized",
              "cap_set_gripper", "cap_yam_real__get_controller_metadata", "cap_program_observe",
              "cap_program_go_home"}
    for name in aspire | set(TOOLS):
        assert f"`{name}`" in text, name
    for point in range(1, 16):
        assert re.search(rf"\bD{point}\b", text), f"D{point}"


@pytest.mark.skipif(not ASPIRE_MCP.is_file(), reason="third_party/ASPIRE is not cloned here")
def test_the_diff_covers_every_tool_that_aspires_mcp_py_defines():
    """Read as text with ast and never run. A re-pulled ASPIRE with a new tool makes this fail."""
    tree = ast.parse(ASPIRE_MCP.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(target, "id", "") == "_DIRECT" for target in node.targets):
            names |= {"cap_" + element.value.replace(".", "__") for element in node.value.elts}
        if isinstance(node, ast.Assign) and any(getattr(target, "id", "") == "specs" for target in node.targets):
            names |= {element.elts[0].value for element in node.value.elts}
    assert {"get_job", "read_operator_messages", "cap_move_synchronized"} <= names and len(names) >= 10
    text = DIFF.read_text(encoding="utf-8")
    assert [name for name in sorted(names) if f"`{name}`" not in text] == []


@pytest.mark.parametrize("path", [SKILL, DIFF], ids=["SKILL.md", "aspire_tool_diff.md"])
def test_the_documents_keep_kevins_writing_rules(path):
    text = path.read_text(encoding="utf-8")
    assert "—" not in text and "–" not in text                   # no em or en dashes
    assert ";" not in text                                                  # no semicolons
    assert not re.search(r", not \w", text)                                 # no "X, not Y"
