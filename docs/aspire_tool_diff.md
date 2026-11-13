# AVAIL's tools against ASPIRE's CAP MCP server

Level 1: AVAIL's tool server keeps ASPIRE's names and argument shapes for the 5 tools both servers have. It differs in transport, in which tools exist, in what results carry and in how it refuses bad input.

Both sides were read on 2026-11-12. ASPIRE is the clone in `third_party/ASPIRE` at commit `fa00f0e` (v1.0), file `src/cap_harness/agent/mcp.py`, with `server.py`, `session.py`, `api.py` and `geometry.py` next to it. AVAIL is `avail/agent/mcp_server.py`. Line numbers name `mcp.py` unless a cell names another file. This compares source code. No ASPIRE server ran, so check the live schema before claiming that an agent written for ASPIRE runs on the sim cell.

## The tools side by side

| Tool | ASPIRE | AVAIL |
|---|---|---|
| `get_capability_graph` | yes | yes |
| `observe` | yes | yes |
| `get_job` | yes | no |
| `read_operator_messages` | yes | no |
| `cap_get_robot_state` | yes | yes |
| `cap_localize_object` | only with a segmentation provider | no |
| `cap_move_to_pose` | only with IK or pose planning | yes |
| `cap_move_to_joints` | yes | no |
| `cap_move_synchronized` | yes | no |
| `cap_set_gripper` | yes | yes |
| `cap_yam_real__get_controller_metadata` | yes | no |
| `cap_program_observe` | yes | no |
| `cap_program_go_home` | yes | no |
| `wam_act` | no | yes |
| `stop_wam` | no | yes |
| `avail_verify` | no | yes |
| `avail_probe` | no | yes |
| `avail_ask_human` | no | yes |
| `avail_done` | no | yes |

## Every difference

| # | Topic | ASPIRE `mcp.py` | AVAIL `mcp_server.py` |
|---|---|---|---|
| 1 | Transport | MCP over streamable HTTP through the `mcp` SDK, stateless, with JSON responses (412-432). `server.py` mounts it at `/mcp` on `127.0.0.1:8222` (`server.py:23`, `server.py:96-97`) | MCP over stdio in plain Python with no SDK, 1 JSON-RPC 2.0 message per line (`serve_stdio`). The same tools run in process through `call_tool` |
| 2 | Protocol version | Whatever the installed SDK negotiates | `2025-06-18`. A client that asks for `2025-03-26` or `2024-11-05` gets that version back. Any other request gets `2025-06-18` |
| 3 | Server identity | Name `cap`, version `0.1.0` (417), while the repo is tagged v1.0 | Name `avail`, version `0.1.0`, the package version |
| 4 | Instructions at initialize | None set in `mcp.py` | The 4 rules of the skill note |
| 5 | Tools only ASPIRE has | `get_job`, `read_operator_messages`, `cap_localize_object`, `cap_move_to_joints`, `cap_move_synchronized`, `cap_yam_real__get_controller_metadata`, `cap_program_observe` and `cap_program_go_home` (196-204, 251-291) | None of these 8 |
| 6 | Tools only AVAIL has | None | `wam_act` and `stop_wam` with the names in GR00T 2's figure, and `avail_verify`, `avail_probe`, `avail_ask_human` and `avail_done` |
| 7 | Conditional tools | Drops `cap_localize_object` without a segmentation provider and `cap_move_to_pose` without IK or pose planning (236-240) | Always lists all 11 tools |
| 8 | Tool order | 4 fixed tools, then the `cap_` tools in registry order, then the packaged programs (251-291) | A fixed order: `get_capability_graph`, `observe`, the 3 `cap_` tools, the 2 WAM tools, the 4 `avail_` tools |
| 9 | Tool fields | `name`, `description` and `inputSchema`. A `cap_` tool's description is its documented Python signature (287, 292-295) | The same 3 fields. Each description is a sentence for the agent |
| 10 | Where schemas come from | Derived from each Python signature, with scalar defaults written into the schema (79-107) | Written by hand. Both refuse unknown keys at every level |
| 11 | Argument check | The `jsonschema` library (298, 306) | Its own check of the subset its schemas use: types, required keys, unknown keys, enums, ranges, item counts, patterns and `oneOf`. NaN, infinity and booleans never pass as numbers. In process, numpy scalars and tuples read as JSON would read them |
| 12 | Names in arguments | Any string for an arm or a camera. The session checks it (332-333 for cameras) | The skills, steps, probes, cameras and arms the cell or the core lists become enums. Any other name or id must match `^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,127}$` |
| 13 | Text that spells accept | No rule | A call with a string argument that spells accept in any case or width is refused before it reaches the cell or the core. No schema has an argument that names a decision |
| 14 | Arm names | Default `primary`, which the station profile maps to `right`. The YAM arms are `left` and `right` (`api.py:1067`, `api.py:1311`) | The cell's arm names, `right` and `left` on the sim cell. Left out, the arm is the one the cell lists at the top. `primary` and `secondary` are not mapped |
| 15 | `cap_move_to_pose` arguments | `target_pose` (required), `tolerance` 0.01, `max_steps` 120, `arm` and `strategy`, a MotionStrategy or null (271-286, `api.py:1305-1313`) | `target_pose` and `arm`. The cell sets the speed and the force cap from config. There is no tolerance, step count or strategy |
| 16 | Pose orientation | `position`, `frame` and exactly 1 of `quaternion_wxyz` or `rpy_deg`, fixed-axis XYZ (110-142). A quaternion more than 1e-5 off unit length is refused (`geometry.py:59-60`) | The same shape and convention. A quaternion within 1e-3 of unit length is rescaled to unit length, and any other is refused. `rpy_deg` becomes a quaternion before the cell sees it, so the cell always gets `{position, quaternion_wxyz, frame}` |
| 17 | `cap_set_gripper` | `position` (required) and `arm`. The range 0 to 1 is checked in code (`api.py:1067-1081`) | The same 2 arguments. The range is in the schema as `minimum` 0 and `maximum` 1. Neither server takes a torque limit |
| 18 | `observe` with no cameras | The default camera only (375-378) | The cell gets None. The sim cell returns every camera |
| 19 | `observe` result | `{ok, observation {observation_seq, timestamp_s, stale, monitor_error, camera_metadata, state}, image_cameras}`, then a text block and a JPEG block per camera (309-314, 326-349). JPEGs are quality 85 with the longest side at most 960 px (180-193) | The cell's Observation next to `ok`: per camera its name, intrinsics, extrinsics and time, with the image as its shape and dtype. No image block is sent |
| 20 | Images after motion | Every motion reply carries every camera's JPEG (408-409) | None |
| 21 | `cap_get_robot_state` result | `{ok, result}` with `joint_positions`, `joint_velocities`, `end_effector_poses`, `gripper_positions`, `base_frame`, `joint_names`, `timestamp_s` and `embodiment`, and no effort (405-407) | The cell's RobotState next to `ok`: per arm `joint_pos`, `joint_vel`, `joint_torque_est`, `gripper_pos`, `gripper_torque`, `tool_pose` and `tool_force_est`, plus `ts`. Effort is in |
| 22 | Motion result | A job runs and the call blocks until it ends. The reply is `{ok, job {..., result {ok, status, postcondition_satisfied, ...}}, observation, image_cameras}` (351-361, 408-409) | The cell's MotionResult next to `ok`: `stopped_by`, `travel_m`, `duration_s`, `force_at_stop_n`, `peak_force_n`, `trace` and `note` |
| 23 | Jobs | Physical tools run as jobs with ids. `observe` may run during a job, and a 2nd physical job is refused (`session.py:410`) | 1 call at a time, run to its end in the calling thread. There are no job ids |
| 24 | Stopping | No stop or cancel tool | `stop_wam` calls the cell's `stop()` |
| 25 | Operator channel | `read_operator_messages` drains free text that arrives through `POST /steer` (383-384, `server.py:29`) | `avail_ask_human` asks a person a typed question (`confirm`, `choose`, `guide_marks`, `guide_text` or `takeover`) and waits for the answer, which the core logs |
| 26 | Capability graph | The tools, the programs with their provenance, and the configured providers (366-374) | The server's name, version and protocol, the tools, the lists of skills, steps, probes, cameras and arms, and what `describe()` on the cell and the core returns when they have it |
| 27 | Result envelope | The result dict as `structuredContent` and as 1 JSON text block, with `isError` when `ok` is false (309-317) | The same. A result without `ok` gets `ok` true, and a result that is no dict goes under `result`. `avail_done` is ok only when the core returns the boolean true with nothing unresolved |
| 28 | Errors | Every exception becomes `{ok: false, error: "<Type>: <message>"}` with `isError`, an unknown tool too (318-324) | The same for refused arguments and for exceptions from the cell or the core. An unknown tool is the JSON-RPC error -32602, as the MCP spec shows it. In process, `call_tool` raises `UnknownTool`. With `strict`, refused arguments and exceptions raise after they are logged |
| 29 | Arrays in results | An array with 1 dimension and at most 32 values becomes a list rounded to 6 decimals. Any other array becomes its shape and dtype (208-227) | Any array of at most 64 numbers becomes a nested list, so a 3 x 3 or 4 x 4 matrix stays a matrix. A larger one becomes its shape and dtype. NaN and infinity become null. Nothing is rounded |
| 30 | Logging | `mcp.py` logs no call itself | Every call, refused calls too, goes to `log("tool_call", payload)` with the tool, its arguments, `ok`, the seconds it took and any error |
| 31 | Threads | An async server that runs each call in a worker thread (423-425) | 1 line at a time in the calling thread. While a call runs, such as a wait for a person, no other message is read |

## The 15 points from the read of AVAIL's docs

On 2026-11-12 a read of AVAIL's docs against ASPIRE's code found 15 points, D1 to D15. Each one maps to rows above.

| Point | The docs against ASPIRE's code | Rows | In the built server |
|---|---|---|---|
| D1 | The docs list 7 robot tools. ASPIRE has 6 more | 5 | 5 robot tools are served, and 8 of ASPIRE's tools are left out |
| D2 | `cap_localize_object` and `cap_move_to_pose` are listed with no condition. Both are conditional in ASPIRE | 5, 7 | `cap_move_to_pose` is always served. `cap_localize_object` is not served. A sim version needs a perception path that keeps the truth wall, and none is built |
| D3 | The docs have a stop and `stop_wam`. ASPIRE has no stop tool | 24 | `stop_wam` calls the cell's `stop()` |
| D4 | The docs guard moves with speed, force and travel caps. ASPIRE's moves take a tolerance, a step count and a strategy | 15 | The cell applies the caps from config. Probes run inside the core through `avail_probe`, so they keep their caps |
| D5 | The docs give the robot state a torque estimate. ASPIRE's state has no effort | 21 | `cap_get_robot_state` returns the joint torque estimate and the gripper torque |
| D6 | The docs put the arm ahead of the position in `set_gripper` and add a torque limit. ASPIRE takes `position` and an `arm` keyword | 17 | ASPIRE's shape: `position` and `arm`, with no torque limit |
| D7 | The docs use the right arm. ASPIRE defaults to `primary` | 14 | The cell's arm names, with the arm it lists at the top as the default. `primary` and `secondary` are refused when the cell lists its arms |
| D8 | The docs name the cameras `top_camera_d405`, `left_camera_d405` and `right_camera_d405`. ASPIRE has `top`, `bottom`, `wrist_left` and `wrist_right` | 12 | The cell's names become an enum. An agent that asks for `top` on the sim cell gets an error that lists the real names |
| D9 | The docs give an observation intrinsics and extrinsics. ASPIRE sends freshness fields and JPEGs | 19 | `observe` returns intrinsics, extrinsics and time, and no pixels |
| D10 | The docs never say what `observe` does with no cameras | 18 | The cell decides. The sim cell returns every camera |
| D11 | The docs give no result envelope | 27, 28 | ASPIRE's envelope, and `avail_done` fails closed |
| D12 | The docs never define MotionResult | 22 | The MotionResult of the Station protocol, next to `ok` |
| D13 | The robot tools take ASPIRE's names and the WAM tools take GR00T 2's. GR00T 2's own figure has `get_image`, `get_robot_state` and `move_to_pose` with no `cap_` | 6 | Kept as the docs say. An agent written for GR00T 2's names needs a mapping |
| D14 | The docs plan an assist page. ASPIRE already has an operator channel | 25 | `avail_ask_human` is AVAIL's channel. `read_operator_messages` is not served |
| D15 | The docs pin ASPIRE at `fa00f0e` (v1.0). Its server reports 0.1.0 | 3 | Recorded. Recheck at the next pull |

## What this does not show

It shows what the 2 files say. It does not show that a live ASPIRE session or a live AVAIL session behaves that way, and no agent written for ASPIRE has driven the sim cell yet. The proxy (`avail/agent/proxy.py`) was tested against a mock of ASPIRE's server, so the 2 real servers have not met.

## Keeping it current

`tests/test_mcp_server.py` reads `mcp.py` with `ast`, without running it, and fails when `mcp.py` defines a tool this file does not name. It also fails when AVAIL serves a tool this file does not name. After a re-pull of ASPIRE, run it and update the tables.
