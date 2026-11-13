---
name: avail-tool-usage
description: Check every robot step with AVAIL before calling it done. Raise a claim with avail_verify after each step, follow the fork it returns, ask a person with avail_ask_human when unsure, and close each cycle with avail_done.
---

# AVAIL tool usage

AVAIL is the check behind each step. A step is done only when calibrated evidence or a person says so. The tools come from AVAIL's MCP server (`avail/agent/mcp_server.py`). Their schemas are authoritative, so read them before the first call. A client that names tools by server shows them as `mcp__<server>__<tool>`.

## The 4 rules

1. A success flag, a pose or a picture does not make a step done. Raise a claim with `avail_verify`.
2. Follow the fork you get back. You may pick among its options. You may not skip it.
3. If you are unsure which predicate applies, ask with `avail_ask_human`.
4. Finish with `avail_done`. If it is not ok, the task is not finished.

## The tools

| Tool | Arguments | What it does |
|---|---|---|
| `get_capability_graph` | none | Lists the tools and the cell's skills, steps, probes, cameras and arms |
| `observe` | `cameras?` | Captures fresh frames: each camera's intrinsics, extrinsics, time and image size. Pixels are not sent |
| `cap_get_robot_state` | none | Reads each arm's joints, joint torque estimate, gripper and tool pose |
| `cap_move_to_pose` | `target_pose`, `arm?` | Moves an arm's tool to an absolute pose. The cell sets the speed and the force cap |
| `cap_set_gripper` | `position`, `arm?` | Sets a gripper from 0 closed to 1 open |
| `wam_act` | `skill`, `item`, `port?` | Runs 1 skill of the stand-in System 1 and returns its feedback and its own success flag |
| `stop_wam` | none | Stops all motion and holds |
| `avail_verify` | `item`, `step` | Raises the step's claim. Returns the verdict, the evidence summary and, when the verdict is not accept, the fork |
| `avail_probe` | `item`, `step`, `probe` | Runs 1 named probe, such as `probe.tug`, and returns its evidence |
| `avail_ask_human` | `item`, `step`, `question`, `options?` | Asks a person about the step's claim and waits for the answer |
| `avail_done` | `cycle_id` | The done gate. ok only when every step the task requires has an accept verdict or a person's sign-off. Otherwise it returns the unresolved list |

Arguments:

- `arm` takes the cell's arm names, `right` and `left` on the sim cell. Left out, it is the cell's first arm.
- `target_pose` holds `position` [x, y, z] in meters, a `frame`, and exactly 1 of `quaternion_wxyz` [w, x, y, z] or `rpy_deg` [roll, pitch, yaw] in degrees. RPY is fixed-axis XYZ, R = Rz(yaw) Ry(pitch) Rx(roll), as on ASPIRE.
- `port` is a number or a name. Left out, it is the item's assigned port.
- `question` is `confirm`, `choose`, `guide_marks`, `guide_text` or `takeover`.
- Names and ids are plain words of letters, digits and `_ . : @ -`, such as `c1`, `retry` or `cannot_tell`.
- No argument of any tool can carry a decision. A text that spells accept is refused in every argument.

Every result has `ok`. A result with `ok` false comes back with `isError` set. A reject or an abstain is a result with `ok` true: the call worked and the claim did not pass.

## What avail_verify returns

- `verdict.decision` is `accept`, `reject` or `abstain`.
  - `accept`: the evidence says the step's predicate holds. Go to the next step.
  - `reject`: the evidence says it does not hold. A fork comes back.
  - `abstain`: the evidence cannot tell. A fork comes back, usually with a question for a person.
- `fork.options` is drawn from `retry`, `repair:<probe>`, `recover:<skill>`, `ask_human` and `abort`. Accept is never one of them.
- When `fork.chosen` is set, the policy has picked. Do that. When it is empty, pick 1 of `fork.options`.
  - `retry`: run the step's skill again with `wam_act`, then call `avail_verify` again.
  - `ask_human`: call `avail_ask_human` with the fork's `question`.
  - `abort`: stop this cycle. It stays unresolved, and `avail_done` says so.
- `avail_probe` lets you look closer after a reject, for example with `probe.tug`. Its evidence informs you. It settles nothing. Only `avail_verify` or a person's answer does.

## Asking a person

- `confirm`: the person sees the frames and the claim and answers yes, no or cannot_tell. A yes signs the step off, and `avail_done` counts it like an accept.
- `choose`: pass the fork's options as `options`. The person picks 1. Do what they picked.
- `guide_marks`: the person marks a grasp point and a release point on a frame. Run the step again with the marks the answer carries.
- `guide_text`: the person writes the step in words.
- When you cannot tell which predicate a step needs, ask with `confirm` before you go on.

## 1 worked cycle

The cable loop in `tasks/cable_loop.yaml` moves 5 cables from the tray into 5 assigned ports, then the cell resets. Each cable takes 4 steps, and each step raises 1 claim. The task file holds the full definition of each claim.

| Step | Claim | Done means, in short |
|---|---|---|
| pick | held | The gripper holds the named cable, straight along the tool |
| carry | at_port | The cable hangs straight above its assigned port |
| insert | seated, or latched on a latch connector | The plug is in its assigned port, flush and straight |
| release | released_stable | The gripper has let go and the plug stayed seated |

Below, `>` is a call and `<` its result. Results are cut to the fields that matter, and `...` marks what is left out.

Read the cell first. The cycle id comes from the task prompt, or from the cell's entry here when the cell reports it.

```
> get_capability_graph {}
< {"ok": true, "skills": ["pick", "carry", "insert", "release"], "steps": ["pick", "carry", "insert", "release"],
   "probes": ["probe.push_seat", "probe.tug"], "arms": ["right", "left"], "cell": {"cycle_id": "cycle_3f9a1c0b7d2e", ...}, ...}
```

Pick cable c1 and check the claim.

```
> wam_act {"skill": "pick", "item": "c1"}
< {"ok": true, "flag": true, "feedback": "..."}
> avail_verify {"item": "c1", "step": "pick"}
< {"ok": true, "verdict": {"decision": "accept", ...}, "evidence": [...]}
```

Carry it to port 1 and check.

```
> wam_act {"skill": "carry", "item": "c1", "port": 1}
< {"ok": true, "flag": true, "feedback": "..."}
> avail_verify {"item": "c1", "step": "carry"}
< {"ok": true, "verdict": {"decision": "accept", ...}, "evidence": [...]}
```

Insert it. The skill's flag says it worked. The evidence says it did not, so the fork picks a retry.

```
> wam_act {"skill": "insert", "item": "c1"}
< {"ok": true, "flag": true, "feedback": "..."}
> avail_verify {"item": "c1", "step": "insert"}
< {"ok": true, "verdict": {"decision": "reject", ...}, "evidence": [...],
   "fork": {"options": ["retry", "ask_human", "abort"], "chosen": "retry", "chosen_by": "policy", ...}}
> wam_act {"skill": "insert", "item": "c1"}
< {"ok": true, "flag": true, "feedback": "..."}
> avail_verify {"item": "c1", "step": "insert"}
< {"ok": true, "verdict": {"decision": "accept", ...}, "evidence": [...]}
```

Release it. The evidence cannot tell, so the fork asks a person, and the person's yes signs the step off.

```
> wam_act {"skill": "release", "item": "c1"}
< {"ok": true, "flag": true, "feedback": "..."}
> avail_verify {"item": "c1", "step": "release"}
< {"ok": true, "verdict": {"decision": "abstain", ...}, "evidence": [...],
   "fork": {"options": ["ask_human", "abort"], "chosen": "ask_human", "question": "confirm", ...}}
> avail_ask_human {"item": "c1", "step": "release", "question": "confirm"}
< {"ok": true, "answer": "yes", "truth_label": true, ...}
```

Cables c2 to c5 go the same way, each into its own port. A call to the gate before c5 is released gets a refusal, with `isError` set.

```
> avail_done {"cycle_id": "cycle_3f9a1c0b7d2e"}
< {"ok": false, "unresolved": [{"item": "c5", "step": "release", ...}]}
```

The cycle stays open. Release c5, verify it, then close the cycle.

```
> avail_done {"cycle_id": "cycle_3f9a1c0b7d2e"}
< {"ok": true, "unresolved": []}
```

Now the cycle is done. The cell resets and the next cycle starts.

## Never

- Call a step done because `wam_act` returned `flag: true`, because a pose matched or because a picture looked right.
- Skip a fork, or pick an option the fork does not list.
- Stop after `avail_done` returned not ok and report the task as finished.
