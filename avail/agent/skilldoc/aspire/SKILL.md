---
name: avail-verification
description: Raise a claim after every task step and finish only through avail_done. A tool's ok, a pose or a camera image does not make a step done.
---

# AVAIL verification

The CAP MCP session is served through AVAIL. Every `cap_*` tool works exactly as before. 4 tools are added:
`avail_verify`, `avail_probe`, `avail_ask_human` and `avail_done`.

## Rules

1. After each step of the task, call `avail_verify` with the `item` and the `step`. A `cap_*` result with
   `ok: true`, a measured pose or a camera image does not make the step done.
2. Follow the fork in the verdict. On `reject`, retry the step or ask a person, as the fork's options say. On
   `abstain`, ask a person with `avail_ask_human`. Do not skip the fork.
3. `avail_probe` returns evidence only. It never marks a step done.
4. Finish with `avail_done`. If it is not `ok`, the task is not finished: report its `unresolved` list and stop.
5. Never state that a step succeeded unless its verdict was `accept` or a person signed it off.

## Steps and items

The operator starts AVAIL with the task's items and steps (for example `--items 5 --steps insert`). Use the same
names in every AVAIL call: items `0`, `1`, ... and the step names given.
