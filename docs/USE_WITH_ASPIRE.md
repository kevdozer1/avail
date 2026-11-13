# Use AVAIL with ASPIRE

AVAIL can sit between ASPIRE's System 2 agent and ASPIRE's CAP tool server. The agent sees every CAP tool exactly
as before, plus 4 tools: `avail_verify`, `avail_probe`, `avail_ask_human` and `avail_done`. AVAIL logs every call
to its ledger, runs its probes through the CAP motion tools, and closes a task only when every required step has an
accept verdict or a person's sign-off. The agent has no argument that can mark a step done.

**What was tested.** The proxy was tested against a mock of ASPIRE's CAP server (`avail/agent/mock_aspire.py`). The
mock's tool names and argument schemas are built by ASPIRE's own code (`cap_harness.agent.mcp.CapMcpBridge`) from
`cap_harness.api.CapApi`'s method signatures at the pinned commit `fa00f0e`. The tests check that every CAP tool
passes through byte for byte, that the 4 AVAIL tools work through the proxy, and that `avail_done` refuses until
every required step is accepted (`tests/test_proxy.py`). ASPIRE's real server was not run, and no real robot moved.

## How it fits

```
Codex (ASPIRE System 2) --stdio--> avail proxy --HTTP--> ASPIRE CAP server :8222/mcp/ --> robot
```

ASPIRE's agent launcher attaches the CAP server as `mcp_servers.cap` by URL
(`scripts/start_system2_agent.sh`). To put AVAIL in front, the same `cap` entry runs the proxy as a stdio command
instead. The tool names do not change, so ASPIRE's approval settings for `cap_*` tools still apply.

## Commands

1. Start ASPIRE's services as its runbook says (`docs/system2.md` in ASPIRE, terminals 1 to 4). The CAP server
   then serves `http://127.0.0.1:8222/mcp/`.

2. Check the proxy reaches it, from the AVAIL checkout:

   ```bash
   printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | \
     uv run avail proxy --upstream-url http://127.0.0.1:8222/mcp/ --items 5 --steps insert
   ```

   The reply lists ASPIRE's CAP tools, then the 4 AVAIL tools.

3. Copy the skill note into ASPIRE's agent skills, so its System 2 agent raises claims and finishes with
   `avail_done`:

   ```bash
   cp -r avail/agent/skilldoc/aspire "$ASPIRE_ROOT/.agents/skills/avail-verification"
   ```

4. Start ASPIRE's agent with the `cap` server replaced by the proxy. From ASPIRE's checkout, the same `codex`
   call as `scripts/start_system2_agent.sh`, with these 2 settings in place of `mcp_servers.cap.url`:

   ```bash
   -c 'mcp_servers.cap.command="uv"' \
   -c 'mcp_servers.cap.args=["run","--directory","/path/to/avail","avail","proxy","--upstream-url","http://127.0.0.1:8222/mcp/","--items","5","--steps","insert"]' \
   -c 'mcp_servers.cap.tools.avail_verify.approval_mode="approve"' \
   -c 'mcp_servers.cap.tools.avail_probe.approval_mode="approve"' \
   -c 'mcp_servers.cap.tools.avail_ask_human.approval_mode="approve"' \
   -c 'mcp_servers.cap.tools.avail_done.approval_mode="approve"'
   ```

5. Every call lands in AVAIL's ledger under `AVAIL_DATA/runs/proxy_<time>/ledger.jsonl`, with who made it: the agent
   or AVAIL's own probe.

## What the proxy decides with, and what to set on a real station

- **The probe.** `probe.tug` reads the tool pose with `cap_get_robot_state`, commands a 3 mm lift with
  `cap_move_to_pose` with the gripper as it is, reads how far the tool really went, and moves it back. A plug held
  fast stalls the lift. A loose one comes up.
- **The thresholds.** A lift held within 0.5 mm accepts, 2 mm or more rejects, and between them the claim goes to a
  person (`--accept-mm`, `--reject-mm`). These are the defaults the mock was tested with. On a real station they
  must be measured: `docs/AUDIT_YOUR_CHECK.md` shows how to measure a check's error rate.
- **Force.** ASPIRE's `cap_move_to_pose` is position controlled and has no force cap of its own. A tug on a real
  plug must be force limited by the station (compliance or a guarded move). Without it, do not run the tug.
- **A person.** `avail_ask_human` needs a person attached. The proxy started from the command line has none, so
  the claim stays open and `avail_done` keeps refusing. That is fail-closed by design.
