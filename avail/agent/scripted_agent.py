"""A scripted agent that runs a cycle through the AVAIL tool server and nothing else.

It follows the skill note (``skilldoc/SKILL.md``) the way an agent would: run a skill with
``wam_act``, raise the step's claim with ``avail_verify``, follow the fork it gets back (a retry,
a person, or giving up), and finish with ``avail_done``. It never reads the cell, the evidence or
any truth, only tool results. Once a cycle, on purpose, it calls ``avail_done`` early, after
``early_done_after`` items, to show the gate refuses a cycle that is not done.

This stands in for a coding agent attached to the server. No model is called.
"""

from __future__ import annotations


def run_cycle(server, *, early_done_after: int | None = 1, max_tries: int = 6) -> dict:
    """1 cycle. Returns the cycle id, the early ``avail_done`` results, the final one, and the tool calls made."""
    calls = {"n": 0}

    def call(name: str, **arguments):
        calls["n"] += 1
        return server.call_tool(name, arguments)

    graph = call("get_capability_graph")
    core = graph.get("core", {})
    cycle_id, items, steps = core["cycle_id"], core["items"], core["steps"]
    early = []
    for n, item in enumerate(items):
        for step in steps:
            if not _run_step(call, item, step, max_tries):
                break                                    # this item is not done: the gate will say so
        if early_done_after is not None and n + 1 == early_done_after and n + 1 < len(items):
            early.append(call("avail_done", cycle_id=cycle_id))   # on purpose, before the cycle is over
    final = call("avail_done", cycle_id=cycle_id)
    return {"cycle_id": cycle_id, "early": early, "final": final, "calls": calls["n"]}


def _run_step(call, item: str, step: str, max_tries: int) -> bool:
    """Skill, claim, and whatever the fork says, until the step is done or the fork gives up."""
    for _ in range(max_tries):
        call("wam_act", skill=step, item=item)
        result = call("avail_verify", item=item, step=step)
        verdict = result.get("verdict") or {}
        if verdict.get("decision") == "accept":
            return True
        fork = result.get("fork") or {}
        chosen = fork.get("chosen")
        if chosen == "retry":
            continue
        if chosen == "ask_human":
            answer = call("avail_ask_human", item=item, step=step, question=fork.get("question") or "confirm")
            if answer.get("outcome") == "done":
                return True
            if answer.get("outcome") == "retry":
                continue
            return False
        return False                                     # abort, or nothing to follow
    return False
