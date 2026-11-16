"""The AVAIL side of the tool server: ``avail_verify``, ``avail_probe``, ``avail_ask_human`` and ``avail_done``.

Core code. ``ToolCore`` is the ``core`` object ``avail.agent.mcp_server.ToolServer`` is handed. It does
what the cycle driver does, 1 tool call at a time: a claim per verify, the task's witness, the fork
policy, the person, the done gate and the ledger, with the same records. The robot side (``wam_act``)
goes through the testbed's cell adapter, which tells this object what each skill call returned, so
a verify can hand the skill's own record to the checks that read it.

An agent drives it through the tools and nothing else (``avail.agent.scripted_agent``). The gate
answers ``avail_done`` from the accept verdicts and the person's sign-offs only, so an early call
comes back not ok with the unresolved list.
"""

from __future__ import annotations

from .driver import CycleDriver, ItemLog


class ToolCore(CycleDriver):
    """1 cycle's AVAIL tools. ``steps``, ``probes`` and ``describe()`` feed the server's schemas and graph."""

    def __init__(self, task, cell, arm: str = "avail_simop", **kwargs):
        super().__init__(task, cell, arm, **kwargs)
        self.steps = [s.name for s in task.steps]
        self.probes = list(getattr(cell, "probes", []))          # the cell names the probes it can run
        self.results: dict[tuple[int, str], object] = {}      # the last skill result per (item, step), from the cell adapter
        self.calls: dict[tuple[int, str], int] = {}           # skill calls per (item, step)
        self.marks: dict[tuple[int, str], dict] = {}          # a person's marks for the next try of a step
        self.items: dict[int, ItemLog] = {}
        self._open: dict[tuple[int, str], dict] = {}          # the last claim, verdict and fork of a step
        self._emit("cycle_start", {"cycle_id": self.cycle_id, "arm": self.arm, "task": self.task.task,
                                   "variant": self.cell.variant, "items": list(range(self.cell.n_items)), "via": "tools"})

    def describe(self) -> dict:
        return {"cycle_id": self.cycle_id, "task": self.task.task, "items": [str(i) for i in range(self.cell.n_items)],
                "steps": self.steps}

    # ---------------------------------------------------------------- what the cell adapter tells it
    def attempt_of(self, item: int, step: str) -> int:
        return self.calls.get((item, step), 0)

    def skill_ran(self, item: int, step: str, result) -> None:
        self.calls[(item, step)] = self.calls.get((item, step), 0) + 1
        self.results[(item, step)] = result
        log = self.items.setdefault(item, ItemLog(item))
        log.attempts[step] = self.calls[(item, step)]
        self.log.motion_s += float(getattr(result, "seconds", 0.0))

    def take_marks(self, item: int, step: str) -> dict | None:
        return self.marks.pop((item, step), None)

    # ---------------------------------------------------------------- the 4 tools
    def _step(self, step: str):
        for s in self.task.steps:
            if s.name == step:
                return s
        raise KeyError(f"the task has no step {step!r}. Its steps: {', '.join(self.steps)}")

    def verify(self, item, step: str) -> dict:
        item, spec = int(item), self._step(step)
        res = self.results.get((item, step))
        if res is None:
            raise ValueError(f"no skill has run for item {item}, step {step}. Call wam_act first")
        claim = self._claim(item, spec, self.attempt_of(item, step) - 1)
        self._emit("claim", claim)
        verdict = self._verify(item, spec, claim, res)
        out = {"verdict": {"decision": verdict.decision, "stage_reached": verdict.stage_reached,
                           "verdict_id": verdict.verdict_id, "witness_version": verdict.witness_version},
               "evidence": [{"source": k.split("/")[0], "feature": k.split("/", 1)[1], "value": v}
                            for k, v in sorted(getattr(self, "_last_features", {}).items())][:40],
               "fork": None}
        if verdict.decision == "accept":
            self.gate.record_accept(item, claim.predicate.name, verdict)
        else:
            plan = self._fork(verdict, self.attempt_of(item, step) - 1, spec)
            out["fork"] = {"fork_id": self._last_fork.fork_id, "options": list(plan.options), "chosen": plan.chosen,
                           "question": plan.question, "reason": plan.reason}
        self._open[(item, step)] = {"claim": claim, "verdict": verdict, "fork": getattr(self, "_last_fork", None)}
        return out

    def probe(self, item, step: str, probe: str) -> dict:
        from ..evidence import base

        item, spec = int(item), self._step(step)
        res = self.results.get((item, step))
        claim = self._claim(item, spec, max(0, self.attempt_of(item, step) - 1))
        params = self.cell.claim_params(item, step, res)
        t0 = self.cell.now()
        ev = base.get(probe).collect(self.cell.view, claim, params)
        self.log.verify_s += max(0.0, self.cell.now() - t0)
        self._emit("evidence", ev)
        return {"evidence_id": ev.evidence_id, "source": ev.source_id, "features": dict(ev.features),
                "side_effect": ev.side_effect}

    def ask_human(self, item, step: str, question: str, options=None) -> dict:
        from ..fork.policy import ForkPlan

        item, spec = int(item), self._step(step)
        opened = self._open.get((item, step))
        if opened is None:
            raise ValueError(f"no claim is open for item {item}, step {step}. Call avail_verify first")
        log = self.items.setdefault(item, ItemLog(item))
        plan = ForkPlan(options=["ask_human"], chosen="ask_human", question=question, reason="the agent asked")
        self._last_fork = opened["fork"] or self._last_fork
        asked = log.interventions
        if asked >= 1 and question != "takeover" and self._asked_before(item, step):
            question = "takeover"                       # back at a person after their own fix failed: they take over
            plan = ForkPlan(options=["ask_human"], chosen="ask_human", question=question, reason="second call")
        self._asked.add((item, step))
        answer = self._ask(item, spec, opened["claim"], opened["verdict"], plan, question, log)
        outcome = self._after_answer(item, spec, opened["claim"], opened["verdict"], plan, answer, log,
                                     self.attempt_of(item, step) - 1)
        if outcome == "retry" and answer.correction_kind == "marks":
            self.marks[(item, step)] = dict(answer.correction)
        return {"answer": answer.answer, "outcome": outcome, "question": question,
                "seconds_to_answer": answer.seconds_to_answer, "seconds_handling": answer.seconds_handling}

    def _asked_before(self, item: int, step: str) -> bool:
        return (item, step) in self._asked

    @property
    def _asked(self) -> set:
        if not hasattr(self, "_asked_set"):
            self._asked_set = set()
        return self._asked_set

    def done(self, cycle_id: str) -> dict:
        if cycle_id != self.cycle_id:
            raise ValueError(f"this server holds cycle {self.cycle_id}, not {cycle_id}")
        ok, unresolved = self.gate.cycle_done(range(self.cell.n_items))
        return {"ok": bool(ok), "unresolved": [{"item": str(u["item"]), "step": u["step"], "predicate": u["predicate"]}
                                               for u in unresolved]}
