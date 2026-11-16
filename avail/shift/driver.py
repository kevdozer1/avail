"""The scripted cycle driver: 1 cycle of a task on a cell, step by step, the way an agent would run it.

Core code. It knows the task only from its task file (``avail.core.task``) and the cell only through
a small duck-typed interface, so it never imports a simulator or anything plug-specific:

- ``cell.variant`` (the plug station: the connector type), ``cell.n_items``
- ``cell.target_of(item) -> dict``: what a claim about the item names besides the item, for the
  cable loop its assigned port
- ``cell.run_skill(step, item, attempt, marks=None) -> result`` with ``ok``, ``flag`` and ``seconds``
- ``cell.claim_params(item, step, result) -> dict`` and ``cell.view`` (what evidence sources get). The
  cell takes from the skill's result what its checks read
- ``cell.now()`` (the cell's clock, seconds), ``cell.begin_assist() -> frame index`` and
  ``cell.end_assist() -> frame index`` (the handoff and the handback in the cell's frame log)
- for a task file with a ``fork.ladder`` (2026-11-16, docs/07): ``cell.rung_begin(name, item, step) -> result or None``
  (the recovery action, which may move the robot, and the step's new result when it made one) and
  ``cell.rung_end(name, item, step)`` (back where the rung started, after a look from elsewhere)
- a person's marks reach the cell as ``run_skill(..., marks=...)`` on the next try, and the cell reads them

Arms (docs/07, E3): ``flag`` trusts each step's own flag. ``vlm`` trusts the flags, except that a
judge decides the steps the task file marks ``judge: true`` (the cable loop's insertion). ``avail`` verifies every claim with the task's witnesses and follows
the fork policy with no person: an abstain or a last reject leaves the item unresolved.
``avail_simop`` ("AVAIL + simulated operator") is the same with an operator behind the fork, in sim one that
answers from simulator truth and takes over with a scripted correction. ``avail_human`` is that arm's name as E3
recorded it (renamed 2026-11-13), still accepted so its ledgers read the same. ``agent_looks`` (S5c, the figure run) is an agent that
checks its own work by looking: after every step a judge sees the step's frames (and, when the testbed gives them,
the robot's readings), and its yes or no decides, with the flag arm's tries.

The done gate (``avail.fork.gate``) is the only thing that declares an AVAIL arm's item done: every
required predicate has an accept verdict or a person's sign-off. The flag and vlm arms do not use
it: they declare done by their own rule, which is what they stand for.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..schemas import Claim, ClaimContext, Predicate, new_id

ARMS = ("flag", "vlm", "avail", "avail_simop", "avail_human", "agent_looks")
JUDGED_ARMS = ("vlm", "agent_looks")
OPERATOR_ARMS = ("avail_simop", "avail_human")        # avail_human: E3's recorded name for avail_simop
GATED_ARMS = ("avail", *OPERATOR_ARMS)


@dataclass
class ItemLog:
    item: int
    declared_done: bool = False
    unresolved: str | None = None              # the step left open, when any
    attempts: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)   # what the skill reported (evaluator-side copy, never used to decide)
    interventions: int = 0
    takeovers: int = 0


@dataclass
class CycleLog:
    cycle_id: str
    arm: str
    items: list = field(default_factory=list)
    motion_s: float = 0.0
    verify_s: float = 0.0
    model_s: float = 0.0
    wait_s: float = 0.0
    handling_s: float = 0.0
    usd: float = 0.0
    assists: int = 0
    takeovers: int = 0
    forks: int = 0

    @property
    def shift_s(self) -> float:
        return self.motion_s + self.verify_s + self.model_s + self.wait_s + self.handling_s


class CycleDriver:
    """Runs 1 cycle. ``witnesses`` maps a predicate to an object with ``verify(view, claim, params)``."""

    def __init__(self, task, cell, arm: str, *, witnesses: dict | None = None, human=None, ledger=None,
                 judge=None, run_id: str = "shift", cycle_id: str | None = None, max_attempts: int | None = None,
                 evaluator=None):
        from ..fork.gate import DoneGate

        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
        if arm in OPERATOR_ARMS and human is None:
            raise ValueError(f"the {arm} arm needs an operator")
        self.task, self.cell, self.arm = task, cell, arm
        self.witnesses = witnesses or {}
        self.human = human
        self.ledger = ledger
        self.judge = judge
        self.run_id = run_id
        self.cycle_id = cycle_id or new_id("cycle")
        self.max_attempts = int(max_attempts if max_attempts is not None else task.fork.get("max_attempts", 2))
        self.ask = dict(task.fork.get("ask", {}))
        self.gate = DoneGate(task, cell.variant)
        self.log = CycleLog(self.cycle_id, arm)
        # The evaluator's hook, from the testbed: after a step is declared done, was it? Its answers go to
        # ``self.audit`` for the report and never into a decision.
        self.evaluator = evaluator
        self.audit: list[dict] = []
        # 2026-11-16 (docs/07, step 2): the escalation ladder from the task file, and 2 more evaluator-side records:
        # every verdict with the claim's truth after its witness ran, and every rung with the truth before and after it
        self.ladder = {k: list(v) for k, v in dict(task.fork.get("ladder", {})).items()}
        self.ladder_after = {k: list(v) for k, v in dict(task.fork.get("ladder_after", {})).items()}
        self.claim_audit: list[dict] = []
        self.rung_audit: list[dict] = []
        self._rung_now: str | None = None

    # ------------------------------------------------------------------ the ledger
    def _emit(self, kind: str, payload) -> None:
        if self.ledger is not None:
            self.ledger.append(kind, payload)

    # ------------------------------------------------------------------ 1 cycle
    def run(self) -> CycleLog:
        self._emit("cycle_start", {"cycle_id": self.cycle_id, "arm": self.arm, "task": self.task.task,
                                   "variant": self.cell.variant, "items": list(range(self.cell.n_items))})
        for item in range(self.cell.n_items):
            self.log.items.append(self._run_item(item))
        return self.log

    def _claim(self, item: int, step, attempt: int) -> Claim:
        predicate = step.claim_for(self.cell.variant)
        spec = self.task.predicates[predicate]
        return Claim(claim_id=new_id("claim"), run_id=self.run_id, cycle_id=self.cycle_id, step_id=f"{item}.{step.name}",
                     predicate=Predicate(name=predicate, target={"item": int(item), **dict(self.cell.target_of(item))},
                                         definition=spec.done_means),
                     raised_by="skill", context=ClaimContext(task=self.task.task, stage=step.name, attempt=attempt))

    def _run_item(self, item: int) -> ItemLog:
        log = ItemLog(item)
        for step in self.task.steps:
            self._decided_by = None
            done = self._run_step(item, step, log)
            if self.evaluator is not None:
                predicate = step.claim_for(self.cell.variant)
                self.audit.append({"item": item, "step": step.name, "predicate": predicate, "declared": bool(done),
                                   "by": self._decided_by, "attempts": log.attempts.get(step.name, 0),
                                   "true": bool(self.evaluator(item, predicate))})
            if not done:
                log.unresolved = step.name
                return log
        if self.arm in GATED_ARMS:
            ok, _ = self.gate.done(item)
            log.declared_done = bool(ok)
        else:
            log.declared_done = True
        return log

    def _run_step(self, item: int, step, log: ItemLog) -> bool:
        """1 step until it is done, given up, or handed to a person. True when the step counts as done."""
        attempt, marks, asks, laddered = 0, None, 0, False
        predicate = step.claim_for(self.cell.variant)
        while True:
            res = self.cell.run_skill(step.name, item, attempt, marks=marks)
            marks = None
            self.log.motion_s += float(res.seconds)
            log.attempts[step.name] = attempt + 1
            if getattr(res, "failure", None):
                log.failures.append(f"{step.name}:{res.failure}")
            claim = self._claim(item, step, attempt)
            self._emit("claim", claim)
            self._emit("tool_call", {"tool": "wam_act", "cycle_id": self.cycle_id, "claim_id": claim.claim_id,
                                     "skill": step.name, "item": item, "attempt": attempt, "ok": bool(res.ok),
                                     "flag": bool(res.flag), "seconds": float(res.seconds)})
            if self.arm == "flag" or (self.arm == "vlm" and not step.judge):
                if res.flag:
                    self._decided_by = "flag"
                    return True
                if attempt + 1 < self.max_attempts:
                    attempt += 1
                    continue
                return False
            if self.arm in JUDGED_ARMS:
                holds = self._judge(item, step, claim, res)
                if holds:
                    self._decided_by = "judge"
                    return True
                # docs/07: the vlm arm's judge decides with up to 2 retries. agent_looks has the flag arm's tries
                if attempt + 1 < (3 if self.arm == "vlm" else self.max_attempts):
                    attempt += 1
                    continue
                return False
            verdict = self._verify(item, step, claim, res)
            if verdict.decision == "accept":
                self.gate.record_accept(item, predicate, verdict)
                self._decided_by = "verdict"
                return True
            plan = self._fork(verdict, attempt, step)
            if plan.chosen == "retry":
                attempt += 1
                continue
            if plan.chosen != "ask_human" or self.human is None:
                return False                             # abort: the item stays unresolved
            if (not laddered and asks == 0 and self.ladder.get(predicate)
                    and verdict.decision in self.ladder_after.get(predicate, ("abstain", "reject"))):
                # before the 1st call to the operator on this step: the task's rungs, each followed by the same witness
                laddered = True
                outcome, claim, verdict, res = self._run_ladder(item, step, claim, verdict, res, attempt)
                if outcome == "accept":
                    return True
                plan = self._fork(verdict, attempt, step)  # what happens next comes from the last verdict
                if plan.chosen == "retry":                 # an abstain on a try with 1 left, rejected after its rung
                    attempt += 1
                    continue
                if plan.chosen != "ask_human":
                    return False
            asks += 1
            if asks > 1:                                 # back at a person after their own fix failed: they take over
                outcome = self._takeover(item, step, claim, verdict, log)
                return outcome == "done"
            answer = self._ask(item, step, claim, verdict, plan, plan.question, log)
            outcome = self._after_answer(item, step, claim, verdict, plan, answer, log, attempt)
            if outcome == "done":
                return True
            if outcome == "retry":
                attempt += 1
                if answer.correction_kind == "marks":
                    marks = answer.correction
                continue
            return False

    # ------------------------------------------------------------------ verification
    def _verify(self, item: int, step, claim: Claim, res):
        predicate = claim.predicate.name
        witness = self.witnesses[predicate]
        params = self.cell.claim_params(item, step.name, res)
        t0 = self.cell.now()
        verdict, evidence = witness.verify(self.cell.view, claim, params)
        # probes move the robot, so their time is the cell's clock. A look or a judge states its own cost
        self.log.verify_s += max(0.0, self.cell.now() - t0) + sum(
            float(e.cost.seconds) for e in evidence if not e.cost.moved_robot)
        self.log.usd += float(verdict.cost_total.usd)
        for ev in evidence:
            self._emit("evidence", ev)
        self._emit("verdict", verdict)
        self._last_features = {f"{e.source_id}/{k}": v for e in evidence for k, v in e.features.items()}
        if self.evaluator is not None:                   # evaluator-side, for the report: never read by a decision
            self.claim_audit.append({"item": int(item), "step": step.name, "predicate": predicate,
                                     "attempt": int(claim.context.attempt), "decision": verdict.decision,
                                     "stage": int(verdict.stage_reached), "rung": self._rung_now,
                                     "true": bool(self.evaluator(item, predicate))})
        return verdict

    def _run_ladder(self, item: int, step, claim: Claim, verdict, res, attempt: int):
        """The escalation ladder (docs/07, 2026-11-16): each rung is the cell's own recovery action, then the predicate's
        witness on a fresh claim of the step at the same try. A rung never decides: only an accept from the witness ends
        the ladder. Returns (``accept`` or ``operator``, the last claim, its verdict, the step's result)."""
        predicate = claim.predicate.name
        for rung in self.ladder.get(predicate, []):
            before = bool(self.evaluator(item, predicate)) if self.evaluator is not None else None
            t0 = self.cell.now()
            out = self.cell.rung_begin(rung, item, step.name)
            moved = max(0.0, self.cell.now() - t0)
            self.log.motion_s += moved
            res = out if out is not None else res
            claim = self._claim(item, step, attempt)
            self._emit("claim", claim)
            self._emit("tool_call", {"tool": f"rung:{rung}", "cycle_id": self.cycle_id, "claim_id": claim.claim_id,
                                     "skill": step.name, "item": item, "attempt": attempt, "after": verdict.decision,
                                     "ok": bool(getattr(out, "ok", True)), "seconds": moved})
            decision_before = verdict.decision
            self._rung_now = rung
            try:
                verdict = self._verify(item, step, claim, res)
            finally:
                self._rung_now = None
            t1 = self.cell.now()
            self.cell.rung_end(rung, item, step.name)
            self.log.motion_s += max(0.0, self.cell.now() - t1)
            if self.evaluator is not None:
                self.rung_audit.append({"item": int(item), "step": step.name, "predicate": predicate, "rung": rung,
                                        "after": decision_before, "decision": verdict.decision, "true_before": before,
                                        "true_after": bool(self.evaluator(item, predicate)), "seconds": moved})
            if verdict.decision == "accept":
                self.gate.record_accept(item, predicate, verdict)
                self._decided_by = f"rung:{rung}"
                return "accept", claim, verdict, res
        return "operator", claim, verdict, res

    def _judge(self, item: int, step, claim: Claim, res) -> bool:
        """A judged arm: a judge's answer on a judged step. Its text never becomes a verdict: it only stops a retry loop."""
        params = self.cell.claim_params(item, step.name, res)
        ev = self.judge.collect(self.cell.view, claim, params)
        self._emit("evidence", ev)
        self.log.model_s += float(ev.cost.seconds)
        self.log.usd += float(ev.cost.usd)
        return bool(ev.features.get("holds", 0.0) >= 0.5)

    def _fork(self, verdict, attempt: int, step):
        from ..fork.policy import decide, fork_record

        # a <predicate>_reject key of the task file's fork.ask sets the question for that predicate's last reject
        plan = decide(verdict.decision, attempt, self.max_attempts, self.human is not None, self.ask,
                      predicate=step.claim_for(self.cell.variant))
        fork = fork_record(verdict.verdict_id, plan)
        self._emit("fork", fork)
        self.log.forks += 1
        self._last_fork = fork
        return plan

    # ------------------------------------------------------------------ a person
    def _ask(self, item: int, step, claim: Claim, verdict, plan, question: str, log: ItemLog, present: bool = False):
        """Ask the person. ``present``: they are at the cell already (a takeover right after their own answer), so
        this is no new call for help and they do not wait to be called."""
        from ..fork.human import HumanRequest

        frames = None
        try:
            obs = self.cell.view.observe()
            frames = {name: fr for name, fr in obs.frames.items()}
        except Exception:                                # a request without pictures still works
            frames = None
        options = ["retry", "takeover", "abort"] if question == "choose" else []
        request = HumanRequest(fork_id=self._last_fork.fork_id, claim=claim, question=question, options=options,
                               frames=frames, context={"item": item, "step": step.name, "predicate": claim.predicate.name,
                                                       "verdict": verdict.decision})
        handoff_frame = None
        if question == "takeover":
            handoff_frame = self.cell.begin_assist()
        answer = self.human.ask(request)
        handback_frame = self.cell.end_assist() if question == "takeover" else None
        wait = 0.0 if present else float(answer.seconds_to_answer)
        self.log.wait_s += wait
        self.log.handling_s += float(answer.seconds_handling)
        if answer.handoff:
            self.log.motion_s += float(answer.handoff.get("robot_seconds", 0.0))
        if not present:
            self.log.assists += 1
        log.interventions += 1
        if question == "takeover":
            self.log.takeovers += 1
            log.takeovers += 1
        self._record_assist(item, step, claim, verdict, question, answer, handoff_frame, handback_frame, wait,
                            offered=list(request.options), present=present)
        return answer

    def _record_assist(self, item, step, claim, verdict, question, answer, handoff_frame, handback_frame,
                       wait: float, offered: list[str] | None = None, present: bool = False) -> None:
        from ..schemas import Assist, Correction, Intervention

        assist_id = new_id("assist")
        correction = None
        if answer.correction_kind:
            correction = Correction(kind=answer.correction_kind, payload=dict(answer.correction))
        assist = Assist(assist_id=assist_id, fork_id=self._last_fork.fork_id, question=question,
                        options=list(offered or []), answer=str(answer.answer),    # what the person was offered
                        truth_label=answer.truth_label, correction=correction,
                        seconds_to_answer=wait, seconds_handling=float(answer.seconds_handling),
                        assistant_id=getattr(self.human, "assistant_id", "human"))
        self._emit("assist", assist)
        self._last_assist = assist
        # The intervention record: 1 record, 3 uses (avail.flywheel.intervention): the label for calibration, the
        # takeover's frames for a correction demo, and the shift's numbers. It is written once its outcome is known.
        # Its label goes with the evidence of the claim at the fork. For a takeover that is the person's look at the
        # step when they took the arm, not the sign-off after their fix, which the Assist carries for the done gate.
        # A claim gets 1 label: a takeover right after the person's own labeled answer on it adds none.
        label = (answer.handoff or {}).get("before") if question == "takeover" else answer.truth_label
        if question == "takeover" and present and getattr(self, "_labeled_claim", None) == claim.claim_id:
            label = None
        if label is not None:
            self._labeled_claim = claim.claim_id
        now = time.time()
        handle = float(answer.seconds_handling)
        self._pending_iv = dict(
            intervention_id=new_id("iv"), run_id=self.run_id, cycle_id=self.cycle_id, item=str(item), step=step.name,
            predicate_name=claim.predicate.name, claim_id=claim.claim_id, verdict_id=verdict.verdict_id,
            fork_id=self._last_fork.fork_id, assist_id=assist_id, question=question,
            trigger=f"{verdict.decision} at attempt {claim.context.attempt + 1}", answer=str(answer.answer),
            truth_label=label, asked_ts=now, answered_ts=now + wait, seconds_waiting=wait,
            seconds_handling=handle, handoff_ts=(now + wait) if question == "takeover" else None,
            handback_ts=(now + wait + handle) if question == "takeover" else None,
            handoff_frame=handoff_frame, handback_frame=handback_frame,
            frames_path=f"frames/{self.cycle_id}.parquet" if question == "takeover" else None,
            features=dict(getattr(self, "_last_features", {})))

    def _close_iv(self, outcome: str) -> None:
        from ..schemas import Intervention

        fields = getattr(self, "_pending_iv", None)
        if fields is None:
            return
        self._emit("intervention", Intervention(**fields, outcome=outcome))
        self._pending_iv = None

    def _after_answer(self, item, step, claim, verdict, plan, answer, log: ItemLog, attempt: int) -> str:
        """What the answer means: done (a person signed it off), retry, or not done."""
        predicate = claim.predicate.name
        question = plan.question
        if question == "confirm":
            if answer.answer == "yes":
                self.gate.record_signoff(item, predicate, self._last_assist)
                self._close_iv("signed_off")
                self._decided_by = "person"
                return "done"
            self._close_iv("retried" if attempt + 1 < self.max_attempts else "unresolved")   # a no with no try left: a takeover follows, as its own record
            if attempt + 1 < self.max_attempts:
                return "retry"
            return self._takeover(item, step, claim, verdict, log, present=True)
        if question == "choose":
            if answer.answer == "retry":
                self._close_iv("retried")
                return "retry"
            if answer.answer == "takeover":
                self._close_iv("unresolved")       # the person chose to take over: that is its own record
                return self._takeover(item, step, claim, verdict, log, present=True)
            self._close_iv("unresolved")
            return "abort"
        if question == "guide_marks":
            self._close_iv("retried")
            return "retry"
        if question == "takeover":
            ok = answer.truth_label is True               # after the handback the person says whether it is done
            if ok:
                self.gate.record_signoff(item, predicate, self._last_assist)
                self._decided_by = "takeover"
            self._close_iv("taken_over_done" if ok else "taken_over_failed")
            return "done" if ok else "abort"
        self._close_iv("unresolved")
        return "abort"

    def _takeover(self, item, step, claim, verdict, log: ItemLog, present: bool = False) -> str:
        """A person takes over the step from where it is. Their sign-off after the handback marks it done.
        ``present``: they are at the cell already, right after answering, so it is the same call for help."""
        from ..fork.policy import ForkPlan

        plan = ForkPlan(options=["ask_human"], chosen="ask_human", question="takeover", reason="the person takes over")
        answer = self._ask(item, step, claim, verdict, plan, "takeover", log, present=present)
        return self._after_answer(item, step, claim, verdict, plan, answer, log, attempt=99)
