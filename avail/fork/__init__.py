"""What happens after a verdict that is not accept. accept is never an option here.

- :mod:`~avail.fork.policy`: the default fork policy and the allowed-options contract
- :mod:`~avail.fork.gate`: the done gate behind ``avail_done``
- :mod:`~avail.fork.human`: the Human protocol, what a person is shown and what they answer
- :mod:`~avail.fork.handoff`: takeover, and which frames of a cycle a person drove

Built in session S5.
"""

from .gate import DoneGate
from .handoff import ASSISTED, AUTONOMOUS, Handback, HandoffLog, Takeover, TeleopSource
from .human import Human, HumanAnswer, HumanRequest, ScriptedHuman, answer_for, check_answer, to_assist
from .policy import DEFAULT_QUESTIONS, ForkPlan, choose, decide, fork_record

__all__ = [
    "ASSISTED", "AUTONOMOUS", "DEFAULT_QUESTIONS", "DoneGate", "ForkPlan", "Handback", "HandoffLog", "Human",
    "HumanAnswer", "HumanRequest", "ScriptedHuman", "Takeover", "TeleopSource", "answer_for", "check_answer",
    "choose", "decide", "fork_record", "to_assist",
]
