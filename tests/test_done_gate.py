"""The done gate (CLAUDE.md rule 11): only evidence or a person marks a step done.

4 layers, each with a test that can fail:

1. A Verdict that accepts must list evidence ids, however it is built or read back.
2. No decider at a fork can express accept: Fork refuses it in options, chosen and shadow.choice.
3. A static rule over the source: outside avail/witness and avail/schemas nothing builds a Verdict
   or changes a decision, and no function under avail/fork or avail/agent takes a decision as free
   text. The scanner is checked on snippets first, then run on the real package.
4. At run time (S5): the fork policy never offers or picks accept, text never becomes a verdict or a
   sign-off, and the gate behind avail_done refuses an early done and says ok once every required
   predicate of every item has an accept or a person's yes, live and rebuilt from the ledger.
"""

from __future__ import annotations

import itertools
import json

import pytest
from pydantic import ValidationError

from avail.fork import (
    DoneGate, ForkPlan, HumanAnswer, HumanRequest, ScriptedHuman, answer_for, check_answer, decide, fork_record, to_assist,
)
from avail.ledger import LedgerWriter, iter_events, load_ledger
from avail.schemas import (
    Assist, Claim, ClaimContext, Cost, Fork, Intervention, Predicate, SchemaError, Verdict, explain, is_fork_option,
    kinds, validate_record,
)
from avail.schemas import gate_rules
from avail.schemas.gate_rules import GateViolation, package_sources, scan_source, scan_tree

EVIDENCE_RULE = "an accept needs evidence"
ACCEPT_BAN = "accept is never a fork option"


def verdict_fields(**changes):
    """A valid accept. Tests change 1 thing at a time."""
    return {
        "verdict_id": "verdict_0001", "claim_id": "claim_0001", "decision": "accept", "score": 0.93,
        "stage_reached": 2, "witness_version": "v3", "evidence_ids": ["ev_0001"],
        "cost_total": Cost(seconds=1.2, usd=0.0, moved_robot=True),
        **changes,
    }


def fork_fields(**changes):
    return {"fork_id": "fork_0001", "verdict_id": "verdict_0002", "options": ["retry", "ask_human"], **changes}


# ------------------------------------------------------------------ 1. a verdict that accepts needs evidence
def test_verdict_accept_without_evidence_ids_raises():
    with pytest.raises(ValidationError) as caught:
        Verdict(**verdict_fields(evidence_ids=[]))
    assert caught.value.error_count() == 1                   # the missing evidence is the only thing wrong
    error = caught.value.errors()[0]
    assert error["loc"] == ("evidence_ids",) and error["type"] == "accept_needs_evidence"
    assert EVIDENCE_RULE in str(caught.value)                # the plain pydantic text says it too
    line = explain(caught.value)[0]
    assert line.startswith(f"evidence_ids: {EVIDENCE_RULE}")
    assert "Fix: list the ids of the Evidence records the witness used" in line


def test_verdict_accept_with_evidence_ids_constructs():
    verdict = Verdict(**verdict_fields())
    assert verdict.decision == "accept" and verdict.evidence_ids == ["ev_0001"]


def test_leaving_evidence_ids_out_of_an_accept_raises_too():
    fields = verdict_fields()
    del fields["evidence_ids"]
    with pytest.raises(ValidationError, match=EVIDENCE_RULE):
        Verdict(**fields)


@pytest.mark.parametrize("decision", ["reject", "abstain"])
def test_only_accept_needs_evidence(decision):
    verdict = Verdict(**verdict_fields(decision=decision, evidence_ids=[]))
    assert verdict.decision == decision and verdict.evidence_ids == []


@pytest.mark.parametrize("ids", [[""], ["   "], ["", "  "], [None], [0]])
def test_an_empty_id_does_not_count_as_evidence(ids):
    with pytest.raises(ValidationError) as caught:
        Verdict(**verdict_fields(evidence_ids=ids))
    assert all(error["loc"][0] == "evidence_ids" for error in caught.value.errors())


def test_a_logged_accept_without_evidence_does_not_read_back():
    row = json.loads(Verdict(**verdict_fields()).model_dump_json())
    assert Verdict.model_validate(row).decision == "accept"                  # the honest row reads back

    row["evidence_ids"] = []
    with pytest.raises(ValidationError, match=EVIDENCE_RULE):
        Verdict.model_validate(row)
    with pytest.raises(ValidationError, match=EVIDENCE_RULE):
        Verdict.model_validate_json(json.dumps(row))
    with pytest.raises(SchemaError, match=EVIDENCE_RULE):
        validate_record("verdict", row)
    del row["evidence_ids"]
    with pytest.raises(ValidationError, match=EVIDENCE_RULE):
        Verdict.model_validate(row)


def test_a_verdict_cannot_become_an_accept_after_it_is_built():
    reject = Verdict(**verdict_fields(decision="reject", evidence_ids=[]))

    with pytest.raises(ValidationError) as caught:           # frozen: no assignment at all
        reject.decision = "accept"
    assert caught.value.errors()[0]["type"] == "frozen_instance"
    with pytest.raises(ValidationError):
        reject.evidence_ids = ["ev_made_up"]
    with pytest.raises(ValidationError):
        del reject.decision

    with pytest.raises(ValidationError, match=EVIDENCE_RULE):                # a copy with an update is validated
        reject.model_copy(update={"decision": "accept"})
    with pytest.raises(ValidationError, match=EVIDENCE_RULE):
        reject.copy(update={"decision": "accept"})

    with pytest.raises(TypeError, match="skips validation"):                 # no unvalidated construction
        Verdict.model_construct(**verdict_fields(evidence_ids=[]))
    with pytest.warns(DeprecationWarning), pytest.raises(TypeError, match="skips validation"):
        Verdict.construct(**verdict_fields(evidence_ids=[]))

    assert reject.decision == "reject" and reject.evidence_ids == []


def test_an_accept_cannot_shed_its_evidence_through_a_copy():
    accept = Verdict(**verdict_fields())
    with pytest.raises(ValidationError, match=EVIDENCE_RULE):
        accept.model_copy(update={"evidence_ids": []})
    assert accept.model_copy(update={"score": 0.99}).evidence_ids == ["ev_0001"]


@pytest.mark.parametrize("text", [
    "Accept", "ACCEPT", "accept.", " accept", "accepted", "yes", "done", "ok", "pass", "true", "seated",
    "The plug looks fully seated, so accept.", "", None, True, 1,
])
def test_free_text_is_not_read_as_a_decision(text):
    """A model's words do not become a decision. Only the 3 exact words are decisions."""
    with pytest.raises(ValidationError) as caught:
        Verdict(**verdict_fields(decision=text))
    assert caught.value.errors()[0]["loc"] == ("decision",)
    assert "use 'accept', 'reject' or 'abstain'" in explain(caught.value)[0]


# ------------------------------------------------------------------ 2. no decider can express accept
def _spellings_of_accept() -> list[str]:
    cased = {"".join(letters) for letters in itertools.product(*((c, c.upper()) for c in "accept"))}
    assert len(cased) == 64
    wrapped = [f"{before}{word}{after}"
               for word in sorted(cased)
               for before, after in (("", ""), ("repair:", ""), ("recover:", ""), (" ", " "), ("", "ed"), ("auto_", ""))]
    return wrapped + ["ａｃｃｅｐｔ", "repair:ａｃｃｅｐｔ", "accept\n", "recover:please_accept_this"]


ACCEPT_SPELLINGS = _spellings_of_accept()


def test_there_are_many_spellings_to_refuse():
    assert len(ACCEPT_SPELLINGS) == 64 * 6 + 4 and len(set(ACCEPT_SPELLINGS)) == len(ACCEPT_SPELLINGS)


def test_a_fork_without_accept_constructs():
    """The control for the 3 tests below: the same forks are fine once accept is gone."""
    fork = Fork(**fork_fields(chosen="retry", chosen_by="agent",
                              shadow={"decider": "jev-d1", "choice": "ask_human", "latency_s": 0.4, "usd": 0.0002}))
    assert fork.options == ["retry", "ask_human"] and fork.chosen == "retry" and fork.shadow.choice == "ask_human"


def test_fork_refuses_accept_in_options():
    for text in ACCEPT_SPELLINGS:
        for options in ([text], ["retry", text], [text, "ask_human"]):
            with pytest.raises(ValidationError) as caught:
                Fork(**fork_fields(options=options))
            where = ("options", options.index(text))
            assert [error["loc"] for error in caught.value.errors()] == [where], text
            assert ACCEPT_BAN in explain(caught.value)[0], text


def test_fork_refuses_accept_as_chosen():
    for text in ACCEPT_SPELLINGS:
        for who in kinds.CHOSEN_BY:
            with pytest.raises(ValidationError) as caught:
                Fork(**fork_fields(chosen=text, chosen_by=who))
            assert [error["loc"] for error in caught.value.errors()] == [("chosen",)], text
            line = explain(caught.value)[0]
            assert ACCEPT_BAN in line and "Fix: use one of retry, ask_human, abort" in line, text


def test_fork_refuses_accept_as_the_shadow_choice():
    for text in ACCEPT_SPELLINGS:
        shadow = {"decider": "jev-d1", "choice": text, "latency_s": 0.4, "usd": 0.0002}
        with pytest.raises(ValidationError) as caught:
            Fork(**fork_fields(shadow=shadow))
        assert [error["loc"] for error in caught.value.errors()] == [("shadow", "choice")], text
        assert ACCEPT_BAN in explain(caught.value)[0], text


def test_a_decider_can_only_choose_from_the_options_it_was_given():
    open_fork = Fork(**fork_fields())
    for who in kinds.CHOSEN_BY:
        assert open_fork.model_copy(update={"chosen": "ask_human", "chosen_by": who}).chosen == "ask_human"
        for choice in ("accept", "abort", "repair:push_seat", "done", ""):
            with pytest.raises(ValidationError):
                open_fork.model_copy(update={"chosen": choice, "chosen_by": who})
    with pytest.raises(ValidationError):
        open_fork.chosen = "retry"                            # and never by changing the fork in place


def test_accept_is_missing_from_the_fork_vocabulary():
    assert "accept" in kinds.DECISIONS
    for word in (*kinds.FORK_OPTION_WORDS, *kinds.FORK_OPTION_PREFIXES):
        assert "accept" not in word.casefold()
    assert not any(is_fork_option(text) for text in ACCEPT_SPELLINGS)
    assert all(is_fork_option(word) for word in kinds.FORK_OPTION_WORDS)


# ------------------------------------------------------------------ 3. the static rule, on snippets
FORK_FILE = "avail/fork/policy.py"
AGENT_FILE = "avail/agent/mcp_server.py"
OTHER_FILE = "avail/shift/runner.py"
WITNESS_FILE = "avail/witness/cascade.py"
SCHEMAS_FILE = "avail/schemas/records.py"

# Each snippet breaks exactly 1 rule, once.
BUILDS_OR_CHANGES_A_VERDICT = {
    "calls Verdict": (
        "from avail.schemas import Verdict\n\ndef judge(fields):\n    return Verdict(**fields)\n",
        "verdict_built"),
    "calls Verdict with an accept": (
        "from avail.schemas import Verdict\n\nv = Verdict(verdict_id='v', claim_id='c', decision='accept')\n",
        "verdict_built"),
    "calls schemas.Verdict": (
        "from avail import schemas\n\ndef judge(fields):\n    return schemas.Verdict(**fields)\n",
        "verdict_built"),
    "calls avail.schemas.Verdict": (
        "import avail.schemas\n\ndef judge(fields):\n    return avail.schemas.Verdict(**fields)\n",
        "verdict_built"),
    "calls Verdict under another name": (
        "from avail.schemas import Verdict as Answer\n\ndef judge(fields):\n    return Answer(**fields)\n",
        "verdict_built"),
    "calls Verdict through an alias": (
        "from avail import schemas\n\nMake = schemas.Verdict\n\ndef judge(fields):\n    return Make(**fields)\n",
        "verdict_built"),
    "model_construct on Verdict": (
        "from avail.schemas import Verdict\n\ndef judge(fields):\n    return Verdict.model_construct(**fields)\n",
        "verdict_built"),
    "model_construct with a decision": (
        "def judge(kind, text):\n    return kind.model_construct(decision=text)\n",
        "verdict_built"),
    "model_validate on a literal": (
        "from avail.schemas import Verdict\n\ndef judge(text):\n"
        "    return Verdict.model_validate({'decision': text, 'evidence_ids': ['made_up']})\n",
        "verdict_built"),
    "validate_record on a literal": (
        "from avail.schemas import validate_record\n\ndef judge(text):\n"
        "    return validate_record('verdict', {'decision': text})\n",
        "verdict_built"),
    "subclasses Verdict": (
        "from avail.schemas import Verdict\n\nclass Loose(Verdict):\n    pass\n",
        "verdict_subclassed"),
    "assigns decision": (
        "def flip(record, text):\n    record.decision = text\n",
        "decision_assigned"),
    "assigns decision on self": (
        "class Result:\n    def __init__(self, text):\n        self.decision = text\n",
        "decision_assigned"),
    "augments decision": (
        "def flip(record):\n    record.decision += 'ed'\n",
        "decision_assigned"),
    "assigns decision in a tuple": (
        "def flip(record, a, b):\n    record.decision, record.score = a, b\n",
        "decision_assigned"),
    "deletes decision": (
        "def flip(record):\n    del record.decision\n",
        "decision_assigned"),
    "setattr decision": (
        "def flip(record, text):\n    setattr(record, 'decision', text)\n",
        "decision_assigned"),
    "object.__setattr__ decision": (
        "def flip(record, text):\n    object.__setattr__(record, 'decision', text)\n",
        "decision_assigned"),
    "__dict__ decision": (
        "def flip(record, text):\n    record.__dict__['decision'] = text\n",
        "decision_assigned"),
    "vars decision": (
        "def flip(record, text):\n    vars(record)['decision'] = text\n",
        "decision_assigned"),
    "__dict__ update decision": (
        "def flip(record, text):\n    record.__dict__.update(decision=text)\n",
        "decision_assigned"),
    "model_copy with a decision": (
        "def flip(record, text):\n    return record.model_copy(update={'decision': text})\n",
        "decision_updated"),
    "model_copy with dict(decision=)": (
        "def flip(record, text):\n    return record.model_copy(update=dict(decision=text), deep=True)\n",
        "decision_updated"),
    "model_copy with a spread decision": (
        "def flip(record, text, more):\n    return record.model_copy(update={**more, **{'decision': text}})\n",
        "decision_updated"),
    "old copy with a decision": (
        "def flip(record, text):\n    return record.copy(update={'decision': text})\n",
        "decision_updated"),
    "replace with a decision": (
        "import copy\n\ndef flip(record, text):\n    return copy.replace(record, decision=text)\n",
        "decision_updated"),
    "bare replace with a decision": (
        "from copy import replace\n\ndef flip(record, text):\n    return replace(record, decision=text)\n",
        "decision_updated"),
    "__replace__ with a decision": (
        "def flip(record, text):\n    return record.__replace__(decision=text)\n",
        "decision_updated"),
}

TAKES_A_DECISION_AS_TEXT = {
    "decision: str": "def resolve(fork, decision: str):\n    return decision\n",
    "decision untyped": "def resolve(fork, decision):\n    return decision\n",
    "verdict untyped": "def resolve(fork, verdict):\n    return verdict\n",
    "verdict: str": "def resolve(fork, verdict: str):\n    return verdict\n",
    "verdict_text": "def resolve(fork, verdict_text: str):\n    return verdict_text\n",
    "answer_text": "def resolve(fork, answer_text: str):\n    return answer_text\n",
    "optional str": "def resolve(fork, decision: str | None = None):\n    return decision\n",
    "Optional[str]": (
        "from typing import Optional\n\ndef resolve(fork, decision: Optional[str] = None):\n    return decision\n"),
    "quoted str": "def resolve(fork, decision: 'str'):\n    return decision\n",
    "Any": "from typing import Any\n\ndef resolve(fork, verdict: Any):\n    return verdict\n",
    "keyword only": "def resolve(fork, *, verdict_text: str = ''):\n    return verdict_text\n",
    "async method": "class Human:\n    async def answer(self, fork, answer_text: str):\n        return answer_text\n",
    "nested function": "def outer():\n    def inner(raw_decision: str):\n        return raw_decision\n    return inner\n",
    "lambda": "pick = lambda decision: decision\n",
    "kwargs": "def resolve(fork, **decision):\n    return decision\n",
    "model answer": "def resolve(fork, model_answer: str):\n    return model_answer\n",
    "decision_str": "def resolve(fork, decision_str: str):\n    return decision_str\n",
    "list of str": "def tally(decisions: list[str]):\n    return len(decisions)\n",
}

CLEAN = {
    "reads a logged verdict": (
        "from avail.schemas import Verdict, validate_record\n\n"
        "def load(row, line):\n"
        "    return Verdict.model_validate(row), Verdict.model_validate_json(line), validate_record('verdict', row)\n"),
    "reads a decision": (
        "from avail.schemas import Verdict\n\n"
        "def summarize(verdict: Verdict, rows):\n"
        "    done = verdict.decision == 'accept'\n"
        "    accepted = [row for row in rows if row['decision'] == 'accept']\n"
        "    return {'decision': verdict.decision, 'done': done, 'accepted': len(accepted)}\n"),
    "typed parameters": (
        "from avail.schemas import Decision, Fork, Verdict\n\n"
        "def options_for(verdict: Verdict, decision: Decision, attempts_left: int) -> list[str]:\n"
        "    return ['retry', 'ask_human'] if decision == 'reject' and attempts_left else ['ask_human']\n\n"
        "def choose(fork: Fork, option: str, verdict_id: str, answer: str) -> Fork:\n"
        "    return fork.model_copy(update={'chosen': option, 'chosen_by': 'agent'})\n"),
    "literal typed decision": (
        "from typing import Literal\n\n"
        "def options_for(decision: Literal['reject', 'abstain']) -> list[str]:\n"
        "    return ['ask_human']\n"),
    "isinstance and annotations": (
        "from avail.schemas import Verdict\n\n"
        "def is_verdict(thing) -> bool:\n"
        "    verdicts: list[Verdict] = []\n"
        "    return isinstance(thing, Verdict) or thing in verdicts\n"),
    "other attributes and copies": (
        "def close(fork, evidence, text):\n"
        "    fork.note = text\n"
        "    label = text.replace('a', 'b')\n"
        "    return fork.model_copy(update={'reason': label}), evidence.model_copy(update={'score': 0.5})\n"),
    "a stub": '"""The default fork policy.\n\nBuilt in session S5.\n"""\n',
}


@pytest.mark.parametrize("name", list(BUILDS_OR_CHANGES_A_VERDICT))
@pytest.mark.parametrize("path", [FORK_FILE, AGENT_FILE, OTHER_FILE, "avail/cli.py"])
def test_scanner_flags_code_that_builds_or_changes_a_verdict(name, path):
    source, rule = BUILDS_OR_CHANGES_A_VERDICT[name]
    found = scan_source(source, path)
    assert [violation.rule for violation in found] == [rule], found
    assert found[0].path == path and found[0].line >= 1
    assert found[0].fix == gate_rules.RULES[rule]
    assert str(found[0]).startswith(f"{path}:{found[0].line}: [{rule}] ") and ". Fix: " in str(found[0])


@pytest.mark.parametrize("name", list(BUILDS_OR_CHANGES_A_VERDICT))
@pytest.mark.parametrize("path", [WITNESS_FILE, SCHEMAS_FILE])
def test_scanner_lets_the_witness_and_the_schemas_build_verdicts(name, path):
    source, _rule = BUILDS_OR_CHANGES_A_VERDICT[name]
    assert scan_source(source, path) == []


@pytest.mark.parametrize("name", list(TAKES_A_DECISION_AS_TEXT))
@pytest.mark.parametrize("path", [FORK_FILE, AGENT_FILE, "avail/fork/human.py", "avail/agent/skilldoc/tools.py"])
def test_scanner_flags_a_fork_or_agent_function_that_takes_a_decision_as_text(name, path):
    found = scan_source(TAKES_A_DECISION_AS_TEXT[name], path)
    assert [violation.rule for violation in found] == ["text_decision_param"], found
    assert "which reads as a decision in free text" in found[0].what
    assert "Take a Verdict, or type the parameter as Decision" in found[0].fix


@pytest.mark.parametrize("name", list(TAKES_A_DECISION_AS_TEXT))
@pytest.mark.parametrize("path", [OTHER_FILE, WITNESS_FILE, "avail/stats/rates.py", "avail/cli.py"])
def test_the_parameter_rule_applies_to_fork_and_agent_only(name, path):
    assert scan_source(TAKES_A_DECISION_AS_TEXT[name], path) == []


@pytest.mark.parametrize("name", list(CLEAN))
@pytest.mark.parametrize("path", [FORK_FILE, AGENT_FILE, OTHER_FILE, WITNESS_FILE])
def test_scanner_passes_clean_code(name, path):
    assert scan_source(CLEAN[name], path) == []


def test_scanner_reports_every_violation_in_a_file_with_its_line():
    source = (
        "from avail.schemas import Verdict\n"                                   # 1
        "\n"                                                                    # 2
        "def decide(fork, verdict_text: str):\n"                                # 3  text_decision_param
        "    verdict = Verdict(decision=verdict_text)\n"                        # 4  verdict_built
        "    verdict.decision = 'accept'\n"                                     # 5  decision_assigned
        "    return verdict.model_copy(update={'decision': 'accept'})\n"        # 6  decision_updated
    )
    found = scan_source(source, FORK_FILE)
    assert [(violation.line, violation.rule) for violation in found] == [
        (3, "text_decision_param"), (4, "verdict_built"), (5, "decision_assigned"), (6, "decision_updated")]
    assert all(isinstance(violation, GateViolation) for violation in found)
    # the same file under avail/witness only has to be honest about itself
    assert scan_source(source, WITNESS_FILE) == []


def test_scanner_fails_closed_on_a_file_it_cannot_read():
    for path in (FORK_FILE, OTHER_FILE, WITNESS_FILE):
        for source in ("def broken(:\n    pass\n", "x = 1\x00\n", "x = (\n"):
            found = scan_source(source, path)
            assert [violation.rule for violation in found] == ["unparsed"], source
            assert "the file does not parse" in found[0].what and "Fix the syntax" in found[0].fix


def test_scanner_reads_names_exactly():
    # An attribute or a key that only looks like decision is something else.
    lookalikes = (
        "def note(record, text):\n"
        "    record.decisions = [text]\n"
        "    record.decision_note = text\n"
        "    setattr(record, 'Decision', text)\n"
        "    record.__dict__['decision_note'] = text\n"
        "    return record.model_copy(update={'decision_note': text})\n"
    )
    assert scan_source(lookalikes, OTHER_FILE) == []
    # validate_record takes its record name in any case, so the scanner does too
    for name in ("verdict", "Verdict", " VERDICT "):
        source = f"def judge(text):\n    return validate_record({name!r}, {{'decision': text}})\n"
        assert [violation.rule for violation in scan_source(source, OTHER_FILE)] == ["verdict_built"], name
    assert scan_source("def load(row):\n    return validate_record('fork', {'options': ['retry']})\n", OTHER_FILE) == []


def test_scanner_wants_a_path_under_the_package():
    with pytest.raises(ValueError, match="relative to the repo root"):
        scan_source("x = 1\n", "tests/test_done_gate.py")
    assert scan_source("x = 1\n", "avail\\fork\\policy.py") == []            # Windows separators are fine


def test_every_rule_has_a_fix_and_a_snippet_that_breaks_it():
    broken = {rule for _source, rule in BUILDS_OR_CHANGES_A_VERDICT.values()} | {"text_decision_param", "unparsed"}
    assert broken == set(gate_rules.RULES)
    assert all(fix and not fix.endswith(".") for fix in gate_rules.RULES.values())
    assert gate_rules.BUILDERS == ("witness", "schemas") and gate_rules.DECIDERS == ("fork", "agent")
    # each fix says what to do instead
    assert "To read a logged verdict use Verdict.model_validate(row)" in gate_rules.RULES["verdict_built"]
    assert "Use Verdict itself" in gate_rules.RULES["verdict_subclassed"]
    for rule in ("decision_assigned", "decision_updated"):
        assert "A decision is set once, by the witness, when the Verdict is built" in gate_rules.RULES[rule]
    assert "give it another name, such as outcome" in gate_rules.RULES["decision_assigned"]
    assert "Take a Verdict, or type the parameter as Decision" in gate_rules.RULES["text_decision_param"]
    assert "Fix the syntax" in gate_rules.RULES["unparsed"]


# ------------------------------------------------------------------ 3. the static rule, on a planted tree and the real tree
def test_scan_tree_finds_violations_planted_in_a_package(tmp_path):
    package = tmp_path / "avail"
    for folder in ("fork", "agent", "witness", "schemas", "shift"):
        (package / folder).mkdir(parents=True)
    (package / "fork" / "policy.py").write_text("def choose(fork, decision: str):\n    return decision\n", "utf-8")
    (package / "agent" / "proxy.py").write_text(
        "from avail.schemas import Verdict\n\ndef fake(text):\n    return Verdict(decision=text)\n", "utf-8")
    (package / "shift" / "runner.py").write_text("def flip(verdict):\n    verdict.decision = 'accept'\n", "utf-8")
    (package / "witness" / "cascade.py").write_text(
        "from avail.schemas import Verdict\n\ndef decide(fields):\n    return Verdict(**fields)\n", "utf-8")
    (package / "schemas" / "records.py").write_text("class Verdict:\n    pass\n\nv = Verdict()\n", "utf-8")
    (package / "fork" / "gate.py").write_text('"""The done gate."""\n', "utf-8")
    (package / "cli.py").write_text("def main(:\n", "utf-8")
    (package / "agent" / "blob.py").write_bytes(b"x = '\xff\xfe'\n")           # not UTF-8: cannot be read as text

    assert [path for path, _file in package_sources(package)] == [
        "avail/agent/blob.py", "avail/agent/proxy.py", "avail/cli.py", "avail/fork/gate.py", "avail/fork/policy.py",
        "avail/schemas/records.py", "avail/shift/runner.py", "avail/witness/cascade.py"]
    found = scan_tree(package)
    assert sorted((violation.path, violation.rule) for violation in found) == [
        ("avail/agent/blob.py", "unparsed"),
        ("avail/agent/proxy.py", "verdict_built"),
        ("avail/cli.py", "unparsed"),
        ("avail/fork/policy.py", "text_decision_param"),
        ("avail/shift/runner.py", "decision_assigned"),
    ]
    unread = [violation for violation in found if violation.path == "avail/agent/blob.py"][0]
    assert "the file cannot be read" in unread.what


def test_the_real_tree_scan_reads_the_whole_package():
    """Guards the 2 tests below against passing because the scan looked at nothing."""
    scanned = [path for path, _file in package_sources()]
    assert len(scanned) >= 40 and len(set(scanned)) == len(scanned)
    assert all(path.startswith("avail/") and path.endswith(".py") for path in scanned)
    for package in ("avail/fork/", "avail/agent/", "avail/witness/", "avail/schemas/", "avail/ledger/", "avail/shift/"):
        assert any(path.startswith(package) for path in scanned), package
    for must_have in ("avail/cli.py", "avail/schemas/records.py", "avail/schemas/gate_rules.py"):
        assert must_have in scanned, must_have
    assert gate_rules.PACKAGE_DIR.name == "avail" and (gate_rules.PACKAGE_DIR / "schemas" / "gate_rules.py").is_file()


def test_the_package_passes_the_gate_rules():
    """The whole of avail/, as it is on disk right now."""
    found = scan_tree()
    assert found == [], "the done gate rules are broken:\n" + "\n".join(str(violation) for violation in found)


def test_fork_and_agent_take_no_decision_as_text():
    """sessions/S1.md step 8: no function in avail/fork accepts free text as a decision. The same for avail/agent."""
    deciders = [(path, file) for path, file in package_sources()
                if path.startswith(("avail/fork/", "avail/agent/"))]
    assert any(path.startswith("avail/fork/") for path, _file in deciders)
    assert any(path.startswith("avail/agent/") for path, _file in deciders)
    found = [violation for path, file in deciders
             for violation in scan_source(file.read_text(encoding="utf-8-sig"), path)]
    assert found == [], "\n".join(str(violation) for violation in found)


def test_the_s5_modules_are_scanned_and_pass():
    """The fork, the gate, the human, the handoff and the rule witness are on disk and honest about themselves."""
    scanned = dict(package_sources())
    for path in ("avail/fork/policy.py", "avail/fork/gate.py", "avail/fork/human.py", "avail/fork/handoff.py",
                 "avail/witness/rules.py", "avail/flywheel/intervention.py", "avail/flywheel/export.py"):
        assert path in scanned, path
        assert scan_source(scanned[path].read_text(encoding="utf-8-sig"), path) == [], path


# ------------------------------------------------------------------ 4. at run time: no decider can return accept
ASK = {"abstain": "confirm", "reject_last": "choose", "at_port_reject": "guide_marks"}


def test_no_decider_at_a_fork_can_return_accept():
    # the policy, over every case it can meet
    for decision, attempt, most, human, ask, predicate in itertools.product(
            ("reject", "abstain"), range(4), range(1, 4), (True, False), (None, ASK), (None, "at_port", "seated")):
        plan = decide(decision, attempt, most, human, ask, predicate=predicate)
        fork = fork_record("verdict_1", plan)
        assert "accept" not in plan.options and plan.chosen != "accept"
        assert not any("accept" in option.casefold() for option in fork.options) and fork.chosen != "accept"
    with pytest.raises(ValueError, match="an accept opens no fork"):
        decide("accept", 0, 2, True)
    # a plan cannot hold accept in any spelling, as an option or as the pick
    for text in ACCEPT_SPELLINGS:
        with pytest.raises(ValueError, match="is not a fork option"):
            ForkPlan(["retry", text], "retry", None, "")
        with pytest.raises(ValueError):
            ForkPlan(["retry"], text, None, "")
    # a person is never offered accept, and cannot pick it off a menu that does not have it
    claim = _claim("seated", 1)
    for text in ("accept", "Accept", "recover:auto_accept"):
        with pytest.raises(ValueError, match="asks a person to accept"):
            HumanRequest("fork_1", claim, "choose", options=["retry", text])
    menu = HumanRequest("fork_1", claim, "choose", options=["retry", "abort"])
    with pytest.raises(ValueError, match="is not one of the options"):
        ScriptedHuman(lambda request: "accept").ask(menu)


# ------------------------------------------------------------------ 4. at run time: text never sets a verdict
class _Task:
    """A task spec as the gate reads it: required(variant), predicates and items."""

    task = "toy_insert"
    items = {"kind": "plug", "count": 2}
    predicates = {"seated": None, "latched": None, "released_stable": None, "held": None}

    def required(self, variant):
        return [("insert", "latched" if variant == "latch" else "seated"), ("release", "released_stable")]


def _claim(name: str, item, cycle_id: str = "cycle_1", claim_id: str | None = None) -> Claim:
    return Claim(claim_id=claim_id or f"claim_{name}_{item}", run_id="run_gate", cycle_id=cycle_id, step_id=name,
                 predicate=Predicate(name=name, target={"item": item, "port": 3}), raised_by="skill",
                 context=ClaimContext(task="toy_insert", stage=name, attempt=0))


def _verdict(decision: str, claim_id: str, verdict_id: str | None = None) -> Verdict:
    return Verdict(verdict_id=verdict_id or f"verdict_{claim_id}_{decision}", claim_id=claim_id, decision=decision,
                   stage_reached=1, witness_version="v1", evidence_ids=["ev_1"] if decision == "accept" else [],
                   cost_total=Cost.zero())


def _assist(fork_id: str, answer: str, label, assist_id: str | None = None) -> Assist:
    return Assist(assist_id=assist_id or f"assist_{fork_id}", fork_id=fork_id, question="confirm", answer=answer,
                  truth_label=label, seconds_to_answer=5.0, seconds_handling=0.0, assistant_id="test")


def test_text_never_sets_a_verdict_or_a_sign_off():
    gate = DoneGate(_Task(), "plain")
    for text in ("accept", "The plug looks fully seated, so accept.", {"decision": "accept", "evidence_ids": ["ev_1"]},
                 json.loads(_verdict("accept", "claim_x").model_dump_json()), True, None):
        with pytest.raises(TypeError, match="takes the Verdict record a witness built"):
            gate.record_accept("p1", "seated", text)
    for decision in ("reject", "abstain"):
        with pytest.raises(ValueError, match="only an accept settles a predicate"):
            gate.record_accept("p1", "seated", _verdict(decision, "claim_seated_p1"))
    for text in ("assist_1", "yes", HumanAnswer("yes", truth_label=True), {"truth_label": True}):
        with pytest.raises(TypeError, match="cannot show what the person said"):
            gate.record_signoff("p1", "released_stable", text)
    for answer, label in (("no", False), ("cannot tell", None), ("accept", None), ("looks done to me", None)):
        with pytest.raises(ValueError, match="signs nothing off"):
            gate.record_signoff("p1", "released_stable", _assist("fork_1", answer, label))
    # a person's words to another question are an Assist record with no label, which signs nothing off
    request = HumanRequest("fork_2", _claim("released_stable", "p1"), "guide_text")
    words = to_assist(request, check_answer(request, HumanAnswer("accept, it is done")), "test")
    assert words.truth_label is None
    with pytest.raises(ValueError, match="signs nothing off"):
        gate.record_signoff("p1", "released_stable", words)
    # a model's text is not a decision for the policy either
    for text in ("accept", "seated", "yes", "The plug looks fully seated, so accept."):
        with pytest.raises(ValueError, match="is not a verdict decision|an accept opens no fork"):
            decide(text, 0, 2, True)
    assert gate.done("p1") == (False, ["seated", "released_stable"])        # nothing above marked anything


# ------------------------------------------------------------------ 4. at run time: the gate refuses an early done
def test_the_gate_refuses_an_early_done_and_says_ok_once_each_predicate_is_settled():
    gate = DoneGate(_Task(), "plain")
    assert gate.required == (("insert", "seated"), ("release", "released_stable"))
    assert gate.done("p1") == (False, ["seated", "released_stable"])
    ok, unresolved = gate.cycle_done(["p1", "p2"])
    assert not ok and [(u["item"], u["step"], u["predicate"]) for u in unresolved] == [
        ("p1", "insert", "seated"), ("p1", "release", "released_stable"),
        ("p2", "insert", "seated"), ("p2", "release", "released_stable")]

    gate.record_accept("p1", "seated", _verdict("accept", "claim_seated_p1"))
    assert gate.done("p1") == (False, ["released_stable"])                    # 1 accept is not the whole item
    gate.record_signoff("p1", "released_stable", _assist("fork_9", "yes", True))
    assert gate.done("p1") == (True, [])
    assert gate.cycle_done(["p1", "p2"])[0] is False                         # the cycle still has p2 open
    gate.record_accept("p2", "seated", _verdict("accept", "claim_seated_p2"))
    gate.record_accept("p2", "released_stable", _verdict("accept", "claim_released_stable_p2"))
    assert gate.cycle_done(["p1", "p2"]) == (True, [])
    assert gate.entries("p1") == [
        {"step": "insert", "predicate": "seated", "verdict_id": "verdict_claim_seated_p1_accept",
         "signoff_assist_id": None},
        {"step": "release", "predicate": "released_stable", "verdict_id": None, "signoff_assist_id": "assist_fork_9"}]


def test_the_gate_asks_for_the_variant_s_own_predicate_and_nothing_else_counts():
    gate = DoneGate(_Task(), "latch")
    gate.record_accept(0, "seated", _verdict("accept", "claim_seated_0"))      # seated is not what latch needs
    gate.record_accept(0, "released_stable", _verdict("accept", "claim_released_stable_0"))
    assert gate.done(0) == (False, ["latched"])
    gate.record_accept("0", "latched", _verdict("accept", "claim_latched_0"))  # the item 0 and the text 0 are 1 item
    assert gate.done(0) == (True, []) and gate.cycle_done([0]) == (True, [])
    with pytest.raises(ValueError, match="defines no predicate 'aligned'"):
        gate.record_accept(0, "aligned", _verdict("accept", "claim_aligned_0"))
    with pytest.raises(ValueError, match="nothing could be done"):
        gate.cycle_done([])
    with pytest.raises(ValueError, match="nothing could be done"):
        gate.cycle_done()                                                    # no items known without a ledger
    with pytest.raises(TypeError, match="a list of the cycle's items"):
        gate.cycle_done("p1")
    with pytest.raises(ValueError, match="give the item's id"):
        gate.done(None)


def test_the_gate_on_the_cable_task_file():
    task_module = pytest.importorskip("avail.core.task")
    task = task_module.load_task("cable_loop")
    gate = DoneGate(task, "latch")
    assert gate.required == (("pick", "held"), ("carry", "at_port"), ("insert", "latched"),
                             ("release", "released_stable"))
    items = list(range(task.items.count))
    ok, unresolved = gate.cycle_done(items)
    assert not ok and len(unresolved) == 4 * len(items)
    for item in items:
        for step, predicate in gate.required:
            assert gate.cycle_done(items)[0] is False                         # done only after the very last one
            gate.record_accept(item, predicate, _verdict("accept", f"claim_{step}_{item}"))
    assert gate.cycle_done(items) == (True, [])


# ------------------------------------------------------------------ 4. the gate rebuilt from the ledger
def _write_cycle(folder, *, finish: bool):
    """A small ledger: 2 items, 1 accept, 1 person's yes through a fork, 1 takeover, and events that must not count.

    With ``finish`` the last predicate gets its accept too.
    """
    with LedgerWriter("run_gate", folder=folder) as ledger:
        ledger.append("cycle_start", {"cycle_id": "cycle_1", "items": [0, 1], "task": "toy_insert"})
        for item in (0, 1):
            for name in ("seated", "released_stable"):
                ledger.append("claim", _claim(name, item))
        ledger.append("verdict", _verdict("accept", "claim_seated_0"))                  # item 0 seated: evidence
        ledger.append("verdict", _verdict("abstain", "claim_released_stable_0", "verdict_rs0"))
        ledger.append("fork", fork_record("verdict_rs0", decide("abstain", 0, 2, True), fork_id="fork_rs0"))
        ledger.append("assist", _assist("fork_rs0", "yes", True, "assist_rs0"))          # item 0 released: a yes
        ledger.append("verdict", _verdict("reject", "claim_seated_1", "verdict_s1"))
        ledger.append("fork", fork_record("verdict_s1", decide("reject", 1, 2, True), fork_id="fork_s1"))
        ledger.append("assist", _assist("fork_s1", "no", False, "assist_s1"))            # a no settles nothing
        ledger.append("intervention", Intervention(
            intervention_id="iv_s1", run_id="run_gate", cycle_id="cycle_1", item=1, step="insert",
            claim_id="claim_seated_1", verdict_id="verdict_s1", fork_id="fork_s1", question="takeover",
            trigger="reject after 2 attempts", answer="taken over", truth_label=True, asked_ts=10.0,
            answered_ts=11.0, seconds_waiting=1.0, seconds_handling=20.0, outcome="taken_over_done"))
        # what must never count: another cycle's accept, a verdict that does not read back, a cycle's text
        ledger.append("claim", _claim("released_stable", 1, cycle_id="cycle_2", claim_id="claim_other"))
        ledger.append("verdict", _verdict("accept", "claim_other"))
        ledger.append("verdict", {"verdict_id": "v_bad", "claim_id": "claim_released_stable_1", "decision": "accept",
                                  "stage_reached": 1, "witness_version": "v1", "evidence_ids": [],
                                  "cost_total": {"seconds": 0, "usd": 0, "moved_robot": False}})
        ledger.append("tool_call", {"tool": "avail_done", "text": "decision: accept, released_stable of item 1 done"})
        if finish:
            ledger.append("verdict", _verdict("accept", "claim_released_stable_1"))


def test_from_ledger_rebuilds_what_the_live_gate_recorded(tmp_path):
    _write_cycle(tmp_path / "run", finish=False)
    gate = DoneGate(_Task(), "plain").from_ledger(iter_events(tmp_path / "run"), "cycle_1")
    assert gate.items == [0, 1]
    assert gate.done(0) == (True, [])
    assert gate.done(1) == (False, ["released_stable"])                       # the bad accept did not count
    assert gate.cycle_done() == (False, [{"item": 1, "step": "release", "predicate": "released_stable"}])
    assert any("does not read back as a Verdict" in problem for problem in gate.problems)
    # the same state as a live gate fed the same records
    live = DoneGate(_Task(), "plain")
    live.record_accept(0, "seated", _verdict("accept", "claim_seated_0"))
    live.record_signoff(0, "released_stable", _assist("fork_rs0", "yes", True, "assist_rs0"))
    iv = Intervention(intervention_id="iv_s1", run_id="run_gate", cycle_id="cycle_1", item=1, step="insert",
                      question="takeover", trigger="reject after 2 attempts", answer="taken over", truth_label=True,
                      asked_ts=10.0, answered_ts=11.0, seconds_waiting=1.0, seconds_handling=20.0,
                      outcome="taken_over_done")
    live.record_signoff(1, "seated", iv)
    for item in (0, 1):
        assert live.entries(item) == gate.entries(item), item
    assert gate.entries(1)[0]["signoff_assist_id"] == "iv_s1"

    _write_cycle(tmp_path / "done", finish=True)
    finished = DoneGate(_Task(), "plain").from_ledger(tmp_path / "done", "cycle_1")
    assert finished.cycle_done() == (True, [])
    other = DoneGate(_Task(), "plain").from_ledger(tmp_path / "done", "cycle_2")
    assert other.items == [] and other.done(1) == (False, ["seated"])          # cycle 2 has only its 1 accept


def test_from_ledger_reads_every_form_a_ledger_comes_in(tmp_path):
    _write_cycle(tmp_path / "run", finish=True)
    tables = load_ledger(tmp_path / "run")
    forms = [tmp_path / "run", iter_events(tmp_path / "run"), list(iter_events(tmp_path / "run")), tables,
             tables.events]
    states = [DoneGate(_Task(), "plain").from_ledger(form, "cycle_1").cycle_done() for form in forms]
    assert states == [(True, [])] * len(forms)
    with pytest.raises(ValueError, match="give the id of the cycle"):
        DoneGate(_Task(), "plain").from_ledger([], "")
    with pytest.raises(TypeError, match="pass the events iter_events gives"):
        DoneGate(_Task(), "plain").from_ledger([("claim", {})], "cycle_1")


def test_a_gate_needs_a_task_with_required_steps():
    with pytest.raises(TypeError, match="pass a TaskSpec"):
        DoneGate({"steps": []}, "plain")
    with pytest.raises(ValueError, match="give the cell's variant"):
        DoneGate(_Task(), "")

    class Empty(_Task):
        def required(self, variant):
            return []

    with pytest.raises(ValueError, match="the gate could never say done"):
        DoneGate(Empty(), "plain")
    assert answer_for(HumanRequest("fork_1", _claim("seated", 0), "confirm"), "yes").truth_label is True
