"""The record types of docs/01_architecture.md: round trips, every validator, the JSON Schema export
and the literal lists.

A refused input must do 2 things: raise, and explain itself in a line that names the field and says
how to fix it. ``refusal`` checks both for every test that uses it.
"""

from __future__ import annotations

import copy
import datetime
import json
import pickle
import re
import time
from pathlib import Path

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from avail import schemas
from avail.schemas import (
    FORK_OPTION_FORMS,
    PART_MODELS,
    PAYLOAD_MODELS,
    RECORD_MODELS,
    Assist,
    Audit,
    Calibration,
    Claim,
    Cost,
    CycleEnd,
    Evidence,
    Fork,
    LedgerEvent,
    Predicate,
    Record,
    SchemaError,
    Stage,
    Verdict,
    Witness,
    explain,
    is_fork_option,
    kinds,
    new_id,
    record_model,
    validate_record,
)
from avail.schemas import base as schema_base
from avail.schemas import export as schema_export

FIX = ". Fix: "


# ------------------------------------------------------------------ valid data, 1 builder per record type
def cost(**changes):
    return {"seconds": 1.25, "usd": 0.004, "moved_robot": True, **changes}


def predicate(**changes):
    return {"name": "seated", "target": {"plug": "p1", "port": 7}, **changes}


def claim(**changes):
    return {
        "claim_id": "claim_0001", "run_id": "run_a", "cycle_id": "cycle_17", "step_id": "insert_p1",
        "predicate": predicate(), "raised_by": "skill", "ts": 1794300000.5,
        "context": {"task": "plug_cycle", "stage": "insert", "attempt": 0},
        **changes,
    }


def evidence(**changes):
    return {
        "evidence_id": "ev_0001", "claim_id": "claim_0001", "source_id": "probe.push_seat", "kind": "active_probe",
        "features": {"travel_mm": 0.4, "force_at_stop_n": 5.5, "clicked": 1.0}, "score": 0.91,
        "cost": cost(usd=0.0), "artifacts": ["runs/run_a/frames/wrist_0001.jpg"], "side_effect": "repaired",
        "ts": 1794300001.75,
        **changes,
    }


def stage(**changes):
    return {
        "source_ids": ["probe.push_seat"], "scorer_id": "push_travel", "tau_reject": 0.2, "tau_accept": 0.8,
        "alpha_k": 0.01, "beta_k": 0.025,
        **changes,
    }


def calibration(**changes):
    return {
        "n_pos": 800, "n_neg": 1200, "split_id": "zoo_calibration", "far_point": 0.008, "far_ucb95": 0.0149,
        "frr_point": 0.031, "date": "2026-11-11",
        **changes,
    }


def witness(**changes):
    return {
        "witness_id": "witness_seated_fric", "version": "v3", "predicate_name": "seated",
        "scope": {"station_id": "mj_plug", "connector": "fric"},
        "stages": [
            stage(source_ids=["passive.pose_residual"], scorer_id="pose", tau_reject=0.1, tau_accept=None,
                  alpha_k=0.0, beta_k=0.025),
            stage(),
            stage(source_ids=["probe.tug"], scorer_id="fused", tau_reject=None, tau_accept=0.9, alpha_k=0.01,
                  beta_k=0.0),
        ],
        "alpha": 0.02, "beta": 0.05, "calibration": calibration(),
        **changes,
    }


def verdict(**changes):
    return {
        "verdict_id": "verdict_0001", "claim_id": "claim_0001", "decision": "accept", "score": 0.93,
        "stage_reached": 2, "witness_version": "v3", "evidence_ids": ["ev_0001", "ev_0002"],
        "cost_total": cost(usd=0.0),
        **changes,
    }


def fork(**changes):
    return {
        "fork_id": "fork_0001", "verdict_id": "verdict_0002", "options": ["repair:push_seat", "retry", "ask_human"],
        "chosen": "retry", "chosen_by": "policy",
        "shadow": {"decider": "jev-d1", "choice": "ask_human", "latency_s": 0.41, "usd": 0.0002},
        "reason": "1 attempt left",
        **changes,
    }


def assist(**changes):
    return {
        "assist_id": "assist_0001", "fork_id": "fork_0001", "question": "confirm",
        "options": ["yes", "no", "cannot tell"], "answer": "no", "truth_label": False,
        "correction": {"kind": "marks", "payload": {"points": [[212, 148], [230, 150]], "arm": "right"}},
        "seconds_to_answer": 18.5, "seconds_handling": 45.0, "assistant_id": "oracle",
        **changes,
    }


def audit(**changes):
    return {"audit_id": "audit_0001", "claim_id": "claim_0001", "truth_label": True, "auditor": "kevin",
            "seconds": 9.5, **changes}


def cycle_end(**changes):
    return {
        "declared_success": True,
        "predicates": [
            {"predicate": predicate(name="at_port"), "verdict_id": "verdict_0001"},
            {"predicate": predicate(), "verdict_id": "verdict_0003", "signoff_assist_id": "assist_0002"},
            {"predicate": {"name": "released_stable", "target": {"plug": "p1"}}, "signoff_assist_id": "assist_0003"},
        ],
        "wall_s": 41.25,
        "true_success": False,
        **changes,
    }


def ledger_event(**changes):
    return {"run_id": "run_a", "seq": 12, "ts": 1794300042.0, "kind": "cycle_end", "payload": cycle_end(), **changes}


def intervention(**changes):
    """A takeover: the person took the arm for the insert step of cable 3 and finished it."""
    return {
        "intervention_id": "iv_0001", "run_id": "run_a", "cycle_id": "cycle_17", "item": "cable_3", "step": "insert",
        "predicate_name": "seated", "claim_id": "claim_0001", "verdict_id": "verdict_0002", "fork_id": "fork_0001",
        "assist_id": "assist_0001", "question": "takeover", "trigger": "reject after 2 attempts",
        "answer": "taken over", "truth_label": True, "asked_ts": 1794300050.0, "answered_ts": 1794300071.5,
        "seconds_waiting": 21.5, "seconds_handling": 38.0, "handoff_ts": 1794300071.5, "handback_ts": 1794300109.5,
        "handoff_frame": 412, "handback_frame": 1172, "frames_path": "frames/cycle_17.parquet",
        "outcome": "taken_over_done", "features": {"passive.jaw_hold/in_band": 1.0, "probe.tug/disp_mm": 0.4},
        **changes,
    }


VALID = {
    "Cost": cost, "Predicate": predicate, "Claim": claim, "Evidence": evidence, "Witness": witness,
    "Verdict": verdict, "Fork": fork, "Assist": assist, "Audit": audit, "LedgerEvent": ledger_event,
    "Intervention": intervention,
}


# ------------------------------------------------------------------ helpers
def refusal(model, data) -> list[str]:
    """The explain lines for data ``model`` must refuse. Every line names a path and carries a fix."""
    with pytest.raises(ValidationError) as caught:
        model.model_validate(data)
    lines = explain(caught.value)
    assert len(lines) == caught.value.error_count()
    for line in lines:
        path, _, rest = line.partition(": ")
        assert path and FIX in rest, line
    return lines


def line_for(lines: list[str], path: str) -> str:
    """The 1 line about ``path``."""
    matching = [line for line in lines if line.startswith(path + ": ")]
    assert len(matching) == 1, f"expected 1 line for {path!r}, got {lines}"
    return matching[0]


def only_line(model, data, path: str) -> str:
    """The line for data whose single problem sits at ``path``."""
    lines = refusal(model, data)
    assert len(lines) == 1, lines
    return line_for(lines, path)


# ------------------------------------------------------------------ round trips
def test_every_record_type_has_a_round_trip_case():
    assert set(VALID) == set(RECORD_MODELS)
    assert list(RECORD_MODELS) == ["Cost", "Predicate", "Claim", "Evidence", "Witness", "Verdict", "Fork", "Assist",
                                   "Audit", "LedgerEvent", "Intervention"]


@pytest.mark.parametrize("name", list(VALID))
def test_round_trip_model_to_json_to_model(name):
    model = RECORD_MODELS[name]
    record = model.model_validate(VALID[name]())
    text = record.model_dump_json()
    as_json = json.loads(text)

    assert model.model_validate_json(text) == record
    assert model.model_validate(as_json) == record
    assert set(as_json) == set(model.model_fields)           # nothing dropped on the way out
    assert json.loads(json.dumps(record.model_dump())) == as_json   # the plain dump is JSON already
    assert model.model_validate_json(model.model_validate_json(text).model_dump_json()) == record


def test_round_trip_changes_nothing_a_reader_would_see():
    given = verdict()
    again = json.loads(Verdict.model_validate(given).model_dump_json())
    assert again == given

    given = fork()
    assert json.loads(Fork.model_validate(given).model_dump_json()) == given

    given = ledger_event()
    assert json.loads(LedgerEvent.model_validate(given).model_dump_json()) == given


def test_round_trip_keeps_null_thresholds_and_the_types_inside_a_target():
    record = Witness.model_validate(witness())
    text = record.model_dump_json()
    assert '"tau_accept":null' in text and '"tau_reject":null' in text
    again = Witness.model_validate_json(text)
    assert again.stages[0].tau_accept is None and again.stages[0].tau_reject == 0.1
    assert again.stages[2].tau_reject is None and again.stages[2].tau_accept == 0.9

    target = Predicate.model_validate_json(Predicate.model_validate(predicate()).model_dump_json()).target
    assert target == {"plug": "p1", "port": 7}
    assert type(target["port"]) is int and type(target["plug"]) is str


def test_round_trip_of_an_open_fork_and_a_bare_evidence_item():
    open_fork = Fork.model_validate({"fork_id": "fork_2", "verdict_id": "verdict_9", "options": ["ask_human"]})
    assert open_fork.chosen is None and open_fork.chosen_by is None and open_fork.shadow is None
    assert Fork.model_validate_json(open_fork.model_dump_json()) == open_fork

    bare = Evidence.model_validate({"evidence_id": "ev_2", "claim_id": "claim_2", "source_id": "passive.skill_flag",
                                    "kind": "passive_proprio", "cost": Cost.zero()})
    assert bare.features == {} and bare.score is None and bare.artifacts == [] and bare.side_effect == "none"
    assert Evidence.model_validate_json(bare.model_dump_json()) == bare


def test_records_survive_copy_and_pickle():
    record = Verdict.model_validate(verdict())
    assert copy.deepcopy(record) == record
    assert pickle.loads(pickle.dumps(record)) == record
    assert record.model_copy() == record and record.model_copy() is not record


# ------------------------------------------------------------------ the base: unknown fields, frozen, validated copies
@pytest.mark.parametrize("name", list(VALID))
def test_unknown_fields_are_refused_on_every_record(name):
    model = RECORD_MODELS[name]
    line = only_line(model, {**VALID[name](), "surprise": 1}, "surprise")
    assert f"{name} has no field named surprise" in line
    assert "remove it or rename it" in line


@pytest.mark.parametrize("model", [*RECORD_MODELS.values(), *PART_MODELS.values()], ids=lambda m: m.__name__)
def test_every_model_forbids_extras_refuses_nan_and_is_frozen(model):
    assert issubclass(model, Record)
    # The payload of a ledger event is a free dict, so CycleEnd is the 1 type that keeps unknown keys.
    assert model.model_config["extra"] == ("allow" if model is CycleEnd else "forbid")
    assert model.model_config["frozen"] is True
    assert model.model_config["allow_inf_nan"] is False


def test_an_unknown_field_gets_a_did_you_mean():
    line = only_line(Evidence, {**evidence(), "feature": {}}, "feature")
    assert "Did you mean features?" in line
    assert ("Evidence has: evidence_id, claim_id, source_id, kind, features, score, cost, artifacts, side_effect, ts"
            in line)

    lines = refusal(Witness, witness(stages=[{**stage(), "tau_acept": 0.5}]))
    assert "Did you mean tau_accept?" in line_for(lines, "stages.0.tau_acept")


def test_a_record_does_not_change_after_it_is_built():
    record = Cost.model_validate(cost())
    with pytest.raises(ValidationError) as caught:
        record.seconds = 99.0
    line = explain(caught.value)[0]
    assert line.startswith("seconds: a record does not change after it is built")
    assert "model_copy(update=" in line
    assert record.seconds == 1.25


def test_model_copy_with_an_update_is_validated():
    record = Cost.model_validate(cost())
    changed = record.model_copy(update={"seconds": 2.0})
    assert changed.seconds == 2.0 and changed.usd == record.usd and record.seconds == 1.25

    with pytest.raises(ValidationError) as caught:
        record.model_copy(update={"seconds": -1.0})
    assert explain(caught.value)[0].startswith("seconds: -1.0 is below 0")
    with pytest.raises(ValidationError) as caught:
        record.model_copy(update={"second": 2.0})
    assert "Cost has no field named second" in explain(caught.value)[0]
    with pytest.raises(ValidationError):
        record.copy(update={"seconds": -1.0})                # the old pydantic name validates too
    assert record.copy(update={"seconds": 3.0}).seconds == 3.0 and record.copy() == record
    with pytest.raises(TypeError, match="does not take exclude"):
        record.copy(exclude={"usd"})


def test_a_deep_copy_with_an_update_shares_nothing():
    record = Evidence.model_validate(evidence())
    changed = record.model_copy(update={"score": 0.5}, deep=True)
    assert changed.score == 0.5 and changed.features == record.features
    assert changed.features is not record.features and changed.cost is not record.cost


@pytest.mark.parametrize("name", list(VALID))
def test_model_construct_is_refused_on_every_record(name):
    model = RECORD_MODELS[name]
    with pytest.raises(TypeError, match="skips validation") as caught:
        model.model_construct(**VALID[name]())
    assert f"use {name}(...) or {name}.model_validate(data)" in str(caught.value)


# ------------------------------------------------------------------ Cost
def test_cost_adds_seconds_and_dollars_and_ors_motion():
    a = Cost(seconds=1.5, usd=0.25, moved_robot=False)
    b = Cost(seconds=2.0, usd=0.5, moved_robot=True)
    total = a + b
    assert (total.seconds, total.usd, total.moved_robot) == (3.5, 0.75, True)
    assert (a + a).moved_robot is False and (b + b).moved_robot is True and (b + a) == total
    assert a == Cost(seconds=1.5, usd=0.25, moved_robot=False)        # the operands are untouched


def test_cost_zero_is_the_identity_and_sum_works():
    zero = Cost.zero()
    assert (zero.seconds, zero.usd, zero.moved_robot) == (0.0, 0.0, False)
    a = Cost(seconds=1.5, usd=0.25, moved_robot=True)
    assert a + zero == a and zero + a == a
    parts = [a, Cost(seconds=0.5, usd=0.0, moved_robot=False), Cost(seconds=2.0, usd=0.75, moved_robot=False)]
    assert sum(parts, Cost.zero()) == Cost(seconds=4.0, usd=1.0, moved_robot=True)
    assert sum(parts) == sum(parts, Cost.zero())
    assert sum([], Cost.zero()) == zero


@pytest.mark.parametrize("other", [1, 1.5, "1", None, True, {"seconds": 1, "usd": 0, "moved_robot": False}])
def test_cost_only_adds_to_a_cost(other):
    with pytest.raises(TypeError):
        Cost.zero() + other
    with pytest.raises(TypeError):
        other + Cost.zero()


@pytest.mark.parametrize("field", ["seconds", "usd"])
def test_cost_refuses_a_negative_amount(field):
    line = only_line(Cost, cost(**{field: -0.5}), field)
    assert "-0.5 is below 0" in line and "give a value of 0 or more" in line


@pytest.mark.parametrize("field", ["seconds", "usd"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_cost_refuses_nan_and_infinity(field, value):
    line = only_line(Cost, cost(**{field: value}), field)
    assert "is not a finite number" in line and "give a finite number" in line


@pytest.mark.parametrize("value", ["1.5", "", None, True, [1.0]])
def test_cost_refuses_an_amount_that_is_not_a_number(value):
    line = only_line(Cost, cost(seconds=value), "seconds")
    assert "is not a number" in line and "give a number" in line


@pytest.mark.parametrize("value", ["yes", "true", "False", 1, 0, 1.0, None])
def test_cost_moved_robot_is_a_real_boolean(value):
    line = only_line(Cost, cost(moved_robot=value), "moved_robot")
    assert "is not true or false" in line and "give a real boolean" in line


@pytest.mark.parametrize("field", ["seconds", "usd", "moved_robot"])
def test_cost_states_all_3_fields(field):
    data = cost()
    del data[field]
    line = only_line(Cost, data, field)
    assert "this required field is missing" in line and f"add {field}" in line


def test_numpy_scalars_are_read_as_plain_numbers():
    record = Cost(seconds=np.float32(1.5), usd=np.int64(2), moved_robot=np.bool_(True))
    assert (record.seconds, record.usd, record.moved_robot) == (1.5, 2.0, True)
    assert type(record.seconds) is float and type(record.usd) is float and type(record.moved_robot) is bool


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nan_is_refused_by_the_number_type_and_by_the_base_class_each_alone(value):
    class NumberOutsideARecord(BaseModel):      # plain pydantic config: the Number type must refuse by itself
        x: schema_base.Number

    class PlainFloatInARecord(Record):          # a plain float: the Record config must refuse by itself
        x: float

    for model in (NumberOutsideARecord, PlainFloatInARecord):
        assert model(x=1.5).x == 1.5
        with pytest.raises(ValidationError) as caught:
            model(x=value)
        line = explain(caught.value)[0]
        assert line.startswith("x: ") and "is not a finite number. Fix: give a finite number" in line


# ------------------------------------------------------------------ new_id
def test_new_id_is_the_prefix_an_underscore_and_12_hex_characters():
    made = new_id("claim")
    assert re.fullmatch(r"claim_[0-9a-f]{12}", made)
    assert re.fullmatch(r"ev_[0-9a-f]{12}", new_id("ev"))
    assert len({new_id("claim") for _ in range(2000)}) == 2000
    assert Claim.model_validate(claim(claim_id=made)).claim_id == made


@pytest.mark.parametrize("prefix", ["", " ", "has space", "9lives", "_x", None, 7, "claim\n"])
def test_new_id_refuses_a_prefix_it_cannot_use(prefix):
    with pytest.raises(ValueError, match="Fix: give a short word"):
        new_id(prefix)


# ------------------------------------------------------------------ Predicate
@pytest.mark.parametrize("name", ["seated", "latched", "at_port", "held", "released_stable", "removed"])
@pytest.mark.parametrize("empty", ["missing", "", "   ", None])
def test_a_known_predicate_gets_its_definition_filled_in(name, empty):
    data = {"name": name, "target": {"plug": "p1"}}
    if empty != "missing":
        data["definition"] = empty
    record = Predicate.model_validate(data)
    assert record.definition == kinds.PREDICATE_DEFINITIONS[name]
    assert len(record.definition) > 20


def test_a_definition_the_caller_gives_is_kept():
    own = "The plug is fully home and the green light is on."
    assert Predicate.model_validate(predicate(definition=own)).definition == own


def test_an_unknown_predicate_must_bring_its_own_definition():
    line = only_line(Predicate, {"name": "aligned", "target": {"plug": "p1"}}, "definition")
    assert "aligned is not a known predicate, so it needs its own definition" in line
    assert "add a definition a human auditor can apply" in line
    assert "seated, latched, at_port, held, released_stable, removed" in line
    for blank in ("", "  ", None):
        only_line(Predicate, {"name": "aligned", "target": {}, "definition": blank}, "definition")

    record = Predicate.model_validate(
        {"name": "aligned", "target": {"plug": "p1"}, "definition": "The plug axis is within 2 deg of the port axis."})
    assert record.definition == "The plug axis is within 2 deg of the port axis."


@pytest.mark.parametrize("name", ["Seated", "SEATED", "seated!", "9lives", "", "at port", "at-port", "_seated",
                                  "seated\n", "séated", None, 7])
def test_a_predicate_name_is_a_lowercase_identifier(name):
    lines = refusal(Predicate, predicate(name=name))
    line = line_for(lines, "name")
    assert "is not a lowercase identifier" in line
    assert "use lowercase letters, digits and underscores, starting with a letter" in line
    assert len(lines) == 1          # a bad name does not also complain about the definition


@pytest.mark.parametrize("value", [1.5, 7.0, True, False, None, [1], {"a": 1}, "", "  "])
def test_a_target_value_is_text_or_a_whole_number(value):
    line = only_line(Predicate, predicate(target={"plug": "p1", "port": value}), "target.port")
    assert "is not text or a whole number" in line
    assert "use a non-empty string or an integer, for example {'plug': 'p1', 'port': 7}" in line


def test_a_target_takes_numpy_whole_numbers_and_needs_text_keys():
    assert Predicate.model_validate(predicate(target={"port": np.int64(7)})).target == {"port": 7}
    assert Predicate.model_validate(predicate(target={})).target == {}
    line = only_line(Predicate, predicate(target={"": "p1"}), 'target.""')
    assert "is not a non-empty string" in line
    line = only_line(Predicate, predicate(target={7: "p1"}), "target.7")
    assert "the key 7 is not a non-empty string" in line and "name each part of the target with text" in line


def test_a_predicate_needs_a_target_dict():
    line = only_line(Predicate, {"name": "seated"}, "target")
    assert "this required field is missing" in line and "add target" in line
    line = only_line(Predicate, predicate(target="p1"), "target")
    assert '"p1" is not a dict' in line and "give a dict" in line


# ------------------------------------------------------------------ Claim
def test_claim_raised_by_is_one_of_the_4():
    for who in ("skill", "policy", "agent", "human"):
        assert Claim.model_validate(claim(raised_by=who)).raised_by == who
    line = only_line(Claim, claim(raised_by="robot"), "raised_by")
    assert '"robot" is not an allowed value' in line and "use 'skill', 'policy', 'agent' or 'human'" in line


@pytest.mark.parametrize("field", ["claim_id", "run_id", "cycle_id", "step_id"])
@pytest.mark.parametrize("value, what", [("", "this is empty"), ("   ", "this is empty"), (17, "17 is not text"),
                                         (None, "null is not text")])
def test_claim_ids_are_non_empty_strings(field, value, what):
    line = only_line(Claim, claim(**{field: value}), field)
    assert what in line and "give a non-empty string" in line


def test_claim_names_every_missing_field():
    lines = refusal(Claim, {})
    assert [line.partition(":")[0] for line in lines] == ["claim_id", "run_id", "cycle_id", "step_id", "predicate",
                                                           "raised_by", "context"]
    assert all("this required field is missing" in line for line in lines)
    assert "new_id('claim') makes a fresh one" in line_for(lines, "claim_id")


def test_claim_context_carries_task_stage_and_attempt():
    record = Claim.model_validate(claim())
    assert (record.context.task, record.context.stage, record.context.attempt) == ("plug_cycle", "insert", 0)
    for field in ("task", "stage", "attempt"):
        context = dict(claim()["context"])
        del context[field]
        only_line(Claim, claim(context=context), f"context.{field}")
    line = only_line(Claim, claim(context={"task": "t", "stage": "s", "attempt": -1}), "context.attempt")
    assert "-1 is below 0" in line
    for value in (1.0, "1", True, None):
        line = only_line(Claim, claim(context={"task": "t", "stage": "s", "attempt": value}), "context.attempt")
        assert "is not a whole number" in line and "give an integer" in line


def test_claim_predicate_problems_carry_the_full_path():
    lines = refusal(Claim, claim(predicate={"name": "Seated", "target": {"port": 1.5}}))
    assert "is not a lowercase identifier" in line_for(lines, "predicate.name")
    assert "is not text or a whole number" in line_for(lines, "predicate.target.port")


def test_ts_defaults_to_now_and_must_be_a_finite_number():
    data = claim()
    del data["ts"]
    before = time.time()
    record = Claim.model_validate(data)
    after = time.time()
    assert type(record.ts) is float and before <= record.ts <= after

    for value, what in (("now", "is not a number"), (None, "is not a number"), (float("nan"), "is not a finite number"),
                        (True, "is not a number")):
        line = only_line(Claim, claim(ts=value), "ts")
        assert what in line and "Unix seconds" in line
    assert Claim.model_validate(claim(ts=1794300000)).ts == 1794300000.0


# ------------------------------------------------------------------ Evidence
def test_features_turn_booleans_and_whole_numbers_into_floats():
    record = Evidence.model_validate(evidence(features={"clicked": True, "held": False, "count": 3, "travel_mm": 0.4}))
    assert record.features == {"clicked": 1.0, "held": 0.0, "count": 3.0, "travel_mm": 0.4}
    assert all(type(value) is float for value in record.features.values())


def test_features_take_numpy_scalars():
    record = Evidence.model_validate(evidence(features={
        "a": np.float32(1.5), "b": np.float64(2.5), "c": np.int64(3), "d": np.bool_(True), "e": np.array(4.5)}))
    assert record.features == {"a": 1.5, "b": 2.5, "c": 3.0, "d": 1.0, "e": 4.5}
    assert all(type(value) is float for value in record.features.values())


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), np.float32("nan"), np.float64("inf"),
                                   10**400])
def test_features_refuse_nan_and_infinity_and_say_to_leave_the_key_out(value):
    line = only_line(Evidence, evidence(features={"travel_mm": 0.4, "gap_mm": value}), "features.gap_mm")
    assert "is not a finite number" in line
    assert "leave this key out of features when the source has no reading" in line


def test_features_refuse_null_and_say_to_leave_the_key_out():
    line = only_line(Evidence, evidence(features={"gap_mm": None}), "features.gap_mm")
    assert "null is not a number" in line
    assert "leave this key out of features when the source has no reading" in line


@pytest.mark.parametrize("value", ["0.4", "", "yes"])
def test_features_refuse_text(value):
    line = only_line(Evidence, evidence(features={"gap_mm": value}), "features.gap_mm")
    assert "is not a number" in line and "Text is refused" in line


@pytest.mark.parametrize("value", [[1.0, 2.0], (1.0, 2.0), {"x": 1.0}, np.zeros(3), {1.0}])
def test_features_are_flat(value):
    line = only_line(Evidence, evidence(features={"force_n": value}), "features.force_n")
    assert "is not a single number" in line
    assert "features is flat, so give each number its own key" in line


def test_features_report_every_bad_key_at_once_and_need_text_keys():
    lines = refusal(Evidence, evidence(features={"a": float("nan"), "b": 1.0, "c": "x", "d": None}))
    assert [line.partition(":")[0] for line in lines] == ["features.a", "features.c", "features.d"]
    line = only_line(Evidence, evidence(features={7: 1.0}), "features.7")
    assert "the key 7 is not a non-empty string" in line
    for blank, path in (("", 'features.""'), ("  ", "features.  ")):
        line = only_line(Evidence, evidence(features={blank: 1.0}), path)
        assert "is not a non-empty string" in line and "name each feature with text, for example travel_mm" in line
    line = only_line(Evidence, evidence(features=[1.0]), "features")
    assert "is not a dict" in line


def test_evidence_kind_and_side_effect_are_from_their_lists():
    for kind in ("passive_proprio", "passive_vision", "active_probe", "judge", "instrumented"):
        assert Evidence.model_validate(evidence(kind=kind)).kind == kind
    line = only_line(Evidence, evidence(kind="probe"), "kind")
    assert "use 'passive_proprio', 'passive_vision', 'active_probe', 'judge' or 'instrumented'" in line

    for effect in ("none", "repaired", "disturbed"):
        assert Evidence.model_validate(evidence(side_effect=effect)).side_effect == effect
    line = only_line(Evidence, evidence(side_effect="broken"), "side_effect")
    assert "use 'none', 'repaired' or 'disturbed'" in line


def test_evidence_score_is_a_finite_number_or_null():
    assert Evidence.model_validate(evidence(score=None)).score is None
    assert Evidence.model_validate(evidence(score=1)).score == 1.0
    for value, what in ((float("nan"), "is not a finite number"), ("0.9", "is not a number"), (True, "is not a number")):
        line = only_line(Evidence, evidence(score=value), "score")
        assert what in line
        assert "higher means the predicate is more likely true, or null when the source gives none" in line


def test_evidence_states_its_cost():
    data = evidence()
    del data["cost"]
    assert "add cost" in only_line(Evidence, data, "cost")
    line = only_line(Evidence, evidence(cost=cost(seconds=-1)), "cost.seconds")
    assert "-1.0 is below 0" in line
    lines = refusal(Evidence, evidence(cost={"seconds": 1.0}))
    assert [line.partition(":")[0] for line in lines] == ["cost.usd", "cost.moved_robot"]
    assert Evidence.model_validate(evidence(cost=Cost.zero())).cost == Cost.zero()


def test_artifacts_are_paths_as_text():
    record = Evidence.model_validate(evidence(artifacts=[Path("frames") / "wrist_1.jpg", "frames/top_1.jpg"]))
    assert record.artifacts == [str(Path("frames") / "wrist_1.jpg"), "frames/top_1.jpg"]
    assert all(type(item) is str for item in record.artifacts)
    for value in (None, "", 7, b"frames/x.jpg"):
        line = only_line(Evidence, evidence(artifacts=["ok.jpg", value]), "artifacts.1")
        assert "is not a file path" in line and "give the path as text" in line


# ------------------------------------------------------------------ Witness
def test_witness_refuses_stage_shares_that_overspend_alpha():
    line = only_line(Witness, witness(alpha=0.015), "alpha")
    assert "the stages spend 0.02 of false-accept budget (the sum of alpha_k), more than alpha = 0.015" in line
    assert "lower the alpha_k values until they add up to at most alpha, or raise alpha" in line


def test_witness_refuses_stage_shares_that_overspend_beta():
    line = only_line(Witness, witness(beta=0.04), "beta")
    assert "the stages spend 0.05 of false-reject budget (the sum of beta_k), more than beta = 0.04" in line
    assert "lower the beta_k values until they add up to at most beta, or raise beta" in line


def test_witness_reports_both_budgets_at_once():
    lines = refusal(Witness, witness(alpha=0.01, beta=0.01))
    assert [line.partition(":")[0] for line in lines] == ["alpha", "beta"]


def test_witness_budget_check_has_a_tolerance_of_1e_9():
    def with_shares(shares, budget):
        return witness(stages=[stage(alpha_k=share, beta_k=0.0) for share in shares], alpha=budget, beta=0.0)

    assert sum([0.1, 0.1, 0.1]) > 0.3                                    # float noise the check must forgive
    Witness.model_validate(with_shares([0.1, 0.1, 0.1], 0.3))
    Witness.model_validate(with_shares([0.01, 0.01], 0.02))              # spending the whole budget is fine
    Witness.model_validate(with_shares([0.01, 0.01 + 5e-10], 0.02))      # inside the tolerance
    Witness.model_validate(with_shares([0.005], 0.02))                   # under the budget
    only_line(Witness, with_shares([0.01, 0.01 + 2e-9], 0.02), "alpha")  # outside it
    only_line(Witness, with_shares([0.02, 1e-6], 0.02), "alpha")

    def with_beta(shares, budget):
        return witness(stages=[stage(alpha_k=0.0, beta_k=share) for share in shares], alpha=0.0, beta=budget)

    Witness.model_validate(with_beta([0.025, 0.025 + 5e-10], 0.05))
    only_line(Witness, with_beta([0.025, 0.025 + 2e-9], 0.05), "beta")


def test_a_null_threshold_means_never_and_must_be_written_out():
    never_accepts = Stage.model_validate(stage(tau_accept=None))
    never_rejects = Stage.model_validate(stage(tau_reject=None))
    assert never_accepts.tau_accept is None and never_rejects.tau_reject is None
    for field, word in (("tau_accept", "never accepts"), ("tau_reject", "never rejects")):
        data = stage()
        del data[field]
        line = only_line(Stage, data, field)
        assert "this required field is missing" in line and f"null means this stage {word}" in line
        for value in (float("inf"), float("-inf"), float("nan")):
            line = only_line(Stage, stage(**{field: value}), field)
            assert "is not a finite number" in line and f"null means this stage {word}" in line
        for value in ("0.5", True, [0.5]):
            line = only_line(Stage, stage(**{field: value}), field)
            assert "is not a number" in line and f"null means this stage {word}" in line
        assert getattr(Stage.model_validate(stage(**{field: 1})), field) == 1.0


def test_thresholds_may_cross():
    # docs/04 section 2: if the 2 thresholds cross, the region between them abstains. The schema allows it.
    crossed = Stage.model_validate(stage(tau_reject=0.9, tau_accept=0.3))
    assert crossed.tau_reject > crossed.tau_accept


@pytest.mark.parametrize("field", ["alpha_k", "beta_k"])
def test_stage_shares_are_between_0_and_1(field):
    assert "is below 0" in only_line(Stage, stage(**{field: -0.01}), field)
    line = only_line(Stage, stage(**{field: 1.5}), field)
    assert "1.5 is above 1" in line and "give a value of 1 or less" in line
    assert "is not a number" in only_line(Stage, stage(**{field: "0.01"}), field)


@pytest.mark.parametrize("field", ["alpha", "beta"])
def test_witness_budgets_are_between_0_and_1(field):
    assert "is above 1" in line_for(refusal(Witness, witness(**{field: 1.01})), field)
    assert "is below 0" in line_for(refusal(Witness, witness(**{field: -0.01})), field)


def test_witness_version_is_text_and_a_whole_number_becomes_text():
    assert Witness.model_validate(witness(version=3)).version == "3"
    assert Witness.model_validate(witness(version="2026-11-11.a")).version == "2026-11-11.a"
    for value in (None, "", "  ", True, 1.5):
        line = only_line(Witness, witness(version=value), "version")
        assert "is not a version label" in line and "for example 'v3'" in line


def test_witness_names_its_predicate_scope_and_stages():
    line = only_line(Witness, witness(predicate_name="Seated"), "predicate_name")
    assert "is not a lowercase identifier" in line
    assert Witness.model_validate(witness(predicate_name="aligned")).predicate_name == "aligned"   # need not be known
    assert "this is empty" in only_line(Witness, witness(scope={"station_id": "", "connector": "fric"}),
                                        "scope.station_id")
    assert "add connector" in only_line(Witness, witness(scope={"station_id": "mj_plug"}), "scope.connector")
    assert Witness.model_validate(witness(stages=[], alpha=0.02)).stages == []      # a witness that always abstains
    assert "is not a list" in only_line(Witness, witness(stages="push"), "stages")
    line = only_line(Witness, witness(stages=[stage(), stage(scorer_id="")]), "stages.1.scorer_id")
    assert "this is empty" in line


def test_calibration_counts_rates_and_date():
    record = Calibration.model_validate(calibration(far_point=None, far_ucb95=None, frr_point=None))
    assert record.far_point is None and record.far_ucb95 is None and record.frr_point is None
    for field in ("n_pos", "n_neg"):
        assert "is below 0" in only_line(Calibration, calibration(**{field: -1}), field)
        assert "is not a whole number" in only_line(Calibration, calibration(**{field: 10.0}), field)
    for field in ("far_point", "far_ucb95", "frr_point"):
        assert "is above 1" in only_line(Calibration, calibration(**{field: 1.2}), field)
        assert "is not a finite number" in only_line(Calibration, calibration(**{field: float("nan")}), field)
    assert "add split_id" in only_line(Calibration, {k: v for k, v in calibration().items() if k != "split_id"},
                                       "split_id")


def test_calibration_date_is_iso_text():
    assert Calibration.model_validate(calibration(date=datetime.date(2026, 11, 11))).date == "2026-11-11"
    assert Calibration.model_validate(calibration(date="2026-11-11T09:30:00")).date == "2026-11-11T09:30:00"
    stamp = datetime.datetime(2026, 11, 11, 9, 30)
    assert Calibration.model_validate(calibration(date=stamp)).date == "2026-11-11T09:30:00"
    for value in ("11/11/2026", "yesterday", "", None, 20261111):
        line = only_line(Calibration, calibration(date=value), "date")
        assert "is not an ISO date" in line and "give a date such as 2026-11-11" in line


# ------------------------------------------------------------------ Verdict
def test_an_accept_with_no_evidence_ids_is_refused():
    line = only_line(Verdict, verdict(evidence_ids=[]), "evidence_ids")
    assert "an accept needs evidence" in line
    assert "list the ids of the Evidence records the witness used, or use the decision abstain" in line
    assert "Only evidence or a person marks a step done" in line

    omitted = verdict()
    del omitted["evidence_ids"]
    assert "an accept needs evidence" in only_line(Verdict, omitted, "evidence_ids")


def test_an_accept_with_evidence_ids_and_any_other_decision_without_them_are_fine():
    assert Verdict.model_validate(verdict()).decision == "accept"
    assert Verdict.model_validate(verdict(evidence_ids=["ev_0001"])).evidence_ids == ["ev_0001"]
    for decision in ("reject", "abstain"):
        record = Verdict.model_validate(verdict(decision=decision, evidence_ids=[]))
        assert record.decision == decision and record.evidence_ids == []


@pytest.mark.parametrize("value", ["ACCEPT", "Accept", "accept ", "accepted", "yes", "done", "pass", "", None, True, 1])
def test_verdict_decision_is_exactly_1_of_the_3_words(value):
    line = only_line(Verdict, verdict(decision=value), "decision")
    assert "is not an allowed value" in line and "use 'accept', 'reject' or 'abstain'" in line


def test_verdict_evidence_ids_are_real_ids():
    lines = refusal(Verdict, verdict(evidence_ids=["ev_0001", "", "  ", 7]))
    assert [line.partition(":")[0] for line in lines] == ["evidence_ids.1", "evidence_ids.2", "evidence_ids.3"]
    assert "An accept must list at least 1" in lines[0]
    assert "is not a list" in only_line(Verdict, verdict(evidence_ids="ev_0001"), "evidence_ids")


def test_verdict_stage_score_version_and_cost():
    assert "is below 0" in only_line(Verdict, verdict(stage_reached=-1), "stage_reached")
    assert "is not a whole number" in only_line(Verdict, verdict(stage_reached=1.5), "stage_reached")
    assert Verdict.model_validate(verdict(decision="abstain", stage_reached=0, score=None, evidence_ids=[])).score is None
    assert "is not a finite number" in only_line(Verdict, verdict(score=float("inf")), "score")
    assert Verdict.model_validate(verdict(witness_version=3)).witness_version == "3"
    assert "is not a version label" in only_line(Verdict, verdict(witness_version=""), "witness_version")
    data = verdict()
    del data["cost_total"]
    assert "add cost_total" in only_line(Verdict, data, "cost_total")
    assert "this is empty" in only_line(Verdict, verdict(claim_id=""), "claim_id")


# ------------------------------------------------------------------ Fork
@pytest.mark.parametrize("option", ["retry", "ask_human", "abort", "repair:push_seat", "repair:tug", "recover:reseat",
                                    "recover:ReSeat-2.0", "recover:skills.reseat_plug"])
def test_fork_options_take_the_5_forms(option):
    record = Fork.model_validate(fork(options=[option], chosen=option))
    assert record.options == [option] and record.chosen == option
    assert is_fork_option(option)


@pytest.mark.parametrize("option", [
    "skip", "Retry", " retry", "retry ", "repair", "repair:", "recover:", "repair:push seat", "fix:push_seat",
    "ask-human", "retry:now", "repair:_x", "recover:réseat", "", 3, None,
])
def test_fork_options_refuse_anything_else(option):
    line = only_line(Fork, fork(options=["retry", option], chosen="retry"), "options.1")
    assert "is not a fork option" in line or "is not text" in line
    assert "use one of retry, ask_human, abort, repair:<probe>, recover:<skill>" in line
    assert not is_fork_option(option)


def test_a_near_miss_fork_option_gets_a_hint():
    assert "Did you mean retry?" in only_line(Fork, fork(options=["Retry"], chosen=None, chosen_by=None), "options.0")
    assert "Did you mean ask_human?" in only_line(Fork, fork(options=["ask-human"], chosen=None), "options.0")
    assert "Did you mean repair:push_seat?" in only_line(Fork, fork(options=[" repair:push_seat"], chosen=None),
                                                        "options.0")
    assert "Did you mean repair:push_seat?" in only_line(Fork, fork(options=["Repair:Push_Seat"], chosen=None),
                                                        "options.0")
    assert "Did you mean" not in only_line(Fork, fork(options=["skip"], chosen=None), "options.0")
    line = only_line(Fork, fork(options=["repair"], chosen=None), "options.0")
    assert "repair needs a name after the colon, for example repair:<probe>" in line
    line = only_line(Fork, fork(options=["recover:"], chosen=None), "options.0")
    assert "recover needs a name after the colon, for example recover:<skill>" in line


def test_fork_options_are_a_non_empty_list_with_no_repeats():
    line = only_line(Fork, fork(options=[], chosen=None, chosen_by=None), "options")
    assert "a fork with no options leaves nothing to choose" in line
    assert "give at least 1 of retry, ask_human, abort, repair:<probe>, recover:<skill>" in line

    line = only_line(Fork, fork(options=["retry", "ask_human", "retry"]), "options")
    assert "retry is listed more than once" in line and "list each option once" in line

    data = fork(chosen=None, chosen_by=None)
    del data["options"]
    assert "add options" in only_line(Fork, data, "options")
    assert "is not a list" in only_line(Fork, fork(options="retry"), "options")


def test_fork_chosen_must_be_one_of_its_options():
    line = only_line(Fork, fork(chosen="abort"), "chosen")
    assert '"abort" is not one of this fork\'s options (repair:push_seat, retry, ask_human)' in line
    assert "pick one of the options, or leave chosen as null while the fork is open" in line
    assert Fork.model_validate(fork(chosen="repair:push_seat")).chosen == "repair:push_seat"


def test_a_fork_with_a_choice_says_who_chose():
    line = only_line(Fork, fork(chosen_by=None), "chosen_by")
    assert "this fork has a choice and does not say who made it" in line
    assert "set chosen_by to one of policy, agent, human" in line
    for who in ("policy", "agent", "human"):
        assert Fork.model_validate(fork(chosen_by=who)).chosen_by == who
    line = only_line(Fork, fork(chosen_by="model"), "chosen_by")
    assert "use 'policy', 'agent' or 'human'" in line


def test_an_open_fork_has_no_choice_yet_and_closes_through_a_validated_copy():
    open_fork = Fork.model_validate(fork(chosen=None, chosen_by=None, shadow=None, reason=""))
    assert open_fork.chosen is None
    waiting = Fork.model_validate(fork(chosen=None, chosen_by="human"))       # a fork waiting on a person
    assert waiting.chosen is None and waiting.chosen_by == "human"

    closed = open_fork.model_copy(update={"chosen": "ask_human", "chosen_by": "policy", "reason": "abstain"})
    assert (closed.chosen, closed.chosen_by, closed.reason) == ("ask_human", "policy", "abstain")
    assert open_fork.chosen is None

    with pytest.raises(ValidationError) as caught:
        open_fork.model_copy(update={"chosen": "abort", "chosen_by": "agent"})
    assert "is not one of this fork's options" in explain(caught.value)[0]
    with pytest.raises(ValidationError) as caught:
        open_fork.model_copy(update={"chosen": "retry"})
    assert explain(caught.value)[0].startswith("chosen_by: ")


def test_fork_shadow_is_logged_with_latency_and_cost():
    record = Fork.model_validate(fork())
    assert (record.shadow.decider, record.shadow.choice, record.shadow.latency_s, record.shadow.usd) == (
        "jev-d1", "ask_human", 0.41, 0.0002)
    silent = Fork.model_validate(fork(shadow={"decider": "jev-d1", "choice": None, "latency_s": 2.0, "usd": 0.0}))
    assert silent.shadow.choice is None
    # the shadow's pick is logged as it came, even when this fork did not offer it
    off_menu = Fork.model_validate(fork(shadow={"decider": "jev-d1", "choice": "abort", "latency_s": 0.4, "usd": 0.0}))
    assert off_menu.shadow.choice == "abort" and "abort" not in off_menu.options

    for field in ("decider", "choice", "latency_s", "usd"):
        shadow = dict(fork()["shadow"])
        del shadow[field]
        assert f"add {field}" in only_line(Fork, fork(shadow=shadow), f"shadow.{field}")
    bad = {"decider": "", "choice": "wait", "latency_s": -1, "usd": float("nan")}
    lines = refusal(Fork, fork(shadow=bad))
    assert "this is empty" in line_for(lines, "shadow.decider")
    assert "is not a fork option" in line_for(lines, "shadow.choice")
    assert "is below 0" in line_for(lines, "shadow.latency_s")
    assert "is not a finite number" in line_for(lines, "shadow.usd")


@pytest.mark.parametrize("text", ["accept", "ACCEPT", "Accept", " accept ", "accepted", "repair:accept",
                                  "recover:Accept", "recover:auto_accept_all", "accept:now", "ａｃｃｅｐｔ"])
def test_fork_refuses_accept_in_options_chosen_and_shadow_choice(text):
    fix = "use one of retry, ask_human, abort, repair:<probe>, recover:<skill>"
    line = only_line(Fork, fork(options=["retry", text], chosen="retry"), "options.1")
    assert "asks a fork to accept, and accept is never a fork option" in line and fix in line

    line = only_line(Fork, fork(chosen=text), "chosen")
    assert "asks a fork to accept, and accept is never a fork option" in line and fix in line

    shadow = {**fork()["shadow"], "choice": text}
    line = only_line(Fork, fork(shadow=shadow), "shadow.choice")
    assert "asks a fork to accept, and accept is never a fork option" in line and fix in line
    assert not is_fork_option(text)


# ------------------------------------------------------------------ Assist and Audit
def test_assist_question_is_one_of_the_5_types():
    for question in ("confirm", "choose", "guide_marks", "guide_text", "takeover"):
        assert Assist.model_validate(assist(question=question)).question == question
    line = only_line(Assist, assist(question="ask"), "question")
    assert "use 'confirm', 'choose', 'guide_marks', 'guide_text' or 'takeover'" in line


@pytest.mark.parametrize("value", ["yes", "no", "true", 1, 0, 1.0, "cannot tell"])
def test_assist_truth_label_is_a_real_boolean_or_null(value):
    line = only_line(Assist, assist(truth_label=value), "truth_label")
    assert "is not true or false" in line and "give a real boolean" in line


def test_assist_truth_label_null_correction_optional_and_times_stated():
    record = Assist.model_validate(assist(truth_label=None, correction=None, answer="cannot tell"))
    assert record.truth_label is None and record.correction is None
    data = assist()
    del data["truth_label"], data["correction"], data["options"]
    record = Assist.model_validate(data)
    assert record.truth_label is None and record.correction is None and record.options == []
    assert Assist.model_validate(assist(truth_label=np.bool_(True))).truth_label is True

    for field in ("seconds_to_answer", "seconds_handling"):
        assert "is below 0" in only_line(Assist, assist(**{field: -1}), field)
        data = assist()
        del data[field]
        assert f"add {field}" in only_line(Assist, data, field)
    for field in ("assist_id", "fork_id", "assistant_id"):
        assert "this is empty" in only_line(Assist, assist(**{field: ""}), field)
    data = assist()
    del data["answer"]
    assert "add answer" in only_line(Assist, data, "answer")
    assert "is not text" in only_line(Assist, assist(answer=True), "answer")


def test_assist_correction_has_a_kind_from_the_list_and_a_payload():
    for kind in ("marks", "step_text", "pose_offset", "takeover"):
        record = Assist.model_validate(assist(correction={"kind": kind, "payload": {}}))
        assert record.correction.kind == kind
    line = only_line(Assist, assist(correction={"kind": "voice", "payload": {}}), "correction.kind")
    assert "use 'marks', 'step_text', 'pose_offset' or 'takeover'" in line
    assert "add payload" in only_line(Assist, assist(correction={"kind": "marks"}), "correction.payload")
    assert "is not a dict" in only_line(Assist, assist(correction={"kind": "marks", "payload": [1, 2]}),
                                        "correction.payload")


@pytest.mark.parametrize("value", [None, "yes", "no", "true", 1, 0])
def test_audit_truth_label_is_a_real_boolean(value):
    line = only_line(Audit, audit(truth_label=value), "truth_label")
    assert "is not true or false" in line and "give a real boolean" in line


def test_audit_fields():
    for label in (True, False):
        assert Audit.model_validate(audit(truth_label=label)).truth_label is label
    data = audit()
    del data["truth_label"]
    assert "add truth_label" in only_line(Audit, data, "truth_label")
    assert "is below 0" in only_line(Audit, audit(seconds=-0.1), "seconds")
    assert "is not a number" in only_line(Audit, audit(seconds="9"), "seconds")
    for field in ("audit_id", "claim_id", "auditor"):
        assert "this is empty" in only_line(Audit, audit(**{field: " "}), field)


# ------------------------------------------------------------------ LedgerEvent
def test_ledger_event_seq_kind_and_payload():
    assert LedgerEvent.model_validate(ledger_event(seq=0)).seq == 0
    assert "-1 is below 0" in only_line(LedgerEvent, ledger_event(seq=-1), "seq")
    for value in (1.0, "1", True, None):
        assert "is not a whole number" in only_line(LedgerEvent, ledger_event(seq=value), "seq")
    line = only_line(LedgerEvent, ledger_event(kind="note", payload={}), "kind")
    assert ("use 'claim', 'evidence', 'verdict', 'fork', 'assist', 'audit', 'tool_call', 'model_call', "
            "'calibration_update', 'drift_alarm', 'cycle_end', 'cycle_start' or 'intervention'") in line
    assert "is not a dict" in only_line(LedgerEvent, ledger_event(kind="tool_call", payload=[1]), "payload")
    data = ledger_event()
    del data["payload"]
    assert "add payload" in only_line(LedgerEvent, data, "payload")
    assert "this is empty" in only_line(LedgerEvent, ledger_event(run_id=""), "run_id")

    data = ledger_event()
    del data["ts"]
    before = time.time()
    assert before <= LedgerEvent.model_validate(data).ts <= time.time()


@pytest.mark.parametrize("kind", ["claim", "evidence", "verdict", "fork", "assist", "audit", "tool_call", "model_call",
                                  "calibration_update", "drift_alarm"])
def test_other_kinds_take_a_free_payload(kind):
    payload = {"anything": [1, 2, {"deep": None}], "declared_success": "not checked here"}
    record = LedgerEvent.model_validate(ledger_event(kind=kind, payload=payload))
    assert record.kind == kind and record.payload == payload
    assert LedgerEvent.model_validate(ledger_event(kind=kind, payload={})).payload == {}


@pytest.mark.parametrize("field", ["declared_success", "predicates", "wall_s"])
def test_a_cycle_end_payload_must_carry_its_3_fields(field):
    payload = cycle_end()
    del payload[field]
    line = only_line(LedgerEvent, ledger_event(payload=payload), f"payload.{field}")
    assert f"a cycle_end event must carry {field}" in line
    assert "add it to the payload" in line
    assert "declared_success (true or false), predicates (a list of {predicate, verdict_id or signoff_assist_id})" in line
    assert "wall_s (seconds), and may carry true_success" in line


def test_an_empty_cycle_end_payload_names_all_3_fields():
    lines = refusal(LedgerEvent, ledger_event(payload={}))
    assert [line.partition(":")[0] for line in lines] == ["payload.declared_success", "payload.predicates",
                                                           "payload.wall_s"]


def test_cycle_end_true_success_is_optional_and_other_keys_ride_along():
    payload = cycle_end()
    del payload["true_success"]
    assert "true_success" not in LedgerEvent.model_validate(ledger_event(payload=payload)).payload

    extra = cycle_end(cycle_id="cycle_17", seed=20001, arm="avail_human")
    record = LedgerEvent.model_validate(ledger_event(payload=extra))
    assert record.payload == extra                                # stored as given, nothing added or rewritten
    assert "definition" not in record.payload["predicates"][0]["predicate"]

    for value in ("yes", 1, 0, "true"):
        line = only_line(LedgerEvent, ledger_event(payload=cycle_end(true_success=value)), "payload.true_success")
        assert "is not true or false" in line and "Sim only, written by the evaluator" in line
    assert LedgerEvent.model_validate(ledger_event(payload=cycle_end(true_success=None))).payload["true_success"] is None


@pytest.mark.parametrize("value", ["yes", "true", 1, 0, None])
def test_cycle_end_declared_success_is_a_real_boolean(value):
    line = only_line(LedgerEvent, ledger_event(payload=cycle_end(declared_success=value)), "payload.declared_success")
    assert "is not true or false" in line and "give a real boolean" in line


def test_cycle_end_wall_time_and_predicate_entries():
    assert "is below 0" in only_line(LedgerEvent, ledger_event(payload=cycle_end(wall_s=-1.0)), "payload.wall_s")
    assert "is not a number" in only_line(LedgerEvent, ledger_event(payload=cycle_end(wall_s="41")), "payload.wall_s")
    assert "is not a list" in only_line(LedgerEvent, ledger_event(payload=cycle_end(predicates="seated")),
                                        "payload.predicates")

    def entries(*items):
        return ledger_event(payload=cycle_end(predicates=list(items)))

    line = only_line(LedgerEvent, entries({"verdict_id": "verdict_1"}), "payload.predicates.0.predicate")
    assert "this required field is missing" in line and "{'name': 'seated', 'target': {'plug': 'p1', 'port': 7}}" in line
    line = only_line(LedgerEvent, entries({"predicate": "seated", "verdict_id": "verdict_1"}),
                     "payload.predicates.0.predicate")
    assert '"seated" is not a dict or an instance of Predicate' in line
    assert "give a dict with the fields of Predicate" in line and "{'name': 'seated', 'target': {" in line
    line = only_line(LedgerEvent, entries({"predicate": predicate()}, {"predicate": predicate(name="Seated")}),
                     "payload.predicates.1.predicate.name")
    assert "is not a lowercase identifier" in line
    line = only_line(LedgerEvent, entries({"predicate": predicate(), "verdit_id": "verdict_1"}),
                     "payload.predicates.0.verdit_id")
    assert "CyclePredicate has no field named verdit_id" in line and "Did you mean verdict_id?" in line
    line = only_line(LedgerEvent, entries({"predicate": predicate(), "verdict_id": ""}),
                     "payload.predicates.0.verdict_id")
    assert "this is empty" in line


def test_a_cycle_end_predicate_may_have_a_verdict_a_sign_off_both_or_neither():
    # The `flag` arm of E3 declares success with no verdict at all, so the record must be able to say so.
    bare = [{"predicate": predicate()}, {"predicate": predicate(name="latched")}]
    record = LedgerEvent.model_validate(ledger_event(payload=cycle_end(predicates=bare, declared_success=True)))
    typed = CycleEnd.model_validate(record.payload)
    assert [(entry.verdict_id, entry.signoff_assist_id) for entry in typed.predicates] == [(None, None), (None, None)]

    typed = CycleEnd.model_validate(cycle_end())
    assert [(entry.predicate.name, entry.verdict_id, entry.signoff_assist_id) for entry in typed.predicates] == [
        ("at_port", "verdict_0001", None), ("seated", "verdict_0003", "assist_0002"),
        ("released_stable", None, "assist_0003")]
    assert typed.predicates[2].predicate.definition == kinds.PREDICATE_DEFINITIONS["released_stable"]
    assert LedgerEvent.model_validate(ledger_event(payload=cycle_end(predicates=[]))).payload["predicates"] == []


def test_cycle_end_keeps_the_keys_it_does_not_know():
    payload = cycle_end(cycle_id="cycle_17", seed=20001)
    typed = CycleEnd.model_validate(payload)
    assert typed.model_extra == {"cycle_id": "cycle_17", "seed": 20001} and typed.wall_s == 41.25
    dumped = typed.model_dump(mode="json")
    assert dumped["cycle_id"] == "cycle_17" and dumped["seed"] == 20001 and dumped["wall_s"] == 41.25
    assert CycleEnd.model_validate_json(typed.model_dump_json()) == typed
    assert validate_record("cycle_end", payload) == typed          # the payload of a logged event reads back
    longer = typed.model_copy(update={"wall_s": 50.0})
    assert longer.wall_s == 50.0 and longer.model_extra == typed.model_extra

    # the 4 fields it knows are still checked, an entry of predicates is still strict, and it is still frozen
    assert "is below 0" in only_line(CycleEnd, cycle_end(cycle_id="cycle_17", wall_s=-1), "wall_s")
    entry = {"predicate": predicate(), "note": "extra"}
    assert "CyclePredicate has no field named note" in only_line(CycleEnd, cycle_end(predicates=[entry]),
                                                                 "predicates.0.note")
    with pytest.raises(ValidationError):
        typed.cycle_id = "cycle_18"
    with pytest.raises(ValidationError):
        typed.model_copy(update={"wall_s": -1.0})


# ------------------------------------------------------------------ explain, validate_record
def test_explain_gives_1_line_per_problem_each_with_its_path_and_a_fix():
    data = verdict(decision="maybe", score=float("nan"), stage_reached=-2, cost_total={"seconds": 1.0})
    data["extra"] = 1
    lines = refusal(Verdict, data)
    assert [line.partition(":")[0] for line in lines] == [
        "decision", "score", "stage_reached", "cost_total.usd", "cost_total.moved_robot", "extra"]
    assert all(FIX in line for line in lines)
    assert len(set(lines)) == len(lines)


def test_a_refusal_explains_itself_the_same_after_pickling():
    # A record refused inside a worker process reaches the parent as a pickled error.
    for model, data in ((Verdict, verdict(evidence_ids=[])), (Fork, fork(options=["retry", "accept"])),
                        (Witness, witness(alpha=0.001)), (Claim, {})):
        with pytest.raises(ValidationError) as caught:
            model.model_validate(data)
        again = pickle.loads(pickle.dumps(caught.value))
        assert explain(again) == explain(caught.value) and explain(again)


def test_explain_quotes_what_the_field_is_for():
    line = only_line(Evidence, {k: v for k, v in evidence().items() if k != "claim_id"}, "claim_id")
    assert line == ("claim_id: this required field is missing. Fix: add claim_id "
                    "(claim_id: the claim this evidence bears on)")
    line = only_line(Stage, stage(alpha_k=1.5), "alpha_k")
    assert line == ("alpha_k: 1.5 is above 1. Fix: give a value of 1 or less "
                    "(alpha_k: this stage's share of the false-accept budget alpha)")


def test_explain_leaves_a_rule_as_it_is_written():
    # A rule's message is complete, so nothing is added after its fix.
    assert only_line(Verdict, verdict(evidence_ids=[]), "evidence_ids") == (
        "evidence_ids: an accept needs evidence, and this verdict lists no evidence ids. Fix: list the ids of the "
        "Evidence records the witness used, or use the decision abstain. Only evidence or a person marks a step done")
    assert only_line(Fork, fork(chosen="abort"), "chosen") == (
        'chosen: "abort" is not one of this fork\'s options (repair:push_seat, retry, ask_human). Fix: pick one of '
        "the options, or leave chosen as null while the fork is open")
    assert only_line(Evidence, evidence(features={"gap_mm": float("nan")}), "features.gap_mm") == (
        "features.gap_mm: nan is not a finite number. Fix: leave this key out of features when the source has no "
        "reading")
    # A scalar's fix is general, so the line says what the field is for.
    assert only_line(Audit, audit(seconds="9"), "seconds") == (
        'seconds: "9" is not a number. Fix: give a number such as 0.5. Text and true or false are refused '
        "(seconds: the seconds the audit took)")


def test_a_long_value_is_cut_short_in_the_message():
    long_text = "x" * 500
    line = only_line(Fork, fork(options=["retry", long_text], chosen="retry"), "options.1")
    assert long_text not in line and '"' + "x" * 56 + "..." in line
    line = only_line(Evidence, evidence(features={"force_n": list(range(200))}), "features.force_n")
    assert len(line) < 250 and "..." in line


def test_explain_names_the_record_when_the_whole_input_is_wrong():
    line = only_line(Verdict, "accept", "Verdict")
    assert line == ('Verdict: "accept" is not a dict or an instance of Verdict. '
                    "Fix: give a dict with the fields of Verdict")
    assert only_line(Cost, [1, 2], "Cost").startswith("Cost: [1, 2] is not a dict or an instance of Cost")


def test_explain_works_on_any_pydantic_model():
    class Other(BaseModel):
        count: int
        name: str

    with pytest.raises(ValidationError) as caught:
        Other.model_validate({"count": "many"})
    lines = explain(caught.value)
    assert len(lines) == 2
    assert lines[0].startswith("count: ") and lines[1].startswith("name: this required field is missing")
    assert all(FIX in line for line in lines)


def test_explain_has_a_fix_for_each_of_pydantics_own_error_types():
    from typing import Literal

    from pydantic import ConfigDict, Field, StrictBool, StrictFloat, StrictInt, field_validator

    class Inner(BaseModel):
        model_config = ConfigDict(extra="forbid")
        size: int = Field(gt=0, lt=10)

    class Other(BaseModel):
        model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
        text: str
        whole: StrictInt
        number: StrictFloat
        flag: StrictBool
        items: list[int] = Field(min_length=2)
        table: dict[str, int]
        word: Literal["a", "b"]
        code: str = Field(min_length=3)
        low: float = Field(ge=1)
        high: float = Field(le=5)
        finite: float
        inner: Inner
        other: Inner
        even: int

        @field_validator("even")
        @classmethod
        def _even(cls, value: int) -> int:
            if value % 2:
                raise ValueError("an odd number is not even")
            return value

    bad = {
        "text": 7, "whole": "7", "number": "7.5", "flag": "yes", "items": [1], "table": [1], "word": "c", "code": "ab",
        "low": 0, "high": 6, "finite": float("inf"), "inner": {"size": 0}, "other": "x", "even": 3, "stray": 1,
    }
    with pytest.raises(ValidationError) as caught:
        Other.model_validate(bad)
    lines = explain(caught.value)
    assert len(lines) == caught.value.error_count() == 15
    want = {
        "text": ("7 is not text", "give a string"),
        "whole": ('"7" is not a whole number', "give an integer"),
        "number": ('"7.5" is not a number', "give a number"),
        "flag": ('"yes" is not true or false', "give a real boolean"),
        "items": ("this has 1 item", "give at least 2 items"),
        "table": ("[1] is not a dict", "give a dict"),
        "word": ('"c" is not an allowed value', "use 'a' or 'b'"),
        "code": ("this has 2 characters", "give at least 3 characters"),
        "low": ("0 is below 1", "give a value of 1 or more"),
        "high": ("6 is above 5", "give a value of 5 or less"),
        "finite": ("inf is not a finite number", "give a finite number"),
        "inner.size": ("0 is not above 0", "give a value above 0"),
        "other": ('"x" is not a dict or an instance of Inner', "give a dict with the fields of Inner"),
        "even": ("an odd number is not even", "change the value to fit the record type"),
        "stray": ("this field is not known here", "remove it"),
    }
    assert [line.partition(": ")[0] for line in lines] == list(want)
    for path, (what, fix) in want.items():
        line = line_for(lines, path)
        assert line.startswith(f"{path}: {what}{FIX}{fix}"), line

    with pytest.raises(ValidationError) as caught:
        Other.model_validate({**bad, "inner": {"size": 10}})
    assert "inner.size: 10 is not below 10. Fix: give a value below 10" in explain(caught.value)


@pytest.mark.parametrize("name, model", [
    ("verdict", Verdict), ("Verdict", Verdict), (" VERDICT ", Verdict), (Verdict, Verdict), ("claim", Claim),
    ("evidence", Evidence), ("fork", Fork), ("assist", Assist), ("audit", Audit), ("cycle_end", CycleEnd),
    ("CycleEnd", CycleEnd), ("ledger_event", LedgerEvent), ("LedgerEvent", LedgerEvent), ("witness", Witness),
    ("cost", Cost), ("predicate", Predicate), ("stage", Stage),
])
def test_record_model_finds_a_type_by_name_kind_or_class(name, model):
    assert record_model(name) is model


@pytest.mark.parametrize("name", ["tool_call", "model_call", "drift_alarm", "verdicts", "", None, 7, dict])
def test_record_model_refuses_a_name_with_no_record_type(name):
    with pytest.raises(ValueError, match="no record type is named") as caught:
        record_model(name)
    assert "Fix: pass a record class, a record name" in str(caught.value)
    assert "claim, evidence, verdict, fork, assist, audit, cycle_end" in str(caught.value)


def test_payload_models_cover_the_typed_ledger_kinds():
    assert PAYLOAD_MODELS == {"claim": Claim, "evidence": Evidence, "verdict": Verdict, "fork": Fork,
                              "assist": Assist, "audit": Audit, "cycle_end": CycleEnd,
                              "intervention": schemas.Intervention}
    assert set(PAYLOAD_MODELS) <= set(kinds.EVENT_KINDS)


def test_validate_record_returns_the_record():
    record = validate_record("verdict", verdict())
    assert isinstance(record, Verdict) and record == Verdict.model_validate(verdict())
    assert validate_record(Verdict, record) is record
    assert validate_record("verdict", record.model_dump_json()) == record            # JSON text
    assert validate_record("verdict", record.model_dump_json().encode()) == record   # and bytes
    assert validate_record("cycle_end", cycle_end()).declared_success is True
    for name, build in VALID.items():
        assert isinstance(validate_record(name, build()), RECORD_MODELS[name])


def test_validate_record_raises_a_schema_error_that_carries_the_lines():
    data = verdict(evidence_ids=[], score="high")
    with pytest.raises(SchemaError) as caught:
        validate_record("verdict", data)
    error = caught.value
    assert isinstance(error, ValueError)
    assert error.record == "Verdict"
    with pytest.raises(ValidationError) as direct:
        Verdict.model_validate(data)
    assert error.problems == explain(direct.value)
    assert len(error.problems) == 1 and error.problems[0].startswith("score: ")      # field errors come first
    assert str(error) == "Verdict is not valid (1 problem):\n  " + error.problems[0]
    assert isinstance(error.__cause__, ValidationError)

    with pytest.raises(SchemaError) as caught:
        validate_record(Verdict, verdict(evidence_ids=[]))
    assert caught.value.problems[0].startswith("evidence_ids: an accept needs evidence")

    with pytest.raises(SchemaError) as caught:
        validate_record("claim", {})
    assert str(caught.value).startswith("Claim is not valid (7 problems):\n  claim_id: ")
    assert len(caught.value.problems) == 7

    again = pickle.loads(pickle.dumps(caught.value))         # it has to survive the trip out of a worker process
    assert type(again) is SchemaError and again.record == "Claim" and again.problems == caught.value.problems
    assert str(again) == str(caught.value)


def test_validate_record_explains_bad_json():
    with pytest.raises(SchemaError) as caught:
        validate_record("verdict", "{not json")
    assert caught.value.problems[0].startswith("Verdict: this is not valid JSON")
    assert "give 1 JSON object" in caught.value.problems[0]


# ------------------------------------------------------------------ JSON Schema export
def test_json_schemas_cover_every_record_type():
    exported = schema_export.json_schemas()
    assert list(exported) == list(RECORD_MODELS)
    for name, schema in exported.items():
        model = RECORD_MODELS[name]
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["title"] == name and schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert list(schema["properties"]) == list(model.model_fields)
        assert set(schema["required"]) == {n for n, f in model.model_fields.items() if f.is_required()}
        assert all(prop.get("description") for prop in schema["properties"].values()), name
        json.dumps(schema)


def test_json_schema_enums_are_the_literal_lists():
    exported = schema_export.json_schemas()
    assert exported["Verdict"]["properties"]["decision"]["enum"] == ["accept", "reject", "abstain"]
    assert exported["Claim"]["properties"]["raised_by"]["enum"] == ["skill", "policy", "agent", "human"]
    assert exported["Evidence"]["properties"]["kind"]["enum"] == [
        "passive_proprio", "passive_vision", "active_probe", "judge", "instrumented"]
    assert exported["Evidence"]["properties"]["side_effect"]["enum"] == ["none", "repaired", "disturbed"]
    assert exported["Assist"]["properties"]["question"]["enum"] == [
        "confirm", "choose", "guide_marks", "guide_text", "takeover"]
    assert exported["Assist"]["$defs"]["Correction"]["properties"]["kind"]["enum"] == [
        "marks", "step_text", "pose_offset", "takeover"]
    assert exported["LedgerEvent"]["properties"]["kind"]["enum"] == [
        "claim", "evidence", "verdict", "fork", "assist", "audit", "tool_call", "model_call", "calibration_update",
        "drift_alarm", "cycle_end", "cycle_start", "intervention"]
    chosen_by = exported["Fork"]["properties"]["chosen_by"]["anyOf"]
    assert {"enum": ["policy", "agent", "human"], "type": "string"} in chosen_by and {"type": "null"} in chosen_by


def test_json_schema_says_the_accept_rule_and_the_cycle_end_rule():
    exported = schema_export.json_schemas()
    assert exported["Verdict"]["allOf"] == [{
        "if": {"properties": {"decision": {"const": "accept"}}, "required": ["decision"]},
        "then": {"properties": {"evidence_ids": {"minItems": 1}}, "required": ["evidence_ids"]},
    }]
    rule = exported["LedgerEvent"]["allOf"][0]
    assert rule["if"] == {"properties": {"kind": {"const": "cycle_end"}}, "required": ["kind"]}
    payload = rule["then"]["properties"]["payload"]
    assert payload["required"] == ["declared_success", "predicates", "wall_s"]
    assert list(payload["properties"]) == ["declared_success", "predicates", "wall_s", "true_success"]
    assert payload["additionalProperties"] is True               # other keys may ride along
    entry = payload["properties"]["predicates"]["items"]
    assert list(entry["properties"]) == ["predicate", "verdict_id", "signoff_assist_id"]
    assert entry["properties"]["predicate"]["required"] == ["name", "target"]
    assert "$ref" not in json.dumps(payload) and "$defs" not in json.dumps(payload)   # the rule stands by itself


def test_json_schema_of_a_fork_option_has_no_room_for_accept():
    options = schema_export.json_schemas()["Fork"]["properties"]["options"]
    assert options["minItems"] == 1 and options["uniqueItems"] is True
    item = options["items"]
    pattern, banned = re.compile(item["pattern"]), re.compile(item["not"]["pattern"])
    for good in ("retry", "ask_human", "abort", "repair:push_seat", "recover:reseat"):
        assert pattern.search(good) and not banned.search(good)
    for bad in ("accept", "Retry", "repair:", "skip", "repair:push seat"):
        assert not pattern.search(bad) or banned.search(bad)
    for sneaky in ("repair:accept", "recover:ACCEPT", "recover:auto_Accept"):
        assert banned.search(sneaky)
    shadow_choice = schema_export.json_schemas()["Fork"]["$defs"]["Shadow"]["properties"]["choice"]["anyOf"]
    assert item in shadow_choice and {"type": "null"} in shadow_choice


def test_avail_schemas_writes_1_file_per_record_type(tmp_path, capsys):
    out = tmp_path / "schemas_out"
    assert schema_export.main(["--out", str(out)]) == 0
    written = sorted(path.name for path in out.iterdir())
    assert written == sorted(f"{name}.schema.json" for name in RECORD_MODELS)
    assert len(written) == 11

    printed = capsys.readouterr().out.split("\n")
    assert [line for line in printed if line] == [str(out / f"{name}.schema.json") for name in RECORD_MODELS]

    exported = schema_export.json_schemas()
    for name in RECORD_MODELS:
        raw = (out / f"{name}.schema.json").read_bytes()
        assert raw.endswith(b"\n") and b"\r" not in raw and not raw.startswith(b"\xef\xbb\xbf")
        assert json.loads(raw.decode("utf-8")) == exported[name]


def test_avail_schemas_defaults_to_the_cache_folder_under_avail_data(tmp_path, monkeypatch, capsys):
    data_root = tmp_path / "avail_data"
    monkeypatch.setenv("AVAIL_DATA", str(data_root))
    assert schema_export.main([]) == 0
    folder = data_root / "cache" / "schemas"
    assert sorted(path.name for path in folder.iterdir()) == sorted(f"{name}.schema.json" for name in RECORD_MODELS)
    assert str(folder / "Verdict.schema.json") in capsys.readouterr().out
    assert [path.name for path in tmp_path.iterdir()] == ["avail_data"]          # nothing written anywhere else


def test_avail_schemas_says_what_to_do_when_there_is_no_data_root(monkeypatch, capsys):
    from avail import paths

    def no_data_root(create: bool = True):
        raise RuntimeError("AVAIL_DATA is not set. Add AVAIL_DATA=... to .env or the environment.")

    monkeypatch.setattr(paths, "cache_dir", no_data_root)
    assert schema_export.main([]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "avail schemas: AVAIL_DATA is not set." in captured.err and "Or pass --out DIR." in captured.err


def test_avail_schemas_runs_through_the_command_line(tmp_path, capsys):
    from avail import cli

    out = tmp_path / "from_cli"
    assert cli.main(["schemas", "--out", str(out)]) == 0
    assert (out / "LedgerEvent.schema.json").is_file()
    assert "Verdict.schema.json" in capsys.readouterr().out
    assert schema_export.write_schemas(out) == [out / f"{name}.schema.json" for name in RECORD_MODELS]   # rerun is fine


# ------------------------------------------------------------------ the literal lists of docs/01_architecture.md
def test_literal_lists_match_docs_01():
    # Written out here on purpose: a silent change to kinds.py must fail this test.
    assert kinds.DECISIONS == ("accept", "reject", "abstain")
    assert kinds.RAISED_BY == ("skill", "policy", "agent", "human")
    assert kinds.EVIDENCE_KINDS == ("passive_proprio", "passive_vision", "active_probe", "judge", "instrumented")
    assert kinds.SIDE_EFFECTS == ("none", "repaired", "disturbed")
    assert kinds.CHOSEN_BY == ("policy", "agent", "human")
    assert kinds.QUESTION_TYPES == ("confirm", "choose", "guide_marks", "guide_text", "takeover")
    assert kinds.CORRECTION_KINDS == ("marks", "step_text", "pose_offset", "takeover")
    assert kinds.EVENT_KINDS == ("claim", "evidence", "verdict", "fork", "assist", "audit", "tool_call",
                                 "model_call", "calibration_update", "drift_alarm", "cycle_end", "cycle_start",
                                 "intervention")
    assert kinds.FORK_OPTION_WORDS == ("retry", "ask_human", "abort")
    assert kinds.FORK_OPTION_PREFIXES == ("repair:", "recover:")
    assert kinds.PREDICATES == ("seated", "latched", "at_port", "held", "released_stable", "removed")


def test_the_fork_vocabulary_has_no_accept_and_reads_as_docs_01_lists_it():
    assert FORK_OPTION_FORMS == "retry, ask_human, abort, repair:<probe>, recover:<skill>"
    for word in (*kinds.FORK_OPTION_WORDS, *kinds.FORK_OPTION_PREFIXES):
        assert "accept" not in word.casefold()
    assert "accept" in kinds.DECISIONS                           # accept exists, as a verdict decision only


def test_every_known_predicate_has_a_definition_an_auditor_can_apply():
    assert tuple(kinds.PREDICATE_DEFINITIONS) == kinds.PREDICATES
    for name, definition in kinds.PREDICATE_DEFINITIONS.items():
        assert re.fullmatch(r"[a-z][a-z0-9_]*", name)
        assert len(definition.split()) >= 5 and definition.endswith("."), name


def test_the_package_re_exports_what_other_modules_import():
    for name in schemas.__all__:
        assert hasattr(schemas, name), name
    wanted = {
        "Predicate", "Claim", "Evidence", "Witness", "Verdict", "Fork", "Assist", "Audit", "LedgerEvent", "Cost",
        "Decision", "RaisedBy", "EvidenceKind", "SideEffect", "ChosenBy", "QuestionType", "CorrectionKind",
        "EventKind", "DECISIONS", "RAISED_BY", "EVIDENCE_KINDS", "SIDE_EFFECTS", "CHOSEN_BY", "QUESTION_TYPES",
        "CORRECTION_KINDS", "EVENT_KINDS", "FORK_OPTION_WORDS", "FORK_OPTION_PREFIXES", "PREDICATES",
        "PREDICATE_DEFINITIONS", "new_id", "explain", "validate_record", "SchemaError",
    }
    assert wanted <= set(schemas.__all__)
    assert len(set(schemas.__all__)) == len(schemas.__all__)
    assert schemas.DECISIONS is kinds.DECISIONS and schemas.Decision is kinds.Decision


# ------------------------------------------------------------------ S5: the predicates come from the task files
# The texts kinds.py held through S4, written out here so a change in the task files fails this test.
S4_DEFINITIONS = {
    "seated": "The plug is in the named port, its shoulder is flush with the socket face within the depth "
              "tolerance, and it sits straight.",
    "latched": "The plug is seated in the named port and its latch is engaged, so it does not come out "
               "under a light pull.",
    "at_port": "The plug is in or at the mouth of the named port and no other port.",
    "held": "The gripper is holding the named plug.",
    "released_stable": "The gripper has let go of the named plug and the plug has stayed where it was.",
    "removed": "The named plug is out of the named port.",
}


def test_the_predicate_definitions_are_the_s4_texts_word_for_word():
    assert kinds.PREDICATE_DEFINITIONS == S4_DEFINITIONS
    assert list(kinds.PREDICATE_DEFINITIONS) == list(S4_DEFINITIONS) == list(kinds.PREDICATES)
    assert type(kinds.PREDICATE_DEFINITIONS) is dict and type(kinds.PREDICATES) is tuple
    assert schemas.PREDICATE_DEFINITIONS is kinds.PREDICATE_DEFINITIONS and schemas.PREDICATES is kinds.PREDICATES
    assert "PREDICATE_DEFINITIONS" in dir(schemas) and "PREDICATES" in dir(kinds)


def test_the_s4_texts_live_in_the_plug_task_file():
    import yaml

    task = yaml.safe_load((kinds.TASKS_DIR / "plug_insert.yaml").read_text(encoding="utf-8"))
    assert {name: spec["done_means"] for name, spec in task["predicates"].items()} == S4_DEFINITIONS
    names = [path.name for path in kinds.task_files()]
    assert "plug_insert.yaml" in names and names == sorted(names)
    assert kinds.load_predicate_definitions() == kinds.PREDICATE_DEFINITIONS     # the same rule, read again


def test_importing_the_schemas_reads_no_task_file():
    import subprocess
    import sys

    from avail.paths import REPO_ROOT

    code = (
        "import avail.ledger, avail.schemas, avail.schemas.kinds as k\n"
        "print(sorted(set(k.LAZY_NAMES) & set(vars(k))))\n"
        "avail.schemas.Predicate(name='held', target={'item': 1}, definition='The arm holds it.')\n"
        "print(sorted(set(k.LAZY_NAMES) & set(vars(k))))\n"
        "avail.schemas.Predicate(name='held', target={'item': 1})\n"
        "print(sorted(set(k.LAZY_NAMES) & set(vars(k))))\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, cwd=REPO_ROOT)
    assert done.returncode == 0, done.stderr
    # nothing is read at import or for a claim with its own definition. The files are read once a claim needs one
    assert done.stdout.splitlines() == ["[]", "[]", "['PREDICATES', 'PREDICATE_DEFINITIONS']"]


def _task_file(folder: Path, name: str, predicates: dict[str, str]) -> Path:
    lines = [f"task: {name}", "predicates:"]
    for predicate_name, text in predicates.items():
        lines += [f"  {predicate_name}:", f"    done_means: {text}", "    checks: []"]
    path = folder / f"{name}.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_the_widest_task_file_gives_the_default_text_and_the_order(tmp_path):
    _task_file(tmp_path, "a_small", {"held": "The gripper holds the named cable in its jaws.",
                                     "lifted": "The named cable is up off the tray."})
    _task_file(tmp_path, "b_wide", {"seated": "The named part is home in its port.",
                                    "held": "The gripper is holding the named part.",
                                    "removed": "The named part is out of its port."})
    definitions = kinds.load_predicate_definitions(tmp_path)
    assert list(definitions) == ["seated", "held", "removed", "lifted"]   # the wide file, then what only the other has
    assert definitions["held"] == "The gripper is holding the named part."
    assert definitions["lifted"] == "The named cable is up off the tray."
    # 2 files of the same width: the name decides
    _task_file(tmp_path, "c_wide", {"held": "The arm has it.", "x1": "One more condition here.",
                                    "x2": "And another condition here."})
    definitions = kinds.load_predicate_definitions(tmp_path)
    assert definitions["held"] == "The gripper is holding the named part."
    assert list(definitions) == ["seated", "held", "removed", "x1", "x2", "lifted"]


def test_a_definition_is_the_file_s_words_on_1_line(tmp_path):
    (tmp_path / "folded.yaml").write_text(
        "task: folded\npredicates:\n  seated:\n    done_means: >\n      The plug is home,\n      and   it sits straight.\n",
        encoding="utf-8")
    assert kinds.load_predicate_definitions(tmp_path) == {"seated": "The plug is home, and it sits straight."}


@pytest.mark.parametrize("body, words", [
    ("task: x\n", "defines no predicates"),
    ("task: x\npredicates: {}\n", "defines no predicates"),
    ("- just\n- a list\n", "defines no predicates"),
    ("task: x\npredicates:\n  seated: {checks: []}\n", "has no done_means text"),
    ("task: x\npredicates:\n  seated:\n    done_means: '   '\n", "has no done_means text"),
    ("task: x\npredicates:\n  Seated:\n    done_means: The plug is home.\n", "is not a lowercase identifier"),
    ("task: [unclosed\n", "cannot be read"),
])
def test_a_task_file_with_no_usable_definition_stops_with_its_path(tmp_path, body, words):
    path = tmp_path / "broken.yaml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(kinds.PredicateDefinitionsError) as caught:
        kinds.load_predicate_definitions(tmp_path)
    assert str(path) in str(caught.value) and words in str(caught.value) and FIX in str(caught.value)
    assert not isinstance(caught.value, ValueError)   # raised inside a validator, it must not become a refusal


def test_a_claim_that_needs_a_missing_definition_stops_with_the_task_file_error(tmp_path, monkeypatch):
    """Raised inside the Predicate validator, the error keeps its own type and message."""
    for name in kinds.LAZY_NAMES:
        monkeypatch.delattr(kinds, name)                   # as if nothing had read the task files yet
    monkeypatch.setattr(kinds, "TASKS_DIR", tmp_path / "no_tasks_here")
    assert Predicate.model_validate(predicate(definition="The plug is home.")).definition == "The plug is home."
    with pytest.raises(kinds.PredicateDefinitionsError, match="no task file"):
        Predicate.model_validate(predicate())
    assert "PREDICATES" not in vars(kinds)                 # a failed read leaves nothing half filled


def test_a_folder_with_no_task_file_stops_with_a_fix(tmp_path):
    for folder in (tmp_path, tmp_path / "missing"):
        with pytest.raises(kinds.PredicateDefinitionsError, match="no task file") as caught:
            kinds.load_predicate_definitions(folder)
        assert "Fix: put the task files in that folder" in str(caught.value)
    assert kinds.task_files(tmp_path / "missing") == []


def test_the_s5_literal_lists():
    assert kinds.INTERVENTION_OUTCOMES == ("signed_off", "retried", "taken_over_done", "taken_over_failed", "unresolved")
    assert kinds.FRAME_TAGS == ("autonomous", "assisted")
    assert kinds.CONFIRM_ANSWERS == ("yes", "no", "cannot_tell")
    assert kinds.CONFIRM_LABELS == {"yes": True, "no": False, "cannot_tell": None}
    assert kinds.EVENT_KINDS[-2:] == ("cycle_start", "intervention")
    for name in ("Intervention", "INTERVENTION_OUTCOMES", "InterventionOutcome", "FRAME_TAGS", "FrameTag",
                 "CONFIRM_ANSWERS"):
        assert name in schemas.__all__ and hasattr(schemas, name), name


# ------------------------------------------------------------------ S5: Intervention
Intervention = schemas.Intervention


def _not_a_takeover(**changes):
    return intervention(handoff_ts=None, handback_ts=None, handoff_frame=None, handback_frame=None, **changes)


def test_an_intervention_is_the_typed_payload_of_its_ledger_kind():
    record = validate_record("intervention", intervention())
    assert isinstance(record, Intervention) and record_model("intervention") is Intervention
    assert record_model("Intervention") is Intervention
    event = LedgerEvent.model_validate(ledger_event(kind="intervention", payload=intervention()))
    assert Intervention.model_validate(event.payload) == record
    start = LedgerEvent.model_validate(ledger_event(kind="cycle_start", payload={"cycle_id": "c1", "items": [0, 1]}))
    assert start.payload == {"cycle_id": "c1", "items": [0, 1]}           # a cycle_start payload is free
    schema = schema_export.json_schemas()["Intervention"]
    assert schema["properties"]["outcome"]["enum"] == list(kinds.INTERVENTION_OUTCOMES)
    assert set(schema["required"]) == {"intervention_id", "run_id", "cycle_id", "item", "step", "question", "trigger",
                                       "answer", "asked_ts", "answered_ts", "seconds_waiting", "seconds_handling",
                                       "outcome"}


def test_an_intervention_needs_only_what_every_question_has():
    required = ("intervention_id", "run_id", "cycle_id", "item", "step", "question", "trigger", "answer", "asked_ts",
                "answered_ts", "seconds_waiting", "seconds_handling", "outcome")
    data = {k: v for k, v in intervention(question="choose", answer="retry", outcome="retried").items() if k in required}
    record = Intervention.model_validate(data)
    assert record.truth_label is None and record.features == {} and record.frames_path is None
    assert (record.claim_id, record.verdict_id, record.fork_id, record.assist_id, record.predicate_name) == (None,) * 5
    assert (record.handoff_ts, record.handback_ts, record.handoff_frame, record.handback_frame) == (None,) * 4
    for field in required:
        less = dict(data)
        del less[field]
        assert f"add {field}" in only_line(Intervention, less, field)


@pytest.mark.parametrize("value, stored", [("cable_3", "cable_3"), (3, "3"), (np.int64(0), "0")])
def test_an_intervention_item_is_text_and_a_whole_number_becomes_text(value, stored):
    assert Intervention.model_validate(intervention(item=value)).item == stored


@pytest.mark.parametrize("value", ["", "  ", None, 1.5, True, [3]])
def test_an_intervention_item_refuses_what_is_not_an_id(value):
    line = only_line(Intervention, intervention(item=value), "item")
    assert "is not an item id" in line and "for example cable_3" in line


def test_intervention_times_run_forward():
    line = only_line(Intervention, intervention(answered_ts=1794300049.0), "answered_ts")
    assert "the answer at 1794300049.000 comes before the question at 1794300050.000" in line
    assert "the answer at or after the question" in line
    line = only_line(Intervention, intervention(handback_ts=1794300070.0), "handback_ts")
    assert "the handback time 1794300070.000 comes before the handoff time 1794300071.500" in line
    line = only_line(Intervention, intervention(handback_frame=411), "handback_frame")
    assert "the handback frame 411 comes before the handoff frame 412" in line
    assert "give handback_frame at or after handoff_frame" in line
    line = only_line(Intervention, intervention(handoff_frame=None), "handback_frame")
    assert "a handback frame is given with no handoff frame" in line and "set handoff_frame as well" in line
    line = only_line(Intervention, intervention(handoff_ts=None), "handback_ts")
    assert "a handback time is given with no handoff time" in line
    # a takeover that never handed back is still a record, and 1 frame under the person is a takeover too
    unfinished = Intervention.model_validate(intervention(handback_ts=None, handback_frame=None,
                                                          outcome="taken_over_failed"))
    assert unfinished.handoff_frame == 412 and unfinished.handback_frame is None
    assert Intervention.model_validate(intervention(handback_frame=412)).handback_frame == 412
    assert Intervention.model_validate(intervention(answered_ts=1794300050.0)).answered_ts == 1794300050.0
    for field in ("seconds_waiting", "seconds_handling"):
        assert "is below 0" in only_line(Intervention, intervention(**{field: -0.5}), field)
    lines = refusal(Intervention, intervention(handoff_frame=-1, handback_frame=-1))
    assert "is below 0" in line_for(lines, "handoff_frame") and "is below 0" in line_for(lines, "handback_frame")
    assert "is not a whole number" in only_line(Intervention, intervention(handback_frame=1172.5), "handback_frame")


@pytest.mark.parametrize("answer, label", [("yes", True), ("no", False), ("cannot_tell", None), ("cannot tell", None),
                                           ("Yes", True)])
def test_a_confirm_answer_carries_its_label(answer, label):
    record = Intervention.model_validate(_not_a_takeover(question="confirm", answer=answer, truth_label=label,
                                                         outcome="signed_off" if label else "retried"))
    assert record.truth_label is label and record.answer == answer


@pytest.mark.parametrize("answer, label, path, words", [
    ("yes", False, "truth_label", 'the answer "yes" carries the truth label true, and this record says false'),
    ("no", None, "truth_label", 'the answer "no" carries the truth label false, and this record says null'),
    ("cannot_tell", True, "truth_label", 'the answer "cannot_tell" carries the truth label null, and this record says true'),
    ("maybe", None, "answer", '"maybe" is not an answer to a confirm question'),
])
def test_a_confirm_answer_and_its_label_must_agree(answer, label, path, words):
    line = only_line(Intervention, _not_a_takeover(question="confirm", answer=answer, truth_label=label), path)
    assert words in line
    assert "A confirm answer is yes with truth_label true, no with truth_label false, or cannot_tell with null" in line


@pytest.mark.parametrize("question, answer, label", [
    ("choose", "retry", None), ("choose", "abort", None), ("guide_marks", "marks", None),
    ("guide_text", "go 2 mm to the left", None), ("takeover", "taken over", True), ("takeover", "taken over", False),
])
def test_other_questions_take_their_own_answers(question, answer, label):
    record = Intervention.model_validate(intervention(question=question, answer=answer, truth_label=label))
    assert (record.question, record.answer, record.truth_label) == (question, answer, label)


@pytest.mark.parametrize("value", ["yes", "true", 1, 0])
def test_an_intervention_truth_label_is_a_real_boolean_or_null(value):
    line = only_line(Intervention, intervention(truth_label=value), "truth_label")
    assert "is not true or false" in line and "give a real boolean" in line


def test_an_intervention_outcome_and_question_come_from_the_lists():
    for outcome in kinds.INTERVENTION_OUTCOMES:
        assert Intervention.model_validate(intervention(outcome=outcome)).outcome == outcome
    line = only_line(Intervention, intervention(outcome="done"), "outcome")
    assert "use 'signed_off', 'retried', 'taken_over_done', 'taken_over_failed' or 'unresolved'" in line
    line = only_line(Intervention, intervention(question="ask"), "question")
    assert "use 'confirm', 'choose', 'guide_marks', 'guide_text' or 'takeover'" in line
    for field in ("trigger", "answer", "step", "cycle_id", "run_id", "intervention_id"):
        assert "this is empty" in only_line(Intervention, intervention(**{field: " "}), field)
    assert "is not a lowercase identifier" in only_line(Intervention, intervention(predicate_name="Seated"),
                                                        "predicate_name")


def test_an_intervention_frame_log_path_is_relative_to_the_run_folder():
    record = Intervention.model_validate(intervention(frames_path=Path("frames") / "cycle_17.parquet"))
    assert record.frames_path.replace("\\", "/") == "frames/cycle_17.parquet"
    for absolute in ("/data/runs/r/frames/c.parquet", "C:/runs/r/frames/c.parquet", "c:\\runs\\c.parquet",
                     "\\\\server\\share\\c.parquet"):
        line = only_line(Intervention, intervention(frames_path=absolute), "frames_path")
        assert "is an absolute path" in line and "give the path relative to the run folder" in line
    assert "is not a file path" in only_line(Intervention, intervention(frames_path=""), "frames_path")


def test_intervention_features_are_flat_finite_numbers():
    record = Intervention.model_validate(intervention(features={"a/b": True, "a/c": np.float32(0.25), "a/d": 3}))
    assert record.features == {"a/b": 1.0, "a/c": 0.25, "a/d": 3.0}
    assert "is not a finite number" in only_line(Intervention, intervention(features={"a/b": float("nan")}),
                                                 "features.a/b")
    assert "is not a single number" in only_line(Intervention, intervention(features={"a/b": [1.0]}), "features.a/b")


# ------------------------------------------------------------------ S5: no plug words in the schemas' code
PLUG_WORDS = re.compile(r"fric|latch|prong|push_seat|mj_plug|NominalGeometry|plug_cycle|\bP0\b|\bN[1-9]\b")


def _names_and_code_strings(tree):
    """Every identifier, and every string constant that is not documentation (a docstring or a description=
    or help= text)."""
    import ast

    docs: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            docs.add(id(node.value))                                    # a docstring or a bare string statement
        elif isinstance(node, ast.keyword) and node.arg in ("description", "help"):
            docs.update(id(sub) for sub in ast.walk(node.value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            yield node.id
        elif isinstance(node, ast.Attribute):
            yield node.attr
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node.name
        elif isinstance(node, ast.arg):
            yield node.arg
        elif isinstance(node, ast.alias):
            yield node.asname or node.name
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
            yield node.value


def test_the_schemas_code_names_no_part_of_the_plug_station():
    import ast

    from avail.paths import REPO_ROOT

    files = sorted((REPO_ROOT / "avail" / "schemas").glob("*.py"))
    assert len(files) >= 7
    found = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        found += [f"{path.name}: {text!r}" for text in _names_and_code_strings(tree) if PLUG_WORDS.search(text)]
    assert found == [], "\n".join(found)
    # the scan sees code strings: the hint that used to name a probe of the plug station is generic now
    hints = "".join(text for text in _names_and_code_strings(ast.parse((REPO_ROOT / "avail/schemas/records.py")
                                                                       .read_text(encoding="utf-8"))))
    assert "needs a name after the colon, for example " in hints
