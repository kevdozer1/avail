"""Tasks as data (avail/core/task.py): the 2 task files load, and a bad task file is refused with every problem named.

No simulator. The files under tasks/ are read as they are on disk. Every other task file here is
written to a temporary folder.
"""

from __future__ import annotations

import copy
import math

import pytest
import yaml
from pydantic import ValidationError

from avail.core.task import (
    TASKS_DIR, DecideSpec, RuleCondition, StepSpec, TaskFileError, TaskSpec, load_task, predicate_definitions,
    task_names, task_path,
)

# the texts the schemas carried through S4. tasks/plug_insert.yaml says it keeps them word for word
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


def base_task() -> dict:
    """A small valid task in the file format, about widgets, for the tests to break 1 piece at a time."""
    return {
        "task": "demo_task",
        "title": "Pick each widget and place it in the bin",
        "items": {"kind": "widget", "count": 2},
        "steps": [
            {"name": "pick", "claim": "held"},
            {"name": "place", "claim": "placed", "claim_by_variant": {"big": "placed_big"}},
        ],
        "predicates": {
            "held": {
                "done_means": "The gripper holds the named widget and nothing else.",
                "checks": ["passive.grip", "vision.tag"],
                "decide": {"kind": "rule",
                           "accept_all": [{"feature": "passive.grip/in_band", "above": 0.5},
                                          {"feature": "vision.tag/match", "above": 0.5}],
                           "reject_any": [{"feature": "vision.tag/mismatch", "above": 0.5}]},
            },
            "placed": {
                "done_means": "The named widget lies inside the bin and stays there.",
                "checks": ["vision.bin@now"],
                "decide": {"kind": "witness", "witness": "{variant}"},
            },
            "placed_big": {
                "done_means": "The named big widget lies flat inside the bin and stays there.",
                "checks": ["vision.bin@now", "probe.nudge"],
                "decide": {"kind": "witness", "witness": "bin_{variant}"},
            },
        },
        "fork": {"max_attempts": 2, "ask": {"abstain": "confirm", "reject_last": "choose"}},
    }


def write_task(folder, data: dict, name: str | None = None):
    path = folder / f"{name or data.get('task', 'unnamed')}.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def refusal(tmp_path, data: dict, name: str | None = None) -> str:
    """The message of the TaskFileError a task file gets."""
    with pytest.raises(TaskFileError) as caught:
        load_task(write_task(tmp_path, data, name))
    return str(caught.value)


# ============================================================================ the real task files
def test_the_task_files_load_and_each_is_named_after_its_task():
    names = task_names()
    assert {"cable_loop", "plug_insert"} <= set(names) and names == sorted(names)
    for name in names:
        spec = load_task(name)
        assert spec.task == name and task_path(name) == TASKS_DIR / f"{name}.yaml"
        for step in spec.steps:                                  # loading already checked this. Said again here
            assert step.claim in spec.predicates
            assert set(step.claim_by_variant.values()) <= set(spec.predicates)


def test_the_plug_task_keeps_the_s4_definitions_word_for_word():
    spec = load_task("plug_insert")
    assert {name: p.done_means for name, p in spec.predicates.items()} == S4_DEFINITIONS
    assert tuple(spec.predicates) == ("seated", "latched", "at_port", "held", "released_stable", "removed")


def test_required_follows_the_steps_and_the_variant():
    plug = load_task("plug_insert")
    assert plug.required("latch") == [("insert", "latched"), ("release", "released_stable")]
    assert plug.required("fric") == [("insert", "seated"), ("release", "released_stable")]
    cable = load_task("cable_loop")
    assert cable.required("latch") == [("pick", "held"), ("carry", "at_port"), ("insert", "latched"),
                                       ("release", "released_stable")]
    assert cable.required("prong") == [("pick", "held"), ("carry", "at_port"), ("insert", "seated"),
                                       ("release", "released_stable")]
    assert cable.variants() == ["latch"] and cable.items.kind == "cable" and cable.items.count == 5


def test_a_cycle_is_every_step_for_every_item_in_order():
    cable = load_task("cable_loop")
    cycle = cable.cycle("latch")
    steps = cable.required("latch")
    assert len(cycle) == cable.items.count * len(steps)
    assert cycle[:len(steps)] == [(0, step, predicate) for step, predicate in steps]
    assert [item for item, _, _ in cycle] == sorted(item for item, _, _ in cycle)


def test_every_rule_in_the_task_files_reads_only_its_own_checks_and_can_accept():
    rules = 0
    for name in task_names():
        for predicate in load_task(name).predicates.values():
            if predicate.decide.kind == "rule":
                rules += 1
                assert predicate.decide.accept_all, predicate.name
                assert {c.check for c in predicate.decide.conditions()} <= set(predicate.checks), predicate.name
    assert rules >= 1, "the task files hold no rule, so this test checked nothing"


def test_a_witness_name_resolves_the_variant():
    plug = load_task("plug_insert")
    assert plug.predicate_for("insert", "latch").decide.witness_for("latch") == "latch"
    assert plug.predicate_for("insert", "fric").name == "seated"
    rule = next(p for p in load_task("cable_loop").predicates.values() if p.decide.kind == "rule")
    with pytest.raises(ValueError, match="rule"):
        rule.decide.witness_for("latch")


def test_a_relative_path_is_found_from_any_working_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_task("tasks/plug_insert.yaml") == load_task("plug_insert")


def test_an_unknown_task_name_lists_the_known_ones():
    with pytest.raises(FileNotFoundError) as caught:
        load_task("cable_lop")
    assert "Did you mean cable_loop?" in str(caught.value) and "plug_insert" in str(caught.value)
    with pytest.raises(FileNotFoundError):
        load_task("no/such/file.yaml")


# ============================================================================ predicate definitions
def test_predicate_definitions_cover_every_task_file_and_the_first_file_wins():
    every = {name for task in task_names() for name in load_task(task).predicates}
    assert set(predicate_definitions()) == every
    widest_first = sorted(task_names(), key=lambda name: (-len(load_task(name).predicates), name))
    assert list(predicate_definitions().items()) == list(predicate_definitions(widest_first).items())
    assert predicate_definitions("plug_insert") == S4_DEFINITIONS
    plug_first = predicate_definitions(["plug_insert", "cable_loop"])
    cable_first = predicate_definitions(["cable_loop", "plug_insert"])
    cable = load_task("cable_loop")
    for name in cable.predicates:
        assert plug_first[name] == S4_DEFINITIONS.get(name, cable.predicates[name].done_means)
        assert cable_first[name] == cable.predicates[name].done_means
    assert cable_first["removed"] == S4_DEFINITIONS["removed"]          # only the plug task defines it


def test_the_default_definitions_are_the_ones_the_schemas_use():
    """1 answer to what a predicate means by default: the claims' fill-in and this function agree."""
    from avail.schemas import kinds

    assert list(predicate_definitions().items()) == list(kinds.PREDICATE_DEFINITIONS.items())


def test_predicate_definitions_read_a_folder_of_their_own(tmp_path):
    first, second = base_task(), base_task()
    second["task"] = "other_task"
    second["predicates"]["held"]["done_means"] = "The gripper has the named widget between its fingers."
    write_task(tmp_path, first)
    write_task(tmp_path, second)
    assert task_names(tmp_path) == ["demo_task", "other_task"]
    defs = predicate_definitions(tasks_dir=tmp_path)
    assert defs["held"] == first["predicates"]["held"]["done_means"]
    assert predicate_definitions(["other_task", "demo_task"], tmp_path)["held"] == second["predicates"]["held"]["done_means"]
    assert task_names(tmp_path / "missing") == []


# ============================================================================ the models
def test_a_task_can_be_built_in_memory_and_round_trips():
    spec = TaskSpec.model_validate(base_task())
    assert spec.predicates["held"].name == "held"                       # a predicate is named by its key
    assert TaskSpec.model_validate(spec.model_dump()) == spec
    assert spec.required("big") == [("pick", "held"), ("place", "placed_big")]
    assert spec.required("small") == [("pick", "held"), ("place", "placed")]
    assert spec.predicate_for("place", "big").decide.witness_for("big") == "bin_big"
    assert spec.step("pick").claim == "held"
    with pytest.raises(KeyError, match="Did you mean place"):
        spec.step("plac")


def test_a_task_spec_is_read_only():
    spec = TaskSpec.model_validate(base_task())
    with pytest.raises(ValidationError):
        spec.task = "renamed"
    with pytest.raises(ValidationError):
        spec.steps[0].claim = "placed"


def test_a_missing_fork_section_is_an_empty_one(tmp_path):
    data = base_task()
    del data["fork"]
    assert load_task(write_task(tmp_path, data)).fork == {}


def test_claim_for_takes_the_variant_override_or_the_default():
    step = StepSpec(name="insert", claim="seated", claim_by_variant={"v2": "locked"})
    assert step.claim_for("v2") == "locked" and step.claim_for("v1") == "seated"


@pytest.mark.parametrize("value, holds", [
    (0.51, True), (0.5, False), (0.2, False), (1, True), (True, True),
    (None, False), (math.nan, False), (math.inf, False), ("0.9", False), ([1.0], False),
])
def test_a_condition_holds_only_on_a_measured_number_strictly_past_its_bound(value, holds):
    assert RuleCondition(feature="a.b/x", above=0.5).holds(value) is holds


def test_a_condition_with_both_bounds_is_a_band():
    band = RuleCondition(feature="a.b/x", above=0.0, below=1.0)
    assert band.holds(0.5) and not band.holds(0.0) and not band.holds(1.0) and not band.holds(2.0)
    assert RuleCondition(feature="a.b/x", below=0.8).holds(0.79) and band.check == "a.b"


def test_a_witness_name_may_hold_only_the_variant():
    assert DecideSpec(kind="witness", witness="{variant}").witness_for("v1") == "v1"
    for template in ("{connector}", "{}", "{variant!r}", "{variant:>5}", "{variant"):
        with pytest.raises(ValidationError):
            DecideSpec(kind="witness", witness=template)


# ============================================================================ bad task files are refused
def test_an_unknown_key_is_refused_wherever_it_is(tmp_path):
    data = base_task()
    data["stepz"] = []
    data["steps"][0]["claims"] = "held"
    data["predicates"]["held"]["decide"]["accept_all"][0]["abvoe"] = 0.5
    data["items"]["colour"] = "red"
    message = refusal(tmp_path, data)
    assert "4 problems" in message
    for where in ("stepz:", "steps.0.claims:", "predicates.held.decide.accept_all.0.abvoe:", "items.colour:"):
        assert f"  {where} is not a key the task format has" in message, where


def test_a_step_that_claims_an_undefined_predicate_is_refused(tmp_path):
    data = base_task()
    data["steps"][0]["claim"] = "hold"
    assert "the step pick claims hold, which predicates does not define. Did you mean held?" in refusal(tmp_path, data)
    data = base_task()
    data["steps"][1]["claim_by_variant"] = {"big": "placed_large"}
    assert "claims placed_large on big" in refusal(tmp_path, data)


def test_a_step_listed_twice_is_refused(tmp_path):
    data = base_task()
    data["steps"].append({"name": "pick", "claim": "held"})
    assert "the step pick is listed more than once" in refusal(tmp_path, data)


def test_a_rule_with_no_accept_condition_is_refused(tmp_path):
    """Every condition of an empty list holds, so such a rule could accept a claim on no evidence."""
    data = base_task()
    data["predicates"]["held"]["decide"]["accept_all"] = []
    assert "a rule needs at least 1 condition in accept_all" in refusal(tmp_path, data)


def test_a_rule_on_a_check_the_predicate_does_not_run_is_refused(tmp_path):
    data = base_task()
    data["predicates"]["held"]["decide"]["reject_any"].append({"feature": "vision.color/mismatch", "above": 0.5})
    message = refusal(tmp_path, data)
    assert "the rule reads features of vision.color, which checks does not list (passive.grip, vision.tag)" in message


def test_a_condition_needs_a_bound_that_can_hold(tmp_path):
    data = base_task()
    data["predicates"]["held"]["decide"]["accept_all"][0] = {"feature": "passive.grip/in_band"}
    assert "gives no bound" in refusal(tmp_path, data)
    data["predicates"]["held"]["decide"]["accept_all"][0] = {"feature": "passive.grip/in_band", "above": 2.0, "below": 1.0}
    assert "never holds" in refusal(tmp_path, data)
    data["predicates"]["held"]["decide"]["accept_all"][0] = {"feature": "grip", "above": 0.5}
    assert "predicates.held.decide.accept_all.0.feature" in refusal(tmp_path, data)


def test_a_bound_is_a_number_and_a_count_is_a_whole_number(tmp_path):
    data = base_task()
    data["predicates"]["held"]["decide"]["accept_all"][0]["above"] = "0.5"
    data["items"]["count"] = True
    message = refusal(tmp_path, data)
    assert "predicates.held.decide.accept_all.0.above" in message and "items.count" in message
    data = base_task()
    data["items"]["count"] = 0
    assert "items.count" in refusal(tmp_path, data)


def test_a_decision_must_fit_its_kind(tmp_path):
    cases = [
        ({"kind": "witness"}, "names its witness"),
        ({"kind": "witness", "witness": "{connector}"}, "may hold only {variant}"),
        ({"kind": "witness", "witness": "w", "accept_all": [{"feature": "vision.bin@now/x", "above": 0.5}]},
         "a witness takes no conditions"),
        ({"kind": "rule", "witness": "w", "accept_all": [{"feature": "vision.bin@now/x", "above": 0.5}]},
         "a rule names no witness"),
        ({"kind": "vote"}, "predicates.placed.decide.kind"),
    ]
    for decide, expected in cases:
        data = base_task()
        data["predicates"]["placed"]["decide"] = decide
        assert expected in refusal(tmp_path, data), decide


def test_an_unquoted_witness_name_says_to_quote_it(tmp_path):
    path = tmp_path / "demo_task.yaml"
    text = yaml.safe_dump(base_task(), sort_keys=False).replace("witness: '{variant}'", "witness: {variant}")
    assert "witness: {variant}\n" in text
    path.write_text(text, encoding="utf-8")
    with pytest.raises(TaskFileError, match='Put it in quotes: witness: "{variant}"'):
        load_task(path)


def test_a_definition_is_a_sentence_a_person_can_apply(tmp_path):
    for words in ("Holds it.", "The gripper holds the widget"):
        data = base_task()
        data["predicates"]["held"]["done_means"] = words
        assert "at least 5 words, ending with a period" in refusal(tmp_path, data), words


def test_check_ids_are_source_ids_listed_once(tmp_path):
    data = base_task()
    data["predicates"]["held"]["checks"] = ["passive.grip", "grip", "vision.tag"]
    assert "'grip' is not a check id" in refusal(tmp_path, data)
    data["predicates"]["held"]["checks"] = ["passive.grip", "vision.tag", "vision.tag"]
    assert "vision.tag is listed more than once" in refusal(tmp_path, data)


def test_the_fork_section_is_checked(tmp_path):
    data = base_task()
    data["fork"] = {"max_atempts": 2}
    assert "fork does not take max_atempts. Did you mean max_attempts?" in refusal(tmp_path, data)
    data["fork"] = {"max_attempts": 0}
    assert "fork.max_attempts is 0" in refusal(tmp_path, data)
    data["fork"] = {"ask": {"abstain": "confrim"}}
    assert "fork.ask.abstain is 'confrim', which is not a question. Did you mean confirm?" in refusal(tmp_path, data)
    data["fork"] = {"ask": "confirm"}
    assert "fork.ask maps what opened a fork" in refusal(tmp_path, data)


def test_a_predicate_named_apart_from_its_key_is_refused(tmp_path):
    data = base_task()
    data["predicates"]["held"]["name"] = "holding"
    assert "the predicate under held is named holding" in refusal(tmp_path, data)


def test_a_file_is_named_after_its_task(tmp_path):
    message = refusal(tmp_path, base_task(), name="something_else")
    assert "the file holds task demo_task" in message and "name the file demo_task.yaml" in message


def test_a_key_written_twice_is_refused_and_not_silently_overwritten(tmp_path):
    path = tmp_path / "demo_task.yaml"
    text = yaml.safe_dump(base_task(), sort_keys=False)
    held = text[text.index("  held:"):text.index("  placed:")]
    path.write_text(text.replace("  placed:", held + "  placed:", 1), encoding="utf-8")
    with pytest.raises(TaskFileError, match="found the key 'held' twice"):
        load_task(path)


@pytest.mark.parametrize("text, expected", [
    ("", "holds nothing"),
    ("- a\n- b\n", "holds a list"),
    ("task: [unclosed\n", "is not valid YAML"),
])
def test_a_file_that_is_not_a_task_mapping_is_refused(tmp_path, text, expected):
    path = tmp_path / "demo_task.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(TaskFileError, match=expected):
        load_task(path)


def test_the_base_task_used_above_is_valid(tmp_path):
    """Guards the refusal tests: each one breaks 1 piece of a file that loads."""
    spec = load_task(write_task(tmp_path, copy.deepcopy(base_task())))
    assert spec.task == "demo_task" and spec.cycle("big")[-1] == (1, "place", "placed_big")
