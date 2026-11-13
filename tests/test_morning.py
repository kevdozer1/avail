"""The morning page's marks and card (S5). No figure is drawn and no file is written here."""

from __future__ import annotations

import math

from avail.analysis.morning import ASK, BAD, GOOD, MUTED, card_rows, step_mark


def test_a_step_is_marked_by_the_checks_1st_verdict():
    assert step_mark({"verdicts": ["accept"]}) == ("✓", GOOD, "")
    assert step_mark({"verdicts": ["reject", "accept"]}) == ("✗", BAD, "retried, passed")
    assert step_mark({"verdicts": ["reject", "reject"]}) == ("✗", BAD, "not done")
    assert step_mark({"verdicts": ["abstain"]})[:2] == ("?", ASK)          # ? is an abstain: the checks could not tell
    assert step_mark({}) == ("", MUTED, "")                                  # a step that never ran has no mark


def test_a_person_stepping_in_shows_in_the_note_under_the_checks_mark():
    """A step the checks rejected keeps its ✗ when a person is asked after the last try. The note says what they did.
    The asks come as the ledger writes them: a chosen takeover is the choose's record, then the takeover's."""
    took = step_mark({"verdicts": ["reject", "reject"], "asks": [("choose", "unresolved"), ("takeover", "taken_over_done")]})
    assert took == ("✗", BAD, "person took over")
    assert step_mark({"verdicts": ["reject", "reject", "accept"], "asks": [("choose", "retried")]})[2] == "person said retry"
    assert step_mark({"verdicts": ["abstain"], "asks": [("confirm", "signed_off")]}) == ("?", ASK, "person signed off")
    assert step_mark({"verdicts": ["abstain"], "asks": [("confirm", "retried"), ("takeover", "taken_over_done")]}) == (
        "?", ASK, "person took over")


def test_the_card_matches_the_reports_units():
    numbers = {"avail_human": {"cycles": 100, "cables": 500, "bad_cables_through": 0, "bad_steps_through": 1,
                               "steps": 2000, "assists_per_100": 113.0, "takeovers_per_100": 69.0,
                               "mtbi_s": 86.98879646026793},
               "flag": {"bad_cables_through": 92, "cables": 500}}
    rows = dict((k, v) for k, v in card_rows(numbers))
    assert rows["bad steps that got through"] == "1 of 2,000"
    assert rows["robot time between calls for help"] == "87 s"                 # S5.md gives 87 s
    assert rows["calls for help per 100 cycles"] == "113" and rows["takeovers per 100 cycles"] == "69"
    assert rows["bad cables through on the same seeds on the skills' own flags"] == "92 of 500"
    numbers["avail_human"]["mtbi_s"] = math.inf
    assert dict(card_rows(numbers))["robot time between calls for help"] == "none"
    assert card_rows({}) == []
