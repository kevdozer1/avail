"""avail audit-check (S5c): a user's check on the near-miss zoo, scored against truth it never sees."""

from __future__ import annotations

import pytest

from avail import audit
from avail.zoo import store


def _zoo_ready():
    try:
        store.load("latch", "calibration", "states")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _zoo_ready(), reason="needs the latch calibration zoo under AVAIL_DATA")


def test_a_check_sees_no_truth():
    seen = []
    audit.audit(lambda s: seen.append(s) or True, "latch", "calibration", limit=20)
    state = seen[0]
    fields = set(vars(state)) | set(state.skill) | set(state.features)
    assert not any("true" in f or f in ("label", "cls", "landed_cls") or f.startswith("spec_") for f in fields
                   if not f.startswith("_"))
    assert "flag" in state.skill and state.cameras


def test_the_skill_flag_and_a_pass_all_check_are_scored_by_failure_type():
    everything = audit.audit(lambda s: True, "latch", "calibration", limit=200)
    assert everything["bad_passed"]["rate"] == 1.0 and everything["good_passed"]["rate"] == 1.0
    flag = audit.audit(audit.BUILTIN["skill_flag"], "latch", "calibration", limit=200)
    assert flag["bad_passed_by_type"]["mouth only"]["k"] == 0                      # the skill knows it missed the mouth
    assert all(c["n"] > 0 for c in flag["bad_passed_by_type"].values())


def test_a_check_from_a_file_and_one_that_raises(tmp_path):
    path = tmp_path / "mine.py"
    path.write_text("def never(state):\n    return False\n\ndef broken(state):\n    raise RuntimeError('x')\n")
    never = audit.audit(audit.load_check(f"{path}:never"), "latch", "calibration", limit=50)
    assert never["bad_passed"]["k"] == 0 and never["good_passed"]["k"] == 0
    broken = audit.audit(audit.load_check(f"{path}:broken"), "latch", "calibration", limit=50)
    assert broken["check_errors"] == 50 and "raised" in audit.report(broken, "broken")


def test_a_test_split_is_refused():
    with pytest.raises(SystemExit):
        audit.main(["--check", "builtin:skill_flag", "--split", "test"])
