"""The docs follow the project's writing rules.

Checked here, mechanically, on README.md and every page under docs/ and reports/:
no em dashes, no semicolons, and none of 3 banned claims. Code spans and fenced blocks
are skipped. A private name from a local context/ list in a tracked file fails the suite.
"""

from __future__ import annotations

import re
import subprocess

import pytest

from avail.paths import REPO_ROOT

REPORTS = sorted([*REPO_ROOT.glob("README.md"), *(REPO_ROOT / "docs").glob("*.md"),
                  *(REPO_ROOT / "reports").glob("*.md")])
BANNED = re.compile(r"\b(first|novel|nobody has)\b", re.IGNORECASE)


def _prose(text: str) -> list[tuple[int, str]]:
    """Lines of prose with code spans and fenced blocks removed."""
    out, fenced = [], False
    for n, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        out.append((n, re.sub(r"`[^`]*`", "", line)))
    return out


@pytest.mark.parametrize("path", REPORTS, ids=lambda p: p.name)
def test_report_has_no_em_dash_no_semicolon_and_no_banned_claim(path):
    problems = []
    for n, line in _prose(path.read_text(encoding="utf-8")):
        if "—" in line:
            problems.append(f"{path.name}:{n}: em dash")
        if ";" in line:
            problems.append(f"{path.name}:{n}: semicolon")
        for m in BANNED.finditer(line):
            problems.append(f"{path.name}:{n}: the word {m.group(0)!r}")
    assert problems == []


@pytest.mark.parametrize("path", [p for p in REPORTS if re.fullmatch(r"S\d+[a-z]?\.md", p.name)], ids=lambda p: p.name)
def test_session_report_has_the_required_parts(path):
    text = path.read_text(encoding="utf-8")
    assert re.search(r"^Level 1:", text, re.MULTILINE), "a report opens with a Level 1 line"
    assert "What this does not show" in text
    assert "next action" in text


def test_no_private_name_is_tracked():
    """context/README.md lists names that must never be committed. Skipped on a clone without context/."""
    listing = REPO_ROOT / "context" / "README.md"
    if not listing.is_file():
        pytest.skip("no context/ folder on this machine")
    names = []
    for line in listing.read_text(encoding="utf-8").splitlines():
        if line.startswith("`") and "," in line:
            names += [part.strip().strip("`") for part in line.split(",")]
    assert len(names) >= 5, "the list of private names was not found where it used to be"
    pattern = re.compile("|".join(re.escape(n) for n in names if n), re.IGNORECASE)
    # tracked files and files that the next commit would add
    tracked = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=str(REPO_ROOT),
                             capture_output=True, text=True, timeout=60).stdout.splitlines()
    hits = []
    for rel in tracked:
        if not rel:
            continue
        path = REPO_ROOT / rel
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if pattern.search(text):
            hits.append(rel)
    assert hits == []
