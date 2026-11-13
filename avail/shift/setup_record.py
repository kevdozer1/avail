"""What a final run's cycles must find unchanged, as file hashes. Core code.

The seed guard hashes config/experiment.yaml when an arm starts. A shift's cycles also depend on the task file, the
testbed's setup files (``setup_files()``), the code that runs and the config as it stays during the arm. ``avail shift
e3`` records all their hashes at E3's 1st run (``AVAIL_DATA/runs/shifts/e3_setup.json``) and compares them before each
arm, and the shift runner compares them before each cycle of a final run and again before the cycle's end is written,
so no cycle that ran under another setup is counted.
"""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

CODE_KEY = "code: avail/**/*.py"
PACKAGE = Path(__file__).resolve().parents[1]           # the avail package that is running


def _digest(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()[:12]


def code_fingerprint() -> str:
    """1 hash over every .py file of the running package, by path and content. Line endings do not change it."""
    h = hashlib.sha256()
    for f in sorted(p for p in PACKAGE.rglob("*.py") if "__pycache__" not in p.parts):
        h.update(f.relative_to(PACKAGE).as_posix().encode("utf-8") + b"\0")
        h.update(f.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()[:12]


def setup_hashes(task_name: str) -> dict[str, str]:
    """The task file's hash, config/experiment.yaml's, those of the testbed's setup files, and the code's. A file is
    named by its path in the repo, under ``AVAIL_DATA/`` for 1 in the data root, or under ``third_party/`` for 1 a
    worktree reaches through its link to the repo's third_party. Line endings do not change a hash."""
    from ..core.task import load_task, task_path
    from ..paths import REPO_ROOT, data_root, third_party_dir

    task = load_task(task_name)
    files = [task_path(task_name), REPO_ROOT / "config" / "experiment.yaml"]
    bed = importlib.import_module(task.testbed) if task.testbed else None
    files += list(getattr(bed, "setup_files", lambda: [])())
    # third_party last: a worktree links it to the repo's, and its files keep their name either way
    bases = ((REPO_ROOT.resolve(), ""), (data_root().resolve(), "AVAIL_DATA/"), (third_party_dir().resolve(), "third_party/"))
    out = {}
    for f in files:
        f = Path(f)
        key = f.as_posix()
        for base, prefix in bases:
            if f.resolve().is_relative_to(base):
                key = prefix + f.resolve().relative_to(base).as_posix()
                break
        out[key] = _digest(f.read_bytes())
    out[CODE_KEY] = code_fingerprint()
    return out


def changed_keys(then: dict, now: dict) -> list[str]:
    """The names whose hash differs between 2 records, a name in only 1 of them included."""
    return sorted(k for k in set(then) | set(now) if then.get(k) != now.get(k))
