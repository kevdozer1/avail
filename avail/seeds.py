"""The seed budget, enforced in code (CLAUDE.md rule 12).

Seed ranges are fixed in ``config/experiment.yaml``. A test range is read once,
by the final run of its experiment. Anything that turns a seed into a simulated
state calls :func:`guard` first. A test seed is refused unless the caller names
the final run, and every allowed test read is appended to
``AVAIL_DATA/test_seed_reads.jsonl`` so a report can state exactly what was read.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .config import TEST_SPLITS, load_experiment
from .paths import seed_reads_path

# which final run may read which test split
FINAL_RUNS = {
    "E1E2": "zoo_test",
    "E3": "shift_test_e3",
    "E4": "shift_test_e4",
    "E7": "shift_test_e7",
    "E1bE2b": "zoo_test_hard",      # S4b: E1 and E2 on hard mode
    "E1cE2c": "zoo_test_hard2",     # S4c: E1 and E2 on hard mode v2
    "FIG": "fig_test",              # S5c: the README figure, 100 cable-loop cycles per arm
    "E3b": "shift_test_e3b",        # the 1 retest of the defect bar after the release fix (H3b)
}


# final runs whose test split has been read and must never be read again
RETIRED_FINAL_RUNS = {
    "E1E2": "it read 16,000 to 17,999 for S4 on 2026-11-11, and Kevin's S4b brief says not to touch that range again",
    "E1bE2b": "it read 64,000 to 65,999 for S4b on 2026-11-11, and the S4c judges read its stored frames on 2026-11-12, "
              "the last read its notes planned",
    "E1cE2c": "it read 70,000 to 71,999 for S4c on 2026-11-12. Its analysis reads the features on disk and needs no read",
    "E3": "it read 30,000 to 34,999 on latch from 2026-11-12 19:42 to 2026-11-13 06:04 and failed H3's defect bar. "
          "Its 1 retest is E3b on new seeds",
    "FIG": "it read 72,000 to 72,099 on 2026-11-13 for the README figure, and the figure's re-render read 1 stored state",
    "E3b": "it read 80,000 to 84,999 on latch on 2026-11-14 and passed H3b (2 of 5,000). It was the only retest",
}


class TestSeedError(RuntimeError):
    """A test seed was requested outside the final run of its experiment."""

    __test__ = False  # not a pytest class


def _commit() -> str | None:
    """The repo's current commit, short, or None when git cannot tell."""
    import subprocess

    from .paths import REPO_ROOT

    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO_ROOT), capture_output=True,
                             text=True, timeout=20)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def split_of(seed: int) -> str | None:
    """The name of the split a seed belongs to, or None when it is outside every range."""
    return load_experiment().seeds.split_of(int(seed))


def seeds_for(split: str) -> range:
    return load_experiment().seeds.seeds(split)


def is_test_seed(seed: int) -> bool:
    return split_of(seed) in TEST_SPLITS


def guard(seeds: Iterable[int], *, purpose: str, final_run: str | None = None,
          record_path: Path | None = None, stored: bool = False) -> None:
    """Refuse test seeds unless this is the named final run. Record every test read.

    ``purpose`` is a short plain description that ends up in the record.
    ``final_run`` is one of ``E1E2``, ``E3``, ``E4``, ``E7``. It unlocks only its own split.
    Seeds outside every configured range are refused too, so a typo cannot wander.

    ``stored`` marks a read of what the final run already stored (frames, tables), with nothing
    simulated, such as the judge run its dated note planned for after the read. It still needs the
    final run's name and is still recorded, with ``stored`` true. It skips only the refusal of a
    changed config, which exists to stop a split from being simulated again under another setup.
    Nothing here lets a test seed be simulated: :func:`assert_simulable` decides that.
    """
    budget = load_experiment().seeds
    touched: dict[str, list[int]] = {}
    for seed in seeds:
        seed = int(seed)
        split = budget.split_of(seed)
        if split is None:
            raise TestSeedError(f"seed {seed} is outside every range in config/experiment.yaml (purpose: {purpose})")
        if split in TEST_SPLITS:
            touched.setdefault(split, []).append(seed)
    if not touched:
        return
    if final_run in RETIRED_FINAL_RUNS:
        raise TestSeedError(f"the final run {final_run} is retired: {RETIRED_FINAL_RUNS[final_run]} (purpose: {purpose})")
    allowed = FINAL_RUNS.get(final_run or "")
    for split, hit in touched.items():
        if split != allowed:
            raise TestSeedError(
                f"{len(hit)} seeds in the test split {split} were requested for: {purpose}. "
                f"Test seeds are read once, by the final run of their experiment. "
                f"Use dev, fit or calibration seeds, or pass the final-run flag for that experiment."
            )
    from .config import config_hash

    path = Path(record_path) if record_path is not None else seed_reads_path()
    current = config_hash()
    # A resumed final run reads its split again under the same config. A changed config may not.
    for split in touched:
        if stored:
            break
        earlier = {r.get("config_hash") for r in test_seed_reads(path) if r.get("split") == split and r.get("config_hash")}
        if earlier - {current}:
            raise TestSeedError(
                f"the test split {split} was already read under config {', '.join(sorted(earlier - {current}))} and the "
                f"config is now {current}. A changed setup needs new, unread test seeds (purpose: {purpose})")
    path.parent.mkdir(parents=True, exist_ok=True)
    commit = _commit()
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        for split, hit in touched.items():
            fh.write(json.dumps({
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "split": split,
                "final_run": final_run,
                "purpose": purpose,
                "count": len(hit),
                "min": min(hit),
                "max": max(hit),
                "pid": os.getpid(),
                "config_hash": current,
                "commit": commit,
                **({"stored": True} if stored else {}),
            }) + "\n")


def config_drift(final_run: str, record_path: Path | None = None) -> tuple[list[str], str] | None:
    """The config hashes a final run's split was read under and the config's hash now, when they differ, else None.
    A read of that split under the config as it is now would be refused by :func:`guard`."""
    from .config import config_hash

    split = FINAL_RUNS[final_run]
    path = Path(record_path) if record_path is not None else seed_reads_path()
    then = {r.get("config_hash") for r in test_seed_reads(path) if r.get("split") == split and r.get("config_hash")}
    now = config_hash()
    return (sorted(then), now) if then - {now} else None


FINAL_RUN_ENV = "AVAIL_FINAL_RUN"


def unlock_final_run(final_run: str | None) -> None:
    """Tell this process and the workers it starts that a final run is in progress.

    The command that was given ``--final-run`` calls this once. Worker processes inherit it.
    """
    if final_run is None:
        os.environ.pop(FINAL_RUN_ENV, None)
        return
    if final_run not in FINAL_RUNS:
        raise TestSeedError(f"unknown final run {final_run!r}. Known: {', '.join(FINAL_RUNS)}")
    if final_run in RETIRED_FINAL_RUNS:
        raise TestSeedError(f"the final run {final_run} is retired: {RETIRED_FINAL_RUNS[final_run]}")
    os.environ[FINAL_RUN_ENV] = final_run


def assert_simulable(seed: int) -> None:
    """The last line of defense, called wherever a seed becomes a simulated state.

    Dev, fit and calibration seeds pass. A test seed passes only inside the final run of its
    experiment. A seed outside every range never passes. Nothing is recorded here: the
    command that runs the experiment records the read once, through :func:`guard`.
    """
    split = load_experiment().seeds.split_of(int(seed))
    if split is None:
        raise TestSeedError(f"seed {seed} is outside every range in config/experiment.yaml")
    unlocked = os.environ.get(FINAL_RUN_ENV, "")
    if split in TEST_SPLITS and (unlocked in RETIRED_FINAL_RUNS or FINAL_RUNS.get(unlocked) != split):
        raise TestSeedError(
            f"seed {seed} is in the test split {split}. Test seeds are read once, by the final run of their "
            f"experiment. Nothing may simulate it before that."
        )


def profile_of_seed(seed: int) -> str | None:
    """The station profile a seed's range belongs to: ``hard`` for the S4b ranges, ``hard2`` for the
    S4c ranges, ``easy`` for the other named ranges, None for dev seeds, which may be built any way."""
    split = load_experiment().seeds.split_of(int(seed))
    if split is None or split == "dev":
        return None
    if split.endswith("_hard2"):
        return "hard2"
    return "hard" if split.endswith("_hard") else "easy"


def assert_profile(seed: int, profile: str) -> None:
    """A seed of a hard range is only ever built in hard mode, and a seed of an easy range only in easy mode."""
    wanted = profile_of_seed(seed)
    if wanted is not None and wanted != profile:
        raise TestSeedError(f"seed {seed} belongs to a {wanted}-mode range and was asked for in {profile} mode")


def test_seed_reads(record_path: Path | None = None) -> list[dict]:
    """Every recorded read of a test split, oldest to newest."""
    path = Path(record_path) if record_path is not None else seed_reads_path()
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


test_seed_reads.__test__ = False  # not a pytest function
