"""Paths and the disk floor. Every large output goes under ``AVAIL_DATA``.

No data-drive path is written literally in this package. The data root comes
from, in order: the process environment, the repo ``.env``, then the documented
default in the committed ``.env.example``. Everything else asks this module for
its directory, so moving the data root is one line in ``.env``.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ENV_VAR = "AVAIL_DATA"
DISK_FLOOR_GB = 10.0
DATA_SUBDIRS = ("runs", "zoo", "videos", "cache")

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")


class DiskFloorError(RuntimeError):
    """Free space on a drive AVAIL writes to fell below the floor."""


def _documented_default() -> str | None:
    """``AVAIL_DATA`` as written in ``.env.example`` (the documented default)."""
    example = REPO_ROOT / ".env.example"
    if not example.is_file():
        return None
    for line in example.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line.startswith(DATA_ENV_VAR + "="):
            value = line.split("=", 1)[1].strip().strip('"').strip("'")
            return value or None
    return None


def data_root(create: bool = False) -> Path:
    """The external data root (``AVAIL_DATA``)."""
    from .config import env

    raw = env(DATA_ENV_VAR) or _documented_default()
    if not raw:
        raise RuntimeError(
            f"{DATA_ENV_VAR} is not set. Add {DATA_ENV_VAR}=... to {REPO_ROOT / '.env'} or the environment."
        )
    root = Path(raw.strip())
    if not root.is_absolute():                      # a relative AVAIL_DATA is read from the repo, not the shell's folder
        root = REPO_ROOT / root
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def _sub(name: str, create: bool = True) -> Path:
    path = data_root() / name
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def runs_dir(create: bool = True) -> Path:
    return _sub("runs", create)


def zoo_dir(create: bool = True) -> Path:
    return _sub("zoo", create)


def videos_dir(create: bool = True) -> Path:
    return _sub("videos", create)


def cache_dir(create: bool = True) -> Path:
    return _sub("cache", create)


def check_run_id(run_id: str) -> str:
    """A run id is a short file-safe name. ValueError otherwise."""
    if not isinstance(run_id, str) or not _RUN_ID.match(run_id) or run_id.endswith("."):
        raise ValueError(
            f"run id {run_id!r} is not usable as a folder name. Use letters, digits, '.', '_' or '-', "
            "start with a letter or digit, at most 100 characters."
        )
    return run_id


def run_dir(run_id: str, create: bool = True) -> Path:
    """``AVAIL_DATA/runs/<run_id>``: one directory per ledger run."""
    check_run_id(run_id)
    path = runs_dir(create) / run_id
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def spend_ledger_path() -> Path:
    """The spend ledger, shared by every session and every paid run."""
    return data_root(create=True) / "spend_ledger.jsonl"


def seed_reads_path() -> Path:
    """The record of every read of a test seed range (CLAUDE.md rule 12)."""
    return data_root(create=True) / "test_seed_reads.jsonl"


def ensure_data_dirs() -> list[Path]:
    """Create ``AVAIL_DATA`` and its standard subfolders."""
    root = data_root(create=True)
    made = []
    for name in DATA_SUBDIRS:
        path = root / name
        path.mkdir(parents=True, exist_ok=True)
        made.append(path)
    return made


def third_party_dir() -> Path:
    """Untrusted reference clones (gitignored)."""
    return REPO_ROOT / "third_party"


def aspire_dir() -> Path:
    return third_party_dir() / "ASPIRE"


def aspire_station_xml() -> Path:
    """The dual-YAM station model, read from its own folder so its mesh paths resolve."""
    return aspire_dir() / "src" / "cap_harness" / "yam_real" / "description" / "station" / "station.xml"


def aspire_station_profile() -> Path:
    """The example station profile that holds ``ready_joints``."""
    return aspire_dir() / "src" / "cap_harness" / "configs" / "yam_real" / "yam-example" / "station.yaml"


def free_space_gb(paths: list[Path] | None = None) -> dict[str, float]:
    """Free gigabytes on the drives holding the repo and the data root."""
    if paths is None:
        paths = [REPO_ROOT]
        try:
            paths.append(data_root())
        except RuntimeError:
            pass
    out: dict[str, float] = {}
    for path in paths:
        anchor = Path(path).resolve().anchor
        if not anchor or anchor in out or not Path(anchor).exists():
            continue
        out[anchor] = round(shutil.disk_usage(anchor).free / 1024**3, 2)
    return out


def check_disk_floor(min_gb: float = DISK_FLOOR_GB, paths: list[Path] | None = None) -> dict[str, float]:
    """Free space per drive. Raises DiskFloorError when any drive is under ``min_gb``."""
    free = free_space_gb(paths)
    low = {drive: gb for drive, gb in free.items() if gb < float(min_gb)}
    if low:
        detail = ", ".join(f"{d} {gb:.1f} GB" for d, gb in sorted(low.items()))
        raise DiskFloorError(f"disk floor {min_gb:.0f} GB crossed: {detail} free. Stop and report.")
    return free
