"""The versioned witness store.

A witness file is JSON: the ``Witness`` record (docs/01_architecture.md), the cascade with its
fused scorers as plain numbers, the per-source checks used by the evidence table, and where
the numbers came from. Files are never rewritten. A new calibration gets the next version.

- ``AVAIL_DATA/witness/<connector>/<predicate>/v<N>.json``  every version ever published
- ``avail/witness/shipped/<connector>.json``                the version the zero-key demo uses,
  committed with the repo and labeled sim
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from ..paths import REPO_ROOT, data_root
from .cascade import Cascade

SHIPPED_DIR = REPO_ROOT / "avail" / "witness" / "shipped"
_VERSION = re.compile(r"^v(\d+)\.json$")


def witness_dir(connector: str, predicate: str, root: Path | None = None) -> Path:
    base = Path(root) if root is not None else data_root(create=True) / "witness"
    path = base / connector / predicate
    path.mkdir(parents=True, exist_ok=True)
    return path


def versions(connector: str, predicate: str, root: Path | None = None) -> list[int]:
    out = []
    for p in witness_dir(connector, predicate, root).iterdir():
        m = _VERSION.match(p.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def build_payload(cascade: Cascade, checks: dict | None = None, provenance: dict | None = None) -> dict:
    """The JSON content of a witness file."""
    return {
        "kind": "avail.witness",
        "sim_or_real": "sim",
        "witness": cascade.to_witness().model_dump(mode="json"),
        "cascade": cascade.to_dict(),
        "checks": checks or {},
        "provenance": provenance or {},
    }


def publish(cascade: Cascade, checks: dict | None = None, provenance: dict | None = None,
            root: Path | None = None) -> tuple[Path, Cascade]:
    """Write the cascade as the next version. Returns the path and the cascade carrying its version."""
    existing = versions(cascade.connector, cascade.predicate, root)
    number = (existing[-1] + 1) if existing else 1
    cascade.version = f"v{number}"
    path = witness_dir(cascade.connector, cascade.predicate, root) / f"v{number}.json"
    if path.exists():
        raise FileExistsError(f"{path} already exists. Witness versions are never rewritten")
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(build_payload(cascade, checks, provenance), indent=1), encoding="utf-8", newline="\n")
    os.replace(tmp, path)
    return path, cascade


def load(connector: str, predicate: str, version: int | None = None, root: Path | None = None) -> tuple[Cascade, dict]:
    """A published cascade and its whole file. The latest version unless one is named."""
    have = versions(connector, predicate, root)
    if not have:
        raise FileNotFoundError(f"no witness published for {connector} {predicate}")
    number = have[-1] if version is None else int(version)
    path = witness_dir(connector, predicate, root) / f"v{number}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Cascade.from_dict(payload["cascade"]), payload


def ship(connector: str, payload: dict) -> Path:
    """Copy a witness file into the repo for the zero-key demo."""
    SHIPPED_DIR.mkdir(parents=True, exist_ok=True)
    path = SHIPPED_DIR / f"{connector}.json"
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8", newline="\n")
    return path


def load_shipped(connector: str) -> tuple[Cascade, dict]:
    path = SHIPPED_DIR / f"{connector}.json"
    if not path.is_file():
        raise FileNotFoundError(f"no shipped witness for {connector}. Run the E1 and E2 analysis, which ships one")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Cascade.from_dict(payload["cascade"]), payload
