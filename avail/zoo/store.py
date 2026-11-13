"""Where the zoo lives on disk, and how its parts are written so a run can stop and resume.

Layout under ``AVAIL_DATA/zoo/<connector>/<split>/``:

- ``states/part_<lo>_<hi>.parquet``     one row per seed: the injected class, truth, what the skill reported
- ``frames/frames_<lo>_<hi>.zip``       the claim-time camera frames, JPEG quality 92, stored uncompressed
- ``features/part_<lo>_<hi>.parquet``   one row per seed: every source's features (S3)
- ``states.parquet``, ``features.parquet``  the parts merged
- ``manifest.json``                     the config hash, the counts and the class mix

A part is written to a temporary name and renamed, so a killed run never leaves half a part.
The data drive is exFAT with 512 KB clusters, which is why frames go into a few zip files.
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np

from ..config import load_experiment
from ..paths import zoo_dir

BLOCK = 50
SPLITS = {"fit": "zoo_fit", "calibration": "zoo_calibration", "test": "zoo_test", "dev": "dev",
          "fit_hard": "zoo_fit_hard", "calibration_hard": "zoo_calibration_hard", "test_hard": "zoo_test_hard",
          "dev_hard": "dev",
          "fit_hard2": "zoo_fit_hard2", "calibration_hard2": "zoo_calibration_hard2", "test_hard2": "zoo_test_hard2",
          "dev_hard2": "dev",
          "cable_fit": "cable_fit", "cable_calibration": "cable_calibration"}      # S5b: cable-loop insertions
JPEG_QUALITY = 92


def check_seeds_in_split(seeds, split: str) -> None:
    """Refuse seeds outside a named split's range, so no dev state lands in a fit, calibration or test table.
    The dev splits take any dev seed."""
    from ..seeds import TestSeedError

    span = split_seeds(split)
    outside = [int(s) for s in seeds if int(s) not in span]
    if outside:
        raise TestSeedError(f"{len(outside)} seeds are outside the {split} split ({span.start} to {span.stop - 1}), "
                            f"for example {outside[:5]}")


def profile_of_split(split: str) -> str:
    """``hard`` for the S4b splits (names ending in ``_hard``), ``hard2`` for the S4c splits
    (``_hard2``), ``easy`` for the rest."""
    if split not in SPLITS:
        raise KeyError(f"unknown split {split!r}. Known: {', '.join(SPLITS)}")
    if split.endswith("_hard2"):
        return "hard2"
    return "hard" if split.endswith("_hard") else "easy"


def split_seeds(split: str) -> range:
    """The seeds of a zoo split. ``dev`` is the whole dev range and callers take a slice of it."""
    if split not in SPLITS:
        raise KeyError(f"unknown split {split!r}. Known: {', '.join(SPLITS)}")
    return load_experiment().seeds.seeds(SPLITS[split])


def split_dir(connector: str, split: str, root: Path | None = None) -> Path:
    base = Path(root) if root is not None else zoo_dir()
    path = base / connector / split
    path.mkdir(parents=True, exist_ok=True)
    return path


def blocks(seeds: list[int], block: int = BLOCK) -> list[list[int]]:
    """Seeds grouped into aligned blocks, so a resumed run meets the same part names."""
    out: dict[int, list[int]] = {}
    for seed in sorted(seeds):
        out.setdefault(seed // block, []).append(seed)
    return [out[k] for k in sorted(out)]


def part_name(block_seeds: list[int], block: int = BLOCK) -> str:
    lo = (block_seeds[0] // block) * block
    return f"part_{lo:06d}_{lo + block - 1:06d}"


def part_path(connector: str, split: str, kind: str, block_seeds: list[int], root: Path | None = None) -> Path:
    folder = split_dir(connector, split, root) / kind
    folder.mkdir(parents=True, exist_ok=True)
    suffix = ".zip" if kind == "frames" else ".parquet"
    name = part_name(block_seeds).replace("part_", "frames_") if kind == "frames" else part_name(block_seeds)
    return folder / (name + suffix)


def write_table_atomic(rows: list[dict], path: Path) -> None:
    import pandas as pd

    tmp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    os.replace(tmp, path)


class FrameZip:
    """Collects JPEG frames for a block and writes them as one stored zip."""

    def __init__(self, path: Path):
        self.path = path
        self._tmp = path.with_suffix(path.suffix + ".tmp")
        self._zip = zipfile.ZipFile(self._tmp, "w", compression=zipfile.ZIP_STORED)

    def add(self, name: str, rgb: np.ndarray) -> str:
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=JPEG_QUALITY)
        self._zip.writestr(name, buf.getvalue())
        return name

    def close(self) -> None:
        self._zip.close()
        os.replace(self._tmp, self.path)


def read_frame(connector: str, split: str, seed: int, name: str, root: Path | None = None) -> np.ndarray:
    """One stored frame as an RGB array. ``name`` is the entry name recorded in the states table."""
    from PIL import Image

    path = part_path(connector, split, "frames", [seed], root)
    with zipfile.ZipFile(path) as zf:
        return np.asarray(Image.open(io.BytesIO(zf.read(name))).convert("RGB"))


def merge(connector: str, split: str, kind: str, root: Path | None = None):
    """Concatenate a kind's parts into ``<kind>.parquet`` and return the frame."""
    import pandas as pd

    folder = split_dir(connector, split, root)
    parts = sorted((folder / kind).glob("part_*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no {kind} parts under {folder / kind}")
    frame = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).sort_values("seed").reset_index(drop=True)
    if frame["seed"].duplicated().any():
        raise RuntimeError(f"duplicate seeds in the {kind} parts under {folder}")
    tmp = folder / f"{kind}.parquet.tmp"
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, folder / f"{kind}.parquet")
    return frame


def load(connector: str, split: str, kind: str, root: Path | None = None):
    import pandas as pd

    path = split_dir(connector, split, root) / f"{kind}.parquet"
    if not path.is_file():
        return merge(connector, split, kind, root)
    return pd.read_parquet(path)


def write_manifest(connector: str, split: str, extra: dict, root: Path | None = None) -> Path:
    from ..config import config_hash

    path = split_dir(connector, split, root) / "manifest.json"
    current = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    current.update({"connector": connector, "split": split, "config_hash": config_hash(), **extra})
    path.write_text(json.dumps(current, indent=1, default=str), encoding="utf-8", newline="\n")
    return path
