"""JSON Schema export: ``avail schemas [--out DIR]``.

Writes 1 ``<Name>.schema.json`` file per record type. The default folder is
``AVAIL_DATA/cache/schemas``, so nothing lands in the repo.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .records import RECORD_MODELS

__all__ = ["JSON_SCHEMA_DIALECT", "json_schemas", "main", "write_schemas"]

JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
FILE_SUFFIX = ".schema.json"


def json_schemas() -> dict[str, dict[str, Any]]:
    """The JSON Schema of every record type, by record name."""
    return {
        name: {"$schema": JSON_SCHEMA_DIALECT, **model.model_json_schema()}
        for name, model in RECORD_MODELS.items()
    }


def write_schemas(out_dir: str | Path) -> list[Path]:
    """Write 1 ``<Name>.schema.json`` per record type into ``out_dir`` and return the paths."""
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for name, schema in json_schemas().items():
        path = folder / f"{name}{FILE_SUFFIX}"
        path.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    """The ``avail schemas`` command. Prints the files it wrote, 1 per line."""
    parser = argparse.ArgumentParser(
        prog="avail schemas",
        description="Write the JSON Schema of every record type, 1 file per type.",
    )
    parser.add_argument("--out", default=None, metavar="DIR",
                        help="the folder to write into (default: AVAIL_DATA/cache/schemas)")
    args = parser.parse_args(argv)
    if args.out is not None:
        out_dir = Path(args.out)
    else:
        from ..paths import cache_dir  # imported here so the record types never need the data root

        try:
            out_dir = cache_dir() / "schemas"
        except RuntimeError as exc:
            print(f"avail schemas: {exc} Or pass --out DIR.", file=sys.stderr)
            return 2
    for path in write_schemas(out_dir):
        print(path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
