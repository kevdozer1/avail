"""1-command setup: clone ASPIRE at the pinned commit into third_party/ASPIRE if it is missing, and check it.

    uv run python scripts/setup_aspire.py

AVAIL's plug and cable station loads ASPIRE's dual-YAM station model (MJCF and meshes) from that clone at run time.
Nothing of ASPIRE is copied into this repo. The clone is gitignored.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
URL = "https://github.com/NVlabs/ASPIRE"
COMMIT = "fa00f0ef2bc1d91d594b9aa502c22133c9f538f7"
TARGET = REPO / "third_party" / "ASPIRE"
NEEDED = ("src/cap_harness/yam_real/description/station/station.xml",
          "src/cap_harness/configs/yam_real/yam-example/station.yaml")


def git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def main() -> int:
    if not TARGET.exists():
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        print(f"cloning {URL} into {TARGET}", flush=True)
        git("clone", "--quiet", URL, str(TARGET))
        git("checkout", "--quiet", COMMIT, cwd=TARGET)
    head = git("rev-parse", "HEAD", cwd=TARGET)
    if head != COMMIT:
        print(f"third_party/ASPIRE is at {head[:7]}, and AVAIL is pinned to {COMMIT[:7]}. Check it out with:\n"
              f"  git -C third_party/ASPIRE checkout {COMMIT}", file=sys.stderr)
        return 1
    missing = [p for p in NEEDED if not (TARGET / p).is_file()]
    if missing:
        print(f"third_party/ASPIRE lacks {', '.join(missing)}", file=sys.stderr)
        return 1
    env = REPO / ".env"
    if not env.exists():
        env.write_text((REPO / ".env.example").read_text(encoding="utf-8"), encoding="utf-8")
        print("wrote .env from .env.example", flush=True)
    print(f"ok: ASPIRE {COMMIT[:7]} in third_party/ASPIRE. Next: uv run avail demo", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
