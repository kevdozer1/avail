"""``avail s4b``: hard-mode tools for the S4b report. Dev seeds only. Evaluator-side.

    avail s4b camera --connector latch --seeds 6000:6040     gap reading error, easy against hard, and the camera's numbers
    avail s4b pairs                                           picture pairs: each near miss, easy and hard, same scene
    avail s4b tables --mode final                             easy and hard results side by side, as markdown
    avail s4b judges                                          the judges on hard near misses, next to camera and probes
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    command, rest = args[0], args[1:]
    if command == "camera":
        from .camera_b import main as camera_main

        return camera_main(rest)
    if command == "pairs":
        from .pairs_b import main as pairs_main

        return pairs_main(rest)
    if command == "tables":
        from .summary_b import main as tables_main

        return tables_main(rest)
    if command == "judges":
        from .judges_b import main as judges_main

        return judges_main(rest)
    print(f"avail s4b: unknown command {command!r}\n{__doc__}", file=sys.stderr)
    return 2
