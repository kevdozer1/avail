"""``avail s4c``: hard mode v2 tools for the S4c report. Evaluator-side.

    avail s4c tables --mode final        hard v1 and hard v2 results side by side, and what the click settles
    avail s4c judges                     the judges on hard v1 near misses, next to the tug and the camera
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    command, rest = args[0], args[1:]
    if command == "tables":
        from .summary_c import main as tables_main

        return tables_main(rest)
    if command == "judges":
        from .judges_c import main as judges_main

        return judges_main(rest)
    print(f"avail s4c: unknown command {command!r}\n{__doc__}", file=sys.stderr)
    return 2
