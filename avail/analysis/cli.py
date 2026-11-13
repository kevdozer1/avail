"""``avail e1e2``: the evidence table (E1) and the bound (E2) from zoo tables on disk.

    avail e1e2 --mode rehearsal --connector latch     calibration stands in for test
    avail e1e2 --mode final                           the reported numbers, on test features already on disk

This command never simulates and never reads a seed. The single read of the test seeds is
``avail zoo features --split test --final-run E1E2``.
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="avail e1e2", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", required=True, choices=["rehearsal", "final"])
    parser.add_argument("--connector", action="append", default=None, help="repeat for several. Default: fric and latch")
    parser.add_argument("--profile", default="easy", choices=["easy", "hard", "hard2"],
                        help="hard runs E1b and E2b (S4b) on the hard-mode splits, hard2 runs E1c and E2c (S4c) on v2's")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--republish", action="store_true",
                        help="a final run that already shipped its witness, again: recompute, publish a new version and ship it")
    args = parser.parse_args(argv)
    from ..paths import REPO_ROOT, runs_dir
    from ..witness.store import SHIPPED_DIR
    from . import run, summary

    connectors = args.connector or ["fric", "latch"]
    shipped_name = {"easy": "{c}.json", "hard": "{c}_hard.json", "hard2": "{c}_hard2.json"}[args.profile]
    if args.mode == "final" and not args.republish and any((SHIPPED_DIR / shipped_name.format(c=c)).is_file() for c in connectors):
        parser.error(f"the {args.profile} final run is done and its witness is shipped. Running it again republishes and "
                     "reships it. Pass --republish if that is really wanted")
    out_dir = runs_dir() / f"{run.RESULT_FOLDER[args.profile]}_{args.mode}"
    results = {}
    for connector in connectors:
        result = run.run_connector(connector, args.mode, profile=args.profile)
        results[connector] = result
        path = run.save_result(result, out_dir)
        print(summary.text_summary(result), flush=True)
        print(f"written to {path}\n", flush=True)
    if not args.no_figures:
        img = {"easy": "s4", "hard": "s4b", "hard2": "s4c"}[args.profile]
        fig_dir = (REPO_ROOT / "docs" / "img" / img) if args.mode == "final" else (out_dir / "figures")
        for path in summary.figures(results, fig_dir, args.mode, profile=args.profile):
            print(f"figure {path}", flush=True)
    if args.mode == "final":
        for line in summary.publish(results, profile=args.profile):
            print(line, flush=True)
    return 0
