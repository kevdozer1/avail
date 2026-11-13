"""``avail zoo``: generate near-miss states and run the evidence sources over them.

    avail zoo --connector latch --split fit                 states for the whole fit range
    avail zoo --connector latch --split dev --seeds 1000:1100
    avail zoo features --connector latch --split fit        every free source on every state
    avail zoo status                                        what exists on disk
    avail zoo judges estimate --connector latch             what a VLM judge run would cost, before any call
    avail zoo look --connector latch --split calibration    each free source per class, harm, repair, worst force

A long run can be cut into short calls with ``--time-budget``. Blocks that are already on
disk are skipped, so the next call picks up where the last one stopped. The test split needs
``--final-run E1E2`` and is read once (CLAUDE.md rule 12).
"""

from __future__ import annotations

import argparse
import json
import os


def _seed_range(text: str) -> list[int]:
    lo, hi = text.split(":")
    return list(range(int(lo), int(hi)))


SPLIT_CHOICES = ["dev", "fit", "calibration", "test", "dev_hard", "fit_hard", "calibration_hard", "test_hard",
                 "dev_hard2", "fit_hard2", "calibration_hard2", "test_hard2"]


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--connector", required=True)
    p.add_argument("--split", required=True, choices=SPLIT_CHOICES,
                   help="a name ending in _hard builds hard mode (S4b), _hard2 hard mode v2 (S4c), each on its own seed range")
    p.add_argument("--seeds", type=_seed_range, default=None, help="low:high, high excluded. Required for the dev splits")
    p.add_argument("--n", type=int, default=None, help="use only the first n seeds of the range")
    p.add_argument("--workers", type=int, default=5,
                   help="5 is safe from a cold start on this machine: a worker commits about 2.2 GB (docs/11)")
    p.add_argument("--time-budget", type=float, default=None, help="seconds. Start no new block after this long")
    p.add_argument("--final-run", default=None,
                   help="E1bE2b unlocked the test_hard split, E1cE2c unlocks the test_hard2 split, once each")


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or [])
    command = "states"
    if argv and argv[0] == "judges":                       # VLM judges over stored frames: estimate first, then run
        from .judges import main as judges_main

        return int(judges_main(argv[1:]) or 0)
    if argv and argv[0] == "look":                         # the first look of the S3 report, from tables on disk
        from ..analysis.look import first_look, md_first_look, md_probes

        look_parser = argparse.ArgumentParser(prog="avail zoo look")
        look_parser.add_argument("--connector", required=True)
        look_parser.add_argument("--split", default="calibration", choices=["dev", "fit", "calibration"])
        look_args = look_parser.parse_args(argv[1:])
        result = first_look(look_args.connector, look_args.split)
        print(md_first_look(result), md_probes(result), sep="\n\n", flush=True)
        return 0
    if argv and argv[0] in ("features", "status", "states"):
        command, argv = argv[0], argv[1:]
    parser = argparse.ArgumentParser(prog=f"avail zoo{'' if command == 'states' else ' ' + command}", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    if command == "status":
        args = parser.parse_args(argv)
        return _status()
    _common(parser)
    args = parser.parse_args(argv)
    from ..paths import check_disk_floor
    from . import store

    check_disk_floor()
    seeds = args.seeds
    if args.split in ("dev", "dev_hard", "dev_hard2") and seeds is None:
        parser.error("the dev splits need --seeds low:high")
    if seeds is None:
        seeds = list(store.split_seeds(args.split))
    if args.n is not None:
        seeds = seeds[: args.n]
    if command == "states":
        from .generate import generate

        summary = generate(args.connector, args.split, seeds, workers=args.workers, time_budget_s=args.time_budget,
                           final_run=args.final_run)
    else:
        from .features import run_features

        summary = run_features(args.connector, args.split, seeds, workers=args.workers, time_budget_s=args.time_budget,
                               final_run=args.final_run)
    print(json.dumps(summary, indent=1, default=str), flush=True)
    return 0 if summary.get("blocks_left", 0) == 0 else 3


def _status() -> int:
    from ..paths import zoo_dir

    root = zoo_dir()
    rows = []
    for connector in sorted(p.name for p in root.iterdir() if p.is_dir()):
        for split in sorted(p.name for p in (root / connector).iterdir() if p.is_dir()):
            folder = root / connector / split
            rows.append((connector, split, len(list((folder / "states").glob("part_*.parquet"))),
                         len(list((folder / "features").glob("part_*.parquet"))),
                         (folder / "states.parquet").is_file(), (folder / "features.parquet").is_file()))
    print("connector  split        state parts  feature parts  states merged  features merged")
    for r in rows:
        print(f"{r[0]:<10} {r[1]:<12} {r[2]:>11}  {r[3]:>13}  {str(r[4]):>13}  {str(r[5]):>15}")
    return 0
