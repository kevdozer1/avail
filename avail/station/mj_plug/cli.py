"""``avail station``: the plug station's acceptance checks, the oracle cycle and the clips."""

from __future__ import annotations

import argparse
import json
import os


def _seed_range(text: str) -> list[int]:
    lo, hi = text.split(":")
    return list(range(int(lo), int(hi)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="avail station", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    acc = sub.add_parser("accept", help="run the acceptance checks of docs/02_plug_station.md on dev seeds")
    acc.add_argument("--connector", required=True)
    acc.add_argument("--seeds", default="200:250", type=_seed_range, help="dev seeds as low:high, high excluded")
    acc.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 6))
    acc.add_argument("--out", default=None, help="where to write the JSON (default AVAIL_DATA/runs/accept_<connector>.json)")

    ora = sub.add_parser("oracle", help="run 1 oracle plug cycle and print what truth says")
    ora.add_argument("--connector", required=True)
    ora.add_argument("--seed", type=int, default=0)

    vid = sub.add_parser("video", help="write the seated-versus-short clip for a connector (mp4)")
    vid.add_argument("--connector", required=True)
    vid.add_argument("--seed", type=int, default=300)

    args = parser.parse_args(argv)
    if args.command == "accept":
        from ...paths import check_disk_floor, runs_dir
        from .accept import run_acceptance, save, summarize

        check_disk_floor()
        res = run_acceptance(args.connector, args.seeds, workers=args.workers)
        path = args.out or str(runs_dir() / f"accept_{args.connector}.json")
        save(res, path)
        print(summarize(res), flush=True)
        print(f"\nwritten to {path}", flush=True)
        return 0
    if args.command == "oracle":
        from .oracle import oracle_cycle

        station, res = oracle_cycle(args.seed, args.connector)
        station.close()
        slim = {k: v for k, v in res.items() if k not in ("truth_held", "truth_released")}
        slim["held"] = res["truth_held"]["predicates"]
        slim["released"] = res["truth_released"]["predicates"] if res["truth_released"] else None
        print(json.dumps(slim, indent=1, default=str), flush=True)
        return 0
    if args.command == "video":
        from ...paths import check_disk_floor, videos_dir
        from .video import make_clip

        check_disk_floor()
        print(json.dumps(make_clip(args.connector, args.seed, videos_dir()), indent=1), flush=True)
        return 0
    return 2
