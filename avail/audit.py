"""``avail audit-check``: how often your success check passes bad states, by failure type, on the near-miss zoo.

    uv run avail audit-check --check builtin:skill_flag --connector latch
    uv run avail audit-check --check my_checks.py:plug_is_in --connector latch --split calibration

A check is a Python function ``check(state) -> bool``: True means "this step is done". It is run on every stored state
of a zoo split: plugs pushed into a port by the plug station with a failure injected (0.5 to 4 mm short, mouth only,
unlatched, the wrong port, a tilted jam, dropped) or none. The function sees what a robot would have at the moment the
skill says done, and never the truth:

- ``state.frames``: the claim-time RGB images, ``{camera name: HxWx3 uint8}``, loaded when first read
- ``state.cameras``: each camera's intrinsics ``K`` and extrinsics ``T`` (camera to world, 4x4)
- ``state.skill``: what the insertion skill reported, ``flag`` (its own success flag), ``stopped_by``, ``travel_mm``,
  ``expected_travel_mm``, ``force_at_stop_n``
- ``state.port``: the target port's nominal position, ``x``, ``y``, ``z`` (m) and ``yaw`` (rad)
- ``state.features``: AVAIL's own checks on the same state, ``{"<source>/<feature>": float}``

The report gives, for each failure type, the share of bad states the check passed, with n and a 95% Wilson
interval, and the share of good states it passed. These are the evaluator's numbers: the truth used to score the
check comes from the simulator and never reaches it. This is the measured version of EmbodiedSWE's rubric tests:
there, hidden rubrics test whether a coding agent's grader can be fooled, and here the zoo measures how often a
success check is fooled, per failure type, with a bound on the rate.

The default split is ``calibration``, which is no test split. Every number is simulation.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from functools import cached_property
from pathlib import Path
from typing import Any, Callable

FAILURE_NAMES = {"P0": "good", "N1": "0.5 to 1 mm short", "N2": "1 to 2 mm short", "N3": "2 to 4 mm short",
                 "N4": "mouth only", "N5": "unlatched", "N6": "wrong port", "N7": "tilted jam", "N8": "dropped"}


class ZooState:
    """What a robot has at the claim. Built from the zoo's tables with every truth column left out."""

    def __init__(self, row: dict, feats: dict, connector: str, split: str):
        self._row, self._connector, self._split = row, connector, split
        self.seed = int(row["seed"])
        self.skill = {"flag": bool(row["skill_flag"]), "stopped_by": row.get("insert_stopped_by"),
                      "travel_mm": row.get("insert_travel_mm"), "expected_travel_mm": row.get("expected_travel_mm"),
                      "force_at_stop_n": row.get("insert_force_at_stop_n")}
        self.port = {"x": row.get("nominal_port_x"), "y": row.get("nominal_port_y"), "z": row.get("nominal_port_z"),
                     "yaw": row.get("nominal_board_yaw")}
        self.cameras = {k[len("calib_"):]: json.loads(v) for k, v in row.items() if k.startswith("calib_") and v}
        self.features = {k[2:]: float(v) for k, v in feats.items()
                         if k[:2] == "A/" and "/_" not in k and isinstance(v, (int, float, bool)) and v == v}

    @cached_property
    def frames(self) -> dict:
        from .zoo import store

        out = {}
        for key, name in self._row.items():
            if key.startswith("frame_") and isinstance(name, str):
                out[key[len("frame_"):]] = store.read_frame(self._connector, self._split, self.seed, name)
        return out


def _wilson(k: int, n: int) -> dict:
    from .stats import wilson

    lo, hi = wilson(k, n) if n else (float("nan"), float("nan"))
    return {"k": int(k), "n": int(n), "rate": k / n if n else float("nan"), "lo": lo, "hi": hi}


def audit(check: Callable[[ZooState], Any], connector: str = "latch", split: str = "calibration",
          limit: int | None = None) -> dict:
    """Run ``check`` on every state of a zoo split and score it against the simulator's truth."""
    from .zoo import store

    from .analysis.run import primary_predicate

    states = store.load(connector, split, "states")
    states = states[states["claim_predicate"] == primary_predicate(connector)]      # the insertion claim only
    feats = store.load(connector, split, "features").set_index("seed")
    if limit:
        states = states.iloc[:limit]
    by_type: dict[str, list[int]] = {}
    good = [0, 0]
    errors = 0
    for row in states.to_dict("records"):
        f = feats.loc[row["seed"]].to_dict() if row["seed"] in feats.index else {}
        try:
            passed = bool(check(ZooState(row, f, connector, split)))
        except Exception:
            errors += 1
            passed = False                                  # a check that raises is counted as not passing
        if bool(row["label"]):
            good[0] += passed
            good[1] += 1
        else:
            kind = FAILURE_NAMES.get(row["cls"], str(row["cls"]))      # the injected failure, as E1 groups them
            hit = by_type.setdefault(kind, [0, 0])
            hit[0] += passed
            hit[1] += 1
    bad_k, bad_n = sum(v[0] for v in by_type.values()), sum(v[1] for v in by_type.values())
    return {"connector": connector, "split": split, "sim_or_real": "sim", "states": int(len(states)),
            "check_errors": errors, "bad_passed": _wilson(bad_k, bad_n), "good_passed": _wilson(*good),
            "bad_passed_by_type": {k: _wilson(*v) for k, v in sorted(by_type.items(), key=lambda kv: -kv[1][1])}}


def _pct(c: dict) -> str:
    if not c["n"]:
        return "none"
    return f"{100 * c['rate']:.1f}% ({c['k']} of {c['n']}, {100 * c['lo']:.1f} to {100 * c['hi']:.1f})"


def report(result: dict, name: str) -> str:
    lines = [f"Check {name} on the {result['connector']} zoo, {result['split']} split, sim. {result['states']} insertion "
             "claims.",
             "", "| Bad states, by failure type | Passed by the check |", "|---|---|"]
    lines += [f"| {kind} | {_pct(c)} |" for kind, c in result["bad_passed_by_type"].items()]
    lines += [f"| all bad states | {_pct(result['bad_passed'])} |", f"| good states | {_pct(result['good_passed'])} |"]
    if result["check_errors"]:
        lines.append(f"\nThe check raised on {result['check_errors']} states, counted as not passing.")
    return "\n".join(lines)


BUILTIN = {
    "skill_flag": lambda s: s.skill["flag"],
    "camera": lambda s: s.features.get("vision.gap@claim/plug_found", 0) >= 0.5
    and s.features.get("vision.gap@claim/gap_mm_est", 99.0) <= 0.5,
}


def load_check(spec: str) -> Callable[[ZooState], Any]:
    """``builtin:<name>``, ``path/to/file.py:function`` or ``package.module:function``."""
    where, _, func = spec.rpartition(":")
    if where == "builtin":
        if func not in BUILTIN:
            raise ValueError(f"no builtin check {func!r}. Builtins: {', '.join(BUILTIN)}")
        return BUILTIN[func]
    if where.endswith(".py"):
        loader = importlib.util.spec_from_file_location("audited_check", Path(where))
        module = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(module)
    else:
        module = importlib.import_module(where)
    return getattr(module, func)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="avail audit-check", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", required=True, help="builtin:<name>, path/to/file.py:function or module:function")
    parser.add_argument("--connector", default="latch", help="the task variant: latch or fric")
    parser.add_argument("--split", default="calibration", help="a zoo split that is no test split")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    from .config import TEST_SPLITS
    from .zoo import store

    if store.SPLITS.get(args.split) in TEST_SPLITS:
        parser.error(f"{args.split} is a test split, read once by its final run. Audit on calibration or fit")
    result = audit(load_check(args.check), args.connector, args.split, args.limit)
    print(json.dumps(result, indent=1, default=float) if args.json else report(result, args.check), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
