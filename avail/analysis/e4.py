"""E4's gate and E4's hypotheses, from a shift's ledgers alone (docs/07, the notes of 2026-11-16).

- ``gate``: calls for help per 100 cycles, cycles with a bad cable shipped, and per witness the share of good claims it
  sent to the operator, each claim labeled by truth right after its witness ran (the ``claims`` list of ``cycle_end``,
  evaluator-side). Also the ladder's rungs: how often each ran, how often the witness then accepted, and truth before
  and after it.
- ``h4``: H4a (cycles with a bad cable shipped, one-sided 95% Clopper-Pearson at most 0.5%, so at most 16 of 5,000) and
  H4b (calls for help per 100 cycles, the one-sided 95% percentile-bootstrap bound over cycles at most 112), with the
  description-only numbers: robots per operator, cycle time against ``skill_done`` on the seeds both ran, takeovers.

Usage::

    uv run python -m avail.analysis.e4 --shift <AVAIL_DATA>/runs/shifts/gate_e4 --gate
    uv run python -m avail.analysis.e4 --shift <AVAIL_DATA>/runs/shifts/e4_final_latch --h4
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from ..stats import bootstrap_mean_upper, cp_upper, wilson


def _ends(shift_dir: Path, arm: str) -> list[dict]:
    from ..ledger.reader import iter_events

    out = []
    for ledger in sorted(Path(shift_dir).glob(f"*_{arm}_*_b*/ledger.jsonl")):
        out += [e["payload"] for e in iter_events(ledger, kinds=["cycle_end"]) if e["payload"].get("arm") == arm]
    return sorted(out, key=lambda e: int(e.get("seed", 0)))


def _cell(k: int, n: int) -> dict:
    lo, hi = wilson(k, n) if n else (float("nan"), float("nan"))
    return {"k": int(k), "n": int(n), "rate": k / n if n else float("nan"), "lo": lo, "hi": hi}


def operator_numbers(ends: list[dict]) -> dict:
    n = len(ends)
    t = {k: sum(float(e.get("time_s", {}).get(k, 0.0)) for e in ends) / n for k in ("motion", "verify", "model", "wait",
                                                                                      "handling")}
    cycle_s = sum(t.values())
    person_s = t["wait"] + t["handling"]
    return {"cycle_s": cycle_s, "time_split_s": t, "operator_s_per_cycle": person_s,
            "robots_per_operator": cycle_s / person_s if person_s else float("inf"),
            "takeovers_per_100": 100.0 * sum(int(e.get("takeovers", 0)) for e in ends) / n}


def gate(shift_dir, arm: str = "avail_simop", cfg: dict | None = None) -> dict:
    from ..config import load_experiment

    g = dict((cfg or load_experiment().e4)["gate"])
    ends = _ends(Path(shift_dir), arm)
    n = len(ends)
    calls = [int(e.get("assists", 0)) for e in ends]
    bad = sum(1 for e in ends if int(e.get("uncaught", 0)) > 0)
    by_pred: dict[str, dict] = defaultdict(lambda: {"good": 0, "good_to_operator": 0, "bad": 0, "bad_accepted": 0,
                                                    "good_rejected": 0, "good_rejected_last_try": 0})
    last = 1                                     # the cable loop's 2 tries: a reject at try index 1 or later is a last reject
    first: dict[str, dict] = defaultdict(lambda: {"good": 0, "good_to_operator": 0})
    rungs: dict[str, dict] = defaultdict(lambda: {"ran": 0, "accepted": 0, "true_before": 0, "true_after": 0,
                                                  "good_turned_bad": 0, "bad_made_good": 0, "after_abstain": 0,
                                                  "after_reject": 0, "seconds": 0.0})
    for e in ends:
        for c in e.get("claims", []):
            p = by_pred[c["predicate"]]
            if c["true"]:
                p["good"] += 1
                p["good_to_operator"] += int(c["decision"] == "abstain")
                p["good_rejected"] += int(c["decision"] == "reject")
                p["good_rejected_last_try"] += int(c["decision"] == "reject" and int(c.get("attempt", 0)) >= last)
            else:
                p["bad"] += 1
                p["bad_accepted"] += int(c["decision"] == "accept")
            if c.get("rung") is None:
                f = first[c["predicate"]]
                if c["true"]:
                    f["good"] += 1
                    f["good_to_operator"] += int(c["decision"] == "abstain")
        for r in e.get("rungs", []):
            x = rungs[r["rung"]]
            x["ran"] += 1
            x["accepted"] += int(r["decision"] == "accept")
            x["true_before"] += int(bool(r["true_before"]))
            x["true_after"] += int(bool(r["true_after"]))
            x["good_turned_bad"] += int(bool(r["true_before"]) and not bool(r["true_after"]))
            x["bad_made_good"] += int((not bool(r["true_before"])) and bool(r["true_after"]))
            x[f"after_{r['after']}"] = x.get(f"after_{r['after']}", 0) + 1
            x["seconds"] += float(r.get("seconds", 0.0))
    witnesses = {}
    for pred, p in by_pred.items():
        rate = _cell(p["good_to_operator"], p["good"])
        # the brief's 20% part counts abstains on good claims. Good claims rejected on their last try also go to the
        # operator, so that share is reported beside it (review, 2026-11-16), as is good rejected at any try
        witnesses[pred] = {**p, "good_to_operator_rate": rate, "pass": bool(rate["rate"] <= g["max_good_to_operator"]),
                           "good_abstain_or_last_reject_rate": _cell(p["good_to_operator"] + p["good_rejected_last_try"],
                                                                     p["good"]),
                           "good_rejected_rate": _cell(p["good_rejected"], p["good"]), "first_look": first.get(pred)}
    calls_100 = 100.0 * sum(calls) / n if n else float("nan")
    out = {"arm": arm, "cycles": n, "calls_per_100": calls_100,
           "calls_per_100_ub95_bootstrap": 100.0 * bootstrap_mean_upper(calls) if n else float("nan"),
           "bad_cycles": bad, "bad_cycles_rate": _cell(bad, n), "witnesses": witnesses, "rungs": dict(rungs),
           **(operator_numbers(ends) if n else {})}
    out["pass"] = {"cycles": n >= int(g["cycles"]), "calls": calls_100 <= float(g["max_calls_per_100"]),
                   "bad_cycles": bad <= int(g["max_bad_cycles"]),
                   "witnesses": all(w["pass"] for w in witnesses.values())}
    out["gate_passed"] = all(out["pass"].values())
    return out


def h4(shift_dir, cfg: dict | None = None) -> dict:
    from ..config import load_experiment

    e4 = dict(cfg or load_experiment().e4)
    a, b = dict(e4["h4a"]), dict(e4["h4b"])
    ends = _ends(Path(shift_dir), a["arm"])
    n = len(ends)
    bad = sum(1 for e in ends if int(e.get("uncaught", 0)) > 0)
    calls = [int(e.get("assists", 0)) for e in ends]
    ucb = cp_upper(bad, n) if n else float("nan")
    calls_ub = 100.0 * bootstrap_mean_upper(calls, resamples=int(b["resamples"]), seed=int(b["rng_seed"]),
                                            confidence=float(b["confidence"])) if n else float("nan")
    flag = {int(e["seed"]): e for e in _ends(Path(shift_dir), "flag")}
    shared = [e for e in ends if int(e["seed"]) in flag]
    line_speed = None
    if shared:
        mine = operator_numbers(shared)["cycle_s"]
        theirs = operator_numbers([flag[int(e["seed"])] for e in shared])["cycle_s"]
        line_speed = {"cycles": len(shared), "avail_simop_cycle_s": mine, "skill_done_cycle_s": theirs,
                      "ratio": mine / theirs if theirs else None}
    flag_bad = sum(1 for e in flag.values() if int(e.get("uncaught", 0)) > 0)
    return {
        "cycles": n, "complete": n == int(a["cycles"]),
        "h4a": {"bad_cycles": bad, "rate": _cell(bad, n), "cp_upper_95": ucb, "bar": float(a["bar"]),
                "max_bad_cycles": int(a["max_bad_cycles"]), "passed": bool(n == int(a["cycles"]) and bad <= int(a["max_bad_cycles"]))},
        "h4b": {"calls": sum(calls), "calls_per_100": 100.0 * sum(calls) / n if n else float("nan"),
                "ub95_bootstrap": calls_ub, "max": float(b["max_calls_per_100"]),
                "passed": bool(n == int(a["cycles"]) and calls_ub <= float(b["max_calls_per_100"]))},
        "both_passed": None,
        "description_only": {**(operator_numbers(ends) if n else {}), "line_speed_vs_skill_done": line_speed,
                             "skill_done": {"cycles": len(flag), "bad_cycles": flag_bad, "rate": _cell(flag_bad, len(flag))}},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shift", required=True, type=Path)
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--h4", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    out = {}
    if args.gate:
        out["gate"] = gate(args.shift)
    if args.h4:
        r = h4(args.shift)
        r["both_passed"] = bool(r["h4a"]["passed"] and r["h4b"]["passed"])
        out["h4"] = r
    text = json.dumps(out, indent=1, default=float)
    print(text, flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
