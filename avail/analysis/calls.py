"""Where the calls for help come from, from a shift's ledgers alone (the night of 2026-11-16, step 1).

A call for help is an assist as the shift card counts it (``avail.shift.card``): a new call to the operator. The
driver (``avail.shift.driver``) asks in 3 ways, and each ask leaves 1 ``intervention`` event:

- **abstain**: the witness abstained, the operator is asked to confirm (``question`` confirm)
- **reject after retries**: the witness rejected the last try, the operator is asked to choose, or to tap the port
  for the carry (``question`` choose or guide_marks)
- **takeover, called back**: the operator's own answer did not fix the step, so they are called back and take over

A takeover right after the operator's own answer is the same call (no 2nd wait), so it counts operator time and no
call. ``seconds_waiting`` is 0 exactly for those, which is how they are told apart here.

Usage::

    uv run python -m avail.analysis.calls --shift <AVAIL_DATA>/runs/shifts/e3b_final_latch --arm avail_simop
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

STEPS = ("pick", "carry", "insert", "release")
REASONS = ("abstain", "reject after retries", "takeover, called back", "takeover, same call")


def reason_of(iv: dict) -> str:
    q = iv.get("question")
    if q == "confirm":
        return "abstain"
    if q in ("choose", "guide_marks"):
        return "reject after retries"
    if q == "takeover":
        return "takeover, called back" if float(iv.get("seconds_waiting", 0.0)) > 0.0 else "takeover, same call"
    return f"other ({q})"


def is_call(iv: dict) -> bool:
    return float(iv.get("seconds_waiting", 0.0)) > 0.0


def _blocks(shift_dir: Path, arm: str, variant: str) -> list[Path]:
    return sorted(p for p in Path(shift_dir).glob(f"*_{arm}_{variant}_b*/ledger.jsonl"))


def load(shift_dir, arm: str = "avail_simop", variant: str = "latch") -> dict:
    """The arm's ended cycles and their interventions, verdicts and claims."""
    from ..ledger.reader import iter_events

    ends, ivs, verdicts, claims = [], [], {}, {}
    for ledger in _blocks(Path(shift_dir), arm, variant):
        for e in iter_events(ledger, kinds=["cycle_end", "intervention", "verdict", "claim"]):
            p, k = e["payload"], e["kind"]
            if k == "cycle_end":
                ends.append(p)
            elif k == "intervention":
                ivs.append(p)
            elif k == "verdict":
                verdicts[p["verdict_id"]] = p
            else:
                claims[p["claim_id"]] = p
    ended = {e["cycle_id"] for e in ends}
    ivs = [iv for iv in ivs if iv.get("cycle_id") in ended]       # only ended cycles count, as on the shift card
    return {"ends": ends, "ivs": ivs, "verdicts": verdicts, "claims": claims}


def per_cycle_calls(data: dict) -> list[int]:
    """Calls for help per ended cycle, in seed order: the unit of H4b's bootstrap."""
    return [int(e.get("assists", 0)) for e in sorted(data["ends"], key=lambda e: int(e.get("seed", 0)))]


def breakdown(data: dict) -> dict:
    """Calls per 100 cycles and operator seconds per cycle, by step and reason, with the claim's truth label where
    the operator gave one (True: the claim was good, so the call was for a claim the witness could not accept)."""
    n = len(data["ends"])
    rows: dict[tuple[str, str], dict] = defaultdict(lambda: {"calls": 0, "asks": 0, "wait_s": 0.0, "handling_s": 0.0,
                                                             "good": 0, "bad": 0, "unlabeled": 0})
    for iv in data["ivs"]:
        key = (iv.get("step", "?"), reason_of(iv))
        r = rows[key]
        r["asks"] += 1
        r["calls"] += int(is_call(iv))
        r["wait_s"] += float(iv.get("seconds_waiting", 0.0))
        r["handling_s"] += float(iv.get("seconds_handling", 0.0))
        label = iv.get("truth_label")
        r["good" if label is True else "bad" if label is False else "unlabeled"] += 1
    total_calls = sum(r["calls"] for r in rows.values())
    card_calls = sum(int(e.get("assists", 0)) for e in data["ends"])
    out = []
    for (step, reason), r in sorted(rows.items(), key=lambda kv: (STEPS.index(kv[0][0]) if kv[0][0] in STEPS else 9,
                                                                   REASONS.index(kv[0][1]) if kv[0][1] in REASONS else 9)):
        out.append({"step": step, "reason": reason, "calls": r["calls"], "asks": r["asks"],
                    "calls_per_100": 100.0 * r["calls"] / n if n else float("nan"),
                    "share_of_calls": r["calls"] / total_calls if total_calls else float("nan"),
                    "wait_s_per_cycle": r["wait_s"] / n, "handling_s_per_cycle": r["handling_s"] / n,
                    "good": r["good"], "bad": r["bad"], "unlabeled": r["unlabeled"]})
    t = {k: sum(float(e.get("time_s", {}).get(k, 0.0)) for e in data["ends"]) / n
         for k in ("motion", "verify", "model", "wait", "handling")} if n else {}
    return {"cycles": n, "calls": total_calls, "calls_on_card": card_calls,
            "calls_per_100": 100.0 * total_calls / n if n else float("nan"), "rows": out, "time_split_s": t}


def decisions(data: dict) -> dict:
    """The witnesses' decisions per predicate and try (description only)."""
    out: dict = defaultdict(lambda: defaultdict(int))
    for v in data["verdicts"].values():
        c = data["claims"].get(v.get("claim_id"))
        if c is None:
            continue
        attempt = int(c.get("context", {}).get("attempt", 0))
        out[c["predicate"]["name"]][f"{v['decision']}@{'1st' if attempt == 0 else 'retry'}"] += 1
    return {k: dict(v) for k, v in out.items()}


def md(b: dict) -> str:
    lines = [f"{b['cycles']} cycles, {b['calls']} calls for help ({b['calls_per_100']:.1f} per 100 cycles, the shift "
             f"card counts {b['calls_on_card']}).", "",
             "| Step | Reason | Calls per 100 cycles | Share of calls | Operator s per cycle (waiting + handling) | "
             "Good claims / bad claims among the asks |", "|---|---|---|---|---|---|"]
    for r in b["rows"]:
        lines.append(f"| {r['step']} | {r['reason']} | {r['calls_per_100']:.1f} | {100 * r['share_of_calls']:.1f}% | "
                     f"{r['wait_s_per_cycle']:.1f} + {r['handling_s_per_cycle']:.1f} | {r['good']} / {r['bad']}"
                     + (f" ({r['unlabeled']} unlabeled)" if r["unlabeled"] else "") + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shift", required=True, type=Path)
    ap.add_argument("--arm", default="avail_simop")
    ap.add_argument("--variant", default="latch")
    ap.add_argument("--out", type=Path, default=None, help="a JSON file for the breakdown")
    args = ap.parse_args(argv)
    data = load(args.shift, args.arm, args.variant)
    b = breakdown(data)
    b["decisions"] = decisions(data)
    print(md(b))
    print(json.dumps(b["decisions"], indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(b, indent=1), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
