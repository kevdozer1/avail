"""The shift card and the E3 table, from the ledger alone. Core code: it reads ledger files and nothing else.

A shift folder holds 1 ledger run per block of cycles (``avail.shift.runner``). Every number below
comes from the ``cycle_end`` and ``intervention`` events in them. The definitions follow docs/07:

- **Defect rate**: cycles that ended with at least 1 item declared done that was not done by truth,
  over cycles, with the one-sided 95% Clopper-Pearson upper bound, against ``customer_bar.defect_rate``.
  Also the same per item, and uncaught items per 100 cycles with a Wilson interval.
- **Uptime**: operating time of cycles that left no item unresolved, over all shift time. Shift time
  is operating time (motion, verification, model calls) plus time waiting on a person and their handling.
- **Calls for help**: assists per 100 cycles, and takeovers per 100 cycles.
- **MTBI**: operating seconds per assist. **Autonomy**: 1 minus person seconds over operating seconds.
- **Line speed**: mean shift seconds per cycle, and that over the flag arm's on the same seeds.

A sim shift card is labeled sim. It says nothing about a real line.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from ..stats import cp_upper, wilson


def _events(shift_dir: Path, kind: str) -> list[dict]:
    from ..ledger.reader import iter_events

    out = []
    for ledger in sorted(Path(shift_dir).glob("*/ledger.jsonl")):
        out += [e["payload"] for e in iter_events(ledger, kinds=[kind])]
    return out


def _cell(k: int, n: int) -> dict:
    lo, hi = wilson(k, n) if n else (float("nan"), float("nan"))
    return {"k": int(k), "n": int(n), "rate": (k / n) if n else float("nan"), "lo": lo, "hi": hi}


def arm_card(ends: list[dict], interventions: list[dict], bar: float) -> dict:
    n = len(ends)
    if n == 0:
        return {"cycles": 0}
    defective = sum(1 for e in ends if e.get("uncaught", 0) > 0)
    items = sum(len(e.get("items", [])) for e in ends)
    uncaught_items = sum(int(e.get("uncaught", 0)) for e in ends)
    true_cycles = sum(1 for e in ends if e.get("true_success"))
    declared_cycles = sum(1 for e in ends if e.get("declared_success"))
    t = {k: sum(float(e.get("time_s", {}).get(k, 0.0)) for e in ends) for k in ("motion", "verify", "model", "wait", "handling")}
    operating = t["motion"] + t["verify"] + t["model"]
    person = t["wait"] + t["handling"]
    shift = operating + person
    clean = [e for e in ends if not any(i.get("unresolved") for i in e.get("items", []))]
    clean_operating = sum(float(e.get("time_s", {}).get(k, 0.0)) for e in clean for k in ("motion", "verify", "model"))
    assists = sum(int(e.get("assists", 0)) for e in ends)
    takeovers = sum(int(e.get("takeovers", 0)) for e in ends)
    unresolved_items = sum(1 for e in ends for i in e.get("items", []) if i.get("unresolved"))
    by_q: dict[str, int] = {}
    for iv in interventions:
        by_q[iv.get("question", "?")] = by_q.get(iv.get("question", "?"), 0) + 1
    steps = [s for e in ends for s in e.get("steps", [])]
    bad_steps = sum(1 for s in steps if s.get("declared") and not s.get("true"))
    return {
        "cycles": n, "items": items,
        "true_success": _cell(true_cycles, n), "declared_success": _cell(declared_cycles, n),
        "defect_rate": {**_cell(defective, n), "ucb95": cp_upper(defective, n), "bar": bar,
                        "certified": bool(cp_upper(defective, n) < bar)},
        "uncaught_items": _cell(uncaught_items, items), "uncaught_items_per_100_cycles": 100.0 * uncaught_items / n,
        "unresolved_items": _cell(unresolved_items, items),
        "assists_per_100": 100.0 * assists / n, "takeovers_per_100": 100.0 * takeovers / n,
        "assists": assists, "takeovers": takeovers, "assists_by_question": by_q,
        "mtbi_s": (operating / assists) if assists else math.inf,
        "autonomy": (1.0 - person / operating) if operating else float("nan"),
        "uptime": (clean_operating / shift) if shift else float("nan"),
        "mean_cycle_s": shift / n, "time_split_s": {k: v / n for k, v in t.items()},
        # per item: description only (Kevin's S5b note in docs/07: H3 stays per cycle, with no new pass or fail)
        "assists_per_100_items": 100.0 * assists / items if items else float("nan"),
        "seconds_per_item": shift / items if items else float("nan"),
        "bad_steps_through": _cell(bad_steps, len(steps)),
        "usd_per_cycle": sum(float(e.get("usd", 0.0)) for e in ends) / n,
        # docs/07's safety line (probe cap violations and safety stops) is not in the ledger's cycle_end events,
        # so the card does not state it. A probe passes the config force cap unchanged to its guarded move.
        "safety_events": None,
    }


def shift_card(shift_dir) -> dict:
    """Every arm and variant of a shift folder."""
    from ..config import load_experiment

    bar = float(load_experiment().customer_bar["defect_rate"])
    ends = _events(Path(shift_dir), "cycle_end")
    ivs = _events(Path(shift_dir), "intervention")
    out: dict = {"sim_or_real": "sim", "shift_dir": str(shift_dir), "arms": {}}
    keys = sorted({(e.get("arm"), e.get("variant")) for e in ends})
    for arm, variant in keys:
        mine = [e for e in ends if e.get("arm") == arm and e.get("variant") == variant]
        cycle_ids = {e.get("cycle_id") for e in mine}
        card = arm_card(mine, [iv for iv in ivs if iv.get("cycle_id") in cycle_ids], bar)
        out["arms"][f"{arm} {variant}"] = card
    for key, card in out["arms"].items():
        arm, variant = key.split(" ")
        flag = out["arms"].get(f"flag {variant}")
        if flag and card.get("cycles"):
            card["line_speed_vs_flag"] = card["mean_cycle_s"] / flag["mean_cycle_s"] if flag.get("mean_cycle_s") else None
    return out


def _pct(c: dict, digits: int = 1) -> str:
    if not c or not c.get("n"):
        return ""
    return f"{100 * c['rate']:.{digits}f}% ({c['k']} of {c['n']}, {100 * c['lo']:.{digits}f} to {100 * c['hi']:.{digits}f})"


def md(card: dict) -> str:
    lines = ["Sim. Each cycle is every item of the task, then a reset.", "",
             "| Arm | Cycles | True success | Cycles with an uncaught failure (bound) | Uncaught items per 100 cycles | "
             "Calls for help per 100 | Takeovers per 100 | MTBI s | Autonomy | Uptime | Cycle s (vs flag) | $ per cycle |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, c in card["arms"].items():
        if not c.get("cycles"):
            continue
        d = c["defect_rate"]
        speed = f"{c['mean_cycle_s']:.0f}" + (f" ({c['line_speed_vs_flag']:.2f}x)" if c.get("line_speed_vs_flag") else "")
        mtbi = "none" if math.isinf(c["mtbi_s"]) else f"{c['mtbi_s']:.0f}"
        lines.append(f"| {key} | {c['cycles']} | {_pct(c['true_success'])} | {_pct(d)}, bound {100 * d['ucb95']:.1f}% | "
                     f"{c['uncaught_items_per_100_cycles']:.1f} | {c['assists_per_100']:.1f} | {c['takeovers_per_100']:.1f} | "
                     f"{mtbi} | {100 * c['autonomy']:.1f}% | {100 * c['uptime']:.1f}% | {speed} | {c['usd_per_cycle']:.3f} |")
    lines += ["", "Per item and where the time goes, description only (no pass or fail).", "",
              "| Arm | Calls for help per 100 items | Seconds per item | Motion s | Checks and probes s | Model calls s | "
              "Waiting on a person s | Handling s | Bad steps through |",
              "|---|---|---|---|---|---|---|---|---|"]
    for key, c in card["arms"].items():
        if not c.get("cycles") or "time_split_s" not in c:
            continue
        t = c["time_split_s"]
        lines.append(f"| {key} | {c['assists_per_100_items']:.1f} | {c['seconds_per_item']:.1f} | {t['motion']:.1f} | "
                     f"{t['verify']:.1f} | {t['model']:.1f} | {t['wait']:.1f} | {t['handling']:.1f} | "
                     f"{_pct(c['bad_steps_through'], 2)} |")
    return "\n".join(lines)


def write(shift_dir) -> Path:
    card = shift_card(shift_dir)
    path = Path(shift_dir) / "shift_card.json"
    path.write_text(json.dumps(card, indent=1, default=float), encoding="utf-8", newline="\n")
    (Path(shift_dir) / "shift_card.md").write_text(md(card) + "\n", encoding="utf-8", newline="\n")
    return path
