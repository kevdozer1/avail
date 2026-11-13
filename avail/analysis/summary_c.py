"""S4c: hard mode v1 and hard mode v2 results side by side, and what the click settles. Evaluator-side.

Reads the saved result files of ``avail e1e2 --profile hard`` (``runs/e1e2b_<mode>``) and of
``avail e1e2 --profile hard2`` (``runs/e1e2c_<mode>``). The click-settle numbers of hard mode v1
were not part of its saved results, so they are computed here from its zoo tables on disk, with
the same code as v2's (docs/07, S4c note 4). That simulates no seed. Easy mode is left out, because
its test range is retired.

    avail s4c tables --mode final        prints every table for both connectors
    avail s4c tables --mode rehearsal    the same on calibration seeds standing in for test
"""

from __future__ import annotations

import json

from .summary_b import (_load, _num, cascade_table, class_table, evidence_table, h1_line, jam_table, pct, price_table,
                        probe_table, short, stage_table)

NAMES = ("Hard v1", "Hard v2")


def click_settles_for(connector: str, mode: str, profile: str) -> dict:
    """The click-settle numbers of one mode, recomputed from the zoo tables on disk (no seed is read)."""
    from ..config import load_experiment
    from . import e1, e2
    from .run import click_settles, primary_predicate, tables_for

    bounds = load_experiment().bounds
    predicate = primary_predicate(connector)
    tabs = tables_for(connector, mode, profile=profile)
    fit, cal, ev = (e1.main_slice(tabs[k], predicate) for k in ("fit", "cal", "eval"))
    cascade, *_ = e2.build(fit, cal, connector, predicate, bounds.alpha, bounds.beta, split_id=tabs["split_id"])
    return click_settles(cascade, fit, cal, ev, connector, predicate, bounds)


def click_table(rows: dict[str, dict]) -> str:
    """What the click settles, per mode."""
    lines = ["| Mode | Claims the click settles | Claims the passive stage decides | Click agrees with that decision | "
             "Without the click: sent to a person | Without the click: bad accepted |", "|---|---|---|---|---|---|"]
    for name, cs in rows.items():
        nc = cs.get("no_click_cascade", {})
        lines.append(f"| {name} | {pct(cs.get('settled'))} | {pct(cs.get('decided_at_passive'))} | "
                     f"{pct(cs.get('click_agrees_with_passive_decision'))} | {short(nc.get('abstain'), 2)} | "
                     f"{short(nc.get('far'), 2)} |")
    return "\n".join(lines)


def headline(v1: dict, v2: dict) -> str:
    """1 table with the numbers the S4c summary leads with."""
    rows = [("Skill's own flag accepts bad plugs", "passive.skill_flag"), ("Click accepts bad plugs", "passive.click"),
            ("Tool pose accepts bad plugs", "passive.pose_residual"), ("Camera accepts bad plugs", "vision.gap@claim"),
            ("Push accepts bad plugs", "probe.push_seat"), ("Tug accepts bad plugs", "probe.tug"),
            ("All passive signals fused accept bad plugs", "fused.passive")]
    lines = [f"| | {NAMES[0]} | {NAMES[1]} |", "|---|---|---|"]
    for label, check in rows:
        a, b = v1["e1"]["table"].get(check, {}), v2["e1"]["table"].get(check, {})
        lines.append(f"| {label} | {short(a.get('far'))} | {short(b.get('far'))} |")
    for label, key in (("Cascade at 2%: bad plugs accepted", "far"), ("Claims sent to a person", "abstain"),
                       ("Good plugs rejected", "frr")):
        lines.append(f"| {label} | {short(v1['e2']['result'][key], 2)} | {short(v2['e2']['result'][key], 2)} |")
    lines.append(f"| Certified bound | {100 * _num(v1['e2']['result']['far_ucb']):.2f}% | "
                 f"{100 * _num(v2['e2']['result']['far_ucb']):.2f}% |")
    lines.append(f"| H1, H2 | {v1['e1']['h1'].get('status')}, {v1['e2']['h2']['status']} | "
                 f"{v2['e1']['h1'].get('status')}, {v2['e2']['h2']['status']} |")
    return "\n".join(lines)


def all_tables(mode: str = "final", connectors=("fric", "latch")) -> str:
    out = []
    for c in connectors:
        v1, v2 = _load(mode, "hard", c), _load(mode, "hard2", c)
        # Easy mode is left out: its test range (16,000 to 17,999) is retired and Kevin's S4b brief says not to touch it
        settles = {"Hard v1 (recomputed)": click_settles_for(c, mode, "hard"),
                   "Hard v2": v2["e2"].get("click_settles")}
        settles = {k: v for k, v in settles.items() if v}
        out += [f"## {c}", "", f"Hard v1: {v1['evaluated_on']}, rows {v1['counts']['evaluation']['rows']:,}. "
                f"Hard v2: {v2['evaluated_on']}, rows {v2['counts']['evaluation']['rows']:,}. "
                f"Config of v2 {v2['config_hash']}.", "",
                "### Headline", "", headline(v1, v2), "",
                "### What the click settles", "", click_table(settles), "",
                "### Evidence table", "", evidence_table(v1, v2, NAMES), "",
                "### Hard v2 by injected class (share of bad plugs accepted)", "", class_table(v2), "",
                "### Hard v2 by landed class (share of bad plugs accepted)", "", class_table(v2, "by_landed_class"), "",
                "### Hard v2, short plugs by cause", "", jam_table(v2), "",
                "### Probes", "", probe_table(v1, v2, NAMES), "",
                "### Cascade", "", cascade_table(v1, v2, ("hard v1", "hard v2")), "",
                "### Hard v2, decided by stage", "", stage_table(v2), "",
                "### Price list", "", price_table(v1, v2, NAMES), "",
                f"H1 hard v1: {h1_line(v1)}", "", f"H1 hard v2: {h1_line(v2)}", "",
                f"Landed classes, hard v2: {json.dumps(v2.get('landed', {}).get('evaluation', {}))}", ""]
    return "\n".join(out)


def demo_results(mode: str = "final", connectors=("fric", "latch")) -> dict:
    """The reported v2 numbers, small enough to ship with the repo next to s4b_results.json."""
    from .summary_b import DEMO_CHECKS, _slim

    out: dict = {"kind": "avail.s4c_results", "sim_or_real": "sim", "mode": mode, "connectors": {}}
    for c in connectors:
        r = _load(mode, "hard2", c)
        table, res = r["e1"]["table"], r["e2"]["result"]
        cs = r["e2"].get("click_settles", {})
        out["connectors"][c] = {
            "evaluated_on": r["evaluated_on"], "rows": r["counts"]["evaluation"]["rows"],
            "good": r["counts"]["evaluation"]["true"], "config_hash": r["config_hash"],
            "checks": {k: {"far": _slim(table[k]["far"]), "near_left_bad": _slim(table[k]["leaves_bad_near_miss"])}
                       for k in DEMO_CHECKS if k in table},
            "cascade": {"far": _slim(res["far"]), "far_ucb": res["far_ucb"], "far_ucb_m": res["far_ucb_m"],
                        "abstain": _slim(res["abstain"]), "frr": _slim(res["frr"]), "mean_seconds": res["mean_seconds"]},
            "click_settles": _slim(cs.get("settled")), "decided_at_passive": _slim(cs.get("decided_at_passive")),
            "h1": r["e1"]["h1"].get("status"), "h2": r["e2"]["h2"]["status"],
        }
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(prog="avail s4c tables")
    parser.add_argument("--mode", default="final", choices=["rehearsal", "final"])
    parser.add_argument("--export-demo", action="store_true",
                        help="also write avail/witness/shipped/s4c_results.json")
    args = parser.parse_args(argv)
    text = all_tables(args.mode)
    print(text, flush=True)
    from ..paths import runs_dir

    path = runs_dir() / f"s4c_tables_{args.mode}.md"
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"\nwritten to {path}", flush=True)
    if args.export_demo:
        from ..witness.store import SHIPPED_DIR

        out = Path(SHIPPED_DIR) / "s4c_results.json"
        out.write_text(json.dumps(demo_results(args.mode), indent=1), encoding="utf-8", newline="\n")
        print(f"demo results: {out}", flush=True)
    return 0
