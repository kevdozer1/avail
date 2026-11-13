"""S4b: easy-mode and hard-mode results side by side, as markdown for reports/S4b.md. Evaluator-side.

Reads the saved result files of ``avail e1e2`` (easy, ``runs/e1e2_<mode>``) and of
``avail e1e2 --profile hard`` (``runs/e1e2b_<mode>``). Computes nothing new: every number is a
number the analysis already wrote, put next to its twin.

    avail s4b tables --mode final        prints every table for both connectors
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .figures import CHECK_NAMES, CLASS_NAMES
from .summary import TABLE_ORDER


def _load(mode: str, profile: str, connector: str) -> dict:
    from ..paths import runs_dir
    from .run import RESULT_FOLDER

    folder = runs_dir() / f"{RESULT_FOLDER[profile]}_{mode}"
    return json.loads((folder / f"e1e2_{connector}.json").read_text(encoding="utf-8"))


def _num(v) -> float:
    if v is None:
        return float("nan")
    if isinstance(v, str):
        return float(v.replace("inf", "inf"))
    return float(v)


def pct(cell: dict | None, digits: int = 1, n_word: str = "of") -> str:
    """'12.3% (45 of 366, 9.4 to 16.0)', or 'none bad' when the denominator is 0."""
    if not cell or not cell.get("n"):
        return "none bad"
    lo, hi = _num(cell.get("lo")), _num(cell.get("hi"))
    return (f"{100 * _num(cell['rate']):.{digits}f}% ({cell['k']} {n_word} {cell['n']:,}, "
            f"{100 * lo:.{digits}f} to {100 * hi:.{digits}f})")


def short(cell: dict | None, digits: int = 1) -> str:
    if not cell or not cell.get("n"):
        return "none bad"
    return f"{100 * _num(cell['rate']):.{digits}f}% ({cell['k']}/{cell['n']:,})"


def evidence_table(easy: dict, hard: dict, names: tuple[str, str] = ("Easy", "Hard")) -> str:
    """1 row per check: false-accept rate in easy mode and in hard mode, and the near misses left bad."""
    table_e, table_h = easy["e1"]["table"], hard["e1"]["table"]
    near = ", ".join(hard["e1"]["h1"].get("classes", []))
    a, b = names
    lines = [f"| Check | {a}: bad plugs accepted | {b}: bad plugs accepted | {a}: near misses {near} left bad | "
             f"{b}: near misses left bad |", "|---|---|---|---|---|"]
    for check in TABLE_ORDER:
        if check not in table_h and check not in table_e:
            continue
        e, h = table_e.get(check, {}), table_h.get(check, {})
        lines.append(f"| {CHECK_NAMES.get(check, check)} | {short(e.get('far'))} | {short(h.get('far'))} | "
                     f"{short(e.get('leaves_bad_near_miss'))} | {short(h.get('leaves_bad_near_miss'))} |")
    return "\n".join(lines)


def class_table(result: dict, key: str = "by_class") -> str:
    """The check-by-class grid of 1 mode. ``key`` by_landed_class gives the classes by truth at claim time."""
    table = result["e1"]["table"]
    present = set(result["counts"]["evaluation"]["classes"])
    if key.endswith("landed_class"):
        present = {c for row in table.values() for c in row.get(key, {})}
    classes = [c for c in ("N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8") if c in present]
    lines = ["| Check | " + " | ".join(f"{c} {CLASS_NAMES.get(c, '')}" for c in classes) + " |",
             "|---|" + "---|" * len(classes)]
    for check in TABLE_ORDER:
        if check not in table:
            continue
        cells = []
        for c in classes:
            cell = table[check][key].get(c)
            cells.append("" if not cell or not cell["n"] else f"{100 * _num(cell['rate']):.0f}% ({cell['k']}/{cell['n']})")
        lines.append(f"| {CHECK_NAMES.get(check, check)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def jam_table(hard: dict) -> str:
    """Hard mode: each check on the short plugs (N1 to N4), by what made them short. Accepted and still bad over all."""
    by = hard["e1"].get("by_jam", {})
    if not by:
        return "(no jam breakdown in this result)"
    kinds = ("stuck joint", "burr", "wedge")
    lines = ["| Check | " + " | ".join(f"{k}: left bad" for k in kinds) + " |", "|---|" + "---|" * len(kinds)]
    for check in TABLE_ORDER:
        if check not in by:
            continue
        lines.append(f"| {CHECK_NAMES.get(check, check)} | " + " | ".join(short(by[check][k]["leaves_bad"]) for k in kinds) + " |")
    return "\n".join(lines)


def probe_table(easy: dict, hard: dict, names: tuple[str, str] = ("Easy", "Hard")) -> str:
    """Harm and repair of each probe, easy and hard, from E1's pass A."""
    a, b = names
    lines = [f"| Probe | {a}: harm | {b}: harm | {a}: repair | {b}: repair |", "|---|---|---|---|---|"]
    for probe in ("probe.push_seat", "probe.tug", "probe.wiggle"):
        e, h = easy["e1"]["table"].get(probe, {}), hard["e1"]["table"].get(probe, {})
        lines.append(f"| {CHECK_NAMES.get(probe, probe)} | {short(e.get('harm'), 2)} | {short(h.get('harm'), 2)} | "
                     f"{short(e.get('repair'))} | {short(h.get('repair'))} |")
    rows = hard["e1"].get("probe_by_jam", {})
    if rows:
        lines += ["", "| Probe, hard mode, short plugs | stuck joint: repaired | burr: repaired | wedge: repaired |",
                  "|---|---|---|---|"]
        for probe, by in rows.items():
            lines.append(f"| {CHECK_NAMES.get(probe, probe)} | " + " | ".join(short(by.get(k)) for k in ("stuck joint", "burr", "wedge")) + " |")
    return "\n".join(lines)


def cascade_table(easy: dict, hard: dict, names: tuple[str, str] = ("easy", "hard")) -> str:
    """E2 and E2b: the cascade at alpha 2%, easy and hard."""
    rows = []
    for name, r in zip(names, (easy, hard)):
        res = r["e2"]["result"]
        rows.append((name, res, r["e2"]["h2"], r["e2"]["resplits"]))
    lines = ["| Mode | Bad accepted | Certified bound | Good rejected | Sent to a person | Mean seconds | Re-split mean FAR | H2 |",
             "|---|---|---|---|---|---|---|---|"]
    for name, res, h2, rs in rows:
        lines.append(f"| {name} | {short(res['far'], 2)} | {100 * _num(res['far_ucb']):.2f}% (m = {res['far_ucb_m']:,}) | "
                     f"{short(res['frr'], 2)} | {short(res['abstain'], 2)} | {_num(res['mean_seconds']):.2f} | "
                     f"{100 * _num(rs['far']['mean']):.3f}% (se {100 * _num(rs['far']['se']):.3f}) | {h2['status']} |")
    return "\n".join(lines)


def price_table(easy: dict, hard: dict, names: tuple[str, str] = ("Easy", "Hard")) -> str:
    """The price list: abstain rate against alpha, easy and hard."""
    a, b = names
    lines = [f"| alpha | {a}: sent to a person | {b}: sent to a person | {a}: bad accepted | {b}: bad accepted |",
             "|---|---|---|---|---|"]
    for e, h in zip(easy["e2"]["sweep"], hard["e2"]["sweep"]):
        lines.append(f"| {100 * e['alpha']:.1f}% | {short(e['abstain'], 2)} | {short(h['abstain'], 2)} | "
                     f"{short(e['far'], 2)} | {short(h['far'], 2)} |")
    return "\n".join(lines)


def stage_table(result: dict) -> str:
    res = result["e2"]["result"]
    lines = ["| Stage | Accepted | Rejected |", "|---|---|---|"]
    for stage, d in res["decided_by_stage"].items():
        lines.append(f"| {stage} | {d['accept']:,} | {d['reject']:,} |")
    lines.append(f"| sent to a person | {res['abstain']['k']:,} | |")
    return "\n".join(lines)


def h1_line(result: dict) -> str:
    h1 = result["e1"]["h1"]
    lb = h1.get("leaves_bad", {})
    if h1.get("status") == "not measured":
        return "not measured"
    return (f"{h1['status']}: camera {pct(lb.get('vision'))} against {h1.get('best_probe')} {pct(lb.get('probe'))}, "
            f"low end over high end {_num(h1.get('ratio_conservative')):.2f}")


def all_tables(mode: str = "final", connectors=("fric", "latch")) -> str:
    out = []
    for c in connectors:
        easy, hard = _load(mode, "easy", c), _load(mode, "hard", c)
        out += [f"## {c}", "", f"Easy: {easy['evaluated_on']}, rows {easy['counts']['evaluation']['rows']:,}. "
                f"Hard: {hard['evaluated_on']}, rows {hard['counts']['evaluation']['rows']:,}.", "",
                "### Evidence table", "", evidence_table(easy, hard), "",
                "### Hard mode by injected class (share of bad plugs accepted)", "", class_table(hard), "",
                "### Hard mode by landed class (share of bad plugs accepted)", "", class_table(hard, "by_landed_class"), "",
                "### Hard mode, short plugs by cause", "", jam_table(hard), "",
                "### Probes", "", probe_table(easy, hard), "",
                "### Cascade", "", cascade_table(easy, hard), "", "### Hard mode, decided by stage", "", stage_table(hard), "",
                "### Price list", "", price_table(easy, hard), "",
                f"H1 easy: {h1_line(easy)}", "", f"H1 hard: {h1_line(hard)}", "",
                f"Landed classes, hard mode: {json.dumps(hard.get('landed', {}).get('evaluation', {}))}", ""]
    return "\n".join(out)


DEMO_CHECKS = ("passive.skill_flag", "passive.pose_residual", "passive.click", "vision.gap@claim", "probe.push_seat",
               "probe.tug", "fused.passive", "fused.all")


def _slim(cell: dict | None) -> dict | None:
    if not cell:
        return None
    return {k: (None if (isinstance(cell.get(k), float) and not math.isfinite(cell[k])) else cell.get(k))
            for k in ("k", "n", "rate", "lo", "hi")}


def demo_results(mode: str = "final", connectors=("fric", "latch")) -> dict:
    """The reported numbers, easy and hard, small enough to ship with the repo for ``avail demo``."""
    out: dict = {"kind": "avail.s4b_results", "sim_or_real": "sim", "mode": mode, "connectors": {}}
    for c in connectors:
        entry = {}
        for profile in ("easy", "hard"):
            r = _load(mode, profile, c)
            table = r["e1"]["table"]
            res = r["e2"]["result"]
            entry[profile] = {
                "evaluated_on": r["evaluated_on"], "rows": r["counts"]["evaluation"]["rows"],
                "good": r["counts"]["evaluation"]["true"], "config_hash": r["config_hash"],
                "checks": {k: {"far": _slim(table[k]["far"]), "near_left_bad": _slim(table[k]["leaves_bad_near_miss"])}
                           for k in DEMO_CHECKS if k in table},
                "repair": {p: _slim(table[p].get("repair")) for p in ("probe.push_seat", "probe.tug") if p in table},
                "harm": {p: _slim(table[p].get("harm")) for p in ("probe.push_seat", "probe.tug") if p in table},
                "cascade": {"far": _slim(res["far"]), "far_ucb": res["far_ucb"], "far_ucb_m": res["far_ucb_m"],
                            "abstain": _slim(res["abstain"]), "frr": _slim(res["frr"]), "mean_seconds": res["mean_seconds"]},
                "h1": r["e1"]["h1"].get("status"), "h2": r["e2"]["h2"]["status"],
            }
        out["connectors"][c] = entry
    return out


def export_demo_results(path: Path | None = None, mode: str = "final") -> Path:
    from ..witness.store import SHIPPED_DIR

    path = Path(path) if path is not None else SHIPPED_DIR / "s4b_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(demo_results(mode), indent=1), encoding="utf-8", newline="\n")
    return path


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="avail s4b tables")
    parser.add_argument("--mode", default="final", choices=["rehearsal", "final"])
    parser.add_argument("--export-demo", action="store_true",
                        help="also write avail/witness/shipped/s4b_results.json for avail demo")
    args = parser.parse_args(argv)
    print(all_tables(args.mode), flush=True)
    if args.export_demo:
        print(f"demo results: {export_demo_results(mode=args.mode)}", flush=True)
    return 0
