"""A first look at how well each free source separates the classes. Development numbers.

Used by the S3 report on calibration seeds: a 3-number summary of each source's main
feature per injected class, the AUC of the source's own hand-made score against the truth
after it acted, and for the probes the harm rate, the repair rate and the worst real force.
Nothing here sets a threshold. That is S4.
"""

from __future__ import annotations

import numpy as np

from ..stats import wilson
from .e1 import _auc
from .run import load_split, primary_predicate

MAIN_FEATURE = {
    "passive.skill_flag": ("flag",),
    "passive.pose_residual": ("axial_mm",),
    "passive.jaw_state": ("jaw_open_error", "jaw_pos"),
    "passive.click": ("snap_mm", "drop_n"),
    "vision.gap@claim": ("gap_mm_est",),
    "probe.push_seat": ("travel_mm",),
    "probe.tug": ("disp_mm",),
    "probe.wiggle": ("lateral_compliance_mm_per_n",),
    "instr.link": ("link_up",),
}
PROBES = ("probe.push_seat", "probe.tug", "probe.wiggle")


def _rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n) if n else (float("nan"), float("nan"))
    return {"k": int(k), "n": int(n), "rate": k / n if n else float("nan"), "lo": lo, "hi": hi}


def first_look(connector: str, split: str = "calibration", root=None) -> dict:
    table = load_split(connector, split, root)
    predicate = primary_predicate(connector)
    main = table[table["claim_predicate"] == predicate].reset_index(drop=True)
    cls = np.asarray(main["cls"])
    y0 = np.asarray(main["y0"]).astype(bool)
    out: dict = {"connector": connector, "split": split, "predicate": predicate, "rows": int(len(main)),
                 "class_counts": {c: int((cls == c).sum()) for c in sorted(set(cls))}, "sources": {}, "probes": {}}
    for source, candidates in MAIN_FEATURE.items():
        feature = next((f for f in candidates if f"A/{source}/{f}" in main.columns), None)
        if feature is None:
            continue
        values = np.asarray(main[f"A/{source}/{feature}"], dtype=float)
        ycol = f"A/{source}/_y"
        y = np.asarray(main[ycol].fillna(False)).astype(bool) if ycol in main.columns else y0
        score = np.asarray(main[f"A/{source}/_score"], dtype=float) if f"A/{source}/_score" in main.columns else np.full(len(main), np.nan)
        per = {}
        for c in sorted(set(cls)):
            v = values[(cls == c) & np.isfinite(values)]
            per[c] = {"n": int(len(v)), "p10": float(np.quantile(v, 0.1)) if len(v) else float("nan"),
                      "median": float(np.median(v)) if len(v) else float("nan"),
                      "p90": float(np.quantile(v, 0.9)) if len(v) else float("nan")}
        out["sources"][source] = {
            "feature": feature, "per_class": per, "auc_of_score": _auc(score, y),
            "measured_share": float(np.isfinite(values).mean()),
            "seconds_median": float(np.nanmedian(main[f"A/{source}/_seconds"])) if f"A/{source}/_seconds" in main.columns else float("nan"),
        }
    for probe in PROBES:
        ycol = f"A/{probe}/_y"
        if ycol not in main.columns:
            continue
        y = np.asarray(main[ycol].fillna(False)).astype(bool)
        row = {"harm": _rate(int((y0 & ~y).sum()), int(y0.sum())), "repair": _rate(int((~y0 & y).sum()), int((~y0).sum())),
               "repair_by_class": {c: _rate(int((~y0 & y & (cls == c)).sum()), int((~y0 & (cls == c)).sum()))
                                   for c in sorted(set(cls)) if c != "P0"},
               "seconds_median": float(np.nanmedian(main[f"A/{probe}/_seconds"])) if f"A/{probe}/_seconds" in main.columns else float("nan")}
        for name in ("_true_peak_push_n", "_true_peak_pull_n", "_true_peak_raw_n"):
            col = f"A/{probe}/{name}"
            if col in main.columns:
                row["worst" + name] = float(np.nanmax(main[col]))
                row["p99" + name] = float(np.nanquantile(main[col], 0.99))
        side = f"A/{probe}/_side"
        if side in main.columns:
            row["side_effects"] = {str(k): int(v) for k, v in main[side].value_counts().items()}
        errors = f"A/{probe}/_error"
        row["errors"] = int(main[errors].notna().sum()) if errors in main.columns else 0
        out["probes"][probe] = row
    return out


def md_first_look(look: dict) -> str:
    """The first look as markdown: one line per source with its per-class medians."""
    classes = list(look["class_counts"])
    lines = [f"`{look['connector']}`, {look['split']} seeds, claim `{look['predicate']}`, {look['rows']} states. "
             "Median of each source's main feature per injected class (10th to 90th percentile).", "",
             "| Source | Feature | " + " | ".join(classes) + " | AUC of its score |", "|---|---|" + "---|" * len(classes) + "---|"]
    for source, row in look["sources"].items():
        cells = []
        for c in classes:
            p = row["per_class"].get(c)
            cells.append("" if not p or not p["n"] else f"{p['median']:.2f} ({p['p10']:.2f} to {p['p90']:.2f})")
        auc = row["auc_of_score"]
        lines.append(f"| `{source}` | `{row['feature']}` | " + " | ".join(cells) + f" | {'' if auc != auc else f'{auc:.3f}'} |")
    return "\n".join(lines)


def md_probes(look: dict) -> str:
    lines = ["| Probe | Harm: good plugs left bad | Repair: bad plugs left good | Worst real push, N | Worst real pull, N | Seconds |",
             "|---|---|---|---|---|---|"]
    for probe, row in look["probes"].items():
        h, r = row["harm"], row["repair"]
        lines.append(
            f"| `{probe}` | {100 * h['rate']:.2f}% ({h['k']} of {h['n']}, {100 * h['lo']:.2f} to {100 * h['hi']:.2f}) | "
            f"{100 * r['rate']:.1f}% ({r['k']} of {r['n']}, {100 * r['lo']:.1f} to {100 * r['hi']:.1f}) | "
            f"{row.get('worst_true_peak_push_n', float('nan')):.1f} | {row.get('worst_true_peak_pull_n', float('nan')):.1f} | "
            f"{row['seconds_median']:.1f} |")
    return "\n".join(lines)
