"""S4b: the VLM judges on hard-mode near misses, next to the camera and the probes on the same states. Evaluator-side.

The judge run of docs/07 (S4b notes) writes 1 table per model and variant under
``AVAIL_DATA/zoo/<connector>/test_hard/judges/``. Every sampled state is a near miss (N1, N2,
and N5 for latch, by injected class), so every one is bad when the claim is raised, and a judge's
accept rate on the sample is its ``leaves_bad`` there. A judge accepts when its answer is yes
(``holds`` true). An answer that failed its schema, and a failed call, count as not accepted
and are counted apart.

On exactly the same seeds, the camera check (``vision.gap@claim``) and the probes (push and tug)
are scored with the scorers and thresholds that E1b fit and calibrated, read from the published
hard-mode witness, and judged against the truth after they acted, as in E1.

    avail s4b judges              prints the comparison for every judges table on disk
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..stats import wilson
from .e1 import CHECKS

COMPARED = ("vision.gap@claim", "probe.push_seat", "probe.tug")


def _rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n) if n else (float("nan"), float("nan"))
    return {"k": int(k), "n": int(n), "rate": k / n if n else float("nan"), "lo": lo, "hi": hi}


def _hard_checks(connector: str, predicate: str) -> dict:
    """The E1b checks (scorer and threshold) from the latest published hard-mode witness."""
    from ..paths import data_root
    from ..witness import store

    _, payload = store.load(connector, predicate, root=data_root() / "witness_hard")
    return payload["checks"]


def compare(connector: str, root=None) -> dict:
    """Every judges table of a connector's test_hard split, each next to the camera and the probes on its seeds."""
    import pandas as pd

    from ..evidence.fused import Scorer
    from ..zoo import store
    from .run import load_split, primary_predicate

    predicate = primary_predicate(connector)
    folder = store.split_dir(connector, "test_hard", root) / "judges"
    tables = sorted(folder.glob("*.parquet")) if folder.is_dir() else []
    out: dict = {"connector": connector, "runs": {}}
    if not tables:
        out["status"] = "no judges table on disk"
        return out
    feats = load_split(connector, "test_hard", root).set_index("seed")
    checks = _hard_checks(connector, predicate)
    for path in tables:
        judged = pd.read_parquet(path)
        seeds = [int(s) for s in judged["seed"] if int(s) in feats.index]
        sub = feats.loc[seeds]
        holds = judged.set_index("seed").loc[seeds, "holds"]
        valid = holds.notna().to_numpy()
        accepted = (holds.fillna(False).astype(bool)).to_numpy()
        bad = ~sub["y0"].astype(bool).to_numpy()
        row = {"states": len(seeds), "bad_at_claim": int(bad.sum()), "invalid_or_failed": int((~valid).sum()),
               "judge_leaves_bad": _rate(int((accepted & bad).sum()), len(seeds)),
               "placeholder": bool(judged["placeholder"].any()), "usd": float(judged["usd"].sum()),
               "classes": {str(k): int(v) for k, v in sub["cls"].value_counts().sort_index().items()}}
        for check in COMPARED:
            if check not in checks:
                continue
            scorer = Scorer.from_dict(checks[check]["scorer"])
            zoo_pass, key, _, _ = CHECKS[check]
            from ..witness.cascade import source_columns

            cols = source_columns(sub.reset_index(), zoo_pass, key)
            frame = pd.DataFrame({name: np.asarray(sub[col], dtype=float) for name, col in cols.items()})
            s = scorer.score(frame)
            acc = np.isfinite(s) & (s > float(checks[check]["tau"]))
            label = f"{zoo_pass}/{key}/_y"
            y = sub[label].fillna(False).astype(bool).to_numpy() if label in sub.columns else ~bad
            row[check] = _rate(int((acc & ~y).sum()), len(seeds))
        out["runs"][path.stem] = row
    return out


def md(result: dict) -> str:
    """The comparison as a markdown table: accepted and still bad, over the sampled near misses."""
    if not result.get("runs"):
        return f"`{result['connector']}`: {result.get('status', 'no runs')}"
    def cell(c):
        return "" if not c or not c.get("n") else f"{100 * c['rate']:.1f}% ({c['k']}/{c['n']})"
    lines = [f"| Judge run ({result['connector']}) | Judge accepts | Invalid or failed | Camera | Push | Tug | Cost |",
             "|---|---|---|---|---|---|---|"]
    for name, row in result["runs"].items():
        lines.append(f"| {name} | {cell(row['judge_leaves_bad'])} | {row['invalid_or_failed']} | "
                     f"{cell(row.get('vision.gap@claim'))} | {cell(row.get('probe.push_seat'))} | "
                     f"{cell(row.get('probe.tug'))} | ${row['usd']:.2f} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    for connector in ("fric", "latch"):
        result = compare(connector)
        print(md(result), flush=True)
        print(json.dumps(result, indent=1, default=float)[:4000], flush=True)
    return 0
