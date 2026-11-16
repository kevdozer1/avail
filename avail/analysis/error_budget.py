"""The error budget (the night of 2026-11-16, step 3): the defect bar as a total budget split across the cable loop's
witnesses through their alphas, chosen on calibration only (docs/07, the notes of 2026-11-16).

For each witness and each alpha on a grid, the scorers stay as fit on the fit split and the thresholds are set again on
the calibration split (``calibrate_cascade``, the weighted budget across a witness's own stages), under the crossing
rule given (``abstain``, as E3 and E3b ran, or ``accept``). On the calibration claims (in sample) that gives the claims
sent to the operator per cycle and the bad claims accepted. 2 union bounds on bad cables shipped per cycle are computed,
each the sum of alpha times bad claims per cycle in calibration (natural rows only: enrichment rows stand for a rare
failure and are left out of the rate) over its witnesses:

- **shipped**: ``released_stable`` and ``held``. A cable ships bad only through a false accept of its release claim, or
  as a wrong cable passed at ``held``, which no later check can see. This is the bound the gate and E4 use.
- **brief**: ``held``, ``at_port`` and ``released_stable``, the sum Kevin's brief names. Reported beside it.

The split chosen under a bound has the fewest claims sent to the operator plus bad claims passed on per cycle, with the
bound at most the budget, every alpha at most ``ALPHA_MAX`` and every stage keeping at least 1 / alpha_k - 1 bad claims.
With ``--max-good-to-operator`` it also keeps every witness at or under that share of good claims sent to the operator.

Usage::

    uv run python -m avail.analysis.error_budget --crossing accept --out <AVAIL_DATA>/runs/analysis/overnight2/budget.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

GRID = (0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.01, 0.011, 0.0115, 0.0125, 0.015, 0.02,
        0.025, 0.03, 0.04, 0.05)
ALPHA_MAX = 0.05
BUDGET = 0.005
PREDICATES = ("held", "at_port", "released_stable")
BOUNDS = {"shipped": ("held", "released_stable"), "brief": ("held", "at_port", "released_stable")}


def _natural(table, enriched_split: str | None):
    from ..config import load_experiment

    if enriched_split is None:
        return np.ones(len(table), dtype=bool)
    seeds = load_experiment().seeds.seeds(enriched_split)
    s = np.asarray(table["seed"], dtype=int)
    return ~((s >= seeds[0]) & (s <= seeds[-1]))


def curve(variant: str, predicate: str, crossing: str = "abstain", grid=GRID) -> dict:
    """The witness's in-sample decisions on its calibration claims at each alpha of the grid."""
    from ..config import load_experiment
    from ..station.mj_plug import cable_shift as cs
    from ..witness.cascade import calibrate_cascade, fit_scorers
    from ..zoo.stages import SKIP_FEATURES, STATION_ID

    stages = cs.check_stages(predicate)
    beta = load_experiment().bounds.beta
    fit = cs.check_table(variant, cs.CHECK_FIT_SPLIT, predicate)
    cal = cs.check_table(variant, cs.CHECK_CALIBRATION_SPLIT, predicate)
    natural = _natural(cal, cs.ENRICHED.get((predicate, cs.CHECK_CALIBRATION_SPLIT)))
    cycles = int(len(set(np.asarray(cal["seed"], dtype=int)[natural])))
    y = np.asarray(cal["y0"].fillna(False)).astype(bool)
    scorers, neg_share, pos_share = fit_scorers(fit, variant, predicate, stages, SKIP_FEATURES)
    rows = []
    for alpha in grid:
        c = calibrate_cascade(scorers, neg_share, pos_share, cal, variant, predicate, float(alpha), beta, stages,
                              station_id=STATION_ID, skip_features=SKIP_FEATURES, split_id=cs.CHECK_CALIBRATION_SPLIT,
                              crossing=crossing)
        d = c.decide_table(cal)
        dec = np.asarray(d["decision"])
        y_dec = np.asarray(d["y_dec"]).astype(bool)
        n_neg_k = [st.n_neg for st in c.stages]
        alpha_k = [st.alpha_k for st in c.stages]
        good = int((natural & y).sum())
        rows.append({
            "alpha": float(alpha), "alpha_k": alpha_k, "n_neg_k": n_neg_k,
            "enough_negatives": all(n >= (1.0 / a - 1.0) for n, a in zip(n_neg_k, alpha_k) if a > 0),
            "to_operator_per_cycle": float(((dec == "abstain") & natural).sum()) / cycles,
            "good_to_operator": int(((dec == "abstain") & natural & y).sum()), "good": good,
            "bad_accepted": int(((dec == "accept") & ~y_dec).sum()), "bad_at_decision": int((~y_dec).sum()),
            "bad_accepted_per_cycle": float(((dec == "accept") & ~y_dec & natural).sum()) / cycles,
            "good_rejected": int(((dec == "reject") & natural & y).sum()),
        })
    return {"predicate": predicate, "cycles": cycles, "bad_per_cycle": float((natural & ~y).sum()) / cycles, "rows": rows}


def choose(curves: dict, bound: tuple[str, ...], budget: float = BUDGET, alpha_max: float = ALPHA_MAX,
           max_good_to_operator: float | None = None) -> dict:
    names = list(curves)
    options = []
    for name in names:
        ok = [r for r in curves[name]["rows"] if r["alpha"] <= alpha_max + 1e-12 and r["enough_negatives"]]
        if max_good_to_operator is not None:
            ok = [r for r in ok if r["good_to_operator"] <= max_good_to_operator * r["good"]]
        options.append(ok)
    best = None
    for combo in itertools.product(*options):
        value = sum(curves[n]["bad_per_cycle"] * r["alpha"] for n, r in zip(names, combo) if n in bound)
        if value > budget + 1e-12:
            continue
        cost = sum(r["to_operator_per_cycle"] + r["bad_accepted_per_cycle"] for r in combo)
        key = (round(cost, 9), value)
        if best is None or key < best["_key"]:
            best = {"_key": key, "alphas": {n: r["alpha"] for n, r in zip(names, combo)}, "union_bound": value,
                    "cost_per_cycle": cost, "per_witness": {n: r for n, r in zip(names, combo)}}
    if best is None:
        least = None
        for combo in itertools.product(*options):
            value = sum(curves[n]["bad_per_cycle"] * r["alpha"] for n, r in zip(names, combo) if n in bound)
            least = value if least is None else min(least, value)
        return {"alphas": None, "note": "no split meets the budget", "least_bound": least}
    best.pop("_key")
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", default="latch")
    ap.add_argument("--budget", type=float, default=BUDGET)
    ap.add_argument("--crossing", default="accept", choices=("abstain", "accept"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    curves = {p: curve(args.variant, p, args.crossing) for p in PREDICATES}
    for p, c in curves.items():
        print(f"== {p}: {c['cycles']} calibration cycles, {c['bad_per_cycle']:.3f} bad claims per cycle")
        for r in c["rows"]:
            print(f"  alpha {r['alpha']:.4f} n_neg {r['n_neg_k']} ok {r['enough_negatives']} | good to operator "
                  f"{r['good_to_operator']}/{r['good']} | bad accepted {r['bad_accepted']}/{r['bad_at_decision']} | "
                  f"good rejected {r['good_rejected']}")
    out = {"crossing": args.crossing, "budget": args.budget, "curves": curves, "at_0.01": {}}
    for name, bound in BOUNDS.items():
        out["at_0.01"][name] = sum(curves[p]["bad_per_cycle"] * 0.01 for p in bound)
        out[name] = choose(curves, bound, args.budget)
        out[f"{name}_with_20pct_rule"] = choose(curves, bound, args.budget, max_good_to_operator=0.20)
        print(f"{name}: at alpha 0.01 everywhere {100 * out['at_0.01'][name]:.2f}% | chosen "
              f"{json.dumps({k: v for k, v in out[name].items() if k != 'per_witness'})} | with the 20% rule "
              f"{json.dumps({k: v for k, v in out[name + '_with_20pct_rule'].items() if k != 'per_witness'})}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
