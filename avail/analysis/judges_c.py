"""Step 2 of Kevin's 2026-11-12 brief: each judge on the hard-mode v1 near misses, next to the tug and the camera.

Evaluator-side. Reads the judges tables under ``AVAIL_DATA/zoo/<connector>/test_hard/judges/``
(1 per model and prompt variant, written by ``scripts/judges_s4c.py``) and the stored E1b features
of the same seeds. Simulates nothing and reads no new seed.

Every sampled state is a near miss (N1 and N2 for ``fric``, N1, N2 and N5 for ``latch``, by injected
class), so every one is bad when the claim is raised. For a judge and for the camera, which move
nothing, the false-accept rate is the share of the sampled states accepted. For the tug and the push,
which act on the plug, the rate is the share accepted and still bad after the probe acted, over the
same states (``leaves_bad``, the H1 measure), and the rate over the states still bad after it acted
is given beside it. The camera, push and tug use the scorers and thresholds of the published hard
witness, version v1, fit and calibrated by E1b.

A judge accepts when its answer says the condition holds. An answer that failed its schema, and a
failed call, count as not accepted, and are counted apart. 95% Wilson intervals throughout.

    uv run avail s4c judges            prints the tables and writes them to AVAIL_DATA/runs/s4c_judges/
"""

from __future__ import annotations

import json

import numpy as np

from ..stats import wilson

PROBES = ("vision.gap@claim", "probe.push_seat", "probe.tug")
WITNESS_VERSION = "v1"
LABELS = {"cheap_open": "Nex N2.5 mini (open weights)", "luna": "GPT-6 Luna (cheap closed)", "fable": "Claude Fable 5.2"}


def _cell(k: int, n: int) -> dict:
    lo, hi = wilson(k, n) if n else (float("nan"), float("nan"))
    return {"k": int(k), "n": int(n), "rate": (k / n) if n else float("nan"), "lo": lo, "hi": hi}


def _checks(connector: str, predicate: str) -> dict:
    from ..paths import data_root

    path = data_root() / "witness_hard" / connector / predicate / f"{WITNESS_VERSION}.json"
    return json.loads(path.read_text(encoding="utf-8"))["checks"]


def compare(connector: str) -> dict:
    import pandas as pd

    from ..evidence.fused import Scorer
    from ..witness.cascade import source_columns
    from ..zoo import store
    from .e1 import CHECKS
    from .run import load_split, primary_predicate

    predicate = primary_predicate(connector)
    folder = store.split_dir(connector, "test_hard") / "judges"
    feats = load_split(connector, "test_hard").set_index("seed")
    checks = _checks(connector, predicate)
    out: dict = {"connector": connector, "predicate": predicate, "witness": f"witness_hard {WITNESS_VERSION}", "runs": {}}
    for path in sorted(folder.glob("*.parquet")):
        model, variant = path.stem.split("__")
        judged = pd.read_parquet(path)
        if bool(judged["placeholder"].any()):
            continue
        seeds = [int(s) for s in judged["seed"]]
        sub = feats.loc[seeds]
        holds = judged.set_index("seed").loc[seeds, "holds"]
        valid = holds.notna().to_numpy()
        accepted = holds.fillna(False).astype(bool).to_numpy()
        bad = ~sub["y0"].astype(bool).to_numpy()
        n = len(seeds)
        row = {"model": model, "variant": variant, "states": n, "bad_at_claim": int(bad.sum()),
               "invalid_or_failed": int((~valid).sum()),
               "classes": {str(k): int(v) for k, v in sub["cls"].value_counts().sort_index().items()},
               "usd": float(judged["original_usd"].fillna(judged["usd"]).sum()),
               "judge": _cell(int((accepted & bad).sum()), n),
               "judge_over_valid": _cell(int((accepted & bad & valid).sum()), int(valid.sum())),
               "judge_by_class": {}}
        cls = sub["cls"].to_numpy()
        for c in sorted(set(cls)):
            m = cls == c
            row["judge_by_class"][str(c)] = _cell(int((accepted & bad & m).sum()), int(m.sum()))
        for check in PROBES:
            scorer = Scorer.from_dict(checks[check]["scorer"])
            zoo_pass, key, _, _ = CHECKS[check]
            cols = source_columns(sub.reset_index(), zoo_pass, key)
            frame = pd.DataFrame({name: np.asarray(sub[col], dtype=float) for name, col in cols.items()})
            s = scorer.score(frame)
            acc = np.isfinite(s) & (s > float(checks[check]["tau"]))
            label = f"{zoo_pass}/{key}/_y"
            y_after = sub[label].fillna(False).astype(bool).to_numpy() if label in sub.columns else ~bad
            row[check] = _cell(int((acc & ~y_after).sum()), n)
            row[check + " over still bad"] = _cell(int((acc & ~y_after).sum()), int((~y_after).sum()))
            row[check + " by_class"] = {str(c): _cell(int((acc & ~y_after & (cls == c)).sum()), int((cls == c).sum()))
                                        for c in sorted(set(cls))}
        out["runs"][path.stem] = row
    return out


def _pct(c: dict | None) -> str:
    if not c or not c.get("n"):
        return ""
    return f"{100 * c['rate']:.1f}% ({c['k']} of {c['n']}, {100 * c['lo']:.1f} to {100 * c['hi']:.1f})"


def md(result: dict) -> str:
    lines = [f"`{result['connector']}`, claim `{result['predicate']}`, test_hard near misses. Bad plugs accepted, "
             f"95% Wilson interval. Camera, push and tug from {result['witness']}, on the judge's own states.", "",
             "| Judge | Prompt | Judge accepts | Camera | Tug | Push | Invalid answers | Cost |",
             "|---|---|---|---|---|---|---|---|"]
    order = {"cheap_open": 0, "luna": 1, "fable": 2}
    for name, row in sorted(result["runs"].items(), key=lambda kv: (order.get(kv[1]["model"], 9), kv[1]["variant"])):
        lines.append(f"| {LABELS.get(row['model'], row['model'])} | {row['variant']} | {_pct(row['judge'])} | "
                     f"{_pct(row['vision.gap@claim'])} | {_pct(row['probe.tug'])} | {_pct(row['probe.push_seat'])} | "
                     f"{row['invalid_or_failed']} | ${row['usd']:.2f} |")
    lines += ["", "By class, judge accepts (camera, tug):", ""]
    for name, row in sorted(result["runs"].items(), key=lambda kv: (order.get(kv[1]["model"], 9), kv[1]["variant"])):
        parts = []
        for c, cell in row["judge_by_class"].items():
            cam, tug = row["vision.gap@claim by_class"][c], row["probe.tug by_class"][c]
            parts.append(f"{c} {cell['k']}/{cell['n']} ({cam['k']}, {tug['k']})")
        lines.append(f"- {LABELS.get(row['model'], row['model'])}, {row['variant']}: " + ", ".join(parts))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from ..paths import runs_dir

    out_dir = runs_dir() / "s4c_judges"
    out_dir.mkdir(parents=True, exist_ok=True)
    texts, results = [], {}
    for connector in ("fric", "latch"):
        result = compare(connector)
        results[connector] = result
        texts.append(md(result))
    text = "\n\n".join(texts)
    print(text, flush=True)
    (out_dir / "judges_vs_probes.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    (out_dir / "judges_vs_probes.json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8",
                                                   newline="\n")
    print(f"\nwritten to {out_dir}", flush=True)
    return 0
