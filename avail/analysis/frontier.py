"""docs/img/frontier.png (the night of 2026-11-16, step 7): what a bad-cable bar costs in calls for help.

Left: x is calls for help per 100 cycles (the shift card's assists), with a 95% percentile-bootstrap interval over
cycles. y is cycles with a bad cable shipped, log scale, with a Wilson 95% interval. A run with 0 such cycles is drawn at
its one-sided 95% upper bound as a downward marker. The dashed line is the 0.5% bar. Right, small: the flywheel's calls
per 100 cycles by day on its held-out set, when that run exists. Everything comes from ledgers and JSON files on disk.

Usage::

    uv run python -m avail.analysis.frontier --out docs/img/frontier.png [--point "E4|<shift dir>|avail_simop"] ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

INK, MUTED, BAR = "#1f2328", "#5b6270", "#d64545"
COLORS = {"skill_done": "#8a8f98", "E3": "#c98a1d", "E3b": "#3d7fd6", "E4": "#1a9e6a", "dev": "#1a9e6a"}


def point(label: str, shift_dir, arm: str, seeds: tuple[int, int] | None = None) -> dict | None:
    from ..stats import bootstrap_mean_upper, cp_upper, wilson
    from .calls import load

    shift_dir = Path(shift_dir)
    if not shift_dir.is_dir():
        return None
    data = load(shift_dir, arm, "latch")
    ends = sorted(data["ends"], key=lambda e: int(e.get("seed", 0)))
    if seeds is not None:
        ends = [e for e in ends if seeds[0] <= int(e["seed"]) <= seeds[1]]
    n = len(ends)
    if n == 0:
        return None
    calls = np.array([int(e.get("assists", 0)) for e in ends], float)
    bad = sum(1 for e in ends if int(e.get("uncaught", 0)) > 0)
    lo, hi = wilson(bad, n)
    if calls.any():
        c_hi = 100 * bootstrap_mean_upper(calls, resamples=2000, seed=40000, confidence=0.975)
        c_lo = -100 * bootstrap_mean_upper(-calls, resamples=2000, seed=40000, confidence=0.975)
    else:
        c_lo = c_hi = 0.0
    return {"label": label, "arm": arm, "cycles": n, "bad": bad, "rate": bad / n, "lo": lo, "hi": hi,
            "ucb95": cp_upper(bad, n), "calls_per_100": 100 * calls.mean(), "calls_lo": c_lo, "calls_hi": c_hi}


def draw(points: list[dict], flywheel: dict | None, out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, fx) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=150, gridspec_kw={"width_ratios": [2.4, 1]})
    for p in points:
        color = next((COLORS[key] for key in sorted(COLORS, key=len, reverse=True) if p["label"].startswith(key)), INK)
        x = p["calls_per_100"]
        xerr = [[x - p["calls_lo"]], [p["calls_hi"] - x]]
        if p["bad"] == 0:
            y = 100 * p["ucb95"]
            ax.errorbar(x, y, xerr=xerr, fmt="v", ms=9, color=color, ecolor=color, elinewidth=1.5, capsize=3)
            text = f"{p['label']}\n0 of {p['cycles']:,} bad, bound\n{x:.0f} calls"
        else:
            y = 100 * p["rate"]
            yerr = [[y - 100 * p["lo"]], [100 * p["hi"] - y]]
            ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", ms=8, color=color, ecolor=color, elinewidth=1.5, capsize=3)
            text = f"{p['label']}\n{p['bad']:,} of {p['cycles']:,} bad\n{x:.0f} calls"
        lift = 16 if 0.2 < y < 1.2 else 3           # a label next to the bar line sits above it
        ax.annotate(text, (x, y), textcoords="offset points", xytext=(9, lift), fontsize=9, color=INK,
                    va="bottom" if lift > 3 else "center")
    right = max(260, max(p["calls_hi"] for p in points) + 30)
    ax.axhline(0.5, ls="--", color=BAR, lw=1.5)
    ax.text(right - 4, 0.58, "bar 0.5%", color=BAR, fontsize=9, ha="right")
    ax.set_yscale("log")
    ax.set_ylim(0.01, 100)
    ax.set_xlim(-8, right)
    ax.set_yticks([0.01, 0.1, 0.5, 1, 10, 100], ["0.01%", "0.1%", "0.5%", "1%", "10%", "100%"])
    ax.set_xlabel("calls for help per 100 cycles")
    ax.set_ylabel("cycles with a bad cable shipped")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
        fx.spines[s].set_visible(False)
    if flywheel and flywheel.get("days"):
        d = flywheel["days"]
        xs = [r["day"] for r in d]
        ys = [r["calls_per_100"] for r in d]
        lo = [r.get("calls_lo", r["calls_per_100"]) for r in d]
        hi = [r.get("calls_hi", r["calls_per_100"]) for r in d]
        fx.fill_between(xs, lo, hi, color=COLORS["E4"], alpha=0.18, lw=0)
        fx.plot(xs, ys, "-o", color=COLORS["E4"], lw=2, ms=5)
        fx.set_xlabel("day")
        fx.set_ylabel("calls per 100 cycles, held-out")
        fx.set_title("flywheel (dev)", fontsize=10, color=INK, loc="left")
        fx.set_ylim(0, max(hi) * 1.25 if max(hi) > 0 else 1)
    else:
        fx.text(0.5, 0.5, "flywheel: not run yet", ha="center", va="center", color=MUTED, fontsize=10,
                transform=fx.transAxes)
        fx.set_axis_off()
    fig.text(0.01, 0.01, "simulation, simulated operator, latch", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    from ..paths import runs_dir

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--point", action="append", default=[], help="label|shift dir|arm, added after the defaults")
    ap.add_argument("--flywheel", type=Path, default=None, help="the flywheel's summary JSON")
    ap.add_argument("--json", type=Path, default=None, help="also write the points as JSON")
    args = ap.parse_args(argv)
    shifts = runs_dir() / "shifts"
    pts = [point("skill_done, E3b seeds", shifts / "e3b_final_latch", "flag"),
           point("E3", shifts / "e3_final_latch", "avail_human"),
           point("E3b", shifts / "e3b_final_latch", "avail_simop")]
    for spec in args.point:
        label, path, arm = spec.split("|")
        pts.append(point(label, Path(path), arm))
    pts = [p for p in pts if p]
    fly = json.loads(args.flywheel.read_text(encoding="utf-8")) if args.flywheel and args.flywheel.is_file() else None
    draw(pts, fly, args.out)
    if args.json:
        args.json.write_text(json.dumps({"points": pts, "flywheel": fly}, indent=1, default=float), encoding="utf-8",
                             newline="\n")
    print(json.dumps(pts, indent=1, default=float), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
