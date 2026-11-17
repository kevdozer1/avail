"""The pictures of docs/VISUALS.md: 1 idea each, readable on a phone, from ledgers alone. Evaluator-side.

    uv run python -m avail.analysis.visuals [--out docs/img/visuals] [--only fleet,fates,...]

Run it again when a new run lands: every figure is drawn from what is on disk under ``AVAIL_DATA/runs``.

- ``fleet``: 1,000 cycles as 1,000 squares, the same seeds for "Robot says done" (the driver's ``flag`` arm) and AVAIL
  (``avail_simop``). Red: a bad cable shipped. Gray: a cable left undone and declared undone.
- ``fates``: every step whose skill truly failed, and what fixed it.
- ``operator_hour``: 5 robots replayed side by side for 1 hour of their own cycles, with 1 operator.
- ``frontier_hero``: what each design cost in calls for help against the bad cycles it let through.
- ``harvest``: wrist frames at the moment a check caught a real failure, from a dev run of E4's configuration.
- ``filmstrip``: the showcase seed, both arms, 6 moments each.

E4's test cycles stored their frames as states, and the seed guard never builds a test seed's station again, so the
wrist frames are rendered from ``gate_e4``, 500 dev cycles of the configuration E4 froze. The showcase film's cycle is
the dev seed its rule picked. Colors are the film's (``avail.analysis.film``).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .film import CHECKING, CODE, COLOR, OPERATOR, STOPPED


def _hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(v) for v in rgb[:3])


C = {CODE: _hex(COLOR[CODE]), CHECKING: _hex(COLOR[CHECKING]), OPERATOR: _hex(COLOR[OPERATOR]),
     STOPPED: _hex(COLOR[STOPPED])}
GOOD, BAD, GRAY, PALE = "#3cbe64", "#e13c3c", "#9aa0a8", "#d5eedb"
INK, MUTED, IDLE = "#1f2328", "#5b6270", "#eceef1"
NOTE = "Simulation, simulated operator"
SHIFTS = {"E3": ("e3_final_latch", "avail_human"), "E3b": ("e3b_final_latch", "avail_simop"),
          "E4": ("e4_final_latch", "avail_simop"), "E5s": ("e5s_final_latch", "avail_simop"),
          "E5n": ("e5n_final_latch", "avail_simop")}
FLEET_SEEDS = (40000, 40999)          # the 1,000 seeds E4 ran with both arms
OPERATOR_HOUR_S = 3600.0
ROBOTS = 5
STEPS = ("pick", "carry", "insert", "release")
STEP_WORD = {"pick": "pick", "carry": "carry", "insert": "insert", "release": "release"}


def shifts_dir() -> Path:
    from ..paths import runs_dir

    return runs_dir() / "shifts"


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "svg.fonttype": "none",
                         "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})
    return plt


def _note(fig, text: str = NOTE) -> None:
    fig.text(0.99, 0.988, text, ha="right", va="top", fontsize=9, color=MUTED)


def _save(fig, out: Path, name: str) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.png", out / f"{name}.svg"]
    fig.savefig(paths[0], dpi=200)
    fig.savefig(paths[1])
    import matplotlib.pyplot as plt

    plt.close(fig)
    return paths


# ----------------------------------------------------------------------------- reading the ledgers
def _ends(shift: str, arm: str) -> list[dict]:
    from .calls import load

    folder = shifts_dir() / shift
    if not folder.is_dir():
        return []
    return sorted(load(folder, arm, "latch")["ends"], key=lambda e: int(e["seed"]))


def cycle_class(end: dict) -> str:
    """``bad`` (a cable declared done that is not), ``undone`` (a cable left undone and declared so), or ``good``."""
    if int(end.get("uncaught", 0)) > 0:
        return "bad"
    if any(not it.get("declared") for it in end.get("items", [])):
        return "undone"
    return "good"


def fleet_numbers(seeds: tuple[int, int] = FLEET_SEEDS, shift: str = "e4_final_latch") -> dict:
    out = {}
    for label, arm in (("Robot says done", "flag"), ("AVAIL", "avail_simop")):
        ends = [e for e in _ends(shift, arm) if seeds[0] <= int(e["seed"]) <= seeds[1]]
        out[label] = {"seeds": [int(e["seed"]) for e in ends], "class": [cycle_class(e) for e in ends]}
    return out


def fates(ends: list[dict], ivs: list[dict] | None = None) -> dict:
    """Every step whose 1st try truly failed (its claim false right after its witness ran), by what came of it:
    ``retry`` (a later plain try accepted, and true), ``rung`` (a recovery rung's claim accepted, and true), ``operator``
    (the operator answered or took over on that step), ``undone`` (left undone and declared so), ``later`` (accepted
    while false, then put right at a later step of the same cable) or ``escaped`` (a bad cable shipped)."""
    helped = set()
    for iv in ivs or []:
        helped.add((iv.get("cycle_id"), int(iv.get("item", -1)), iv.get("step")))
    count, by_step = Counter(), Counter()
    for e in ends:
        steps = {(s["item"], s["step"]): s for s in e.get("steps", [])}
        items = {it["item"]: it for it in e.get("items", [])}
        first = {}
        for c in e.get("claims", []):
            k = (c["item"], c["step"])
            if k not in first and int(c["attempt"]) == 0 and c.get("rung") is None:
                first[k] = c
        for k, c in first.items():
            if c["true"]:
                continue
            by_step[k[1]] += 1
            s = steps.get(k)
            if s is None or not s["declared"]:
                fate = "undone"
            elif not s["true"]:
                fate = "escaped" if items.get(k[0], {}).get("uncaught") else "later"
            elif (e["cycle_id"], int(k[0]), k[1]) in helped or s["by"] in ("takeover", "operator", "signoff"):
                fate = "operator"
            elif str(s["by"]).startswith("rung:"):
                fate = "rung"
            else:
                fate = "retry"
            count[fate] += 1
    return {"total": int(sum(count.values())), "fates": dict(count), "by_step": dict(by_step), "cycles": len(ends)}


def fates_numbers(shift: str = "e4_final_latch", arm: str = "avail_simop") -> dict:
    from .calls import load

    folder = shifts_dir() / shift
    if not folder.is_dir():
        return {}
    data = load(folder, arm, "latch")
    return fates(data["ends"], data["ivs"])


# ----------------------------------------------------------------------------- 1. fleet
E5_FLEET = {"E5n": ("AVAIL, E5n", (45000, 45999)), "E5s": ("AVAIL, stress", (86000, 86999))}


def e5_fleet_panels() -> dict:
    """1 more waffle per E5 arm that has ledgers: E5n's 1st 1,000 seeds, and E5s's 1,000 with stuck_release forced."""
    out = {}
    for arm, (label, (lo, hi)) in E5_FLEET.items():
        shift, driver_arm = SHIFTS[arm]
        ends = [e for e in _ends(shift, driver_arm) if lo <= int(e["seed"]) <= hi]
        if ends:
            out[label] = {"seeds": [int(e["seed"]) for e in ends], "class": [cycle_class(e) for e in ends]}
    return out


def draw_fleet(out: Path, numbers: dict | None = None, extra: dict | None = None) -> list[Path]:
    """2 waffles of 1,000 squares on the same seeds, plus any ``extra`` waffle (label to classes): by default each E5
    arm that has ledgers, on its own seeds."""
    plt = _style()
    numbers = dict(numbers or fleet_numbers())
    extra = e5_fleet_panels() if extra is None else extra
    if extra:
        numbers.update(extra)
    panels = list(numbers.items())
    cols, rows = 25, 40
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 7.4))
    axes = np.atleast_1d(axes)
    color = {"bad": BAD, "undone": GRAY, "good": PALE}
    for ax, (label, d) in zip(axes, panels):
        cls = d["class"]
        for i, c in enumerate(cls[: cols * rows]):
            r, q = divmod(i, cols)
            ax.add_patch(plt.Rectangle((q, rows - 1 - r), 0.86, 0.86, color=color[c], lw=0))
        ax.set_xlim(-0.2, cols)
        ax.set_ylim(-0.2, rows)
        ax.set_aspect("equal")
        ax.axis("off")
        n_bad = sum(1 for c in cls if c == "bad")
        ax.text(cols / 2, rows + 0.8, label, ha="center", va="bottom", fontsize=15, color=INK, clip_on=False)
        ax.text(cols / 2, rows + 3.4, f"{n_bad:,}", ha="center", va="bottom", fontsize=40, fontweight="bold",
                color=BAD if n_bad else INK, clip_on=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BAD), plt.Rectangle((0, 0), 1, 1, color=GRAY),
               plt.Rectangle((0, 0), 1, 1, color=PALE)]
    fig.legend(handles, ["bad cable shipped", "left undone, said so", "all 5 done"], loc="lower center", ncol=3,
               frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.035), handlelength=1.0, handleheight=1.0)
    same = f"The same {len(panels[0][1]['class']):,} E4 seeds for the left 2"
    fig.text(0.01, 0.012, f"1 square = 1 cycle of 5 cables. " + (same + ", each E5 arm on its own seeds" if extra
                                                                  else f"The same {len(panels[0][1]['class']):,} seeds per side"),
             ha="left", va="bottom", fontsize=9, color=MUTED)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.80, bottom=0.1, wspace=0.12)
    _note(fig)
    return _save(fig, out, "fleet")


# ----------------------------------------------------------------------------- 2. fates
FATE_ORDER = ("retry", "rung", "operator", "later", "undone", "escaped")
FATE_WORD = {"retry": "a plain retry", "rung": "a recovery rung", "operator": "the operator",
             "later": "caught at a later step", "undone": "left undone, said so", "escaped": "escaped"}
FATE_COLOR = {"retry": C[CODE], "rung": "#c96a10", "operator": C[OPERATOR], "later": C[CHECKING], "undone": GRAY,
              "escaped": BAD}


def draw_fates(out: Path, numbers: dict | None = None, title_run: str = "E4", name: str = "fates") -> list[Path]:
    """E4's failed steps, and with the default arguments 1 more figure per E5 arm that has ledgers (fates_e5n,
    fates_e5s)."""
    if numbers is None and title_run == "E4":
        paths = []
        for arm in ("E5n", "E5s"):
            shift, driver_arm = SHIFTS[arm]
            more = fates_numbers(shift, driver_arm)
            if more and more.get("cycles"):
                paths += draw_fates(out, more, arm + (", stuck_release forced" if arm == "E5s" else ""),
                                    f"fates_{arm.lower()}")
        return draw_fates(out, fates_numbers(), "E4", "fates") + paths
    plt = _style()
    total = numbers["total"]
    f = numbers["fates"]
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    gap = 0.014 * total
    shown = [k for k in FATE_ORDER if f.get(k, 0) > 0]
    min_h = 0.05 * total                                  # a band too thin to read is drawn at this height
    heights = [max(float(f[k]), min_h) for k in shown]
    span = sum(heights) + gap * (len(shown) - 1)
    x0, x1, w = 0.0, 1.0, 0.07
    ax.add_patch(plt.Rectangle((x0, (span - total) / 2), w, total, color="#4a4f57", lw=0))
    ax.text(x0, (span - total) / 2 + total + 0.02 * total, f"{total:,}\nfailed steps", ha="left", va="bottom",
            fontsize=14, color=INK, fontweight="bold")
    yl = (span - total) / 2 + total
    yr = span
    t = np.linspace(0, 1, 60)
    ease = 3 * t ** 2 - 2 * t ** 3
    for k, h in zip(shown, heights):
        n = float(f[k])
        left_top, left_bot = yl, yl - n
        right_top, right_bot = yr, yr - h
        xs = x0 + w + (x1 - x0 - 2 * w) * t
        top = left_top + (right_top - left_top) * ease
        bot = left_bot + (right_bot - left_bot) * ease
        ax.fill_between(xs, bot, top, color=FATE_COLOR[k], alpha=0.55, lw=0)
        ax.add_patch(plt.Rectangle((x1 - w, right_bot), w, h, color=FATE_COLOR[k], lw=0))
        pct = 100.0 * n / total
        label = f"{FATE_WORD[k]}  {int(n):,}" + (f"  ({pct:.0f}%)" if pct >= 1 else "")
        ax.text(x1 + 0.02, (right_top + right_bot) / 2, label, va="center", fontsize=12.5, color=INK,
                fontweight="bold" if k in ("retry", "rung") else "normal")
        yl -= n
        yr -= h + gap
    own = f.get("retry", 0) + f.get("rung", 0)
    ax.annotate("", xy=(x1 + 0.005, span), xytext=(x1 + 0.005, span - f.get("retry", 0) - f.get("rung", 0) - gap),
                arrowprops={"arrowstyle": "-", "color": INK, "lw": 0})
    ax.set_title(f"Fixed with no person: {100.0 * own / total:.0f}%", fontsize=18, color=INK, loc="left", pad=14,
                 fontweight="bold")
    ax.set_xlim(-0.02, 1.95)
    ax.set_ylim(-0.04 * total, span + 0.16 * total)
    ax.axis("off")
    fig.text(0.02, 0.012, f"{title_run}, {numbers['cycles']:,} cycles. A failed step: its skill's 1st try left the step "
             "not done", ha="left", va="bottom", fontsize=9, color=MUTED)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.06)
    _note(fig)
    return _save(fig, out, name)


# ----------------------------------------------------------------------------- 3. operator hour
def cycle_segments(events: list[dict], end: dict) -> list[tuple[str, float, int]]:
    """1 cycle as (mode, seconds, calls) in ledger order: a skill or a rung moving (CODE), a check (CHECKING), the
    operator's time to answer and to handle (OPERATOR, the same time model as robots per operator). Motion and checks
    are scaled to the cycle's own totals, so the cycle lasts exactly its recorded time."""
    seg = []
    for e in events:
        k, p = e["kind"], e["payload"]
        if k == "tool_call":
            seg.append([CODE, float(p.get("seconds", 0.0)), 0])
        elif k == "verdict":
            seg.append([CHECKING, float(p.get("cost_total", {}).get("seconds", 0.0)), 0])
        elif k == "intervention":
            s = float(p.get("seconds_waiting", 0.0)) + float(p.get("seconds_handling", 0.0))
            seg.append([OPERATOR, s, 1 if float(p.get("seconds_waiting", 0.0)) > 0 else 0])
    t = end.get("time_s", {})
    for mode, key in ((CODE, "motion"), (CHECKING, "verify")):
        have = sum(s[1] for s in seg if s[0] == mode)
        want = float(t.get(key, have))
        if have > 0:
            for s in seg:
                if s[0] == mode:
                    s[1] *= want / have
    return [(m, float(s), int(c)) for m, s, c in seg if s > 0]


def robot_cycles(shift: str = "e4_final_latch", arm: str = "avail_simop", robots: int = ROBOTS,
                 per_robot: int = 30, start: int = 40000, stride: int = 1000) -> list[list[list]]:
    """Each robot's consecutive cycles: robot r runs the seeds from ``start + r * stride`` in order."""
    from ..ledger.reader import iter_events

    folder = shifts_dir() / shift
    out = []
    for r in range(robots):
        lo = start + r * stride
        wanted = set(range(lo, lo + per_robot))
        cycles: dict[int, dict] = {}
        for b in sorted({s // 10 * 10 for s in wanted}):
            ledger = folder / f"{shift}_{arm}_latch_b{b}" / "ledger.jsonl"
            if not ledger.is_file():
                continue
            cur = None
            for e in iter_events(ledger):
                if e["kind"] == "cycle_start":
                    cur = {"events": [], "seed": int(str(e["payload"]["cycle_id"]).rsplit("_c", 1)[1])}
                elif e["kind"] == "cycle_end" and cur is not None:
                    if cur["seed"] in wanted:
                        cycles[cur["seed"]] = {"segments": cycle_segments(cur["events"], e["payload"]),
                                               "seed": cur["seed"], "bad": int(e["payload"].get("uncaught", 0))}
                    cur = None
                elif cur is not None:
                    cur["events"].append(e)
        out.append([cycles[s] for s in sorted(cycles)])
    return out


def replay_hour(robots: list[list[dict]], horizon: float = OPERATOR_HOUR_S) -> dict:
    """1 operator for all robots, first come first served. A robot that calls while the operator is busy waits
    (STOPPED) until the operator is free. Returns each lane's spans and the operator's busy time inside the hour."""
    lanes = [[] for _ in robots]
    op_spans = []
    t = [0.0] * len(robots)
    queue = [[(m, s, c) for cyc in cycles for (m, s, c) in cyc["segments"]] for cycles in robots]
    pos = [0] * len(robots)
    op_free = 0.0
    waited = 0.0
    calls = 0
    while True:
        live = [r for r in range(len(robots)) if t[r] < horizon and pos[r] < len(queue[r])]
        if not live:
            break
        r = min(live, key=lambda i: t[i])
        mode, dur, call = queue[r][pos[r]]
        pos[r] += 1
        if mode == OPERATOR:
            begin = max(t[r], op_free)
            if begin > t[r]:
                lanes[r].append((t[r], begin, STOPPED))
                waited += min(begin, horizon) - t[r]
            lanes[r].append((begin, begin + dur, OPERATOR))
            op_spans.append((begin, begin + dur, r))
            op_free = begin + dur
            t[r] = begin + dur
            calls += call
        else:
            lanes[r].append((t[r], t[r] + dur, mode))
            t[r] += dur
    busy = sum(max(0.0, min(b, horizon) - a) for a, b, _ in op_spans if a < horizon)
    return {"lanes": lanes, "operator": op_spans, "busy_s": busy, "busy_share": busy / horizon,
            "robot_wait_s": waited, "calls": calls, "horizon_s": horizon}


def draw_operator_hour(out: Path, hour: dict | None = None) -> list[Path]:
    plt = _style()
    if hour is None:
        hour = replay_hour(robot_cycles())
    horizon = hour["horizon_s"]
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    n = len(hour["lanes"])
    h = 0.62
    for r, spans in enumerate(hour["lanes"]):
        y = n - r
        for a, b, mode in spans:
            if a >= horizon:
                continue
            ax.broken_barh([(a / 60, (min(b, horizon) - a) / 60)], (y - h / 2, h), color=C[mode], lw=0)
        ax.text(-0.6, y, f"robot {r + 1}", ha="right", va="center", fontsize=12, color=INK)
    y = 0
    ax.broken_barh([(0, horizon / 60)], (y - h / 2, h), color=IDLE, lw=0)
    for a, b, r in hour["operator"]:
        if a >= horizon:
            continue
        ax.broken_barh([(a / 60, (min(b, horizon) - a) / 60)], (y - h / 2, h), color=C[OPERATOR], lw=0)
        if (min(b, horizon) - a) / 60 >= 1.0:
            ax.text((a + min(b, horizon)) / 120, y, str(r + 1), ha="center", va="center", fontsize=8, color="white",
                    fontweight="bold")
    ax.text(-0.6, y, "operator", ha="right", va="center", fontsize=12, color=C[OPERATOR], fontweight="bold")
    ax.axhline(0.55, color="#c9ccd1", lw=0.8)
    ax.set_xlim(0, horizon / 60)
    ax.set_ylim(-0.6, n + 0.6)
    ax.set_yticks([])
    ax.set_xticks([0, 15, 30, 45, 60], ["0", "15", "30", "45", "60"])
    ax.set_xlabel("minutes", fontsize=11, color=MUTED, labelpad=2)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"1 operator, 5 robots: busy {100 * hour['busy_share']:.0f}% of the hour", fontsize=17, color=INK,
                 loc="left", fontweight="bold", pad=12)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C[m]) for m in (CODE, CHECKING, OPERATOR, STOPPED)]
    ax.legend(handles, ["robot moving", "checking", "with the operator", "waiting for the operator"], loc="upper center",
              bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False, fontsize=10.5, handlelength=1.0)
    fig.text(0.01, 0.012, "E4 cycles replayed in order, real durations. Operator time per call: time to answer plus "
             "handling, as in robots per operator", ha="left", va="bottom", fontsize=8.5, color=MUTED)
    fig.subplots_adjust(left=0.11, right=0.985, top=0.88, bottom=0.2)
    _note(fig)
    return _save(fig, out, "operator_hour")


# ----------------------------------------------------------------------------- 4. frontier, simplified
def hero_points() -> list[dict]:
    from .frontier import point

    pts = []
    sd = point("Robot says done", shifts_dir() / "e4_final_latch", "flag", seeds=FLEET_SEEDS)
    if sd:
        pts.append(sd)
    for label in ("E3", "E3b", "E4", "E5s", "E5n"):
        shift, arm = SHIFTS[label]
        p = point(label, shifts_dir() / shift, arm)
        if p:
            pts.append(p)
    return pts


def draw_frontier_hero(out: Path, pts: list[dict] | None = None) -> list[Path]:
    plt = _style()
    pts = pts or hero_points()
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    colors = {"Robot says done": "#8a8f98", "E3": "#c98a1d", "E3b": C[CHECKING], "E4": GOOD, "E5n": "#0f7a52",
              "E5s": "#6b4bb8"}
    xy = {}
    for p in pts:
        x, y = p["calls_per_100"], 100 * max(p["rate"], 1e-9)
        lo, hi = 100 * p["lo"], 100 * p["hi"]
        col = colors.get(p["label"], INK)
        hollow = p["label"] == "E5s"                  # stuck_release forced on every cable: drawn apart from the chain
        face = "white" if hollow else col
        if p["bad"] == 0:
            y = 100 * p["ucb95"]
            ax.plot([x], [y], marker="v", ms=12, color=col, mfc=face, ls="none")
        else:
            ax.errorbar([x], [y], yerr=[[y - lo], [hi - y]], fmt="s" if hollow else "o", ms=10, color=col, mfc=face,
                        capsize=4, lw=1.6)
        if p["calls_hi"] > p["calls_lo"]:
            ax.plot([p["calls_lo"], p["calls_hi"]], [y, y], color=col, lw=1.6)
        xy[p["label"]] = (x, y)
        dx, ha = (8, "left")
        if p["label"] in ("E3b", "E5s"):           # E5s sits at the right edge: its label goes to its left
            dx, ha = (-10, "right")
        ax.annotate("E5s, stress" if hollow else p["label"], (x, y), xytext=(dx, 8), textcoords="offset points", ha=ha,
                    fontsize=14, color=col, fontweight="bold")
    chain = [k for k in ("E3", "E3b", "E4", "E5n") if k in xy]
    for a, b in zip(chain, chain[1:]):
        ax.annotate("", xy=xy[b], xytext=xy[a], arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1.4,
                                                            "shrinkA": 9, "shrinkB": 9, "connectionstyle": "arc3,rad=0.15"})
    ax.axhline(0.5, color=BAD, ls="--", lw=1.4)
    ax.text(0.02, 0.5, "bar 0.5%", transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=11, color=BAD)
    ax.set_yscale("log")
    floor = min([0.01] + [100 * p["lo"] for p in pts if p["bad"] > 0 and p["lo"] > 0])
    ax.set_ylim(10 ** np.floor(np.log10(floor)), 100)
    ticks = [t for t in (0.001, 0.01, 0.1, 0.5, 1, 10, 100) if t >= 10 ** np.floor(np.log10(floor))]
    ax.set_yticks(ticks, [f"{t:g}%" for t in ticks])
    ax.set_xlim(-10, max(p["calls_hi"] for p in pts) * 1.12 + 5)
    ax.set_xlabel("calls for help per 100 cycles")
    ax.set_ylabel("cycles with a bad cable")
    for s in ("right", "top"):
        ax.spines[s].set_visible(False)
    ax.set_title("Fewer bad cables, fewer calls", fontsize=18, color=INK, loc="left", fontweight="bold", pad=12)
    note = "95% intervals. 5,000 cycles per run, 1,000 for robot says done"
    if "E5s" in xy:
        note += " and for E5s, whose every cable's 1st release sticks"
    fig.text(0.01, 0.012, note, ha="left", va="bottom", fontsize=8.5, color=MUTED)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.9, bottom=0.14)
    _note(fig)
    return _save(fig, out, "frontier_hero")


# ----------------------------------------------------------------------------- 5. harvest
def caught_claims(shift: str, arm: str = "avail_simop") -> list[dict]:
    """Every 1st-try claim a check rejected or abstained on while truth said the step was not done, with its evidence,
    from a shift's ledgers (the frame time comes from the frame log and the skill's own seconds)."""
    from ..ledger.reader import iter_events

    out = []
    for ledger in sorted((shifts_dir() / shift).glob(f"*_{arm}_latch_b*/ledger.jsonl")):
        claims, ev, calls, verdicts, cur = {}, defaultdict(dict), {}, {}, []
        for e in iter_events(ledger):
            k, p = e["kind"], e["payload"]
            if k == "claim":
                claims[p["claim_id"]] = p
            elif k == "evidence":
                ev[p["claim_id"]][p["source_id"]] = p["features"]
            elif k == "tool_call" and p.get("tool") == "wam_act":
                calls[p["claim_id"]] = p
            elif k == "verdict":
                verdicts[p["claim_id"]] = p
            elif k == "cycle_end":
                for c in p.get("claims", []):
                    if c["true"] or c["decision"] == "accept" or int(c["attempt"]) != 0 or c.get("rung") is not None:
                        continue
                    cid = next((i for i, cl in claims.items() if cl["cycle_id"] == p["cycle_id"]
                                and cl["step_id"] == f"{c['item']}.{c['step']}" and int(cl["context"]["attempt"]) == 0),
                               None)
                    if cid is None or cid not in verdicts:
                        continue
                    out.append({"cycle_id": p["cycle_id"], "seed": int(p["seed"]), "item": int(c["item"]),
                                "step": c["step"], "predicate": c["predicate"], "decision": c["decision"],
                                "evidence": ev.get(cid, {}), "seconds": float(calls.get(cid, {}).get("seconds", 0.0)),
                                "frames": str(ledger.parent / "frames" / f"{p['cycle_id']}.parquet")})
                claims, ev, calls, verdicts = {}, defaultdict(dict), {}, {}
    return out


def caught_per_1000(shift: str = "e4_final_latch", arm: str = "avail_simop") -> tuple[int, int, float]:
    """Claims a check said no to while truth agreed (every try), per 1,000 cycles: each is a labeled failure."""
    ends = _ends(shift, arm)
    n = sum(1 for e in ends for c in e.get("claims", []) if not c["true"] and c["decision"] != "accept")
    return n, len(ends), 1000.0 * n / max(1, len(ends))


def harvest_line(c: dict) -> str:
    """The check's signal, then its conclusion, in a few words."""
    ev, d = c["evidence"], c["decision"]
    g = lambda s, k, default=None: ev.get(s, {}).get(k, default)          # noqa: E731
    no = "unsure" if d == "abstain" else "no"
    if c["step"] == "pick":
        if g("passive.jaw_hold", "closed_empty"):
            sig = "jaws shut on nothing"
        elif g("vision.cable_tag", "mismatch"):
            sig = "wrong color in the jaws"
        elif not g("vision.cable_tag", "present", 1.0):
            sig = "no cable color seen"
        else:
            sig = f"jaws at {g('passive.jaw_hold', 'jaw_mm', 0.0):.1f} mm"
        return f"{sig}: held? {no}"
    if c["step"] == "carry":
        lat = g("passive.hover_pose", "lateral_mm")
        if g("passive.hover_pose", "port_match") == 0.0:
            sig = "over another port"
        elif g("passive.jaw_hold", "closed_empty"):
            sig = "nothing in the jaws"
        elif lat is not None:
            sig = f"{lat:.1f} mm off the port"
        else:
            sig = "pose unread"
        return f"{sig}: at port? {no}"
    if c["step"] == "insert":
        if g("probe.tug", "disp_mm") is not None:
            sig = f"pull moved it {g('probe.tug', 'disp_mm'):.1f} mm"
        elif g("probe.push_seat", "travel_mm") is not None:
            sig = f"push went {g('probe.push_seat', 'travel_mm'):.1f} mm more"
        else:
            sig = "no click" if not g("passive.click", "clicked") else "click, off depth"
        return f"{sig}: latched? {no}"
    if g("passive.jaw_hold", "open") == 0.0:
        sig = "jaws did not open"
    elif g("passive.release_record", "lifted_closed") == 1.0:
        sig = "lifted with jaws shut"
    elif g("vision.gap", "plug_found") == 0.0:
        sig = "no plug at the port"
    else:
        gap = g("vision.gap", "gap_mm_est")
        sig = f"gap {gap:.1f} mm" if gap is not None else "plug moved"
    return f"{sig}: released? {no}"


def _pick_harvest(caught: list[dict], per_step: int = 3) -> list[dict]:
    """3 per step, the most common signals first, at most 1 per cycle, in seed order within a signal."""
    chosen, used = [], set()
    for step in STEPS:
        rows = [c for c in caught if c["step"] == step]
        by_line = defaultdict(list)
        for c in rows:
            by_line[harvest_line(c)].append(c)
        lines = sorted(by_line, key=lambda k: (-len(by_line[k]), k))
        picked, nxt = [], {line: 0 for line in lines}
        while len(picked) < per_step and any(nxt[line] < len(by_line[line]) for line in lines):
            for line in lines:                              # round robin: each signal once, then again
                while nxt[line] < len(by_line[line]) and by_line[line][nxt[line]]["cycle_id"] in used:
                    nxt[line] += 1
                if nxt[line] < len(by_line[line]) and len(picked) < per_step:
                    c = by_line[line][nxt[line]]
                    picked.append(c)
                    used.add(c["cycle_id"])
                    nxt[line] += 1
        chosen.extend(picked)
    return chosen


def wrist_frame(c: dict) -> np.ndarray:
    """The wrist camera at the claim: the end of the skill's 1st try of that step, re-rendered from the frame log."""
    import pandas as pd

    from ..station.mj_plug.cable_replay import Replay

    frames = pd.read_parquet(c["frames"])
    scene = json.loads(Path(c["frames"].replace(".parquet", ".scene.json")).read_text(encoding="utf-8"))
    rows = frames[(frames["item"] == c["item"]) & (frames["step"] == c["step"])]
    t_claim = float(rows["t"].iloc[0]) + c["seconds"]
    idx = int(rows.index[np.searchsorted(rows["t"].to_numpy(float), t_claim - 1e-6).clip(0, len(rows) - 1)])
    rep = Replay(c["seed"], "latch", frames, scene)
    try:
        return rep.render(idx, camera="right_camera_d405")
    finally:
        rep.close()


def draw_harvest(out: Path, shift: str = "gate_e4", rate_shift: str = "e4_final_latch") -> list[Path]:
    plt = _style()
    chosen = _pick_harvest(caught_claims(shift))
    n, cycles, per = caught_per_1000(rate_shift)
    fig, axes = plt.subplots(3, 4, figsize=(10.0, 7.6))
    for j, step in enumerate(STEPS):
        axes[0, j].set_title(STEP_WORD[step], fontsize=16, color=INK, fontweight="bold", pad=8)
        col = [c for c in chosen if c["step"] == step]
        for i in range(3):
            ax = axes[i, j]
            ax.axis("off")
            if i >= len(col):
                continue
            ax.imshow(wrist_frame(col[i]))
            ax.text(0.5, -0.04, harvest_line(col[i]), transform=ax.transAxes, ha="center", va="top", fontsize=10.5,
                    color=INK)
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, ec=BAD, lw=2.5))
    fig.text(0.5, 0.045, f"{per:,.0f} labeled failures per 1,000 cycles", ha="center", va="bottom", fontsize=17,
             color=INK, fontweight="bold")
    fig.text(0.01, 0.012, f"Wrist camera at each claim a check caught: dev cycles of E4's setup. Rate: E4, {n:,} "
             f"caught claims in {cycles:,} cycles", ha="left", va="bottom", fontsize=8.5, color=MUTED)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.12, hspace=0.32, wspace=0.04)
    _note(fig)
    return _save(fig, out, "harvest")


# ----------------------------------------------------------------------------- 6. filmstrip
def strip_moments(run: dict) -> list[tuple[float, set[int]]]:
    """6 moments of a recorded cycle: the start, then the end of each cable's last release try. With each, the cables
    the run had declared done by then."""
    rel = {}
    for s in run["skills"]:
        if s["step"] == "release":
            rel[int(s["item"])] = max(float(s["t_end"]), rel.get(int(s["item"]), 0.0))
    done_at = {}
    for c in run["checks"]:
        if c["predicate"] == "released_stable" and c["decision"] == "accept":
            done_at[int(c["step_id"].split(".")[0])] = float(c["t_end"])
    if not run["checks"]:                                  # the skill's own flag declares each cable at its release
        done_at = dict(rel)
    for o in run["ops"]:                                    # an operator who signs off or takes over finishes the cable
        if o.get("step") == "release":
            done_at.setdefault(int(o["item"]), float(o.get("t_end", o["t"])))
    moments = [(0.0, set())]
    for k in sorted(rel):
        t = max(rel[k], done_at.get(k, rel[k]))
        moments.append((t, {i for i, tt in done_at.items() if tt <= t + 1e-6}))
    return moments


def draw_filmstrip(out: Path, rec: Path | None = None) -> list[Path]:
    from PIL import Image

    from ..paths import runs_dir
    from .film import Stage, load

    plt = _style()
    rec = Path(rec) if rec else runs_dir() / "showcase" / "film"
    rows = [("flag", "Robot says done"), ("avail_simop", "AVAIL")]
    fig, axes = plt.subplots(2, 6, figsize=(13.0, 4.9))
    seed = None
    for r, (arm, label) in enumerate(rows):
        run = load(rec, arm)
        seed = run["seed"]
        truth = {int(it["item"]): bool(it["true"]) for it in run["end"]["items"]}
        stage = Stage(rec, arm, run["seed"])
        try:
            pts = [stage.project(p) for p in stage.ports.values()]
            pts = [p for p in pts if p is not None]
            cx, cy = np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts])
            moments = strip_moments(run)
            for j, (t, done) in enumerate(moments[:6]):
                img = Image.fromarray(stage.frame(min(t, float(stage.t[-1])))).convert("RGB")
                w, h = 700, 455
                x0 = int(np.clip(cx - w / 2 - 40, 0, img.width - w))
                y0 = int(np.clip(cy - h / 2 - 70, 0, img.height - h))
                ax = axes[r, j]
                ax.imshow(np.asarray(img.crop((x0, y0, x0 + w, y0 + h)).resize((430, 280))))
                ax.set_xticks([])
                ax.set_yticks([])
                last = j == min(5, len(moments) - 1)
                ok = all(truth.values())
                for s in ax.spines.values():
                    s.set_color((GOOD if ok else BAD) if last else "#c9ccd1")
                    s.set_linewidth(5 if last else 1)
                for k in range(5):                          # the work order: 1 badge per cable, done or not yet
                    x = 0.12 + 0.19 * k
                    fill = GOOD if k in done else "#f5cd28"
                    if last and k in done:
                        ax.plot([x], [-0.14], marker="o", ms=21, mfc="none", mec=GOOD if truth.get(k, True) else BAD,
                                mew=2.6, transform=ax.transAxes, clip_on=False)
                    ax.plot([x], [-0.14], marker="o", ms=14, mfc=fill, mec=INK, mew=0.8, transform=ax.transAxes,
                            clip_on=False)
                    ax.text(x, -0.14, str(k + 1), transform=ax.transAxes, ha="center", va="center", fontsize=8.5,
                            color=INK, fontweight="bold", clip_on=False)
                if j == 0:
                    ax.set_ylabel(label, fontsize=14, color=INK, fontweight="bold", labelpad=8)
        finally:
            stage.close()
    for j, word in enumerate(["start", "cable 1", "cable 2", "cable 3", "cable 4", "cable 5"]):
        axes[0, j].set_title(word, fontsize=12, color=MUTED, pad=6)
    fig.text(0.01, 0.012, f"The showcase cycle, dev seed {seed:,}, run once per arm. Ring: the cable as it truly ended",
             ha="left", va="bottom", fontsize=8.5, color=MUTED)
    fig.subplots_adjust(left=0.04, right=0.995, top=0.92, bottom=0.12, hspace=0.42, wspace=0.05)
    _note(fig)
    return _save(fig, out, "filmstrip")


# ----------------------------------------------------------------------------- the whole set
FIGURES = ("fleet", "fates", "operator_hour", "frontier_hero", "harvest", "filmstrip")


def numbers() -> dict:
    """The numbers each figure states, for docs/VISUALS.md and the reports."""
    fl = fleet_numbers()
    hour = replay_hour(robot_cycles())
    n, cycles, per = caught_per_1000()
    return {"fleet": {k: {"cycles": len(v["class"]), **Counter(v["class"])} for k, v in fl.items()},
            "fates": fates_numbers(), "operator_hour": {k: hour[k] for k in ("busy_s", "busy_share", "robot_wait_s",
                                                                             "calls", "horizon_s")},
            "frontier_hero": hero_points(), "harvest": {"caught_claims": n, "cycles": cycles, "per_1000": per}}


def main(argv: list[str] | None = None) -> int:
    from ..paths import REPO_ROOT

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "img" / "visuals")
    ap.add_argument("--only", default="", help="comma separated, from " + ", ".join(FIGURES))
    ap.add_argument("--json", type=Path, default=None, help="also write the numbers as JSON")
    args = ap.parse_args(argv)
    todo = [f for f in (args.only.split(",") if args.only else FIGURES) if f]
    drawn = {}
    for name in todo:
        paths = globals()[f"draw_{name}"](args.out)
        drawn[name] = [str(p) for p in paths]
        print(f"{name}: {', '.join(Path(p).name for p in paths)}", flush=True)
    if args.json:
        args.json.write_text(json.dumps(numbers(), indent=1, default=float), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
