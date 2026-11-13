"""The zoo's figures and markdown tables (E1, E2 and their hard-mode twins), from analysis results alone.

Every number drawn or printed here comes out of an analysis result that was computed from
zoo tables on disk. Nothing here runs a robot or reads ground truth by itself.

Figures follow a fixed, validated palette (1 hue for magnitude, 2 hues at most for series)
and carry 1 idea each with few words. Sim and real numbers never share a figure.

Moved here from ``avail.shift.report`` in S5, with ``CLASS_NAMES`` and ``CHECK_NAMES``: they
name the plug station's failure classes and checks, which is testbed knowledge.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

# palette: validated with the dataviz validator on the light surface (2026-11-11)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834")
BLUE_RAMP = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
DPI = 150

CLASS_NAMES = {
    "P0": "seated", "N1": "0.5 to 1 mm short", "N2": "1 to 2 mm short", "N3": "2 to 4 mm short",
    "N4": "mouth only", "N5": "unlatched", "N6": "wrong port", "N7": "tilted jam", "N8": "dropped",
    "N9": "not released",
}
CHECK_NAMES = {
    "passive.skill_flag": "skill's own flag", "passive.pose_residual": "tool pose", "passive.jaw_state": "gripper reading",
    "passive.click": "click in the insertion", "vision.gap@claim": "camera, wrist view", "vision.gap@relook": "camera, closer look",
    "probe.wiggle": "wiggle", "probe.push_seat": "push", "probe.tug": "tug", "instr.link": "link signal",
    "fused.passive": "all passive signals, fused", "fused.all": "push, tug and look, fused",
}


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": INK_2,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })
    return plt


def _pct(x: float, digits: int = 0) -> str:
    if x is None or not math.isfinite(x):
        return "n/a"
    return f"{100 * x:.{digits}f}%"


def _title(fig, title: str, subtitle: str) -> None:
    fig.text(0.02, 0.975, title, ha="left", va="top", fontsize=13, fontweight="semibold", color=INK)
    fig.text(0.02, 0.975 - 0.34 / fig.get_figheight(), subtitle, ha="left", va="top", fontsize=9.5, color=INK_2)


def fig_far_by_check(table: dict, path: Path, title: str, subtitle: str, checks: list[str] | None = None,
                     key: str = "far") -> Path:
    """Bars: the share of bad plugs each check accepts, with its Wilson interval."""
    plt = _mpl()
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath

    checks = [c for c in (checks or list(table)) if c in table and table[c][key]["n"] > 0]
    checks = sorted(checks, key=lambda c: -table[c][key]["rate"])
    n = len(checks)
    slot = 0.34
    fig_h = 0.95 + slot * n + 0.45
    fig = plt.figure(figsize=(8.4, fig_h), dpi=DPI)
    ax = fig.add_axes([0.27, 0.45 / fig_h, 0.50, slot * n / fig_h])
    ax.set_xlim(0, 1.0)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([CHECK_NAMES.get(c, c) for c in checks], fontsize=10)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"], fontsize=9)
    ax.tick_params(length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.canvas.draw()
    to_px = ax.transData.transform
    thick = 20.0 * DPI / 150.0                               # bar thickness in pixels, under the 24 px cap
    radius = 4.0 * DPI / 150.0
    for i, check in enumerate(checks):
        cell = table[check][key]
        x0, y = to_px((0.0, i))
        x1, _ = to_px((cell["rate"], i))
        if x1 - x0 >= 1.0:
            r = min(radius, (x1 - x0))
            top, bot = y + thick / 2, y - thick / 2
            verts = [(x0, bot), (x1 - r, bot), (x1, bot), (x1, bot + r), (x1, top - r), (x1, top), (x1 - r, top), (x0, top), (x0, bot)]
            codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CURVE3, MPath.CURVE3,
                     MPath.LINETO, MPath.CLOSEPOLY]
            fig.add_artist(PathPatch(MPath(verts, codes), facecolor=SERIES[0], edgecolor="none", transform=None, zorder=3))
        ax.plot([cell["lo"], cell["hi"]], [i, i], color=INK_2, linewidth=1.2, solid_capstyle="butt", zorder=4)
        ax.text(min(max(cell["hi"], cell["rate"]) + 0.015, 1.02), i, f"{_pct(cell['rate'])}", va="center", ha="left",
                fontsize=10, color=INK, clip_on=False)
        ax.text(1.13, i, f"{cell['k']} of {cell['n']}", va="center", ha="left", fontsize=8.5, color=MUTED, clip_on=False)
    _title(fig, title, subtitle)
    fig.text(0.27, 0.12 / fig_h, "Line: 95% Wilson interval. Gray: bad plugs accepted, of bad plugs seen.", fontsize=8.5,
             color=MUTED, ha="left", va="bottom")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def fig_class_grid(table: dict, path: Path, title: str, subtitle: str, checks: list[str] | None = None,
                   classes: list[str] | None = None, key: str = "by_class", min_n: int = 5) -> Path:
    """Grid: check by failure class, share of bad plugs of that class accepted. One hue, darker is worse."""
    plt = _mpl()
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    from matplotlib.patches import Rectangle

    checks = [c for c in (checks or list(table)) if c in table]
    all_classes = sorted({c for chk in checks for c in table[chk][key]})
    classes = [c for c in (classes or all_classes) if c in all_classes and c != "P0"]
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    cw, ch = 0.92, 0.36
    fig_w, fig_h = 2.6 + cw * len(classes) + 0.3, 1.45 + ch * len(checks) + 0.3
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI)
    ax = fig.add_axes([2.5 / fig_w, 0.3 / fig_h, cw * len(classes) / fig_w, ch * len(checks) / fig_h])
    ax.set_xlim(0, len(classes))
    ax.set_ylim(len(checks), 0)
    gap_x, gap_y = 2.0 / (cw * DPI), 2.0 / (ch * DPI)        # a 2 px surface gap between cells
    for i, check in enumerate(checks):
        for j, cls in enumerate(classes):
            cell = table[check][key].get(cls)
            if cell is None or cell["n"] < min_n:
                ax.text(j + 0.5, i + 0.5, "few" if cell and cell["n"] > 0 else "", ha="center", va="center", fontsize=8, color=MUTED)
                continue
            color = cmap(cell["rate"])
            ax.add_patch(Rectangle((j + gap_x / 2, i + gap_y / 2), 1 - gap_x, 1 - gap_y, facecolor=color, edgecolor="none"))
            r, g, b = to_rgb(color)
            light = 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.55
            ax.text(j + 0.5, i + 0.5, f"{100 * cell['rate']:.0f}", ha="center", va="center", fontsize=9.5,
                    color=INK if light else "#ffffff")
    ax.set_xticks([j + 0.5 for j in range(len(classes))])
    ax.set_xticklabels([CLASS_NAMES.get(c, c).replace(" to ", " to\n", 1) if "short" in CLASS_NAMES.get(c, "") else
                        CLASS_NAMES.get(c, c).replace(" ", "\n", 1) for c in classes], fontsize=8.5, color=INK_2)
    ax.xaxis.tick_top()
    ax.set_yticks([i + 0.5 for i in range(len(checks))])
    ax.set_yticklabels([CHECK_NAMES.get(c, c) for c in checks], fontsize=10)
    ax.tick_params(length=0)
    for side in ax.spines:
        ax.spines[side].set_visible(False)
    _title(fig, title, subtitle)
    fig.text(2.5 / fig_w, 0.08 / fig_h, "Darker is worse. Blank: no bad plug of that kind was left once the check had acted.",
             fontsize=8.5, color=MUTED, ha="left", va="bottom")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def fig_abstain_curve(sweeps: dict[str, list[dict]], path: Path, title: str, subtitle: str, gate: float = 0.15) -> Path:
    """Lines: abstain rate against alpha, 1 line per connector."""
    plt = _mpl()

    fig = plt.figure(figsize=(6.4, 3.9), dpi=DPI)
    ax = fig.add_axes([0.11, 0.15, 0.70, 0.60])
    top = max([gate] + [row["abstain"]["rate"] for rows in sweeps.values() for row in rows])
    ends = {}
    for k, (connector, rows) in enumerate(sorted(sweeps.items())):
        xs = [row["alpha"] for row in rows]
        ys = [row["abstain"]["rate"] for row in rows]
        ax.plot(xs, ys, color=SERIES[k], linewidth=2, solid_capstyle="round", solid_joinstyle="round", zorder=3,
                marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.6, label=connector)
        ends[connector] = (xs[-1], ys[-1])
    # direct labels at the line ends, pushed apart when the lines end close together
    gap, last = top * 0.07, None
    for connector, (x, y) in sorted(ends.items(), key=lambda kv: kv[1][1]):
        y_text = y if last is None else max(y, last + gap)
        ax.text(x * 1.12, y_text, connector, va="center", ha="left", fontsize=10, color=INK, clip_on=False)
        last = y_text
    ax.axhline(gate, color=AXIS, linewidth=1.0, zorder=1)
    ax.text(0.0046, gate + top * 0.02, f"H2 limit {_pct(gate)}", fontsize=8.5, color=MUTED, va="bottom", ha="left")
    ax.set_xscale("log")
    alphas = sorted({row["alpha"] for rows in sweeps.values() for row in rows})
    ax.set_xticks(alphas)
    ax.set_xticklabels([_pct(a, 1 if a < 0.01 else 0) for a in alphas])
    ax.minorticks_off()
    ax.set_xlim(min(alphas) * 0.85, max(alphas) * 1.15)
    ax.set_ylim(0, top * 1.18)
    from matplotlib.ticker import MultipleLocator

    ax.yaxis.set_major_locator(MultipleLocator(0.05 if top <= 0.4 else 0.1))
    ax.yaxis.set_major_formatter(lambda v, _: _pct(v))
    ax.set_xlabel("promised false-accept rate (alpha)", fontsize=9.5)
    ax.set_ylabel("claims sent to a person", fontsize=9.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=len(sweeps), frameon=False, fontsize=9, handlelength=1.6,
              borderaxespad=0.2, columnspacing=1.4)
    _title(fig, title, subtitle)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def fig_resplit(values: list[float], alpha: float, path: Path, title: str, subtitle: str) -> Path:
    """Histogram: false-accept rate across re-splits, with alpha marked."""
    plt = _mpl()

    fig = plt.figure(figsize=(6.0, 3.4), dpi=DPI)
    ax = fig.add_axes([0.10, 0.17, 0.84, 0.56])
    v = np.asarray(values, dtype=float)
    hi = max(float(v.max()), alpha) * 1.25 + 1e-6
    bins = np.linspace(0, hi, 26)
    counts, edges = np.histogram(v, bins=bins)
    width = edges[1] - edges[0]
    ax.bar(edges[:-1] + width / 2, counts, width=width * 0.86, color=SERIES[0], edgecolor="none", zorder=3)
    ax.axvline(alpha, color=INK_2, linewidth=1.2, zorder=4)
    ax.text(alpha, counts.max() * 1.07, f" alpha {_pct(alpha)}", fontsize=9, color=INK_2, ha="left", va="bottom")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, counts.max() * 1.22)
    from matplotlib.ticker import MaxNLocator

    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax.xaxis.set_major_formatter(lambda x, _: _pct(x, 1))
    ax.set_xlabel("false-accept rate on the held-out half", fontsize=9.5)
    ax.set_ylabel("re-splits", fontsize=9.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    _title(fig, title, subtitle)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def fig_pairs(pairs: list[dict], path: Path, title: str, subtitle: str) -> Path:
    """Picture pairs: a near miss next to its seated twin, from the wrist camera.

    ``pairs`` is a list of {"seated": rgb, "miss": rgb, "label": text}. Images are shown as given.
    """
    plt = _mpl()

    n = len(pairs)
    fig = plt.figure(figsize=(2.05 * 2 * n + 0.3 * (n - 1) + 0.4, 3.25), dpi=DPI)
    w = fig.get_figwidth()
    for k, pair in enumerate(pairs):
        left = (0.2 + k * (4.1 + 0.3)) / w
        for j, (key, cap) in enumerate((("seated", "seated"), ("miss", pair["label"]))):
            ax = fig.add_axes([left + j * 2.05 / w, 0.10, 2.0 / w, 2.0 / 3.25])
            ax.imshow(pair[key], interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines:
                ax.spines[side].set_visible(False)
            ax.set_xlabel(cap, fontsize=9.5, color=INK_2, labelpad=4)
    _title(fig, title, subtitle)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------- tables as markdown
def _cell(c: dict | None, digits: int = 1) -> str:
    if not c or not c.get("n"):
        return "no bad states"
    return f"{100 * c['rate']:.{digits}f}% ({c['k']} of {c['n']}, {100 * c['lo']:.{digits}f} to {100 * c['hi']:.{digits}f})"


def md_evidence_table(table: dict, checks: list[str] | None = None, near: str = "N1, N2, N5") -> str:
    """The E1 table: one row per check. ``near`` names the near-miss classes the connector has."""
    checks = [c for c in (checks or list(table)) if c in table]
    lines = [f"| Check | Accepts bad plugs (FAR) | Near misses {near} | Accepts good plugs | Balanced accuracy | Seconds | Harm | Repair |",
             "|---|---|---|---|---|---|---|---|"]
    for c in checks:
        r = table[c]
        harm = _cell(r["harm"]) if "harm" in r else ""
        repair = _cell(r["repair"]) if "repair" in r else ""
        sec = r.get("seconds_median")
        lines.append(f"| {CHECK_NAMES.get(c, c)} (`{c}`) | {_cell(r['far'])} | {_cell(r['far_near_miss'])} | "
                     f"{_cell(r['true_accept'])} | {r['balanced_accuracy']:.3f} | "
                     f"{'' if sec is None or not math.isfinite(sec) else f'{sec:.1f}'} | {harm} | {repair} |")
    return "\n".join(lines)


def md_class_table(table: dict, classes: list[str], checks: list[str] | None = None, key: str = "by_class") -> str:
    """The source-by-class grid as a table: share of bad plugs of each class accepted, with n."""
    checks = [c for c in (checks or list(table)) if c in table]
    classes = [c for c in classes if c != "P0"]
    lines = ["| Check | " + " | ".join(f"{c} {CLASS_NAMES.get(c, '')}" for c in classes) + " |",
             "|---|" + "---|" * len(classes)]
    for chk in checks:
        cells = []
        for c in classes:
            cell = table[chk][key].get(c)
            cells.append("" if not cell or not cell["n"] else f"{100 * cell['rate']:.0f}% ({cell['k']}/{cell['n']})")
        lines.append(f"| {CHECK_NAMES.get(chk, chk)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
