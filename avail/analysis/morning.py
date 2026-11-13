"""The morning page: what the night found, pictures first, at most 5 panels. Evaluator-side.

    uv run python -m avail.analysis.morning --shift dev2 --headline "<1 sentence>" [--cycle <cycle id> --seed <seed>] [--no-video]

Writes ``AVAIL_DATA/runs/morning/index.html`` with its images beside it, and copies both to
``reports/morning.html`` and ``reports/morning_img/``. Every number comes from a file the night wrote:
the S4, S4b and S4c result files, the judge comparison, the dev shift's ledgers and the chosen
cycle's frame log.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import shutil
from pathlib import Path

import numpy as np

from ..paths import REPO_ROOT, runs_dir

STEP_NAMES = ("pick", "carry", "insert", "release")
CHECK_ROWS = (("passive.skill_flag", "skill's own flag"), ("passive.jaw_state", "gripper reading"),
              ("passive.pose_residual", "tool pose"), ("passive.click", "click"), ("vision.gap@claim", "camera"),
              ("probe.push_seat", "push"), ("probe.tug", "tug"), ("fused.passive", "all passive, fused"))
CLASS_COLS = (("N1", "0.5-1 mm short"), ("N2", "1-2 mm short"), ("N3", "2-4 mm short"), ("N4", "mouth only"),
              ("N5", "unlatched"), ("N6", "wrong port"), ("N7", "tilted jam"), ("N8", "dropped"))

INK, MUTED, GOOD, BAD, ASK, GRID = "#1d1d1f", "#6e6e73", "#1a7f37", "#c4314b", "#b26b00", "#d2d2d7"

# The judges' good plugs passed (low, high, n), from S4c's exploratory good plugs (reports/S4c.md, the judges): not a
# planned measurement, and no file keeps them. The open-weights judge's 2 prompts gave the 2 counts, which prompt gave
# which was not kept. Fable did not run on good plugs.
JUDGE_GOOD_PLUGS = {"fric": {"cheap_open": (13, 16, 44), "luna": (0, 0, 44)},
                    "latch": {"cheap_open": (11, 12, 48), "luna": (0, 0, 48)}}


def _load_result(folder: str, connector: str) -> dict | None:
    path = runs_dir() / folder / f"e1e2_{connector}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


# ----------------------------------------------------------------------------- panel 3: who catches what
def fig_catches(out: Path, connector: str = "latch") -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = [("Easy (S4)", "e1e2_final"), ("Hard (S4b)", "e1e2b_final"), ("Hard v2 (S4c)", "e1e2c_final")]
    results = [(name, _load_result(folder, connector)) for name, folder in modes]
    results = [(n, r) for n, r in results if r is not None]
    if not results:
        return None
    fig, axes = plt.subplots(1, len(results), figsize=(3.6 * len(results), 3.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, r) in zip(axes, results):
        table = r["e1"]["table"]
        for i, (check, _) in enumerate(CHECK_ROWS):
            row = table.get(check, {}).get("by_class", {})
            for j, (cls, _) in enumerate(CLASS_COLS):
                cell = row.get(cls)
                if not cell or not cell.get("n"):
                    continue
                rate = float(cell["rate"])
                y = len(CHECK_ROWS) - 1 - i
                if rate <= 0.10:                       # catches it: accepts 10% or less of that kind
                    ax.scatter(j, y, s=110, color=INK, zorder=3)
                elif rate <= 0.50:                     # catches some
                    ax.scatter(j, y, s=110, facecolors="white", edgecolors=INK, linewidths=1.5, zorder=3)
                    ax.scatter(j, y, s=32, color=INK, zorder=4)
                else:                                  # lets most through
                    ax.scatter(j, y, s=110, facecolors="white", edgecolors=GRID, linewidths=1.5, zorder=3)
        ax.set_xticks(range(len(CLASS_COLS)))
        ax.set_xticklabels([c[1] for c in CLASS_COLS], rotation=60, ha="right", fontsize=10, color=MUTED)
        ax.set_yticks(range(len(CHECK_ROWS)))
        ax.set_yticklabels([c[1] for c in reversed(CHECK_ROWS)], fontsize=11, color=INK)
        ax.set_title(name, fontsize=13, color=INK, loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        ax.set_xlim(-0.6, len(CLASS_COLS) - 0.4)
        ax.set_ylim(-0.6, len(CHECK_ROWS) - 0.4)
    fig.tight_layout()
    path = out / f"catches_{connector}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------- panel 4: judges against the tug
def fig_judges(out: Path) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = runs_dir() / "s4c_judges" / "judges_vs_probes.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharex=True)
    for ax, connector in zip(axes, ("fric", "latch")):
        runs = data[connector]["runs"]
        e1b = _load_result("e1e2b_final", connector)
        table = (e1b or {}).get("e1", {}).get("table", {})
        rows = []
        for key, label, judge in (("fable__crop_ref", "Fable 5.2 judge", "fable"),
                                  ("cheap_open__crop_ref", "open-weights judge", "cheap_open"),
                                  ("luna__crop_ref", "GPT-6 Luna judge", "luna")):
            if key in runs:
                rows.append((label, runs[key]["judge"], JUDGE_GOOD_PLUGS[connector].get(judge)))
        ref = runs.get("cheap_open__crop_ref") or next(iter(runs.values()))
        for col, label in (("vision.gap@claim", "camera check"), ("probe.tug", "tug")):
            good = table.get(col, {}).get("true_accept")
            rows.append((label, ref[col], (good["k"], good["k"], good["n"]) if good else None))
        rows = rows[::-1]
        for i, (label, bad, good) in enumerate(rows):
            ax.barh(i + 0.19, 100 * bad["rate"], color=BAD, height=0.36)
            ax.text(100 * bad["rate"] + 1.5, i + 0.19, f"bad {100 * bad['rate']:.0f}%", va="center", fontsize=7.5, color=INK)
            if good is None:
                ax.text(1.5, i - 0.19, "good: not measured", va="center", fontsize=7.5, color=MUTED)
                continue
            lo, hi, n = good
            ax.barh(i - 0.19, 100 * hi / n, color=GOOD, height=0.36)
            span = f"{100 * lo / n:.0f}%" if lo == hi else f"{100 * lo / n:.0f} to {100 * hi / n:.0f}%"
            ax.text(100 * hi / n + 1.5, i - 0.19, f"good {span}", va="center", fontsize=7.5, color=INK)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows], fontsize=9)
        ax.set_xlim(0, 125)
        ax.set_title(f"{connector}: plugs passed, bad and good", fontsize=10, loc="left", color=INK)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=0, labelsize=8, colors=MUTED)
    fig.tight_layout()
    p = out / "judges_vs_tug.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ----------------------------------------------------------------------------- panel 2: the filmstrip
def _marks_for_cycle(shift_dir: Path, cycle_id: str) -> dict:
    """For each (item, step): the first verdict, how the step ended, and its first claim's frame time."""
    from ..ledger.reader import iter_events

    marks: dict = {}
    for ledger in shift_dir.glob("*/ledger.jsonl"):
        claims = {}
        for e in iter_events(ledger):
            p = e["payload"]
            if e["kind"] == "claim" and p.get("cycle_id") == cycle_id:
                item, step = p["step_id"].split(".")
                claims[p["claim_id"]] = (int(item), step)
                marks.setdefault((int(item), step), {"verdicts": [], "asks": []})
            elif e["kind"] == "verdict" and p.get("claim_id") in claims:
                marks[claims[p["claim_id"]]]["verdicts"].append(p["decision"])
            elif e["kind"] == "intervention" and p.get("cycle_id") == cycle_id:
                marks.setdefault((int(p["item"]), p["step"]), {"verdicts": [], "asks": []})["asks"].append(
                    (p["question"], p["outcome"]))
    return marks


def _fonts(text_size: int, mark_size: int):
    """DejaVu Sans, the font matplotlib ships: the other panels use it, and it has ✓ and ✗."""
    import os

    import matplotlib
    from PIL import ImageFont

    folder = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
    try:
        return (ImageFont.truetype(os.path.join(folder, "DejaVuSans.ttf"), text_size),
                ImageFont.truetype(os.path.join(folder, "DejaVuSans-Bold.ttf"), mark_size))
    except OSError:
        default = ImageFont.load_default()
        return default, default


def step_mark(m: dict) -> tuple[str, str, str]:
    """What the checks said at the step's 1st claim, and how the step ended: (mark, its color, a short note).

    ✓ the checks passed it, ✗ they caught a failure, ? they could not tell (an abstain). The note says how the step
    ended, a person's part in it included."""
    verdicts, asks = m.get("verdicts", []), m.get("asks", [])
    first = verdicts[0] if verdicts else None
    if first == "accept":
        sym, color, note = "✓", GOOD, ""
    elif first == "reject":
        sym, color = "✗", BAD
        note = "retried, passed" if verdicts[-1] == "accept" else "not done"
    elif first is None and not asks:
        return "", MUTED, ""
    else:
        sym, color, note = "?", ASK, ""
    if asks:
        question, outcome = asks[-1]
        note = {"taken_over_done": "person took over", "taken_over_failed": "takeover failed",
                "signed_off": "person signed off", "retried": "person said retry",
                "unresolved": "not done"}.get(outcome, outcome)
    return sym, color, note


def fig_filmstrip(out: Path, shift_dir: Path, cycle_id: str, seed: int, connector: str) -> Path | None:
    from PIL import Image, ImageDraw

    from ..station.mj_plug.cable_replay import Replay, load

    parquet = next(shift_dir.glob(f"*/frames/{cycle_id}.parquet"), None)
    if parquet is None:
        return None
    frames, scene = load(parquet)
    marks = _marks_for_cycle(shift_dir, cycle_id)
    rep = Replay(seed, connector, frames, scene)
    tile_w, tile_h, label_h, left = 240, 180, 40, 96
    n_items = int(frames["item"].max()) + 1
    canvas = Image.new("RGB", (tile_w * 4 + left, (tile_h + label_h) * n_items + 30), "white")
    draw = ImageDraw.Draw(canvas)
    font, big = _fonts(17, 24)
    for j, step in enumerate(STEP_NAMES):
        draw.text((left + j * tile_w + 8, 4), step, fill=MUTED, font=font)
    for item in range(n_items):
        y0 = 30 + item * (tile_h + label_h)
        draw.text((8, y0 + tile_h // 2 - 10), f"cable {item + 1}", fill=INK, font=font)
        for j, step in enumerate(STEP_NAMES):
            sel = frames[(frames["item"] == item) & (frames["step"] == step)]
            if sel.empty:
                continue
            idx = int(sel.index[-1])                   # the end of the step, after any retry or takeover
            img = Image.fromarray(rep.render(idx, camera="right_camera_d405")).resize((tile_w - 6, tile_h - 6))
            canvas.paste(img, (left + j * tile_w + 3, y0 + 3))
            sym, color, note = step_mark(marks.get((item, step), {}))
            draw.text((left + j * tile_w + 8, y0 + tile_h + 2), sym, fill=color, font=big)
            draw.text((left + j * tile_w + 38, y0 + tile_h + 7), note, fill=MUTED, font=font)
    rep.close()
    path = out / "filmstrip.png"
    canvas.save(path)
    return path


def cycle_video(path: Path, shift_dir: Path, cycle_id: str, seed: int, connector: str, stride: int = 1) -> Path | None:
    """1 cycle as an mp4 in real time at the frame log's own rate: the wrist camera beside the clip camera, and a
    caption with the cable, the step, the verdict and who has control."""
    import imageio
    from PIL import Image, ImageDraw

    from ..station.mj_plug.cable_replay import Replay, load

    parquet = next(shift_dir.glob(f"*/frames/{cycle_id}.parquet"), None)
    if parquet is None:
        return None
    frames, scene = load(parquet)
    marks = _marks_for_cycle(shift_dir, cycle_id)
    rep = Replay(seed, connector, frames, scene)
    font, _ = _fonts(24, 24)
    gaps = np.diff(frames["t"].to_numpy(dtype=float))
    rate = 1.0 / float(np.mean(gaps[gaps > 0])) if (gaps > 0).any() else 20.0       # dev1's logs ran at 19.2 Hz
    writer = imageio.get_writer(str(path), fps=rate / stride, codec="libx264", quality=7, macro_block_size=16)
    try:
        for idx in range(0, len(frames), stride):
            row = frames.iloc[idx]
            wrist = rep.render(idx, camera="right_camera_d405")
            side = rep.render(idx, camera="video_cam")
            canvas = Image.new("RGB", (1280, 480 + 64), "white")
            canvas.paste(Image.fromarray(wrist), (0, 0))
            canvas.paste(Image.fromarray(side), (640, 0))
            item, step, tag = int(row["item"]), str(row["step"]), str(row["tag"])
            m = marks.get((item, step), {"verdicts": [], "asks": []})
            verdicts = ", then ".join(m["verdicts"]) if m["verdicts"] else ""
            who = "A person has control (takeover)" if tag == "assisted" else "The robot has control"
            text = f"Cable {item + 1}, {step}" + (f": {verdicts}" if verdicts else "") + f". {who}"
            ImageDraw.Draw(canvas).text((16, 480 + 18), text, fill=(200, 40, 60) if tag == "assisted" else (20, 20, 20),
                                        font=font)
            writer.append_data(np.asarray(canvas))
    finally:
        writer.close()
        rep.close()
    return path


# ----------------------------------------------------------------------------- panel 5: the dev shift's numbers
def shift_numbers(shift_dir: Path, variant: str = "latch") -> dict:
    """Per arm, from the ledgers: cycles, cables, bad cables and bad steps declared done, calls for help,
    takeovers, operating seconds per call for help, and seconds per cycle."""
    from ..ledger.reader import iter_events

    arms: dict = {}
    for ledger in sorted(shift_dir.glob("*/ledger.jsonl")):
        for e in iter_events(ledger, kinds=["cycle_end"]):
            p = e["payload"]
            if p.get("variant") != variant:
                continue
            a = arms.setdefault(p["arm"], {"cycles": 0, "cables": 0, "bad_cables_through": 0, "bad_steps_through": 0,
                                           "steps": 0, "assists": 0, "takeovers": 0, "operating_s": 0.0, "shift_s": 0.0,
                                           "true_cycles": 0, "usd": 0.0})
            t = p.get("time_s", {})
            a["cycles"] += 1
            a["true_cycles"] += bool(p.get("true_success"))
            a["cables"] += len(p.get("items", []))
            a["bad_cables_through"] += int(p.get("uncaught", 0))
            a["steps"] += len(p.get("steps", []))
            a["bad_steps_through"] += sum(1 for s in p.get("steps", []) if s.get("declared") and not s.get("true"))
            a["assists"] += int(p.get("assists", 0))
            a["takeovers"] += int(p.get("takeovers", 0))
            a["operating_s"] += sum(float(t.get(k, 0.0)) for k in ("motion", "verify", "model"))
            a["shift_s"] += sum(float(t.get(k, 0.0)) for k in ("motion", "verify", "model", "wait", "handling"))
            a["usd"] += float(p.get("usd", 0.0))
    for a in arms.values():
        n = max(a["cycles"], 1)
        a["assists_per_100"] = 100.0 * a["assists"] / n
        a["takeovers_per_100"] = 100.0 * a["takeovers"] / n
        a["mtbi_s"] = a["operating_s"] / a["assists"] if a["assists"] else float("inf")
        a["cycle_s"] = a["shift_s"] / n
    return arms


def pick_cycle(shift_dir: Path, arm: str = "avail_human", variant: str = "latch") -> dict | None:
    """A cycle of ``arm`` with a caught failure (a reject verdict) and a takeover, every cable done by truth at
    the end: the fewest takeovers, at least 1, then the most reject verdicts, then the lowest seed."""
    from ..ledger.reader import iter_events

    best = None
    for ledger in sorted(shift_dir.glob(f"*_{arm}_{variant}_b*/ledger.jsonl")):
        cycle_of: dict[str, str] = {}                 # claim id to its cycle
        caught: dict[str, int] = {}                   # cycle id to its reject verdicts
        for e in iter_events(ledger, kinds=["claim", "verdict", "cycle_end"]):
            p = e["payload"]
            if e["kind"] == "claim":
                cycle_of[p["claim_id"]] = p["cycle_id"]
            elif e["kind"] == "verdict" and p.get("decision") == "reject" and p.get("claim_id") in cycle_of:
                cid = cycle_of[p["claim_id"]]
                caught[cid] = caught.get(cid, 0) + 1
            elif e["kind"] == "cycle_end":
                n = caught.get(p["cycle_id"], 0)
                if n and p.get("takeovers", 0) >= 1 and p.get("true_success"):
                    key = (p["takeovers"], -n, p["seed"])
                    if best is None or key < best[0]:
                        best = (key, {"cycle_id": p["cycle_id"], "seed": p["seed"], "takeovers": p["takeovers"],
                                      "rejects": n, "assists": p.get("assists", 0)})
    return best[1] if best else None


def card_rows(numbers: dict, arm: str = "avail_human") -> list[tuple[str, str]]:
    a, flag = numbers.get(arm), numbers.get("flag")
    if not a:
        return []
    mtbi = "none" if math.isinf(a["mtbi_s"]) else f"{a['mtbi_s']:.0f} s"
    rows = [("cycles of 5 cables", f"{a['cycles']}"),
            ("bad steps that got through", f"{a['bad_steps_through']:,} of {a['steps']:,}"),
            ("bad cables that got through", f"{a['bad_cables_through']:,} of {a['cables']:,}"),
            ("calls for help per 100 cycles", f"{a['assists_per_100']:.0f}"),
            ("takeovers per 100 cycles", f"{a['takeovers_per_100']:.0f}"),
            ("robot time between calls for help", mtbi)]
    if flag:
        rows.append(("bad cables through on the same seeds on the skills' own flags",
                     f"{flag['bad_cables_through']:,} of {flag['cables']:,}"))
    return rows


# ----------------------------------------------------------------------------- the page
def page(out: Path, headline: str, panels: list[tuple[str, str, str | None]], card_rows: list[tuple[str, str]]) -> Path:
    css = """
:root { --ink:#1d1d1f; --muted:#6e6e73; --bg:#ffffff; --line:#e5e5ea; --link:#0b57d0; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --ink:#f5f5f7; --muted:#a1a1a6; --bg:#111113; --line:#2c2c2e; --link:#8ab4f8; } }
:root[data-theme="dark"] { --ink:#f5f5f7; --muted:#a1a1a6; --bg:#111113; --line:#2c2c2e; --link:#8ab4f8; }
a { color: var(--link); }
body { background: var(--bg); color: var(--ink); font: 16px/1.45 system-ui, -apple-system, Segoe UI, sans-serif; margin: 0; }
main { max-width: 1040px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 23px; line-height: 1.3; margin: 8px 0 28px; font-weight: 650; }
section { margin: 0 0 40px; }
h2 { font-size: 15px; color: var(--muted); font-weight: 600; margin: 0 0 10px; text-transform: none; }
img { max-width: 100%; height: auto; display: block; border-radius: 6px; background: #fff; }
.card { display: grid; grid-template-columns: repeat(auto-fit, minmax(128px, 1fr)); gap: 12px; }
.card div { border: 1px solid var(--line); border-radius: 8px; padding: 12px; }
.card b { display: block; font-size: 21px; font-weight: 650; white-space: nowrap; }
.card span { color: var(--muted); font-size: 13px; }
.note { color: var(--muted); font-size: 13px; margin-top: 6px; }
"""
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>",
             f"<title>AVAIL morning</title><style>{css}</style></head><body><main>",
             f"<h1>{html.escape(headline)}</h1>"]
    for title, img, note in panels:
        if note == "video":                            # the filmstrip's cycle as a video, beside it
            note_html = "<p class='note'><a href='cycle.mp4'>The same cycle as video (mp4)</a></p>"
        else:
            note_html = f"<p class='note'>{html.escape(note)}</p>" if note else ""
        parts.append(f"<section><h2>{html.escape(title)}</h2><img src='{img}' alt='{html.escape(title)}'>"
                     + note_html + "</section>")
    if card_rows:
        parts.append("<section><h2>The dev shift, sim</h2><div class='card'>"
                     + "".join(f"<div><b>{html.escape(v)}</b><span>{html.escape(k)}</span></div>" for k, v in card_rows)
                     + "</div></section>")
    parts.append("</main></body></html>")
    path = out / "index.html"
    path.write_text("\n".join(parts), encoding="utf-8", newline="\n")
    return path


def publish_copy(out: Path) -> Path:
    """Copy the page and its images into reports/, the images and the video into reports/morning_img/."""
    dest_img = REPO_ROOT / "reports" / "morning_img"
    dest_img.mkdir(parents=True, exist_ok=True)
    text = (out / "index.html").read_text(encoding="utf-8")
    for f in [*out.glob("*.png"), *out.glob("*.mp4")]:
        shutil.copy2(f, dest_img / f.name)
        text = text.replace(f"src='{f.name}'", f"src='morning_img/{f.name}'").replace(
            f"href='{f.name}'", f"href='morning_img/{f.name}'")
    dest = REPO_ROOT / "reports" / "morning.html"
    dest.write_text(text, encoding="utf-8", newline="\n")
    return dest


def fig_dev_change(out: Path, before: str, after: str, connector: str = "latch", arm: str = "avail_human") -> Path | None:
    """2 simple bars per chart: calls for help per 100 cycles and seconds per cycle, 1 dev shift against the next."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nums = {name: shift_numbers(runs_dir() / "shifts" / name, connector) for name in (before, after)}
    if not all(arm in n for n in nums.values()):
        return None
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
    flag_s = {name: n.get("flag", {}).get("cycle_s") for name, n in nums.items()}
    for ax, (key, title) in zip(axes, (("assists_per_100", "calls for help per 100 cycles"), ("cycle_s", "seconds per cycle"))):
        vals = [nums[name][arm][key] for name in (before, after)]
        ax.bar([0, 1], vals, color=[MUTED, INK], width=0.55)
        for x, v in enumerate(vals):
            ax.text(x, v * 1.02, f"{v:.0f}", ha="center", va="bottom", fontsize=10, color=INK)
        if key == "cycle_s" and all(flag_s.values()):
            for x, name in enumerate((before, after)):
                ax.plot([x - 0.3, x + 0.3], [flag_s[name]] * 2, color=BAD, lw=1.5)
            ax.text(1.35, flag_s[after], "flag", color=BAD, fontsize=8, va="center")
        ax.set_xticks([0, 1])
        ax.set_xticklabels([before, after], fontsize=9)
        ax.set_ylim(0, max(vals) * 1.25)
        ax.set_title(title, fontsize=10, loc="left", color=INK)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.set_yticks([])
    fig.tight_layout()
    p = out / "dev_change.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def e3_progress(connector: str = "latch") -> dict:
    """Cycles ended per E3 arm against the plan, from the E3 ledgers. Progress only: no outcome is read here."""
    from ..config import load_experiment
    from ..ledger.reader import iter_events

    e3 = load_experiment().e3
    plan = {arm: int(e3["avail_human_cycles"].get(connector, e3["avail_human_cycles"]["default"]))
            if arm == "avail_human" else int(e3["cycles_per_arm"]) for arm in e3["arms"] if arm != "vlm"}
    done = {arm: 0 for arm in plan}
    for ledger in sorted((runs_dir() / "shifts" / f"e3_final_{connector}").glob("*/ledger.jsonl")):
        for e in iter_events(ledger, kinds=["cycle_end"]):
            arm = e["payload"].get("arm")
            if arm in done:
                done[arm] += 1
    return {"plan": plan, "done": done}


def fig_e3_progress(out: Path, connector: str = "latch") -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prog = e3_progress(connector)
    arms = list(prog["plan"])
    fig, ax = plt.subplots(figsize=(7.6, 0.55 * len(arms) + 0.9))
    for i, arm in enumerate(arms[::-1]):
        n, k = prog["plan"][arm], prog["done"][arm]
        ax.barh(i, 100.0, color=GRID, height=0.5)
        ax.barh(i, 100.0 * k / n, color=INK, height=0.5)
        ax.text(101.5, i, f"{k:,} of {n:,} cycles", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels(arms[::-1], fontsize=9)
    ax.set_xlim(0, 135)
    ax.set_xticks([])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    p = out / "e3_progress.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m avail.analysis.morning")
    parser.add_argument("--shift", required=True, help="the dev shift the filmstrip's cycle comes from")
    parser.add_argument("--before", default=None, help="an earlier dev shift: the 2 bars compare it with --after")
    parser.add_argument("--after", default=None, help="the later dev shift of the 2 bars")
    parser.add_argument("--e3", action="store_true", help="add E3's progress, cycles ended per arm")
    parser.add_argument("--cycle", default=None, help="default: the avail_human cycle pick_cycle chooses")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--connector", default="latch")
    parser.add_argument("--headline", required=True)
    parser.add_argument("--card", default=None, help="a JSON list of [label, value] pairs. Default: from the ledgers")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args(argv)
    if args.cycle is not None and args.seed is None:
        parser.error("--cycle needs --seed: the filmstrip replays the cycle's scene from its seed")
    out = runs_dir() / "morning"
    out.mkdir(parents=True, exist_ok=True)
    shift_dir = runs_dir() / "shifts" / args.shift
    numbers = shift_numbers(shift_dir, args.connector)
    (out / "shift_numbers.json").write_text(json.dumps(numbers, indent=1, default=float), encoding="utf-8")
    cycle, seed = args.cycle, args.seed
    if cycle is None:
        chosen = pick_cycle(shift_dir, variant=args.connector)
        if chosen is None:
            raise SystemExit("no avail_human cycle has both a caught failure and a takeover. Pass --cycle")
        cycle, seed = chosen["cycle_id"], int(chosen["seed"])
        print("cycle:", json.dumps(chosen), flush=True)
    panels = []
    if args.before and args.after:
        change = fig_dev_change(out, args.before, args.after, args.connector)
        if change:
            panels.append((f"{args.before} to {args.after}, avail_human on {args.connector}, 100 dev cycles each, sim. "
                           "The red line is flag's time", change.name, None))
    if args.e3:
        panels.append((f"E3 on {args.connector}, the test read, sim: cycles ended per arm",
                       fig_e3_progress(out, args.connector).name, None))
    film = fig_filmstrip(out, shift_dir, cycle, seed, args.connector)
    marker = out / "cycle.json"                         # which cycle cycle.mp4 shows, so a rebuild can keep it
    if args.no_video:
        same = marker.is_file() and json.loads(marker.read_text(encoding="utf-8")).get("cycle_id") == cycle
        video = (out / "cycle.mp4") if same and (out / "cycle.mp4").is_file() else None
    else:
        video = cycle_video(out / "cycle.mp4", shift_dir, cycle, seed, args.connector)
        marker.write_text(json.dumps({"cycle_id": cycle, "seed": seed, "shift": args.shift}), encoding="utf-8")
    if film:
        panels.append(("1 simulated cycle from the wrist camera, 1 frame at the end of each step. ✓ the checks passed "
                       "it, ✗ they caught a failure, ? they could not tell. The words beside a mark say what happened "
                       "next, such as a person taking over",
                       film.name, "video" if video else None))
    catches = fig_catches(out, args.connector)
    if catches:
        panels.append((f"Who catches what ({args.connector}, sim). Filled: the check passes at most 10% of that kind of "
                       "bad plug. Dot in a ring: 10% to 50%. Grey ring: over 50%. Blank: the probe repaired every one, "
                       "so none was left to judge",
                       catches.name, None))
    judges = fig_judges(out)
    if judges:
        panels.append(("Judges and checks, hard mode v1, sim: red, bad near misses passed. Green, good plugs passed",
                       judges.name, "The judges' good plugs are an exploratory sample. Fable did not run on good plugs."))
    if args.before and args.after:
        rows = []                                       # the 2 bars carry the shift's numbers
    else:
        rows = [tuple(r) for r in json.loads(args.card)] if args.card else card_rows(numbers)
    path = page(out, args.headline, panels, rows)
    print(path, publish_copy(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
