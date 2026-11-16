"""The showcase: 1 cycle by both arms side by side, and bad cables shipped over a shift's paired cycles. Evaluator-side.

    uv run python -m avail.analysis.showcase --ledger <shift folder> --showcase <dev shift folder> --out <folder>

``--ledger`` is the shift whose paired cycles make the bottom of the figure and its cost line: the seeds both
``skill_done`` (the driver's ``flag`` arm) and ``avail_simop`` ran. ``--showcase`` is a dev shift: the cycle shown
on top and in the video is picked from it by the rule in ``docs/07_experiments.md`` (2026-11-13), never from a test
seed. The figure:

- top, 2 rows of 5 wrist-camera tiles, 1 per cable, at the end of that cable's work: "Skill says done" above,
  "AVAIL + simulated operator" below. A green check: the cable was declared done and is done. A red cross: it was
  declared done and is not (shipped bad). A grey cross: it was held back, not declared done.
- bottom, bad cables shipped against cycles, cumulative, over the paired cycles, 1 line per arm, labeled in place,
  and the cost line underneath: calls for help per 100 cycles, time per cycle as a multiple of ``skill_done``'s, and
  robots per operator (the cycle time over the operator's attention time per cycle: answering plus handling).

The simulated operator answers from simulator truth and takes over with a scripted correction, and the figure says so.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

SKILL, SIMOP = "flag", "avail_simop"
ALIASES = {SIMOP: (SIMOP, "avail_human")}            # avail_human: the arm's name before 2026-11-13
TITLES = {SKILL: "Skill says done", SIMOP: "AVAIL + simulated operator"}
GREEN, RED, GREY, INK, MUTED = "#2e9e5b", "#d64545", "#9aa0a8", "#1f2328", "#5b6270"
CAMERA = "right_camera_d405"
OPERATOR_NOTE = "The simulated operator answers from simulator truth and takes over with a scripted correction."


# ----------------------------------------------------------------------------- ledgers
def cycles(shift_dir: Path, arm: str) -> dict[int, dict]:
    """The arm's ended cycles by seed (the last when a seed reran), each with the folder of its ledger run."""
    out = {}
    for name in ALIASES.get(arm, (arm,)):
        for path in sorted(glob.glob(str(Path(shift_dir) / f"*_{name}_*" / "ledger.jsonl"))):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    e = json.loads(line)
                    if e["kind"] == "cycle_end" and e["payload"].get("arm") == name:
                        out[int(e["payload"]["seed"])] = {**e["payload"], "_run": str(Path(path).parent)}
    return out


def paired(shift_dir: Path) -> list[tuple[int, dict, dict]]:
    """(seed, skill_done's cycle, avail_simop's cycle) for every seed both arms ran, in seed order."""
    a, b = cycles(shift_dir, SKILL), cycles(shift_dir, SIMOP)
    return [(s, a[s], b[s]) for s in sorted(set(a) & set(b))]


def pick(pairs: list[tuple[int, dict, dict]]) -> tuple[int, str]:
    """The showcase seed by the written rule, and which clause chose it."""
    def clean(c):
        return c["declared_success"] and c["true_success"] and not c["uncaught"]

    for clause, keep in (("1", lambda s, b: s["uncaught"] > 0 and clean(b) and b["assists"] > 0),
                         ("3a", lambda s, b: s["uncaught"] > 0 and clean(b)),
                         ("3b", lambda s, b: s["uncaught"] > 0)):
        seeds = [seed for seed, s, b in pairs if keep(s, b)]
        if seeds:
            return min(seeds), clause
    raise ValueError("no paired cycle where skill_done declared a bad cable done")


def cost(cs: list[dict], base_s: float) -> dict:
    """Calls for help per 100 cycles, time per cycle against ``base_s``, and robots per operator."""
    n = max(1, len(cs))
    cycle_s = float(np.mean([c["wall_s"] for c in cs])) if cs else float("nan")
    attention = float(np.mean([float(c["time_s"].get("wait", 0.0)) + float(c["time_s"].get("handling", 0.0)) for c in cs]))
    return {"cycles": len(cs), "calls_per_100": 100.0 * sum(c["assists"] for c in cs) / n, "cycle_s": cycle_s,
            "time_x": cycle_s / base_s if base_s else float("nan"),
            "robots_per_operator": cycle_s / attention if attention > 0 else float("inf"),
            "bad_cables": int(sum(c["uncaught"] for c in cs)), "bad_cycles": int(sum(1 for c in cs if c["uncaught"]))}


# ----------------------------------------------------------------------------- pictures
def _frames(c: dict):
    from ..station.mj_plug import cable_replay

    return cable_replay.load(Path(c["_run"]) / "frames" / f"{c['cycle_id']}.parquet")


def tiles(c: dict, seed: int, connector: str = "latch") -> list[np.ndarray]:
    """1 wrist frame per cable at the end of its work, re-rendered from the frame log. Shift dev seeds only."""
    from ..seeds import assert_simulable
    from ..station.mj_plug.cable_replay import Replay

    assert_simulable(seed)                              # a test seed is refused: the showcase comes from dev seeds
    frames, scene = _frames(c)
    rep = Replay(seed, connector, frames, scene)
    try:
        out = []
        for k in range(5):
            rows = np.flatnonzero(rep.frames["item"].to_numpy() == k)
            out.append(rep.render(int(rows[-1]), camera=CAMERA) if rows.size else np.zeros((480, 640, 3), np.uint8))
        return out
    finally:
        rep.close()


def mark(item: dict) -> tuple[str, str]:
    if item["declared"] and item["true"]:
        return "✓", GREEN
    if item["declared"]:
        return "✗", RED
    return "✗", GREY


def figure(out: Path, show_seed: int, show_pair: tuple[dict, dict], show_tiles: dict, pairs: list, clause: str,
           ledger_name: str, showcase_name: str) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 8.2), dpi=150)
    fig.patch.set_facecolor("white")
    for r, (arm, c) in enumerate(((SKILL, show_pair[0]), (SIMOP, show_pair[1]))):
        y0 = 0.73 - r * 0.215
        fig.text(0.04, y0 + 0.19, TITLES[arm], fontsize=12, color=INK, weight="bold")
        items = {int(i["item"]): i for i in c["items"]}
        for k in range(5):
            ax = fig.add_axes([0.04 + k * 0.188, y0, 0.175, 0.175])
            ax.imshow(show_tiles[arm][k][80:480, 70:570])
            ax.set_axis_off()
            sym, color = mark(items[k])
            ax.text(0.04, 0.96, sym, transform=ax.transAxes, fontsize=22, color=color, weight="bold", va="top",
                    bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.85})
    fig.text(0.04, 0.488, f"1 cycle, dev seed {show_seed:,}, both arms, each cable at the end of its work.   "
                          "✓ done.   ✗ red: declared done, not done.   ✗ grey: held back.", fontsize=10, color=MUTED)
    ax = fig.add_axes([0.08, 0.16, 0.86, 0.29])
    skill = [s for _, s, _ in pairs]
    simop = [b for _, _, b in pairs]
    x = np.arange(1, len(pairs) + 1)
    for cs, arm, color in ((skill, SKILL, RED), (simop, SIMOP, GREEN)):
        y = np.cumsum([c["uncaught"] for c in cs])
        ax.plot(x, y, color=color, lw=2)
        ax.text(x[-1] * 1.01, y[-1], f"{TITLES[arm]}: {int(y[-1]):,}", color=color, fontsize=11, va="center")
    ax.set_xlim(0, len(pairs) * 1.42)
    ax.set_xticks(np.linspace(0, len(pairs), 6).round())
    ax.spines["bottom"].set_bounds(0, len(pairs))
    ax.set_xlabel("cycles", color=MUTED)
    ax.xaxis.set_label_coords(len(pairs) / (2 * len(pairs) * 1.42), -0.12)
    ax.set_ylabel("bad cables shipped", color=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)
    base = cost(skill, 0.0)["cycle_s"]
    c = cost(simop, base)
    fig.text(0.08, 0.075, f"{TITLES[SIMOP]}: {c['calls_per_100']:.0f} calls for help per 100 cycles, "
                          f"{c['time_x']:.2f}x the time per cycle, {c['robots_per_operator']:.1f} robots per operator.",
             fontsize=11, color=INK)
    fig.text(0.08, 0.018, f"Simulation, 5-cable loop, latch. Bottom: {len(pairs):,} paired cycles of {ledger_name}, the same "
                          f"seeds for both arms. Top: {showcase_name}, picked by the written rule ({clause}).\n{OPERATOR_NOTE}",
             fontsize=9, color=MUTED, linespacing=1.5)
    paths = []
    for ext in ("svg", "png"):
        p = Path(out) / f"showcase.{ext}"
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, facecolor="white")
        paths.append(p)
    plt.close(fig)
    return paths


def _font(size: int):
    from matplotlib import font_manager
    from PIL import ImageFont

    return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)


def catches(c: dict) -> list[dict]:
    """Where the operator arm caught a step and called its operator, from its ledger: per call that ended in a takeover,
    the step, the frame where that step's 1st try ended (the catch) and the handoff and handback frames."""
    path = Path(c["_run"]) / "ledger.jsonl"
    out, first_try, claim_of, why = [], {}, {}, {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            e = json.loads(line)
            p = e["payload"]
            if e["kind"] == "evidence" and p.get("claim_id") in claim_of:
                key = claim_of[p["claim_id"]]
                feats = p.get("features", {})
                if key not in why:
                    if feats.get("open") == 0.0:
                        why[key] = "the jaws did not open"
                    elif feats.get("lifted_closed") == 1.0:
                        why[key] = "it lifted with the jaws shut"
                    elif feats.get("gap_mm_est", 0.0) > 2.0:
                        why[key] = f"the camera sees a {feats['gap_mm_est']:.0f} mm gap"
                continue
            if p.get("cycle_id") != c["cycle_id"]:
                continue
            if e["kind"] == "claim" and p["context"]["attempt"] == 0:
                item_s, step_s = p["step_id"].split(".")
                claim_of[p["claim_id"]] = (int(item_s), step_s)
            if e["kind"] == "tool_call" and p.get("tool") == "wam_act" and int(p.get("attempt", -1)) == 0:
                first_try[(int(p["item"]), str(p["skill"]))] = float(p["seconds"])
            if e["kind"] == "intervention" and p.get("handoff_frame") is not None:
                out.append({"item": int(p["item"]), "step": str(p["step"]), "handoff": int(p["handoff_frame"]),
                            "handback": int(p["handback_frame"])})
    frames, _ = _frames(c)
    item, step, t = frames["item"].to_numpy(), frames["step"].astype(str).to_numpy(), frames["t"].to_numpy(dtype=float)
    for k in out:
        k["why"] = why.get((k["item"], k["step"]))
        rows = np.flatnonzero((item == k["item"]) & (step == k["step"]))
        if not rows.size:
            k["catch"] = k["handoff"]
            continue
        # the 1st try ends this long after the step's 1st frame, on the sim clock the frame log shares: the claim
        end_t = t[rows[0]] + first_try.get((k["item"], k["step"]), 0.0)
        at = int(rows[np.searchsorted(t[rows], end_t).clip(0, rows.size - 1)])
        k["catch"] = int(min(max(int(rows[0]), at - 3), k["handoff"]))     # 3 frames inside the 1st try, before a retry moves
    return out


def video(out: Path, seed: int, pair: tuple[dict, dict], connector: str = "latch", speed: int = 4,
          banner: str | None = None, name: str = "showcase.mp4") -> tuple[Path, list[dict]]:
    """The cycle by both arms side by side, wrist camera: Skill says done on the left, AVAIL + simulated operator on
    the right, at ``speed`` times real time. Sized to be read on a phone: large type, 1 short caption per side, and a
    callout on the right from the moment AVAIL catches a step until the operator hands it back. The shorter cycle
    holds its last frame. Returns the path and the catches (with the frame each is in, for a still)."""
    import imageio
    from PIL import Image, ImageDraw

    from ..seeds import assert_simulable
    from ..station.mj_plug.cable_replay import Replay

    assert_simulable(seed)
    reps = []
    for c in pair:
        frames, scene = _frames(c)
        reps.append(Replay(seed, connector, frames, scene))
    caught = catches(pair[1])
    shipped = int(pair[0]["uncaught"])
    # the skill's bad cables, flagged as each one's work ends: (item, last frame of its work)
    bad_ends = []
    skill_items = reps[0].frames["item"].to_numpy()
    for it in pair[0]["items"]:
        rows = np.flatnonzero(skill_items == int(it["item"]))
        if it["declared"] and not it["true"] and rows.size:
            bad_ends.append((int(it["item"]), int(rows[-1])))
    big, mid, small = _font(30), _font(28), _font(22)
    top = 96                                              # the banner: the label, then what the operator is and the speed
    height = top + 480 + 128                              # 704: a multiple of 16, so the encoder does not stretch it
    path = Path(out) / name
    try:
        n = max(len(r.frames) for r in reps)
        writer = imageio.get_writer(str(path), fps=20.0, codec="libx264", quality=7, macro_block_size=16)
        try:
            for idx in range(0, n, speed):
                canvas = Image.new("RGB", (1280, height), "white")
                draw = ImageDraw.Draw(canvas)
                draw.rectangle((0, 0, 1280, top), fill=(255, 236, 179) if banner else (240, 240, 240))
                if banner:
                    draw.text((16, 8), banner, fill=(90, 60, 0), font=mid)
                draw.text((16, 54), f"Simulation, {speed}x speed. The operator is simulated: it answers from simulator "
                                    "truth and takes over with a script.", fill=(90, 60, 0), font=small)
                for side, (rep, arm) in enumerate(zip(reps, (SKILL, SIMOP))):
                    i = min(idx, len(rep.frames) - 1)
                    canvas.paste(Image.fromarray(rep.render(i, camera=CAMERA)), (640 * side, top))
                    row = rep.frames.iloc[i]
                    x, y = 640 * side + 16, top + 480 + 10
                    draw.text((x, y), TITLES[arm], fill=(20, 20, 20), font=big)
                    if idx >= len(rep.frames):
                        line, color = (f"Done. {shipped} bad cables shipped" if arm == SKILL else "Done. 0 bad cables"), \
                            ((200, 40, 50) if arm == SKILL and shipped else (30, 130, 70))
                    elif str(row["tag"]) == "assisted":
                        line, color = f"Cable {int(row['item']) + 1}: operator takes over", (200, 40, 50)
                    else:
                        line, color = f"Cable {int(row['item']) + 1}: {row['step']}", (70, 70, 70)
                    draw.text((x, y + 44), line, fill=color, font=mid)
                    if arm == SKILL:
                        for item_bad, end in bad_ends:
                            if end - 40 <= i <= end + 160:
                                box = (16, top + 16, 640 - 16, top + 106)
                                draw.rectangle(box, fill=(200, 40, 50))
                                draw.text((box[0] + 12, box[1] + 10), f"Cable {item_bad + 1} declared done.",
                                          fill="white", font=mid)
                                draw.text((box[0] + 12, box[1] + 48), "It is not: shipped bad.", fill="white", font=mid)
                    if arm == SIMOP:
                        for k in caught:
                            if k["catch"] <= i < k["handoff"]:
                                box = (640 + 16, top + 16, 1280 - 16, top + 106)
                                draw.rectangle(box, fill=(30, 110, 60))
                                draw.text((box[0] + 12, box[1] + 10), f"AVAIL rejects cable {k['item'] + 1}'s {k['step']}:",
                                          fill="white", font=mid)
                                draw.text((box[0] + 12, box[1] + 48), (k.get("why") or "the check says no") + ".",
                                          fill="white", font=mid)
                                k.setdefault("still_t", idx)
                writer.append_data(np.asarray(canvas))
                for k in caught:
                    if k.get("still_t") == idx and "still" not in k:
                        k["still"] = canvas.copy()
        finally:
            writer.close()
        return path, caught
    finally:
        for r in reps:
            r.close()


def still(path: Path, seed: int, pair: tuple[dict, dict], banner: str | None = None, connector: str = "latch") -> Path | None:
    """1 picture of the 1st cable the skill shipped bad that AVAIL caught: on the left the skill's arm as that cable's
    work ended (declared done), on the right AVAIL's arm at the moment it rejected the same step."""
    from PIL import Image, ImageDraw

    from ..seeds import assert_simulable
    from ..station.mj_plug.cable_replay import Replay

    assert_simulable(seed)
    caught = {(k["item"]): k for k in catches(pair[1])}
    bad = [int(it["item"]) for it in pair[0]["items"] if it["declared"] and not it["true"] and int(it["item"]) in caught]
    if not bad:
        return None
    item, k = bad[0], caught[bad[0]]
    big, mid, small = _font(30), _font(26), _font(22)
    top = 96
    canvas = Image.new("RGB", (1280, top + 480 + 120), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 1280, top), fill=(255, 236, 179) if banner else (240, 240, 240))
    if banner:
        draw.text((16, 8), banner, fill=(90, 60, 0), font=mid)
    draw.text((16, 54), "Simulation. The operator is simulated: it answers from simulator truth and takes over with a "
                        "script.", fill=(90, 60, 0), font=small)
    for side, (c, arm) in enumerate(zip(pair, (SKILL, SIMOP))):
        frames, scene = _frames(c)
        rep = Replay(seed, connector, frames, scene)
        try:
            idx = int(np.flatnonzero(rep.frames["item"].to_numpy() == item)[-1]) if side == 0 else k["catch"]
            canvas.paste(Image.fromarray(rep.render(idx, camera=CAMERA)), (640 * side, top))
        finally:
            rep.close()
        draw.text((640 * side + 16, top + 490), TITLES[arm], fill=(20, 20, 20), font=big)
    box = (16, top + 16, 640 - 16, top + 106)
    draw.rectangle(box, fill=(200, 40, 50))
    draw.text((box[0] + 12, box[1] + 10), f"Cable {item + 1} declared done.", fill="white", font=mid)
    draw.text((box[0] + 12, box[1] + 48), "It is not: shipped bad.", fill="white", font=mid)
    box = (640 + 16, top + 16, 1280 - 16, top + 106)
    draw.rectangle(box, fill=(30, 110, 60))
    draw.text((box[0] + 12, box[1] + 10), f"AVAIL rejects cable {item + 1}'s {k['step']}:", fill="white", font=mid)
    draw.text((box[0] + 12, box[1] + 48), (k.get("why") or "the check says no") + ".", fill="white", font=mid)
    draw.text((16, top + 534), f"Cable {item + 1}, {k['step']}, same seed", fill=(70, 70, 70), font=mid)
    draw.text((656, top + 534), "Next: the operator takes over", fill=(70, 70, 70), font=mid)
    canvas.save(path)
    return Path(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m avail.analysis.showcase", description=__doc__.split("\n\n")[0])
    ap.add_argument("--ledger", required=True, help="the shift folder whose paired cycles make the bottom and the cost line")
    ap.add_argument("--showcase", required=True, help="a dev shift folder to pick the showcase cycle from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="use only the 1st N paired cycles of --ledger")
    args = ap.parse_args(argv)
    pairs = paired(Path(args.ledger))[: args.limit] if args.limit else paired(Path(args.ledger))
    show_pairs = paired(Path(args.showcase))
    seed, clause = pick(show_pairs)
    pair = next((s, b) for sd, s, b in show_pairs if sd == seed)
    show_tiles = {SKILL: tiles(pair[0], seed), SIMOP: tiles(pair[1], seed)}
    paths = figure(Path(args.out), seed, pair, show_tiles, pairs, clause, Path(args.ledger).name, Path(args.showcase).name)
    if not args.no_video:
        path, caught = video(Path(args.out), seed, pair)
        paths.append(path)
        if caught and "still" in caught[0]:
            caught[0]["still"].save(Path(args.out) / "showcase_catch.png")
            paths.append(Path(args.out) / "showcase_catch.png")
    base = cost([s for _, s, _ in pairs], 0.0)["cycle_s"]
    summary = {"showcase_seed": seed, "rule_clause": clause, "paired_cycles": len(pairs),
               "skill_done": cost([s for _, s, _ in pairs], base), "avail_simop": cost([b for _, _, b in pairs], base)}
    (Path(args.out) / "showcase.json").write_text(json.dumps(summary, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=1))
    print("written: " + ", ".join(str(p) for p in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
