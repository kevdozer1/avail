"""The README figure: 1 bad plug and 3 verdicts on top, 4 waffles of 100 cycles below. Evaluator-side.

    uv run python -m avail.analysis.fig_readme --shift fig_final --out docs/img/fig
    uv run python -m avail.analysis.fig_readme --shift e3_final_latch --out docs/img/fig --no-judge   (E3's 3 arms)

Bottom: 1 square per cycle, in seed order. Green: every cable truly done. Red: declared done with a bad cable.
Grey: held back (a cable left open, none declared done while bad). Under each waffle: time per cycle as a multiple of
``skill_done``'s and calls for help per 100 cycles.

Top: a cycle where the skill declared an insertion done and the evaluator found it unlatched. Its wrist view at the
claim is re-rendered from the frame log. AVAIL's verdict is the ``avail`` arm's on the same seed and cable, used only
when that arm reached the identical state (the same joint angles at the claim). The judge's verdict is 1 call of the
figure run's judge on the re-rendered claim, kept in ``<out>.example.json`` so it is asked once.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

ARMS = (("skill_done", "flag", "Skill says done"), ("agent_looks", "agent_looks", "Agent looks at the camera"),
        ("avail", "avail", "AVAIL"), ("avail_simop", "avail_simop", "AVAIL + simulated operator"))
LEGACY = {"avail_simop": ("avail_human",)}        # E3's and FIG's recorded name for avail_simop (renamed 2026-11-13)
GREEN, RED, GREY, INK, MUTED = "#2e9e5b", "#d64545", "#c9ccd1", "#1f2328", "#5b6270"
CAPTION = ("Simulation, 5-cable loop, latch connector.\nThe simulated operator answers from simulator truth "
           "and takes over with a scripted correction.")


# ----------------------------------------------------------------------------- ledgers
def _events(shift_dir: Path, arm: str, kinds: tuple[str, ...]):
    for path in sorted(glob.glob(str(Path(shift_dir) / f"*_{arm}_*" / "ledger.jsonl"))):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                e = json.loads(line)
                if e["kind"] in kinds:
                    yield Path(path).parent, e


def cycles(shift_dir: Path, arm: str) -> list[dict]:
    """The arm's ended cycles, 1 per seed (the last when a seed reran), in seed order."""
    by_seed = {}
    for name in (arm, *LEGACY.get(arm, ())):
        for run, e in _events(shift_dir, name, ("cycle_end",)):
            p = e["payload"]
            if p.get("arm") == name:
                by_seed[int(p["seed"])] = {**p, "_run": str(run)}
    return [by_seed[s] for s in sorted(by_seed)]


def outcome(c: dict) -> str:
    if c["uncaught"]:
        return "red"
    if c["true_success"] and c["declared_success"]:
        return "green"
    return "grey"


def arm_summary(shift_dir: Path) -> dict:
    out = {}
    for name, driver_arm, title in ARMS:
        cs = cycles(shift_dir, driver_arm)
        if not cs:
            continue
        kinds = [outcome(c) for c in cs]
        out[name] = {"title": title, "n": len(cs), "outcomes": kinds,
                     "green": kinds.count("green"), "red": kinds.count("red"), "grey": kinds.count("grey"),
                     "cycle_s": float(np.mean([c["wall_s"] for c in cs])),
                     "assists_per_100": 100.0 * sum(c["assists"] for c in cs) / len(cs),
                     "usd": float(sum(c.get("usd", 0.0) for c in cs))}
    base = out.get("skill_done", {}).get("cycle_s")
    for s in out.values():
        s["time_x"] = s["cycle_s"] / base if base else float("nan")
    return out


# ----------------------------------------------------------------------------- the example
def _frames(run: Path, cycle_id: str):
    from ..station.mj_plug import cable_replay

    return cable_replay.load(Path(run) / "frames" / f"{cycle_id}.parquet")


def _claim_row(frames, item: int, step: str):
    rows = frames[(frames["item"] == item) & (frames["step"] == step)]
    return None if rows.empty else rows.index[-1]


def _reason(evidence: dict, stage: int) -> str:
    """AVAIL's decisive reason, from the features of the stage that decided."""
    f = lambda src, key: evidence.get(src, {}).get(key)          # noqa: E731
    if stage == 1:
        if f("passive.click", "clicked") == 0.0 and (f("passive.pose_residual", "axial_mm") or 0) < 1.0:
            return "no latch click"
        axial = f("passive.pose_residual", "axial_mm")
        return f"plug {axial:.1f} mm short" if axial is not None else "passive signals"
    if stage == 2 and f("probe.push_seat", "travel_mm") is not None:
        return f"push moved {f('probe.push_seat', 'travel_mm'):.1f} mm"
    if stage == 3 and f("probe.tug", "disp_mm") is not None:
        return f"tug moved {f('probe.tug', 'disp_mm'):.1f} mm"
    if f("vision.gap@now", "gap_mm_est") is not None:
        return f"camera reads a {f('vision.gap@now', 'gap_mm_est'):.1f} mm gap"
    return f"stage {stage}"


def _avail_verdict(shift_dir: Path, seed: int, item: int):
    """The avail arm's 1st insertion verdict on this seed and cable, its evidence, and its cycle's frame log."""
    claim, verdict, evidence, run = None, None, {}, None
    for r, e in _events(shift_dir, "avail", ("claim", "verdict", "evidence")):
        p = e["payload"]
        if e["kind"] == "claim":
            if (claim is None and p["step_id"] == f"{item}.insert" and p["cycle_id"].endswith(f"_c{seed}")
                    and p["context"]["attempt"] == 0):
                claim, run = p, r
        elif claim is not None and p.get("claim_id") == claim["claim_id"]:
            if e["kind"] == "verdict":
                verdict = p
            else:
                evidence[p["source_id"]] = p["features"]
    return claim, verdict, evidence, run


def find_example(shift_dir: Path) -> dict | None:
    """A flag-arm insertion declared done that the evaluator found unlatched, where the avail arm reached the same
    state and rejected it."""
    for c in cycles(shift_dir, "flag"):
        for it in c["items"]:
            if "insert:unlatched" not in it["failures"]:
                continue
            step = next((s for s in c["steps"] if s["item"] == it["item"] and s["step"] == "insert"), None)
            if not step or not step["declared"] or step["true"] or step["attempts"] != 1:
                continue
            claim, verdict, evidence, run = _avail_verdict(shift_dir, c["seed"], it["item"])
            if verdict is None or verdict["decision"] == "accept":
                continue
            flag_frames, _ = _frames(c["_run"], c["cycle_id"])
            idx = _claim_row(flag_frames, it["item"], "insert")
            avail_frames, _ = _frames(run, claim["cycle_id"])
            same = avail_frames[(avail_frames["item"] == it["item"]) & (avail_frames["step"] == "insert")]
            q = np.asarray(flag_frames.loc[idx, "qpos"], dtype=float)
            if idx is None or not any(np.allclose(np.asarray(r, dtype=float), q, atol=1e-9) for r in same["qpos"]):
                continue
            return {"seed": int(c["seed"]), "item": int(it["item"]), "cycle_id": c["cycle_id"], "run": c["_run"],
                    "frame": int(idx), "skill_flag": True, "true": False, "failure": "insert:unlatched",
                    "avail": {"decision": verdict["decision"], "stage": int(verdict["stage_reached"]),
                              "reason": _reason(evidence, int(verdict["stage_reached"]))}}
    return None


def _render_and_ask(ex: dict, ask_judge: bool) -> tuple[np.ndarray, dict | None]:
    """The wrist crop of the claim, re-rendered, and the figure judge's answer on it when asked."""
    import mujoco

    from ..core.task import load_task
    from ..evidence.judge import build_request
    from ..schemas import Claim, ClaimContext, Predicate, new_id
    from ..station.mj_plug import cable_replay, cable_shift
    from ..zoo.judges import reference_image

    frames, scene = _frames(ex["run"], ex["cycle_id"])
    rp = cable_replay.Replay(ex["seed"], "latch", frames, scene)
    try:
        row = rp.frames.loc[ex["frame"]]
        rp._scene_at(int(row["frame_index"]))
        rp.data.qpos[:] = np.asarray(row["qpos"], dtype=float)
        mujoco.mj_forward(rp.model, rp.data)
        cell, item = rp.cell, ex["item"]
        params = cell.claim_params(item, "insert", None)
        task = load_task("cable_loop")
        pred = "latched"
        claim = Claim(claim_id=new_id("claim"), run_id="fig", cycle_id=ex["cycle_id"], step_id=f"{item}.insert",
                      predicate=Predicate(name=pred, target={"item": item, **cell.target_of(item)},
                                          definition=task.predicates[pred].done_means),
                      raised_by="skill", context=ClaimContext(task="cable_loop", stage="insert", attempt=0))
        req = build_request(claim, {**params, "judge": {"model": "x", "variant": "crop", "views": None}})
        wrist = req.images[0]
        answer = None
        if ask_judge:
            judge = cable_shift.make_fig_judge()
            judge._refs["latch"] = reference_image("latch", profile="easy")
            ev = judge.collect(cell.view, claim, params)
            answer = {"holds": float(ev.features.get("holds", float("nan"))),
                      "confidence": ev.features.get("confidence"), "usd": float(ev.cost.usd), "model": judge.model}
        return wrist, answer
    finally:
        rp.close()


# ----------------------------------------------------------------------------- drawing
def draw(summary: dict, wrist: np.ndarray | None, ex: dict | None, out: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = [a for a, _, _ in ARMS if a in summary]
    fig = plt.figure(figsize=(10, 7.4), dpi=150)
    fig.patch.set_facecolor("white")
    if wrist is not None and ex is not None:
        rows = np.where(wrist.reshape(wrist.shape[0], -1).max(axis=1) > 8)[0]      # drop the band outside the frame
        cols = np.where(wrist.max(axis=(0, 2)) > 8)[0]
        if rows.size and cols.size:
            wrist = wrist[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
        ax = fig.add_axes([0.06, 0.53, 0.30, 0.42])
        ax.imshow(wrist)
        ax.set_axis_off()
        ax.text(0.0, -0.05, "Looks seated. The latch has not caught.", transform=ax.transAxes, fontsize=11,
                color=MUTED, va="top")
        verdicts = [("Skill", True, None)]
        if ex.get("judge"):
            verdicts.append(("Agent looking", ex["judge"]["holds"] >= 0.5, None))
        verdicts.append(("AVAIL", ex["avail"]["decision"] == "accept", ex["avail"]["reason"]))
        for k, (who, done, why) in enumerate(verdicts):
            y = 0.86 - 0.11 * k
            wrong = done                                   # the plug is bad: saying done is wrong
            fig.text(0.42, y, "✓" if done else "✗", fontsize=24, color=RED if wrong else GREEN,
                     va="center", weight="bold")
            said = "done" if done else "not done"
            fig.text(0.47, y, f"{who}: {said}" + (f" ({why})" if why else ""), fontsize=13, color=INK, va="center")
    width = 0.205
    for k, name in enumerate(arms):
        s = summary[name]
        left = 0.06 + k * (width + 0.025)
        ax = fig.add_axes([left, 0.155, width, 0.275])
        colors = {"green": GREEN, "red": RED, "grey": GREY}
        for i, kind in enumerate(s["outcomes"][:100]):
            r, c = divmod(i, 10)
            ax.add_patch(plt.Rectangle((c, 9 - r), 0.86, 0.86, color=colors[kind], lw=0))
        ax.set_xlim(-0.1, 10)
        ax.set_ylim(-0.1, 10)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(s["title"], fontsize=11, color=INK, pad=6)
        n = "" if s["n"] == 100 else f", n = {s['n']}"
        fig.text(left + width / 2, 0.118, f"{s['time_x']:.2f}x time{n}", ha="center", fontsize=11, color=INK)
        fig.text(left + width / 2, 0.088, f"{s['assists_per_100']:.0f} calls for help / 100", ha="center",
                 fontsize=11, color=MUTED)
    fig.text(0.06, 0.488, "■ every cable done", color=GREEN, fontsize=11)
    fig.text(0.24, 0.488, "■ declared done with a bad cable", color=RED, fontsize=11)
    fig.text(0.52, 0.488, "■ held back", color="#8a8f98", fontsize=11)
    fig.text(0.06, 0.008, CAPTION, fontsize=10, color=MUTED, linespacing=1.4)
    paths = []
    for ext in ("svg", "png"):
        p = out.with_suffix(f".{ext}")
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, facecolor="white")
        paths.append(p)
    plt.close(fig)
    return paths


def main(argv: list[str] | None = None) -> int:
    from ..paths import runs_dir

    ap = argparse.ArgumentParser(prog="python -m avail.analysis.fig_readme", description=__doc__.split("\n\n")[0])
    ap.add_argument("--shift", default="fig_final")
    ap.add_argument("--out", default="docs/img/fig")
    ap.add_argument("--no-judge", action="store_true", help="draw without asking the judge about the top frame")
    ap.add_argument("--final-run", default=None, help="the final run whose seeds the shift read, such as FIG")
    args = ap.parse_args(argv)
    shift_dir = runs_dir() / "shifts" / args.shift
    out = Path(args.out)
    summary = arm_summary(shift_dir)
    cache = out.with_suffix(".example.json")
    ex = json.loads(cache.read_text(encoding="utf-8")) if cache.is_file() else find_example(shift_dir)
    wrist = None
    if ex is not None and args.final_run:
        from ..seeds import guard, unlock_final_run

        # the claim's joint angles are written from the frame log into a fresh cell: no physics runs
        guard([ex["seed"]], purpose="README figure: re-render 1 stored claim frame and ask the judge about it",
              final_run=args.final_run, stored=True)
        unlock_final_run(args.final_run)
    if ex is not None:
        wrist, answer = _render_and_ask(ex, ask_judge=not args.no_judge and not ex.get("judge"))
        if answer:
            ex["judge"] = answer
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(ex, indent=1), encoding="utf-8", newline="\n")
    paths = draw(summary, wrist, ex, out)
    (out.with_suffix(".summary.json")).write_text(json.dumps(
        {a: {k: v for k, v in s.items() if k != "outcomes"} for a, s in summary.items()}, indent=1),
        encoding="utf-8", newline="\n")
    print(json.dumps({"example": ex, "arms": {a: {k: v for k, v in s.items() if k != "outcomes"}
                                              for a, s in summary.items()}}, indent=1, default=str))
    print("written: " + ", ".join(str(p) for p in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
