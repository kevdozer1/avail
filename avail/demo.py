"""``avail demo``: the same small zoo in easy mode and in hard mode, the free checks, and a report with pictures.

No key, no GPU. It injects 36 states for 1 connector on dev seeds kept for the demo, once in
easy mode and once in hard mode (S4b: a 1080p fisheye webcam on the wrist, jams that no push
passes, a drifting force estimate, connector lots), runs every free evidence source on each,
applies the thresholds of the witnesses that ship with the repo, prints a small evidence table
for each mode and writes an HTML report: the hard-mode camera's pipeline on 1 state, the
reported easy and hard numbers, this run's numbers, the same scenes through both cameras, and
the jams. Simulation only.

The numbers a demo prints stand on 36 states per mode, so their intervals are wide. The
reported numbers stand on 2,000 test states per connector per mode and are in
``docs/RESULTS.md``.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import time
import webbrowser
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

DEMO_SEED_START = 9000              # dev seeds kept for the demo
SHOW_CHECKS = ["passive.skill_flag", "passive.pose_residual", "passive.click", "vision.gap@claim",
               "probe.push_seat", "probe.tug", "fused.passive", "fused.all"]
PROFILES = ("easy", "hard")
CARD_ORDER = {"N1": 0, "N2": 1, "N5": 2, "N3": 3, "N6": 4, "N7": 5, "N4": 6, "N8": 7, "P0": 8}


def _png_b64(rgb: np.ndarray, size: int | None = None, quality: int = 85) -> str:
    from PIL import Image

    img = Image.fromarray(rgb)
    if size is not None:
        img = img.resize((size, int(round(size * img.height / img.width))), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _job(args: tuple) -> dict:
    """Worker: every free source on one seed in one mode, plus a 48 mm wrist crop for the report."""
    seed, connector, profile = args
    from .analysis.pairs_b import _crop
    from .zoo.features import feature_row

    extra: dict = {}

    def grab(prepared) -> None:
        crop, px_mm = _crop(prepared)
        m = prepared.outcome.truth["metrics"]
        extra.update({"crop_b64": _png_b64(crop), "px_per_mm": px_mm, "true_shortfall_mm": m["shortfall_mm"],
                      "skill_flag": bool(prepared.outcome.flag), "jam_kind": prepared.outcome.spec.jam_kind})

    row = feature_row(seed, connector, on_prepared=grab, profile=profile)
    row.update(extra)
    row["profile"] = profile
    return row


def _stages_job(args: tuple) -> dict:
    """Worker: 1 near-miss state through the easy render and through every step of the hard-mode webcam."""
    seed, connector, cls = args
    from .station.mj_plug.inject import sample_error
    from .zoo.harness import prepare

    easy = prepare(seed, connector, spec=sample_error(seed, connector, cls), profile="easy")
    easy_rgb = easy.params["frames"].get("claim/right_camera_d405").rgb
    easy.close()
    hard = prepare(seed, connector, spec=sample_error(seed, connector, cls, profile="hard"), profile="hard")
    try:
        hard.station._keep_stages = {}
        work = hard.station.observe(["right_camera_d405"]).frames["right_camera_d405"].rgb
        stages = hard.station._keep_stages["right_camera_d405"]
        info = dict(hard.station._frame_info["right_camera_d405"])
    finally:
        hard.close()
    return {"seed": seed, "cls": cls, "easy": _png_b64(easy_rgb, 480), "raw": _png_b64(stages["raw"], 640),
            "rectified": _png_b64(stages["rectified"], 640), "work": _png_b64(work, 480), "info": info}


def _load_witness(connector: str, profile: str, witness_file: str | None = None):
    from .witness import store
    from .witness.cascade import Cascade

    if witness_file and profile == "easy":
        payload = json.loads(Path(witness_file).read_text(encoding="utf-8"))
        return Cascade.from_dict(payload["cascade"]), payload
    return store.load_shipped(connector if profile == "easy" else f"{connector}_hard")


def _run_pool(jobs: list[tuple], fn, workers: int) -> list:
    """Run jobs on a fresh process per few jobs, a few at a time (docs/11: pools that retire workers can hang)."""
    out = []
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=12) as pool:
        pending, queue = set(), list(jobs)
        while queue or pending:
            while queue and len(pending) < workers:
                pending.add(pool.submit(fn, queue.pop(0)))
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            out += [f.result() for f in done]
    return out


def _evaluate(rows: list[dict], connector: str, profile: str, witness_file: str | None) -> dict:
    import pandas as pd

    from .analysis import e1
    from .analysis.run import primary_predicate
    from .evidence.fused import Scorer
    from .witness.cascade import evaluate

    cascade, payload = _load_witness(connector, profile, witness_file)
    table = pd.DataFrame(sorted(rows, key=lambda r: r["seed"]))
    predicate = primary_predicate(connector)
    main = e1.main_slice(table, predicate)
    scorers = {k: Scorer.from_dict(v["scorer"]) for k, v in payload["checks"].items()}
    thresholds = {k: {"tau": v["tau"], "tau_ba": v["tau"]} for k, v in payload["checks"].items()}
    usable = {k: s for k, s in scorers.items() if k in ("fused.passive", "fused.all") or k in e1.CHECKS}
    return {"profile": profile, "table": main, "evidence": e1.evaluate_checks(usable, thresholds, main),
            "decisions": cascade.decide_table(main), "cascade": evaluate(cascade, main), "witness_version": cascade.version,
            "provenance": payload.get("provenance", {})}


def run_demo(connector: str, n: int, seed_start: int, workers: int, witness_file: str | None = None,
             profiles: tuple[str, ...] = PROFILES) -> dict:
    """Run the demo in each mode and return the tables. Raises FileNotFoundError when a witness does not ship."""
    from .seeds import guard

    for profile in profiles:                                   # fail before simulating anything
        _load_witness(connector, profile, witness_file)
    seeds = list(range(seed_start, seed_start + n))
    guard(seeds, purpose=f"demo, {connector}")
    t0 = time.perf_counter()
    jobs = [(s, connector, p) for p in profiles for s in seeds]
    stage_job = [("stages", (seed_start, connector, "N1"))] if "hard" in profiles else []
    results = _run_pool([("row", j) for j in jobs] + stage_job, _dispatch, workers)
    rows = [r for kind, r in results if kind == "row"]
    stages = next((r for kind, r in results if kind == "stages"), None)
    out = {"connector": connector, "seeds": [seeds[0], seeds[-1]], "n": n, "modes": {}, "stages": stages,
           "sim_wall_s": time.perf_counter() - t0}
    for profile in profiles:
        out["modes"][profile] = _evaluate([r for r in rows if r["profile"] == profile], connector, profile, witness_file)
    out["predicate"] = next(iter(out["modes"].values()))["table"]["claim_predicate"].iloc[0] if out["modes"] else ""
    return out


def _dispatch(item: tuple):
    kind, args = item
    return (kind, _stages_job(args) if kind == "stages" else _job(args))


def _reported() -> dict | None:
    from .witness.store import SHIPPED_DIR

    path = SHIPPED_DIR / "s4b_results.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def text_table(res: dict) -> str:
    from .analysis.figures import CHECK_NAMES

    lines = [f"AVAIL demo. {res['connector']} connector, claim '{res['predicate']}', simulation, dev seeds "
             f"{res['seeds'][0]} to {res['seeds'][1]}. Each check is tuned to accept 95% of good plugs.", ""]
    modes = list(res["modes"])
    head = f"{'check':<28}" + "".join(f"{m + ': bad plugs accepted':<34}" for m in modes)
    lines.append(head)
    for check in SHOW_CHECKS:
        cells = []
        for m in modes:
            row = res["modes"][m]["evidence"].get(check)
            far = row["far"] if row else None
            cells.append(f"{far['k']} of {far['n']}" if far and far["n"] else "no bad plugs left")
        lines.append(f"{CHECK_NAMES.get(check, check):<28}" + "".join(f"{c:<34}" for c in cells))
    lines.append("")
    for m in modes:
        c = res["modes"][m]["cascade"]
        lines.append(f"{m} cascade: accepted {c['accept_rate']['k']}, rejected {c['reject_rate']['k']}, asked a person "
                     f"{c['abstain']['k']}. Bad plugs accepted: {c['far']['k']} of {c['far']['n']}.")
    return "\n".join(lines)


_CSS = """
:root { --surface:#fcfcfb; --plane:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --line:#e1e0d9; --bad:#d03b3b; --good:#006300; --easy:#2a78d6; --hard:#eb6834; }
@media (prefers-color-scheme: dark) { :root { --surface:#1a1a19; --plane:#0d0d0d; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781; --line:#2c2c2a; --bad:#e66767; --good:#0ca30c; --easy:#6da7ec; --hard:#f08a5d; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--plane); color:var(--ink); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1040px; margin: 0 auto; padding: 28px 16px 56px; }
h1 { font-size: 26px; margin: 0 0 4px; } h2 { font-size: 18px; margin: 34px 0 8px; } h3 { font-size: 15px; margin: 18px 0 6px; }
p { margin: 6px 0; color: var(--ink2); } .lead { color: var(--ink); font-size: 17px; }
.card { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--line); white-space: nowrap; }
th { color: var(--ink2); font-weight: 600; font-size: 13px; } tr:last-child td { border-bottom: 0; }
.muted { color: var(--muted); } .bar { display:inline-block; height:10px; border-radius:0 4px 4px 0; vertical-align:middle; margin-right:8px; }
.bar.easy { background: var(--easy); } .bar.hard { background: var(--hard); }
.tag { display:inline-block; font-size:12px; font-weight:600; padding:1px 7px; border-radius:9px; color:#fff; }
.tag.easy { background: var(--easy); } .tag.hard { background: var(--hard); }
.stages { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
.pairs { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 14px; }
figure { margin: 0; background: var(--surface); border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
figure img { width: 100%; display: block; } .duo { display: grid; grid-template-columns: 1fr 1fr; gap: 2px; background: var(--line); }
.duo img { image-rendering: pixelated; } figcaption { padding: 8px 10px 10px; font-size: 13px; color: var(--ink2); }
figcaption b { color: var(--ink); } .no { color: var(--bad); font-weight: 600; } .yes { color: var(--good); font-weight: 600; }
ul { margin: 6px 0 6px 20px; padding: 0; color: var(--ink2); } li { margin: 2px 0; }
"""


def _cell(c: dict | None, mode: str, scale: int = 150) -> str:
    if not c or not c.get("n"):
        return '<span class="muted">none bad after it acted</span>'
    width = max(1, int(round(scale * float(c["rate"]))))
    lo, hi = c.get("lo"), c.get("hi")
    span = f' <span class="muted">({100 * lo:.1f} to {100 * hi:.1f}%)</span>' if lo is not None and hi is not None else ""
    return f'<span class="bar {mode}" style="width:{width}px"></span>{100 * float(c["rate"]):.1f}% ({c["k"]:,} of {c["n"]:,}){span}'


def _verdict(d: str) -> str:
    return {"accept": '<span class="yes">accepted</span>', "reject": '<span class="no">rejected</span>',
            "abstain": "asked a person"}[d]


def _reported_html(rep: dict | None, connector: str) -> str:
    from .analysis.figures import CHECK_NAMES

    if not rep or connector not in rep.get("connectors", {}):
        return '<p class="muted">No reported numbers ship with this copy. Run the S4 and S4b analyses to make them.</p>'
    easy, hard = rep["connectors"][connector]["easy"], rep["connectors"][connector]["hard"]
    rows = []
    for check in SHOW_CHECKS:
        e, h = easy["checks"].get(check), hard["checks"].get(check)
        if not e and not h:
            continue
        rows.append(f"<tr><td>{html.escape(CHECK_NAMES.get(check, check))}</td><td>{_cell(e and e['far'], 'easy')}</td>"
                    f"<td>{_cell(h and h['far'], 'hard')}</td></tr>")
    ce, ch = easy["cascade"], hard["cascade"]

    def casc(c: dict) -> str:
        return (f"{c['far']['k']} of {c['far']['n']:,} bad plugs accepted, certified under {100 * c['far_ucb']:.2f}%, "
                f"{c['abstain']['k']} of {c['abstain']['n']:,} claims sent to a person")

    rep_e = easy["repair"].get("probe.push_seat") or {}
    rep_h = hard["repair"].get("probe.push_seat") or {}
    return f"""<div class="card"><table><thead><tr><th>Check</th><th><span class="tag easy">easy</span> bad plugs accepted</th>
<th><span class="tag hard">hard</span> bad plugs accepted</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<div class="card" style="margin-top:12px"><table><tbody>
<tr><td><span class="tag easy">easy</span> cascade at 2%</td><td>{casc(ce)}</td></tr>
<tr><td><span class="tag hard">hard</span> cascade at 2%</td><td>{casc(ch)}</td></tr>
<tr><td>Push repairs a bad plug</td><td><span class="tag easy">easy</span> {100 * float(rep_e.get('rate', float('nan'))):.0f}% &nbsp;
<span class="tag hard">hard</span> {100 * float(rep_h.get('rate', float('nan'))):.0f}%</td></tr>
<tr><td>H1 (camera worse than the probes on near misses)</td><td><span class="tag easy">easy</span> {html.escape(str(easy['h1']))} &nbsp;
<span class="tag hard">hard</span> {html.escape(str(hard['h1']))}</td></tr>
<tr><td>H2 (the promised rate holds)</td><td><span class="tag easy">easy</span> {html.escape(str(easy['h2']))} &nbsp;
<span class="tag hard">hard</span> {html.escape(str(hard['h2']))}</td></tr>
</tbody></table></div>
<p class="muted">{easy['rows']:,} easy and {hard['rows']:,} hard test states for {html.escape(connector)}, simulation, each test range read once.
Every rate with its n and a 95% Wilson interval.</p>"""


def _stages_html(st: dict | None) -> str:
    if not st:
        return ""
    info = st["info"]
    cap = [("easy: the render the checks read", st["easy"], "640 x 480, the depth camera's view, rendered clean"),
           ("hard: what the sensor sends", st["raw"], "1920 x 1080 through a 130 degree fisheye. The corners the "
            "rectified frame never uses are not rendered"),
           ("hard: rectified with the calibration", st["rectified"], "close to the lens and not equal to it"),
           ("hard: the frame the checks read", st["work"],
            f"640 x 360. Exposure {1000 * info.get('exposure_s', 0):.0f} ms, gain {info.get('gain', 1):.1f}, "
            f"blur sd {info.get('blur_sigma_px', 0):.1f} px at 1080p, MJPEG {info.get('jpeg_bytes', 0) / 1000:.0f} kB, "
            f"{info.get('port_raw_px_per_mm', 0):.1f} px/mm at the port in the raw frame")]
    figs = "".join(f'<figure><img alt="{html.escape(t)}" src="data:image/jpeg;base64,{b}"><figcaption><b>{html.escape(t)}</b>'
                   f"<br>{html.escape(c)}</figcaption></figure>" for t, b, c in cap)
    return f'<div class="stages">{figs}</div>'


def write_html(res: dict, path: Path) -> Path:
    """The demo report."""
    from .analysis.figures import CHECK_NAMES, CLASS_NAMES

    esc = html.escape
    modes = res["modes"]
    connector = res["connector"]
    live_rows = []
    for check in SHOW_CHECKS:
        cells = "".join(f"<td>{_cell(modes[m]['evidence'].get(check, {}).get('far'), m, 110)}</td>" for m in modes)
        live_rows.append(f"<tr><td>{esc(CHECK_NAMES.get(check, check))}</td>{cells}</tr>")
    head = "".join(f'<th><span class="tag {m}">{m}</span> bad plugs accepted</th>' for m in modes)
    casc_rows = []
    for m in modes:
        c = modes[m]["cascade"]
        casc_rows.append(f'<tr><td><span class="tag {m}">{m}</span></td><td>{c["accept_rate"]["k"]}</td><td>{c["reject_rate"]["k"]}</td>'
                         f'<td>{c["abstain"]["k"]}</td><td>{c["far"]["k"]} of {c["far"]["n"]}</td><td>{c["frr"]["k"]} of {c["frr"]["n"]}</td>'
                         f'<td>{c["mean_seconds"]:.1f} s</td></tr>')

    # the same scenes through both cameras, near misses first
    cards = []
    jam_rows = []
    if "easy" in modes and "hard" in modes:
        te, th = modes["easy"]["table"].set_index("seed"), modes["hard"]["table"].set_index("seed")
        de, dh = modes["easy"]["decisions"].copy(), modes["hard"]["decisions"].copy()
        de.index = modes["easy"]["table"]["seed"].to_numpy()       # 1 decision per row of the main slice, in order
        dh.index = modes["hard"]["table"]["seed"].to_numpy()
        common = [s for s in te.index if s in th.index]
        common.sort(key=lambda s: (CARD_ORDER.get(str(th.loc[s, "cls"]), 9), s))
        for s in common[:12]:
            re_, rh = te.loc[s], th.loc[s]
            cls = str(rh["cls"])
            detail = CLASS_NAMES.get(cls, cls)
            short = rh.get("true_shortfall_mm")
            if cls in ("N1", "N2", "N3") and short == short:
                detail = f"{float(short):.1f} mm short"
            jam = rh.get("jam_kind")
            jam_text = f", held by a {jam}" if isinstance(jam, str) else ""
            truth = "good" if bool(rh["y0"]) else "bad"
            cards.append(
                f'<figure><div class="duo"><img alt="easy wrist crop" src="data:image/jpeg;base64,{re_["crop_b64"]}">'
                f'<img alt="hard wrist crop" src="data:image/jpeg;base64,{rh["crop_b64"]}"></div>'
                f"<figcaption><b>{esc(detail)}</b> (truth: {truth}){esc(jam_text)}. Left easy, right hard, 48 mm of the scene<br>"
                f'<span class="tag easy">easy</span> {_verdict(str(de.loc[s, "decision"]))} '
                f'<span class="muted">stage {int(de.loc[s, "stage_reached"])}</span> &nbsp; '
                f'<span class="tag hard">hard</span> {_verdict(str(dh.loc[s, "decision"]))} '
                f'<span class="muted">stage {int(dh.loc[s, "stage_reached"])}</span></figcaption></figure>')
        for s in th.index:
            r = th.loc[s]
            if not isinstance(r.get("jam_kind"), str):
                continue
            y_push = bool(r.get("A/probe.push_seat/_y")) if r.get("A/probe.push_seat/_y") == r.get("A/probe.push_seat/_y") else False
            slipped = r.get("A/probe.tug/slipped")
            jam_rows.append(f"<tr><td>{s}</td><td>{esc(str(r['cls']))}</td><td>{esc(r['jam_kind'])}</td>"
                            f"<td>{float(r['true_shortfall_mm']):.2f} mm</td>"
                            f"<td>{'seated it' if y_push else 'could not seat it'} "
                            f"<span class='muted'>(score {float(r.get('A/probe.push_seat/_score', float('nan'))):.2f})</span></td>"
                            f"<td>{'held the pull' if slipped == 0 else 'slipped' if slipped == 1 else ''} "
                            f"<span class='muted'>(score {float(r.get('A/probe.tug/_score', float('nan'))):.2f})</span></td>"
                            f"<td>{_verdict(str(dh.loc[s, 'decision']))}</td></tr>")
    rep = _reported()
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>AVAIL demo</title><style>{_CSS}</style></head><body><main>
<h1>AVAIL demo</h1>
<p class="lead">A robot step is done when evidence says so. Here the same plugs go through 2 stations: the clean one of S4 and the hard one of S4b.</p>
<p>Simulation. {esc(connector)} connector, claim "{esc(res['predicate'])}", dev seeds {res['seeds'][0]} to {res['seeds'][1]} in each mode.
Hard mode changes 4 things:</p>
<ul><li>the wrist camera is a 1080p USB webcam with a 130 degree fisheye lens, rectified with an imperfect calibration, with auto exposure, noise, blur and MJPEG</li>
<li>half of the short plugs are held by a jam that no capped push passes: a burr, which lets go when pulled, or a wedge, which can hold the pull</li>
<li>the force estimate drifts and reads high while the load rises, from gearbox friction</li>
<li>the detent force follows the connector's lot, within plus or minus 30%</li></ul>
<h2>The hard-mode camera, step by step</h2>
{_stages_html(res.get('stages'))}
<h2>The reported numbers</h2>
{_reported_html(rep, connector)}
<h2>This run</h2>
<div class="card"><table><thead><tr><th>Check</th>{head}</tr></thead><tbody>{''.join(live_rows)}</tbody></table></div>
<div class="card" style="margin-top:12px"><table><thead><tr><th>Cascade</th><th>Accepted</th><th>Rejected</th><th>Asked a person</th>
<th>Bad plugs accepted</th><th>Good plugs rejected</th><th>Mean check time</th></tr></thead><tbody>{''.join(casc_rows)}</tbody></table></div>
<p class="muted">Each check is tuned to accept 95% of good plugs. A probe is judged on the plug as it left it. With {res['n']} states per mode the intervals are wide.</p>
<h2>The same scene through both cameras</h2>
<div class="pairs">{''.join(cards)}</div>
<h2>Jams a push cannot fix</h2>
<div class="card"><table><thead><tr><th>Seed</th><th>Class</th><th>Jam</th><th>Short by</th><th>Push</th><th>Tug</th><th>AVAIL</th></tr></thead>
<tbody>{''.join(jam_rows) or '<tr><td colspan="7" class="muted">no jammed plug among these seeds</td></tr>'}</tbody></table></div>
<p class="muted">A push that meets a jam stops on force, much as it does on a seated plug. A wedge that holds more than the tug's cap holds the pull as a seated plug does.
The click in the insertion trace is what still tells them apart: a jammed plug never clicks.</p>
<h2>What this does not show</h2>
<p>These are simulated plugs and simulated pictures, also in hard mode. Its scene is still ideal: flat-colored parts with sharp edges and no shadows.
Joint readings are exact, so every good plug clicks and no short plug does. No VLM judge ran, because the demo uses no API key.
{res['n']} states per mode cannot support a rate below several percent: the reported numbers stand on 2,000 test states per connector and mode
and are in docs/RESULTS.md.</p>
</main></body></html>"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8", newline="\n")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="avail demo", description=__doc__.split("\n\n")[0])
    parser.add_argument("--connector", default="latch")
    parser.add_argument("--n", type=int, default=36, help="states per mode")
    parser.add_argument("--profile", default="both", choices=["both", "easy", "hard"])
    parser.add_argument("--seed-start", type=int, default=DEMO_SEED_START)
    # each worker holds a simulator and a renderer, about 2.3 GB of commit charge on Windows, and this
    # machine's desktop leaves about 11 GB, so 4 is the safe default (docs/11_machine_gotchas.md)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default=None, help="the HTML report (default AVAIL_DATA/runs/demo/report.html)")
    parser.add_argument("--witness-file", default=None, help="an easy-mode witness file in place of the shipped one")
    parser.add_argument("--no-open", action="store_true", help="do not open the report in a browser")
    args = parser.parse_args(argv)
    from .paths import check_disk_floor, runs_dir

    profiles = PROFILES if args.profile == "both" else (args.profile,)
    t0 = time.perf_counter()
    check_disk_floor()
    print(f"AVAIL demo: {args.n} simulated states per mode ({', '.join(profiles)}) for the {args.connector} connector "
          f"on {args.workers} workers...", flush=True)
    try:
        res = run_demo(args.connector, args.n, args.seed_start, args.workers, args.witness_file, profiles)
    except FileNotFoundError as exc:
        print(f"avail demo: {exc}", flush=True)
        return 2
    print(text_table(res), flush=True)
    out = Path(args.out) if args.out else runs_dir() / "demo" / "report.html"
    path = write_html(res, out)
    print(f"\nreport: {path}", flush=True)
    print(f"done in {time.perf_counter() - t0:.0f} s", flush=True)
    if not args.no_open:
        webbrowser.open(path.as_uri())
    return 0
