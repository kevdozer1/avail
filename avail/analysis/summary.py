"""Turn E1 and E2 results into text, figures and a published witness. Evaluator-side."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import figures as report          # named apart: figures() below is this module's own
from .e1 import CHECKS, NEAR_MISS

TABLE_ORDER = ["passive.skill_flag", "passive.jaw_state", "passive.pose_residual", "passive.click", "vision.gap@claim",
               "probe.wiggle", "probe.push_seat", "probe.tug", "fused.passive", "fused.all", "instr.link"]
PAIR_SEED = 1500                      # a dev seed, for the picture pairs only
CROP = 150


def _r(cell: dict, digits: int = 1) -> str:
    if not cell or not cell.get("n"):
        return "n/a"
    return f"{100 * cell['rate']:.{digits}f}% ({cell['k']}/{cell['n']}, {100 * cell['lo']:.{digits}f}-{100 * cell['hi']:.{digits}f})"


def text_summary(result: dict) -> str:
    """A console summary of one connector's E1 and E2."""
    e1r, e2r = result["e1"], result["e2"]
    c = result["counts"]
    lines = [
        f"== {result['connector']} ({result['predicate']}), {result['mode']}, sim. Evaluated on {result['evaluated_on']}",
        f"   rows: fit {c['fit']['rows']}, calibration {c['calibration']['rows']}, evaluation {c['evaluation']['rows']} "
        f"({c['evaluation']['true']} true at claim time)",
        f"   E1 at {100 * result['true_accept_for_e1']:.0f}% true-accept. FAR = accepts among bad plugs (k/n, 95% Wilson):",
    ]
    for check in TABLE_ORDER:
        if check not in e1r["table"]:
            continue
        row = e1r["table"][check]
        extra = ""
        if "repair" in row:
            extra = f"  repair {_r(row['repair'], 0)}  harm {_r(row['harm'], 1)}"
        lines.append(f"     {check:<22} FAR {_r(row['far']):<34} near-miss {_r(row['far_near_miss']):<30} "
                     f"TAR {100 * row['true_accept']['rate']:.1f}%  BA {row['balanced_accuracy']:.3f}{extra}")
    h1 = e1r["h1"]
    lb = h1.get("leaves_bad", {})
    far = h1.get("far_ratio", {})
    lines.append(f"   H1 {h1['status']} on {', '.join(h1['classes'])}, accepted and still bad among all near misses: "
                 f"{h1.get('best_vision')} {_r(lb.get('vision', {}))} vs {h1.get('best_probe')} {_r(lb.get('probe', {}))}, "
                 f"ratio {h1.get('ratio')}, low over high {h1.get('ratio_conservative')}. "
                 f"On false-accept rates: ratio {far.get('ratio', 'n/a')} against {far.get('probe', 'no probe with bad plugs left')}")
    res = e2r["result"]
    lines.append(f"   E2 at alpha {result['alpha']}, beta {result['beta']}: FAR {_r(res['far'], 2)}, certified upper bound "
                 f"{100 * res['far_ucb']:.2f}% (m {res['far_ucb_m']}), FRR {_r(res['frr'], 2)}, abstain {_r(res['abstain'], 2)}, "
                 f"mean {res['mean_seconds']:.2f} s")
    rs = e2r["resplits"]
    lines.append(f"   re-splits ({rs['n_resplits']}): mean FAR {100 * rs['far']['mean']:.3f}% (se {100 * rs['far']['se']:.3f}, "
                 f"sd {100 * rs['far']['sd']:.3f}), abstain mean {100 * rs['abstain']['mean']:.2f}%")
    eq = e2r["equal_budget"]
    lines.append(f"   equal budget for comparison: FAR {_r(eq['far'], 2)}, re-split mean {100 * eq['resplit_far']['mean']:.3f}% "
                 f"(se {100 * eq['resplit_far']['se']:.3f})")
    lines.append(f"   H2 {e2r['h2']['status']}: {e2r['h2']['parts']}")
    nc = e2r.get("no_camera")
    if nc:
        lines.append(f"   exploratory, no camera anywhere: FAR {_r(nc['far'], 2)}, bound {100 * nc['far_ucb']:.2f}% (m {nc['far_ucb_m']}), "
                     f"FRR {_r(nc['frr'], 2)}, abstain {_r(nc['abstain'], 2)}, mean {nc['mean_seconds']:.2f} s, "
                     f"decided by stage {nc['decided_by_stage']}")
    lines.append("   stages: " + "; ".join(
        f"{s['stage']} alpha_k {s['alpha_k']:.4f} accept>{'never' if s['tau_accept'] is None else round(s['tau_accept'], 2)} "
        f"reject<{'never' if s['tau_reject'] is None else round(s['tau_reject'], 2)}" for s in e2r["stages"]))
    lines.append("   decided by stage: " + str(res["decided_by_stage"]) + f"  abstain {res['abstain']['k']}")
    lines.append("   price list: " + "; ".join(
        f"alpha {row['alpha']}: abstain {100 * row['abstain']['rate']:.1f}% FAR {100 * row['far']['rate']:.2f}%" for row in e2r["sweep"]))
    rel = result["release"]
    if rel.get("status") == "measured":
        lines.append(f"   release slice (released_stable, jaw_state): FAR {_r(rel['far'])}, TAR {_r(rel['true_accept'])}")
    else:
        lines.append(f"   release slice: {rel.get('status')} (eval n {rel.get('n_eval')})")
    return "\n".join(lines)


def _project(frame, point) -> tuple[float, float]:
    """Pixel of a base-frame point in a calibrated frame."""
    t = np.linalg.inv(frame.extrinsics)
    p = t[:3, :3] @ np.asarray(point, float) + t[:3, 3]
    k = frame.intrinsics
    return float(k[0, 0] * p[0] / p[2] + k[0, 2]), float(k[1, 1] * p[1] / p[2] + k[1, 2])


def _wrist_crop(prepared) -> np.ndarray:
    from PIL import Image

    arm = prepared.params["nominal"].holding_arm
    frame = prepared.params["frames"].get(f"claim/{arm}_camera_d405")
    port = prepared.params["nominal"].port_position(prepared.outcome.target_port)
    u, v = _project(frame, port)
    h, w = frame.rgb.shape[:2]
    x0 = int(min(max(u - CROP / 2, 0), w - CROP))
    y0 = int(min(max(v - CROP * 0.45, 0), h - CROP))
    crop = frame.rgb[y0:y0 + CROP, x0:x0 + CROP]
    return np.asarray(Image.fromarray(crop).resize((CROP * 3, CROP * 3), Image.NEAREST))


def picture_pairs(connector: str, seed: int = PAIR_SEED) -> list[dict]:
    """A seated plug and its near-miss twins: same seed, same scene, same camera. Dev seed, renders in sim."""
    from ..zoo.harness import prepare

    classes = [c for c in NEAR_MISS if not (c == "N5" and connector != "latch")]
    seated = prepare(seed, connector, "P0")
    seated_crop = _wrist_crop(seated)
    seated.close()
    pairs = []
    for cls in classes:
        miss = prepare(seed, connector, cls)
        short = miss.outcome.truth["metrics"]["shortfall_mm"]
        label = "unlatched, fully in" if cls == "N5" else f"{short:.1f} mm short"
        pairs.append({"seated": seated_crop, "miss": _wrist_crop(miss), "label": label, "cls": cls})
        miss.close()
    return pairs


def figures(results: dict, fig_dir: Path, mode: str, profile: str = "easy") -> list[Path]:
    """Write the E1 and E2 figures. 1 idea each. Hard mode (S4b) labels every figure and skips the pairs,
    which ``avail s4b pairs`` draws easy and hard side by side."""
    fig_dir = Path(fig_dir)
    out = []
    from .run import PROFILE_LABEL

    tag = "sim, test seeds" if mode == "final" else "sim, REHEARSAL on calibration seeds"
    mark = PROFILE_LABEL[profile]
    for connector, result in results.items():
        table = result["e1"]["table"]
        n_bad = result["counts"]["evaluation"]["rows"] - result["counts"]["evaluation"]["true"]
        checks = [c for c in TABLE_ORDER if c in table and c != "instr.link"]
        out.append(report.fig_far_by_check(
            table, fig_dir / f"far_by_check_{connector}.png", "How often each check accepts a bad plug" + mark,
            f"{connector} connector, {tag}. Each check tuned to accept 95% of good plugs. {n_bad} bad plugs at claim time.", checks))
        classes = [c for c in ("N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8") if c in result["counts"]["evaluation"]["classes"]]
        out.append(report.fig_class_grid(
            table, fig_dir / f"check_by_class_{connector}.png", "Which failure each check lets through" + mark,
            f"{connector} connector, {tag}. Percent of bad plugs of each kind that the check accepts.", checks, classes))
        rs = result["e2"]["resplits"]
        out.append(report.fig_resplit(
            rs["far_values"], result["alpha"], fig_dir / f"resplit_{connector}.png", "Does the promised rate hold?" + mark,
            f"{connector} connector, {tag}. Cascade false-accept rate over {rs['n_resplits']} re-splits."))
        if profile != "easy":
            continue
        try:
            pairs = picture_pairs(connector)
            out.append(report.fig_pairs(pairs, fig_dir / f"pairs_{connector}.png", "Looks seated, is not",
                                        f"{connector} connector, wrist camera, sim, dev seed {PAIR_SEED}. Each pair is the same scene."))
        except Exception as exc:                            # a missing clone or renderer must not lose the other figures
            print(f"picture pairs skipped for {connector}: {type(exc).__name__}: {exc}", flush=True)
    sweeps = {c: r["e2"]["sweep"] for c, r in results.items()}
    out.append(report.fig_abstain_curve(sweeps, fig_dir / "abstain_vs_alpha.png", "The price of a tighter promise" + mark,
                                        f"{tag}. Share of claims that end in a call to a person."))
    return out


def publish(results: dict, profile: str = "easy") -> list[str]:
    """Publish each connector's cascade as a new witness version and ship it for the demo.

    Hard mode (S4b) publishes under ``AVAIL_DATA/witness_hard`` with station id ``mj_plug_hard``.
    It does not replace the shipped easy-mode witness: for the demo's hard-mode half it is saved
    beside it as ``avail/witness/shipped/<connector>_hard.json``.
    """
    from ..paths import data_root
    from ..witness import store

    lines = []
    for connector, result in results.items():
        cascade = result["_cascade"]
        provenance = {"mode": result["mode"], "profile": profile, "evaluated_on": result["evaluated_on"],
                      "calibrated_on": result["calibrated_on"],
                      "config_hash": result["config_hash"], "counts": result["counts"],
                      "e2": {"far": result["e2"]["result"]["far"], "far_ucb": result["e2"]["result"]["far_ucb"],
                             "abstain": result["e2"]["result"]["abstain"], "frr": result["e2"]["result"]["frr"]}}
        if profile in ("hard", "hard2"):                    # S4b, S4c: beside the easy witness, never over it
            cascade.station_id = f"mj_plug_{profile}"
            path, cascade = store.publish(cascade, checks=result["_checks"], provenance=provenance,
                                          root=data_root(create=True) / f"witness_{profile}")
            shipped = store.ship(f"{connector}_{profile}", store.build_payload(cascade, result["_checks"], provenance))
            lines.append(f"{profile} witness {connector} {cascade.predicate} {cascade.version} published at {path}, "
                         f"saved for the demo at {shipped}. The easy-mode witness is unchanged")
            continue
        path, cascade = store.publish(cascade, checks=result["_checks"], provenance=provenance)
        payload = store.build_payload(cascade, result["_checks"], provenance)
        shipped = store.ship(connector, payload)
        lines.append(f"witness {connector} {cascade.predicate} {cascade.version} published at {path} and shipped at {shipped}")
    return lines


def md_tables(result: dict) -> dict[str, str]:
    """The markdown tables of one connector, for a report."""
    table = result["e1"]["table"]
    classes = [c for c in ("N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8") if c in result["counts"]["evaluation"]["classes"]]
    checks = [c for c in TABLE_ORDER if c in table]
    near = ", ".join(c for c in NEAR_MISS if c in result["counts"]["evaluation"]["classes"])
    return {
        "evidence": report.md_evidence_table(table, checks, near),
        "by_class": report.md_class_table(table, classes, checks, "by_class"),
        "leaves_bad": report.md_class_table(table, classes, checks, "leaves_bad_by_class"),
    }


__all__ = ["CHECKS", "figures", "md_tables", "picture_pairs", "publish", "text_summary"]
