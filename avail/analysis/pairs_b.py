"""S4b picture pairs: each near-miss class next to its seated twin, from the wrist camera, easy and hard.

Same dev seed, same scene, same injected error in both modes (hard mode draws its extras from
streams of its own). Each crop shows the same 48 mm of the scene around the port, so the hard
camera's fewer pixels per millimeter show as a coarser picture. Evaluator-side, dev seed 1500.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .e1 import NEAR_MISS
from .summary import PAIR_SEED, _project

CROP_MM = 48.0                     # the easy crop was 150 px at about 3.1 px/mm
SHOW_PX = 300                      # every crop is shown at this size


def _crop(prepared) -> tuple[np.ndarray, float]:
    """The wrist crop around the port, CROP_MM wide, scaled to SHOW_PX. Returns (image, pixels per mm of the frame)."""
    from PIL import Image

    from ..evidence.vision import measure_gap

    nominal = prepared.params["nominal"]
    frame = prepared.params["frames"].get(f"claim/{nominal.holding_arm}_camera_d405")
    port = nominal.port_position(prepared.outcome.target_port)
    gap = measure_gap(frame, nominal, prepared.outcome.target_port)
    px_mm = float(gap.get("px_per_mm", 3.1))
    size = int(round(CROP_MM * px_mm))
    u, v = _project(frame, port)
    h, w = frame.rgb.shape[:2]
    size = min(size, h, w)
    x0 = int(min(max(u - size / 2, 0), w - size))
    y0 = int(min(max(v - size * 0.45, 0), h - size))
    crop = frame.rgb[y0:y0 + size, x0:x0 + size]
    return np.asarray(Image.fromarray(crop).resize((SHOW_PX, SHOW_PX), Image.NEAREST)), px_mm


def pairs(connector: str, seed: int = PAIR_SEED) -> list[dict]:
    """Per near-miss class: the seated and near-miss crops in easy mode and in hard mode."""
    from ..station.mj_plug.inject import sample_error
    from ..zoo.harness import prepare

    classes = [c for c in NEAR_MISS if not (c == "N5" and connector != "latch")]
    out = []
    crops: dict = {}
    for profile in ("easy", "hard"):
        seated = prepare(seed, connector, spec=sample_error(seed, connector, "P0", profile=profile), profile=profile)
        crops[(profile, "P0")] = _crop(seated)
        seated.close()
        for cls in classes:
            spec = sample_error(seed, connector, cls, profile=profile)
            miss = prepare(seed, connector, spec=spec, profile=profile)
            short = miss.outcome.truth["metrics"]["shortfall_mm"]
            label = "unlatched, fully in" if cls == "N5" else f"{short:.1f} mm short"
            if spec.jam_kind:
                label += f", {spec.jam_kind} jam"
            crops[(profile, cls)] = (*_crop(miss), label)
            miss.close()
    for cls in classes:
        out.append({"cls": cls, "easy_seated": crops[("easy", "P0")][0], "easy_miss": crops[("easy", cls)][0],
                    "hard_seated": crops[("hard", "P0")][0], "hard_miss": crops[("hard", cls)][0],
                    "easy_label": crops[("easy", cls)][2], "hard_label": crops[("hard", cls)][2],
                    "easy_px_mm": crops[("easy", cls)][1], "hard_px_mm": crops[("hard", cls)][1]})
    return out


def figure(rows: list[dict], path: Path, connector: str, seed: int = PAIR_SEED) -> Path:
    """1 row per near-miss class: easy seated, easy near miss, a gap, hard seated, hard near miss."""
    from .figures import _mpl, _title

    plt = _mpl()
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(11.0, 2.9 * n + 1.1), squeeze=False)
    for i, row in enumerate(rows):
        cells = (("easy_seated", "seated"), ("easy_miss", row["easy_label"]), ("hard_seated", "seated"),
                 ("hard_miss", row["hard_label"]))
        for j, (key, label) in enumerate(cells):
            ax = axes[i][j]
            ax.imshow(row[key])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xlabel(label, fontsize=10)
        axes[i][0].set_ylabel(row["cls"], fontsize=11, rotation=0, labelpad=18)
    axes[0][0].set_title(f"easy, {rows[0]['easy_px_mm']:.1f} px/mm", fontsize=10, loc="left")
    axes[0][2].set_title(f"hard, {rows[0]['hard_px_mm']:.1f} px/mm", fontsize=10, loc="left")
    _title(fig, "The same near misses through the hard-mode camera",
           f"{connector} connector, holding wrist camera, sim, dev seed {seed}. Each crop is 48 mm of the scene.")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def camera_stages(path: Path, connector: str = "latch", seed: int = PAIR_SEED, cls: str = "N1") -> Path:
    """1 state through both cameras: the easy-mode render, then the hard-mode webcam's raw, rectified and working frames."""
    from ..station.mj_plug.inject import sample_error
    from ..zoo.harness import prepare
    from .figures import _mpl, _title

    easy = prepare(seed, connector, spec=sample_error(seed, connector, cls), profile="easy")
    easy_rgb = easy.params["frames"].get("claim/right_camera_d405").rgb
    easy.close()
    hard = prepare(seed, connector, spec=sample_error(seed, connector, cls, profile="hard"), profile="hard")
    hard.station._keep_stages = {}
    work = hard.station.observe(["right_camera_d405"]).frames["right_camera_d405"].rgb
    stages = hard.station._keep_stages["right_camera_d405"]
    info = hard.station._frame_info["right_camera_d405"]
    hard.close()
    plt = _mpl()
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.6), gridspec_kw={"width_ratios": [4 / 3, 16 / 9, 16 / 9, 16 / 9]})
    panels = ((easy_rgb, "easy: the render the checks read, 640 x 480"),
              (stages["raw"], "hard: what the sensor sends, 1920 x 1080"),
              (stages["rectified"], "hard: rectified with the calibration"),
              (work, "hard: the frame the checks read, 640 x 360"))
    for ax, (img, label) in zip(axes, panels):
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(label, fontsize=9)
    _title(fig, "The wrist camera, easy and hard",
           f"{connector} connector, {cls}, sim, dev seed {seed}. Hard frame: exposure {1000 * info['exposure_s']:.0f} ms, gain "
           f"{info['gain']:.1f}, blur sd {info['blur_sigma_px']:.1f} px at 1080p. The raw frame's corners lie outside the rays "
           "the rectified frame uses and are not rendered.")
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    import argparse

    from ..paths import REPO_ROOT

    parser = argparse.ArgumentParser(prog="avail s4b pairs")
    parser.add_argument("--connector", action="append", default=None)
    parser.add_argument("--out", default=str(REPO_ROOT / "docs" / "img" / "s4b"))
    args = parser.parse_args(argv)
    for connector in args.connector or ["fric", "latch"]:
        rows = pairs(connector)
        print(figure(rows, Path(args.out) / f"pairs_easy_hard_{connector}.png", connector), flush=True)
    print(camera_stages(Path(args.out) / "camera_stages_latch.png"), flush=True)
    return 0
