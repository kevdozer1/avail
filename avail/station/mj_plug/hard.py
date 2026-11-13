"""Hard mode's per-seed draws (S4b): the connector lot, the cable's route, the camera units and the jam mix.

Each draw is a pure function of (seed, connector) from its own random stream. The draws of easy
mode (the class, the shortfall, the scene) use other streams and do not move, so a seed shows
the same scene and the same injected error in both modes, and a picture can put them side by side.

Simulator side. Evidence sources never import this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ...config import load_experiment
from . import aspire
from .webcam import LensUnit, draw_unit

WRIST_CAMERAS = (aspire.WRIST_CAMERA["left"], aspire.WRIST_CAMERA["right"])
CABLE_SEGMENTS = 8


@dataclass(frozen=True)
class HardDraw:
    """What hard mode adds to one seed."""

    lot: int                          # the connector lot, 0-based
    lot_factor: float                 # that lot's detent force against nominal
    cable_azimuth: float              # rad in the plug frame, 0 along +x (the fingers), pi/2 along +y
    cable_gap_m: float                # how far from the plug body's side the hanging cable passes
    lenses: dict[str, LensUnit]       # per wrist camera
    jam_u: float                      # uniform draws behind the jam decision, kept apart so a forced class keeps them
    jam_kind_u: float
    wedge_u: float


@lru_cache(maxsize=8)
def lot_factors(connector: str) -> tuple[float, ...]:
    """Each lot's detent force factor for a connector type. Fixed by the config, the same for every split."""
    exp = load_experiment()
    force = exp.hard().force
    names = sorted(exp.connectors)
    rng = np.random.default_rng([names.index(connector), 7070])
    p = force.detent_lot_pct / 100.0
    return tuple(float(round(v, 6)) for v in rng.uniform(1.0 - p, 1.0 + p, size=force.lots))


def sample_hard(seed: int, connector: str) -> HardDraw:
    """Hard mode's draws for a seed. A pure function of (seed, connector)."""
    exp = load_experiment()
    hard = exp.hard()
    names = sorted(exp.connectors)
    c = names.index(connector)
    rng = np.random.default_rng([int(seed), c, 6464])
    lots = lot_factors(connector)
    lot = int(rng.integers(0, len(lots)))
    half = math.radians(hard.plug_cable.sector_deg / 2.0)
    side = math.pi / 2.0 if rng.random() < 0.5 else -math.pi / 2.0
    azimuth = side + float(rng.uniform(-half, half))
    gap = float(rng.uniform(*hard.plug_cable.side_gap_mm)) * 1e-3
    jam_u, jam_kind_u, wedge_u = (float(v) for v in rng.random(3))
    lenses = {}
    for k, cam in enumerate(WRIST_CAMERAS):
        lenses[cam] = draw_unit(hard.wrist_camera, np.random.default_rng([int(seed), c, k, 8181]))
    return HardDraw(lot=lot, lot_factor=lots[lot], cable_azimuth=round(azimuth, 6), cable_gap_m=round(gap, 6),
                    lenses=lenses, jam_u=jam_u, jam_kind_u=jam_kind_u, wedge_u=wedge_u)


def jam_of(draw: HardDraw, cls: str, connector: str) -> tuple[str | None, float | None]:
    """(kind, wedge hold in N) when this seed's short plug is made by a jam, else (None, None)."""
    exp = load_experiment()
    jams = exp.hard().jams
    if cls not in jams.classes or draw.jam_u >= jams.share:
        return None, None
    kinds = sorted(jams.kinds)
    weights = np.array([jams.kinds[k] for k in kinds], dtype=float)
    cut = np.cumsum(weights / weights.sum())
    kind = kinds[int(np.searchsorted(cut, draw.jam_kind_u, side="right"))] if draw.jam_kind_u < 1.0 else kinds[-1]
    if kind != "wedge":
        return kind, None
    lo, hi = jams.wedge_hold_x_retention
    hold = (lo + (hi - lo) * draw.wedge_u) * exp.connector(connector).retention_seated_n
    return kind, round(float(hold), 4)


def cable_points(draw: HardDraw, connector: str, geom) -> list[tuple[float, float, float]]:
    """The cable's center line in the plug frame, meters: up out of the plug's top, over, down beside the body, then off.

    The plug frame has z along the plug axis with the shoulder at 0, and x across the fingers.
    The cable comes down outside the body at ``cable_gap_m`` from its side, to a little above the
    shoulder, then runs straight away from the plug at that height.
    """
    cable = load_experiment().hard().plug_cable
    radius = cable.diameter_mm * 1e-3 / 2.0
    bx, by, bz = geom.body
    ux, uy = math.cos(draw.cable_azimuth), math.sin(draw.cable_azimuth)
    half_y = by / 2.0 + (0.0016 if connector == "latch" else 0.0)     # the latch tab stands 1.6 mm off the +y face
    extent = min(bx / 2.0 / max(abs(ux), 1e-9), half_y / max(abs(uy), 1e-9))
    d = extent + draw.cable_gap_m + radius
    low = cable.end_above_shoulder_mm * 1e-3 + radius
    top = bz
    pts = [(0.0, 0.0, top), (0.0, 0.0, top + 0.003), (0.5 * d * ux, 0.5 * d * uy, top + 0.005),
           (0.9 * d * ux, 0.9 * d * uy, top + 0.003), (d * ux, d * uy, top - 0.003),
           (d * ux, d * uy, 0.5 * (top - 0.003 + low)), (d * ux, d * uy, low),
           ((d + 0.012) * ux, (d + 0.012) * uy, low), ((d + 0.025) * ux, (d + 0.025) * uy, low)]
    assert len(pts) == CABLE_SEGMENTS + 1
    return pts
