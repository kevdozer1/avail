"""Connector retention: the force a socket puts on a plug along the port axis.

Two parts, both functions of how deep the plug is, both from ``config/experiment.yaml``:

- ``axial_friction_n``: dry sliding friction. It is handed to MuJoCo as friction loss on the
  plug's axial degree of freedom, so the constraint solver holds the plug still below the
  friction level and the model stays stable at a 1 ms step.
- ``detent_n``: the conservative force F(d). A barrier just before full depth gives the click
  on the way in, and the same barrier resists the way out. The latch connector's table is
  steep on the inside (easy in, hard out). ``latch_blocked`` swaps in a table with no barrier.

The force acts only while the tongue tip is inside a socket's bore. Any socket will do, so a
plug in the wrong port is held like any other. In v0 the board lies flat, so the port axis is
the base frame's z and the plug's axial degree of freedom is its free joint's z translation.

This module reads simulator state. It is part of the simulated world, on the far side of the
truth wall from every evidence source.
"""

from __future__ import annotations

import math
from bisect import bisect_right

import numpy as np

from ...config import ConnectorSpec
from .scene import BOARD_CENTER_XY, TABLE_TOP_Z, Layout, SceneConfig

AXIAL_DAMPING = 4.0          # N s/m on the plug's axial motion while captured, for numerical calm
CAPTURE_MARGIN = 0.0004      # m, slack on the bore outline when deciding whether the tip is inside
# MuJoCo's friction loss is a soft constraint. With its default impedance a 25 g plug moving at
# 8 mm/s feels 0.03 N where the table says 1.5 N. This impedance makes it act as dry friction.
FRICTION_SOLIMP = (0.999, 0.999, 0.001, 0.5, 2.0)


class Table:
    """A piecewise-linear table, zero outside its range unless ``hold_ends``."""

    def __init__(self, points: list[tuple[float, float]], hold_ends: bool = False):
        self.xs = [float(p[0]) for p in points]
        self.ys = [float(p[1]) for p in points]
        self.hold_ends = hold_ends

    def __call__(self, x: float) -> float:
        xs, ys = self.xs, self.ys
        if x <= xs[0]:
            return ys[0] if (self.hold_ends or x == xs[0]) else 0.0
        if x >= xs[-1]:
            return ys[-1] if (self.hold_ends or x == xs[-1]) else 0.0
        i = bisect_right(xs, x)
        x0, x1 = xs[i - 1], xs[i]
        return ys[i - 1] + (ys[i] - ys[i - 1]) * (x - x0) / (x1 - x0)

    @property
    def peak_positive(self) -> float:
        return max(0.0, max(self.ys))

    @property
    def peak_negative(self) -> float:
        return max(0.0, -min(self.ys))


class Retention:
    """Applies the connector's axial forces to the plug. Call ``update`` before every sim step.

    Hard mode (S4b) adds 2 things. ``detent_scale`` multiplies the detent tables: the connector
    lot's factor. ``wedge`` is ``(port, stop shortfall, hold in N, seat, ramp, span)``, lengths in
    mm (``profiles.hard.jams``), for a plug jammed by a misalignment. The jam stop keeps it from
    going deeper. Once the tip is within ``seat`` of the stop the plug binds where it is, and a
    barrier like the detent's keeps it from coming out: a pull toward that point that rises to
    ``hold`` over ``ramp`` of pull-out, stays there until ``span``, and fades to nothing over the
    next ``span``, so a plug that breaks free does not let a stored pull go all at once. It binds
    again only when pushed back onto the stop. Nothing pulls the plug in on the way down. The plug
    is not tilted. Friction loss would hold it too, but in MuJoCo it creeps under a steady pull,
    and the barrier does not.
    """

    def __init__(self, model, lay: Layout, cfg: SceneConfig, spec: ConnectorSpec, latch_blocked: bool = False,
                 detent_scale: float = 1.0, wedge: tuple[int, float, float, float, float, float] | None = None):
        if not spec.modeled:
            raise ValueError(f"connector {lay.connector!r} has no force tables in config/experiment.yaml")
        self.lay = lay
        self.depth_full_mm = spec.full_depth_mm
        self.friction = Table(spec.axial_friction_n, hold_ends=True)
        scale = float(detent_scale)
        self.detent_scale = scale
        self.detent_engaged = Table([(x, y * scale) for x, y in spec.detent_n])
        blocked = spec.detent_latch_blocked_n
        self.detent_blocked = Table([(x, y * scale) for x, y in blocked]) if blocked else Table([(0.0, 0.0), (1.0, 0.0)])
        self.latch_blocked = bool(latch_blocked)
        self.wedge = wedge
        self.wedged = False           # hard mode: the plug has reached its jam stop and is held there
        self.wedged_at = math.nan     # hard mode: the shortfall at which it bound, mm
        self.tip = lay.ids["site:plug_tip"]
        self.zdof = lay.ids["plug_dadr"] + 2
        self.ox = lay.geom.opening_half[0] + CAPTURE_MARGIN
        self.oy = lay.geom.opening_half[1] + CAPTURE_MARGIN
        self.face_z = TABLE_TOP_Z + lay.geom.face_z
        c, s = math.cos(cfg.board_yaw), math.sin(cfg.board_yaw)
        bx, by = BOARD_CENTER_XY[0] + cfg.board_xy[0], BOARD_CENTER_XY[1] + cfg.board_xy[1]
        self.ports_xy = [(bx + c * px - s * py, by + s * px + c * py) for px, py in lay.ports_board]
        self.cos, self.sin = c, s
        self.base_frictionloss = float(lay.base["dof_frictionloss"][self.zdof])
        self.base_damping = float(lay.base["dof_damping"][self.zdof])
        model.dof_solimp[self.zdof] = FRICTION_SOLIMP
        # what the last update saw (privileged, for the evaluator)
        self.captured = False
        self.port = 0                 # 1-based index of the capturing port, 0 when free
        self.depth_mm = 0.0
        self.shortfall_mm = float("nan")
        self.detent_force = 0.0       # N toward the seat
        self.friction_level = 0.0     # N

    @property
    def detent(self) -> Table:
        return self.detent_blocked if self.latch_blocked else self.detent_engaged

    def nominal_peaks(self) -> dict[str, float]:
        """Insertion peak and seated retention implied by the tables, in N."""
        slide = self.friction(self.depth_full_mm)
        return {"insertion_peak_n": slide + self.detent.peak_negative, "retention_n": slide + self.detent.peak_positive}

    def update(self, model, data) -> None:
        tip = data.site_xpos[self.tip]
        tx, ty, tz = float(tip[0]), float(tip[1]), float(tip[2])
        depth = self.face_z - tz
        port, captured = 0, False
        if depth > 0.0:
            c, s = self.cos, self.sin
            for i, (px, py) in enumerate(self.ports_xy):
                dx, dy = tx - px, ty - py
                if abs(dx) > 0.02 or abs(dy) > 0.02:
                    continue
                lx, ly = c * dx + s * dy, -s * dx + c * dy          # into the board frame
                if abs(lx) <= self.ox and abs(ly) <= self.oy:
                    port, captured = i + 1, True
                break
        self.captured, self.port = captured, port
        if not captured:
            self.wedged, self.wedged_at = False, math.nan
            self.depth_mm, self.shortfall_mm = max(0.0, depth * 1000.0), float("nan")
            self.detent_force = self.friction_level = 0.0
            model.dof_frictionloss[self.zdof] = self.base_frictionloss
            model.dof_damping[self.zdof] = self.base_damping
            data.qfrc_applied[self.zdof] = 0.0
            return
        depth_mm = depth * 1000.0
        shortfall = self.depth_full_mm - depth_mm
        self.depth_mm, self.shortfall_mm = depth_mm, shortfall
        self.friction_level = self.friction(depth_mm)
        self.detent_force = self.detent(shortfall)
        if self.wedge is not None:
            w_port, w_stop, w_hold, w_seat, w_ramp, w_span = self.wedge
            at_stop = port == w_port and shortfall - w_stop <= w_seat
            if at_stop and (not self.wedged or shortfall < self.wedged_at):
                self.wedged, self.wedged_at = True, shortfall          # it binds where it is, deepest point kept
            if self.wedged:
                out_mm = shortfall - self.wedged_at if port == w_port else math.inf
                if out_mm >= 2.0 * w_span:
                    self.wedged, self.wedged_at = False, math.nan
                elif out_mm > 0.0:
                    rise = min(1.0, out_mm / w_ramp)
                    fade = 1.0 if out_mm <= w_span else (2.0 * w_span - out_mm) / w_span
                    self.detent_force += w_hold * rise * fade
        model.dof_frictionloss[self.zdof] = self.base_frictionloss + self.friction_level
        model.dof_damping[self.zdof] = self.base_damping + AXIAL_DAMPING
        data.qfrc_applied[self.zdof] = -self.detent_force           # toward the seat is down

    def release(self, model, data) -> None:
        """Put the plug's axial degree of freedom back to its base values."""
        model.dof_frictionloss[self.zdof] = self.base_frictionloss
        model.dof_damping[self.zdof] = self.base_damping
        data.qfrc_applied[self.zdof] = 0.0


def retention_curve(spec: ConnectorSpec, latch_blocked: bool = False, step_mm: float = 0.01) -> dict[str, np.ndarray]:
    """The force tables sampled against shortfall, for plots and tests. Positive pulls toward the seat."""
    friction = Table(spec.axial_friction_n, hold_ends=True)
    table = spec.detent_latch_blocked_n if (latch_blocked and spec.detent_latch_blocked_n) else spec.detent_n
    detent = Table(table) if not (latch_blocked and not spec.detent_latch_blocked_n) else Table([(0.0, 0.0), (1.0, 0.0)])
    short = np.arange(0.0, spec.full_depth_mm + step_mm, step_mm)
    return {
        "shortfall_mm": short,
        "detent_n": np.array([detent(x) for x in short]),
        "friction_n": np.array([friction(spec.full_depth_mm - x) for x in short]),
    }
