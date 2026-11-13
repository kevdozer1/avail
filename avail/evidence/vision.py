"""vision.gap: a classical estimate of the gap between the plug shoulder and the socket face.

The source reads 1 stored camera frame and the nominal geometry. It never moves the robot.

Method, for a calibrated view that faces 1 side of the plug from above:

1. Project the taught port into the frame with the frame's own calibration. Pick the side face
   of the plug that looks at the camera.
2. Resample the image on the plane of that face (the plane that holds the plug body's near
   side), on a grid of 0.1 mm. In this picture the plug axis is vertical and the board is at
   the bottom.
3. Find the near wall of the socket housing: a box as dark as the darkest things in the
   picture, with brighter board on both sides and, where no neighbor socket hides it, under its
   foot. The taught port is off by about 1 mm and the camera by a few millimeters, so the wall
   is searched for within 8 mm of where the calibration puts it. The wall's own width sets the
   scale across the picture, which also absorbs an error in the frame's horizontal focal length.
4. Walk up from the middle of the wall along 2 bands of columns that lie inside the body's
   width and outside the tongue's width. In each band the first step up is the top of the wall,
   where the darker wall meets the socket's top face, and the largest step up is the shoulder of
   the plug body. Each step is placed where the brightness crosses halfway between its 2 levels.
5. The camera looks down on the socket, so with the plug home a strip of the socket's top face
   still shows between the wall and the shoulder. Its height follows from the geometry and is
   taken off. What is left, averaged over the bands that could be read, is ``gap_mm_est``.
   Where the foot of the wall is in view, the wall's height in the picture corrects the scale
   along the plug axis.
6. The middle columns say how much light tongue shows under the body (``tongue_mm``). A large
   gap with no tongue in it is an empty socket, and then no plug was found at the port.

No gap is reported when the top of the wall and the shoulder cannot be told apart in either
band, when the shoulder sits below the socket face by more than the measurement can be wrong,
or when the gap is far larger than the insertion depth. In each case ``plug_found`` is 0.

Features: ``gap_mm_est``, ``confidence`` (0 to 1), ``plug_found``, ``housing_found``,
``port_in_view``, ``port_u``, ``port_v``, ``px_per_mm``, ``view_elevation_deg``, ``tongue_mm``,
``slant_deg`` (tilt of the shoulder edge against the socket's top edge), ``bands_read`` (1 or
2), ``edge_contrast``, ``top_contrast``, ``wall_contrast``, ``shift_x_mm``, ``shift_z_mm``,
``scale_ratio`` and ``scale_z_ratio``. A feature that could not be measured is left out.

Views (``params['view']``): ``claim`` reads the holding arm's wrist frame stored at claim time,
``now`` takes a fresh frame from that camera first and stores it as ``now/<camera>`` (used after
the probes have acted), ``relook`` reads a frame stored by a closer look.

Score: ``-gap_mm_est`` when a plug was found at the port, and ``-full_depth_mm`` (the whole
tongue showing) when the frame shows no plug there. Higher means more likely seated. The score
says nothing about a latch: an unlatched plug sits at full depth and looks seated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..schemas import Claim, Cost, Evidence
from ..station.base import CameraFrame, NominalGeometry, Station
from .base import make_evidence, register

MM = 1e-3
RES_MM = 0.1                       # grid step of the resampled picture
HALF_WIDTH_MM = 16.0               # the picture spans this far to each side of the taught port
BELOW_MM = 13.0                    # and this far below the housing's foot
ABOVE_MM = 27.0                    # and this far above the socket face
SEARCH_MM = 8.0                    # how far the housing may sit from where the calibration puts it
SCALES = np.arange(0.62, 1.1201, 0.04)   # lateral scales tried when looking for the housing wall
WALL_SHARE = 0.55                  # the upper share of the wall that is searched for when its foot is hidden
DARK_FACTOR, DARK_OFFSET = 1.8, 4.0      # the wall is at most this much brighter than the picture's darkest 3 percent
MIN_ELEVATION_DEG = 22.0           # below this the socket's top face is too thin a strip to read
MAX_ELEVATION_DEG = 80.0           # above this the camera looks down the plug axis and the side face vanishes
MIN_VALID_SHARE = 0.85             # share of the picture that must fall inside the frame
MIN_WALL_CONTRAST = 10.0           # gray levels between the wall and the darkest of its neighbors
MIN_SHOULDER_STEP = 12.0           # gray levels across the shoulder edge
MIN_TOP_STEP = 4.0                 # gray levels across the top edge of the wall
MIN_FOOT_STEP = 8.0                # gray levels across the foot of the wall
FLANK_MARGIN_MM = 0.6
CENTER_MARGIN_MM = 0.5
EDGE_SIGMA_MM = 0.2
EMPTY_GAP_MM = 3.0                 # above this gap a plug must show tongue
BELOW_FACE_MM = 0.6                # a shoulder this far below the socket face is not a plug in this socket


@dataclass
class _View:
    """The camera and the port in one frame of reference: across the face, toward the camera, up the plug axis."""

    k: np.ndarray                  # 3 x 3 intrinsics
    r_cb: np.ndarray               # camera to base rotation
    c: np.ndarray                  # camera center, base frame
    p: np.ndarray                  # taught face center of the port, base frame
    a: np.ndarray                  # unit vector across the face
    n: np.ndarray                  # unit vector out of the face, toward the camera
    z: np.ndarray                  # unit vector up the plug axis
    cam: np.ndarray                # camera center in (a, n, z) coordinates, mm
    wb: float                      # half-width of the plug body across the face, mm
    db: float                      # distance of the body's near face from the plug axis, mm
    wt: float                      # half-width of the tongue across the face, mm
    wh: float                      # half-width of the housing across the face, mm
    dh: float                      # distance of the housing's near wall from the port axis, mm
    hh: float                      # height of the housing, mm
    depth: float                   # full insertion depth, mm

    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Base-frame points (meters) to pixel coordinates and depth along the optical axis."""
        pc = (pts - self.c) @ self.r_cb
        depth = pc[..., 2]
        safe = np.where(np.abs(depth) < 1e-9, 1e-9, depth)
        uv = pc @ self.k.T
        return uv[..., 0] / safe, uv[..., 1] / safe, depth

    def seen_at(self, across: float, out: float, up: float) -> tuple[float, float]:
        """Where a point (mm, port frame) shows in the picture of the body's face plane: (across, up) in mm."""
        ca, cn, cz = self.cam
        lam = (self.db - cn) / (out - cn)
        return ca + lam * (across - ca), cz + lam * (up - cz)


@dataclass
class _Edge:
    pos: float                     # mm, where the profile crosses halfway between the 2 levels
    height: float                  # gray levels
    low: float                     # level before the step
    high: float                    # level after the step


def _port_view(frame: CameraFrame, nominal: NominalGeometry, port: int) -> _View | None:
    k = np.asarray(frame.intrinsics, dtype=float)
    t = np.asarray(frame.extrinsics, dtype=float)
    if k.shape != (3, 3) or t.shape != (4, 4) or not (np.all(np.isfinite(k)) and np.all(np.isfinite(t))):
        return None
    r_cb, c = t[:3, :3], t[:3, 3]
    p = np.asarray(nominal.port_position(port), dtype=float)
    r_tool = np.asarray(nominal.tool_rotation(), dtype=float)
    x_p, z_p = r_tool[:, 0], -r_tool[:, 2]            # across the fingers, and up the plug axis
    y_p = np.cross(z_p, x_p)
    d = c - p
    cx, cy = float(x_p @ d), float(y_p @ d)
    bx, by, _ = nominal.plug_body_m
    tx, ty = nominal.plug_tongue_m
    hx, hy = nominal.housing_half_m
    if min(bx, by, tx, ty, hx, hy, nominal.housing_height_m, nominal.full_depth_m) <= 0:
        return None
    if abs(cy) >= abs(cx):
        n = y_p * (1.0 if cy >= 0 else -1.0)
        wb, db, wt, wh, dh = bx / 2, by / 2, tx / 2, hx, hy
    else:
        n = x_p * (1.0 if cx >= 0 else -1.0)
        wb, db, wt, wh, dh = by / 2, bx / 2, ty / 2, hy, hx
    a = np.cross(n, z_p)
    cam = np.array([a @ d, n @ d, z_p @ d]) / MM
    return _View(k=k, r_cb=r_cb, c=c, p=p, a=a, n=n, z=z_p, cam=cam, wb=wb / MM, db=db / MM, wt=wt / MM,
                 wh=wh / MM, dh=dh / MM, hh=nominal.housing_height_m / MM, depth=nominal.full_depth_m / MM)


def _rectify(frame: CameraFrame, view: _View) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The image resampled on the plane of the body's near face. Returns (rgb, valid, across mm, up mm)."""
    xs = np.arange(-HALF_WIDTH_MM, HALF_WIDTH_MM + 1e-9, RES_MM)
    zs = np.arange(-(view.hh + BELOW_MM), ABOVE_MM + 1e-9, RES_MM)
    # A point of the plane is p + (x a + db n + z up) mm. Its pixel is a ratio of 2 expressions linear in x and z.
    origin = view.k @ ((view.p + view.db * MM * view.n - view.c) @ view.r_cb)
    per_x = view.k @ ((view.a * MM) @ view.r_cb)
    per_z = view.k @ ((view.z * MM) @ view.r_cb)
    col, row = xs[None, :], zs[:, None]
    depth = origin[2] + per_x[2] * col + per_z[2] * row
    safe = np.where(np.abs(depth) < 1e-9, 1e-9, depth)
    u = (origin[0] + per_x[0] * col + per_z[0] * row) / safe
    v = (origin[1] + per_x[1] * col + per_z[1] * row) / safe
    rgb = np.asarray(frame.rgb)
    h, w = rgb.shape[:2]
    valid = (depth > 1e-6) & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    u = np.where(valid, u, -10.0).astype(np.float32)
    v = np.where(valid, v, -10.0).astype(np.float32)
    out = cv2.remap(np.ascontiguousarray(rgb[..., :3]), u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out.astype(np.float32), valid, xs, zs


def _box_means(table: np.ndarray, r0: np.ndarray, r1: np.ndarray, c0: np.ndarray, c1: np.ndarray) -> np.ndarray:
    """Mean over the half-open boxes [r0, r1) x [c0, c1) from a summed-area table with a zero border."""
    total = table[r1, c1] - table[r0, c1] - table[r1, c0] + table[r0, c0]
    return total / np.maximum((r1 - r0) * (c1 - c0), 1)


def _find_wall(value: np.ndarray, xs: np.ndarray, zs: np.ndarray, view: _View, with_foot: bool, dark_limit: float) -> dict | None:
    """The housing's near wall: a dark box with brighter board on both sides.

    With ``with_foot`` the box is the whole wall and the board under its foot must be brighter
    too. Without it the box is the upper part of the wall, for views where a neighbor socket
    hides the foot. The search runs over position and over the picture's lateral scale, so the
    wall's own width sets the scale across the face. A box brighter than ``dark_limit`` is not a wall.
    """
    x_lo0, z_lo0 = view.seen_at(-view.wh, view.dh, -view.hh)
    x_hi0, z_hi0 = view.seen_at(view.wh, view.dh, 0.0)
    x_mid0, half0 = 0.5 * (x_lo0 + x_hi0), 0.5 * (x_hi0 - x_lo0)
    height0 = z_hi0 - z_lo0
    table = np.zeros((value.shape[0] + 1, value.shape[1] + 1))
    table[1:, 1:] = np.cumsum(np.cumsum(value, axis=0), axis=1)
    table2 = np.zeros_like(table)
    table2[1:, 1:] = np.cumsum(np.cumsum(value * value, axis=0), axis=1)

    def col(x):
        return np.clip(np.rint((x - xs[0]) / RES_MM).astype(int), 0, len(xs))

    def row(z):
        return np.clip(np.rint((z - zs[0]) / RES_MM).astype(int), 0, len(zs))

    shifts = np.arange(-SEARCH_MM, SEARCH_MM + 1e-9, 0.2)
    dx, dz = np.meshgrid(shifts, shifts)
    inset, band = 0.5, 2.0
    z_box_lo = z_hi0 - (1.0 if with_foot else WALL_SHARE) * height0 + dz
    z_box_hi = z_hi0 + dz
    r0, r1 = row(z_box_lo + inset), row(z_box_hi - inset)
    rs0, rs1 = row(z_box_lo + 1.0), row(z_box_hi - 1.0)
    rb0, rb1 = row(z_box_lo - inset - band), row(z_box_lo - inset)
    best = None
    for kappa in SCALES:
        x_lo, x_hi = x_mid0 - half0 * kappa + dx, x_mid0 + half0 * kappa + dx
        c0, c1 = col(x_lo + inset), col(x_hi - inset)
        inside = _box_means(table, r0, r1, c0, c1)
        inside2 = _box_means(table2, r0, r1, c0, c1)
        spread = np.sqrt(np.maximum(inside2 - inside * inside, 0.0))
        left = _box_means(table, rs0, rs1, col(x_lo - inset - band), col(x_lo - inset))
        right = _box_means(table, rs0, rs1, col(x_hi + inset), col(x_hi + inset + band))
        around = np.minimum(left, right)
        if with_foot:
            around = np.minimum(around, _box_means(table, rb0, rb1, c0, c1))
        score = np.where(inside <= dark_limit, around - inside - 2.0 * spread, -np.inf)
        k = np.unravel_index(int(np.argmax(score)), score.shape)
        if np.isfinite(score[k]) and (best is None or score[k] > best["score"]):
            best = {"score": float(score[k]), "contrast": float(around[k] - inside[k]), "level": float(inside[k]),
                    "x_lo": float(x_lo[k]), "x_hi": float(x_hi[k]), "z_lo": float(z_box_lo[k]), "z_hi": float(z_box_hi[k])}
    if best is None or best["contrast"] < MIN_WALL_CONTRAST:
        return None
    best.update(x_mid0=float(x_mid0), half0=float(half0), z_lo0=float(z_lo0), z_hi0=float(z_hi0), height0=float(height0),
                with_foot=with_foot)
    return best


def _edges(profile: np.ndarray, coords: np.ndarray, lo: float, hi: float, rising: bool = True,
           min_height: float = 3.0) -> list[_Edge]:
    """Every step of one direction between lo and hi, lowest coordinate first.

    A step is a peak of the smoothed slope. Its 2 levels are read where the slope has died down
    on each side, and its position is where the profile crosses halfway between them. That is
    good to a fraction of a camera pixel on an edge the renderer or the lens has blurred.
    """
    p = np.asarray(profile, dtype=float) * (1.0 if rising else -1.0)
    smooth = gaussian_filter1d(p, 1.0, mode="nearest")
    slope = gaussian_filter1d(p, EDGE_SIGMA_MM / RES_MM, order=1, mode="nearest")
    n = len(p)
    i0 = max(int(np.searchsorted(coords, lo)), 1)
    i1 = min(int(np.searchsorted(coords, hi)), n - 1)
    reach = int(round(1.5 / RES_MM))
    out: list[_Edge] = []
    for k in range(i0, i1):
        if not (slope[k] > 0 and slope[k] >= slope[k - 1] and slope[k] > slope[k + 1]):
            continue
        cut = 0.15 * slope[k]
        a = k
        while a > max(k - reach, 1) and cut < slope[a - 1] <= slope[a]:
            a -= 1
        b = k
        while b < min(k + reach, n - 2) and cut < slope[b + 1] <= slope[b]:
            b += 1
        a, b = a - 1, b + 1
        low, high = float(smooth[a]), float(smooth[b])
        if high - low < min_height:
            continue
        half = 0.5 * (low + high)
        j = a
        while j < b - 1 and smooth[j + 1] < half:
            j += 1
        rise = smooth[j + 1] - smooth[j]
        frac = (half - smooth[j]) / rise if rise > 1e-9 else 0.5
        out.append(_Edge(pos=float(coords[j] + min(max(frac, 0.0), 1.0) * RES_MM), height=high - low, low=low, high=high))
    return out


def _strongest(edges: list[_Edge]) -> _Edge | None:
    return max(edges, key=lambda e: e.height) if edges else None


def _band(value: np.ndarray, xs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Mean over the columns between lo and hi (mm), per row."""
    i0 = int(np.searchsorted(xs, lo))
    i1 = max(int(np.searchsorted(xs, hi)), i0 + 1)
    return value[:, i0:i1].mean(axis=1)


def _read_band(profile: np.ndarray, zs: np.ndarray, z_from: float, z_to: float, wall_rows: np.ndarray) -> tuple[_Edge | None, _Edge | None]:
    """The top of the wall and the shoulder of the plug body along 1 band of columns. Either can be None.

    The shoulder is the largest step up. The top of the wall is the first step up, and it has to
    start from the wall's own level and sit clear below the shoulder.
    """
    steps = _edges(profile, zs, z_from, z_to, min_height=MIN_TOP_STEP)
    shoulder = _strongest(steps)
    if shoulder is None or shoulder.height < MIN_SHOULDER_STEP:
        return None, None
    first = steps[0]
    wall_level = float(np.median(profile[wall_rows]))
    if first is shoulder or first.pos > shoulder.pos - 0.3 or abs(first.low - wall_level) > max(3.0, 0.5 * first.height):
        return None, shoulder
    return first, shoulder


def _foot_in_view(view: _View, nominal: NominalGeometry, port: int) -> bool:
    """False when a neighbor socket between the camera and this one hides the foot of the wall."""
    z_foot = view.seen_at(0.0, view.dh, -view.hh)[1]
    for other in range(1, len(nominal.ports_board) + 1):
        if other == port:
            continue
        d = (np.asarray(nominal.port_position(other), dtype=float) - view.p) / MM
        across, out = float(view.a @ d), float(view.n @ d)
        if out <= view.dh or out >= view.cam[1] - 2.0 * view.dh or abs(across) > 2.5 * view.wh:
            continue
        if view.seen_at(across, out - view.dh, 0.0)[1] > z_foot - 1.0:
            return False
    return True


def _strip_pixels(view: _View, strip_mm: float) -> float:
    """How many camera pixels the strip of socket face spans when the plug is home."""
    pts = view.p + (view.db * view.n + np.array([[0.0], [strip_mm]]) * view.z) * MM
    u, v, _ = view.project(pts)
    return float(math.hypot(u[1] - u[0], v[1] - v[0]))


def measure_gap(frame: CameraFrame, nominal: NominalGeometry, port: int, debug: dict | None = None) -> dict:
    """Everything vision.gap can read from 1 frame, as a flat dict of floats. A key that is missing was not measurable."""
    out: dict[str, float] = {"plug_found": 0.0, "housing_found": 0.0, "port_in_view": 0.0, "confidence": 0.0}
    view = _port_view(frame, nominal, port)
    if view is None:
        return out
    rgb = np.asarray(frame.rgb)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return out
    h, w = rgb.shape[:2]
    u, v, depth = view.project(view.p[None, :])
    u, v, depth = float(u[0]), float(v[0]), float(depth[0])
    if depth <= 1e-6:
        return out
    out["port_u"], out["port_v"] = u, v
    out["px_per_mm"] = float(view.k[1, 1] / depth * MM)
    ca, cn, cz = view.cam
    elevation = math.degrees(math.atan2(cz, math.hypot(ca, cn)))
    out["view_elevation_deg"] = elevation
    if not (0 <= u <= w - 1 and 0 <= v <= h - 1):
        return out
    if not (MIN_ELEVATION_DEG <= elevation <= MAX_ELEVATION_DEG) or cn <= view.dh + 5.0:
        return out
    img, valid, xs, zs = _rectify(frame, view)
    if debug is not None:
        debug.update(img=img, valid=valid, xs=xs, zs=zs, view=view)
    if valid.mean() < MIN_VALID_SHARE:
        return out
    out["port_in_view"] = 1.0
    brightness = img.max(axis=2).astype(np.float64)
    dark_limit = DARK_FACTOR * float(np.percentile(brightness[valid], 3.0)) + DARK_OFFSET
    value = np.where(valid, brightness, 255.0)                  # outside the frame never looks like a dark wall

    # ---- the housing's near wall: sides and lateral scale
    foot_in_view = _foot_in_view(view, nominal, port)
    wall = _find_wall(value, xs, zs, view, True, dark_limit) if foot_in_view else None
    if wall is None:
        wall = _find_wall(value, xs, zs, view, False, dark_limit)
    if wall is None:
        return out
    x_lo, x_hi, z_lo, z_hi = wall["x_lo"], wall["x_hi"], wall["z_lo"], wall["z_hi"]
    rows = (zs >= z_lo + 0.25 * (z_hi - z_lo)) & (zs <= z_lo + 0.75 * (z_hi - z_lo))
    across = value[rows].mean(axis=0)
    left = _strongest(_edges(across, xs, x_lo - 1.2, x_lo + 1.2, rising=False))
    right = _strongest(_edges(across, xs, x_hi - 1.2, x_hi + 1.2, rising=True))
    if left is None or right is None:
        return out
    center = 0.5 * (left.pos + right.pos)
    scale = float((right.pos - left.pos) / (2.0 * wall["half0"]))   # lateral scale of the picture, from the wall's width
    if not SCALES[0] - 0.05 <= scale <= SCALES[-1] + 0.05:
        return out
    out["housing_found"] = 1.0
    out["shift_x_mm"] = float(center - wall["x_mid0"])
    out["scale_ratio"] = scale
    out["wall_contrast"] = float(wall["contrast"] / 255.0)

    # ---- the top of the wall and the shoulder of the plug body, read along each of the 2 flank bands
    lo_x, hi_x = (view.wt + FLANK_MARGIN_MM) * scale, (view.wb - FLANK_MARGIN_MM) * scale
    if hi_x - lo_x < 0.6:
        return out
    flank_l = _band(value, xs, center - hi_x, center - lo_x)
    flank_r = _band(value, xs, center + lo_x, center + hi_x)
    z_mid = 0.5 * (z_lo + z_hi)
    i_mid = int(np.searchsorted(zs, z_mid))
    j0, j1 = int(np.searchsorted(xs, center - hi_x)), int(np.searchsorted(xs, center + hi_x)) + 1
    outside = np.flatnonzero(~valid[i_mid:, j0:j1].all(axis=1))          # rows above the wall that leave the frame
    z_to = zs[-1] - 1.0 if outside.size == 0 else min(zs[-1] - 1.0, zs[i_mid + int(outside[0])] - 0.5)
    wall_rows = (zs >= z_lo + 1.0) & (zs <= z_mid)
    read = [_read_band(profile, zs, z_mid, z_to, wall_rows) for profile in (flank_l, flank_r)]
    if debug is not None:
        debug.update(wall=wall, center=center, scale=scale, flank_l=flank_l, flank_r=flank_r, x_left=left.pos,
                     x_right=right.pos, lo_x=lo_x, hi_x=hi_x, z_box=(z_lo, z_hi), read=read)
    shoulders = [s for _, s in read if s is not None]
    if not shoulders:
        return out
    out["edge_contrast"] = float(np.mean([s.height for s in shoulders]) / 255.0)
    pairs = [(t, s) for t, s in read if t is not None]
    if not pairs:
        return out                                              # the wall's top and the shoulder cannot be told apart
    z_top = float(np.mean([t.pos for t, _ in pairs]))
    z_shoulder = float(np.mean([s.pos for _, s in pairs]))
    top_height = float(np.mean([t.height for t, _ in pairs]))
    low = float(np.mean([s.low for _, s in pairs]))
    high = float(np.mean([s.high for _, s in pairs]))
    out["bands_read"] = float(len(pairs))
    out["top_contrast"] = top_height / 255.0
    out["shift_z_mm"] = float(z_top - wall["z_hi0"])

    # ---- scale along the plug axis, from the wall's own height where its foot is in view
    scale_z = 1.0
    if foot_in_view:
        middle = _band(value, xs, center - 0.6 * view.wh * scale, center + 0.6 * view.wh * scale)
        foot = _strongest(_edges(middle, zs, z_top - 1.12 * wall["height0"], z_top - 0.88 * wall["height0"],
                                 rising=False, min_height=MIN_FOOT_STEP))
        if foot is not None:
            scale_z = (z_top - foot.pos) / wall["height0"]
            if debug is not None:
                debug.update(z_foot=foot.pos)
    out["scale_z_ratio"] = float(scale_z)
    seated_strip = -view.seen_at(0.0, view.dh, 0.0)[1]          # height of socket face that shows with the plug home
    gap = (z_shoulder - z_top) / scale_z - seated_strip
    slant = None
    if len(pairs) == 2:                                         # against the socket's own top edge, so camera roll cancels
        (t_l, s_l), (t_r, s_r) = pairs
        slant = math.degrees(math.atan2((s_r.pos - t_r.pos) - (s_l.pos - t_l.pos), view.wt + view.wb))
    elif len(shoulders) == 2:
        slant = math.degrees(math.atan2(shoulders[1].pos - shoulders[0].pos, view.wt + view.wb))
    if slant is not None:
        out["slant_deg"] = float(slant)

    # ---- how much tongue shows under the body, along the middle columns
    half = max((view.wt - CENTER_MARGIN_MM) * scale, 0.3)
    mid = gaussian_filter1d(_band(value, xs, center - half, center + half), 1.0, mode="nearest")
    lit = low + 0.4 * (high - low)                              # brighter than this is body or tongue, darker is socket
    k = min(int(np.searchsorted(zs, z_shoulder)) + 2, len(zs) - 2)
    k_stop = max(int(np.searchsorted(zs, z_top - 0.5)), 1)
    while k > k_stop and mid[k] > lit:
        k -= 1
    z_tongue_end = float(zs[k])
    if mid[k] <= lit < mid[k + 1]:
        z_tongue_end += (lit - mid[k]) / (mid[k + 1] - mid[k]) * RES_MM
    tongue = max(0.0, (z_shoulder - z_tongue_end) / scale_z)
    out["tongue_mm"] = float(tongue)
    if debug is not None:
        debug.update(z_shoulder=z_shoulder, z_top=z_top, z_tongue_end=z_tongue_end, mid=mid, half=half, gap_raw=gap)

    if gap < -BELOW_FACE_MM or gap > view.depth + 2.0:
        return out                                              # not a plug body sitting in this socket
    if gap > EMPTY_GAP_MM and tongue < 0.3 * (gap - 2.0):
        return out                                              # a large gap with no tongue in it: an empty socket
    out["plug_found"] = 1.0
    out["gap_mm_est"] = float(gap)
    conf = min(1.0, (high - low) / 40.0) * min(1.0, top_height / 10.0) * min(1.0, _strip_pixels(view, seated_strip) / 2.0)
    conf *= math.exp(-abs(scale_z - 1.0) / 0.1) * (1.0 if len(pairs) == 2 else 0.6)
    if slant is not None:
        conf *= math.exp(-max(0.0, abs(slant) - 1.5) / 4.0)
    out["confidence"] = float(np.clip(conf, 0.0, 1.0))
    return out


class VisionGap:
    """Passive vision: how many millimeters of tongue show between the plug body and the socket face."""

    source_id = "vision.gap"
    kind = "passive_vision"
    predicates = {"seated", "latched", "at_port"}

    def estimate_cost(self, claim: Claim) -> Cost:
        return Cost(seconds=0.1, usd=0.0, moved_robot=False)

    def collect(self, station: Station, claim: Claim, params: dict) -> Evidence:
        t0 = float(station.robot_state().ts)
        nominal = params.get("nominal")
        frames = params.get("frames")
        which = params.get("view", "claim")
        frame = None
        if nominal is not None and frames is not None:
            arm = nominal.free_arm if which == "relook" else params.get("arm", nominal.holding_arm)
            key = f"{which}/{arm}_camera_d405"
            if which == "now":                              # a fresh frame from the wrist camera, taken here
                camera = f"{arm}_camera_d405"
                frames.put(key, station.observe([camera]).frames[camera])
            frame = frames.get(key)
        port = claim.predicate.target.get("port")
        if frame is None or not isinstance(port, int) or not 1 <= port <= len(nominal.ports_board):
            return make_evidence(self, claim, {}, score=None, seconds=float(station.robot_state().ts) - t0)
        features = measure_gap(frame, nominal, port)
        if features.get("plug_found") and "gap_mm_est" in features:
            score = -features["gap_mm_est"]
        else:
            score = -nominal.full_depth_m / MM
        return make_evidence(self, claim, features, score=score, seconds=float(station.robot_state().ts) - t0)


register(VisionGap())
