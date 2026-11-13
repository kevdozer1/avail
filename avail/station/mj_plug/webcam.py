"""Hard mode's wrist camera: a 1080p USB 2.0 webcam with a 130 degree fisheye-style lens (S4b).

Simulator side. The station renders a wide pinhole view with MuJoCo and this module turns it
into the frame a calibrated pipeline hands to an evidence source, in the order a real camera
and pipeline make it:

1. The lens. Each 1080p sensor pixel looks along the ray this seed's lens unit gives it: an
   equidistant fisheye (image radius = f x angle) with the unit's own focal length, optical
   center and k1, k2 terms. Its value is read from the pinhole render along that ray.
2. Motion. The camera's pose over the last 0.1 s gives each row its own moment (rolling
   shutter) and a smear over the exposure (motion blur). A camera that is still gets neither.
3. Blur. The lens's own blur and the defocus of a fixed-focus lens at the port's distance.
4. The sensor. Auto exposure meters the central half of the frame and sets the exposure time,
   then the gain. Photo-electrons get shot noise and read noise, so a dim scene is noisier.
   White balance is off by this episode's error and a little more each frame. 8-bit sRGB out.
5. MJPEG at the stream's quality, with 4:2:2 color.
6. The pipeline rectifies with its calibration of the lens, which is close to the unit and not
   equal to it, then downscales to the working size. The frame carries that calibration's
   intrinsics, so what is left of the lens error stays in the picture.

The pinhole render only needs to cover the rays the rectified frame uses, about 49 degrees off
axis, so the raw frame's far corners are left dark. Nothing downstream reads them.

The station gives this module 2 things a renderer needs and an evidence source never sees: the
camera's pose history, for the motion, and the distance to the port, for the defocus.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from ...config import WristCameraSpec

COVER_MARGIN = 1.06            # the pinhole render covers the rectified frame's rays with this much room on each side
UNIT_FOCAL_FLOOR = 0.97        # the shortest focal length the render plans for, against the nominal
MOTION_MIN_PX = 0.05           # a row shift below this, in 1080p pixels, is not applied
BLUR_MIN_PX = 0.2              # a motion smear below this is not applied
CHROMA_CODES = {"4:4:4": 0, "4:2:2": 1, "4:2:0": 2}


# ----------------------------------------------------------------------------- the lens
def nominal_focal_px(spec: WristCameraSpec) -> float:
    """Pixels per radian of the nominal equidistant lens: the half diagonal over the half field of view."""
    w, h = spec.sensor_px
    return 0.5 * math.hypot(w, h) / math.radians(0.5 * spec.diagonal_fov_deg)


def distort(theta: np.ndarray, k) -> np.ndarray:
    """Kannala-Brandt: the angle the image radius stands for, from the true angle off axis."""
    t2 = theta * theta
    return theta * (1.0 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))))


def undistort(theta_d: np.ndarray, k, iterations: int = 8) -> np.ndarray:
    """The inverse of ``distort`` by Newton's method. Exact to float precision for these small k."""
    theta = np.array(theta_d, dtype=np.float64, copy=True)
    for _ in range(iterations):
        t2 = theta * theta
        f = theta * (1.0 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3])))) - theta_d
        df = 1.0 + t2 * (3 * k[0] + t2 * (5 * k[1] + t2 * (7 * k[2] + t2 * 9 * k[3])))
        theta -= f / df
    return theta


@dataclass(frozen=True)
class LensUnit:
    """1 camera unit and the pipeline's calibration of it, in 1080p pixels with pixel centers at integers."""

    f: float
    cx: float
    cy: float
    k: tuple[float, float, float, float]
    f_est: float
    cx_est: float
    cy_est: float
    k_est: tuple[float, float, float, float]
    awb: tuple[float, float]           # this episode's white balance error on red and on blue

    def as_dict(self) -> dict:
        return {"f": self.f, "cx": self.cx, "cy": self.cy, "k": list(self.k), "f_est": self.f_est,
                "cx_est": self.cx_est, "cy_est": self.cy_est, "k_est": list(self.k_est), "awb": list(self.awb)}


def draw_unit(spec: WristCameraSpec, rng: np.random.Generator) -> LensUnit:
    """A lens unit and its calibration, drawn from the spreads in the config."""
    w, h = spec.sensor_px
    f0 = nominal_focal_px(spec)
    f = f0 * (1.0 + rng.uniform(-1.0, 1.0) * spec.unit_focal_pct / 100.0)
    cx = (w - 1) / 2.0 + rng.uniform(-1.0, 1.0) * spec.unit_center_px
    cy = (h - 1) / 2.0 + rng.uniform(-1.0, 1.0) * spec.unit_center_px
    k = (float(rng.normal(0.0, spec.unit_k_sd[0])), float(rng.normal(0.0, spec.unit_k_sd[1])), 0.0, 0.0)
    f_est = f * (1.0 + rng.normal(0.0, spec.calib_focal_pct_sd / 100.0))
    cx_est = cx + rng.normal(0.0, spec.calib_center_px_sd)
    cy_est = cy + rng.normal(0.0, spec.calib_center_px_sd)
    k_est = (k[0] + float(rng.normal(0.0, spec.calib_k_sd[0])), k[1] + float(rng.normal(0.0, spec.calib_k_sd[1])), 0.0, 0.0)
    awb = (float(rng.normal(0.0, spec.awb_error_sd)), float(rng.normal(0.0, spec.awb_error_sd)))
    return LensUnit(float(f), float(cx), float(cy), k, float(f_est), float(cx_est), float(cy_est), k_est, awb)


def render_plan(spec: WristCameraSpec) -> dict:
    """The pinhole render that covers every ray the rectified frame uses: size, focal length and fovy.

    The render keeps the nominal lens's pixels per radian at the center, so the center of the
    raw frame samples it 1 to 1.
    """
    w, h = spec.sensor_px
    f0 = nominal_focal_px(spec)
    f_low = f0 * UNIT_FOCAL_FLOOR
    half_tan_x = COVER_MARGIN * (w / 2.0) / f_low
    half_tan_y = COVER_MARGIN * (h / 2.0) / f_low
    width = int(2 * math.ceil(f0 * half_tan_x))
    height = int(2 * math.ceil(f0 * half_tan_y))
    fovy = 2.0 * math.degrees(math.atan(0.5 * height / f0))
    return {"width": width, "height": height, "f": f0, "fovy_deg": fovy}


# ----------------------------------------------------------------------------- color
def _srgb_to_linear_lut() -> np.ndarray:
    v = np.arange(256, dtype=np.float64) / 255.0
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _linear_to_srgb_lut(n: int = 65536) -> np.ndarray:
    v = np.linspace(0.0, 1.0, n)
    s = np.where(v <= 0.0031308, 12.92 * v, 1.055 * np.power(v, 1.0 / 2.4) - 0.055)
    return np.clip(np.rint(s * 255.0), 0, 255).astype(np.uint8)


TO_LINEAR = _srgb_to_linear_lut()
TO_SRGB = _linear_to_srgb_lut()


# ----------------------------------------------------------------------------- the camera
@dataclass
class Motion:
    """The camera's pose before the frame, for the rolling shutter and the motion blur.

    ``times`` are seconds before the moment of the render, ascending toward 0. ``points_cam`` is
    the reference point (the port) in the camera's OpenCV frame at each of those times, meters.
    """

    times: np.ndarray
    points_cam: np.ndarray


@dataclass
class FrameInfo:
    exposure_s: float = 0.0
    gain: float = 1.0
    meter: float = 0.0
    ev_jitter: float = 0.0
    blur_sigma_px: float = 0.0
    defocus_sigma_px: float = 0.0
    shutter_shift_px: float = 0.0
    motion_blur_px: float = 0.0
    jpeg_bytes: int = 0
    awb_gains: tuple[float, float] = (1.0, 1.0)
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        out = {k: v for k, v in self.__dict__.items() if k != "extra"}
        out["awb_gains"] = list(self.awb_gains)
        out.update(self.extra)
        return out


class Webcam:
    """1 wrist camera unit with its calibration. ``process`` makes 1 working frame from 1 pinhole render."""

    def __init__(self, spec: WristCameraSpec, unit: LensUnit):
        self.spec, self.unit = spec, unit
        self.plan = render_plan(spec)
        self.sensor_w, self.sensor_h = spec.sensor_px
        self.work_w, self.work_h = spec.working_px
        self._raw_maps = None                                          # built on the first frame
        self._rect_maps = None
        cxr, cyr = (self.sensor_w - 1) / 2.0, (self.sensor_h - 1) / 2.0
        self.k_rect = np.array([[unit.f_est, 0.0, cxr], [0.0, unit.f_est, cyr], [0.0, 0.0, 1.0]])
        scale_x, scale_y = self.sensor_w / self.work_w, self.sensor_h / self.work_h
        k = self.k_rect
        # INTER_AREA averages whole blocks, so a full-size pixel center c lands at (c + 0.5) / s - 0.5
        self.k_work = np.array([[k[0, 0] / scale_x, 0.0, (k[0, 2] + 0.5) / scale_x - 0.5],
                                [0.0, k[1, 1] / scale_y, (k[1, 2] + 0.5) / scale_y - 0.5],
                                [0.0, 0.0, 1.0]])

    # ---------------------------------------------------------------- geometry
    def _raw_from_pinhole(self) -> tuple[np.ndarray, np.ndarray]:
        """For each raw sensor pixel, where its ray lands in the pinhole render. The true lens.

        Both maps are radial: a pixel at radius r from the optical center lands at radius
        scale(r) x r from the render's center, along the same direction. scale is tabulated once
        on a fine grid of r, which is exact to well under a hundredth of a pixel.
        """
        u = self.unit
        plan = self.plan
        dx = (np.arange(self.sensor_w, dtype=np.float32) - np.float32(u.cx))[None, :]
        dy = (np.arange(self.sensor_h, dtype=np.float32) - np.float32(u.cy))[:, None]
        r = np.sqrt(dx * dx + dy * dy)
        r_max = float(r.max()) + 2.0
        grid = np.linspace(0.0, r_max, 16384)
        theta = undistort(grid / u.f, u.k)
        ok = theta < math.radians(85.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(ok, plan["f"] * np.tan(np.where(ok, theta, 0.0)) / grid, 1e6)
        scale[0] = plan["f"] / u.f
        s = np.interp(r, grid, scale).astype(np.float32)
        cxp, cyp = np.float32((plan["width"] - 1) / 2.0), np.float32((plan["height"] - 1) / 2.0)
        return (cxp + s * dx).astype(np.float32), (cyp + s * dy).astype(np.float32)

    def _rect_from_raw(self) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
        """For each pixel of the rectified 1080p frame, where the calibration puts it in the raw frame. Radial, as above."""
        u = self.unit
        cxr, cyr = (self.sensor_w - 1) / 2.0, (self.sensor_h - 1) / 2.0
        fr = u.f_est
        x = ((np.arange(self.sensor_w, dtype=np.float32) - np.float32(cxr)) / np.float32(fr))[None, :]
        y = ((np.arange(self.sensor_h, dtype=np.float32) - np.float32(cyr)) / np.float32(fr))[:, None]
        rho = np.sqrt(x * x + y * y)
        grid = np.linspace(0.0, float(rho.max()) + 0.01, 16384)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = u.f_est * distort(np.arctan(grid), u.k_est) / grid
        scale[0] = u.f_est
        s = np.interp(rho, grid, scale).astype(np.float32)
        map_x = (np.float32(u.cx_est) + s * x).astype(np.float32)
        map_y = (np.float32(u.cy_est) + s * y).astype(np.float32)
        k_rect = np.array([[fr, 0.0, cxr], [0.0, fr, cyr], [0.0, 0.0, 1.0]])
        return (map_x, map_y), k_rect

    def project_raw(self, points_cam: np.ndarray) -> np.ndarray:
        """Camera-frame points (OpenCV axes, meters) to raw 1080p pixels through the true lens."""
        p = np.atleast_2d(np.asarray(points_cam, dtype=np.float64))
        rho = np.hypot(p[:, 0], p[:, 1])
        theta = np.arctan2(rho, p[:, 2])
        phi = np.arctan2(p[:, 1], p[:, 0])
        rd = self.unit.f * distort(theta, self.unit.k)
        return np.stack([rd * np.cos(phi) + self.unit.cx, rd * np.sin(phi) + self.unit.cy], axis=1)

    def raw_px_per_mm(self, point_cam: np.ndarray, step_m: float = 1e-4) -> float:
        """Raw 1080p pixels per millimeter of motion across the line of sight at a camera-frame point."""
        p = np.asarray(point_cam, dtype=np.float64)
        sight = p / np.linalg.norm(p)
        across = np.cross(sight, [0.0, 1.0, 0.0])
        if np.linalg.norm(across) < 1e-9:
            across = np.cross(sight, [1.0, 0.0, 0.0])
        across /= np.linalg.norm(across)
        a, b = self.project_raw(np.stack([p, p + step_m * across]))
        return float(np.linalg.norm(b - a) / (step_m * 1000.0))

    def defocus_sigma_px(self, distance_m: float) -> float:
        """Gaussian sd, in 1080p pixels, of a fixed-focus lens's blur disk at ``distance_m`` (thin lens)."""
        s = self.spec
        pitch_mm = s.pixel_pitch_um / 1000.0
        f_mm = self.unit.f * pitch_mm
        sf_mm, d_mm = s.focus_distance_m * 1000.0, max(float(distance_m) * 1000.0, 2.0 * f_mm)
        coc_mm = (f_mm * f_mm / (s.f_number * (sf_mm - f_mm))) * abs(d_mm - sf_mm) / d_mm
        return float(coc_mm / pitch_mm / 4.0)        # a uniform disk of diameter D has a per-axis sd of D / 4

    # ---------------------------------------------------------------- the frame
    def process(self, pinhole_rgb: np.ndarray, rng: np.random.Generator, *, distance_m: float,
                motion: Motion | None = None, keep: dict | None = None) -> tuple[np.ndarray, np.ndarray, FrameInfo]:
        """The working frame, its intrinsics, and what the camera did to make it.

        ``keep``, when given, receives the raw 1080p frame as the sensor delivered it (``raw``,
        before MJPEG) and the rectified 1080p frame (``rectified``), for pictures only.
        """
        spec, unit = self.spec, self.unit
        plan = self.plan
        if pinhole_rgb.shape[:2] != (plan["height"], plan["width"]):
            raise ValueError(f"the pinhole render is {pinhole_rgb.shape[:2]}, expected {(plan['height'], plan['width'])}")
        info = FrameInfo()
        if self._raw_maps is None:
            self._raw_maps = self._raw_from_pinhole()
            self._rect_maps, _ = self._rect_from_raw()
        # 1. the lens: raw sensor frame from the render
        raw = cv2.remap(np.ascontiguousarray(pinhole_rgb[..., :3]), *self._raw_maps, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        lin = TO_LINEAR[raw]                                          # float32, linear light

        # auto exposure first, because the exposure time sets how far the camera moved during it
        h, w = self.sensor_h, self.sensor_w
        center = lin[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
        meter = float(np.mean(0.2126 * center[..., 0] + 0.7152 * center[..., 1] + 0.0722 * center[..., 2]))
        jitter = float(rng.normal(0.0, spec.ae_jitter_ev))
        want = spec.ae_target / max(meter, 1e-4) * (2.0 ** jitter)    # output per unit of rendered light
        need = want * spec.full_well_e / spec.white_e_per_s           # exposure seconds times gain
        exposure = min(spec.max_exposure_s, need)
        gain = float(np.clip(need / spec.max_exposure_s, 1.0, spec.max_gain))
        info.meter, info.ev_jitter, info.exposure_s, info.gain = meter, jitter, float(exposure), gain

        # 2. motion: rolling shutter and motion blur, from where the reference point was in the frame
        if motion is not None and len(motion.times) >= 2:
            lin = self._apply_motion(lin, motion, exposure, info)

        # 3. blur: the lens and the defocus at the port's distance
        defocus = self.defocus_sigma_px(distance_m)
        sigma = math.hypot(spec.lens_psf_px, defocus)
        info.defocus_sigma_px, info.blur_sigma_px = defocus, sigma
        if sigma > 0.05:
            lin = cv2.GaussianBlur(lin, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)

        # 4. the sensor: electrons, shot and read noise, gain, white balance, 8-bit sRGB
        electrons = lin * np.float32(spec.white_e_per_s * exposure)
        noise = rng.standard_normal(electrons.shape, dtype=np.float32)
        electrons = electrons + np.sqrt(electrons + np.float32(spec.read_noise_e ** 2)) * noise
        np.clip(electrons, 0.0, spec.full_well_e, out=electrons)
        out = electrons * np.float32(gain / spec.full_well_e)
        awb_r = 1.0 + unit.awb[0] + float(rng.normal(0.0, spec.awb_jitter_sd))
        awb_b = 1.0 + unit.awb[1] + float(rng.normal(0.0, spec.awb_jitter_sd))
        out[..., 0] *= np.float32(awb_r)
        out[..., 2] *= np.float32(awb_b)
        info.awb_gains = (awb_r, awb_b)
        np.clip(out, 0.0, 1.0, out=out)
        srgb = TO_SRGB[np.rint(out * (len(TO_SRGB) - 1)).astype(np.int32)]

        # 5. MJPEG
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(srgb).save(buf, format="JPEG", quality=int(spec.jpeg_quality),
                                   subsampling=CHROMA_CODES[spec.chroma_subsampling])
        info.jpeg_bytes = buf.tell()
        decoded = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))

        # 6. rectify with the calibration, then downscale
        rect = cv2.remap(decoded, *self._rect_maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        work = cv2.resize(rect, (self.work_w, self.work_h), interpolation=cv2.INTER_AREA)
        if keep is not None:
            keep["raw"], keep["rectified"] = srgb, rect
        return np.ascontiguousarray(work), self.k_work.copy(), info

    def _apply_motion(self, lin: np.ndarray, motion: Motion, exposure: float, info: FrameInfo) -> np.ndarray:
        """Shift each row to where the camera was while that row was exposed, and smear it along the motion."""
        spec = self.spec
        h = self.sensor_h
        times = np.asarray(motion.times, dtype=np.float64)
        px = self.project_raw(motion.points_cam)                       # where the point was, raw pixels, over time
        now = px[-1]

        def at(t: np.ndarray) -> np.ndarray:
            t = np.clip(t, times[0], times[-1])
            return np.stack([np.interp(t, times, px[:, 0]), np.interp(t, times, px[:, 1])], axis=-1)

        rows = np.linspace(0.0, h - 1.0, 9)
        ends = -spec.readout_s * (1.0 - rows / (h - 1.0))              # each row's read time, before the render
        samples = np.linspace(0.0, 1.0, 7)
        shifts = []
        for end in ends:
            window = end - exposure * (1.0 - samples)
            shifts.append(at(window).mean(axis=0) - now)
        shifts = np.array(shifts)                                      # 9 rows x 2
        port_row = int(np.clip(round(now[1]), 0, h - 1))
        end_port = -spec.readout_s * (1.0 - port_row / (h - 1.0))
        smear = at(np.array([end_port]))[0] - at(np.array([end_port - exposure]))[0]
        info.shutter_shift_px = float(np.abs(shifts).max())
        info.motion_blur_px = float(np.hypot(*smear))
        out = lin
        if info.shutter_shift_px >= MOTION_MIN_PX:
            # A row exposed when the content sat at now + d shows it there, so the row samples the render at x - d.
            row_dx = np.interp(np.arange(h), rows, shifts[:, 0]).astype(np.float32)
            row_dy = np.interp(np.arange(h), rows, shifts[:, 1]).astype(np.float32)
            grid_x = np.arange(self.sensor_w, dtype=np.float32)[None, :] - row_dx[:, None]
            grid_y = np.arange(h, dtype=np.float32)[:, None] - row_dy[:, None]
            grid_x = np.broadcast_to(grid_x, (h, self.sensor_w)).astype(np.float32)
            grid_y = np.broadcast_to(grid_y, (h, self.sensor_w)).astype(np.float32)
            out = cv2.remap(out, grid_x, grid_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if info.motion_blur_px >= BLUR_MIN_PX:
            out = cv2.filter2D(out, -1, line_kernel(smear), borderType=cv2.BORDER_REPLICATE)
        return out


def line_kernel(vector) -> np.ndarray:
    """A normalized smear kernel along ``vector`` (pixels), drawn anti-aliased."""
    dx, dy = float(vector[0]), float(vector[1])
    length = math.hypot(dx, dy)
    size = 2 * int(math.ceil(length / 2.0)) + 3
    kernel = np.zeros((size, size), dtype=np.float32)
    c = size // 2
    n = max(8, int(math.ceil(length * 4)))
    for t in np.linspace(-0.5, 0.5, n):
        x, y = c + t * dx, c + t * dy
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        fx, fy = x - x0, y - y0
        for (xx, yy, wgt) in ((x0, y0, (1 - fx) * (1 - fy)), (x0 + 1, y0, fx * (1 - fy)),
                              (x0, y0 + 1, (1 - fx) * fy), (x0 + 1, y0 + 1, fx * fy)):
            if 0 <= xx < size and 0 <= yy < size:
                kernel[yy, xx] += wgt
    return kernel / kernel.sum()
