"""judge.vlm: ask a vision-language model whether the claimed predicate holds.

The source sends 1 to 3 camera views with a short text: the predicate's definition, the
connector type in plain words, the target and what each image is. The model answers with JSON
``{"holds": true|false, "confidence": 0..1, "reason": "..."}``. ``p_yes`` is ``confidence`` when
``holds`` is true, else ``1 - confidence``. Every call goes through ``avail.router``: it is
cached, charged to a spend session and refused before any network I/O when a cap would be
crossed. Reasoning is off and usage is on, which are the router's defaults.

What ``collect`` reads from ``params``:

- ``frames``     a ``FrameStore``: ``claim/<camera>`` frames at claim time, ``relook/<camera>`` after a closer look
- ``nominal``    the taught geometry. The cropped variants use it to find the port in each frame
- ``connector``  the connector type (default ``nominal.connector``)
- ``arm``        the arm that holds the plug (default ``nominal.holding_arm``)
- ``judge``      a dict of settings:

  - ``model``      a key of ``config/models.yaml``. It has no default, so nothing is spent by accident
  - ``variant``    ``full``, ``crop`` (default) or ``crop_ref``
  - ``router``     the ``Router`` to call through. Default ``Router()``: no spend session, so only the
                   fake model and cached answers work
  - ``reference``  for ``crop_ref``: an RGB array of a correctly seated plug of the same type
  - ``views``      a list of frame keys to send, in order, in place of the default views
  - ``max_tokens`` the output allowance, in place of the default
  - ``logprobs``   true asks the provider for token log-probabilities and adds ``p_yes_logprob``

The 3 variants (docs/03_evidence_sources.md, after FailBench's report that cropping helped):

- ``full``      1 to 3 whole frames: the holding arm's wrist view at claim time, then the relook
                view when there is one, then the overhead view. The text does not say where the
                target port is in a frame.
- ``crop``      the same views, each cropped to a square around the taught port. The port is
                projected into the frame with the frame's own intrinsics and extrinsics, and the
                square is centered on that pixel. The square is 80 mm on a side at the port's
                distance from the camera, at least 96 pixels of the frame and at most the frame's
                shorter side. A crop under 384 pixels on a side is resized up to 384 with cubic
                interpolation. The part of the square that falls outside the frame is black. The
                real port sits as far from the center as the calibration and the taught position
                are off, so the text says at or near the center.
- ``crop_ref``  the crops, after 1 reference image of a correctly seated plug of the same type.
                The text says that image 1 is the reference.

2 cases of the cropped variants:

- A calibrated view whose frame does not hold the port is left out, because it does not show
  the port.
- A frame without calibration cannot be projected into. Its crop is then the central square of
  the frame, the text says so and no longer says where the port is, and the feature
  ``crop_on_port`` is 0.

The text is a function of the claim's predicate (name, definition, target), the connector type,
the variant and which views are sent. It never carries a claim id, a file name, a pixel
position or any number measured on the state, so 2 states with the same claim get the same
text. The images are camera pixels and nothing is drawn on them.

Features: ``p_yes``, ``holds`` (1.0 or 0.0), ``confidence``, ``n_images``, ``cached`` (1.0 when the
answer came from the router's cache), ``placeholder`` (1.0 when the model is the offline fake),
``crop_on_port`` (cropped variants only), and ``p_yes_logprob`` when asked for and returned. An
answer that fails its schema gives ``json_error`` 1.0, no ``p_yes`` and score None. ``truncated``
is 1.0 when the answer was cut at the output allowance.

Score: ``p_yes``. Cost: ``seconds`` is the measured latency of the call and ``usd`` is what the
router charged, which is 0 for a cached answer and for the fake model.

Missing frames, a missing model or a missing reference give an evidence with no features and
score None. A refusal by the router (no key, no spend session, a cap) and a failed call are
raised as the router raised them: a judge that went quiet would look like a judge that
abstains.

The fake model's answers are placeholders. They say nothing about a judge, and ``placeholder``
marks every evidence that holds one.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field, replace
from typing import Any

import cv2
import numpy as np

from ..router import Router, default_models, estimate_usd, get_adapter, resolve_route
from ..schemas import Claim, Cost, Evidence
from ..station.base import CameraFrame, NominalGeometry, Station
from .base import FrameStore, make_evidence, register

VARIANTS = ("full", "crop", "crop_ref")
DEFAULT_VARIANT = "crop"
MAX_VIEWS = 3

#: the answer of docs/03_evidence_sources.md
ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "holds": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
    },
    "required": ["holds", "confidence", "reason"],
    "additionalProperties": False,
}
SCHEMA_NAME = "judge_answer"

#: output allowance of 1 call. The answer is about 60 tokens
MAX_TOKENS = 300
#: for a model that cannot turn reasoning off: its reasoning tokens count against the allowance
MAX_TOKENS_REASONING = 1000
#: the middle of the 2 to 10 s that docs/03_evidence_sources.md gives for a judge call
TYPICAL_SECONDS = 6.0

CROP_MM = 80.0                # side of the crop at the port's distance from the camera
CROP_MIN_PX = 96              # a crop is never smaller than this in the frame
CROP_OUT_PX = 384             # a crop under this many pixels on a side is resized up to it
LOGPROBS_BODY = {"logprobs": True, "top_logprobs": 5}

CONNECTOR_WORDS = {
    "fric": "a friction-fit plug such as USB-A. Friction alone holds it in the socket.",
    "latch": "a latching plug such as RJ45. A latch on the plug snaps into the socket when the plug is fully in.",
    "prong": "a plug with prongs such as a mains power plug. The prongs slide into slots in the socket.",
}

ROLE_HOLDING = "holding_wrist"
ROLE_FREE = "free_wrist"
ROLE_OVERHEAD = "overhead"
ROLE_OTHER = "other"
_CAMERA_WORDS = {
    ROLE_HOLDING: "the camera on the wrist of the arm that holds the plug",
    ROLE_FREE: "the camera on the wrist of the other arm",
    ROLE_OVERHEAD: "the overhead camera",
    ROLE_OTHER: "another camera",
}


# ----------------------------------------------------------------------------- views
@dataclass(frozen=True)
class View:
    """One image of the plug being judged: which frame it is, and how it was cropped."""

    key: str                      # the frame key, for example claim/right_camera_d405
    when: str                     # claim or relook
    role: str                     # one of the ROLE_ names
    crop: str = "none"            # none (a whole frame), port (centered on the taught port) or center


def _split_key(key: str) -> tuple[str, str]:
    when, _, camera = key.partition("/")
    return (when, camera) if camera else ("claim", when)


def view_of(key: str, holding: str, free: str) -> View:
    """The view a frame key names. The role comes from the camera's name: ``<arm>_...`` or ``top...``."""
    when, camera = _split_key(key)
    if holding and camera.startswith(holding + "_"):
        role = ROLE_HOLDING
    elif free and camera.startswith(free + "_"):
        role = ROLE_FREE
    elif camera.startswith("top"):
        role = ROLE_OVERHEAD
    else:
        role = ROLE_OTHER
    return View(key=key, when=when, role=role)


def select_views(frames: FrameStore, holding: str, free: str, wanted: list[str] | None = None) -> list[View]:
    """The views to send, in order. At most 3.

    Default: the holding arm's wrist view at claim time, the relook view when there is one (the
    free arm's camera when it took one), then the overhead view. ``wanted`` replaces the default
    with the caller's own keys, and a key that has no frame is skipped.
    """
    if wanted is not None:
        keys = [key for key in wanted if isinstance(key, str) and frames.get(key) is not None]
        return [view_of(key, holding, free) for key in keys[:MAX_VIEWS]]
    claim = [view_of(key, holding, free) for key in frames.keys("claim/")]
    relook = [view_of(key, holding, free) for key in frames.keys("relook/")]
    picked: list[View] = []
    picked += [v for v in claim if v.role == ROLE_HOLDING][:1]
    picked += ([v for v in relook if v.role == ROLE_FREE] or relook)[:1]
    picked += ([v for v in claim if v.role == ROLE_OVERHEAD] or [v for v in claim if v.role == ROLE_OTHER])[:1]
    return picked


def typical_views(variant: str, n: int = 2) -> list[View]:
    """The views a call usually sends, for a cost estimate made before any frame exists."""
    crop = "none" if variant == "full" else "port"
    wrist = View("claim/holding", "claim", ROLE_HOLDING, crop)
    relook = View("relook/free", "relook", ROLE_FREE, crop)
    overhead = View("claim/top", "claim", ROLE_OVERHEAD, crop)
    return {1: [wrist], 2: [wrist, overhead]}.get(int(n), [wrist, relook, overhead])


# ----------------------------------------------------------------------------- crops
@dataclass
class Crop:
    """A square cut from a frame. ``x0``, ``y0`` and ``side`` are the window in the frame, in pixels."""

    rgb: np.ndarray
    on_port: bool                 # True: centered on the projected port. False: the central square of the frame
    x0: int
    y0: int
    side: int
    scale: float                  # crop pixels per frame pixel
    port_uv: tuple[float, float] | None = None

    def locate(self, u: float, v: float) -> tuple[float, float]:
        """Where the frame pixel (u, v) lands in the crop."""
        return (u - self.x0 + 0.5) * self.scale - 0.5, (v - self.y0 + 0.5) * self.scale - 0.5


def _rgb(frame: CameraFrame | None) -> np.ndarray | None:
    """The frame's pixels as H x W x 3 uint8, or None when it holds no usable image."""
    if frame is None:
        return None
    rgb = np.asarray(getattr(frame, "rgb", None))
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] not in (3, 4) or rgb.shape[0] < 2 or rgb.shape[1] < 2:
        return None
    return np.ascontiguousarray(rgb[:, :, :3])


def _calibration(frame: CameraFrame) -> tuple[np.ndarray, np.ndarray] | None:
    """(intrinsics, camera-to-base extrinsics) when the frame carries a usable calibration, else None."""
    k, t = getattr(frame, "intrinsics", None), getattr(frame, "extrinsics", None)
    if k is None or t is None:
        return None
    try:
        k, t = np.asarray(k, dtype=float), np.asarray(t, dtype=float)
    except (TypeError, ValueError):
        return None
    if k.shape != (3, 3) or t.shape != (4, 4) or not (np.all(np.isfinite(k)) and np.all(np.isfinite(t))):
        return None
    return k, t


def project_point(frame: CameraFrame, point) -> tuple[float, float, float] | None:
    """A base-frame point (meters) in a frame: (u, v, depth along the optical axis in meters).

    None when the frame has no usable calibration or the point is not in front of the camera.
    The extrinsics are camera to base in the OpenCV convention (see ``CameraFrame``).
    """
    calibration = _calibration(frame)
    if calibration is None:
        return None
    k, t = calibration
    p = np.asarray(point, dtype=float).reshape(-1)
    if p.shape != (3,) or not np.all(np.isfinite(p)):
        return None
    pc = t[:3, :3].T @ (p - t[:3, 3])
    if pc[2] <= 1e-6:
        return None
    uvw = k @ pc
    if abs(uvw[2]) < 1e-12:
        return None
    return float(uvw[0] / uvw[2]), float(uvw[1] / uvw[2]), float(pc[2])


def _cut(rgb: np.ndarray, x0: int, y0: int, side: int) -> np.ndarray:
    """The window [x0, x0 + side) x [y0, y0 + side) of ``rgb``. What falls outside the frame is black."""
    h, w = rgb.shape[:2]
    out = np.zeros((side, side, 3), dtype=np.uint8)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + side, w), min(y0 + side, h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = rgb[sy0:sy1, sx0:sx1]
    return out


def _sized(cut: np.ndarray) -> tuple[np.ndarray, float]:
    """A cut resized up to ``CROP_OUT_PX`` when it is smaller. Returns (pixels, crop pixels per frame pixel)."""
    side = cut.shape[0]
    if side >= CROP_OUT_PX:
        return cut, 1.0
    resized = cv2.resize(cut, (CROP_OUT_PX, CROP_OUT_PX), interpolation=cv2.INTER_CUBIC)
    return np.ascontiguousarray(resized), CROP_OUT_PX / side


def claimed_port(claim: Claim, nominal: NominalGeometry | None) -> int | None:
    """The port named in the claim when it is one of the board's ports, else None."""
    port = claim.predicate.target.get("port")
    if nominal is None or isinstance(port, bool) or not isinstance(port, int):
        return None
    return port if 1 <= port <= len(nominal.ports_board) else None


def crop_to_port(frame: CameraFrame, nominal: NominalGeometry, port: int) -> Crop | None:
    """The square around the taught port of a calibrated frame, with the port at its center.

    The side is ``CROP_MM`` at the port's distance from the camera, at least ``CROP_MIN_PX`` and
    at most the frame's shorter side. None when the frame has no calibration, or when the port
    is behind the camera or outside the frame: that view does not show the port.
    """
    rgb = _rgb(frame)
    calibration = _calibration(frame)
    if rgb is None or calibration is None:
        return None
    hit = project_point(frame, nominal.port_position(port))
    if hit is None:
        return None
    u, v, depth = hit
    h, w = rgb.shape[:2]
    if not (0.0 <= u <= w - 1 and 0.0 <= v <= h - 1):
        return None
    focal = 0.5 * (abs(calibration[0][0, 0]) + abs(calibration[0][1, 1]))
    side = int(round(focal * CROP_MM * 1e-3 / depth))
    side = max(CROP_MIN_PX, min(side, h, w))
    # the window starts on a whole pixel, so the port lands within half a frame pixel of the window's middle
    x0, y0 = int(round(u - (side - 1) / 2)), int(round(v - (side - 1) / 2))
    pixels, scale = _sized(_cut(rgb, x0, y0, side))
    return Crop(rgb=pixels, on_port=True, x0=x0, y0=y0, side=side, scale=scale, port_uv=(u, v))


def center_crop(frame: CameraFrame) -> Crop | None:
    """The central square of a frame: the fallback for a frame that carries no calibration."""
    rgb = _rgb(frame)
    if rgb is None:
        return None
    h, w = rgb.shape[:2]
    side = min(h, w)
    x0, y0 = (w - side) // 2, (h - side) // 2
    pixels, scale = _sized(_cut(rgb, x0, y0, side))
    return Crop(rgb=pixels, on_port=False, x0=x0, y0=y0, side=side, scale=scale)


def crop_view(frame: CameraFrame, nominal: NominalGeometry | None, port: int | None) -> Crop | None:
    """The crop of 1 view for the cropped variants.

    A calibrated frame is cropped around the taught port, and None means the frame does not
    show the port. A frame that cannot be projected into (no calibration, no taught geometry
    or no port in the claim) gives its central square.
    """
    if _calibration(frame) is None or nominal is None or port is None:
        return center_crop(frame)
    return crop_to_port(frame, nominal, port)


# ----------------------------------------------------------------------------- the text
def connector_words(connector: str | None) -> str:
    """The connector type in plain words."""
    if connector in CONNECTOR_WORDS:
        return CONNECTOR_WORDS[connector]
    return "a plug that pushes straight into its socket."


def _target_words(claim: Claim) -> str:
    return ", ".join(f"{name} {value}" for name, value in claim.predicate.target.items())


def _view_sentence(index: int, view: View) -> str:
    camera = _CAMERA_WORDS.get(view.role, _CAMERA_WORDS[ROLE_OTHER])
    when = ("in a closer look taken after that report" if view.when == "relook"
            else "when the robot reported the step done")
    text = f"Image {index} shows the plug being judged, seen by {camera} {when}."
    if view.crop == "port":
        text += " It is cropped to a square around the target port, which is at or near the center of the image."
    elif view.crop == "center":
        text += " It is cropped to the central square of the camera frame."
    return text


def build_prompt(claim: Claim, connector: str | None, variant: str, views: list[View],
                 n_ports: int | None = None) -> str:
    """The text of 1 judge call. ``views`` are the images of the plug being judged, in the order sent.

    The text depends on the claim's predicate, the connector type, the variant and the views.
    Nothing else about the state enters it.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown judge variant {variant!r}. Known: {', '.join(VARIANTS)}")
    predicate = claim.predicate
    reference = variant == "crop_ref"
    lines = [
        "A robot is plugging a connector into a port on a board. It has just reported 1 step of that work as done. "
        "Decide from the camera images whether the condition below holds. Judge only what the images show.",
        "",
        f"Condition ({predicate.name}): {predicate.definition}",
        f"Plug type: {connector_words(connector)}",
        f"Target: {_target_words(claim)}." + (f" The board has {int(n_ports)} ports." if n_ports else ""),
        "",
    ]
    total = len(views) + (1 if reference else 0)
    lines.append(f"There {'is 1 image' if total == 1 else f'are {total} images, in this order'}.")
    index = 1
    if reference:
        lines.append("Image 1 is a reference. It shows another plug of the same type that is correctly seated in its "
                     "port. It is a different plug on a different board. Use it to see what a correct result looks "
                     "like.")
        index = 2
    for offset, view in enumerate(views):
        lines.append(_view_sentence(index + offset, view))
    if any(view.crop == "port" for view in views):
        lines.append("A black area at the edge of an image lies outside the camera frame.")
    lines += [
        "",
        "Answer with 1 JSON object and no other text. It has 3 fields.",
        "holds: true or false. Your best judgment of whether the condition holds for the target.",
        "confidence: a number from 0 to 1 that says how sure you are that your value of holds is right. "
        "0.5 means you cannot tell.",
        "reason: 1 short sentence that says what in the images decided it.",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- the request
@dataclass
class JudgeRequest:
    """Everything 1 call sends. ``problem`` says why nothing can be sent, and is None when it can."""

    model: str = ""
    variant: str = DEFAULT_VARIANT
    text: str = ""
    images: list[np.ndarray] = field(default_factory=list)
    views: list[View] = field(default_factory=list)     # the views of the plug being judged, as sent
    left_out: int = 0                                   # calibrated views that do not show the port
    problem: str | None = None


def _reference(value) -> np.ndarray | None:
    """The reference image as H x W x 3 uint8, or None when it is not one."""
    if value is None:
        return None
    rgb = np.asarray(value)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[0] < 2 or rgb.shape[1] < 2:
        return None
    return np.ascontiguousarray(rgb)


def build_request(claim: Claim, params: dict) -> JudgeRequest:
    """The text and the images of 1 call, from the claim and ``params``. No model is called here."""
    cfg = params.get("judge")
    if not isinstance(cfg, dict):
        return JudgeRequest(problem="params holds no judge settings")
    variant = cfg.get("variant", DEFAULT_VARIANT)
    if variant not in VARIANTS:
        raise ValueError(f"unknown judge variant {variant!r}. Known: {', '.join(VARIANTS)}")
    model = cfg.get("model")
    if not isinstance(model, str) or not model.strip():
        return JudgeRequest(variant=variant, problem="no model was named in params['judge']['model']")
    request = JudgeRequest(model=model, variant=variant)
    frames = params.get("frames")
    if frames is None or not callable(getattr(frames, "get", None)) or not callable(getattr(frames, "keys", None)):
        request.problem = "params holds no frames"
        return request
    nominal = params.get("nominal")
    nominal = nominal if isinstance(nominal, NominalGeometry) else None
    arm = params.get("arm")
    holding = arm if isinstance(arm, str) and arm else (nominal.holding_arm if nominal is not None else "right")
    if nominal is not None and nominal.free_arm != holding:
        free = nominal.free_arm
    else:
        free = "left" if holding != "left" else "right"
    wanted = cfg.get("views")
    port = claimed_port(claim, nominal)
    for view in select_views(frames, holding, free, list(wanted) if wanted is not None else None):
        frame = frames.get(view.key)
        pixels = _rgb(frame)
        if pixels is None:
            continue
        if variant == "full":
            request.images.append(pixels)
            request.views.append(view)
            continue
        crop = crop_view(frame, nominal, port)
        if crop is None:
            request.left_out += 1
            continue
        request.images.append(crop.rgb)
        request.views.append(replace(view, crop="port" if crop.on_port else "center"))
    if not request.images:
        request.problem = ("no view shows the target port" if request.left_out else "no camera frame to send")
        return request
    if variant == "crop_ref":
        reference = _reference(cfg.get("reference"))
        if reference is None:
            request.images, request.views = [], []
            request.problem = "crop_ref needs params['judge']['reference'], an RGB array of a correctly seated plug"
            return request
        request.images.insert(0, reference)
    connector = params.get("connector") or (nominal.connector if nominal is not None else None)
    request.text = build_prompt(claim, connector, variant, request.views,
                                n_ports=len(nominal.ports_board) if nominal is not None else None)
    return request


# ----------------------------------------------------------------------------- the answer
def read_answer(value) -> tuple[float, bool, float] | None:
    """(p_yes, holds, confidence) of an answer that passed its schema, or None when it is not one.

    ``p_yes`` is ``confidence`` when ``holds`` is true, else ``1 - confidence``.
    """
    if not isinstance(value, dict):
        return None
    holds, confidence = value.get("holds"), value.get("confidence")
    if not isinstance(holds, bool) or isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return None
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return None
    return (confidence if holds else 1.0 - confidence), holds, confidence


def _bare(token) -> str:
    return re.sub(r"[^a-z]", "", str(token).lower())


def p_yes_from_logprobs(raw) -> float | None:
    """P(holds is true) from the token log-probabilities of an answer, or None when there are none.

    Reads the chat-completions form ``choices[0].logprobs.content``: a list of tokens, each with
    ``logprob`` and ``top_logprobs``. The token that carries the value of ``holds`` is the one
    that spells true or false after the key. Its alternatives give the 2 probabilities, which
    are renormalized to sum to 1. This parser has only been run on hand-made payloads. No
    provider answer has been through it yet.
    """
    try:
        content = raw["choices"][0]["logprobs"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    if not isinstance(content, list):
        return None
    seen = ""
    for item in content:
        if not isinstance(item, dict):
            return None
        token = item.get("token")
        if '"holds"' in seen and _bare(token) in ("true", "false"):
            mass = {"true": 0.0, "false": 0.0}
            for alt in (item.get("top_logprobs") or [item]):
                word = _bare(alt.get("token")) if isinstance(alt, dict) else ""
                logprob = alt.get("logprob") if isinstance(alt, dict) else None
                if word in mass and isinstance(logprob, (int, float)) and not isinstance(logprob, bool):
                    mass[word] += math.exp(min(float(logprob), 0.0))
            total = mass["true"] + mass["false"]
            return mass["true"] / total if total > 0 else None
        seen += str(token)
    return None


def output_allowance(model_key: str, models=None) -> int:
    """``max_tokens`` for 1 judge call: more for a model that cannot turn reasoning off."""
    table = models if models is not None else default_models()
    entry, _note = resolve_route(table.get(model_key))
    no_off_switch = get_adapter(entry.adapter).reasoning_note("disabled", entry) is not None
    return MAX_TOKENS_REASONING if no_off_switch else MAX_TOKENS


@dataclass
class JudgeResult:
    """What 1 call gave. ``response`` is the router's answer, and None when nothing was sent."""

    request: JudgeRequest
    response: Any = None
    features: dict[str, float] = field(default_factory=dict)
    p_yes: float | None = None
    holds: bool | None = None
    confidence: float | None = None
    reason: str | None = None
    json_error: str | None = None
    seconds: float = 0.0
    usd: float = 0.0


def judge(claim: Claim, params: dict) -> JudgeResult:
    """Build the request, call the model through the router and read the answer.

    Raises what the router raises: ``KeyError`` for an unknown model key,
    ``MissingCredentialError``, ``SpendCapExceeded``, ``ModelCallError`` and the other router
    errors. Nothing is sent when the request has a ``problem``.
    """
    request = build_request(claim, params)
    result = JudgeResult(request=request)
    if request.problem is not None:
        return result
    cfg = params["judge"]
    router = cfg.get("router")
    if router is None:
        router = Router()
    models = router.models
    entry = models.get(request.model)
    max_tokens = cfg.get("max_tokens")
    if max_tokens is None:
        max_tokens = output_allowance(request.model, models)
    kwargs: dict[str, Any] = {}
    if cfg.get("logprobs"):
        kwargs["extra"] = {"body": dict(LOGPROBS_BODY)}
    started = time.perf_counter()
    response = router.chat(request.model, request.text, images=request.images, json_schema=ANSWER_SCHEMA,
                           max_tokens=int(max_tokens), schema_name=SCHEMA_NAME, **kwargs)
    result.seconds = time.perf_counter() - started
    result.response = response
    result.usd = float(response.usd)
    features = {
        "n_images": float(len(request.images)),
        "cached": 1.0 if response.cached else 0.0,
        "placeholder": 1.0 if entry.adapter == "fake" else 0.0,
    }
    if request.variant != "full":
        features["crop_on_port"] = 1.0 if all(view.crop == "port" for view in request.views) else 0.0
    if response.truncated:
        features["truncated"] = 1.0
    answer = read_answer(response.json) if response.json_ok else None
    if answer is None:
        features["json_error"] = 1.0
        result.json_error = response.json_error or "the answer holds no true or false holds with a confidence from 0 to 1"
    else:
        result.p_yes, result.holds, result.confidence = answer
        reason = response.json.get("reason")
        result.reason = reason if isinstance(reason, str) else None
        features.update(p_yes=result.p_yes, holds=1.0 if result.holds else 0.0, confidence=result.confidence)
        if cfg.get("logprobs"):
            from_tokens = p_yes_from_logprobs(response.raw)
            if from_tokens is not None:
                features["p_yes_logprob"] = from_tokens
    result.features = features
    return result


# ----------------------------------------------------------------------------- the source
class JudgeVlm:
    """Asks a vision-language model whether the claimed predicate holds. See the module docstring.

    ``model`` and ``variant`` are what ``estimate_cost`` prices, since it gets no params. With no
    model it prices the leading entry of ``e1_minimum`` in ``config/models.yaml``. ``collect`` never
    uses them: a call names its model in ``params['judge']['model']``.
    """

    source_id = "judge.vlm"
    kind = "judge"
    predicates = {"seated", "latched", "at_port", "released_stable"}

    def __init__(self, model: str | None = None, variant: str = DEFAULT_VARIANT):
        if variant not in VARIANTS:
            raise ValueError(f"unknown judge variant {variant!r}. Known: {', '.join(VARIANTS)}")
        self.model = model
        self.variant = variant

    def estimate_cost(self, claim: Claim) -> Cost:
        """6 s, the middle of the 2 to 10 s in docs/03, and the router's worst case in dollars for 1 call.

        ``usd`` is 0.0 when the model table cannot price the call. A cost cannot say unknown, so
        read 0.0 from this source as not priced.
        """
        usd = 0.0
        try:
            models = default_models()
            model = self.model or (models.e1_minimum[0] if models.e1_minimum else None)
            if model:
                views = typical_views(self.variant)
                text = build_prompt(claim, max(CONNECTOR_WORDS, key=lambda name: len(CONNECTOR_WORDS[name])),
                                    self.variant, views, n_ports=8)
                images = len(views) + (1 if self.variant == "crop_ref" else 0)
                usd = float(estimate_usd(model, text, image_count=images, json_schema=ANSWER_SCHEMA,
                                         max_tokens=output_allowance(model, models), models=models))
        except Exception:  # noqa: BLE001 - an estimate never fails a caller. An unpriced call reads as 0.0
            usd = 0.0
        if not math.isfinite(usd) or usd < 0:
            usd = 0.0
        return Cost(seconds=TYPICAL_SECONDS, usd=usd, moved_robot=False)

    def ask(self, claim: Claim, params: dict) -> JudgeResult:
        """The whole answer of 1 call: what ``collect`` turns into an evidence. Evaluators use it for their tables."""
        return judge(claim, params)

    def collect(self, station: Station, claim: Claim, params: dict) -> Evidence:
        """Ask the model and return the evidence. The station is not touched."""
        result = judge(claim, params)
        score = result.p_yes if claim.predicate.name in self.predicates else None
        return make_evidence(self, claim, result.features, score=score, seconds=result.seconds, usd=result.usd)


JUDGE_VLM = register(JudgeVlm())
