"""judge.vlm and the runner that puts stored zoo frames in front of it.

Everything here is offline: the model is the fake one, the HTTP transport fails the test if it
is called, sockets refuse to connect, no key is read from the environment or .env, and
AVAIL_DATA points at a temp folder. The fake model's answers are placeholders, so no test here
says anything about how well a judge sees. The tests pin the plumbing: what is sent, what the
answer becomes, what it costs and what is refused.

The top part needs no simulator: frames, cameras and zoo tables are built by hand. The last
part runs on real injected states of the plug station (dev seeds 3000 and 3017).
"""

from __future__ import annotations

import ast
import json
import math
import re
import socket
import sys

import httpx
import numpy as np
import pandas as pd
import pytest

from avail import config
from avail.evidence import judge
from avail.evidence.base import REGISTRY, EvidenceSource, FrameStore
from avail.paths import REPO_ROOT, aspire_station_xml, zoo_dir
from avail.router import (
    DiskCache,
    HttpClient,
    MissingCredentialError,
    ModelCallError,
    ModelsConfig,
    Router,
    SpendCapExceeded,
    SpendCaps,
    canonical_json,
    clear_fake_script,
    fake_calls,
    fake_script,
    load_models,
    normalize_image,
)
from avail.router import http as router_http
from avail.schemas import EVIDENCE_KINDS, PREDICATE_DEFINITIONS, Claim, ClaimContext, Cost, Evidence, Predicate
from avail.seeds import TestSeedError, test_seed_reads
from avail.station.base import CameraFrame, NominalGeometry
from avail.walls import scan_source
from avail.zoo import judges, store

SOURCE_FILE = "avail/evidence/judge.py"
KEY_VARS = tuple(sorted({k for k in config.dotenv_values(REPO_ROOT / ".env.example") if k.endswith("_KEY")}
                        | {"OPENROUTER_API_KEY", "ELORIAN_API_KEY", "TYPESAFE_JEV_API_KEY", "TYPESAFE_API_KEY",
                           "LIQUID_API_KEY"}))
PLAIN_KEY = "av-test-" + "Q7" * 20          # built at run time. It is not a real key
W, H = 640, 480
WRIST, TOP, OTHER_WRIST = "right_camera_d405", "top_camera_d405", "left_camera_d405"
TWIN_SEED = 3017                            # a dev seed whose seated state holds the plug and claims latched
CLASS_NAME = re.compile(r"\b(P0|N[1-9])\b")

needs_station = pytest.mark.skipif(not aspire_station_xml().is_file(), reason="third_party/ASPIRE is not cloned")


def _refuse_network(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"a test tried the network: {request.method} {request.url}")


def _refuse_socket(self, address, *args, **kwargs):
    raise AssertionError(f"a test tried to open a socket to {address!r}")


def refusing_client() -> HttpClient:
    """An HTTP client whose transport fails the test if it is ever called."""
    return HttpClient(transport=httpx.MockTransport(_refuse_network), sleep=lambda s: None)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """No real data root, no keys from the environment or .env, no network, no leftover fake script."""
    monkeypatch.setenv("AVAIL_DATA", str(tmp_path / "data"))
    for name in KEY_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "_DOTENV", {})
    monkeypatch.setattr(router_http, "_DEFAULT_CLIENT", refusing_client())
    monkeypatch.setattr(socket.socket, "connect", _refuse_socket)
    clear_fake_script()
    yield
    clear_fake_script()


# ----------------------------------------------------------------------------- hand-made claims, cameras and frames
def make_claim(predicate: str = "latched", port: int = 7, claim_id: str = "claim_secret_41") -> Claim:
    return Claim(claim_id=claim_id, run_id="run_hidden", cycle_id="cycle_hidden", step_id="insert",
                 predicate=Predicate(name=predicate, target={"plug": "p1", "port": port}),
                 raised_by="skill", context=ClaimContext(task="plug_cycle", stage="insert", attempt=0))


def hand_nominal(connector: str = "latch") -> NominalGeometry:
    ports = tuple((x, y) for x in (-0.02, 0.02) for y in (-0.06, -0.02, 0.02, 0.06))
    return NominalGeometry(connector=connector, full_depth_m=0.012, grasp_height_m=0.016, board_xy=(0.5, -0.12),
                           board_yaw=0.05, face_z=0.766, ports_board=ports, pitch_m=0.04)


def noise(tag: int) -> np.ndarray:
    return np.random.default_rng(tag).integers(0, 256, size=(H, W, 3), dtype=np.uint8)


def look_down_frame(name: str, rgb: np.ndarray, point, uv, focal: float, depth: float) -> CameraFrame:
    """A frame from a camera ``depth`` meters above ``point``, looking straight down, with ``point`` at pixel ``uv``."""
    cx, cy = W / 2, H / 2
    rot = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])     # camera x, y, z axes in the base frame
    in_camera = np.array([(uv[0] - cx) * depth / focal, (uv[1] - cy) * depth / focal, depth])
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = np.asarray(point, dtype=float) - rot @ in_camera
    k = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]])
    return CameraFrame(name=name, rgb=rgb, intrinsics=k, extrinsics=t, ts=0.0)


VIEWS = {WRIST: ((316.0, 402.0), 432.9, 0.14), TOP: ((369.0, 283.0), 432.2, 1.0), OTHER_WRIST: ((430.0, 370.0), 432.9, 0.5)}


def hand_frames(nominal: NominalGeometry, port: int, *, relook: bool = False, calibrated: bool = True, tag: int = 0) -> FrameStore:
    frames = FrameStore()
    point = nominal.port_position(port)
    for i, (camera, (uv, focal, depth)) in enumerate(VIEWS.items()):
        frame = look_down_frame(camera, noise(100 * tag + i), point, uv, focal, depth)
        if not calibrated:
            frame = CameraFrame(name=camera, rgb=frame.rgb, intrinsics=None, extrinsics=None, ts=0.0)
        frames.put(f"claim/{camera}", frame)
    if relook:
        frames.put(f"relook/{OTHER_WRIST}", look_down_frame(OTHER_WRIST, noise(100 * tag + 9), point, (320.0, 240.0), 432.9, 0.08))
    return frames


def hand_params(variant: str = "crop", *, port: int = 7, model: str | None = "fake", connector: str = "latch",
                reference: np.ndarray | None = None, **frame_kw) -> tuple[Claim, dict]:
    nominal = hand_nominal(connector)
    settings: dict = {"variant": variant}
    if model is not None:
        settings["model"] = model
    if reference is not None:
        settings["reference"] = reference
    params = {"frames": hand_frames(nominal, port, **frame_kw), "nominal": nominal, "connector": connector,
              "arm": "right", "judge": settings}
    return make_claim(port=port), params


def reference_picture() -> np.ndarray:
    return np.random.default_rng(999).integers(0, 256, size=(judge.CROP_OUT_PX, judge.CROP_OUT_PX, 3), dtype=np.uint8)


def answer(holds: bool = True, confidence: float = 0.9) -> dict:
    return {"holds": holds, "confidence": confidence, "reason": "scripted"}


def in_the_middle(crop: judge.Crop, where: float) -> bool:
    """The window starts on a whole pixel, so the port lands within half a frame pixel of the crop's middle."""
    middle = crop.rgb.shape[0] / 2 - 0.5
    return abs(where - middle) <= crop.scale / 2 + 1e-6


# ============================================================================ the source: no simulator needed
def test_the_source_is_registered_as_a_judge_for_the_4_plug_predicates():
    source = judge.JUDGE_VLM
    assert REGISTRY["judge.vlm"] is source and isinstance(source, EvidenceSource)
    assert (source.source_id, source.kind) == ("judge.vlm", "judge") and source.kind in EVIDENCE_KINDS
    assert source.predicates == {"seated", "latched", "at_port", "released_stable"}
    assert judge.VARIANTS == ("full", "crop", "crop_ref")
    assert judge.ANSWER_SCHEMA["required"] == ["holds", "confidence", "reason"]
    assert judge.ANSWER_SCHEMA["properties"]["confidence"] == {"type": "number", "minimum": 0, "maximum": 1}


def test_source_file_passes_the_truth_wall():
    source = (REPO_ROOT / SOURCE_FILE).read_text(encoding="utf-8")
    assert scan_source(source, SOURCE_FILE) == []
    # the scanner does bite on this file: the same source with a truth import fails
    peeking = source + "\nfrom ..station.mj_plug.truth import truth\n"
    assert {v.rule for v in scan_source(peeking, SOURCE_FILE)} >= {"truth_import"}


def test_source_file_imports_only_what_a_source_may():
    """The standard library, numpy, cv2, PIL and 5 corners of avail. Nothing from the simulator or the zoo."""
    allowed_avail = ("avail.schemas", "avail.station.base", "avail.evidence.base", "avail.config", "avail.router")
    allowed_other = set(sys.stdlib_module_names) | {"numpy", "cv2", "PIL"}
    package = SOURCE_FILE[:-len(".py")].split("/")[:-1]
    imported = []
    for node in ast.walk(ast.parse((REPO_ROOT / SOURCE_FILE).read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = package[:len(package) - (node.level - 1)] if node.level else []
            imported.append(".".join([*base, *([node.module] if node.module else [])]))
    assert "avail.router" in imported, "the scan found no router import, so it read the wrong file"
    for name in imported:
        if name.split(".")[0] == "avail":
            assert any(name == ok or name.startswith(ok + ".") for ok in allowed_avail), name
        else:
            assert name.split(".")[0] in allowed_other, name


@pytest.mark.parametrize("variant", judge.VARIANTS)
def test_the_prompt_carries_the_definition_the_connector_and_the_port(variant):
    claim, params = hand_params(variant, port=7, reference=reference_picture())
    text = judge.build_request(claim, params).text
    assert PREDICATE_DEFINITIONS["latched"] in text and "Condition (latched)" in text
    assert "a latching plug such as RJ45" in text
    assert "Target: plug p1, port 7." in text and "The board has 8 ports." in text
    # the required answer, described in plain words and never as JSON inside the text
    for name in ("holds:", "confidence:", "reason:"):
        assert name in text
    assert "{" not in text and "}" not in text
    # nothing of the claim but its predicate: no ids, no context
    for hidden in ("claim_secret_41", "run_hidden", "cycle_hidden", "plug_cycle"):
        assert hidden not in text
    # plain direct prose
    assert ";" not in text and chr(0x2014) not in text
    # the same claim about a friction-fit plug names that connector in its place
    fric_claim, fric_params = hand_params(variant, port=7, connector="fric", reference=reference_picture())
    fric = judge.build_request(fric_claim, fric_params).text
    assert "a friction-fit plug such as USB-A" in fric and "RJ45" not in fric
    assert judge.build_request(make_claim(port=3), params).text == text.replace("port 7", "port 3")


def test_the_text_says_what_each_image_is_and_which_one_is_the_reference():
    claim, params = hand_params("full", relook=True)
    full = judge.build_request(claim, params).text
    assert "There are 3 images, in this order." in full
    wrist = "Image 1 shows the plug being judged, seen by the camera on the wrist of the arm that holds the plug"
    assert wrist in full and "Image 2 shows the plug being judged, seen by the camera on the wrist of the other arm in a closer look" in full
    assert "Image 3 shows the plug being judged, seen by the overhead camera" in full
    assert "cropped" not in full and "reference" not in full          # whole frames, and no word on where the port is
    claim, params = hand_params("crop")
    crop = judge.build_request(claim, params).text
    assert crop.count("cropped to a square around the target port, which is at or near the center of the image") == 2
    assert "reference" not in crop
    claim, params = hand_params("crop_ref", reference=reference_picture())
    ref = judge.build_request(claim, params).text
    assert "There are 3 images, in this order." in ref
    assert "Image 1 is a reference." in ref and "correctly seated" in ref
    assert "Image 2 shows the plug being judged, seen by the camera on the wrist of the arm that holds the plug" in ref
    assert "Image 3 shows the plug being judged, seen by the overhead camera" in ref
    one = judge.build_prompt(claim, "latch", "full", judge.typical_views("full", 1))
    assert "There is 1 image." in one


def test_the_text_does_not_change_with_the_pixels_or_with_where_the_camera_stopped():
    """2 states with the same claim: other pixels, and a wrist camera that stopped 4 mm higher. The text is the same."""
    texts = {}
    for variant in judge.VARIANTS:
        claim, params = hand_params(variant, tag=1, reference=reference_picture())
        other_claim, other = hand_params(variant, tag=2, reference=reference_picture())
        nominal = other["nominal"]
        moved = look_down_frame(WRIST, noise(77), nominal.port_position(7), (316.0, 390.0), 432.9, 0.144)
        other["frames"].put(f"claim/{WRIST}", moved)
        a, b = judge.build_request(claim, params), judge.build_request(other_claim, other)
        assert a.text == b.text
        assert not np.array_equal(a.images[-2], b.images[-2])           # the states do differ, in the pixels
        texts[variant] = a.text
    assert len(set(texts.values())) == 3


@pytest.mark.parametrize("holds, confidence, p_yes", [(True, 0.8, 0.8), (False, 0.8, 0.2), (True, 0.5, 0.5),
                                                       (False, 1.0, 0.0), (True, 1, 1.0), (False, 0.35, 0.65)])
def test_p_yes_is_confidence_when_holds_is_true_else_1_minus_confidence(holds, confidence, p_yes):
    claim, params = hand_params("crop")
    with fake_script(lambda req: answer(holds, confidence)):
        evidence = judge.JUDGE_VLM.collect(None, claim, params)
    assert isinstance(evidence, Evidence) and evidence.claim_id == claim.claim_id
    assert (evidence.source_id, evidence.kind, evidence.side_effect) == ("judge.vlm", "judge", "none")
    assert evidence.features["p_yes"] == pytest.approx(p_yes) and evidence.score == pytest.approx(p_yes)
    assert evidence.features["holds"] == (1.0 if holds else 0.0)
    assert evidence.features["confidence"] == pytest.approx(confidence)
    assert evidence.features["n_images"] == 2.0 and evidence.features["cached"] == 0.0
    assert evidence.features["placeholder"] == 1.0                    # the fake model: never a measurement
    assert evidence.features["crop_on_port"] == 1.0
    assert evidence.cost.usd == 0.0 and evidence.cost.seconds > 0 and evidence.cost.moved_robot is False
    assert judge.read_answer(answer(holds, confidence)) == (pytest.approx(p_yes), holds, pytest.approx(confidence))


def test_a_predicate_the_source_does_not_bear_on_gets_features_and_no_score():
    _claim, params = hand_params("full")
    held = make_claim(predicate="held")
    with fake_script(lambda req: answer(True, 0.9)):
        evidence = judge.JUDGE_VLM.collect(None, held, params)
    assert evidence.features["p_yes"] == pytest.approx(0.9) and evidence.score is None


@pytest.mark.parametrize("reply", [
    "the plug looks fine to me",                                       # not JSON
    {"holds": True},                                                   # fields missing
    {"holds": "yes", "confidence": 0.9, "reason": "text where a boolean belongs"},
    {"holds": True, "confidence": 1.7, "reason": "confidence out of range"},
    {"holds": True, "confidence": 0.9, "reason": "one field too many", "verdict": "accept"},
    '{"holds": true, "confid',                                         # cut off
])
def test_an_answer_that_fails_its_schema_gives_no_p_yes(reply):
    claim, params = hand_params("crop")
    with fake_script([reply]):
        result = judge.JUDGE_VLM.ask(claim, params)
        evidence = judge.JUDGE_VLM.collect(None, claim, params)        # the same request, from the cache
    for features in (result.features, evidence.features):
        assert features["json_error"] == 1.0 and features["n_images"] == 2.0
        assert not {"p_yes", "holds", "confidence"} & set(features)
    assert evidence.score is None and evidence.features["cached"] == 1.0
    assert result.p_yes is None and result.holds is None and result.json_error
    for not_an_answer in (None, [], {"holds": 1, "confidence": 0.9}, {"holds": True, "confidence": True},
                          {"holds": True, "confidence": float("nan")}, {"holds": True, "confidence": -0.1}):
        assert judge.read_answer(not_an_answer) is None


def test_the_3_variants_send_the_expected_images_in_order():
    seen = []

    def script(req):
        seen.append(req)
        return answer()

    reference = reference_picture()
    sent = {}
    with fake_script(script):
        for variant in judge.VARIANTS:
            claim, params = hand_params(variant, relook=True, reference=reference)
            request = judge.build_request(claim, params)
            evidence = judge.JUDGE_VLM.collect(None, claim, params)
            sent[variant] = request.images
            assert evidence.features["n_images"] == len(request.images) == seen[-1].image_count
            # the model got these images, in this order
            assert list(seen[-1].images) == [normalize_image(image).sha256 for image in request.images]
            assert seen[-1].text == request.text and seen[-1].json_schema == judge.ANSWER_SCHEMA
            assert seen[-1].reasoning == {"enabled": False} and seen[-1].role == "judge"
    frames = params["frames"]
    # full: the holding arm's wrist view, then the closer look, then the overhead view, whole
    assert len(sent["full"]) == 3 and all(image.shape == (H, W, 3) for image in sent["full"])
    for image, key in zip(sent["full"], (f"claim/{WRIST}", f"relook/{OTHER_WRIST}", f"claim/{TOP}")):
        assert np.array_equal(image, frames.get(key).rgb)
    # crop: the same 3 views, each a square that is smaller than a whole frame. The closer look is 8 cm from the
    # port, so its 80 mm square is 433 pixels of the frame and is sent as it is. The other 2 are resized up to 384
    assert [image.shape for image in sent["crop"]] == [(384, 384, 3), (433, 433, 3), (384, 384, 3)]
    for image in sent["crop"]:
        assert image.dtype == np.uint8 and image.shape[0] <= H and image.shape[1] <= W
        assert image.size < sent["full"][0].size
    # crop_ref: the reference, then the same crops
    assert len(sent["crop_ref"]) == 4 and np.array_equal(sent["crop_ref"][0], reference)
    for with_ref, plain in zip(sent["crop_ref"][1:], sent["crop"]):
        assert np.array_equal(with_ref, plain)
    # without a closer look there are 2 views, and the caller can name its own
    claim, params = hand_params("full")
    assert [v.key for v in judge.build_request(claim, params).views] == [f"claim/{WRIST}", f"claim/{TOP}"]
    params["judge"]["views"] = [f"claim/{TOP}", "claim/no_such_camera", f"claim/{OTHER_WRIST}"]
    assert [v.key for v in judge.build_request(claim, params).views] == [f"claim/{TOP}", f"claim/{OTHER_WRIST}"]


def test_the_crop_is_centered_on_the_projected_port():
    nominal = hand_nominal()
    port = 6
    point = nominal.port_position(port)
    rgb = noise(5)
    rgb[396:409, 310:323] = (255, 0, 0)                                # a red patch on the pixel the port projects to
    frame = look_down_frame(WRIST, rgb, point, (316.0, 402.0), 432.9, 0.14)
    u, v, depth = judge.project_point(frame, point)
    assert (u, v, depth) == (pytest.approx(316.0), pytest.approx(402.0), pytest.approx(0.14))
    crop = judge.crop_to_port(frame, nominal, port)
    side = round(432.9 * 0.080 / 0.14)                                 # 80 mm at the port's distance: 247 pixels
    assert crop.on_port and crop.side == side == 247 and crop.rgb.shape == (384, 384, 3)
    assert crop.scale == pytest.approx(384 / 247) and crop.port_uv == (pytest.approx(316.0), pytest.approx(402.0))
    # the port's pixel is inside the window and lands on the middle of the crop
    assert crop.x0 <= u < crop.x0 + crop.side and crop.y0 <= v < crop.y0 + crop.side
    cu, cv = crop.locate(u, v)
    assert in_the_middle(crop, cu) and in_the_middle(crop, cv)
    assert (crop.x0, crop.y0) == (193, 279)                            # 316 - 123 and 402 - 123: 123 pixels each side
    red, green, blue = crop.rgb[int(round(cv)), int(round(cu))]
    assert red > 240 and green < 15 and blue < 15
    # the window runs past the bottom of the frame, and that part is black
    assert crop.y0 + crop.side > H
    rows_inside = int(math.floor((H - crop.y0) * crop.scale))
    assert not crop.rgb[rows_inside + 4:].any() and crop.rgb[:rows_inside - 4].any()

    # close enough that the square is 384 pixels or more: no resize, and the pixels are the frame's own
    near = look_down_frame(WRIST, rgb, point, (300.2, 200.3), 1000.0, 0.2)
    exact = judge.crop_to_port(near, nominal, port)
    assert exact.side == 400 and exact.scale == 1.0 and exact.rgb.shape == (400, 400, 3)
    assert (exact.x0, exact.y0) == (101, 1)
    assert np.array_equal(exact.rgb, rgb[1:401, 101:501])
    assert in_the_middle(exact, exact.locate(300.2, 200.3)[0]) and in_the_middle(exact, exact.locate(300.2, 200.3)[1])
    # far away, the square is never under 96 pixels of the frame
    far = judge.crop_to_port(look_down_frame(TOP, rgb, point, (369.0, 283.0), 432.2, 1.0), nominal, port)
    assert far.side == judge.CROP_MIN_PX == 96 and far.rgb.shape == (384, 384, 3) and far.scale == 4.0
    # a view that does not show the port gives no crop: outside the frame, or behind the camera
    assert judge.crop_to_port(look_down_frame(WRIST, rgb, point, (700.0, 240.0), 432.9, 0.14), nominal, port) is None
    assert judge.crop_to_port(look_down_frame(WRIST, rgb, point, (316.0, -5.0), 432.9, 0.14), nominal, port) is None
    behind = look_down_frame(WRIST, rgb, point, (316.0, 402.0), 432.9, 0.14)
    behind.extrinsics[:3, 3] = point - np.array([0.0, 0.0, 0.2])       # the camera is now under the board, looking down
    assert judge.project_point(behind, point) is None and judge.crop_to_port(behind, nominal, port) is None
    # a frame without calibration: the central square, marked as such
    bare = CameraFrame(name=WRIST, rgb=rgb, intrinsics=None, extrinsics=None, ts=0.0)
    assert judge.crop_to_port(bare, nominal, port) is None
    center = judge.crop_view(bare, nominal, port)
    assert not center.on_port and (center.x0, center.y0, center.side) == (80, 0, 480)
    assert np.array_equal(center.rgb, rgb[:, 80:560])


def test_a_calibrated_view_that_does_not_show_the_port_is_left_out_of_the_crops():
    claim, params = hand_params("crop", port=7)
    nominal = params["nominal"]
    away = look_down_frame(WRIST, noise(3), nominal.port_position(7), (900.0, 402.0), 432.9, 0.14)
    params["frames"].put(f"claim/{WRIST}", away)
    request = judge.build_request(claim, params)
    assert [v.key for v in request.views] == [f"claim/{TOP}"] and request.left_out == 1 and len(request.images) == 1
    assert "There is 1 image." in request.text
    params["judge"]["variant"] = "full"                                # the whole frames still go out
    assert len(judge.build_request(claim, params).images) == 2
    # no view shows the port: nothing to send, and no call
    params["judge"]["variant"] = "crop"
    params["frames"].put(f"claim/{TOP}", look_down_frame(TOP, noise(4), nominal.port_position(7), (-50.0, 283.0), 432.2, 1.0))
    nothing = judge.build_request(claim, params)
    assert nothing.problem == "no view shows the target port" and nothing.images == []
    evidence = judge.JUDGE_VLM.collect(None, claim, params)
    assert evidence.features == {} and evidence.score is None and fake_calls() == 0


def test_frames_without_calibration_get_the_central_square_and_the_text_says_so():
    claim, params = hand_params("crop", calibrated=False)
    request = judge.build_request(claim, params)
    assert [v.crop for v in request.views] == ["center", "center"]
    assert all(image.shape == (480, 480, 3) for image in request.images)
    assert request.text.count("cropped to the central square of the camera frame") == 2
    assert "center of the image" not in request.text and "black area" not in request.text
    with fake_script(lambda req: answer()):
        evidence = judge.JUDGE_VLM.collect(None, claim, params)
    assert evidence.features["crop_on_port"] == 0.0 and evidence.features["p_yes"] == pytest.approx(0.9)


def test_a_second_identical_call_is_served_from_the_cache_for_0():
    claim, params = hand_params("crop_ref", reference=reference_picture())
    first = judge.JUDGE_VLM.collect(None, claim, params)
    second = judge.JUDGE_VLM.collect(None, claim, params)
    assert first.features["cached"] == 0.0 and second.features["cached"] == 1.0
    assert first.cost.usd == 0.0 and second.cost.usd == 0.0 and fake_calls() == 1     # the model was asked once
    assert {k: v for k, v in second.features.items() if k != "cached"} == {k: v for k, v in first.features.items() if k != "cached"}
    assert first.evidence_id != second.evidence_id
    # another variant, another port or 1 changed pixel is another request
    other_claim, other = hand_params("crop")
    assert judge.JUDGE_VLM.collect(None, other_claim, other).features["cached"] == 0.0
    params["frames"].get(f"claim/{WRIST}").rgb[402, 316] = (1, 2, 3)
    assert judge.JUDGE_VLM.collect(None, claim, params).features["cached"] == 0.0
    assert fake_calls() == 3


def priced_models() -> ModelsConfig:
    """A model table with a fake model that has prices, so a call costs money without any network."""
    base = dict(display="test entry", adapter="fake", vision=True, roles=["judge", "test"], verified=True,
                source="test", accessed="2026-11-11")
    return ModelsConfig.model_validate({
        "version": "test", "currency": "USD", "unit": "usd_per_million_tokens",
        "models": {"priced_fake": dict(base, id="fake-priced", price_in=10.0, price_out=50.0),
                   "fake": dict(base, id="fake", price_in=0.0, price_out=0.0)}})


def test_the_cost_is_what_the_router_charged_and_0_for_a_cached_answer(tmp_path):
    router = Router(session="T", caps=SpendCaps(total_usd=5.0, sessions={"T": 5.0}), models=priced_models(),
                    cache=DiskCache(tmp_path / "cache"), http=refusing_client())
    claim, params = hand_params("crop", model="priced_fake")
    params["judge"]["router"] = router
    reply = answer(False, 0.75)
    with fake_script(lambda req: reply):
        request = judge.build_request(claim, params)
        first = judge.JUDGE_VLM.collect(None, claim, params)
        again = judge.JUDGE_VLM.collect(None, claim, params)
    # the fake counts ceil(chars / 4) tokens of text and 765 per image, priced here at 10 and 50 USD per million
    tokens_in = math.ceil(len(request.text) / 4) + 765 * 2
    tokens_out = math.ceil(len(json.dumps(reply)) / 4)
    charged = (tokens_in * 10.0 + tokens_out * 50.0) / 1e6
    assert first.cost.usd == pytest.approx(charged) and charged > 0.015
    assert router.spent() == pytest.approx(charged)                   # the session was charged that, once
    assert again.cost.usd == 0.0 and again.features["cached"] == 1.0 and router.spent() == pytest.approx(charged)
    assert first.features["p_yes"] == pytest.approx(0.25) and first.features["placeholder"] == 1.0


def test_a_paid_model_without_a_key_is_refused_before_any_network_io(monkeypatch, tmp_path):
    """The real model table. The transport and the sockets fail the test if anything is sent."""
    claim, params = hand_params("crop", model="opus")
    params["judge"]["router"] = Router(session="S3", http=refusing_client())
    with pytest.raises(MissingCredentialError, match="OPENROUTER_API_KEY"):
        judge.JUDGE_VLM.collect(None, claim, params)
    del params["judge"]["router"]                                      # the default router has no spend session
    with pytest.raises(MissingCredentialError, match="OPENROUTER_API_KEY"):
        judge.JUDGE_VLM.collect(None, claim, params)
    # with a key and no spend session the router still refuses, before it sends anything
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    with pytest.raises(SpendCapExceeded) as refused:
        judge.JUDGE_VLM.collect(None, claim, params)
    assert refused.value.scope == "no_session" and PLAIN_KEY not in str(refused.value)
    assert not (tmp_path / "data" / "spend_ledger.jsonl").exists() and fake_calls() == 0
    with pytest.raises(KeyError, match="unknown model key"):           # a typo in the model key is loud too
        judge.JUDGE_VLM.collect(None, *hand_params("crop", model="opuss"))


def test_missing_inputs_give_an_evidence_with_no_features_and_no_call():
    claim, params = hand_params("crop")
    cases = {
        "no judge settings": {k: v for k, v in params.items() if k != "judge"},
        "no model": {**params, "judge": {"variant": "crop"}},
        "an empty model name": {**params, "judge": {"model": "  ", "variant": "crop"}},
        "no frames": {**{k: v for k, v in params.items() if k != "frames"}},
        "an empty frame store": {**params, "frames": FrameStore()},
        "crop_ref without a reference": {**params, "judge": {"model": "fake", "variant": "crop_ref"}},
        "a reference that is not an image": {**params, "judge": {"model": "fake", "variant": "crop_ref", "reference": "ref.png"}},
    }
    for what, bad in cases.items():
        assert judge.build_request(claim, bad).problem, what
        evidence = judge.JUDGE_VLM.collect(None, claim, bad)
        assert evidence.features == {} and evidence.score is None and evidence.cost.usd == 0.0, what
    assert fake_calls() == 0
    with pytest.raises(ValueError, match="unknown judge variant"):
        judge.build_request(claim, {**params, "judge": {"model": "fake", "variant": "zoomed"}})


def test_estimate_cost_is_the_routers_worst_case_for_the_default_model_and_variant():
    claim = make_claim()
    table = load_models()
    default = table.e1_minimum[0]
    entry = table.models[default]
    cost = judge.JUDGE_VLM.estimate_cost(claim)
    assert isinstance(cost, Cost) and cost.seconds == 6.0 and cost.moved_robot is False
    longest = max(judge.CONNECTOR_WORDS, key=lambda name: len(judge.CONNECTOR_WORDS[name]))
    text = judge.build_prompt(claim, longest, "crop", judge.typical_views("crop"), n_ports=8)
    tokens_in = math.ceil((len(text) + len(canonical_json(judge.ANSWER_SCHEMA))) / 3.5) + 2 * (entry.image_tokens or 1600)
    allowance = judge.output_allowance(default, table)
    assert allowance == (judge.MAX_TOKENS_REASONING if entry.reasoning_mandatory else judge.MAX_TOKENS)
    assert cost.usd == pytest.approx((tokens_in * entry.price_in + allowance * entry.price_out) / 1e6) and cost.usd > 0
    # the reference is 1 more image, and the fake model is free
    with_ref = judge.JudgeVlm(model=default, variant="crop_ref").estimate_cost(claim)
    assert with_ref.usd > cost.usd + 0.99 * (entry.image_tokens or 1600) * entry.price_in / 1e6
    assert judge.JudgeVlm(model="fake").estimate_cost(claim) == Cost(seconds=6.0, usd=0.0, moved_robot=False)
    assert judge.JudgeVlm(model="no_such_model").estimate_cost(claim).usd == 0.0     # not priced, and no crash
    assert judge.output_allowance("luna", table) == judge.MAX_TOKENS == 300          # luna can turn reasoning off


def test_token_log_probabilities_give_a_second_p_yes_when_asked_for():
    def token(text, logprob, alternatives=()):
        return {"token": text, "logprob": logprob,
                "top_logprobs": [{"token": t, "logprob": lp} for t, lp in ((text, logprob), *alternatives)]}

    def raw(value_token):
        content = [token("{", -0.01), token('"holds', -0.01), token('":', -0.01), value_token, token(",", -0.01),
                   token(' "confidence', -0.02), token('":', -0.01), token(" 0", -0.3), token(".9", -0.5), token("}", -0.01)]
        return {"choices": [{"logprobs": {"content": content}}]}

    yes = raw(token(" true", math.log(0.9), [(" false", math.log(0.1))]))
    assert judge.p_yes_from_logprobs(yes) == pytest.approx(0.9)
    # the answer was false: 0.7 on false, 0.2 on true, the rest on other tokens. Renormalized over the 2
    no = raw(token(" false", math.log(0.7), [(" true", math.log(0.2)), (" maybe", math.log(0.1))]))
    assert judge.p_yes_from_logprobs(no) == pytest.approx(0.2 / 0.9)
    alone = raw({"token": "true", "logprob": math.log(0.6)})           # no alternatives listed
    assert judge.p_yes_from_logprobs(alone) == pytest.approx(1.0)
    for nothing in ({}, {"choices": []}, {"choices": [{"logprobs": None}]}, {"choices": [{"logprobs": {"content": []}}]},
                    raw(token(" 1", -0.1))):
        assert judge.p_yes_from_logprobs(nothing) is None
    # asking for them adds the 2 body fields to the request, and not asking sends none
    seen = []
    claim, params = hand_params("full")
    with fake_script(lambda req: seen.append(req.extra) or answer()):
        plain = judge.JUDGE_VLM.collect(None, claim, params)
        params["judge"]["logprobs"] = True
        asked = judge.JUDGE_VLM.collect(None, claim, params)
    assert seen == [{}, {"body": {"logprobs": True, "top_logprobs": 5}}]
    assert "p_yes_logprob" not in plain.features and "p_yes_logprob" not in asked.features   # the fake returns none


# ============================================================================ the runner: hand-made zoo tables
def make_zoo(root, seeds, classes, *, calibrated: bool = True, flip_truth: bool = False) -> None:
    """A small states table with its frames zip, in the layout ``avail zoo`` writes. Dev seeds, nothing simulated."""
    nominal = hand_nominal()
    frames = store.FrameZip(store.part_path("latch", "dev", "frames", seeds, root))
    rows = []
    for seed, cls in zip(seeds, classes):
        port = 1 + seed % 8
        point = nominal.port_position(port)
        positive = (cls == "P0") != flip_truth
        row = {"seed": seed, "connector": "latch", "split": "dev", "cls": cls, "landed_cls": cls,
               "claim_predicate": "latched", "target_port": port, "label": positive,
               "true_shortfall_mm": 0.03 if positive else 1.57, "true_seated": positive, "true_latched": positive,
               "true_nearest_port": port, "skill_flag": True}
        for i, (camera, (uv, focal, depth)) in enumerate(VIEWS.items()):
            frame = look_down_frame(camera, noise(seed * 10 + i), point, uv, focal, depth)
            row[f"frame_{camera}"] = frames.add(f"{seed}_{camera}.jpg", frame.rgb)
            if calibrated:
                row[f"calib_{camera}"] = json.dumps({"K": [round(float(v), 6) for v in frame.intrinsics.ravel()],
                                                     "T": [round(float(v), 9) for v in frame.extrinsics.ravel()]})
        if calibrated:
            row["nominal_port_x"], row["nominal_port_y"], row["nominal_port_z"] = (float(v) for v in point)
            row["nominal_board_yaw"] = float(nominal.board_yaw)
        row["frames_zip"] = frames.path.name
        rows.append(row)
    frames.close()
    store.write_table_atomic(rows, store.part_path("latch", "dev", "states", seeds, root))
    store.merge("latch", "dev", "states", root)


ZOO_SEEDS = [3100, 3101, 3102, 3103, 3104, 3105]
ZOO_CLASSES = ["P0", "N2", "N6", "P0", "N5", "N9"]
REQUIRED_COLUMNS = ["seed", "model", "variant", "p_yes", "holds", "confidence", "usd", "latency_s", "cached",
                    "placeholder", "json_ok"]


def test_subsample_is_stratified_by_class_and_repeatable(tmp_path):
    counts = {"P0": 80, "N1": 30, "N2": 30, "N5": 20, "N6": 25, "N9": 15}
    rows, seed = [], 3200
    for cls, n in counts.items():
        for _ in range(n):
            rows.append({"seed": seed, "cls": cls, "label": cls == "P0"})
            seed += 1
    np.random.default_rng(1).shuffle(rows)
    store.write_table_atomic(rows, store.part_path("latch", "dev", "states", [3200], tmp_path))
    store.merge("latch", "dev", "states", tmp_path)
    class_of = {r["seed"]: r["cls"] for r in rows}

    picked = judges.subsample("latch", "dev", 40, root=tmp_path)
    assert picked == sorted(set(picked)) and len(picked) == 40 and set(picked) <= set(class_of)
    taken = pd.Series([class_of[s] for s in picked]).value_counts().to_dict()
    assert taken == {"P0": 16, "N1": 6, "N2": 6, "N5": 4, "N6": 5, "N9": 3}        # 40 of 200: 1 in 5 of each class
    assert judges.subsample("latch", "dev", 40, root=tmp_path) == picked            # the same sample every time
    other = judges.subsample("latch", "dev", 40, seed=1, root=tmp_path)
    assert other != picked and pd.Series([class_of[s] for s in other]).value_counts().to_dict() == taken
    # a share that does not divide: the largest remainders get the leftover states, and the total is exact
    odd = judges.subsample("latch", "dev", 33, root=tmp_path)
    assert len(odd) == 33 and pd.Series([class_of[s] for s in odd]).value_counts().to_dict() == {
        "P0": 13, "N1": 5, "N2": 5, "N5": 3, "N6": 4, "N9": 3}
    # equal shares give the small classes as many states as the large ones
    equal = judges.subsample("latch", "dev", 42, allocation="equal", root=tmp_path)
    assert pd.Series([class_of[s] for s in equal]).value_counts().to_dict() == dict.fromkeys(counts, 7)
    # a class that runs out gives its share to the others
    capped = judges.subsample("latch", "dev", 150, allocation="equal", root=tmp_path)
    assert pd.Series([class_of[s] for s in capped]).value_counts().to_dict() == {
        "P0": 30, "N1": 30, "N2": 30, "N5": 20, "N6": 25, "N9": 15}
    assert judges.subsample("latch", "dev", 500, root=tmp_path) == sorted(class_of)  # more than there is: everything
    for bad in (0, -3, 2.5, True):
        with pytest.raises(ValueError):
            judges.subsample("latch", "dev", bad, root=tmp_path)
    with pytest.raises(ValueError, match="allocation"):
        judges.subsample("latch", "dev", 10, allocation="random", root=tmp_path)
    # any split but dev is closed, and it is refused before a table is looked for
    for split in ("test", "calibration", "fit"):
        with pytest.raises(TestSeedError):
            judges.subsample("latch", split, 10, root=tmp_path)
    assert test_seed_reads() == []


def test_the_estimate_total_is_calls_times_the_per_call_cost(capsys):
    table = load_models()
    result = judges.estimate("latch", 400, models=["opus", "luna", "fake"])
    rows = result["rows"]
    assert [(r["model"], r["variant"]) for r in rows] == [(m, v) for m in ("opus", "luna", "fake") for v in judge.VARIANTS]
    for row in rows:
        assert row["calls"] == 400 and row["usd_total"] == pytest.approx(400 * row["usd_per_call"])
        assert row["images"] == (3 if row["variant"] == "crop_ref" else 2)
        assert row["image_px"] == ("640x480" if row["variant"] == "full" else "384x384")
    assert result["total_usd"] == pytest.approx(sum(r["usd_total"] for r in rows)) and result["total_usd"] > 1.0
    # the per-call cost is what the router reserves: the text and schema at 3.5 characters a token, 1600 tokens
    # an image, and the whole output allowance, at the prices of the model table
    by_key = {(r["model"], r["variant"]): r for r in rows}
    for model in ("opus", "luna"):
        entry = table.models[model]
        for variant in judge.VARIANTS:
            text = judges._typical_text("latch", variant, 2)
            n_images = 3 if variant == "crop_ref" else 2
            tokens_in = (math.ceil((len(text) + len(canonical_json(judge.ANSWER_SCHEMA))) / 3.5)
                         + n_images * (entry.image_tokens or 1600))
            allowance = 1000 if entry.reasoning_mandatory else 300
            expected = (tokens_in * entry.price_in + allowance * entry.price_out) / 1e6
            assert by_key[(model, variant)]["usd_per_call"] == pytest.approx(expected)
            assert by_key[(model, variant)]["max_tokens"] == allowance
            assert by_key[(model, variant)]["blocked"] == "no key: set OPENROUTER_API_KEY in .env"
    assert all(by_key[("fake", v)]["usd_per_call"] == 0.0 and by_key[("fake", v)]["placeholder"] for v in judge.VARIANTS)
    assert by_key[("opus", "crop_ref")]["usd_per_call"] > by_key[("opus", "crop")]["usd_per_call"]
    assert judges.estimate("latch", 0, models=["opus"])["total_usd"] == 0.0
    half = judges.estimate("latch", 200, models=["opus"], variants=["crop"])
    assert half["total_usd"] == pytest.approx(by_key[("opus", "crop")]["usd_total"] / 2)
    # the default list: e1_minimum, plus fable when its entry is verified
    expected_default = list(table.e1_minimum) + (["fable"] if table.models["fable"].verified else [])
    assert judges.default_judge_models() == expected_default
    assert {r["model"] for r in judges.estimate("latch", 1)["rows"]} == set(expected_default)
    with pytest.raises(KeyError):
        judges.estimate("latch", 10, models=["no_such_model"])
    with pytest.raises(ValueError):
        judges.estimate("latch", 10, models=["fake"], variants=["zoomed"])
    # the printed table carries the rows, the total, why nothing can be called now, and no answer of any model
    printed_result = judges.print_estimate("latch", 400, models=["opus", "fake"], variants=["crop"])
    out = capsys.readouterr().out
    assert "No call is made." in out and f"{printed_result['total_usd']:.2f}" in out
    assert "opus: no key: set OPENROUTER_API_KEY in .env" in out and "placeholders" in out
    assert fake_calls() == 0


def test_a_stored_state_reaches_the_judge_as_frames_a_claim_and_the_taught_port_only(tmp_path):
    make_zoo(tmp_path, ZOO_SEEDS, ZOO_CLASSES)
    table = store.load("latch", "dev", "states", tmp_path)
    row = table.to_dict("records")[1]
    claim, params = judges.state_inputs(row, "latch", "dev", tmp_path)
    assert set(params) == {"frames", "nominal", "connector", "arm"} and params["connector"] == "latch"
    assert params["frames"].keys() == [f"claim/{WRIST}", f"claim/{TOP}"]            # the free arm's view is not used
    assert (claim.claim_id, claim.predicate.name, claim.predicate.target) == ("zoo_latch_3101", "latched",
                                                                               {"plug": "p1", "port": row["target_port"]})
    nominal = params["nominal"]
    stored = [row["nominal_port_x"], row["nominal_port_y"], row["nominal_port_z"]]
    assert np.allclose(nominal.port_position(row["target_port"]), stored, atol=1e-12)
    assert len(nominal.ports_board) == 8 and (nominal.holding_arm, nominal.free_arm) == ("right", "left")
    frame = params["frames"].get(f"claim/{WRIST}")
    assert frame.rgb.shape == (H, W, 3) and frame.intrinsics.shape == (3, 3) and frame.extrinsics.shape == (4, 4)
    u, v, depth = judge.project_point(frame, nominal.port_position(row["target_port"]))
    assert (u, v, depth) == (pytest.approx(316.0, abs=1e-3), pytest.approx(402.0, abs=1e-3), pytest.approx(0.14, abs=1e-6))
    # a cell that holds no calibration reads as none
    for cell in (None, float("nan"), "", "not json", json.dumps({"K": [1, 2, 3]}), json.dumps({"K": [0] * 9, "T": ["x"] * 16})):
        assert judges._calibration(cell) == (None, None)
    assert judges.nominal_of({**row, "nominal_port_x": float("nan")}, "latch") is None


def test_run_writes_1_row_per_seed_and_a_rerun_is_cached_and_free(tmp_path, capsys):
    make_zoo(tmp_path, ZOO_SEEDS, ZOO_CLASSES)

    def script(req):                    # an answer that depends on the request, so the table shows what was mapped
        return answer(req.image_count == 3, 0.8)

    with fake_script(script):
        summaries = {}
        for variant in judge.VARIANTS:
            first = judges.run("latch", "dev", ZOO_SEEDS, "fake", variant, root=tmp_path, reference=reference_picture())
            again = judges.run("latch", "dev", ZOO_SEEDS, "fake", variant, root=tmp_path, reference=reference_picture())
            summaries[variant] = (first, again)
        assert fake_calls() == 18                                       # 6 states, 3 variants, each asked once
    out = capsys.readouterr().out
    assert out.count("placeholders from the fake model") == 6 and "estimate for latch dev, fake, crop: 6 calls" in out
    for variant, (first, again) in summaries.items():
        assert (first["states"], first["calls"], first["cached"], first["json_ok"]) == (6, 6, 0, 6)
        assert (again["states"], again["calls"], again["cached"], again["json_ok"]) == (6, 0, 6, 6)
        assert first["usd"] == 0.0 and again["usd"] == 0.0 and first["estimate_usd"] == 0.0
        assert first["placeholder"] is True and first["refused"] is None and first["stopped"] is None
        assert first["crop_rule"] == ("none" if variant == "full" else "port")
        path = tmp_path / "latch" / "dev" / "judges" / f"fake__{variant}.parquet"
        assert again["path"] == str(path) and path.is_file() and not list(path.parent.glob("*.tmp"))
        table = pd.read_parquet(path)
        assert list(table.columns[:len(REQUIRED_COLUMNS)]) == REQUIRED_COLUMNS
        assert list(table["seed"]) == ZOO_SEEDS and set(table["model"]) == {"fake"} and set(table["variant"]) == {variant}
        assert table["cached"].all() and table["placeholder"].all() and table["json_ok"].all()
        assert (table["usd"] == 0.0).all() and (table["latency_s"] > 0).all()
        expected = 0.8 if variant == "crop_ref" else 0.2                # 3 images only with the reference
        assert np.allclose(table["p_yes"], expected) and (table["holds"] == (variant == "crop_ref")).all()
        assert np.allclose(table["confidence"], 0.8) and set(table["reason"]) == {"scripted"}
        assert list(table["n_images"]) == [3 if variant == "crop_ref" else 2] * 6
        assert list(table["port"]) == [1 + s % 8 for s in ZOO_SEEDS] and set(table["predicate"]) == {"latched"}
        # no truth in the judges table
        assert not {"cls", "landed_cls", "label"} & set(table.columns) and not [c for c in table.columns if c.startswith("true_")]


def test_what_the_runner_sends_does_not_change_with_the_class_or_the_truth(tmp_path):
    """2 tables with the same frames and claims. Every class and every truth column differs. The requests are equal."""
    make_zoo(tmp_path / "a", ZOO_SEEDS, ZOO_CLASSES)
    make_zoo(tmp_path / "b", ZOO_SEEDS, ["N1", "P0", "P0", "N8", "P0", "N3"], flip_truth=True)
    sent = {"a": [], "b": []}
    for name in sent:
        def script(req, name=name):
            sent[name].append((req.text, req.images))
            return answer()

        with fake_script(script):
            for variant in judge.VARIANTS:
                judges.run("latch", "dev", ZOO_SEEDS, "fake", variant, root=tmp_path / name,
                           reference=reference_picture(), echo=False)
    assert len(sent["a"]) == 18 and sent["a"] == sent["b"]
    for text, images in sent["a"]:
        assert not CLASS_NAME.search(text)
        for seed in ZOO_SEEDS:
            assert str(seed) not in text
        for word in ("zoo_", ".jpg", "frames_", "shortfall", "label", "seed"):
            assert word not in text
        assert len(images) in (2, 3)


def test_a_table_without_calibration_gets_central_square_crops_and_the_run_says_so(tmp_path, capsys):
    make_zoo(tmp_path, ZOO_SEEDS, ZOO_CLASSES, calibrated=False)
    texts = []
    with fake_script(lambda req: texts.append(req.text) or answer()):
        summary = judges.run("latch", "dev", ZOO_SEEDS, "fake", "crop", root=tmp_path)
        full = judges.run("latch", "dev", ZOO_SEEDS, "fake", "full", root=tmp_path, echo=False)
    assert summary["crop_rule"] == "center" and full["crop_rule"] == "none"
    assert "holds no camera calibration" in capsys.readouterr().out
    table = pd.read_parquet(summary["path"])
    assert set(table["crop_rule"]) == {"center"} and table["json_ok"].all()
    assert all("central square of the camera frame" in text for text in texts[:6])
    assert all("cropped" not in text for text in texts[6:])


def test_run_refuses_a_test_seed_and_every_split_but_dev(tmp_path, monkeypatch):
    for split, seeds in (("dev", [16000]), ("dev", [3100, 16001]), ("test", [16000]), ("calibration", [14000]),
                         ("fit", [10000]), ("dev", [99999])):
        with pytest.raises(TestSeedError):
            judges.run("latch", split, seeds, "fake", "full", root=tmp_path)
    with pytest.raises(TestSeedError):
        judges.save_example("latch", "test", 16000, "full", tmp_path / "out", root=tmp_path)
    assert fake_calls() == 0 and not list(tmp_path.rglob("*.parquet")) and test_seed_reads() == []
    with pytest.raises(ValueError, match="not in the dev split"):       # a fit seed under the dev split's name
        judges.run("latch", "dev", [10000], "fake", "full", root=tmp_path)
    # the name of the final run goes to the seed guard, which is what opens another split
    calls = []
    monkeypatch.setattr(judges, "guard", lambda seeds, **kw: calls.append((list(seeds), kw)))
    with pytest.raises(FileNotFoundError):                              # the guard let it through. There is no table here
        judges.run("latch", "calibration", [14000, 14001], "fake", "full", final_run="E1E2", root=tmp_path)
    # a judge reads stored frames and simulates nothing, so it reaches the guard as a stored read (S4c)
    assert calls == [([14000, 14001], {"purpose": "judge run, latch calibration, fake full", "final_run": "E1E2",
                                       "stored": True})]


def test_run_refuses_a_paid_model_without_a_key_and_says_so(tmp_path, monkeypatch, capsys):
    make_zoo(tmp_path, ZOO_SEEDS, ZOO_CLASSES)
    summary = judges.run("latch", "dev", ZOO_SEEDS, "opus", "crop", session="S3", root=tmp_path)
    out = capsys.readouterr().out
    assert "OPENROUTER_API_KEY" in summary["refused"] and summary["path"] is None and summary["states"] == 0
    assert "judge run refused." in out and "No call was made." in out and "estimate for" not in out
    # with a key: no spend session, a session that needs an OK, and an estimate above what the session has left
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    assert "no spend session" in judges.run("latch", "dev", ZOO_SEEDS, "opus", "crop", root=tmp_path, echo=False)["refused"]
    needs_ok = judges.run("latch", "dev", ZOO_SEEDS, "opus", "crop", session="full_judge_run", root=tmp_path, echo=False)
    assert "Kevin's OK" in needs_ok["refused"]
    tight = Router(session="T", caps=SpendCaps(total_usd=1.0, sessions={"T": 0.05}), http=refusing_client())
    over = judges.run("latch", "dev", ZOO_SEEDS, "opus", "crop", router=tight, root=tmp_path)
    out = capsys.readouterr().out
    assert "estimate for latch dev, opus, crop: 6 calls" in out          # the estimate is printed before the refusal
    assert over["estimate_usd"] > 0.05 and "above what session T has left ($0.0500 of its $0.05 cap)" in over["refused"]
    assert PLAIN_KEY not in out and PLAIN_KEY not in json.dumps([summary, needs_ok, over], default=str)
    # nothing was sent, written or charged in any of the 4
    assert not list(tmp_path.rglob("judges/*.parquet")) and tight.spent() == 0.0 and fake_calls() == 0


def test_run_stops_at_max_states_and_survives_a_failed_call(tmp_path, capsys, monkeypatch):
    make_zoo(tmp_path, ZOO_SEEDS, ZOO_CLASSES)
    with fake_script(lambda req: answer()):
        cut = judges.run("latch", "dev", ZOO_SEEDS, "fake", "full", max_states=4, root=tmp_path)
    assert cut["states"] == 4 and list(pd.read_parquet(cut["path"])["seed"]) == ZOO_SEEDS[:4]
    assert "Running the leading 4." in capsys.readouterr().out
    # 1 failed call gets a row and the run goes on. 3 in a row stop it, with what was gathered written
    boom = [answer(), ModelCallError("the provider answered 503", status=503), answer(),
            ModelCallError("503 again", status=503), ModelCallError("503 again", status=503),
            ModelCallError("503 again", status=503)]
    with fake_script(boom):
        stopped = judges.run("latch", "dev", ZOO_SEEDS, "fake", "crop", root=tmp_path, echo=False)
    table = pd.read_parquet(stopped["path"])
    assert "3 calls in a row failed" in stopped["stopped"] and stopped["states"] == 6 and stopped["json_ok"] == 2
    assert list(table["json_ok"]) == [True, False, True, False, False, False]
    assert table["error"].iloc[1].startswith("ModelCallError: the provider answered 503") and table["p_yes"].isna().sum() == 4
    # a cap reached in the middle of a run (another process spent the room) stops it at once
    real, count = judge.judge, {"n": 0}

    def third_call_hits_the_cap(claim, params):
        count["n"] += 1
        if count["n"] == 3:
            raise SpendCapExceeded("Session S3 cap $8.0000 would be crossed", scope="session", session="S3")
        return real(claim, params)

    monkeypatch.setattr(judge, "judge", third_call_hits_the_cap)
    capped = judges.run("latch", "dev", ZOO_SEEDS, "fake", "full", root=tmp_path, echo=False)
    assert capped["states"] == 2 and capped["stopped"].startswith("SpendCapExceeded") and count["n"] == 3
    assert list(pd.read_parquet(capped["path"])["seed"]) == ZOO_SEEDS[:2]


def test_the_command_line_estimates_runs_and_shows_without_a_key(capsys):
    make_zoo(zoo_dir(), ZOO_SEEDS, ZOO_CLASSES)                         # AVAIL_DATA is a temp folder here
    assert judges.main(["estimate", "--connector", "latch", "--n", "10", "--models", "luna,fake", "--variants", "crop"]) == 0
    out = capsys.readouterr().out
    assert "latch, 10 states" in out and "luna" in out and "total" in out
    assert judges.main(["run", "--connector", "latch", "--model", "fake", "--variant", "full,crop", "--n", "4"]) == 0
    out = capsys.readouterr().out
    assert out.count("4 states, 4 answered by the model") == 2 and out.count("placeholders") == 2
    table = pd.read_parquet(zoo_dir() / "latch" / "dev" / "judges" / "fake__crop.parquet")
    assert len(table) == 4 and set(table["seed"]) <= set(ZOO_SEEDS) and table["placeholder"].all()
    assert judges.main(["run", "--connector", "latch", "--model", "opus", "--variant", "crop", "--session", "S3"]) == 2
    assert "OPENROUTER_API_KEY" in capsys.readouterr().out
    assert judges.main(["show", "--connector", "latch", "--seed", "3102", "--variant", "crop"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["crop_rule"] == "port" and len(shown["images"]) == 2 and shown["problem"] is None
    from PIL import Image

    assert np.asarray(Image.open(shown["images"][0])).shape == (384, 384, 3)
    with open(shown["prompt"], encoding="utf-8") as fh:
        assert "Condition (latched)" in fh.read()


# ============================================================================ on real injected states (dev seeds)
@pytest.fixture(scope="module")
def twins():
    """A seated plug and its near-miss twin on the same dev seed: the claim and the params of each, station closed."""
    from avail.zoo.harness import prepare

    out = {}
    for cls in ("P0", "N2"):
        prepared = prepare(TWIN_SEED, "latch", cls)
        try:
            assert prepared.outcome.landed_class == cls, "the injection did not land where it was aimed"
            out[cls] = (prepared.claim, prepared.fresh_params())
        finally:
            prepared.close()
    return out


@needs_station
def test_a_seated_plug_and_its_near_miss_twin_get_the_same_text(twins):
    reference = reference_picture()
    for variant in judge.VARIANTS:
        requests = {}
        for cls, (claim, params) in twins.items():
            requests[cls] = judge.build_request(claim, {**params, "judge": {"model": "fake", "variant": variant,
                                                                            "reference": reference}})
            assert requests[cls].problem is None
        seated, short = requests["P0"], requests["N2"]
        assert seated.text == short.text                                # the text does not track the injected class
        for text in (seated.text, short.text):
            assert not CLASS_NAME.search(text) and str(TWIN_SEED) not in text
            assert twins["P0"][0].claim_id not in text and "zoo" not in text
            assert PREDICATE_DEFINITIONS["latched"] in text and "RJ45" in text
            assert f"port {twins['P0'][0].predicate.target['port']}" in text
        # the 2 states differ where they may: in the wrist camera's pixels
        wrist = 1 if variant == "crop_ref" else 0
        assert len(seated.images) == len(short.images) == (3 if variant == "crop_ref" else 2)
        assert not np.array_equal(seated.images[wrist], short.images[wrist])
        if variant != "full":
            assert all(view.crop == "port" for view in seated.views + short.views)
            assert all(image.shape == (384, 384, 3) for image in seated.images)


@needs_station
def test_the_crop_of_a_live_wrist_frame_holds_the_projected_port(twins):
    claim, params = twins["P0"]
    nominal, port = params["nominal"], claim.predicate.target["port"]
    frame = params["frames"].get(f"claim/{WRIST}")
    u, v, depth = judge.project_point(frame, nominal.port_position(port))
    assert 0 <= u < frame.width and 0 <= v < frame.height and 0.08 < depth < 0.25      # in view, 8 to 25 cm away
    crop = judge.crop_to_port(frame, nominal, port)
    assert crop.on_port and crop.port_uv == (pytest.approx(u), pytest.approx(v))
    assert crop.side == round(frame.intrinsics[0, 0] * 0.080 / depth) and crop.side < min(frame.width, frame.height)
    assert crop.x0 <= u < crop.x0 + crop.side and crop.y0 <= v < crop.y0 + crop.side
    cu, cv = crop.locate(u, v)
    assert in_the_middle(crop, cu) and in_the_middle(crop, cv)
    assert crop.rgb.shape == (384, 384, 3) and crop.rgb.size < frame.rgb.size
    # the overhead view shows the port too, from about 1 m, so its square is the 96 pixel minimum
    top = judge.crop_to_port(params["frames"].get(f"claim/{TOP}"), nominal, port)
    assert top is not None and top.side == 96 and top.rgb.shape == (384, 384, 3)


@needs_station
def test_collect_on_a_live_state_gives_a_placeholder_evidence(twins):
    """The smoke test of the source: the whole path on a real injected state. The answer itself is a placeholder."""
    claim, params = twins["N2"]
    params = {**params, "judge": {"model": "fake", "variant": "crop"}}
    evidence = judge.JUDGE_VLM.collect(None, claim, params)
    assert isinstance(evidence, Evidence) and evidence.claim_id == claim.claim_id and evidence.kind == "judge"
    assert evidence.features["placeholder"] == 1.0 and evidence.features["n_images"] == 2.0
    assert evidence.features["crop_on_port"] == 1.0 and evidence.features["cached"] == 0.0
    assert 0.0 <= evidence.score <= 1.0 and evidence.cost.usd == 0.0 and evidence.side_effect == "none"
    assert judge.JUDGE_VLM.collect(None, claim, params).features["cached"] == 1.0


@needs_station
def test_the_reference_picture_is_built_once_on_a_dev_seed_and_then_read_from_the_cache(monkeypatch):
    from avail.zoo import harness

    assert judges.REFERENCE_SEED == 3000
    path = judges.reference_path("latch")
    assert not path.is_file()
    picture = judges.reference_image("latch")
    assert path.is_file() and path.parent.name == "reference" and path.parent.parent.name == "cache"
    assert picture.shape == (384, 384, 3) and picture.dtype == np.uint8 and picture.std() > 10
    # the second call simulates nothing and returns the same pixels
    monkeypatch.setattr(harness, "prepare", lambda *a, **k: pytest.fail("the reference was simulated twice"))
    assert np.array_equal(judges.reference_image("latch"), picture)
