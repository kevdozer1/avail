"""Environment, secrets and the two config files.

``.env`` is parsed here with a CRLF-safe reader (``splitlines`` drops ``\\r``. A
bash loader on this machine once produced paths ending in ``\\r``). Values in the
process environment win over ``.env``. Secret values are returned to the caller
and never printed, logged or put in an exception message: a missing key raises
:class:`MissingCredentialError`, which names the variable to set and nothing else.

``config/experiment.yaml`` holds the seed budget, the bounds, the station noise
and the probe caps. ``config/models.yaml`` holds the judge model table. The
typed model table lives in :mod:`avail.router.models` because the router owns
prices and spend.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .paths import REPO_ROOT

ENV_FILE = REPO_ROOT / ".env"
EXPERIMENT_YAML = REPO_ROOT / "config" / "experiment.yaml"
MODELS_YAML = REPO_ROOT / "config" / "models.yaml"

TEST_SPLITS = ("zoo_test", "shift_test_e3", "shift_test_e4", "shift_test_e7", "zoo_test_hard", "zoo_test_hard2",
               "fig_test", "shift_test_e3b")
PROFILES = ("easy", "hard", "hard2")
HARD_PROFILES = ("hard", "hard2")          # profiles built on the hard-mode station: webcam, jams, force, lots


# --------------------------------------------------------------------------- .env
def parse_dotenv(text: str) -> dict[str, str]:
    """``KEY=value`` lines to a dict. Blank lines and ``#`` comments are skipped."""
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


_DOTENV: dict[str, str] | None = None


def dotenv_values(path: Path | None = None, reload: bool = False) -> dict[str, str]:
    """The repo ``.env`` as a dict (cached). A missing file gives an empty dict."""
    global _DOTENV
    if path is not None:
        p = Path(path)
        return parse_dotenv(p.read_text(encoding="utf-8-sig")) if p.is_file() else {}
    if _DOTENV is None or reload:
        _DOTENV = parse_dotenv(ENV_FILE.read_text(encoding="utf-8-sig")) if ENV_FILE.is_file() else {}
    return _DOTENV


def env(name: str, default: str | None = None) -> str | None:
    """A setting from the process environment, then ``.env``. Empty counts as unset."""
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        value = dotenv_values().get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


class MissingCredentialError(RuntimeError):
    """A model's API key is absent. The message names the variable, never a value."""


def has_secret(names: list[str] | tuple[str, ...]) -> bool:
    return any(env(n) for n in names)


def load_secret(names: list[str] | tuple[str, ...], label: str) -> str:
    """The first set variable among ``names``. Raises MissingCredentialError naming ``names[0]``."""
    for name in names:
        value = env(name)
        if value:
            return value
    others = f" (or {', '.join(names[1:])})" if len(names) > 1 else ""
    raise MissingCredentialError(
        f"{label} credential not found. Set {names[0]}{others} in the environment "
        f"or add {names[0]}=... to {ENV_FILE}."
    )


def hardware_allowed() -> bool:
    """True only when ``AVAIL_ALLOW_HARDWARE=1`` (CLAUDE.md rule 14). Nothing in S0 to S4 moves hardware."""
    return env("AVAIL_ALLOW_HARDWARE") == "1"


# ---------------------------------------------------------------- experiment.yaml
class _Section(BaseModel):
    """A config section. Unknown keys are kept so later sessions can add to the file."""

    model_config = ConfigDict(extra="allow")


class SeedBudget(_Section):
    dev: tuple[int, int]
    zoo_fit: tuple[int, int]
    zoo_calibration: tuple[int, int]
    zoo_test: tuple[int, int]
    shift_dev: tuple[int, int]
    shift_test_e3: tuple[int, int]
    shift_test_e4: tuple[int, int]
    shift_test_e7: tuple[int, int]
    zoo_fit_hard: tuple[int, int]           # S4b
    zoo_calibration_hard: tuple[int, int]
    zoo_test_hard: tuple[int, int]
    zoo_fit_hard2: tuple[int, int]          # S4c
    zoo_calibration_hard2: tuple[int, int]
    zoo_test_hard2: tuple[int, int]
    cable_fit: tuple[int, int] = (24000, 25999)          # S5b: the cable loop's insertions
    cable_calibration: tuple[int, int] = (26000, 27999)
    fig_test: tuple[int, int] = (72000, 72099)            # S5c: the README figure's run, read once by FIG
    check_fit: tuple[int, int] = (28000, 28999)           # S5c fix: the hand-set checks, fit
    check_calibration: tuple[int, int] = (29000, 29999)   # S5c fix: and their calibration
    shift_test_e3b: tuple[int, int] = (80000, 84999)      # E3b: the 1 retest of the defect bar, read once
    check_fit_drag: tuple[int, int] = (85000, 85499)      # S5c enrichment: the release's tip and drag, fit
    check_calibration_drag: tuple[int, int] = (85500, 85999)   # and calibration

    @model_validator(mode="after")
    def _disjoint(self) -> "SeedBudget":
        spans = sorted((lo, hi, name) for name, (lo, hi) in self.ranges().items())
        for (lo, hi, name) in spans:
            if lo > hi or lo < 0:
                raise ValueError(f"seeds.{name} must be [low, high] with 0 <= low <= high, got [{lo}, {hi}]")
        for (_, hi_a, a), (lo_b, _, b) in zip(spans, spans[1:]):
            if lo_b <= hi_a:
                raise ValueError(f"seeds.{a} and seeds.{b} overlap. Seed ranges must be disjoint")
        return self

    def ranges(self) -> dict[str, tuple[int, int]]:
        return {name: tuple(getattr(self, name)) for name in type(self).model_fields}

    def seeds(self, split: str) -> range:
        if split not in type(self).model_fields:
            raise KeyError(f"unknown seed split {split!r}. Known: {', '.join(type(self).model_fields)}")
        lo, hi = getattr(self, split)
        return range(lo, hi + 1)

    def split_of(self, seed: int) -> str | None:
        for name, (lo, hi) in self.ranges().items():
            if lo <= seed <= hi:
                return name
        return None


class Bounds(_Section):
    alpha: float = Field(gt=0, lt=1)
    beta: float = Field(gt=0, lt=1)
    alpha_sweep: list[float]
    certificate_confidence: float = Field(gt=0, lt=1)
    true_accept_for_e1: float = Field(gt=0, lt=1)


class RenderSize(_Section):
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class Board(_Section):
    rows: int = Field(gt=0)
    cols: int = Field(gt=0)
    pitch_mm: float = Field(gt=0)
    mode: str


class Randomization(_Section):
    board_xy_mm: float
    board_yaw_deg: float
    camera_xyz_mm: float
    camera_rot_deg: float
    lights: tuple[int, int]
    distractor_cables: tuple[int, int]


class Noise(_Section):
    force_estimate_sigma_n: float = Field(ge=0)
    force_estimate_bias_drift_n_per_min: float = Field(ge=0)
    port_localization_sigma_mm: float = Field(ge=0)
    stiction_under_travel_mm: dict[str, float]


class StationConfig(_Section):
    render: RenderSize
    cameras: list[str]
    board: Board
    randomization: Randomization
    noise: Noise


class ConnectorGeometry(_Section):
    """Plug and socket shape in millimeters. x is the direction the fingers squeeze."""

    tongue_xy: tuple[float, float]
    body_xyz: tuple[float, float, float]
    clearance: float = Field(gt=0)
    chamfer: float = Field(ge=0)


def _table(points: list[tuple[float, float]] | None, name: str) -> list[tuple[float, float]] | None:
    if points is None:
        return None
    xs = [float(p[0]) for p in points]
    if len(points) < 2 or any(b <= a for a, b in zip(xs, xs[1:])):
        raise ValueError(f"{name} must be a list of at least 2 [x, value] points with strictly increasing x")
    return [(float(p[0]), float(p[1])) for p in points]


class ConnectorSpec(_Section):
    full_depth_mm: float = Field(gt=0)
    insertion_peak_n: float = Field(gt=0)
    retention_seated_n: float = Field(gt=0)
    retention_2mm_short_n: tuple[float, float]
    retention_latch_blocked_n: float | None = None
    seated_min_depth_mm: float | None = None
    geometry_mm: ConnectorGeometry | None = None
    axial_friction_n: list[tuple[float, float]] | None = None          # [depth_mm, N]
    detent_n: list[tuple[float, float]] | None = None                  # [shortfall_mm, N toward the seat]
    detent_latch_blocked_n: list[tuple[float, float]] | None = None

    @model_validator(mode="after")
    def _tables(self) -> "ConnectorSpec":
        self.axial_friction_n = _table(self.axial_friction_n, "axial_friction_n")
        self.detent_n = _table(self.detent_n, "detent_n")
        self.detent_latch_blocked_n = _table(self.detent_latch_blocked_n, "detent_latch_blocked_n")
        return self

    @property
    def modeled(self) -> bool:
        """True when the connector has the geometry and force tables the plug station needs."""
        return self.geometry_mm is not None and self.axial_friction_n is not None and self.detent_n is not None


class ZooConfig(_Section):
    per_connector: int = Field(gt=0)
    positive_share: float = Field(gt=0, lt=1)
    near_miss_share_of_negatives: float = Field(gt=0, lt=1)
    judge_subsample_per_connector: int = Field(gt=0)
    classes: list[str]


class PushSeatCaps(_Section):
    force_cap_x_insertion_peak: float = Field(gt=0)
    travel_cap_mm: float = Field(gt=0)
    speed_mm_s: float = Field(gt=0)


class TugCaps(_Section):
    force_cap_x_retention: float = Field(gt=0)
    travel_cap_mm: float = Field(gt=0)
    speed_mm_s: float = Field(gt=0)
    slip_mm: float = Field(gt=0)


class WiggleCaps(_Section):
    force_cap_n: float = Field(gt=0)
    travel_cap_mm: float = Field(gt=0)
    cycles: int = Field(gt=0)


class RelookConfig(_Section):
    standoff_mm: float = Field(gt=0)


class ProbeCaps(_Section):
    push_seat: PushSeatCaps
    tug: TugCaps
    wiggle: WiggleCaps
    relook: RelookConfig


class WristCameraSpec(_Section):
    """Hard mode's wrist camera: a 1080p webcam with a 130 degree fisheye-style lens (S4b)."""

    sensor_px: tuple[int, int]
    diagonal_fov_deg: float = Field(gt=0, lt=180)
    projection: str
    unit_focal_pct: float = Field(ge=0)
    unit_center_px: float = Field(ge=0)
    unit_k_sd: tuple[float, float]
    calib_focal_pct_sd: float = Field(ge=0)
    calib_center_px_sd: float = Field(ge=0)
    calib_k_sd: tuple[float, float]
    pixel_pitch_um: float = Field(gt=0)
    f_number: float = Field(gt=0)
    focus_distance_m: float = Field(gt=0)
    lens_psf_px: float = Field(ge=0)
    full_well_e: float = Field(gt=0)
    read_noise_e: float = Field(ge=0)
    white_e_per_s: float = Field(gt=0)
    max_exposure_s: float = Field(gt=0)
    max_gain: float = Field(ge=1)
    ae_target: float = Field(gt=0, lt=1)
    ae_jitter_ev: float = Field(ge=0)
    awb_error_sd: float = Field(ge=0)
    awb_jitter_sd: float = Field(ge=0)
    readout_s: float = Field(ge=0)
    jpeg_quality: int = Field(ge=1, le=100)
    chroma_subsampling: str
    working_px: tuple[int, int]

    @model_validator(mode="after")
    def _known(self) -> "WristCameraSpec":
        if self.projection != "equidistant":
            raise ValueError("profiles.hard.wrist_camera.projection: only 'equidistant' is modeled")
        if self.chroma_subsampling not in ("4:4:4", "4:2:2", "4:2:0"):
            raise ValueError("profiles.hard.wrist_camera.chroma_subsampling must be 4:4:4, 4:2:2 or 4:2:0")
        return self


class PlugCableSpec(_Section):
    diameter_mm: float = Field(gt=0)
    side_gap_mm: tuple[float, float]
    end_above_shoulder_mm: float
    sector_deg: float = Field(gt=0, le=180)


class JamSpec(_Section):
    classes: list[str]
    share: float = Field(ge=0, le=1)
    kinds: dict[str, float]
    wedge_hold_x_retention: tuple[float, float]
    wedge_seat_mm: float = Field(gt=0)
    wedge_ramp_mm: float = Field(gt=0)
    wedge_span_mm: float = Field(gt=0)

    @model_validator(mode="after")
    def _kinds(self) -> "JamSpec":
        if set(self.kinds) - {"burr", "wedge"} or not self.kinds or sum(self.kinds.values()) <= 0:
            raise ValueError("profiles.hard.jams.kinds must give positive weights to burr and wedge only")
        return self


class ForceSpec(_Section):
    bias_drift_n_per_min: float = Field(ge=0)
    bias_walk_n_per_sqrt_s: float = Field(ge=0)
    hysteresis_n: float = Field(ge=0)
    tare_with_noise: bool
    detent_lot_pct: float = Field(ge=0, lt=100)
    lots: int = Field(gt=0)


class JointReadingSpec(_Section):
    """Hard mode v2 (S4c): what a joint reading carries of the servo's encoder, on the output shaft."""

    counts_per_turn: int = Field(gt=0)
    zero_phase_counts: tuple[float, float]


class GearLashSpec(_Section):
    """Hard mode v2 (S4c): the servo's gear lash, as lost motion when a joint's torque turns."""

    lash_deg: tuple[float, float]
    motor_speed_rad_s: float = Field(gt=0)
    breakaway_nm: float = Field(gt=0)

    @model_validator(mode="after")
    def _range(self) -> "GearLashSpec":
        lo, hi = self.lash_deg
        if not 0 <= lo <= hi:
            raise ValueError("profiles.<name>.gear_lash.lash_deg must be [low, high] with 0 <= low <= high")
        return self


class CurrentReadingSpec(_Section):
    """Hard mode v2 (S4c): what a joint torque reading carries of the servo's current sensing."""

    count_ma: float = Field(gt=0)
    stall_current_a: float = Field(gt=0)
    noise_counts_sd: float = Field(ge=0)


class HardProfile(_Section):
    """``profiles.hard``: what hard mode adds on top of easy mode (S4b).

    ``profiles.hard2`` (S4c) names ``base: hard`` and adds ``joint_readings``, ``gear_lash`` and
    ``current_readings``. The base block's sections are copied in before validation.
    """

    base: str | None = None
    wrist_camera: WristCameraSpec
    plug_cable: PlugCableSpec
    jams: JamSpec
    force: ForceSpec
    joint_readings: JointReadingSpec | None = None
    gear_lash: GearLashSpec | None = None
    current_readings: CurrentReadingSpec | None = None


class ExperimentConfig(_Section):
    """``config/experiment.yaml``. Sections used by S0 to S4 are typed. The rest stay as plain dicts."""

    version: str
    seeds: SeedBudget
    customer_bar: dict[str, Any]
    bounds: Bounds
    predicates: dict[str, dict[str, Any]]
    station: StationConfig
    connectors: dict[str, ConnectorSpec]
    zoo: ZooConfig
    probes: ProbeCaps
    cascade: dict[str, Any]
    fork: dict[str, Any]
    human: dict[str, Any]
    audit: dict[str, Any]
    flywheel: dict[str, Any]
    drift: dict[str, Any]
    standin_s1: dict[str, Any]
    e3: dict[str, Any]
    e4: dict[str, Any]
    e7: dict[str, Any]
    economics: dict[str, Any]
    spend_caps_usd: dict[str, float]
    profiles: dict[str, HardProfile] = Field(default_factory=dict)
    fig: dict[str, Any] = Field(default_factory=dict)        # S5c: the README figure's final run
    e3b: dict[str, Any] = Field(default_factory=dict)        # the 1 retest of the defect bar (H3b)
    check_witness: dict[str, Any] = Field(default_factory=dict)   # 2026-11-16: the check witnesses' alphas and crossing

    @field_validator("version", mode="before")
    @classmethod
    def _version_text(cls, value: Any) -> Any:
        return value.isoformat() if hasattr(value, "isoformat") else value

    @field_validator("profiles", mode="before")
    @classmethod
    def _inherit(cls, value: Any) -> Any:
        """A profile that names ``base`` starts from that profile's sections. Its own sections win."""
        if not isinstance(value, dict):
            return value
        out = {}
        for name, block in value.items():
            if isinstance(block, dict) and block.get("base") is not None:
                base = value.get(block["base"])
                if not isinstance(base, dict) or base.get("base") is not None:
                    raise ValueError(f"profiles.{name}.base must name a profile without a base of its own")
                block = {**{k: v for k, v in base.items() if k != "base"}, **block}
            out[name] = block
        return out

    def connector(self, name: str) -> ConnectorSpec:
        if name not in self.connectors:
            raise KeyError(f"unknown connector {name!r}. Known: {', '.join(sorted(self.connectors))}")
        return self.connectors[name]

    def hard(self, profile: str = "hard") -> HardProfile:
        """A hard-mode block, ``profiles.hard`` by default. Raises when the config has none."""
        if profile not in self.profiles:
            raise KeyError(f"config/experiment.yaml has no profiles.{profile} block, so that mode cannot be built")
        return self.profiles[profile]

    def push_force_cap_n(self, connector: str) -> float:
        """The push probe's force cap for a connector. Config only (CLAUDE.md rule 15)."""
        return self.probes.push_seat.force_cap_x_insertion_peak * self.connector(connector).insertion_peak_n

    def tug_force_cap_n(self, connector: str) -> float:
        """The tug probe's force cap for a connector. Config only (CLAUDE.md rule 15)."""
        return self.probes.tug.force_cap_x_retention * self.connector(connector).retention_seated_n


def _read_yaml(path: Path) -> Any:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


@lru_cache(maxsize=8)
def _load_experiment_cached(path: str, mtime_ns: int) -> ExperimentConfig:
    return ExperimentConfig.model_validate(_read_yaml(Path(path)))


def load_experiment(path: Path | None = None) -> ExperimentConfig:
    """The experiment config, validated. Cached per file version."""
    p = Path(path or EXPERIMENT_YAML)
    return _load_experiment_cached(str(p), p.stat().st_mtime_ns)


def load_models_raw(path: Path | None = None) -> dict[str, Any]:
    """``config/models.yaml`` as a plain dict. The router validates the ``models`` block."""
    data = _read_yaml(Path(path or MODELS_YAML))
    if not isinstance(data, dict) or "models" not in data:
        raise ValueError(f"{path or MODELS_YAML} must be a mapping with a 'models' key")
    return data


def config_hash(path: Path | None = None) -> str:
    """A short hash of the experiment config, quoted in every report. Line endings do not change it."""
    raw = Path(path or EXPERIMENT_YAML).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()[:12]
