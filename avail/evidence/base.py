"""The EvidenceSource protocol, the registry, and the only handle a source gets on a station.

An evidence source turns cameras, proprioception and nominal geometry into features, a
score and a cost. It never sees ground truth (CLAUDE.md rule 10). Two things enforce that:

- ``StationView`` is what ``collect`` receives. It carries the Station protocol and
  nothing else, so a source cannot reach the simulator through it.
- ``tests/test_truth_wall.py`` scans every file in this package for imports of the truth
  module, of MuJoCo, or of the sim backend, and for private attributes of a station.

What ``collect(station, claim, params)`` may find in ``params`` (the caller fills what it has):

- ``nominal``     a ``NominalGeometry``: taught poses and part dimensions, with localization error
- ``connector``   the connector type, for example ``latch``
- ``arm``         the arm holding the plug (default ``nominal.holding_arm``)
- ``insert``      what the insertion skill itself recorded: ``flag``, ``stopped_by``, ``travel_mm``,
                  ``expected_travel_mm``, ``force_at_stop_n``, ``peak_force_n`` and ``trace``
                  (``t_s``, ``travel_mm``, ``force_n`` lists)
- ``frames``      a ``FrameStore``: camera frames by key, shared between sources. Claim-time frames
                  are stored as ``claim/<camera>``. ``vision.gap`` with view ``now`` adds ``now/<camera>``
- ``view``        for vision sources, which view to use: ``claim`` (stored at claim time) or ``now`` (taken fresh)

A source returns features as a flat dict of finite floats. A feature it could not measure is
left out. ``score`` is optional. When given, higher means the predicate is more likely to hold
once the source has finished acting.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

from ..core.robot import CameraFrame, MotionResult, Observation, Pose, RobotState, Station
from ..schemas import Claim, Cost, Evidence, new_id

PROTOCOL_METHODS = ("observe", "robot_state", "move_to_pose", "move_relative", "set_gripper", "stop", "wait")


@runtime_checkable
class EvidenceSource(Protocol):
    source_id: str                      # for example "probe.push_seat"
    kind: str                           # one of avail.schemas.EVIDENCE_KINDS
    predicates: set[str]                # the predicates this source bears on

    def estimate_cost(self, claim: Claim) -> Cost:
        """What a call is expected to cost, before making it."""
        ...

    def collect(self, station: Station, claim: Claim, params: dict) -> Evidence:
        """Measure, and return the evidence."""
        ...


class StationView:
    """A station as an evidence source may see it: the Station protocol and nothing else.

    Instrumented stations also report their own signals through ``read_signal``. Every other
    attribute raises, so a source that reaches for simulator state fails loudly.
    """

    __slots__ = ("__station",)

    def __init__(self, station: Station):
        object.__setattr__(self, "_StationView__station", station)

    @property
    def station_id(self) -> str:
        return self.__station.station_id

    def observe(self, cameras: list[str] | None = None) -> Observation:
        return self.__station.observe(cameras)

    def robot_state(self) -> RobotState:
        return self.__station.robot_state()

    def move_to_pose(self, arm: str, pose: Pose, speed: float, max_force: float | None = None) -> MotionResult:
        return self.__station.move_to_pose(arm, pose, speed, max_force)

    def move_relative(self, arm: str, delta_m, frame: str, speed: float, max_force: float, max_travel: float) -> MotionResult:
        return self.__station.move_relative(arm, delta_m, frame, speed, max_force, max_travel)

    def set_gripper(self, arm: str, position: float, max_torque: float | None = None) -> MotionResult:
        return self.__station.set_gripper(arm, position, max_torque)

    def stop(self) -> None:
        self.__station.stop()

    def wait(self, seconds: float) -> None:
        self.__station.wait(seconds)

    def read_signal(self, name: str) -> dict:
        reader = getattr(self.__station, "read_signal", None)
        if reader is None:
            raise KeyError(f"station {self.station_id!r} reports no signals. It is not instrumented")
        return reader(name)

    def __getattr__(self, name: str):
        raise AttributeError(
            f"an evidence source may not read {name!r} from a station. Use the Station protocol: "
            + ", ".join(PROTOCOL_METHODS)
        )

    def __setattr__(self, name: str, value) -> None:
        raise AttributeError("a StationView is read-only")


class FrameStore:
    """Camera frames by key, shared between sources during one verification."""

    def __init__(self) -> None:
        self._frames: dict[str, CameraFrame] = {}

    def put(self, key: str, frame: CameraFrame) -> str:
        self._frames[key] = frame
        return key

    def put_observation(self, prefix: str, obs: Observation) -> list[str]:
        return [self.put(f"{prefix}/{name}", frame) for name, frame in obs.frames.items()]

    def get(self, key: str) -> CameraFrame | None:
        return self._frames.get(key)

    def keys(self, prefix: str = "") -> list[str]:
        return sorted(k for k in self._frames if k.startswith(prefix))

    def __contains__(self, key: str) -> bool:
        return key in self._frames


REGISTRY: dict[str, EvidenceSource] = {}


def register(source: EvidenceSource) -> EvidenceSource:
    """Add a source to the registry. A new source is 1 file that calls this once."""
    for attr in ("source_id", "kind", "predicates", "estimate_cost", "collect"):
        if not hasattr(source, attr):
            raise TypeError(f"{type(source).__name__} is not an evidence source: it has no {attr}")
    if source.source_id in REGISTRY and REGISTRY[source.source_id] is not source:
        raise ValueError(f"an evidence source named {source.source_id!r} is already registered")
    REGISTRY[source.source_id] = source
    return source


BUILTIN_MODULES = ("passive", "probes", "vision", "instrumented", "judge")


def _import_source_module(module: str) -> bool:
    """Import ``avail.evidence.<module>`` so the sources in it register. False when the module does not exist."""
    import importlib

    try:
        importlib.import_module(f"{__package__}.{module}")
    except ModuleNotFoundError as exc:
        if exc.name != f"{__package__}.{module}":
            raise
        return False
    return True


def load_builtin() -> dict[str, EvidenceSource]:
    """Import the built-in source modules so they register, and return the registry."""
    for module in BUILTIN_MODULES:
        _import_source_module(module)
    return REGISTRY


def get(source_id: str) -> EvidenceSource:
    """The registered source named ``source_id``.

    A source not registered yet is looked for in the built-in modules, then in a file of its own:
    ``<kind>.<name>`` in ``avail/evidence/<name>.py``, the layout of a check that is 1 file.
    """
    if source_id not in REGISTRY:
        load_builtin()
    if source_id not in REGISTRY:
        name = source_id.partition(".")[2]
        if name.isidentifier() and not name.startswith("_"):
            _import_source_module(name)
    if source_id not in REGISTRY:
        raise KeyError(f"unknown evidence source {source_id!r}. Known: {', '.join(sorted(REGISTRY))}")
    return REGISTRY[source_id]


def sources_for(predicate: str) -> list[EvidenceSource]:
    """The sources registered so far that bear on ``predicate``, the built-in ones included.

    A check that lives in a file of its own is listed once something imported it or asked ``get`` for it.
    """
    load_builtin()
    return [s for _, s in sorted(REGISTRY.items()) if predicate in s.predicates]


def make_evidence(source: EvidenceSource, claim: Claim, features: dict, *, score: float | None = None,
                  seconds: float = 0.0, usd: float = 0.0, moved_robot: bool = False, side_effect: str = "none",
                  artifacts: list[str] | None = None, ts: float | None = None) -> Evidence:
    """Build the Evidence record for a measurement. Non-finite features are dropped, as the schema asks."""
    clean = {}
    for key, value in features.items():
        if value is None:
            continue
        value = float(value)
        if value == value and abs(value) != float("inf"):
            clean[key] = value
    if score is not None:
        score = float(score)
        if score != score or abs(score) == float("inf"):
            score = None
    return Evidence(
        evidence_id=new_id("ev"), claim_id=claim.claim_id, source_id=source.source_id, kind=source.kind,
        features=clean, score=score, cost=Cost(seconds=max(0.0, float(seconds)), usd=max(0.0, float(usd)), moved_robot=bool(moved_robot)),
        artifacts=list(artifacts or []), side_effect=side_effect, ts=time.time() if ts is None else float(ts),
    )
