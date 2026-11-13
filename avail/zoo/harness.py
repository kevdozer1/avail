"""Put one injected state in front of the evidence sources. Evaluator-side.

``prepare(seed, connector)`` runs the injected insertion and hands back everything a
source's ``collect`` needs: the claim, the params and a ``StationView``. It also hands back
the station itself and the outcome with ground truth, which stay on the evaluator's side of
the wall. The zoo features run, the smoke tests and the demo all go through here, so every
source sees a state the same way.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..evidence.base import FrameStore, StationView
from ..schemas import Claim, ClaimContext, Predicate
from ..station.mj_plug.inject import ARM, ErrorSpec, InsertOutcome, sample_error, scripted_insert
from ..station.mj_plug.station import MjPlugStation
from ..station.mj_plug.truth import truth


@dataclass
class Prepared:
    station: MjPlugStation        # privileged. Never pass this to a source
    outcome: InsertOutcome        # privileged: holds ground truth
    view: StationView             # what a source gets
    claim: Claim
    params: dict

    def label(self) -> bool:
        """Is the claimed predicate true right now? Ground truth, evaluator only."""
        tr = truth(self.station, self.outcome.target_port, release_ref=self.outcome.release_ref)
        return bool(tr["predicates"][self.claim.predicate.name])

    def truth(self) -> dict:
        return truth(self.station, self.outcome.target_port, release_ref=self.outcome.release_ref)

    def fresh_params(self) -> dict:
        """A copy of the params with its own frame store holding the claim-time frames."""
        out = dict(self.params)
        store = FrameStore()
        for key in self.params["frames"].keys("claim/"):
            store.put(key, self.params["frames"].get(key))
        out["frames"] = store
        return out

    def close(self) -> None:
        self.station.close()


def make_claim(outcome: InsertOutcome) -> Claim:
    return Claim(
        claim_id=f"zoo_{outcome.connector}_{outcome.seed}", run_id="zoo", cycle_id=f"seed{outcome.seed}", step_id="insert",
        predicate=Predicate(name=outcome.claim_predicate, target={"plug": "p1", "port": outcome.target_port}),
        raised_by="skill", context=ClaimContext(task="plug_cycle", stage="insert", attempt=0),
    )


def insert_record(outcome: InsertOutcome) -> dict:
    """What the insertion skill itself recorded. Nothing privileged."""
    res = outcome.insert
    return {
        "flag": bool(outcome.flag), "stopped_by": res.stopped_by, "travel_mm": res.travel_m * 1000.0,
        "expected_travel_mm": outcome.expected_travel_m * 1000.0, "force_at_stop_n": res.force_at_stop_n,
        "peak_force_n": res.peak_force_n, "seconds": res.duration_s, "trace": res.trace,
    }


def prepare(seed: int, connector: str, cls: str | None = None, spec: ErrorSpec | None = None, *,
            noise: bool = True, render: bool = True, profile: str = "easy") -> Prepared:
    """Inject a state and wrap it for the evidence sources.

    ``cls`` forces a class (for tests on dev seeds). Without it the seed's own class is used,
    which is what the zoo does. ``profile`` is ``easy`` or ``hard`` (S4b).
    """
    if spec is None:
        spec = sample_error(seed, connector, cls, profile=profile)
    station, outcome = scripted_insert(seed, connector, spec, noise=noise, profile=profile)
    frames = FrameStore()
    if render:
        frames.put_observation("claim", station.observe())
    params = {
        "nominal": station.nominal(), "connector": connector, "arm": ARM,
        "insert": insert_record(outcome), "frames": frames,
    }
    return Prepared(station=station, outcome=outcome, view=StationView(station), claim=make_claim(outcome), params=params)
