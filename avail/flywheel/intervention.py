"""1 intervention record, 3 uses: a calibration label, a correction demo and a shift metric.

- :func:`as_label` gives the calibration row of docs/06: the predicate, the evidence features at
  the fork, the person's label, the source ``assist`` and a sampling probability of 1, because
  every case that reaches a person is labeled.
- :func:`as_demo` gives the frames of a takeover from the cycle's frame log, from 2 s before the
  handoff to the handback, with the handoff and handback frames marked. That is the shape
  HG-DAgger style training wants: a person's correction in a state the robot reached by itself.
- :func:`as_metric` gives what the shift card counts: the question, the seconds and whether the
  person took over.

Each takes an ``Intervention`` record, or a dict that reads as one.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ..schemas import Intervention

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["LABEL_SOURCE", "PRE_ROLL_S", "TAKEOVER_OUTCOMES", "as_demo", "as_label", "as_metric", "demo_window",
           "is_takeover"]

PRE_ROLL_S = 2.0
"""How much of the robot's own run before a handoff a demo keeps, in seconds of the frame log's clock."""

LABEL_SOURCE = "assist"
TAKEOVER_OUTCOMES = ("taken_over_done", "taken_over_failed")


def _record(iv: Intervention | Mapping[str, Any]) -> Intervention:
    if isinstance(iv, Intervention):
        return iv
    if isinstance(iv, Mapping):
        return Intervention.model_validate(iv)
    raise TypeError(f"got a {type(iv).__name__}. Fix: pass an Intervention record or its payload dict")


def is_takeover(iv: Intervention | Mapping[str, Any]) -> bool:
    """True when the person took control: a takeover question, a takeover outcome or a handoff frame."""
    iv = _record(iv)
    return iv.question == "takeover" or iv.outcome in TAKEOVER_OUTCOMES or iv.handoff_frame is not None


def as_label(iv: Intervention | Mapping[str, Any], *, predicate: str | None = None,
             witness_version: str | None = None) -> dict[str, Any] | None:
    """The calibration row of 1 intervention, or None when the person's answer is not a label.

    ``predicate`` and ``witness_version`` fill in what the record does not carry, for example from
    the claim and the verdict in the ledger. ``predicate`` is the record's ``predicate_name`` when not
    given, and None when neither says.
    """
    iv = _record(iv)
    if iv.truth_label is None:
        return None
    return {
        "predicate": predicate or iv.predicate_name,
        "features": dict(iv.features),
        "label": iv.truth_label,
        "source": LABEL_SOURCE,
        "sampling_probability": 1.0,
        "witness_version": witness_version,
        "intervention_id": iv.intervention_id, "run_id": iv.run_id, "cycle_id": iv.cycle_id, "item": iv.item,
        "step": iv.step, "claim_id": iv.claim_id, "verdict_id": iv.verdict_id, "question": iv.question,
        "trigger": iv.trigger, "ts": iv.answered_ts,
    }


def demo_window(iv: Intervention | Mapping[str, Any], frames: pd.DataFrame,
                pre_roll_s: float = PRE_ROLL_S) -> tuple[pd.DataFrame | None, str]:
    """(the demo frames, "") for a takeover, or (None, why there is no demo).

    The frames run from ``pre_roll_s`` before the handoff frame, on the frame log's own clock
    ``t``, to the handback frame, in frame order. ``is_handoff`` and ``is_handback`` mark the 2 ends.
    """
    iv = _record(iv)
    if not is_takeover(iv):
        return None, "not a takeover"
    if iv.handoff_frame is None or iv.handback_frame is None:
        return None, "the takeover has no handoff frame and handback frame"
    missing = [column for column in ("frame_index", "t") if column not in frames.columns]
    if missing:
        return None, f"the frame log has no {' and no '.join(missing)} column"
    index = frames["frame_index"]
    at_handoff = frames.loc[index == iv.handoff_frame, "t"]
    if at_handoff.empty:
        return None, f"the handoff frame {iv.handoff_frame} is not in the frame log"
    if not bool((index == iv.handback_frame).any()):
        return None, f"the handback frame {iv.handback_frame} is not in the frame log"
    start = float(at_handoff.iloc[0]) - float(pre_roll_s)
    keep = (index <= iv.handback_frame) & ((index >= iv.handoff_frame) | (frames["t"] >= start - 1e-9))
    out = frames.loc[keep].sort_values("frame_index", kind="stable").reset_index(drop=True)
    out["is_handoff"] = out["frame_index"] == iv.handoff_frame
    out["is_handback"] = out["frame_index"] == iv.handback_frame
    return out, ""


def as_demo(iv: Intervention | Mapping[str, Any], frames_df: pd.DataFrame,
            pre_roll_s: float = PRE_ROLL_S) -> pd.DataFrame | None:
    """The frames of a takeover with ``is_handoff`` and ``is_handback``, or None (see :func:`demo_window`)."""
    return demo_window(iv, frames_df, pre_roll_s)[0]


def as_metric(iv: Intervention | Mapping[str, Any]) -> dict[str, Any]:
    """What the shift card counts for 1 intervention."""
    iv = _record(iv)
    return {
        "intervention_id": iv.intervention_id, "cycle_id": iv.cycle_id, "item": iv.item, "step": iv.step,
        "question": iv.question, "trigger": iv.trigger, "outcome": iv.outcome,
        "seconds_waiting": iv.seconds_waiting, "seconds_handling": iv.seconds_handling,
        "takeover": is_takeover(iv), "labeled": iv.truth_label is not None,
    }
