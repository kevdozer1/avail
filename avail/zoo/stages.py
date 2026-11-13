"""The plug station's cascade: its stages, the zoo columns they read, and the features its scorers leave out.

Testbed knowledge, kept out of the core since S5. ``avail.witness.cascade`` takes the stage
definitions, the skipped features and the station id as arguments, and a stored witness
carries its own, so a witness file is applied with the stages it was calibrated with.
"""

from __future__ import annotations

from ..witness.cascade import StageDef

# the id the simulated plug station reports (MjPlugStation.station_id). Written out here so
# the analysis need not import the simulator to name the station its witnesses belong to
STATION_ID = "mj_plug"

# feature names that are bookkeeping and say nothing about the plug: the port the tool pose
# points at is an index, and a scorer must not read it as a measurement
SKIP_FEATURES: tuple[str, ...] = ("port_index_est",)

PLUG_STAGES: tuple[StageDef, ...] = (
    StageDef("passive", ("passive.skill_flag", "passive.pose_residual", "passive.jaw_state", "passive.click",
                         "vision.gap@claim"), "A", "y0"),
    StageDef("push", ("probe.push_seat",), "B", "B/probe.push_seat/_y"),
    StageDef("tug", ("probe.tug",), "B", "B/probe.tug/_y"),
    StageDef("look", ("vision.gap@now",), "B", "B/vision.gap@now/_y"),
)
