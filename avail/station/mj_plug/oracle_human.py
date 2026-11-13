"""OracleHuman for the cable loop: a person in sim who answers from truth and can take over. Privileged.

One of the 3 places allowed to read truth (CLAUDE.md rule 10). It answers the 4 kinds of request a
fork can make (``avail.fork.human``):

- ``confirm``: is the step done? Yes or no from truth, wrong with probability ``human.human_error``.
- ``choose``: 1 of the fork's options. ``retry`` when truth says a retry of the step can fix it (a short
  plug in the right port, fingers that did not open, the wrong cable in the jaws), else ``takeover``.
- ``guide_marks``: a tap on the wrist image at the assigned port's center, with ``marks_sigma_px`` of
  scatter, which the skill turns back into a position on the board.
- ``takeover``: the person drives the robot from where it is and finishes the step. In sim the
  scripted correction below plays the teleop source (``avail.fork.handoff.TeleopSource``). Every
  frame between the handoff and the handback is tagged assisted in the cell's frame log.

The answer delay is log-normal with the configured median and spread. A takeover is charged
``human.fix_handle_s`` of handling on top of the robot time it took.
"""

from __future__ import annotations

import math

import numpy as np

from ...config import load_experiment
from ..base import Pose
from . import scene
from .cable_cell import ARM, CLEARANCE_M, EXTRA_TRAVEL_M, INSERT_SPEED, RELEASE_LIFT_M, CableCell, cell_config

PREDICATE_OF_STEP = {"pick": "held", "carry": "at_port", "insert": None, "release": "released_stable"}


class OracleHuman:
    """Answers from truth, and takes over by a scripted correction. Sim only."""

    assistant_id = "oracle_human"

    def __init__(self, cell: CableCell, seed: int):
        self.cell = cell
        exp = load_experiment()
        self.cfg = exp.human
        self.extra = cell_config()["human_extra"]
        self.teleop = cell_config()["teleop"]
        self.rng = np.random.default_rng([int(seed), sorted(exp.connectors).index(cell.connector), 7777])

    # ------------------------------------------------------------------ the Human protocol
    def ask(self, request):
        from ...fork.human import HumanAnswer

        ctx = request.context
        item, step = int(ctx["item"]), str(ctx["step"])
        predicate = str(ctx["predicate"])
        delay = float(self.cfg["answer_delay_median_s"]) * math.exp(float(self.cfg["answer_delay_sigma_log"]) * self.rng.normal())
        question = request.question
        if question == "confirm":
            done = self.cell.truth_of(item, predicate)
            if self.rng.random() < float(self.cfg["human_error"]):
                done = not done
            return HumanAnswer(answer="yes" if done else "no", truth_label=bool(done), correction_kind=None, correction={},
                               seconds_to_answer=delay, seconds_handling=float(self.extra["confirm_handling_s"]), handoff=None)
        if question == "choose":
            options = list(request.options)
            pick = "retry" if (self._retry_fixes(item, step) and "retry" in options) else None
            if pick is None:
                pick = "takeover" if "takeover" in options else options[0]
            return HumanAnswer(answer=pick, truth_label=None, correction_kind=None, correction={},
                               seconds_to_answer=delay, seconds_handling=0.0, handoff=None)
        if question == "guide_marks":
            mark = self._port_mark(item, request)
            return HumanAnswer(answer="marks", truth_label=None, correction_kind="marks", correction=mark,
                               seconds_to_answer=delay, seconds_handling=0.0, handoff=None)
        if question == "takeover":
            back = self.take_over(self.cell, request)
            # after the handback the person says whether the step is done now, from what they see: truth here.
            # What they saw when they took the arm (back.before) labels the robot's claim
            return HumanAnswer(answer="taken over", truth_label=bool(back.ok), correction_kind="takeover",
                               correction={"ok": back.ok, "note": back.note}, seconds_to_answer=delay,
                               seconds_handling=float(self.cfg["fix_handle_s"]),
                               handoff={"ok": back.ok, "robot_seconds": back.seconds, "note": back.note,
                                        "before": back.before})
        raise ValueError(f"OracleHuman cannot answer a {question!r} request")

    def _retry_fixes(self, item: int, step: str) -> bool:
        """Would the robot fix this by trying the step again? Truth decides."""
        cell = self.cell
        if step == "pick":
            return cell.physical != item and cell.holding()          # the wrong cable: putting it back and picking works
        if step == "release":
            return cell.physical == item and cell.holding()          # fingers that did not open
        if step == "insert" and cell.physical == item and cell.holding():
            from .truth import plug_metrics

            m = plug_metrics(cell.station)
            return bool(m["in_bore"] and m["nearest_port"] == cell.cables[item].port and m["tilt_deg"] <= 2.0
                        and item not in cell.latch_blocked_for)
        return False

    def _port_mark(self, item: int, request) -> dict:
        """Where a person taps on the claim-time wrist frame: the assigned port's center, with scatter."""
        from ...evidence.cable_tag import project

        cell = self.cell
        frames = (request.frames or {})
        frame = frames.get(f"{ARM}_camera_d405")
        port_pos, _ = scene.port_pose_world(cell.station._cfg, cell.station._lay, cell.cables[item].port)
        if frame is None:
            frame = cell.station.observe([f"{ARM}_camera_d405"]).frames[f"{ARM}_camera_d405"]
        uv = project(frame, port_pos)
        sigma = float(self.extra["marks_sigma_px"])
        if uv is None:
            return {"camera": f"{ARM}_camera_d405", "u": None, "v": None}
        uv = uv + self.rng.normal(0.0, sigma, size=2)
        return {"camera": f"{ARM}_camera_d405", "u": float(uv[0]), "v": float(uv[1]), "kind": "port_center"}

    # ------------------------------------------------------------------ the teleop source
    def take_over(self, cell: CableCell, request):
        """The person finishes the step from where the robot is. Returns a Handback."""
        from ...fork.handoff import Handback

        st = cell.station
        ctx = request.context
        item, step = int(ctx["item"]), str(ctx["step"])
        t0 = float(st._mj_data.time)
        exp = load_experiment()
        notes = []
        before = bool(cell.truth_of(item, str(ctx["predicate"])))    # what the person sees as they take the arm
        # the hands: the named cable straight in the jaws, a blocked latch cleared
        if cell.physical is not None and cell.physical != item and cell.holding():
            wrong = cell.physical
            st.set_gripper(ARM, 1.0)
            cell._hide_physical()
            cell._vis_to_slot(wrong)                      # the person puts the wrong cable back in its slot
            cell._record_scene()
            notes.append(f"put cable {wrong} back")
        straight = cell.physical == item and cell.holding() and cell.in_hand_tilt_deg() <= 2.0
        from .truth import plug_metrics

        if not straight:
            in_bore = cell.physical == item and plug_metrics(st)["in_bore"]
            if in_bore and cell.holding():
                st.move_relative(ARM, [0.0, 0.0, 0.025], "base", 0.01, cell.pull_out_force_n(), 0.03)
            st.set_gripper(ARM, 1.0)
            if cell.physical != item:
                cell._make_physical(item, "placed" if cell.cables[item].where == "placed" else "slot")
            cell._attach_in_hand(0.0)                     # the person hands it to the gripper, straight
            notes.append("handed the cable to the gripper")
        cable = cell.cables[item]
        if step in ("carry", "insert", "release") and plug_metrics(st)["in_bore"]:
            st.move_relative(ARM, [0.0, 0.0, 0.025], "base", 0.01, cell.pull_out_force_n(), 0.03)
        # the plug is out of the port before the latch is cleared: cleared with the plug in, the latch holds it there.
        # On dev1 the person cleared it 1st, and in 5 of 8 such takeovers could not pull the plug out and missed the port
        if item in cell.latch_blocked_for:
            cell.latch_blocked_for.discard(item)
            st._retention.latch_blocked = False
            notes.append("cleared the latch")
        if step in ("carry", "insert", "release"):
            port_pos, yaw = scene.port_pose_world(st._cfg, st._lay, cable.port)
            cell._carry_plug_to(port_pos, yaw, CLEARANCE_M)
            cell.hover_xy[item] = (float(port_pos[0]), float(port_pos[1]))
            notes.append("drove it over its port")
        if step in ("insert", "release"):
            cap = float(self.teleop["seat_force_x_insertion_peak"]) * exp.connector(cell.connector).insertion_peak_n
            depth = st._lay.geom.full_depth
            travel = CLEARANCE_M + depth + EXTRA_TRAVEL_M
            st.move_relative(ARM, [0.0, 0.0, -travel], "base", INSERT_SPEED, cap, travel)
            st.wait(0.3)
            notes.append("pushed it home")
        if step == "release":
            st.set_gripper(ARM, 1.0)
            cell.release_ref[item] = plug_metrics(st)["plug_pos"]
            pose = st.robot_state().arms[ARM].tool_pose
            st.move_to_pose(ARM, Pose((pose.pos[0], pose.pos[1], pose.pos[2] + RELEASE_LIFT_M), pose.quat), 0.03)
            st.wait(exp.predicates["released_stable"]["settle_s"])
            notes.append("let go")
        predicate = ctx["predicate"]
        ok = cell.truth_of(item, predicate)
        return Handback(ok=bool(ok), seconds=float(st._mj_data.time) - t0, note=", ".join(notes), before=before)
