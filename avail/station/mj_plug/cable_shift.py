"""The cable loop's testbed module for ``avail.shift.runner``: cells, witnesses, the person, the judge, the evaluator.

Simulator side. The task file ``tasks/cable_loop.yaml`` names this module as its testbed. The core
runner imports it by that name and calls the functions below, so the core never imports the station.

Witnesses. ``seated`` and ``latched`` use E2's cascade, fit on the cable loop's own insertions of the
``cable_fit`` seeds and calibrated on those of the ``cable_calibration`` seeds (``cable_zoo.py``, S5b)
at the customer bar's alpha (``customer_bar.alpha_for_bar``, 0.01) and ``bounds.beta``. It is built once
and saved under ``AVAIL_DATA/witness_e3``. No test seed enters it. Its stages are E2's without the camera:
passive signals (the skill's flag, the tool pose, the gripper, the click), then the push, then the
tug. On dev cycles the wrist camera lost the plug at its port in 4 of 15 insertions, each with a
cable already seated in a neighboring port, which hides the socket's near wall from this camera.
The zoo's board never had a neighbor, so its camera check was calibrated on views a full board does
not give. S4 measured this same cascade without the camera doing as well as with it on the zoo.
``held``, ``at_port`` and ``released_stable`` use the task file's rule witnesses.
"""

from __future__ import annotations

from functools import lru_cache

from ...config import load_experiment
from ...paths import data_root
from .cable_cell import CableCell

E3_WITNESS_DIR = "witness_e3"


def cable_stages():
    """E2's stages without the camera, as the cable loop uses them."""
    from ...witness.cascade import StageDef

    return (
        StageDef("passive", ("passive.skill_flag", "passive.pose_residual", "passive.jaw_state", "passive.click"), "A", "y0"),
        StageDef("push", ("probe.push_seat",), "B", "B/probe.push_seat/_y"),
        StageDef("tug", ("probe.tug",), "B", "B/probe.tug/_y"),
    )


def make_cell(seed: int, variant: str, profile: str = "easy") -> CableCell:
    return CableCell(seed, variant, profile=profile)


def judge_reference_files() -> dict:
    """The picture of a seated plug that the E3 judge's prompt shows beside each claim, per E3 variant: easy mode, dev
    seed 3000, under AVAIL_DATA/cache/reference. Empty when config runs no vlm arm or its prompt shows no reference."""
    from ...zoo.judges import reference_path

    e3 = load_experiment().e3
    if "vlm" not in e3.get("arms", []) or dict(e3.get("vlm_judge", {})).get("prompt", "crop_ref") != "crop_ref":
        return {}
    return {v: reference_path(v, "easy", create=False) for v in e3.get("variants", [])}


def station_files() -> list:
    """The station's robot model a cycle loads: station.xml, the meshes it names (its assets folder) and the profile with
    the ready joints, under third_party/ASPIRE, which git does not track. Only the files that exist."""
    from ...paths import aspire_station_profile, aspire_station_xml

    xml = aspire_station_xml()
    assets = xml.parent / "assets"
    files = [xml, aspire_station_profile(), *(sorted(assets.glob("*")) if assets.is_dir() else [])]
    return [f for f in files if f.is_file()]


def station_problem() -> str | None:
    """Why a final run could not build the station, or None: its robot model is missing. git does not track
    third_party, so a worktree of a commit lacks it until its third_party is linked to the repo's."""
    from ...paths import aspire_station_profile, aspire_station_xml

    missing = [str(p) for p in (aspire_station_xml(), aspire_station_profile()) if not p.is_file()]
    if missing:
        return (f"the station's robot model is missing ({', '.join(missing)}). Run uv run python scripts/setup_aspire.py, "
                "or in a worktree link its third_party folder to the repo's")
    return None


def setup_files() -> list:
    """The files besides config/experiment.yaml that set what the cycles do: the cell's failure profile, the taught
    palette, the model table that holds the judge's entry (config/models.yaml, hashed whole), the witness each E3
    variant loads for its insertion (the latest version under AVAIL_DATA/witness_e3), the judge's reference picture
    of each and the station's robot model. The E3 command records their hashes at its 1st run and checks them before
    each arm, and the shift runner checks them before each cycle."""
    from ...analysis.run import primary_predicate
    from ...paths import REPO_ROOT
    from ...witness import store
    from .cable_cell import CELL_YAML
    from .cable_teach import PALETTE_FILE

    files = [CELL_YAML, PALETTE_FILE, REPO_ROOT / "config" / "models.yaml"]
    root = data_root() / E3_WITNESS_DIR
    for variant in load_experiment().e3.get("variants", []):
        predicate = primary_predicate(variant)
        have = store.versions(variant, predicate, root=root) if (root / variant / predicate).is_dir() else []
        if have:
            files.append(store.witness_dir(variant, predicate, root) / f"v{max(have)}.json")
    files += [path for path in judge_reference_files().values() if path.is_file()]
    return files + station_files()


CALIBRATION_SPLIT = "cable_calibration"            # S5b: the witness is calibrated on the cable loop's own insertions
FIT_SPLIT = "cable_fit"


def _witness_matches(cascade, alpha: float) -> bool:
    """Whether a stored cascade is the one ``e3_cascade`` asks for: the same alpha, the cable loop's stages and the
    cable loop's own calibration split (a witness calibrated on the zoo's 1-plug states is stale since S5b)."""
    return (abs(float(cascade.alpha) - float(alpha)) < 1e-12
            and [d.name for d in cascade.stage_defs] == [d.name for d in cable_stages()]
            and dict(cascade.calibration or {}).get("split_id") == CALIBRATION_SPLIT)


def prepare_problem(variants, alpha: float) -> str | None:
    """Why a final run would find something ``avail shift prepare`` makes missing or stale, or None: a variant's witness
    (missing, or built for another alpha or other stages) or its judge reference picture. It reads and creates no
    folder.

    A final run builds neither: ``e3_cascade`` and the judge refuse to. 1 missing at E3's 1st arm would stop the run at
    the 1st arm that needs it, after an earlier arm had read the split, and making it then would change the setup the
    E3 record holds. So the E3 command asks this before any seed is read, and adds what to do, which depends on
    whether E3 has started."""
    from ...analysis.run import primary_predicate
    from ...witness import store

    root = data_root() / E3_WITNESS_DIR
    references = judge_reference_files()
    found = []
    for variant in variants:
        predicate = primary_predicate(variant)
        have = store.versions(variant, predicate, root=root) if (root / variant / predicate).is_dir() else []
        if not have:
            found.append(f"{variant} has no {predicate} witness under {root}")
        else:
            cascade, _ = store.load(variant, predicate, root=root)
            if not _witness_matches(cascade, alpha):
                found.append(f"the latest {variant} {predicate} witness (v{max(have)}) was built for another alpha, "
                             f"other stages or another calibration split than the run asks for (alpha {alpha}, stages "
                             f"{', '.join(d.name for d in cable_stages())}, split {CALIBRATION_SPLIT})")
        path = references.get(variant)
        if path is not None and not path.is_file():
            found.append(f"{variant} has no judge reference picture ({path})")
    return ". ".join(found) if found else None


def prepare(task, variant: str, alpha: float) -> dict:
    """What ``avail shift prepare`` builds for 1 variant, outside any final run: its witnesses, and the judge's
    reference picture when the E3 judge's prompt shows one (dev seed 3000, easy mode)."""
    from ...zoo.judges import reference_image

    made = {p: type(w).__name__ for p, w in make_witnesses(task, variant, alpha).items()}
    path = judge_reference_files().get(variant)
    if path is not None:
        reference_image(variant, profile="easy")
        made["judge reference"] = str(path)
    return made


@lru_cache(maxsize=4)
def e3_cascade(variant: str, alpha: float):
    """The cascade for ``seated`` (fric) or ``latched`` (latch) at the customer bar's alpha. Built once, then read."""
    from ...analysis import e1
    from ...analysis.run import primary_predicate
    from ...witness import store
    from ...witness.cascade import Cascade

    from ...witness.cascade import calibrate_cascade, fit_scorers
    from ...zoo.stages import SKIP_FEATURES, STATION_ID

    predicate = primary_predicate(variant)
    root = data_root(create=True) / E3_WITNESS_DIR
    stages = cable_stages()
    names = [d.name for d in stages]
    try:
        _, payload = store.load(variant, predicate, root=root)
        cascade = Cascade.from_dict(payload["cascade"])
        if _witness_matches(cascade, alpha):
            return cascade
    except FileNotFoundError:
        pass
    import os

    from ...seeds import FINAL_RUN_ENV

    if os.environ.get(FINAL_RUN_ENV):                 # a final run reads its witness and never fits or publishes one
        raise RuntimeError(f"no {E3_WITNESS_DIR} witness for {variant} {predicate} at alpha {alpha} with stages {names}, "
                           f"and a final run is open. Run `avail shift prepare --variant {variant}` before the final run")
    beta = load_experiment().bounds.beta
    fit = e1.main_slice(cable_table(variant, FIT_SPLIT), predicate)
    cal = e1.main_slice(cable_table(variant, CALIBRATION_SPLIT), predicate)
    scorers, neg_share, pos_share = fit_scorers(fit, variant, predicate, stages, SKIP_FEATURES)
    cascade = calibrate_cascade(scorers, neg_share, pos_share, cal, variant, predicate, alpha, beta, stages,
                                station_id=STATION_ID, skip_features=SKIP_FEATURES, split_id=CALIBRATION_SPLIT)
    provenance = {"purpose": "S5b and E3: the cascade without the camera, at the customer bar's alpha, on the cable "
                             "loop's own insertions", "alpha": alpha, "beta": beta, "stages": names,
                  "fit": f"{FIT_SPLIT} ({len(fit)} insertion claims)",
                  "calibration": f"{CALIBRATION_SPLIT} ({len(cal)} insertion claims)"}
    _, cascade = store.publish(cascade, checks={}, provenance=provenance, root=root)
    return cascade


def cable_table(variant: str, split: str):
    """The cable loop's insertion claims of a split (``avail.station.mj_plug.cable_zoo``), checked to hold only that
    split's seeds."""
    from ...zoo import store

    table = store.load(variant, split, "features")
    store.check_seeds_in_split(table["seed"], split)
    return table


def make_witnesses(task, variant: str, alpha: float) -> dict:
    from ...witness.rules import RuleWitness

    out = {}
    for step, predicate in task.required(variant):
        spec = task.predicates[predicate]
        if spec.decide.kind == "witness":
            out[predicate] = e3_cascade(variant, float(alpha))
        else:
            out[predicate] = RuleWitness.from_predicate(spec, version=f"{task.task}-rules")
    return out


def make_human(cell: CableCell, seed: int):
    from .oracle_human import OracleHuman

    return OracleHuman(cell, seed)


class _Judge:
    """The vlm arm's judge: the evidence source ``judge.vlm`` with the model and prompt settings filled in."""

    def __init__(self, model: str, variant: str, session: str):
        from ...evidence import base
        from ...router import Router

        self.source = base.get("judge.vlm")
        self.model, self.variant = model, variant
        self.router = Router(session=session, approved=True, phase="shift_vlm")
        self._refs: dict = {}

    def collect(self, view, claim, params):
        import os

        from ...seeds import FINAL_RUN_ENV
        from ...zoo.judges import reference_image, reference_path

        connector = params["connector"]
        if connector not in self._refs and self.variant == "crop_ref":
            # a final run reads its reference picture and never builds one, as it never builds a witness
            if os.environ.get(FINAL_RUN_ENV) and not reference_path(connector, "easy", create=False).is_file():
                raise RuntimeError(f"no judge reference picture for {connector}, and a final run is open. Run "
                                   f"`avail shift prepare --variant {connector}` before the final run")
            self._refs[connector] = reference_image(connector, profile="easy")
        call = dict(params)
        call["judge"] = {"model": self.model, "variant": self.variant, "router": self.router,
                         "reference": self._refs.get(connector)}
        return self.source.collect(view, claim, call)


def judge_problem() -> str | None:
    """Why the E3 judge could not be called, or None: its model is not in config/models.yaml, or the router would refuse
    it now (no key in the environment or .env, an entry not verified). The vlm arm runs last, hours after E3's 1st read,
    so the E3 command asks this before any seed is read."""
    from ...router.models import default_models
    from ...router.smoke import skip_reason

    model = dict(load_experiment().e3.get("vlm_judge", {})).get("model", "fable")
    entry = default_models().models.get(model)
    if entry is None:
        return f"config/models.yaml has no model {model}, which e3.vlm_judge names"
    reason = skip_reason(entry)
    return f"the E3 judge {model} cannot be called now: {reason}" if reason else None


def make_judge(model: str | None = None, variant: str | None = None, session: str = "S5_vlm_arm"):
    """The vlm arm's judge. Its model and prompt come from config (``e3.vlm_judge``), which the seed guard
    hashes, so an E3 run cannot change its judge between arms."""
    judge = dict(load_experiment().e3.get("vlm_judge", {}))
    return _Judge(model or judge.get("model", "fable"), variant or judge.get("prompt", "crop_ref"), session)


def evaluator(cell: CableCell):
    """The evaluator's per-step truth, for the shift report only: was this predicate of this cable true now?"""
    return cell.truth_of


def evaluate(cell: CableCell) -> dict:
    """The evaluator's truth at the end of the cycle: did each cable end seated (and latched), let go and still?"""
    return {k: bool(cell.item_success(k)) for k in range(cell.n_items)}


class ToolCell:
    """The robot side of the tool server over a cable cell: ASPIRE's CAP tools and ``wam_act``.

    ``wam_act`` runs 1 stand-in skill and tells the AVAIL core what it returned, so ``avail_verify``
    can hand the skill's own record to the checks that read it. Pixels are not sent."""

    arms = ("right", "left")
    cameras = ("top_camera_d405", "left_camera_d405", "right_camera_d405")

    def __init__(self, cell: CableCell, core):
        from .cable_cell import STEPS

        self.cell, self.core = cell, core
        self.skills = list(STEPS)
        self.steps = list(STEPS)

    def describe(self) -> dict:
        return {"variant": self.cell.variant, "items": [str(i) for i in range(self.cell.n_items)],
                "ports": {str(i): self.cell.port_of(i) for i in range(self.cell.n_items)}}

    def observe(self, cameras=None) -> dict:
        obs = self.cell.station.observe(list(cameras) if cameras else None)
        return {"frames": {n: {"intrinsics": f.intrinsics.tolist(), "extrinsics": f.extrinsics.tolist(), "ts": f.ts,
                               "shape": list(f.rgb.shape)} for n, f in obs.frames.items()}}

    def robot_state(self) -> dict:
        rs = self.cell.station.robot_state()
        return {"arms": {a: {"joint_pos": s.joint_pos.tolist(), "gripper_pos": s.gripper_pos,
                             "tool_pose": {"position": list(s.tool_pose.pos), "quaternion_wxyz": list(s.tool_pose.quat)},
                             "tool_force_est": s.tool_force_est.tolist()} for a, s in rs.arms.items()}, "ts": rs.ts}

    def move_to_pose(self, target_pose: dict, arm: str) -> dict:
        from ..base import Pose

        pose = Pose(tuple(target_pose["position"]), tuple(target_pose["quaternion_wxyz"]))
        res = self.cell.station.move_to_pose(arm, pose, 0.05)
        return {"ok": bool(res.ok), "stopped_by": res.stopped_by, "travel_m": res.travel_m, "note": res.note}

    def set_gripper(self, position: float, arm: str) -> dict:
        res = self.cell.station.set_gripper(arm, float(position))
        return {"ok": bool(res.ok), "stopped_by": res.stopped_by}

    def run_skill(self, skill: str, item, port=None) -> dict:
        i = int(item)
        res = self.cell.run_skill(skill, i, self.core.attempt_of(i, skill), marks=self.core.take_marks(i, skill))
        self.core.skill_ran(i, skill, res)
        return {"ok": True, "flag": bool(res.flag), "seconds": float(res.seconds), "note": res.note}

    def stop(self) -> dict:
        self.cell.station.stop()
        return {"ok": True}


def tool_session(task_name: str, variant: str, seeds: list[int], *, shift: str, run_root, early_done_after: int = 1) -> dict:
    """Cycles run by the scripted agent through the tool server alone, 1 ledger per cycle. Dev seeds only.

    Each cycle also calls ``avail_done`` early on purpose, after its 2nd cable, which the gate must refuse."""
    import json
    from pathlib import Path

    from ...agent.mcp_server import ToolServer
    from ...agent.scripted_agent import run_cycle
    from ...core.task import load_task
    from ...ledger.writer import LedgerWriter
    from ...seeds import guard
    from ...shift.driver import ItemLog
    from ...shift.runner import _cycle_end
    from ...shift.toolcore import ToolCore

    guard(seeds, purpose=f"tool session {shift}, {task_name} {variant}")
    task = load_task(task_name)
    alpha = float(load_experiment().customer_bar["alpha_for_bar"])
    out = []
    for seed in seeds:
        run_id = f"{shift}_tools_{variant}_c{seed}"
        run_dir = Path(run_root) / run_id
        cell = make_cell(seed, variant)
        try:
            with LedgerWriter(run_id, folder=run_dir) as ledger:
                core = ToolCore(task, cell, "avail_human", witnesses=make_witnesses(task, variant, alpha),
                                human=make_human(cell, seed), ledger=ledger, run_id=run_id, cycle_id=f"{run_id}",
                                evaluator=evaluator(cell))
                server = ToolServer(ToolCell(cell, core), core, log=ledger.append)
                result = run_cycle(server, early_done_after=early_done_after)
                core.log.items = [core.items.get(i) or ItemLog(i) for i in range(cell.n_items)]
                steps_of = {s.claim_for(variant): s.name for s in task.steps}
                for it in core.log.items:
                    ok, open_predicates = core.gate.done(it.item)
                    it.declared_done = bool(ok)
                    # the 1st step the gate still holds open, as the driver records it, so the card counts it
                    it.unresolved = None if ok else steps_of.get(open_predicates[0], open_predicates[0])
                _cycle_end(ledger, core, core.log, evaluate(cell), seed, "avail_human_tools", variant, task)
            frames = frames_table(cell)
            (run_dir / "frames").mkdir(parents=True, exist_ok=True)
            frames.to_parquet(run_dir / "frames" / f"{run_id}.parquet", index=False)
            (run_dir / "frames" / f"{run_id}.scene.json").write_text(json.dumps(scene_events(cell)), encoding="utf-8")
            out.append({"seed": seed, "early_done": result["early"], "final_done": result["final"], "calls": result["calls"]})
        finally:
            close(cell)
    return {"shift": shift, "cycles": out}


def frames_table(cell: CableCell):
    import pandas as pd

    return pd.DataFrame(cell.frames)


def scene_events(cell: CableCell) -> list:
    return cell.scene_events


def close(cell: CableCell) -> None:
    cell.close()
