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
WITNESS_ROOT_ENV = "AVAIL_WITNESS_ROOT"         # 2026-11-16 (docs/07, step 5): a flywheel day's own witness folder


def witness_root(create: bool = False):
    """Where the cable loop's witnesses are read from: AVAIL_DATA/witness_e3, or the folder ``AVAIL_WITNESS_ROOT`` names
    for a dev run of the flywheel. A final run refuses the override, and nothing is ever built under it."""
    import os
    from pathlib import Path

    from ...seeds import FINAL_RUN_ENV

    override = os.environ.get(WITNESS_ROOT_ENV, "").strip()
    if override:
        if os.environ.get(FINAL_RUN_ENV):
            raise RuntimeError(f"{WITNESS_ROOT_ENV} is set inside the final run {os.environ[FINAL_RUN_ENV]}. A final run "
                               f"reads AVAIL_DATA/{E3_WITNESS_DIR} alone")
        return Path(override)
    return data_root(create=create) / E3_WITNESS_DIR


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
    root = witness_root()
    for variant in load_experiment().e3.get("variants", []):
        for predicate in (primary_predicate(variant), *CHECKED):       # the insertion's and the S5c fix's witnesses
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

    root = witness_root()
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
        for checked in CHECKED:                        # S5c fix: the witnesses that replaced the hand-set rules
            have = store.versions(variant, checked, root=root) if (root / variant / checked).is_dir() else []
            if not have:
                found.append(f"{variant} has no calibrated {checked} witness under {root}")
            elif not _check_matches(store.load(variant, checked, root=root)[0], checked, alpha):
                found.append(f"the latest {variant} {checked} witness (v{max(have)}) was built for another alpha, other "
                             f"stages or another calibration split (alpha {alpha}, split {CHECK_CALIBRATION_SPLIT})")
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
    root = witness_root(create=True)
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

    if os.environ.get(FINAL_RUN_ENV) or os.environ.get(WITNESS_ROOT_ENV):   # a final run and a flywheel day only read
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


# ----------------------------------------------------------------------------- the checks that were hand-set (S5c fix)
CHECK_FIT_SPLIT, CHECK_CALIBRATION_SPLIT = "check_fit", "check_calibration"


def check_stages(predicate: str):
    """The stages of the calibrated witness for ``held``, ``at_port`` or ``released_stable``. ``at_port`` also reads the
    hover pose against the board's offset as the robot measured it (``passive.hover_offset``). ``released_stable``:
    the jaws, the release record and the lift's forces, then the camera after the retreat, then a closer look from an
    inspection pose by the port (``vision.release_relook``, the only source here that moves the arm, touching nothing)."""
    from ...witness.cascade import StageDef

    if predicate == "held":
        return (StageDef("passive", ("passive.jaw_hold", "vision.cable_tag"), "A", "y0"),)
    if predicate == "at_port":
        return (StageDef("passive", ("passive.hover_pose", "passive.hover_offset", "passive.jaw_hold", "vision.cable_tag"),
                         "A", "y0"),)
    if predicate == "released_stable":
        # 2026-11-14, fix 2: the passive signals and the camera at the end of the lift, then the re-grip pull. The relook
        # (2026-11-13) left the witness: it moved the arm before the pull and barely changed what the camera read
        return (StageDef("passive", ("passive.jaw_hold", "passive.release_record", "passive.release_lift", "vision.gap@now"),
                         "A", "y0"),
                StageDef("tug", ("probe.release_tug",), "B", "B/probe.release_tug/_y"))
    raise KeyError(f"no calibrated check for {predicate!r}")


CHECKED = ("held", "at_port", "released_stable")


def check_settings() -> dict:
    """The check witnesses' alphas, crossing rule and design date from config (`check_witness`, docs/07 2026-11-16).
    Empty when the config names none, and then each witness is built as E3b's were."""
    return dict(load_experiment().check_witness or {})


def check_alpha(predicate: str, alpha: float) -> float:
    """A check witness's alpha: its own from config when it has one, else the customer bar's."""
    return float(dict(check_settings().get("alphas", {})).get(predicate, alpha))


def _check_matches(cascade, predicate: str, alpha: float) -> bool:
    s = check_settings()
    return (abs(float(cascade.alpha) - float(check_alpha(predicate, alpha))) < 1e-12
            and [d.name for d in cascade.stage_defs] == [d.name for d in check_stages(predicate)]
            and [list(d.sources) for d in cascade.stage_defs] == [list(d.sources) for d in check_stages(predicate)]
            and dict(cascade.calibration or {}).get("split_id") == CHECK_CALIBRATION_SPLIT
            and getattr(cascade, "crossing", "abstain") == s.get("crossing", "abstain")
            and dict(cascade.calibration or {}).get("design") == s.get("design"))


#: the enrichment of released_stable (docs/07, 2026-11-13): the release's own failure injected on its own seed ranges,
#: joined to the natural rows of its table. E3b's seeds are never enriched
ENRICHED = {("released_stable", CHECK_FIT_SPLIT): "check_fit_drag",
            ("released_stable", CHECK_CALIBRATION_SPLIT): "check_calibration_drag"}


def check_table(variant: str, split: str, predicate: str):
    """The claims of 1 predicate in a check split (``check_zoo.py``), checked to hold only that split's seeds, with
    the predicate's enrichment rows when it has them."""
    import pandas as pd

    from ...zoo import store

    parts = []
    for name in (split, ENRICHED.get((predicate, split))):
        if name is None:
            continue
        table = store.load(variant, name, "features")
        store.check_seeds_in_split(table["seed"], name)
        parts.append(table[table["claim_predicate"] == predicate])
    out = pd.concat(parts, ignore_index=True)
    if predicate == "released_stable":
        # 2026-11-16 (docs/07, step 1): the pull's held_at_cap and force_frac, functions of what the pull stored
        from ...evidence.release_tug import derive

        out = derive(out, variant)
    return out


@lru_cache(maxsize=8)
def check_cascade(variant: str, predicate: str, alpha: float):
    """The calibrated witness for a hand-set predicate, at the customer bar's alpha. Built once, then read."""
    import os

    from ...seeds import FINAL_RUN_ENV
    from ...witness import store
    from ...witness.cascade import Cascade, calibrate_cascade, fit_scorers
    from ...zoo.stages import SKIP_FEATURES, STATION_ID

    root = witness_root(create=True)
    stages = check_stages(predicate)
    settings = check_settings()
    alpha = check_alpha(predicate, alpha)
    try:
        _, payload = store.load(variant, predicate, root=root)
        cascade = Cascade.from_dict(payload["cascade"])
        if _check_matches(cascade, predicate, alpha):
            return cascade
    except FileNotFoundError:
        pass
    if os.environ.get(FINAL_RUN_ENV) or os.environ.get(WITNESS_ROOT_ENV):   # a final run and a flywheel day only read
        raise RuntimeError(f"no calibrated {predicate} witness for {variant} at alpha {alpha}, and a final run is open. "
                           f"Run `avail shift prepare --variant {variant}` before the final run")
    beta = load_experiment().bounds.beta
    fit = check_table(variant, CHECK_FIT_SPLIT, predicate)
    cal = check_table(variant, CHECK_CALIBRATION_SPLIT, predicate)
    scorers, neg_share, pos_share = fit_scorers(fit, variant, predicate, stages, SKIP_FEATURES)
    cascade = calibrate_cascade(scorers, neg_share, pos_share, cal, variant, predicate, alpha, beta, stages,
                                station_id=STATION_ID, skip_features=SKIP_FEATURES, split_id=CHECK_CALIBRATION_SPLIT,
                                crossing=str(settings.get("crossing", "abstain")))
    if settings.get("design"):
        cascade.calibration["design"] = settings["design"]
    provenance = {"purpose": f"S5c fix: {predicate} calibrated in place of the hand-set rule, after E3. "
                             f"Design {settings.get('design', 'S5c')}: alpha from the error budget, crossing "
                             f"{settings.get('crossing', 'abstain')}", "alpha": alpha,
                  "beta": beta, "stages": [d.name for d in stages],
                  "fit": f"{CHECK_FIT_SPLIT} ({len(fit)} claims, {int((~fit['y0'].astype(bool)).sum())} bad)",
                  "calibration": f"{CHECK_CALIBRATION_SPLIT} ({len(cal)} claims, {int((~cal['y0'].astype(bool)).sum())} bad)"}
    _, cascade = store.publish(cascade, checks={}, provenance=provenance, root=root)
    return cascade


def make_witnesses(task, variant: str, alpha: float) -> dict:
    from ...witness.rules import RuleWitness

    out = {}
    for step, predicate in task.required(variant):
        spec = task.predicates[predicate]
        if spec.decide.kind == "witness" and predicate in CHECKED:
            out[predicate] = check_cascade(variant, predicate, float(alpha))
        elif spec.decide.kind == "witness":
            out[predicate] = e3_cascade(variant, float(alpha))
        else:
            out[predicate] = RuleWitness.from_predicate(spec, version=f"{task.task}-rules")
    return out


def make_human(cell: CableCell, seed: int):
    """The simulated operator. A dev run can ask for one who is not perfect (docs/07, 2026-11-16, step 4) through
    ``AVAIL_SIMOP_OPERATOR``. A final run refuses that."""
    import os

    from ...seeds import FINAL_RUN_ENV
    from .imperfect_operator import ENV, ImperfectHuman, from_env
    from .oracle_human import OracleHuman

    spec = from_env()
    if spec is not None:
        if os.environ.get(FINAL_RUN_ENV):
            raise RuntimeError(f"{ENV} is set inside the final run {os.environ[FINAL_RUN_ENV]}: a final run's operator is "
                               "the configured one")
        return ImperfectHuman(cell, seed, *spec)
    return OracleHuman(cell, seed)


AWAY_FROM_PORT = ("pick",)                        # steps a judge's crop centers on the gripper instead of the port


def robot_readings(view, params: dict) -> list[str]:
    """The robot's own readings at a claim, in words a judge can use: the gripper reading and where the tool is
    against its taught pose with the plug seated in the target port. Robot state and taught geometry only."""
    import numpy as np

    arm = params.get("arm", "right")
    state = view.robot_state().arms[arm]
    nominal = params["nominal"]
    lines = [f"Gripper reading {float(state.gripper_pos):.2f}, on a scale from 0 (closed) to 1 (open). "
             f"Closed on a plug it reads about {nominal.expected_jaw_pos():.2f}."]
    port = params.get("port")
    if port:
        off = (np.asarray(state.tool_pose.pos, dtype=float) - np.asarray(nominal.seated_tool_pose(int(port)).pos)) * 1000
        lines.append(f"The tool is {float(off[2]):.1f} mm above, and {float(np.hypot(off[0], off[1])):.1f} mm to the side "
                     f"of, where it would be with the plug seated in the target port.")
    return lines


class _Judge:
    """The vlm arm's judge: the evidence source ``judge.vlm`` with the model and prompt settings filled in.
    ``with_state``: the prompt also carries the robot's readings at the claim (``robot_readings``)."""

    def __init__(self, model: str, variant: str, session: str, with_state: bool = False):
        from ...evidence import base
        from ...router import Router

        self.source = base.get("judge.vlm")
        self.model, self.variant, self.with_state = model, variant, bool(with_state)
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
        if self.with_state:
            call["judge"]["state"] = robot_readings(view, params)
        if params.get("step") in AWAY_FROM_PORT:      # the cable is in the tray, not at its port: crop around the jaws
            call["judge"]["crop_point"] = view.robot_state().arms[params.get("arm", "right")].tool_pose.pos
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


def make_judge(model: str | None = None, variant: str | None = None, session: str = "S5_vlm_arm",
               with_state: bool = False):
    """The vlm arm's judge. Its model and prompt come from config (``e3.vlm_judge``), which the seed guard
    hashes, so an E3 run cannot change its judge between arms."""
    judge = dict(load_experiment().e3.get("vlm_judge", {}))
    return _Judge(model or judge.get("model", "fable"), variant or judge.get("prompt", "crop_ref"), session, with_state)


FIG_PRICE_IN_MAX = 3.0                            # Kevin, 2026-11-13: no model above $3 per million input tokens


def fig_judge_problem(model: str) -> str | None:
    """Why the figure run's judge may not be called, or None: a Fable model, a price above the night's limit, or a call
    the router would refuse now (no key, an entry not verified)."""
    from ...router.models import default_models
    from ...router.smoke import skip_reason

    models = default_models()
    entry = models.models.get(model)
    if entry is None:
        return f"config/models.yaml has no model {model}, which fig.judge names"
    if "fable" in f"{model} {entry.id} {entry.display}".lower():
        return f"fig.judge names {model}, a Fable model, and tonight's brief rules Fable out"
    priced = models.models.get(getattr(entry, "model_key", None) or model, entry)
    price = getattr(priced, "price_in", None)
    if price is None or float(price) > FIG_PRICE_IN_MAX:
        return f"{model} costs {price} per million input tokens, above the ${FIG_PRICE_IN_MAX:.0f} limit"
    reason = skip_reason(entry)
    return f"the figure judge {model} cannot be called now: {reason}" if reason else None


def make_fig_judge(session: str = "FIG_agent_looks"):
    """The FIG run's agent_looks judge: config ``fig.judge`` (model, prompt, robot_state), on every step."""
    judge = dict(load_experiment().fig.get("judge", {}))
    return _Judge(judge["model"], judge.get("prompt", "crop_ref"), session, with_state=bool(judge.get("robot_state")))


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
                core = ToolCore(task, cell, "avail_simop", witnesses=make_witnesses(task, variant, alpha),
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
                _cycle_end(ledger, core, core.log, evaluate(cell), seed, "avail_simop_tools", variant, task)
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
