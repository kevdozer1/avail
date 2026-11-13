"""The staged decision: cheapest evidence first, 2 thresholds per stage, abstain at the end.

A stage scores the evidence gathered so far with a fused scorer and compares the score with
its 2 thresholds. Above ``tau_accept`` the claim is accepted. Below ``tau_reject`` it is
rejected. Between them the next stage runs. After the last stage with no decision the
verdict is abstain, which the fork turns into a call for a person.

This module works on evidence only: feature tables from the zoo, or live evidence gathered
through a ``StationView``. Witnesses are the only code that constructs a ``Verdict``
(CLAUDE.md rule 11), and every accepting verdict lists the evidence it stands on.

It knows no station. What a cascade collects comes in as arguments: the stage definitions
(``StageDef``), the feature names that are bookkeeping and stay out of the scorers, and the
id of the station the cascade is calibrated on. A stored witness carries all 3. The plug
station's are in ``avail.zoo.stages``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from ..evidence.fused import Scorer, fit_scorer
from ..schemas import Calibration, Cost, Stage, Verdict, Witness, WitnessScope, new_id
from ..stats import ABSTAIN, ACCEPT, REJECT, cp_upper, decide, decide_many, wilson
from .calibrate import budget_split, stage_thresholds

# the station id given to a stored cascade that does not name its station
UNKNOWN_STATION = "unknown"


@dataclass(frozen=True)
class StageDef:
    """What a stage collects and where its columns live in a zoo features table."""

    name: str
    sources: tuple[str, ...]          # in the order they are collected. "vision.gap@now" means vision.gap on a fresh frame
    zoo_pass: str                     # "A" (claim-time, nothing has acted) or "B" (the cascade order)
    label: str                        # the column holding the truth after this stage acted


def source_columns(table, zoo_pass: str, source: str, skip_features: tuple[str, ...] = ()) -> dict[str, str]:
    """Canonical feature name to table column for one source: ``probe.tug/disp_mm`` -> ``B/probe.tug/disp_mm``.

    A feature whose name starts with ``_`` is bookkeeping (a label, a time, an error) and is left
    out, and so is every feature named in ``skip_features`` and every column that holds no numbers.
    """
    prefix = f"{zoo_pass}/{source}/"
    out = {}
    for col in table.columns:
        if col.startswith(prefix):
            feat = col[len(prefix):]
            if feat.startswith("_") or feat in skip_features:
                continue
            if not np.issubdtype(table[col].dtype, np.number) and table[col].dtype != bool:
                continue
            out[f"{source}/{feat}"] = col
    return out


def stage_frame(table, stages: tuple[StageDef, ...], upto: int, skip_features: tuple[str, ...] = ()):
    """The evidence available after stage ``upto`` (0-based), with canonical column names."""
    import pandas as pd

    cols = {}
    for stage in stages[: upto + 1]:
        for source in stage.sources:
            cols.update(source_columns(table, stage.zoo_pass, source, skip_features))
    return pd.DataFrame({name: np.asarray(table[col], dtype=float) for name, col in cols.items()}, index=table.index)


def stage_seconds(table, stage: StageDef) -> np.ndarray:
    total = np.zeros(len(table))
    for source in stage.sources:
        col = f"{stage.zoo_pass}/{source}/_seconds"
        if col in table.columns:
            total += np.nan_to_num(np.asarray(table[col], dtype=float))
    return total


@dataclass
class FittedStage:
    name: str
    sources: tuple[str, ...]
    scorer: Scorer
    tau_reject: float
    tau_accept: float
    alpha_k: float
    beta_k: float
    n_neg: int = 0
    n_pos: int = 0

    def to_dict(self) -> dict:
        return {"name": self.name, "sources": list(self.sources), "scorer": self.scorer.to_dict(),
                "tau_reject": None if not math.isfinite(self.tau_reject) else self.tau_reject,
                "tau_accept": None if not math.isfinite(self.tau_accept) else self.tau_accept,
                "alpha_k": self.alpha_k, "beta_k": self.beta_k, "n_neg": self.n_neg, "n_pos": self.n_pos}

    @staticmethod
    def from_dict(d: dict) -> "FittedStage":
        return FittedStage(d["name"], tuple(d["sources"]), Scorer.from_dict(d["scorer"]),
                           -math.inf if d["tau_reject"] is None else float(d["tau_reject"]),
                           math.inf if d["tau_accept"] is None else float(d["tau_accept"]),
                           float(d["alpha_k"]), float(d["beta_k"]), int(d.get("n_neg", 0)), int(d.get("n_pos", 0)))


@dataclass
class Cascade:
    """A calibrated cascade for 1 predicate on 1 variant of a cell (for the plug station, 1 connector type).

    ``station_id``, ``stage_defs``, ``calibration`` and ``skip_features`` are keyword arguments.
    ``stage_defs`` says what each stage collects and where its columns live in a features table.
    ``skip_features`` names features that are bookkeeping and stay out of the scorers. A stored
    cascade carries both and its station id.
    """

    predicate: str
    connector: str
    alpha: float
    beta: float
    stages: list[FittedStage]
    version: str = "v0"
    station_id: str = field(kw_only=True)
    stage_defs: tuple[StageDef, ...] = field(kw_only=True)
    calibration: dict = field(default_factory=dict, kw_only=True)
    skip_features: tuple[str, ...] = field(default=(), kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(self.station_id, str) or not self.station_id.strip():
            raise ValueError(f"station_id must name the station the cascade is calibrated on, got {self.station_id!r}")
        self.stage_defs = tuple(self.stage_defs)
        self.skip_features = tuple(self.skip_features)
        if len(self.stage_defs) != len(self.stages):
            raise ValueError(f"the cascade has {len(self.stages)} fitted stages and {len(self.stage_defs)} stage "
                             "definitions. Pass the stage definitions the stages were fit with")

    # ---------------------------------------------------------------- on a zoo table
    def decide_table(self, table):
        """Run the cascade over a zoo features table. Returns a frame with one decision per row."""
        import pandas as pd

        n = len(table)
        decision = np.array([ABSTAIN] * n, dtype=object)
        reached = np.full(n, len(self.stages), dtype=int)
        score = np.full(n, np.nan)
        seconds = np.zeros(n)
        y_dec = np.zeros(n, dtype=bool)
        open_rows = np.ones(n, dtype=bool)
        for k, stage in enumerate(self.stages):
            sdef = self.stage_defs[k]
            frame = stage_frame(table, self.stage_defs, k, self.skip_features)
            s = stage.scorer.score(frame)
            seconds[open_rows] += stage_seconds(table, sdef)[open_rows]
            labels = np.asarray(table[sdef.label].fillna(False)).astype(bool)
            d = np.asarray(decide_many(s, stage.tau_reject, stage.tau_accept), dtype=object)
            score[open_rows] = s[open_rows]
            y_dec[open_rows] = labels[open_rows]           # the truth after the last stage that ran
            decided = open_rows & (d != ABSTAIN)
            decision[decided] = d[decided]
            reached[decided] = k + 1
            open_rows = open_rows & ~decided
        return pd.DataFrame({"decision": decision, "stage_reached": reached, "score": score,
                             "seconds": seconds, "y_dec": y_dec}, index=table.index)

    # ---------------------------------------------------------------- live
    def verify(self, view, claim, params: dict):
        """Gather evidence stage by stage on a live station and return (verdict, evidence list)."""
        from ..evidence import base

        features: dict[str, float] = {}
        evidence = []
        total = Cost(seconds=0.0, usd=0.0, moved_robot=False)
        decision, score, reached = ABSTAIN, None, len(self.stages)
        for k, stage in enumerate(self.stages):
            for key in stage.sources:
                source_id, _, viewname = key.partition("@")
                call = dict(params)
                if viewname:
                    call["view"] = viewname
                ev = base.get(source_id).collect(view, claim, call)
                evidence.append(ev)
                total = total + ev.cost
                for name, value in ev.features.items():
                    if name not in self.skip_features:
                        features[f"{key}/{name}"] = float(value)
            score = stage.scorer.score_one(features)
            d = decide(score, stage.tau_reject, stage.tau_accept)
            if d != ABSTAIN:
                decision, reached = d, k + 1
                break
        verdict = Verdict(verdict_id=new_id("verdict"), claim_id=claim.claim_id, decision=decision, score=score,
                          stage_reached=reached, witness_version=self.version,
                          evidence_ids=[e.evidence_id for e in evidence], cost_total=total)
        return verdict, evidence

    # ---------------------------------------------------------------- records
    def to_witness(self) -> Witness:
        cal = self.calibration
        return Witness(
            witness_id=f"witness_{self.station_id}_{self.connector}_{self.predicate}", version=self.version,
            predicate_name=self.predicate, scope=WitnessScope(station_id=self.station_id, connector=self.connector),
            stages=[Stage(source_ids=[s.partition("@")[0] for s in st.sources], scorer_id=st.scorer.scorer_id,
                          tau_reject=None if not math.isfinite(st.tau_reject) else st.tau_reject,
                          tau_accept=None if not math.isfinite(st.tau_accept) else st.tau_accept,
                          alpha_k=st.alpha_k, beta_k=st.beta_k) for st in self.stages],
            alpha=self.alpha, beta=self.beta,
            calibration=Calibration(n_pos=int(cal.get("n_pos", 0)), n_neg=int(cal.get("n_neg", 0)),
                                    split_id=str(cal.get("split_id", "unknown")), far_point=cal.get("far_point"),
                                    far_ucb95=cal.get("far_ucb95"), frr_point=cal.get("frr_point"),
                                    date=str(cal.get("date", date.today().isoformat()))),
        )

    def to_dict(self) -> dict:
        return {"predicate": self.predicate, "connector": self.connector, "alpha": self.alpha, "beta": self.beta,
                "version": self.version, "station_id": self.station_id, "calibration": self.calibration,
                "stages": [s.to_dict() for s in self.stages],
                "stage_defs": [{"name": d.name, "sources": list(d.sources), "zoo_pass": d.zoo_pass, "label": d.label}
                               for d in self.stage_defs],
                "skip_features": list(self.skip_features)}

    @staticmethod
    def from_dict(d: dict) -> "Cascade":
        """A cascade from what ``to_dict`` wrote.

        A file written before S5 lists no skipped features. That changes no score: a scorer reads
        only the features it was fit on. A file that names no station is read as ``unknown``.
        """
        defs = tuple(StageDef(x["name"], tuple(x["sources"]), x["zoo_pass"], x["label"]) for x in d["stage_defs"])
        return Cascade(d["predicate"], d["connector"], float(d["alpha"]), float(d["beta"]),
                       [FittedStage.from_dict(s) for s in d["stages"]], d.get("version", "v0"),
                       station_id=str(d.get("station_id") or UNKNOWN_STATION), stage_defs=defs,
                       calibration=d.get("calibration", {}), skip_features=tuple(d.get("skip_features", ())))


def fit_scorers(fit, connector: str, predicate: str, stage_defs: tuple[StageDef, ...],
                skip_features: tuple[str, ...] = ()) -> tuple[list[Scorer], list[float], list[float]]:
    """Fit one fused scorer per stage on the fit split. Also returns the share of false and true states per stage."""
    scorers, neg_share, pos_share = [], [], []
    for k, sdef in enumerate(stage_defs):
        ok = fit[sdef.label].notna()
        rows = fit[ok]
        y = np.asarray(rows[sdef.label]).astype(bool)
        frame = stage_frame(rows, stage_defs, k, skip_features)
        scorers.append(fit_scorer(f"fused.{connector}.{predicate}.{sdef.name}", frame, list(frame.columns), y))
        neg_share.append(float((~y).mean()))
        pos_share.append(float(y.mean()))
    return scorers, neg_share, pos_share


def calibrate_cascade(scorers: list[Scorer], neg_share: list[float], pos_share: list[float], cal, connector: str,
                      predicate: str, alpha: float, beta: float, stage_defs: tuple[StageDef, ...], *, station_id: str,
                      skip_features: tuple[str, ...] = (), version: str = "v0", split_id: str = "zoo_calibration",
                      budget: str = "weighted") -> Cascade:
    """Set every stage's thresholds on calibration rows. Scorers stay as fit.

    ``stage_defs`` and ``skip_features`` are the ones the scorers were fit with, and ``station_id``
    names the station the calibration rows come from. ``budget`` is how alpha and beta are shared
    between stages: ``weighted`` (the default, see ``avail.witness.calibrate``) or ``equal``
    (docs/04 section 4 as first written, kept so the difference can be measured).
    """
    if len(scorers) != len(stage_defs):
        raise ValueError(f"{len(scorers)} scorers for {len(stage_defs)} stages. Pass the stage definitions the "
                         "scorers were fit with")
    scores, labels, n_neg, n_pos = [], [], [], []
    for k, sdef in enumerate(stage_defs):
        ok = np.asarray(cal[sdef.label].notna())
        y = np.asarray(cal[sdef.label][ok]).astype(bool)
        s = scorers[k].score(stage_frame(cal[ok], stage_defs, k, skip_features))
        scores.append(s)
        labels.append(y)
        n_neg.append(int((~y).sum()))
        n_pos.append(int(y.sum()))
    if budget == "weighted":
        alpha_k, beta_k = budget_split(neg_share, pos_share, alpha, beta, n_neg, n_pos)
    elif budget == "equal":
        flat = [1.0] * len(stage_defs)
        alpha_k, beta_k = budget_split(flat, flat, alpha, beta, n_neg, n_pos)
    else:
        raise ValueError(f"budget must be 'weighted' or 'equal', got {budget!r}")
    stages = []
    for k, sdef in enumerate(stage_defs):
        thr = stage_thresholds(scores[k], labels[k], alpha_k[k], beta_k[k])
        stages.append(FittedStage(sdef.name, sdef.sources, scorers[k], thr["tau_reject"], thr["tau_accept"],
                                  alpha_k[k], beta_k[k], thr["n_neg"], thr["n_pos"]))
    return Cascade(predicate, connector, alpha, beta, stages, version=version, station_id=station_id,
                   stage_defs=tuple(stage_defs), skip_features=tuple(skip_features),
                   calibration={"n_pos": n_pos[0], "n_neg": n_neg[0], "split_id": split_id,
                                "date": date.today().isoformat(), "neg_share_fit": neg_share, "pos_share_fit": pos_share})


def evaluate(cascade: Cascade, table, conf: float = 0.95) -> dict:
    """The cascade's rates on a table it was not calibrated on, each with its n and its interval.

    FAR is accepts among states where the predicate is false at the decision. FRR is rejects
    among states where it is true at the decision. The certified bound is the one-sided
    Clopper-Pearson upper bound on FAR with its m (docs/04, section 3).
    """
    dec = cascade.decide_table(table)
    y = np.asarray(dec["y_dec"]).astype(bool)
    acc = np.asarray(dec["decision"] == ACCEPT)
    rej = np.asarray(dec["decision"] == REJECT)
    abst = np.asarray(dec["decision"] == ABSTAIN)
    n = len(dec)
    m_neg, m_pos = int((~y).sum()), int(y.sum())
    fa, fr = int((acc & ~y).sum()), int((rej & y).sum())

    def rate(k: int, m: int) -> dict:
        lo, hi = wilson(k, m, conf) if m else (float("nan"), float("nan"))
        return {"k": k, "n": m, "rate": k / m if m else float("nan"), "lo": lo, "hi": hi}

    out = {
        "n": n, "far": rate(fa, m_neg), "frr": rate(fr, m_pos), "abstain": rate(int(abst.sum()), n),
        "far_ucb": cp_upper(fa, m_neg, conf) if m_neg else float("nan"), "far_ucb_m": m_neg,
        "uncaught_per_claim": rate(fa, n), "uncaught_per_claim_ucb": cp_upper(fa, n, conf) if n else float("nan"),
        "accept_rate": rate(int(acc.sum()), n), "reject_rate": rate(int(rej.sum()), n),
        "true_accept": rate(int((acc & y).sum()), m_pos),
        "mean_seconds": float(np.mean(dec["seconds"])) if n else float("nan"),
        "median_seconds": float(np.median(dec["seconds"])) if n else float("nan"),
        "stage_reached": {int(k): int(v) for k, v in dec["stage_reached"].value_counts().sort_index().items()},
        "decided_by_stage": {
            st.name: {"accept": int((acc & (dec["stage_reached"] == i + 1)).sum()),
                      "reject": int((rej & (dec["stage_reached"] == i + 1)).sum())}
            for i, st in enumerate(cascade.stages)},
    }
    if "cls" in table.columns:
        per = {}
        cls = np.asarray(table["cls"])
        for c in sorted(set(cls)):
            sel = cls == c
            per[c] = {"n": int(sel.sum()), "accept": int((acc & sel).sum()), "reject": int((rej & sel).sum()),
                      "abstain": int((abst & sel).sum()), "false_accept": int((acc & ~y & sel).sum()),
                      "false_reject": int((rej & y & sel).sum()), "bad_at_decision": int((~y & sel).sum())}
        out["per_class"] = per
    return out
