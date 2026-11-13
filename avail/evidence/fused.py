"""The fused scorer: a logistic model over the features gathered so far.

One model per predicate, connector and stage. It is fit on the fit split, its thresholds
come from the calibration split, and its numbers are reported on the test split
(docs/03_evidence_sources.md, "Fused scorer"). A ``Scorer`` is plain numbers, so a witness
file can carry it and apply it without scikit-learn.

Missing features are part of life: a probe that never slipped has no slip force. Each
feature is filled with its fit-split median, and a feature that is missing often enough gets
a second column that says so.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

MISSING_MIN_SHARE = 0.01      # a feature missing in at least this share of fit rows gets a missing flag


@dataclass
class Scorer:
    """A fitted logistic model. ``score`` is the log-odds that the predicate holds. Higher means more likely."""

    scorer_id: str
    features: list[str]
    medians: list[float]
    flag_missing: list[bool]
    means: list[float]
    scales: list[float]
    coef: list[float]
    intercept: float
    n_fit: int = 0
    n_pos: int = 0
    constant: bool = False                    # true when the fit split had 1 class only: the score says nothing
    notes: str = ""
    columns: list[str] = field(default_factory=list)   # design-matrix column names, for reading the coefficients

    def design(self, rows) -> np.ndarray:
        """The standardized design matrix for a table (DataFrame) or a list of feature dicts."""
        x = _raw_matrix(rows, self.features)
        cols = []
        for j, name in enumerate(self.features):
            col = x[:, j]
            missing = ~np.isfinite(col)
            cols.append(np.where(missing, self.medians[j], col))
            if self.flag_missing[j]:
                cols.append(missing.astype(float))
        if not cols:
            return np.zeros((x.shape[0], 0))
        d = np.column_stack(cols)
        return (d - np.asarray(self.means)) / np.asarray(self.scales)

    def score(self, rows) -> np.ndarray:
        d = self.design(rows)
        if d.shape[1] == 0:
            return np.full(d.shape[0], self.intercept)
        return d @ np.asarray(self.coef) + self.intercept

    def score_one(self, features: dict) -> float:
        return float(self.score([features])[0])

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "Scorer":
        return Scorer(**data)

    def top_terms(self, k: int = 5) -> list[tuple[str, float]]:
        """The largest standardized coefficients, for a person who asks what the scorer leans on."""
        order = np.argsort(-np.abs(np.asarray(self.coef)))[:k]
        return [(self.columns[i], float(self.coef[i])) for i in order]


def _raw_matrix(rows, features: list[str]) -> np.ndarray:
    if hasattr(rows, "columns"):                           # a DataFrame
        out = np.full((len(rows), len(features)), np.nan)
        for j, name in enumerate(features):
            if name in rows.columns:
                out[:, j] = np.asarray(rows[name], dtype=float)
        return out
    out = np.full((len(rows), len(features)), np.nan)
    for i, row in enumerate(rows):
        for j, name in enumerate(features):
            value = row.get(name)
            if value is not None:
                out[i, j] = float(value)
    return out


def fit_scorer(scorer_id: str, rows, features: list[str], labels, c: float = 1.0) -> Scorer:
    """Fit a logistic scorer on fit-split rows. ``labels`` is the truth after the stage's action."""
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(labels, dtype=bool)
    x = _raw_matrix(rows, features)
    keep = [j for j in range(x.shape[1]) if np.isfinite(x[:, j]).any()]          # drop features never measured
    features = [features[j] for j in keep]
    x = x[:, keep]
    medians, flags, cols, names = [], [], [], []
    for j, name in enumerate(features):
        col = x[:, j]
        missing = ~np.isfinite(col)
        med = float(np.median(col[~missing]))
        medians.append(med)
        cols.append(np.where(missing, med, col))
        names.append(name)
        flag = bool(missing.mean() >= MISSING_MIN_SHARE)
        flags.append(flag)
        if flag:
            cols.append(missing.astype(float))
            names.append(name + " (missing)")
    n, n_pos = int(len(y)), int(y.sum())
    if not cols or n_pos == 0 or n_pos == n:
        return Scorer(scorer_id, features, medians, flags, [0.0] * len(cols), [1.0] * len(cols), [0.0] * len(cols),
                      0.0, n, n_pos, constant=True, notes="the fit split had 1 class or no usable feature", columns=names)
    d = np.column_stack(cols)
    means = d.mean(axis=0)
    scales = d.std(axis=0)
    scales = np.where(scales < 1e-9, 1.0, scales)
    z = (d - means) / scales
    model = LogisticRegression(C=c, max_iter=5000)
    model.fit(z, y)
    return Scorer(scorer_id, features, medians, flags, [float(v) for v in means], [float(v) for v in scales],
                  [float(v) for v in model.coef_[0]], float(model.intercept_[0]), n, n_pos, columns=names)
