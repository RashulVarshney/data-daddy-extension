"""Train / predict / persist the column sensitivity classifier."""

from __future__ import annotations

import pathlib

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from ..common.seed import SEED
from .features import ColumnFeaturizer, is_generic_header
from .labels import LABEL2ID, LABELS
from .synth import ColumnInstance, generic_header


def header_dropout(cols: list[ColumnInstance], p: float, seed: int) -> list[tuple[list, str | None]]:
    """Randomly replace the header with a col_N style generic name or nothing (prob p)."""
    rng = np.random.default_rng(seed)
    out = []
    for c in cols:
        if rng.random() < p:
            out.append((c.values, generic_header(rng) if rng.random() < 0.6 else None))
        else:
            out.append((c.values, c.header))
    return out


def make_estimator(kind: str, seed: int = SEED):
    if kind == "logreg":
        return LogisticRegression(C=4.0, max_iter=400, class_weight="balanced", random_state=seed)
    if kind == "lightgbm":
        return lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.08,
            num_leaves=31,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.5,
            class_weight="balanced",
            random_state=seed,
            n_jobs=8,
            verbose=-1,
        )
    raise ValueError(kind)


class ColumnClassifier:
    def __init__(self, kind: str = "logreg", seed: int = SEED, dropout: float = 0.5):
        self.kind, self.seed, self.dropout = kind, seed, dropout
        self.feat = ColumnFeaturizer()
        self.est = make_estimator(kind, seed)

    def fit(self, cols: list[ColumnInstance]):
        pairs = header_dropout(cols, self.dropout, self.seed)
        X = self.feat.fit_transform(pairs, self.seed)
        y = np.array([LABEL2ID[c.label] for c in cols])
        self.est.fit(X, y)
        return self

    def predict_proba(self, cols_pairs) -> np.ndarray:
        X = self.feat.transform(cols_pairs, self.seed)
        p = np.zeros((X.shape[0], len(LABELS)))
        p[:, self.est.classes_] = self.est.predict_proba(X)
        return p

    def predict(self, cols_pairs) -> list[str]:
        return [LABELS[i] for i in self.predict_proba(cols_pairs).argmax(1)]

    def predict_instances(self, cols: list[ColumnInstance], headerless: bool = False) -> list[str]:
        return self.predict([(c.values, None if headerless else c.header) for c in cols])

    def save(self, path: str | pathlib.Path):
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | pathlib.Path) -> ColumnClassifier:
        return joblib.load(path)


def macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    return float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0))


_ = is_generic_header
