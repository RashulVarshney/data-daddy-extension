"""Scan a table: per-column predictions, dataset risk score, tier policy."""

from __future__ import annotations

import pathlib

import pandas as pd

from ..policy.risk import detect_quasi_identifiers, quasi_identifier_uniqueness, risk_score
from ..policy.tiers import tier_policy
from .labels import LABELS
from .model import ColumnClassifier

DEFAULT_MODEL = pathlib.Path("models/classifier.joblib")


def scan_dataframe(df: pd.DataFrame, model: ColumnClassifier, headerless: bool = False) -> dict:
    headers = [str(c) for c in df.columns]
    cols = [[None if pd.isna(v) else str(v) for v in df[c]] for c in df.columns]
    proba = model.predict_proba(
        [(cv, None if headerless else h) for cv, h in zip(cols, headers, strict=True)]
    )
    labels = [LABELS[i] for i in proba.argmax(1)]
    qi_idx = detect_quasi_identifiers(headers, cols, labels)
    uniq = None
    if len(qi_idx) >= 2:
        uniq = quasi_identifier_uniqueness([tuple(cols[i][r] for i in qi_idx) for r in range(len(df))])
    risk = risk_score(labels, uniq, len(qi_idx))
    return {
        "n_rows": int(len(df)),
        "n_columns": len(headers),
        "columns": [
            {
                "name": h,
                "predicted_class": lab,
                "confidence": round(float(p.max()), 4),
                "top3": {LABELS[j]: round(float(p[j]), 4) for j in p.argsort()[::-1][:3]},
            }
            for h, lab, p in zip(headers, labels, proba, strict=True)
        ],
        "quasi_identifier_columns": [headers[i] for i in qi_idx],
        "quasi_identifier_uniqueness": None if uniq is None else round(uniq, 4),
        "risk": risk,
        "tier_policy": tier_policy(risk["level"]),
    }
