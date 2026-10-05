"""Dataset-level risk score (0..1). Pure functions, no I/O."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

# severity weight of one detected class (how identifying / harmful on its own)
CLASS_WEIGHT = {
    "GOVT_ID": 1.0,
    "ADDRESS": 0.7,
    "DOB": 0.6,
    "EMAIL": 0.5,
    "PHONE": 0.5,
    "LAT_LONG": 0.5,
    "PERSON_NAME": 0.4,
    "IP_ADDRESS": 0.3,
}
LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
THRESHOLDS = [(0.80, "CRITICAL"), (0.50, "HIGH"), (0.25, "MEDIUM")]
QI_WEIGHT = 0.8  # how much a fully-unique quasi-identifier combination contributes

_QI_HEADER = re.compile(
    r"(zip|postal|pin_?code|postcode|^age$|age_?(years|yrs)?$|gender|^sex$|birth_?year|^yob$)", re.I
)
_GENDER = {"m", "f", "male", "female", "man", "woman", "other", "non-binary", "nb", "x"}


def class_signal(proportion: float) -> float:
    """Presence matters more than proportion: 0.6 floor, saturating at 1/3 of columns."""
    return 0.6 + 0.4 * min(1.0, 3.0 * proportion)


def risk_level(score: float) -> str:
    for t, name in THRESHOLDS:
        if score >= t:
            return name
    return "LOW"


def risk_score(
    column_labels: Sequence[str], qi_uniqueness: float | None = None, n_qi_columns: int = 0
) -> dict:
    """column_labels: predicted class per column (incl. NONE). qi_uniqueness: fraction of rows unique on the
    detected quasi-identifier combination (None if not computed). Returns score, level, components."""
    n = max(len(column_labels), 1)
    counts = Counter(x for x in column_labels if x != "NONE")
    keep = 1.0
    parts = {}
    for lab, k in counts.items():
        s = CLASS_WEIGHT.get(lab, 0.3) * class_signal(k / n)
        parts[lab] = round(s, 4)
        keep *= 1 - s
    direct = 1 - keep
    qi = QI_WEIGHT * qi_uniqueness if (qi_uniqueness is not None and n_qi_columns >= 2) else 0.0
    score = 1 - (1 - direct) * (1 - qi)
    level = risk_level(score)
    # hard rules: a government id is never below HIGH; id + any other direct identifier is CRITICAL
    if "GOVT_ID" in counts and LEVELS.index(level) < LEVELS.index("HIGH"):
        level = "HIGH"
    if "GOVT_ID" in counts and counts.keys() & {"PERSON_NAME", "ADDRESS", "DOB"}:
        level = "CRITICAL"
    score = max(score, {"HIGH": 0.5, "CRITICAL": 0.8}.get(level, 0.0))
    return {
        "score": round(float(min(score, 1.0)), 4),
        "level": level,
        "components": {"direct": round(direct, 4), "quasi_identifier": round(qi, 4), "per_class": parts},
    }


def detect_quasi_identifiers(
    headers: Sequence[str], columns: Sequence[Sequence], labels: Sequence[str]
) -> list[int]:
    """Indices of quasi-identifier columns: DOB-class columns, header hints (zip/age/gender/...), or gender-like values."""
    idx = []
    for i, (h, col, lab) in enumerate(zip(headers, columns, labels, strict=True)):
        vals = {str(v).strip().lower() for v in col if v is not None and str(v).strip() != ""}
        if (
            lab == "DOB"
            or _QI_HEADER.search(str(h or "").strip())
            or (vals and len(vals) <= 6 and vals <= _GENDER)
        ):
            idx.append(i)
    return idx


def quasi_identifier_uniqueness(rows: Sequence[tuple]) -> float:
    """Fraction of rows whose quasi-identifier tuple appears exactly once."""
    if not rows:
        return 0.0
    c = Counter(rows)
    return sum(1 for r in rows if c[r] == 1) / len(rows)
