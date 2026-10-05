from __future__ import annotations

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .labels import LABELS


def metrics(y_true: list[str], y_pred: list[str]) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    present = [i for i, n in enumerate(s) if n > 0]
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            sum(f[i] for i in present) / max(len(present), 1)
        ),  # over classes present in y_true
        "per_class": {
            LABELS[i]: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(s[i]),
            }
            for i in range(len(LABELS))
        },
        "confusion_matrix": {
            "labels": LABELS,
            "matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        },
    }


def failure_cases(cols, y_pred: list[str], limit: int = 20) -> list[dict]:
    out = []
    for c, p in zip(cols, y_pred, strict=True):
        if p != c.label:
            out.append(
                {
                    "true": c.label,
                    "pred": p,
                    "header": c.header,
                    "header_style": c.header_style,
                    "locale": c.locale,
                    "source": c.source,
                    "values": [str(v) for v in c.values if v][:4],
                }
            )
    return out[:limit] if limit else out
