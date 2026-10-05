"""Fetch real public tabular datasets (OpenML via sklearn) used as NONE-class columns.

Writes data/none_real/<name>.csv and data/none_real/manifest.json (source id, license, fetch date).
Columns that are person names, coordinates or otherwise PII-like are dropped (listed in EXCLUDE).
"""

import datetime as dt
import json
import pathlib

from sklearn.datasets import fetch_openml

OUT = pathlib.Path("data/none_real")
# (name, openml data_id). Chosen: widely used, public, no direct identifiers.
DATASETS = [
    ("adult", 1590),
    ("credit-g", 31),
    ("bank-marketing", 1461),
    ("diabetes", 37),
    ("blood-transfusion", 1464),
    ("wine-quality-red", 40691),
    ("car", 21),
    ("nursery", 26),
    ("mushroom", 24),
    ("spambase", 44),
    ("ilpd", 1480),
    ("phoneme", 1489),
    ("banknote-authentication", 1462),
    ("kc1", 1067),
    ("vehicle", 54),
    ("tic-tac-toe", 50),
    ("heart-statlog", 53),
    ("hepatitis", 55),
    ("segment", 36),
    ("liver-disorders", 8),
    ("titanic", 40945),
    ("cylinder-bands", 6332),
    ("wdbc", 1510),
    ("kr-vs-kp", 3),
    ("monks-problems-1", 333),
]
# Titanic 'name', 'ticket', 'cabin', 'home.dest' contain real people / personal details -> excluded.
EXCLUDE = {"titanic": {"name", "ticket", "cabin", "home.dest", "boat", "body"}}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, did in DATASETS:
        try:
            b = fetch_openml(data_id=did, as_frame=True, parser="auto")
        except Exception as e:  # noqa: BLE001
            print("FAIL", name, did, repr(e)[:120])
            continue
        df = b.frame.drop(columns=list(EXCLUDE.get(name, [])), errors="ignore")
        df.to_csv(OUT / f"{name}.csv", index=False)
        d = b.details or {}
        manifest.append(
            {
                "name": name,
                "openml_id": did,
                "rows": len(df),
                "cols": df.shape[1],
                "license": d.get("licence", "unspecified"),
                "url": f"https://www.openml.org/d/{did}",
                "fetched": dt.date.today().isoformat(),
            }
        )
        print("ok", name, df.shape, d.get("licence"))
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
