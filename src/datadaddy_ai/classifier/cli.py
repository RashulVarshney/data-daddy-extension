"""scan_csv <path> [--json] [--headerless] [--model PATH]"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from .model import ColumnClassifier
from .scan import DEFAULT_MODEL, scan_dataframe


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="scan_csv")
    ap.add_argument("path")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--headerless", action="store_true", help="ignore header names")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--max-rows", type=int, default=5000)
    a = ap.parse_args(argv)
    df = pd.read_csv(a.path, nrows=a.max_rows, dtype=str, keep_default_na=False, na_values=[""])
    rep = scan_dataframe(df, ColumnClassifier.load(a.model), a.headerless)
    if a.json:
        print(json.dumps(rep, indent=2))
        return
    print(f"{a.path}: {rep['n_rows']} rows x {rep['n_columns']} cols")
    for c in rep["columns"]:
        print(f"  {c['name']:<28} {c['predicted_class']:<12} {c['confidence']:.2f}")
    print(
        f"risk={rep['risk']['score']} level={rep['risk']['level']} tier={rep['tier_policy']['required_tier']} "
        f"raw_blocked={rep['tier_policy']['raw_rows_blocked']}"
    )


if __name__ == "__main__":
    main()
