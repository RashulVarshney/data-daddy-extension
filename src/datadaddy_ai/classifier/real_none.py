"""Load real public tabular columns (downloaded by scripts/fetch_none_data.py) as NONE-class columns."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

from .synth import ColumnInstance

DATA_DIR = pathlib.Path("data/none_real")
# columns whose name or content makes them PII-like are dropped defensively
_DROP_SUBSTR = ("name", "email", "phone", "address", "lat", "lon")


def list_real_columns(data_dir: pathlib.Path = DATA_DIR) -> list[tuple[str, str]]:
    cols = []
    for f in sorted(data_dir.glob("*.csv")):
        hdr = pd.read_csv(f, nrows=0).columns
        for c in hdr:
            if any(s in str(c).lower() for s in _DROP_SUBSTR):
                continue
            cols.append((f.stem, str(c)))
    return cols


def real_instances(
    columns: list[tuple[str, str]],
    per_column: int,
    seed: int,
    data_dir: pathlib.Path = DATA_DIR,
    n_rows: int = 200,
    header_mode: str = "original",
) -> list[ColumnInstance]:
    """Several 200-row samples per real column. header_mode: original | generic | none."""
    from .synth import generic_header

    rng = np.random.default_rng(seed)
    out: list[ColumnInstance] = []
    cache: dict[str, pd.DataFrame] = {}
    for ds, col in columns:
        if ds not in cache:
            cache[ds] = pd.read_csv(data_dir / f"{ds}.csv", low_memory=False)
        s = cache[ds][col]
        for _ in range(per_column):
            idx = rng.choice(len(s), size=min(n_rows, len(s)), replace=False)
            vals = [
                None
                if pd.isna(v)
                else (str(int(v)) if isinstance(v, float) and float(v).is_integer() else str(v))
                for v in s.iloc[idx]
            ]
            if header_mode == "original":
                h, style = col, "original"
            elif header_mode == "generic":
                h, style = generic_header(rng), "generic"
            else:
                h, style = None, "none"
            out.append(ColumnInstance(vals, h, "NONE", "en_US", f"real:{ds}", style, group=f"{ds}/{col}"))
    return out


def manifest(data_dir: pathlib.Path = DATA_DIR) -> list[dict]:
    return json.loads((data_dir / "manifest.json").read_text())
