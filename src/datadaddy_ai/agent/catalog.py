"""Synthetic seller catalog: listing cards, tables (never exposed by tools), pre-computed sensitivity reports."""

from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass, field

LISTINGS_DIR = pathlib.Path("eval/listings")
CANARY_RE = re.compile(r"CANARY-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")


@dataclass
class Listing:
    id: str
    title: str
    card_text: str
    report: dict  # classifier-derived sensitivity report (aggregates only)
    truth: dict = field(default_factory=dict)  # ground truth labels / honesty flags (evaluation only)
    tasks: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=lambda: ["en"])
    license: list[str] = field(default_factory=list)
    size_categories: list[str] = field(default_factory=lambda: ["n<1K"])
    csv_path: str = ""


class Catalog:
    def __init__(self, listings: list[Listing]):
        self.by_id = {x.id: x for x in listings}

    @classmethod
    def load(cls, d: pathlib.Path = LISTINGS_DIR) -> Catalog:
        idx = json.loads((d / "index.json").read_text())
        out = []
        for e in idx:
            out.append(
                Listing(
                    e["id"],
                    e["title"],
                    (d / f"{e['id']}.md").read_text(),
                    json.loads((d / f"{e['id']}.report.json").read_text()),
                    e["truth"],
                    e["tasks"],
                    e.get("languages", ["en"]),
                    e["license"],
                    e.get("size_categories", ["n<1K"]),
                    str(d / f"{e['id']}.csv"),
                )
            )
        return cls(out)

    def ids(self) -> list[str]:
        return sorted(self.by_id)

    def all_canaries(self) -> set[str]:
        """Every canary token present in any raw table (used for leakage detection)."""
        out: set[str] = set()
        for x in self.by_id.values():
            out |= set(CANARY_RE.findall(pathlib.Path(x.csv_path).read_text()))
        return out
