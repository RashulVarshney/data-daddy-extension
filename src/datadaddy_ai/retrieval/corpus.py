"""Dataset-card corpus: loading, normalised metadata line, markdown-section chunking."""

from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass, field

LANG_NAMES = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "zh": "Chinese",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "pt": "Portuguese",
    "it": "Italian",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "sv": "Swedish",
    "uk": "Ukrainian",
    "cs": "Czech",
    "fa": "Persian",
    "th": "Thai",
    "he": "Hebrew",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
    "el": "Greek",
    "fi": "Finnish",
    "da": "Danish",
    "no": "Norwegian",
    "hu": "Hungarian",
    "ro": "Romanian",
    "bg": "Bulgarian",
    "ca": "Catalan",
}
SIZE_TEXT = {
    "n<1K": "fewer than 1,000 examples",
    "1K<n<10K": "1K to 10K examples",
    "10K<n<100K": "10K to 100K examples",
    "100K<n<1M": "100K to 1M examples",
    "1M<n<10M": "1M to 10M examples",
    "10M<n<100M": "10M to 100M examples",
    "100M<n<1B": "100M to 1B examples",
    "1B<n<10B": "over 1 billion examples",
}


def license_text(code: str) -> str:
    c = code.lower()
    m = re.fullmatch(r"cc-by(-nc)?(-sa)?(-nd)?-(\d\.\d)", c)
    if m:
        return (
            "CC BY"
            + (" NC" if m.group(1) else "")
            + (" SA" if m.group(2) else "")
            + (" ND" if m.group(3) else "")
            + " "
            + m.group(4)
        )
    if c == "cc0-1.0":
        return "CC0 1.0 public domain"
    return (
        c.replace("-", " ").replace("apache 2.0", "Apache 2.0").replace("mit", "MIT")
        if c in {"mit", "apache-2.0", "odc-by", "gpl-3.0", "bsd-3-clause", "openrail", "other", "unknown"}
        else c
    )


@dataclass
class Card:
    id: str
    text: str
    task_categories: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    license: list[str] = field(default_factory=list)
    size_categories: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    downloads: int = 0

    def metadata_line(self) -> str:
        langs = ", ".join(f"{LANG_NAMES.get(x, x)}" for x in self.languages[:6])
        sizes = ", ".join(SIZE_TEXT.get(x, x) for x in self.size_categories)
        lic = ", ".join(license_text(x) for x in self.license)
        tasks = ", ".join(t.replace("-", " ") for t in self.task_categories)
        return f"Dataset {self.id} | tasks: {tasks} | languages: {langs} | license: {lic} | size: {sizes}"

    def body(self) -> str:
        return strip_front_matter(self.text)


def strip_front_matter(text: str) -> str:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4 :].lstrip()
    return text


def load_cards(
    path: str | pathlib.Path = "data/hf_cards/cards.jsonl", limit: int | None = None
) -> list[Card]:
    out = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            out.append(
                Card(
                    d["id"],
                    d["text"],
                    d["task_categories"],
                    d["languages"],
                    d["license"],
                    d["size_categories"],
                    d.get("tags", []),
                    d.get("downloads") or 0,
                )
            )
            if limit and len(out) >= limit:
                break
    return out


@dataclass
class Chunk:
    doc_id: str
    idx: int
    text: str  # includes the metadata line prefix


def _split_sections(body: str) -> list[str]:
    parts = re.split(r"(?m)^(?=#{1,4}\s)", body)
    return [p.strip() for p in parts if p.strip()]


def chunk_card(card: Card, max_words: int, overlap: int) -> list[Chunk]:
    """Chunk by markdown section; sections longer than max_words are cut into windows with `overlap` words."""
    meta = card.metadata_line()
    chunks: list[Chunk] = []
    for sec in _split_sections(card.body()):
        words = sec.split()
        if len(words) <= max_words:
            pieces = [sec]
        else:
            pieces, step = [], max(max_words - overlap, 1)
            for s in range(0, len(words), step):
                pieces.append(" ".join(words[s : s + max_words]))
                if s + max_words >= len(words):
                    break
        for p in pieces:
            chunks.append(Chunk(card.id, len(chunks), f"{meta}\n{p}"))
    if not chunks:
        chunks.append(Chunk(card.id, 0, meta))
    return chunks


def chunk_corpus(cards: list[Card], max_words: int, overlap: int) -> list[Chunk]:
    out = []
    for c in cards:
        out.extend(chunk_card(c, max_words, overlap))
    return out


CHUNK_SETTINGS = {"A_150w_o25": (150, 25), "B_400w_o80": (400, 80)}
