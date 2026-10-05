"""Fetch ~2000 public Hugging Face dataset cards -> data/hf_cards/cards.jsonl (gitignored).

Selection: the most-downloaded public datasets (sorted by downloads) that have a README of >= 300 chars.
Keeps id, task_categories, languages, license, size_categories, tags, downloads, card text.
"""

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import pathlib

from huggingface_hub import HfApi, hf_hub_download

OUT = pathlib.Path("data/hf_cards")


def fetch_card(ds):
    try:
        p = hf_hub_download(ds.id, "README.md", repo_type="dataset", cache_dir=str(OUT / "_cache"))
        text = pathlib.Path(p).read_text(encoding="utf-8", errors="ignore")
    except Exception:  # noqa: BLE001
        return None
    cd = ds.card_data.to_dict() if ds.card_data else {}

    def as_list(x):
        return [x] if isinstance(x, str) else list(x or [])

    return {
        "id": ds.id,
        "task_categories": as_list(cd.get("task_categories")),
        "languages": as_list(cd.get("language")),
        "license": as_list(cd.get("license")),
        "size_categories": as_list(cd.get("size_categories")),
        "tags": list(ds.tags or [])[:40],
        "downloads": ds.downloads,
        "last_modified": str(ds.last_modified),
        "text": text,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=2000)
    ap.add_argument("--candidates", type=int, default=4200)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    cands = list(
        api.list_datasets(
            sort="downloads", limit=a.candidates, expand=["cardData", "downloads", "tags", "lastModified"]
        )
    )
    print("candidates", len(cands))
    rows = []
    with cf.ThreadPoolExecutor(16) as ex:
        for r in ex.map(fetch_card, cands):
            if r and len(r["text"]) >= 300:
                rows.append(r)
            if len(rows) >= a.target:
                break
    rows = rows[: a.target]
    with open(OUT / "cards.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    json.dump(
        {
            "fetched": dt.date.today().isoformat(),
            "n": len(rows),
            "candidates": len(cands),
            "selection": "top by downloads with README >= 300 chars",
        },
        open(OUT / "manifest.json", "w"),
    )
    print("saved", len(rows))


if __name__ == "__main__":
    main()
