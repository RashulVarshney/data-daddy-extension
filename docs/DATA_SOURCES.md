# Data sources

Everything downloaded is recorded here with source, license and fetch date. Downloaded data is **not committed** (`data/` is gitignored); re-create it with the scripts named below.

**All PII-like tables are synthetic** (Faker, seed 1337). Government-ID-like values are random formatted strings that are **invalid by construction** and clearly synthetic: SSN-like with area 000/666/9xx, Aadhaar-like starting 0/1, PAN-like with a 4th letter in X/Y/Z/Q, UK NINO-like with never-issued prefixes (BG, GB, NK, KN, TN, NT, ZZ), German tax-id-like starting with 0. No real personal data is created or used anywhere in the training or agent data.

## NONE-class columns: `scripts/fetch_none_data.py` (via `sklearn.datasets.fetch_openml`)

| Dataset | URL | License | Fetched | Use |
|---|---|---|---|---|
| OpenML `adult` (id 1590) | https://www.openml.org/d/1590 | OpenML licence field: "Public" (not an SPDX id; see dataset page) | 2026-10-05 | NONE-class columns (48842 rows x 15 cols) |
| OpenML `credit-g` (id 31) | https://www.openml.org/d/31 | OpenML licence field: "Public" (not an SPDX id; see dataset page) | 2026-10-05 | NONE-class columns (1000 rows x 21 cols) |
| OpenML `bank-marketing` (id 1461) | https://www.openml.org/d/1461 | OpenML licence field: "Public" (not an SPDX id; see dataset page) | 2026-10-05 | NONE-class columns (45211 rows x 17 cols) |
| OpenML `diabetes` (id 37) | https://www.openml.org/d/37 | OpenML licence field: "Public" (not an SPDX id; see dataset page) | 2026-10-05 | NONE-class columns (768 rows x 9 cols) |
| OpenML `blood-transfusion` (id 1464) | https://www.openml.org/d/1464 | OpenML licence field: "Public" (not an SPDX id; see dataset page) | 2026-10-05 | NONE-class columns (748 rows x 5 cols) |
| OpenML `wine-quality-red` (id 40691) | https://www.openml.org/d/40691 | OpenML licence field: "public" (not an SPDX id; see dataset page) | 2026-10-05 | NONE-class columns (1599 rows x 12 cols) |
| OpenML `car` (id 21) | https://www.openml.org/d/21 | OpenML licence field: "Public" (not an SPDX id; see dataset page) | 2026-10-05 | NONE-class columns (1728 rows x 7 cols) |
| OpenML `nursery` (id 26) | https://www.openml.org/d/26 | OpenML licence field: "Public" (not an SPDX id; see dataset page) | 2026-10-05 | NONE-class columns (12960 rows x 9 cols) |
| OpenML `mushroom` (id 24) | https://www.openml.org/d/24 | OpenML licence field: "Public" (not an SPDX id; see dataset page) | 2026-10-05 | NONE-class columns (8124 rows x 23 cols) |
| OpenML `spambase` (id 44) | https://www.openml.org/d/44 | OpenML licence field: "Public" (not an SPDX id; see dataset page) | 2026-10-05 | NONE-class columns (4601 rows x 58 cols) |
| OpenML `ilpd` (id 1480) | https://www.openml.org/d/1480 | OpenML licence field: "Public" (not an SPDX id; see dataset page) | 2026-10-05 | NONE-class columns (583 rows x 11 cols) |
| OpenML `phoneme` (id 1489) | https://www.openml.org/d/1489 | OpenML licence field: "Public" (not an SPDX id; see dataset page) | 2026-10-05 | NONE-class columns (5404 rows x 6 cols) |
| OpenML `banknote-authentication` (id 1462) | https://www.openml.org/d/1462 | OpenML licence field: "Public" (not an SPDX id; see dataset page) | 2026-10-05 | NONE-class columns (1372 rows x 5 cols) |
| OpenML `kc1` (id 1067) | https://www.openml.org/d/1067 | OpenML licence field: "Public" (not an SPDX id; see dataset page) | 2026-10-05 | NONE-class columns (2109 rows x 22 cols) |
| OpenML `vehicle` (id 54) | https://www.openml.org/d/54 | OpenML licence field: "Public" (not an SPDX id; see dataset page) | 2026-10-05 | NONE-class columns (846 rows x 19 cols) |
| OpenML `tic-tac-toe` (id 50) | https://www.openml.org/d/50 | OpenML licence field: "Public" (not an SPDX id; see dataset page) | 2026-10-05 | NONE-class columns (958 rows x 10 cols) |
| OpenML `heart-statlog` (id 53) | https://www.openml.org/d/53 | OpenML licence field: "Public" (not an SPDX id; see dataset page) | 2026-10-05 | NONE-class columns (270 rows x 14 cols) |
| OpenML `hepatitis` (id 55) | https://www.openml.org/d/55 | OpenML licence field: "Public" (not an SPDX id; see dataset page) | 2026-10-05 | NONE-class columns (155 rows x 20 cols) |
| OpenML `segment` (id 36) | https://www.openml.org/d/36 | OpenML licence field: "Public" (not an SPDX id; see dataset page) | 2026-10-05 | NONE-class columns (2310 rows x 20 cols) |
| OpenML `liver-disorders` (id 8) | https://www.openml.org/d/8 | OpenML licence field: "Public" (not an SPDX id; see dataset page) | 2026-10-05 | NONE-class columns (345 rows x 6 cols) |
| OpenML `titanic` (id 40945) | https://www.openml.org/d/40945 | OpenML licence field: "Public" (not an SPDX id; see dataset page) | 2026-10-05 | NONE-class columns (1309 rows x 8 cols) |
| OpenML `cylinder-bands` (id 6332) | https://www.openml.org/d/6332 | OpenML licence field: "Public" (not an SPDX id; see dataset page) | 2026-10-05 | NONE-class columns (540 rows x 38 cols) |
| OpenML `wdbc` (id 1510) | https://www.openml.org/d/1510 | OpenML licence field: "Public" (not an SPDX id; see dataset page) | 2026-10-05 | NONE-class columns (569 rows x 31 cols) |
| OpenML `kr-vs-kp` (id 3) | https://www.openml.org/d/3 | OpenML licence field: "Public" (not an SPDX id; see dataset page) | 2026-10-05 | NONE-class columns (3196 rows x 37 cols) |
| OpenML `monks-problems-1` (id 333) | https://www.openml.org/d/333 | OpenML licence field: "Public" (not an SPDX id; see dataset page) | 2026-10-05 | NONE-class columns (556 rows x 7 cols) |

Titanic's `name`, `ticket`, `cabin`, `home.dest`, `boat`, `body` columns were dropped (real people); any column whose name contains name/email/phone/address/lat/lon is dropped defensively.

## Hugging Face dataset cards: `scripts/fetch_hf_cards.py`

- Source: Hugging Face Hub (`huggingface_hub.HfApi.list_datasets`, then each repo's `README.md`), 2000 cards chosen as the most-downloaded public datasets that have a README of at least 300 characters (4200 candidates scanned).
- Fetched: 2026-10-05. Stored in `data/hf_cards/cards.jsonl` (gitignored).
- License: each card/dataset keeps its own license, recorded per card in the `license` field. Card text is used only for local retrieval experiments and is attributed to its dataset repo id. It is not redistributed in this repository.
- The queries in `eval/queries_manual_review.jsonl` reference public dataset repo ids only.

## Sanity-check CSVs (CLI demo only, `data/sanity/`, not committed)

| File | Source | License | Fetched |
|---|---|---|---|
| legislators-current.csv | https://unitedstates.github.io/congress-legislators/legislators-current.csv (unitedstates/congress-legislators) | CC0 / public domain (US public officials' public information) | 2026-10-05 |
| titanic.csv | https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv | Not stated in the source repo; historical passenger list (real names of long-deceased people). Used only for a local sanity check, not redistributed | 2026-10-05 |
| usgs_2.5_week.csv | https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_week.csv | US government work, public domain | 2026-10-05 |

## Models
- `BAAI/bge-small-en-v1.5` (MIT) and `cross-encoder/ms-marco-MiniLM-L-6-v2` (Apache-2.0) from the Hugging Face Hub.
- `en_core_web_sm` 3.8.0 spaCy model (MIT) for the Presidio baseline.
