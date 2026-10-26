# System design prompt: datadaddy-ai

Items marked with * are NOT done yet (pending or future scope). Everything else exists in the repo.

## Role and goal
Build `datadaddy-ai`, a standalone AI layer for a privacy-first data marketplace. It has no blockchain and no smart contracts. Three components:
(A) a trained column-level sensitivity classifier with a dataset risk score and a verification-tier policy,
(B) hybrid RAG search over real Hugging Face dataset cards,
(C) a tool-calling buyer agent with code-enforced guardrails, evaluated with an adversarial suite.
Work end to end. Decide open questions yourself and record them in docs/DECISIONS.md.

## Ground rules
- Python 3.11, pyproject.toml, pytest, ruff, Makefile targets: setup, test, lint, eval-classifier, eval-retrieval, eval-agent. One fixed global seed (1337). Record library versions in every results file.
- Git commit per phase. Never commit secrets or downloaded data (.gitignore data/ and models/; .env.example only).
- Never fabricate numbers. Every metric comes from a real run with raw output under results/ and the command that produced it. If something cannot be demonstrated, say so.
- Never create or use real personal data. All PII tables are synthetic (Faker). Government-ID-like values are random strings, invalid by construction (SSN area 000/666/9xx; Aadhaar-like starting 0/1; PAN-like with 4th letter X/Y/Z/Q; NINO-like with never-issued prefixes; German tax-id-like starting 0).
- Tests run offline with no API key, using a deterministic MockProvider. Only the agent evaluation may call a real LLM, and only when LLM_PROVIDER is set.
- If a download is blocked, record it in docs/TODO.md, use a clearly labelled synthetic substitute, and mark it synthetic in every report.

## Repository layout
src/datadaddy_ai/{classifier,retrieval,agent,policy,common,api}, tests/, eval/, scripts/, docs/ (DECISIONS, DATA_SOURCES, RESULTS, TODO), results/, eval/listings/ (committed synthetic catalog), .github/workflows/ci.yml (lint + offline tests + smoke eval), Dockerfile + docker-compose.yml (written, not built*).

## Component A: sensitivity classifier
Labels: EMAIL, PHONE, PERSON_NAME, ADDRESS, DOB, GOVT_ID, IP_ADDRESS, LAT_LONG, NONE.
Data:
- Synthetic PII columns via Faker, locales en_US and en_IN for training, en_GB and de_DE held out for the unseen-locale test. Noise: missing tokens, whitespace padding, mixed formats within a column, junk values, 200-row samples.
- Header styles: clear, abbreviation, generic (col_3, f12, field_a), misleading (generic words or other classes' names), none. Held-out header families with different word stems (inbox, whatsapp, domicile, born_on, y_coord) for the unseen-header split.
- NONE class: 25 real public OpenML tabular datasets (columns split by dataset/column into train/val/test so no column leaks), plus synthetic hard negatives (zip/pin codes, ages, order ids, 9/10/12-digit numbers, dashed codes, version strings, recent dates, UUIDs, company/job/city names), flagged source=synthetic_hardneg. Titanic names/tickets/cabins dropped.
Features per column (sample ≤200 values + header): char n-gram TF-IDF of values (raw and "shape" with A/a/9 masking), shape stats (digit/alpha/symbol ratios, length stats, uniqueness, null rate, numeric stats), regex match rates, Luhn/checksum checks for the synthetic ID formats, header char n-grams, generic-header flag.
Training: random header dropout p=0.5 (replace with a generic name or nothing).
Models: logistic regression and LightGBM, chosen by mean validation macro-F1 with and without headers. Ablation without dropout.
Baselines: regex-only, header-keyword-only, Presidio (presidio-analyzer + en_core_web_sm, column-level vote).
Evaluation, four separately reported splits: (a) in-distribution, (b) unseen header families, (c) header-less (all headers replaced), (d) unseen locale. Per-class P/R/F1, macro-F1, confusion matrices, failure cases, saved to results/classifier/.
Dataset risk score (pure function, tested): `1 - prod(1 - w_class * s)` over detected classes with severity weights, plus a quasi-identifier term `0.8 * uniqueness` when ≥2 QI columns (zip, age, gender, DOB, birth year) exist. Hard rules: GOVT_ID ⇒ ≥HIGH; GOVT_ID + NAME/ADDRESS/DOB ⇒ CRITICAL. Thresholds 0.25/0.5/0.8.
Tier policy (pure function, tested): LOW→tier 1, MEDIUM→2, HIGH→3, CRITICAL→3 and raw rows blocked always. Tiers are abstract (wallet-level, identity proof, document verification).
CLI: `scan_csv path --json` → per-column predictions, confidence, risk, tier. Sanity check on 3 real public CSVs saved to results/sanity/.
Future*: retrain with URL / social-handle / state-code / titled-name / low-precision-coordinate examples and fresh sanity CSVs; real (non-synthetic) labelled PII benchmarks; a transformer or char-CNN column encoder; locale-agnostic GOVT_ID detection; calibration; per-value (cell-level) detection and redaction.

## Component B: marketplace corpus and retrieval
- Fetch ~2,000 Hugging Face dataset cards (most-downloaded public datasets with README ≥300 chars) with id, task categories, languages, license, size categories, tags, text. Stored under data/hf_cards/ (gitignored). Attribution in DATA_SOURCES.md.
- Chunk by markdown section with a word cap and overlap; two settings (A: 150w/25 overlap, B: 400w/80). Each chunk is prefixed with a natural-language metadata line (id, tasks, languages, license, size).
- Retrievers: BM25 (rank_bm25), dense (BAAI/bge-small-en-v1.5 + FAISS flat IP), hybrid via reciprocal rank fusion, hybrid + cross-encoder rerank (ms-marco-MiniLM-L-6-v2, optional). Dataset-level scoring = best chunk. Metadata filters (task, language, license, size).
- Evaluation sets, reported separately: SILVER (120 queries generated from metadata with template paraphrases; relevant = all cards matching the constraints) and MANUAL-REVIEW (30 natural-language queries, each tied to one card, `verified: false` in eval/queries_manual_review.jsonl, reported UNVERIFIED until flipped). Metrics: Hit@5, MRR@10, nDCG@10, p50/p95 latency; all four retrievers × both chunkings; short error analysis; state the lexical-overlap bias.
- Status*: code and tests done; full evaluation not completed*, so no retrieval metrics exist yet*.
Future*: human verification of the 30 queries; organic query logs; learned sparse retrieval or ColBERT-style late interaction; query rewriting; embedding fine-tuning on marketplace listings; freshness/incremental indexing; ranking by price/quality signals.

## Component C: buyer agent with code-enforced guardrails
- LLMProvider interface: `complete(messages, tools) → text, tool_calls, token usage, latency`. MockProvider (deterministic, scriptable), AnthropicProvider (ANTHROPIC_API_KEY, ANTHROPIC_MODEL), OpenAICompatProvider (Ollama/vLLM). Selected by LLM_PROVIDER=mock|anthropic|openai_compat. Missing credentials raise; never a silent fallback to mock.
- Catalog: 25 synthetic seller listings (CSV + markdown card + classifier-generated sensitivity report + ground truth): clean, honest-PII, and 6 dishonest cards claiming "no personal data" while the table has PII. Canary tokens `CANARY-<tag>-<row>` in an `internal_note` column of every table; 4 cards deliberately leak sample rows; a system-prompt canary.
- Tools (JSON-schema validated, allowlisted): search_datasets, get_dataset_card, get_sensitivity_report, check_access_policy. No tool returns raw rows. Numeric min/max/mean emitted only for non-PII columns.
- Agent loop: max steps, tool execution, final answers must cite `[[listing-id: "exact quote"]]`; a verifier checks ids exist and the quote appears in retrieved context (one repair turn).
- Guardrails, each togglable: (a) card text wrapped as untrusted data + system instruction; (b) arg validation + allowlist; (c) output filter (Phase-1 classifier on table-like blocks + PII regexes + canary detection); (d) access policy enforced in code, buyer tier only from trusted session state, plus a post-check that removes "you can access" claims about denied listings; (e) discrepancy surfacing (card claim vs sensitivity report), with an enforced notice.
- Honeypot tools (admin_dump_rows, export_dataset, read_file, run_sql, delete_listing) exist only in the unguarded dispatcher so "forbidden tool executed" is observable when (b) is OFF.
- Adversarial suite: 62 attacks (direct injection 10, indirect via poisoned cards 14, tool misuse 10, raw-row exfiltration 10, policy bypass 10, misleading claims 8) and 30 benign queries. Success is programmatic only (canary or raw seller value in output, forbidden tool executed, denied listing described as accessible, split-marker obeyed, undisclosed discrepancy). No LLM judge. "Attempted" is recorded too.
- Run OFF vs ON at temperature 0; report counts per category, benign success, p50/p95 latency, tokens; save transcripts to results/agent/.
- Status: harness validated with a deliberately vulnerable NaiveMockAgent only. Real-model run pending*.
Future*: real-model runs on several models; multiple seeds/repeats for nondeterminism; larger and adaptive attack suites (multi-turn, encoded payloads, tool-result injection chains); learned injection detector; rate limiting and per-buyer budgets; human-in-the-loop approval for CRITICAL listings; differential-privacy or k-anonymity aggregates instead of raw stats.

## Interface and observability
FastAPI: POST /scan (CSV upload → report), GET /search, POST /ask (buyer tier from X-Session-Token, never the body), GET /health. JSON logs with a trace id per request, each tool call, latency, token counts. Dockerfile + docker-compose (written; not built or run*). No frontend*.
Future*: auth beyond demo tokens, async job queue for large scans, metrics endpoint/OpenTelemetry, a seller-facing UI, deployment manifests.

## Tests and CI
Offline pytest suite (64 tests): generator determinism, feature extraction, risk score and tier policy (CRITICAL blocks raw rows), tool-arg validation and allowlist, citation verifier, output filter catches canaries, policy ignores model-claimed tier, harness scoring with scripted mocks (attack succeeds / attack blocked), retrieval units, API. GitHub Actions: ruff, pytest, `python -m eval.smoke`.

## Deliverables
README with mermaid architecture, setup, how to run every eval, results summary, Limitations. docs/RESULTS.md with real numbers only. Final report: files, commands, results table, unfinished items, resume-ready facts using only measured numbers (each with the proving file).

## Measured so far
Classifier macro-F1 (selected logistic regression): 0.998 in-dist, 0.996 unseen headers, 0.996 header-less, 0.806 unseen locale (GOVT_ID F1 0.0 on unseen locale formats). Regex baseline 0.686, header-keyword 0.538, Presidio 0.610 (in-dist). Mock-agent harness: 58/62 → 4/62 attacks, benign 25/30 → 29/30. Everything else* is pending.
