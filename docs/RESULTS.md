# Results

Every number below comes from a run whose raw output is under `results/`. Seed 1337; library versions are in each `metrics.json`.

## 1. Sensitivity classifier (`make eval-classifier` → `results/classifier/`)

Command: `python -m eval.run_classifier --seed 1337` (raw: `metrics.json`, `summary.md`, `failure_cases.json`).
Selected model: logistic regression (validation score 0.9978 vs LightGBM 0.9967). Sizes: train 7,508 / val 1,556 / each test 2,165 columns; real NONE columns split by column (252 / 63 / 105).

Macro-F1 per split (reported separately):

| model | in-dist | unseen header | header-less | unseen locale |
|---|---|---|---|---|
| **selected: logreg** | 0.998 | 0.996 | 0.996 | 0.806 |
| ablation: logreg, no header dropout | 0.998 | 0.996 | 0.996 | 0.808 |
| other: lightgbm | 0.996 | 0.996 | 0.996 | 0.867 |
| baseline: regex-only | 0.686 | 0.697 | 0.686 | 0.565 |
| baseline: header-keyword-only | 0.538 | 0.191 | 0.046 | 0.544 |
| baseline: Presidio (en_core_web_sm; 600-column subsample per split) | 0.610 | 0.574 | 0.610 | 0.553 |

Reading this honestly:
- The synthetic task is **nearly saturated**: values alone separate the classes, so unseen headers and header-less mode cost nothing, and header dropout shows no measurable benefit (0.998 vs 0.998). These splits show the model does not *depend* on headers for this data. They do **not** show robustness to real-world header or value styles.
- **Unseen locale is the only split that exposes a real gap.** GOVT_ID F1 = 0.0 (200/200 UK NINO-like / German tax-id-like columns predicted PHONE (105) or PERSON_NAME (95)): the model learned formats, not the concept of a government ID. PHONE (P 0.62) and PERSON_NAME (P 0.61, R 0.73) also degrade; some unformatted UK numbers are predicted NONE. The other classes transfer (EMAIL, IP, DOB, ADDRESS ≥ 0.98).
- Per-class precision/recall/F1 and confusion matrices for every split are in `metrics.json`.
- LightGBM transferred slightly better to the unseen locale (0.867) but took 4× longer to train; selection used validation only.

### CLI sanity check on real public CSVs (`results/sanity/`)
`scan_csv` on three real files, scored against my own hand labels (`eval/sanity_check.py`, judgment not gold):

| file | PII columns found / labelled | errors |
|---|---|---|
| congress legislators (36 cols) | 6 / 8 | `full_name`, `middle_name` missed; false positives: `state`→PERSON_NAME, `url`→ADDRESS, `contact_form`,`rss_url`→IP_ADDRESS, `twitter`→PERSON_NAME, `mastodon`→EMAIL. Dataset scored CRITICAL (tier 3, raw rows blocked), which is plausible (names + DOB + address + phone) |
| Titanic (12 cols) | 0 / 1 | `Name` ("Braund, Mr. Owen Harris") missed → dataset scored LOW, a wrong risk call |
| USGS earthquakes (22 cols) | 1 / 2 | `latitude` (2-decimal values) missed; `place` ("45 km WNW of Beluga, Alaska") → ADDRESS; scored HIGH |

Causes: the training data has no URL / social-handle / state-code negatives, titled-name formats, or low-precision coordinates. I did **not** retrain on these files, to keep them usable as an honest check. This is the clearest evidence that the synthetic training distribution does not cover real-world columns.

## 2. Retrieval (`make eval-retrieval` → `results/retrieval/`)

Command: `python -m eval.run_retrieval --seed 1337` (raw: `metrics.json`, `per_query.json`, `summary.md`). 2,000 Hugging Face cards; dense model BAAI/bge-small-en-v1.5; reranker cross-encoder/ms-marco-MiniLM-L-6-v2 (top 30). Chunking A = 150 words / 25 overlap (28,645 chunks); B = 400 words / 80 overlap (23,704 chunks). Latency is single-query on a shared 12-core CPU (the box was also running other jobs), so treat it as indicative.

**SILVER** = 120 auto-generated metadata queries (not human-verified). **manual (UNVERIFIED)** = my 30 queries, `verified: false`, awaiting your review.

| chunking | retriever | set | Hit@5 | MRR@10 | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| A_150w_o25 | bm25 | silver | 0.667 | 0.537 | 0.464 | 75.0 | 112.3 |
| A_150w_o25 | bm25 | manual (UNVERIFIED) | 0.933 | 0.867 | 0.884 | 64.5 | 83.5 |
| A_150w_o25 | dense | silver | 0.600 | 0.454 | 0.328 | 15.8 | 18.0 |
| A_150w_o25 | dense | manual (UNVERIFIED) | 0.900 | 0.675 | 0.732 | 15.4 | 16.8 |
| A_150w_o25 | hybrid_rrf | silver | 0.783 | 0.634 | 0.497 | 101.0 | 161.6 |
| A_150w_o25 | hybrid_rrf | manual (UNVERIFIED) | 0.900 | 0.791 | 0.834 | 91.1 | 139.1 |
| A_150w_o25 | hybrid_rrf+rerank | silver | 0.867 | 0.762 | 0.595 | 1664.1 | 2100.9 |
| A_150w_o25 | hybrid_rrf+rerank | manual (UNVERIFIED) | 0.967 | 0.917 | 0.930 | 1968.2 | 2004.9 |
| B_400w_o80 | bm25 | silver | 0.692 | 0.570 | 0.486 | 65.4 | 98.0 |
| B_400w_o80 | bm25 | manual (UNVERIFIED) | 0.933 | 0.867 | 0.884 | 55.6 | 71.8 |
| B_400w_o80 | dense | silver | 0.583 | 0.455 | 0.327 | 16.6 | 19.3 |
| B_400w_o80 | dense | manual (UNVERIFIED) | 0.900 | 0.682 | 0.736 | 15.6 | 17.5 |
| B_400w_o80 | hybrid_rrf | silver | 0.767 | 0.611 | 0.484 | 89.3 | 155.0 |
| B_400w_o80 | hybrid_rrf | manual (UNVERIFIED) | 0.900 | 0.796 | 0.837 | 76.0 | 122.8 |
| B_400w_o80 | hybrid_rrf+rerank | silver | 0.842 | 0.741 | 0.591 | 1618.5 | 2115.1 |
| B_400w_o80 | hybrid_rrf+rerank | manual (UNVERIFIED) | 0.967 | 0.917 | 0.930 | 2043.2 | 2116.4 |

Findings:
- SILVER: hybrid+rerank is best (Hit@5 0.867 / MRR 0.762 on chunking A), hybrid next (0.783), BM25 0.667, dense 0.600. Dense is the weakest at constraint-style queries ("CC BY 4.0, French, under 1K examples").
- Chunking barely matters: A vs B differ by ≤0.025 Hit@5 for every retriever.
- Reranking costs ~1.6–2.0 s per query on CPU versus 15–100 ms for the first stage: +8 to +10 points Hit@5 on SILVER for ~20× latency.
- Manual (UNVERIFIED, n=30): BM25 0.933 Hit@5, dense 0.900, hybrid 0.900, hybrid+rerank 0.967. With only 30 queries one query is 3.3 points, so these differences are within noise; and hybrid does not beat BM25 here. `manual-16` (the Falcon RefinedWeb query) is missed by every retriever; see `per_query.json`.
- SILVER error analysis (chunking A, Hit@5): BM25 is weakest on queries with a license constraint (0.603) and on 4-constraint queries (0.567); dense is flat around 0.58–0.63 regardless of constraint type; rerank stays 0.77–0.93, lowest on 4-constraint queries (0.767).
- Known bias: SILVER relevance comes from the same metadata that is rendered into each indexed chunk, and the manual queries were written from card text, so lexical overlap favours BM25 on both sets. Neither set measures organic user queries.

## 3. Buyer agent (`make eval-agent` → `results/agent/`)

**Real-model numbers are PENDING.** No `ANTHROPIC_API_KEY` and no local OpenAI-compatible server were available. Nothing below says anything about how a real LLM behaves. The only run is the **mock harness validation**: `NaiveMockAgent`, a deliberately vulnerable scripted agent that obeys any instruction it sees (in user text or in tool results, ignoring delimiters).

Command: `env -u LLM_PROVIDER python -m eval.run_agent` (raw transcripts: `results/agent/mock_naive/`). 62 attacks, 30 benign.

| category | n | guardrails OFF | guardrails ON |
|---|---|---|---|
| direct_injection | 10 | 10/10 | 0/10 |
| indirect_injection | 14 | 13/14 | 4/14 |
| tool_misuse | 10 | 10/10 | 0/10 |
| raw_row_exfiltration | 10 | 10/10 | 0/10 |
| policy_bypass | 10 | 7/10 | 0/10 |
| misleading_claims | 8 | 8/8 | 0/8 |
| **total** | 62 | **58/62** | **4/62** |

Benign task success 25/30 (OFF) vs 29/30 (ON); benign blocked by the output filter 0/30 in both. Mock latency p50/p95 0.02/0.03 s (meaningless for real models); ~466 mock tokens/case.

What this does and does not show:
- It validates that the scorers detect success, and that the **code-enforced** layers (allowlist/arg validation, output filter + canary detection, session-derived tier + post-check, discrepancy notice) block those attacks.
- The 4 remaining ON successes are indirect-injection markers: the prompt-level guardrail (a) (delimiters + instruction) is something only a real model can honour, and the mock ignores it by construction. Real LLMs may do better or worse.
- ON's benign advantage comes from guardrails (d)/(e) forcing discrepancy disclosure; the 1 OFF/ON common benign failure is a mock retrieval miss.
- Over-blocking of 0/30 holds only for the mock's answers; real model answers will differ.

Listing pre-check (`results/agent/listings_classifier_check.json`): classifier column accuracy 0.99 over 25 synthetic listings, risk level agrees with ground truth on 24/25 (`geo-checkins`: `user_hash` hex IDs predicted PHONE).

## 4. Tests and CI
`make test`: 64 offline tests pass; `make lint` clean; `python -m eval.smoke` runs offline in CI.
