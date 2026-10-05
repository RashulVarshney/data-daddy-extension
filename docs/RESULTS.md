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

**PENDING.** The corpus (2,000 Hugging Face dataset cards), chunkers, BM25 / dense / hybrid / rerank code, SILVER generator (120 queries) and 30 MANUAL-REVIEW queries (`verified: false`) are built and unit-tested, but the full evaluation had not finished when this session was wrapped up (embedding 28.6k + 23.7k chunks on CPU). No retrieval metrics are reported here. See docs/TODO.md. When it runs, MANUAL-REVIEW numbers are labeled UNVERIFIED until the flag is flipped, and the known bias applies: SILVER relevance is derived from the same metadata that is rendered into each indexed chunk, and queries derived from card text favour lexical overlap, so BM25 will look better than it would on organic queries.

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
