# datadaddy-ai: project description

## Why this exists
DataDaddy was a privacy-first data marketplace concept I built at a hackathon. A marketplace for data has a trust problem that its blockchain/payment layer does not solve: sellers upload tables, and neither side can easily tell what is in them. Sellers can be careless or dishonest ("anonymized, no personal data" on a file full of emails), buyers need to find relevant datasets among thousands of listings, and an AI assistant helping buyers becomes a new attack surface (poisoned listing text, requests to dump rows, claims of higher verification). This project is the AI layer for that idea, built standalone with no blockchain, so each piece can be measured on its own.

## What it does
1. **Looks inside a table and decides how sensitive it is.** A classifier reads a CSV column by column (a sample of values plus the header) and labels each column as email, phone, person name, address, date of birth, government ID, IP address, lat/long, or none. It turns that into a 0–1 dataset risk score (LOW to CRITICAL), including a re-identification check on combinations like zip + age + gender. A policy then says which buyer verification tier is required (1, 2 or 3) and whether raw-row access is blocked completely (always for CRITICAL).
2. **Finds datasets.** A hybrid search over ~2,000 real Hugging Face dataset cards combines keyword search (BM25), semantic search (small embedding model + FAISS) and reciprocal rank fusion, with an optional cross-encoder reranker and metadata filters.
3. **Answers buyer questions safely.** A tool-calling agent can search listings, read cards, fetch the automated sensitivity report and check access policy. Its final answers must cite listings with exact quotes, which are verified. Five guardrails are enforced in code (and can each be switched off for evaluation): untrusted-text wrapping, argument validation with an allowlist, an output filter, access policy driven only by trusted session state, and surfacing of card-vs-report discrepancies.
4. **Proves (or fails to prove) that the guardrails work.** An adversarial suite of 62 attacks and 30 benign queries is scored by code, not by an LLM judge: canary strings leaking, forbidden tools executing, denied datasets described as accessible, injected markers obeyed, dishonest cards left undisclosed.
5. **Serves it.** A small FastAPI app exposes /scan, /search and /ask with structured JSON logs and a trace id per request.

## How it works
- **Data.** All PII tables are synthetic (Faker; en_US/en_IN for training, en_GB/de_DE held out). Government-ID-like values are fake and invalid by construction. Non-PII columns come from 25 real public OpenML datasets (split by column to avoid leakage), plus synthetic look-alike negatives (zip codes, long numeric ids, version strings).
- **Features and models.** Character n-gram TF-IDF of the raw values and of their "shape", shape statistics, regex match rates, checksum checks and header n-grams, trained with random header dropout so the model cannot lean only on header names. Logistic regression and LightGBM are compared on validation; Presidio, regex-only and header-keyword baselines are included.
- **Risk and policy** are small pure functions with unit tests (severity-weighted noisy-OR over detected classes, quasi-identifier uniqueness term, hard rules for government IDs).
- **Retrieval** chunks cards by markdown section with a size cap and overlap (two settings compared), indexes each chunk with a metadata line, and scores documents by their best chunk.
- **Agent safety is mostly code, not prompts.** The tool executor rejects unknown tools and invalid arguments, overrides any model-supplied buyer tier with the session's, wraps card text as untrusted, adds discrepancy flags to reports, and a post-processor removes access claims about denied listings. The output filter runs the classifier plus regexes and canary detection before an answer is delivered. No tool ever returns raw rows.
- **Testing without an API key.** Everything runs offline with a deterministic mock provider; only the agent evaluation can call a real model, and it fails loudly if credentials are missing.

## What has actually been measured
- Classifier macro-F1: 0.998 in-distribution, 0.996 unseen header styles, 0.996 header-less, 0.806 unseen locale. Baselines on the in-distribution split: regex 0.686, header-keyword 0.538, Presidio 0.610.
- The honest reading: the synthetic task is nearly saturated, so the header-related splits barely test anything. The unseen-locale split shows a real gap (government-ID detection fails completely on unseen formats). A scan of three real public CSVs showed clear mistakes (URLs labelled as IP addresses, names missed, one dataset scored LOW when it should not be).
- Agent guardrails against a deliberately vulnerable scripted agent: 58/62 attacks succeed with guardrails off, 4/62 with them on; benign success 25/30 → 29/30. This validates the harness and the code-enforced layers; it says nothing about real LLMs.
- 64 offline tests pass; lint is clean; the repo is pushed to GitHub.

## What is not done yet*
- *Retrieval metrics (Hit@5, MRR@10, nDCG@10, latency) for BM25 vs dense vs hybrid vs rerank on both chunkings. The code exists; the evaluation did not finish.
- *Human verification of the 30 manual-review queries.
- *Real-model agent results (needs an Anthropic key or a local Ollama-style server).
- *Docker image build and run test.
- *Retraining the classifier on more diverse real-world column shapes and unseen-locale ID formats.
- *Larger, adaptive attack suites, repeated runs for LLM nondeterminism, a frontend, and production auth.

## Limitations
Synthetic classifier data; silver labels derived from the same metadata that is indexed (lexical bias); a small attack suite; LLM nondeterminism; no blockchain integration; a standalone prototype rather than production software.
