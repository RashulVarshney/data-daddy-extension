# TODO / blockers

1. **Retrieval evaluation not completed** (user asked to wrap up). Run `make eval-retrieval` (needs `python scripts/fetch_hf_cards.py` first; ~30+ min on CPU for embeddings, cached under data/hf_cards/*.npy). Then fill docs/RESULTS.md §2 and the README table. `eval/queries_manual_review.jsonl` is `verified: false`: please review and flip.
2. **No real-model agent numbers.** Set `LLM_PROVIDER=anthropic` + `ANTHROPIC_API_KEY` + `ANTHROPIC_MODEL`, or `LLM_PROVIDER=openai_compat` + Ollama vars, then `make eval-agent`. Not exercised against a real API in this session (provider code unit-tested only for conversion and credential failures).
3. Classifier covers few real-world column shapes (URLs, handles, state codes, titled names, low-precision coordinates); GOVT_ID does not transfer to unseen locale formats. Add diverse negatives/positives and fresh sanity CSVs.
4. Dockerfile / docker-compose were written but **not built or run** (disk constraints).
5. Presidio baseline is evaluated on a 600-column subsample per split.
6. Push: the user supplied https://github.com/RashulVarshney/data-daddy-extension as the push remote. See git log / final report for whether the push succeeded.
