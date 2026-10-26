# TODO / blockers

1. Retrieval evaluation finished (see RESULTS §2). Still needed: verify `eval/queries_manual_review.jsonl` (flip `verified`), then re-run or relabel.
2. **No real-model agent numbers.** Set `LLM_PROVIDER=anthropic` + `ANTHROPIC_API_KEY` + `ANTHROPIC_MODEL`, or `LLM_PROVIDER=openai_compat` + Ollama vars, then `make eval-agent`. Not exercised against a real API in this session (provider code unit-tested only for conversion and credential failures).
3. Classifier covers few real-world column shapes (URLs, handles, state codes, titled names, low-precision coordinates); GOVT_ID does not transfer to unseen locale formats. Add diverse negatives/positives and fresh sanity CSVs.
4. Dockerfile / docker-compose were written but **not built or run** (disk constraints).
5. Presidio baseline is evaluated on a 600-column subsample per split.
6. Push: the user supplied https://github.com/RashulVarshney/data-daddy-extension as the push remote. See git log / final report for whether the push succeeded.
