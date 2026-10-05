# Decisions

Each entry records an open decision and the default chosen.

1. **Python**: 3.11.17 installed through uv. System Python is 3.12, but the brief says 3.11.
2. **Seed**: global seed `1337` (`datadaddy_ai.common.seed.SEED`), passed to every generator, split and model.
3. **Packaging**: CPU-only torch is installed separately in `make setup`. Resolving the `dense` extra together with the rest picked a very old sentence-transformers. The CI and test extras exclude torch so tests stay light.
4. **Remote**: the user supplied `https://github.com/RashulVarshney/data-daddy-extension` as the push target. This overrides the original "never push" rule, so commits are pushed there. See docs/TODO.md for what was pushed.
5. **Tests are offline**: the classifier tests use a tiny model trained on a small synthetic set. Retrieval tests use BM25 and a hashing "dense" stub. No downloads happen in pytest.
