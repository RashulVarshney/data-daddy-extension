PY ?= .venv/bin/python
SEED ?= 1337

.PHONY: setup test lint eval-classifier eval-retrieval eval-agent smoke

setup:
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) torch --index-url https://download.pytorch.org/whl/cpu
	uv pip install --python $(PY) sentence-transformers faiss-cpu
	uv pip install --python $(PY) -e ".[dev]"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check . && $(PY) -m ruff format --check .

eval-classifier:
	$(PY) -m eval.run_classifier --seed $(SEED)

eval-retrieval:
	$(PY) -m eval.run_retrieval --seed $(SEED)

# Real-model numbers need LLM_PROVIDER=anthropic|openai_compat; with unset/mock only harness validation runs.
eval-agent:
	$(PY) -m eval.run_agent --seed $(SEED)

smoke:
	$(PY) -m eval.smoke
