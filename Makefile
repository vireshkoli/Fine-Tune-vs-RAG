.DEFAULT_GOAL := help
.PHONY: help setup setup-gpu setup-check lint fmt type test check splits verify-splits clean-pyc

PY := uv run

help:  ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup:  ## Install dependencies (CPU only; add GPU extras with `make setup-gpu`)
	uv sync --group dev

setup-gpu:  ## Install GPU dependencies on the lab machine
	uv sync --group dev --extra gpu

setup-check:  ## Preflight: GPUs, disk, cache isolation, HF token scope
	$(PY) python scripts/00_setup_check.py

lint:  ## Lint with ruff
	$(PY) ruff check .
	$(PY) ruff format --check .

fmt:  ## Autoformat and autofix
	$(PY) ruff check --fix .
	$(PY) ruff format .

type:  ## Type-check with mypy (strict)
	$(PY) mypy

test:  ## Run the CPU test suite
	$(PY) pytest

check: lint type test  ## Everything CI runs

splits:  ## Build and freeze the evaluation splits (needs network)
	$(PY) python scripts/01_build_splits.py --check-leakage

verify-splits:  ## Rebuild and prove the split still matches the committed manifest
	$(PY) python scripts/01_build_splits.py --verify

clean-pyc:  ## Remove Python caches (never touches .artifacts/)
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache .ruff_cache .pytest_cache
