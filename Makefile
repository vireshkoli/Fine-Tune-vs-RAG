.DEFAULT_GOAL := help
.PHONY: help setup setup-gpu setup-check lint fmt type test check splits verify-splits index index-estimate eval report train train-estimate clean-pyc

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

CORPUS ?= parity
CONFIG ?= configs/train/qlora_r16.yaml
ARM ?= base
SEED ?=

index:  ## Build a retrieval index (CORPUS=parity|external)
	$(PY) python scripts/02_build_index.py --corpus $(CORPUS)

index-estimate:  ## Report corpus size and projected embedding GPU-hours, then stop
	$(PY) python scripts/02_build_index.py --corpus $(CORPUS) --estimate-only

eval:  ## Evaluate one arm on the frozen test set (ARM=base|rag-external|…)
	$(PY) python scripts/03_eval_arm.py --arm $(ARM) $(if $(SEED),--seed $(SEED),)

report:  ## Regenerate every table and figure from committed run JSONs
	$(PY) python scripts/07_make_report.py

train:  ## QLoRA fine-tune (CONFIG=configs/train/qlora_r16.yaml)
	$(PY) python scripts/04_train.py --config $(CONFIG)

train-estimate:  ## Time a few steps and project total GPU-hours, then stop
	$(PY) python scripts/04_train.py --config $(CONFIG) --estimate-only

clean-pyc:  ## Remove Python caches (never touches .artifacts/)
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache .ruff_cache .pytest_cache
