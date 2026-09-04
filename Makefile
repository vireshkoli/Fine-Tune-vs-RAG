.DEFAULT_GOAL := help
.PHONY: help setup setup-gpu setup-check lint fmt type test check check-ci splits verify-splits index index-estimate eval report train train-estimate contamination errors verify-recoverable teardown docker-build clean-pyc space-data push-dry push

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

check: lint type test  ## Everything CI runs (in the *local* environment)

# CI installs only the CPU dependency set, so a local `make check` on a machine
# with the GPU extras can pass while CI fails — that happened, twice: an absent
# `transformers` made mypy reject subclassing TrainerCallback, and made
# tests/test_train.py fail at collection. This target reproduces CI's
# environment in a throwaway venv so the divergence cannot hide again.
check-ci:  ## Run lint, types and tests in a CPU-only venv, exactly as CI does
	UV_PROJECT_ENVIRONMENT=/tmp/fvr-ci-venv uv sync --frozen --group dev
	UV_PROJECT_ENVIRONMENT=/tmp/fvr-ci-venv HF_HUB_OFFLINE=1 uv run ruff check .
	UV_PROJECT_ENVIRONMENT=/tmp/fvr-ci-venv HF_HUB_OFFLINE=1 uv run ruff format --check .
	UV_PROJECT_ENVIRONMENT=/tmp/fvr-ci-venv HF_HUB_OFFLINE=1 uv run mypy
	UV_PROJECT_ENVIRONMENT=/tmp/fvr-ci-venv HF_HUB_OFFLINE=1 uv run pytest -q

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

verify-recoverable:  ## Prove every artifact survives the lab machine being wiped
	$(PY) python scripts/09_verify_recoverable.py

teardown:  ## Dry-run the artifact deletion manifest (add EXECUTE=1 to delete)
	$(PY) python scripts/10_teardown.py $(if $(EXECUTE),--execute,)

contamination:  ## Run permutation, position-bias and verbatim probes (ARM=base)
	$(PY) python scripts/05_contamination.py --arm $(ARM) $(if $(ADAPTER),--adapter $(ADAPTER),)

errors:  ## Categorise every arm's failures and emit review CSVs
	$(PY) python scripts/06_error_analysis.py

space-data:  ## Rebuild the demo Space payload from committed run JSONs
	$(PY) python scripts/08_push_to_hub.py --build-data --target space

push-dry:  ## Print the exact Hub upload manifest without uploading anything
	$(PY) python scripts/08_push_to_hub.py --build-data

push:  ## Publish the adapter and the demo Space to the Hugging Face Hub
	$(PY) python scripts/08_push_to_hub.py --build-data --execute

docker-build:  ## Build both images (train needs CUDA; serve runs on CPU)
	docker build -f docker/train.Dockerfile -t fine-tune-vs-rag:train .
	docker build -f docker/serve.Dockerfile -t fine-tune-vs-rag:serve .

clean-pyc:  ## Remove Python caches (never touches .artifacts/)
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache .ruff_cache .pytest_cache
