# Training image — GPU required.
#
# Pinned to a CUDA 12.8 runtime because the reference hardware (A40, driver
# 570.133.07) cannot run the CUDA 13 wheels that PyPI now ships by default.
# pyproject pins torch to the cu128 index for the same reason.
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependency layer first, so source edits do not invalidate the (large) wheel cache.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --group dev --extra gpu

COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY Makefile ./
RUN uv sync --frozen --group dev --extra gpu

# Caches stay inside the project so a container teardown cannot reach a shared
# host cache mounted alongside it — the same invariant fvr/config.py enforces.
ENV HF_HOME=/app/.artifacts/hub \
    HF_DATASETS_CACHE=/app/.artifacts/datasets \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ENTRYPOINT ["uv", "run", "python"]
CMD ["scripts/04_train.py", "--config", "configs/train/qlora_r16.yaml"]
