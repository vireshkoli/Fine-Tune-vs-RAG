# Serving image — runs on CPU *or* GPU.
#
# CPU-capable on purpose: the demo has to work for a reviewer with no GPU, and
# a serving image that silently requires CUDA is not a demo. torch resolves to
# the CPU wheel here; mount a GPU with `--gpus all` and it is used if present.
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
# No --extra gpu: the serving path needs the CPU dependency set only. Anything
# that genuinely needs CUDA belongs in the training image.
RUN uv sync --frozen --no-install-project

COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY results/ results/
RUN uv sync --frozen

ENV HF_HOME=/app/.artifacts/hub \
    TOKENIZERS_PARALLELISM=false

# Default to regenerating the report, which is pure CPU and proves the image
# works without a model or a GPU.
ENTRYPOINT ["uv", "run", "python"]
CMD ["scripts/07_make_report.py"]
