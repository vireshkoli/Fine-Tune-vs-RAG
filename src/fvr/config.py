"""Typed configuration — the single source of truth for paths, seeds and secrets.

Two deliberately separate objects:

``ProjectConfig``
    Everything non-secret, loaded from YAML under ``configs/``. Committed to git.

``Secrets``
    Tokens only, loaded from environment or ``.env``. Never from YAML, so a
    secret cannot be committed by accident — the separation is structural
    rather than a convention someone has to remember.

Importing this module also pins the Hugging Face cache into
``$PROJECT_ROOT/.artifacts``. That must happen *before* ``huggingface_hub`` or
``transformers`` is imported, because those read their cache locations at
import time. Every entry point therefore imports ``fvr.config`` first; the
rule is enforced by ``tests/test_config.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Final, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ``src/fvr/config.py`` -> ``src/fvr`` -> ``src`` -> repo root
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

#: The one directory this project may delete. See ``scripts/10_teardown.py``.
ARTIFACTS_DIRNAME: Final[str] = ".artifacts"


class Paths(BaseModel):
    """Filesystem layout, all resolved absolute and all under ``PROJECT_ROOT``."""

    model_config = ConfigDict(frozen=True)

    root: Path = PROJECT_ROOT
    artifacts: Path = PROJECT_ROOT / ARTIFACTS_DIRNAME
    hub: Path = PROJECT_ROOT / ARTIFACTS_DIRNAME / "hub"
    datasets: Path = PROJECT_ROOT / ARTIFACTS_DIRNAME / "datasets"
    indices: Path = PROJECT_ROOT / ARTIFACTS_DIRNAME / "indices"
    checkpoints: Path = PROJECT_ROOT / ARTIFACTS_DIRNAME / "checkpoints"
    results: Path = PROJECT_ROOT / "results"
    configs: Path = PROJECT_ROOT / "configs"

    @field_validator("*", mode="after")
    @classmethod
    def _absolute(cls, value: Path) -> Path:
        return value.resolve()

    def deletable(self) -> tuple[Path, ...]:
        """Paths teardown is permitted to remove. Everything else is off limits."""
        return (self.hub, self.datasets, self.indices, self.checkpoints)

    def ensure(self) -> None:
        """Create the artifact tree. Never creates anything outside the project."""
        for path in (self.artifacts, *self.deletable(), self.results):
            path.mkdir(parents=True, exist_ok=True)


class ProjectConfig(BaseModel):
    """Non-secret configuration, loaded from ``configs/base.yaml``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int = 42
    #: GPU reserved for timed inference. Kept exclusive so latency is comparable.
    inference_device: int = 0
    #: GPU reserved for the judge. Never runs while a timed run is in flight.
    judge_device: int = 1
    #: Fail the setup check below this much free disk (the project needs ~74 GiB).
    min_free_disk_gib: float = 90.0
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    paths: Paths = Field(default_factory=Paths)


class Secrets(BaseSettings):
    """Tokens, from the environment or ``.env`` only — never from YAML."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    hf_token: str | None = Field(default=None, alias="HF_TOKEN")
    wandb_api_key: str | None = Field(default=None, alias="WANDB_API_KEY")


def load_config(path: Path | str | None = None) -> ProjectConfig:
    """Load ``ProjectConfig`` from YAML, falling back to defaults when absent."""
    config_path = Path(path) if path is not None else PROJECT_ROOT / "configs" / "base.yaml"
    if not config_path.is_file():
        return ProjectConfig()

    raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if raw is None:
        return ProjectConfig()
    if not isinstance(raw, dict):
        raise TypeError(f"{config_path} must contain a YAML mapping, got {type(raw).__name__}")
    return ProjectConfig(**raw)


def bootstrap_env(config: ProjectConfig | None = None) -> Paths:
    """Pin every ML cache into ``.artifacts/`` and create the tree.

    This is what keeps the project's downloads out of the shared
    ``~/.cache/huggingface`` on the lab machine, so teardown stays surgical.
    Call before importing ``huggingface_hub`` / ``transformers`` / ``torch``.
    """
    paths = (config or load_config()).paths
    paths.ensure()

    os.environ.setdefault("HF_HOME", str(paths.hub))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(paths.hub))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(paths.hub))
    os.environ.setdefault("HF_DATASETS_CACHE", str(paths.datasets))
    os.environ.setdefault("TORCH_HOME", str(paths.artifacts / "torch"))
    # Tokenizer threads fight the dataloader workers and make latency noisy.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return paths
