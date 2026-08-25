"""Typed training configuration. No magic numbers in code."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class LoraConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    r: int = Field(default=16, ge=1)
    alpha: int = 32
    dropout: float = 0.05
    #: Attention *and* MLP projections. Attention-only adapters underfit on
    #: knowledge-heavy tasks; the MLP is where most factual capacity lives.
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


class TrainConfig(BaseModel):
    """From ``configs/train/*.yaml``."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    name: str
    model_config_path: str = "configs/model/qwen3-8b.yaml"

    lora: LoraConfig = Field(default_factory=LoraConfig)

    #: NF4 double-quantised base weights — the "Q" in QLoRA. Roughly 5.5GB for
    #: an 8B model, which is what makes this reproducible on a 16GB card.
    quantization: Literal["nf4", "int8", "none"] = "nf4"

    learning_rate: float = 2e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    epochs: float = 2.0
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 1024
    #: Trades ~30% throughput for a large activation-memory saving. Mandatory
    #: below 24GB; kept on everywhere so the recipe is portable.
    gradient_checkpointing: bool = True

    max_train_samples: int | None = 30000
    include_explanation: bool = True

    #: Frequent enough that an interrupted run loses minutes, not hours. The
    #: plan assumes training *will* fail at least once.
    save_steps: int = 200
    eval_steps: int = 200
    logging_steps: int = 10
    save_total_limit: int = 3
    #: Checkpoint selection is on the val split, never on test.
    metric_for_best_model: str = "eval_loss"

    seed: int = 42
    optim: str = "paged_adamw_8bit"
    report_to: str = "wandb"

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_batch_size * self.gradient_accumulation_steps


def load_train_config(path: Path | str) -> TrainConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return TrainConfig(**raw)
