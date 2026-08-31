"""QLoRA fine-tuning, config-driven.

Resumption is built in from the start rather than added after a run is lost:
``save_steps`` is tuned so an interruption costs minutes, ``train`` accepts a
checkpoint to continue from, and the OOM guard writes the exact command to
resume with. The plan treats "training will fail at least once" as a design
input, and this module is where that belief is cashed out.

The base model is loaded through :mod:`fvr.models.loader`, the same entry point
the evaluation arms use, so "the fine-tune started from the identical weights"
is a property of the code rather than a claim in the README.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fvr.models.loader import ModelConfig
from fvr.train.callbacks import OOMGuardCallback, ThroughputCallback, latest_checkpoint
from fvr.train.config import TrainConfig


@dataclass
class TrainResult:
    """What a run produced, including the GPU-seconds the cost model needs."""

    output_dir: Path
    train_gpu_seconds: float
    final_step: int
    train_loss: float | None
    best_eval_loss: float | None
    resumed_from: str | None
    n_train: int
    n_eval: int
    trainable_params: int
    total_params: int

    @property
    def trainable_pct(self) -> float:
        return 100.0 * self.trainable_params / self.total_params if self.total_params else 0.0

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "output_dir": str(self.output_dir),
                    "train_gpu_seconds": self.train_gpu_seconds,
                    "final_step": self.final_step,
                    "train_loss": self.train_loss,
                    "best_eval_loss": self.best_eval_loss,
                    "resumed_from": self.resumed_from,
                    "n_train": self.n_train,
                    "n_eval": self.n_eval,
                    "trainable_params": self.trainable_params,
                    "total_params": self.total_params,
                    "trainable_pct": self.trainable_pct,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def build_peft_model(model: Any, config: TrainConfig) -> Any:
    """Attach LoRA adapters to a quantised base."""
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model, prepare_model_for_kbit_training

    if config.quantization != "none":
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.gradient_checkpointing
        )

    return get_peft_model(
        model,
        PeftLoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            target_modules=list(config.lora.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )


def count_parameters(model: Any) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def train(
    config: TrainConfig,
    model_config: ModelConfig,
    train_dataset: Any,
    eval_dataset: Any,
    output_dir: Path,
    *,
    resume: bool = True,
    max_steps: int | None = None,
) -> TrainResult:
    """Run QLoRA fine-tuning and return the artefacts plus timing.

    ``eval_dataset`` is the **validation** split. Checkpoint selection uses it
    and nothing else; the test split is never loaded here, which
    ``tests/test_train.py`` asserts directly.
    """
    import torch
    from trl import SFTConfig, SFTTrainer  # type: ignore[attr-defined]

    from fvr.models.loader import load_base_model

    output_dir = Path(output_dir)
    # str(None) is the string "None", which Trainer then treats as a real path
    # and dies with "Can't find a valid checkpoint at None". Convert only when
    # a checkpoint actually exists.
    checkpoint = latest_checkpoint(output_dir) if resume else None
    resume_from = str(checkpoint) if checkpoint is not None else None

    quantised = model_config.model_copy(update={"quantization": config.quantization})
    loaded = load_base_model(quantised, use_cache=False)
    model = build_peft_model(loaded.model, config)
    trainable, total = count_parameters(model)

    sft_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.epochs,
        max_steps=max_steps if max_steps is not None else -1,
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        gradient_checkpointing=config.gradient_checkpointing,
        max_length=config.max_seq_length,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=False,
        bf16=True,
        optim=config.optim,
        seed=config.seed,
        report_to=config.report_to,
        run_name=config.name,
    )

    guard = OOMGuardCallback(output_dir, config.per_device_batch_size, config.max_seq_length)
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=loaded.tokenizer,
        callbacks=[ThroughputCallback(config.effective_batch_size, model_config.device), guard],
    )

    started = time.perf_counter()
    try:
        output = trainer.train(resume_from_checkpoint=resume_from)
    except torch.OutOfMemoryError as exc:
        marker = guard.write_marker(trainer.state, exc)
        raise RuntimeError(
            f"CUDA OOM at step {trainer.state.global_step}. Wrote {marker} with resume "
            "instructions; progress up to the last checkpoint is intact."
        ) from exc
    elapsed = time.perf_counter() - started

    trainer.save_model(str(output_dir / "adapter"))
    loaded.tokenizer.save_pretrained(str(output_dir / "adapter"))

    eval_losses = [h["eval_loss"] for h in trainer.state.log_history if "eval_loss" in h]

    return TrainResult(
        output_dir=output_dir,
        train_gpu_seconds=elapsed,
        final_step=trainer.state.global_step,
        train_loss=float(output.training_loss) if output is not None else None,
        best_eval_loss=min(eval_losses) if eval_losses else None,
        resumed_from=resume_from,
        n_train=len(train_dataset),
        n_eval=len(eval_dataset),
        trainable_params=trainable,
        total_params=total,
    )
