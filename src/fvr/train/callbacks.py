"""Training callbacks: throughput, hardware utilisation, and an OOM guard.

The OOM guard exists because the plan assumes training will fail at least once
and treats that as a design input rather than an accident. A CUDA OOM normally
takes the process down with a stack trace and no indication of where to resume
from; this catches it, writes a resumable marker, and prints the specific knob
to turn — so the recovery is one command instead of an afternoon.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

try:
    from transformers import TrainerCallback
except ImportError:  # pragma: no cover - CPU-only CI has no GPU stack
    # transformers is an optional `gpu` extra, but this module also holds pure
    # functions (latest_checkpoint, the OOM marker) that CI must be able to
    # import and test. Falling back to `object` keeps the module importable
    # without it; the callbacks are only ever *constructed* inside training,
    # which always has transformers installed.
    TrainerCallback = object  # type: ignore[assignment,misc]


class ThroughputCallback(TrainerCallback):
    """Logs samples/s and GPU memory alongside the loss.

    Throughput is recorded because it feeds the cost model: the amortised
    training term in ``fvr.eval.cost`` needs real GPU-seconds, not an estimate.
    """

    def __init__(self, effective_batch_size: int, device: int = 0) -> None:
        self.effective_batch_size = effective_batch_size
        self.device = device
        self._start: float | None = None
        self._last_step = 0

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._start = time.perf_counter()
        self._last_step = state.global_step

    def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **kwargs: Any) -> None:
        if logs is None or self._start is None:
            return
        steps = state.global_step - self._last_step
        if steps <= 0:
            return
        elapsed = time.perf_counter() - self._start
        logs["samples_per_second"] = steps * self.effective_batch_size / max(elapsed, 1e-9)
        try:
            import torch

            if torch.cuda.is_available():
                logs["gpu_mem_reserved_gib"] = torch.cuda.max_memory_reserved(self.device) / 2**30
        except ImportError:  # pragma: no cover - CPU-only CI
            pass
        self._start = time.perf_counter()
        self._last_step = state.global_step


class OOMGuardCallback(TrainerCallback):
    """Writes a resume marker and actionable advice when CUDA runs out of memory."""

    def __init__(self, output_dir: Path, per_device_batch_size: int, seq_length: int) -> None:
        self.output_dir = Path(output_dir)
        self.per_device_batch_size = per_device_batch_size
        self.seq_length = seq_length

    def write_marker(self, state: Any, error: BaseException) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        marker = self.output_dir / "OOM_RESUME.json"
        halved = max(1, self.per_device_batch_size // 2)
        marker.write_text(
            json.dumps(
                {
                    "global_step": getattr(state, "global_step", None),
                    "error": str(error)[:2000],
                    "resume_with": (
                        f"uv run python scripts/04_train.py "
                        f"--config <same> --resume {self.output_dir}"
                    ),
                    "suggestions": [
                        f"halve per_device_batch_size to {halved} and double "
                        "gradient_accumulation_steps to keep the effective batch identical",
                        f"reduce max_seq_length below {self.seq_length}",
                        "confirm gradient_checkpointing is true",
                        "check nvidia-smi for another tenant on the training device",
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return marker


def latest_checkpoint(output_dir: Path) -> Path | None:
    """Newest ``checkpoint-N`` directory, or ``None``.

    Sorted numerically, not lexically: ``checkpoint-1000`` sorts before
    ``checkpoint-200`` as a string, which would resume from the wrong place.
    """
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return None
    checkpoints = [
        p for p in output_dir.glob("checkpoint-*") if p.is_dir() and p.name.split("-")[-1].isdigit()
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: int(p.name.split("-")[-1]))
