"""Orchestrates one arm over the frozen test set and writes a result JSON.

Two invariants are enforced here rather than trusted:

* **GPU exclusivity while timing.** Latency is measured only when nothing else
  is using the inference device, so a judge or a training run cannot silently
  inflate p95.
* **Results are self-describing.** Every JSON records the model revision, the
  adapter, the split hash and the package versions that produced it. A number
  that cannot be traced back to the exact conditions that produced it is not
  evidence.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fvr.data.schema import Prediction, Question
from fvr.eval.latency import LatencyRecorder, LatencySummary
from fvr.eval.metrics import bootstrap_interval, minimum_detectable_effect
from fvr.inference.arms import Arm
from fvr.prompts.templates import build_prompt

if TYPE_CHECKING:  # pragma: no cover
    from fvr.inference.engine import InferenceEngine

#: Items timed individually for latency. The rest are scored in batches, which
#: is far faster but says nothing about single-request latency.
DEFAULT_LATENCY_SAMPLES = 60


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None


def environment_fingerprint() -> dict[str, Any]:
    """Everything needed to explain why a number came out the way it did."""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_sha": _git_sha(),
        "timestamp": datetime.now(UTC).isoformat(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
    except ImportError:  # pragma: no cover - CPU-only CI
        info["torch"] = None
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except ImportError:  # pragma: no cover
        pass
    return info


@dataclass
class ArmResult:
    """One arm, one seed, over one frozen split."""

    arm: str
    seed: int
    split_sha256: str
    n_items: int
    accuracy: float
    ci_low: float
    ci_high: float
    latency: LatencySummary
    predictions: list[Prediction]
    model_info: dict[str, Any]
    environment: dict[str, Any]
    minimum_detectable_effect: float
    per_subject: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "seed": self.seed,
            "split_sha256": self.split_sha256,
            "n_items": self.n_items,
            "accuracy": self.accuracy,
            "ci_95": [self.ci_low, self.ci_high],
            "minimum_detectable_effect": self.minimum_detectable_effect,
            "latency": self.latency.as_dict(),
            "per_subject": self.per_subject,
            "model": self.model_info,
            "environment": self.environment,
            "predictions": [
                {
                    "question_id": p.question_id,
                    "predicted_idx": p.predicted_idx,
                    "option_logprobs": p.option_logprobs,
                    "prompt_tokens": p.prompt_tokens,
                }
                for p in self.predictions
            ],
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2) + "\n", encoding="utf-8")


def _per_subject_accuracy(
    questions: list[Question], predictions: list[Prediction]
) -> dict[str, dict[str, float]]:
    by_id = {q.id: q for q in questions}
    buckets: dict[str, list[bool]] = {}
    for pred in predictions:
        question = by_id[pred.question_id]
        correct = pred.is_correct(question)
        if correct is not None:
            buckets.setdefault(question.subject, []).append(correct)
    return {
        subject: {"n": len(values), "accuracy": sum(values) / len(values)}
        for subject, values in sorted(buckets.items())
    }


def run_arm(
    arm: Arm,
    engine: InferenceEngine,
    questions: list[Question],
    *,
    seed: int,
    split_sha256: str,
    batch_size: int = 8,
    latency_samples: int = DEFAULT_LATENCY_SAMPLES,
    retrieve: Any = None,
    progress: Any = None,
) -> ArmResult:
    """Score every question, then time a subset one at a time.

    Throughput and latency are measured separately on purpose: batching makes
    the whole run affordable, but a batched wall-clock divided by batch size is
    not a latency a user would ever experience.
    """
    prompts = [build_prompt(q, retrieve(q) if retrieve is not None else ()) for q in questions]

    predictions: list[Prediction] = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        for question, scored in zip(
            questions[start : start + batch_size], engine.score_batch(chunk), strict=True
        ):
            predictions.append(
                Prediction(
                    question_id=question.id,
                    arm=arm.name,
                    seed=seed,
                    predicted_idx=scored.scores.predicted_idx,
                    option_logprobs=list(scored.scores.logprobs),
                    prompt_tokens=scored.prompt_tokens,
                )
            )
        if progress is not None:
            progress(min(start + batch_size, len(prompts)), len(prompts))

    recorder = LatencyRecorder()
    for prompt in prompts[:latency_samples]:
        recorder.time_call(lambda p=prompt: engine.score_one(p))  # type: ignore[misc]

    by_id = {q.id: q for q in questions}
    correct = [
        result for p in predictions if (result := p.is_correct(by_id[p.question_id])) is not None
    ]
    interval = bootstrap_interval(correct, seed=seed)

    return ArmResult(
        arm=arm.name,
        seed=seed,
        split_sha256=split_sha256,
        n_items=len(correct),
        accuracy=interval.point,
        ci_low=interval.low,
        ci_high=interval.high,
        latency=recorder.summary(),
        predictions=predictions,
        model_info=engine.loaded.describe(),
        environment=environment_fingerprint(),
        minimum_detectable_effect=minimum_detectable_effect(len(correct)),
        per_subject=_per_subject_accuracy(questions, predictions),
    )
