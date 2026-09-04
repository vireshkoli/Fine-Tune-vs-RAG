"""Builds the demo Space's precomputed payload from committed results.

The Space ships **precomputed-first**: every answer it shows was produced by a
real evaluated run and is read from JSON, so the demo works with zero GPU
quota, costs nothing, and cannot drift from the benchmark. Live inference is a
later flag, not a dependency.

Item selection is deliberately *not* random. A random 60 of 1,000 items is
mostly cases where every arm agrees, which demonstrates nothing. Items are
stratified by disagreement pattern instead, so the demo shows the phenomenon
the benchmark measures — where retrieval rescues an item the weights miss, and
where it does the opposite.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fvr.data.schema import OPTION_LABELS, Question
from fvr.inference.arms import ARMS, ARMS_BY_NAME
from fvr.report.aggregate import Aggregate, RunRecord

#: Reference arms for the disagreement buckets. ``rag-parity`` is the honest
#: comparator for ``qlora``: same base model, same information, one in an index
#: and one in the weights.
BASE_ARM = "base"
FINETUNE_ARM = "qlora"
RETRIEVAL_ARM = "rag-parity"

#: How many items to include per bucket. Small on purpose — the payload is
#: committed to git and shipped to a free Space.
PER_BUCKET = 10


def softmax(logprobs: Sequence[float]) -> list[float]:
    """Normalise option log-probs into a distribution over the four choices.

    The scorer returns raw log-probabilities, which are not normalised across
    options. The demo shows confidence as a percentage, so it needs to be.
    """
    if not logprobs:
        return []
    top = max(logprobs)
    weights = [math.exp(value - top) for value in logprobs]
    total = sum(weights)
    return [w / total for w in weights] if total else [0.0] * len(logprobs)


def bucket_for(*, base_ok: bool | None, finetune_ok: bool | None, retrieval_ok: bool | None) -> str:
    """Which disagreement pattern an item belongs to.

    Ordered most-informative first: the buckets that separate weights from index
    are claimed before the generic agreement buckets.
    """
    if finetune_ok and not retrieval_ok:
        return "weights_only"
    if retrieval_ok and not finetune_ok:
        return "index_only"
    if base_ok and not (finetune_ok or retrieval_ok):
        return "both_broke_it"
    if not base_ok and finetune_ok and retrieval_ok:
        return "both_fixed_it"
    if base_ok and finetune_ok and retrieval_ok:
        return "everyone_right"
    return "everyone_wrong"


@dataclass(frozen=True)
class DemoItem:
    """One question with every arm's precomputed answer."""

    question: Question
    bucket: str
    answers: dict[str, dict[str, Any]]

    def as_json(self) -> dict[str, Any]:
        return {
            "id": self.question.id,
            "question": self.question.question,
            "options": self.question.options,
            "answer_idx": self.question.answer_idx,
            "answer_label": self.question.answer_label,
            "subject": self.question.subject,
            "bucket": self.bucket,
            "answers": self.answers,
        }


def _answer(run: RunRecord, question_id: str) -> dict[str, Any] | None:
    for pred in run.payload["predictions"]:
        if pred["question_id"] == question_id:
            idx = pred.get("predicted_idx")
            probs = softmax(pred.get("option_logprobs") or [])
            return {
                "predicted_idx": idx,
                "predicted_label": OPTION_LABELS[idx] if idx is not None else None,
                "confidence": round(probs[idx], 4) if idx is not None and probs else None,
                "prompt_tokens": pred.get("prompt_tokens"),
            }
    return None


def _index_predictions(run: RunRecord) -> dict[str, dict[str, Any]]:
    return {pred["question_id"]: pred for pred in run.payload["predictions"]}


def select_items(
    aggregate: Aggregate,
    questions: Mapping[str, Question],
    *,
    per_bucket: int = PER_BUCKET,
    seed: int = 42,
) -> list[DemoItem]:
    """Stratified pick across disagreement buckets, deterministic for a seed."""
    runs = {arm: aggregate.for_arm(arm)[0] for arm in aggregate.arms if aggregate.for_arm(arm)}
    for required in (BASE_ARM, FINETUNE_ARM, RETRIEVAL_ARM):
        if required not in runs:
            raise KeyError(f"no committed run for {required!r}; cannot stratify the demo")

    by_arm = {arm: _index_predictions(run) for arm, run in runs.items()}

    def correct(arm: str, qid: str) -> bool | None:
        pred = by_arm[arm].get(qid)
        question = questions.get(qid)
        if pred is None or question is None or question.answer_idx is None:
            return None
        return bool(pred["predicted_idx"] == question.answer_idx)

    buckets: dict[str, list[str]] = {}
    for qid in by_arm[BASE_ARM]:
        if qid not in questions:
            continue
        name = bucket_for(
            base_ok=correct(BASE_ARM, qid),
            finetune_ok=correct(FINETUNE_ARM, qid),
            retrieval_ok=correct(RETRIEVAL_ARM, qid),
        )
        buckets.setdefault(name, []).append(qid)

    rng = random.Random(seed)
    items: list[DemoItem] = []
    for name in sorted(buckets):
        pool = sorted(buckets[name])
        rng.shuffle(pool)
        for qid in pool[:per_bucket]:
            answers = {
                arm: answer
                for arm, run in runs.items()
                if (answer := _answer(run, qid)) is not None
            }
            items.append(DemoItem(question=questions[qid], bucket=name, answers=answers))
    return items


#: Presentation order. ``Aggregate`` yields arms in filename order, which puts
#: ``qlora-rag-parity`` second and reads as noise; the demo shows them in the
#: order the experiment defines them instead.
_ARM_ORDER = {arm.name: i for i, arm in enumerate(ARMS)}


def arm_summaries(aggregate: Aggregate) -> list[dict[str, Any]]:
    """Headline metrics per arm, straight from the run JSONs."""
    summaries = []
    for name in sorted(aggregate.arms, key=lambda n: _ARM_ORDER.get(n, len(_ARM_ORDER))):
        run = aggregate.for_arm(name)[0]
        arm = ARMS_BY_NAME.get(name)
        summaries.append(
            {
                "name": name,
                "description": arm.description if arm else "",
                "headline": bool(arm.headline) if arm else False,
                "corpus": arm.corpus if arm else "none",
                "uses_adapter": bool(arm.uses_adapter) if arm else False,
                "accuracy": run.accuracy,
                "ci_95": list(run.ci),
                "p50_ms": round(run.p50_ms, 1),
                "p95_ms": round(run.p95_ms, 1),
                "prompt_tokens": round(run.mean_prompt_tokens, 1),
                "groundedness": run.groundedness,
            }
        )
    return summaries


def build_payload(
    aggregate: Aggregate,
    questions: Mapping[str, Question],
    *,
    per_bucket: int = PER_BUCKET,
    seed: int = 42,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The complete ``responses.json`` the Space reads."""
    aggregate.assert_same_split()
    reference = aggregate.runs[0]
    items = select_items(aggregate, questions, per_bucket=per_bucket, seed=seed)
    return {
        "schema_version": 1,
        "provenance": {
            "split_sha256": reference.payload["split_sha256"],
            "n_test_items": reference.payload["n_items"],
            "git_sha": reference.payload["environment"]["git_sha"],
            "model": reference.payload["model"]["repo_id"],
            "model_revision": reference.payload["model"]["revision"],
            "seed": reference.seed,
            "note": (
                "Every answer here was produced by a real evaluated run over the frozen "
                "test set. Nothing in this file is generated at demo time."
            ),
        },
        "arms": arm_summaries(aggregate),
        "items": [item.as_json() for item in items],
        **(dict(extra) if extra else {}),
    }


def write_payload(payload: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
