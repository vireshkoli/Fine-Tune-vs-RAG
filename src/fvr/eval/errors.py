"""Error analysis over committed predictions.

Runs entirely on the JSON already in ``results/runs/`` — no model, no GPU. That
matters beyond convenience: it means the error taxonomy can be recomputed by
anyone who clones the repo, long after the lab machine is wiped.

The categories are chosen to answer questions the accuracy table cannot:

``fixed_by_retrieval`` / ``broken_by_retrieval``
    Items one arm gets right and the other wrong. The counts are the same
    discordant pairs McNemar tests, but here they carry their text, so a reader
    can see *what kind* of question moved rather than only how many.
``confident_wrong``
    Wrong with a large log-prob margin. These are the dangerous failures in a
    clinical setting: the model is not hedging, it is confidently mistaken.
``near_miss``
    Wrong with a tiny margin — the right answer was the runner-up. Different
    problem, different fix.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fvr.data.schema import OPTION_LABELS, Question

#: Log-prob gap between the chosen option and the runner-up above which a wrong
#: answer counts as confident rather than a coin flip.
CONFIDENT_MARGIN = 1.0
NEAR_MISS_MARGIN = 0.15


@dataclass(frozen=True)
class ErrorCase:
    """One scored item, with everything needed to judge it by eye."""

    question_id: str
    subject: str
    question: str
    options: list[str]
    gold_label: str
    predicted_label: str
    margin: float
    category: str
    arm: str
    retrieved_titles: list[str]

    def as_row(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "category": self.category,
            "question_id": self.question_id,
            "subject": self.subject,
            "gold": self.gold_label,
            "predicted": self.predicted_label,
            "margin": round(self.margin, 3),
            "question": self.question,
            "gold_text": self.options[OPTION_LABELS.index(self.gold_label)],
            "predicted_text": self.options[OPTION_LABELS.index(self.predicted_label)],
            "retrieved": " | ".join(self.retrieved_titles[:3]),
        }


def _margin(logprobs: Sequence[float] | None) -> float:
    if not logprobs or len(logprobs) < 2:
        return 0.0
    ordered = sorted(logprobs, reverse=True)
    return ordered[0] - ordered[1]


def categorise(
    question: Question,
    predicted_idx: int | None,
    logprobs: Sequence[float] | None,
    *,
    comparison_correct: bool | None = None,
) -> str:
    """Label one item.

    ``comparison_correct`` is whether a *reference* arm got it right, which is
    what turns a plain error into "retrieval broke this" or "retrieval fixed
    this".
    """
    if predicted_idx is None or question.answer_idx is None:
        return "unscorable"
    correct = predicted_idx == question.answer_idx
    margin = _margin(logprobs)

    if comparison_correct is not None:
        if correct and not comparison_correct:
            return "fixed_by_retrieval"
        if not correct and comparison_correct:
            return "broken_by_retrieval"
    if correct:
        return "correct"
    if margin >= CONFIDENT_MARGIN:
        return "confident_wrong"
    if margin <= NEAR_MISS_MARGIN:
        return "near_miss"
    return "wrong"


def collect_errors(
    run_payload: dict[str, Any],
    questions: dict[str, Question],
    *,
    reference_payload: dict[str, Any] | None = None,
) -> list[ErrorCase]:
    """Every non-correct item in a run, categorised."""
    reference_correct: dict[str, bool] = {}
    if reference_payload is not None:
        for pred in reference_payload["predictions"]:
            question = questions.get(pred["question_id"])
            if question is not None and question.answer_idx is not None:
                reference_correct[pred["question_id"]] = (
                    pred["predicted_idx"] == question.answer_idx
                )

    cases: list[ErrorCase] = []
    for pred in run_payload["predictions"]:
        question = questions.get(pred["question_id"])
        if question is None or question.answer_idx is None:
            continue
        category = categorise(
            question,
            pred.get("predicted_idx"),
            pred.get("option_logprobs"),
            comparison_correct=reference_correct.get(pred["question_id"]),
        )
        if category in {"correct", "unscorable"}:
            continue
        predicted_idx = pred.get("predicted_idx")
        cases.append(
            ErrorCase(
                question_id=question.id,
                subject=question.subject,
                question=question.question,
                options=question.options,
                gold_label=OPTION_LABELS[question.answer_idx],
                predicted_label=OPTION_LABELS[predicted_idx] if predicted_idx is not None else "?",
                margin=_margin(pred.get("option_logprobs")),
                category=category,
                arm=str(run_payload["arm"]),
                retrieved_titles=[],
            )
        )
    return cases


def sample_for_review(
    cases: Sequence[ErrorCase], *, per_category: int = 10, seed: int = 42
) -> list[ErrorCase]:
    """A stratified sample to hand-inspect, deterministic for a given seed.

    Stratified rather than random: the rare categories are the informative ones,
    and a flat sample of 10 from a 400-error pool would return almost nothing
    but ordinary wrong answers.
    """
    import random

    rng = random.Random(seed)
    by_category: dict[str, list[ErrorCase]] = {}
    for case in cases:
        by_category.setdefault(case.category, []).append(case)

    sampled: list[ErrorCase] = []
    for category in sorted(by_category):
        bucket = sorted(by_category[category], key=lambda c: c.question_id)
        rng.shuffle(bucket)
        sampled.extend(bucket[:per_category])
    return sampled


def category_counts(cases: Sequence[ErrorCase]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        counts[case.category] = counts.get(case.category, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def write_review_csv(cases: Sequence[ErrorCase], path: Path) -> None:
    """A CSV to open in a spreadsheet and annotate by hand."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [case.as_row() for case in cases]
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*rows[0].keys(), "human_label", "notes"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "human_label": "", "notes": ""})


def write_summary(counts: dict[str, dict[str, int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(counts, indent=2, sort_keys=True) + "\n", encoding="utf-8")
