"""The free-text arms: generate an answer, then have a judge grade it.

Why this arm exists. Constrained A/B/C/D scoring measures whether the model can
*rank four candidates*, which is a narrower and easier task than producing the
answer unaided — a model that has never heard of a condition can still eliminate
three implausible options. So every MCQ number in this project, including the
headline, measures something weaker than "does it know the medicine". This arm
is what separates the two, and it is the reason the MCQ results carry the
caveat rather than being presented as knowledge.

Reference answers are the gold *option text*, not the explanation. Explanations
in MedMCQA are OCR-damaged and frequently open with answer-key boilerplate, so
grading against them would grade the judge's tolerance for noise. The option
text is short, clean, and is exactly what "the answer" means for an exam item.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fvr.data.schema import Question

#: The plan sized the free-text set at 300: enough to separate arms that differ
#: by a few points, small enough that 6 arms x 3 judge seeds stays affordable.
DEFAULT_N_ITEMS = 300


class UnlabelledQuestionError(Exception):
    """A question with no gold answer cannot be a free-text reference."""


def reference_answer(question: Question) -> str:
    """The gold answer text, used as the judge's reference."""
    if question.answer_idx is None:
        raise UnlabelledQuestionError(f"{question.id} has no gold answer")
    return question.options[question.answer_idx].strip()


def select_freetext_items(
    questions: Sequence[Question], *, n: int = DEFAULT_N_ITEMS, seed: int = 42
) -> list[Question]:
    """A deterministic subset of the frozen test split.

    Sorted by id before sampling so the selection depends only on the seed, not
    on the order the caller happened to load the split in. Every arm must be
    given the *same* items or the judged comparison is not paired.
    """
    labelled = sorted((q for q in questions if q.answer_idx is not None), key=lambda q: q.id)
    if len(labelled) < n:
        raise ValueError(f"asked for {n} items but only {len(labelled)} are labelled")
    rng = random.Random(seed)
    chosen = rng.sample(labelled, n)
    return sorted(chosen, key=lambda q: q.id)


@dataclass(frozen=True)
class FreeTextAnswer:
    """One generated answer, with the accounting needed to cost it."""

    question_id: str
    subject: str
    question: str
    reference: str
    answer: str
    prompt_tokens: int
    completion_tokens: int
    n_passages: int = 0

    def as_json(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "subject": self.subject,
            "question": self.question,
            "reference": self.reference,
            "answer": self.answer,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "n_passages": self.n_passages,
        }


@dataclass
class FreeTextRun:
    """Every generated answer for one arm, before any judging."""

    arm: str
    seed: int
    split_sha256: str
    model: dict[str, Any]
    environment: dict[str, Any]
    answers: list[FreeTextAnswer] = field(default_factory=list)
    latency: dict[str, float | int] = field(default_factory=dict)
    retrieval: dict[str, Any] | None = None
    device_occupancy: dict[str, Any] | None = None

    @property
    def empty_answers(self) -> int:
        """Answers that came back blank.

        Reported rather than dropped: an arm that refuses or emits nothing is
        failing, and silently excluding those items would score it on the subset
        where it happened to speak.
        """
        return sum(1 for a in self.answers if not a.answer.strip())

    @property
    def mean_completion_tokens(self) -> float:
        if not self.answers:
            return 0.0
        return sum(a.completion_tokens for a in self.answers) / len(self.answers)

    def to_json(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "seed": self.seed,
            "split_sha256": self.split_sha256,
            "n_items": len(self.answers),
            "empty_answers": self.empty_answers,
            "mean_completion_tokens": round(self.mean_completion_tokens, 2),
            "latency": self.latency,
            "retrieval": self.retrieval,
            "device_occupancy": self.device_occupancy,
            "model": self.model,
            "environment": self.environment,
            "answers": [a.as_json() for a in self.answers],
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2, ensure_ascii=False) + "\n", "utf-8")


def load_run(path: Path) -> dict[str, Any]:
    return dict(json.loads(Path(path).read_text(encoding="utf-8")))
