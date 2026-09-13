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
import re
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


_OPTION_LETTER = r"(?:[a-e]|i{1,3}|iv|vi{0,3})"

#: Gold answers that only mean something relative to the hidden options, or to
#: items enumerated in the stem. With the options removed there is nothing for a
#: judge to grade: no free-text answer can "agree with" `B>A>D>C`.
#:
#: Found by the first smoke run, not anticipated. Validated by reading every hit
#: on the frozen test split — 23 of 1,000 items, each genuinely ungradable. A
#: first, looser version also flagged `3.1` (a Mount & Hume class code), `3-5%`
#: and `Both GH and prolactin`, all perfectly gradable; the rules below are the
#: tightened ones, and those three are pinned as negatives in the tests.
OPTION_DEPENDENT_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "all/none of the above",
        re.compile(r"\b(all|none)\b.{0,15}\b(above|of these)\b|^\s*(all|none)\s*$", re.I),
    ),
    (
        "letter or ordering combination",
        re.compile(rf"^\W*{_OPTION_LETTER}(?:\s*(?:>|,|&|and|-)\s*{_OPTION_LETTER})+\W*$", re.I),
    ),
    (
        "numbered-statement combination",
        re.compile(r"^\W*[1-9](?:\s*(?:,|&|and)\s*[1-9])+\W*$", re.I),
    ),
    ("true/false grid", re.compile(r"(?:\b[A-E][\.\)]\s*\S+\s*){3,}")),
    (
        "both/neither of the options",
        re.compile(
            r"^\s*(both|neither)\s*(?:of\s+)?(?:the\s+)?(?:above|these)?\s*$"
            r"|^\s*(both|neither)\s+[a-e]\s*(?:and|&|,|nor|or)\s*[a-e]\b",
            re.I,
        ),
    ),
    (
        "option letters in prose",
        re.compile(
            r"\boptions?\s+[a-e]\b|^\W*[a-e](?:\s*,\s*[a-e])+\s+(?:true|false|correct)", re.I
        ),
    ),
)


def option_dependent_reason(question: Question) -> str | None:
    """Why this item cannot be graded with its options hidden, or None if it can."""
    gold = reference_answer(question)
    for name, rule in OPTION_DEPENDENT_RULES:
        if rule.search(gold):
            return name
    return None


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

    Option-dependent items are excluded *before* sampling, so the set is still
    exactly ``n`` gradable items rather than ``n`` minus whatever the sample
    happened to draw.
    """
    labelled = sorted(
        (q for q in questions if q.answer_idx is not None and option_dependent_reason(q) is None),
        key=lambda q: q.id,
    )
    if len(labelled) < n:
        raise ValueError(
            f"asked for {n} items but only {len(labelled)} are labelled and gradable free-text"
        )
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
    #: Test-split items left out because their gold answer needs the options.
    #: Recorded so the free-text set's size is explained, not just stated.
    excluded_option_dependent: int = 0

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
            "excluded_option_dependent": self.excluded_option_dependent,
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
