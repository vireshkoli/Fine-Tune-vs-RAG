"""Groundedness: is a RAG arm's answer actually supported by what it retrieved?

A RAG arm can be right for the wrong reason — the model already knew the answer
and the retrieved passages were irrelevant. Accuracy alone cannot tell those
apart, so retrieval gets measured on its own terms:

``retrieval_hit``
    Did any retrieved passage contain the gold answer text? This is the honest
    ceiling on what retrieval could possibly contribute.
``support``
    Lexical overlap between the chosen option and the retrieved context.

Both are cheap and deterministic, which is the point: they run on every item of
every RAG arm, unlike the LLM judge, which is reserved for free-text scoring in
Phase 6. A high accuracy paired with a low retrieval-hit rate is the signature
of an arm whose retrieval is decorative.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from fvr.data.schema import Passage, Question

_WORD = re.compile(r"[a-z0-9]+")
#: Words too common in clinical prose to count as evidence of support.
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
        "which",
        "who",
        "whom",
        "this",
        "these",
        "those",
        "not",
        "no",
        "than",
        "then",
    ]
)


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS and len(w) > 2}


@dataclass(frozen=True)
class GroundednessScore:
    """Per-item retrieval diagnostics."""

    question_id: str
    n_passages: int
    context_chars: int
    #: Fraction of the gold answer's content words present in the context.
    gold_coverage: float
    #: Same, for the option the arm actually chose.
    predicted_coverage: float
    #: True when the gold answer is substantially present in the context.
    retrieval_hit: bool
    top_score: float | None

    @property
    def is_grounded(self) -> bool:
        """Whether the chosen answer is meaningfully supported by the context."""
        return self.predicted_coverage >= 0.5


def score_groundedness(
    question: Question,
    passages: Sequence[Passage],
    predicted_idx: int | None,
    *,
    hit_threshold: float = 0.6,
) -> GroundednessScore:
    """Measure how far the retrieved context supports the gold and chosen options."""
    context = " ".join(p.text for p in passages)
    context_words = _content_words(context)

    def coverage(text: str) -> float:
        words = _content_words(text)
        if not words:
            return 0.0
        return len(words & context_words) / len(words)

    gold = (
        coverage(question.options[question.answer_idx]) if question.answer_idx is not None else 0.0
    )
    predicted = coverage(question.options[predicted_idx]) if predicted_idx is not None else 0.0

    return GroundednessScore(
        question_id=question.id,
        n_passages=len(passages),
        context_chars=len(context),
        gold_coverage=gold,
        predicted_coverage=predicted,
        retrieval_hit=gold >= hit_threshold,
        top_score=passages[0].score if passages else None,
    )


@dataclass(frozen=True)
class GroundednessSummary:
    """Aggregate retrieval diagnostics for one arm."""

    n: int
    retrieval_hit_rate: float
    mean_gold_coverage: float
    mean_predicted_coverage: float
    grounded_rate: float
    mean_passages: float
    mean_context_chars: float

    @classmethod
    def from_scores(cls, scores: Sequence[GroundednessScore]) -> GroundednessSummary:
        if not scores:
            raise ValueError("no groundedness scores to summarise")
        n = len(scores)
        return cls(
            n=n,
            retrieval_hit_rate=sum(s.retrieval_hit for s in scores) / n,
            mean_gold_coverage=sum(s.gold_coverage for s in scores) / n,
            mean_predicted_coverage=sum(s.predicted_coverage for s in scores) / n,
            grounded_rate=sum(s.is_grounded for s in scores) / n,
            mean_passages=sum(s.n_passages for s in scores) / n,
            mean_context_chars=sum(s.context_chars for s in scores) / n,
        )

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n": self.n,
            "retrieval_hit_rate": self.retrieval_hit_rate,
            "mean_gold_coverage": self.mean_gold_coverage,
            "mean_predicted_coverage": self.mean_predicted_coverage,
            "grounded_rate": self.grounded_rate,
            "mean_passages": self.mean_passages,
            "mean_context_chars": self.mean_context_chars,
        }

    def __str__(self) -> str:
        return (
            f"hit={self.retrieval_hit_rate:.3f} grounded={self.grounded_rate:.3f} "
            f"ctx={self.mean_context_chars:.0f} chars over {self.mean_passages:.1f} passages"
        )
