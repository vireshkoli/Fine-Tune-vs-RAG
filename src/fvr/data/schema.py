"""Canonical data types.

Every dataset is normalised into these on load, so nothing downstream — the
retriever, the arms, the scorer — has to know which corpus a row came from.
Answers are stored as an index into ``options`` rather than a letter, because
the contamination probe permutes option order and a letter would silently
become wrong.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Fixed option labels. Four options everywhere; MedMCQA and MedQA both use four.
OPTION_LABELS: tuple[str, ...] = ("A", "B", "C", "D")


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    #: Labelled rows deliberately held back — never trained on, never evaluated on.
    RESERVE = "reserve"


class Question(BaseModel):
    """One multiple-choice item, normalised."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    question: str = Field(min_length=1)
    options: Annotated[list[str], Field(min_length=2, max_length=4)]
    #: Index into ``options``. ``None`` means the label is withheld upstream
    #: (MedMCQA ships ``cop == -1`` for its test split), which makes the row unusable.
    answer_idx: int | None = None
    subject: str = "Unknown"
    #: Source explanation. Used to build the parity retrieval index, never shown at eval.
    explanation: str | None = None
    source: str = "medmcqa"

    @model_validator(mode="after")
    def _answer_in_range(self) -> Self:
        if self.answer_idx is not None and not 0 <= self.answer_idx < len(self.options):
            raise ValueError(
                f"answer_idx {self.answer_idx} out of range for {len(self.options)} options"
            )
        return self

    @property
    def is_labelled(self) -> bool:
        return self.answer_idx is not None

    @property
    def answer_label(self) -> str | None:
        """The letter, derived rather than stored so permutation cannot desync it."""
        return None if self.answer_idx is None else OPTION_LABELS[self.answer_idx]

    def content_hash(self) -> str:
        """Stable hash of the semantic content, ignoring id and metadata.

        Used for cross-split leakage detection: MedMCQA contains near-duplicate
        items, and the same question appearing in train and test would inflate
        every arm that trained on it.
        """
        payload = "␟".join(
            [self.question.strip().lower(), *(o.strip().lower() for o in self.options)]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Passage(BaseModel):
    """A retrievable chunk, carrying enough provenance to cite it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    text: str = Field(min_length=1)
    #: Which index this belongs to — ``parity`` (train explanations) or ``external``.
    corpus: str
    #: For the parity corpus, the train question this text came from. Lets the
    #: leakage test prove no test-derived text ever entered an index.
    source_question_id: str | None = None
    title: str | None = None
    score: float | None = None


class Prediction(BaseModel):
    """One arm's answer to one question, with everything needed to score and cost it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: str
    arm: str
    seed: int
    predicted_idx: int | None = None
    #: Log-probability per option, from constrained scoring. Enables calibration analysis.
    option_logprobs: list[float] | None = None
    generated_text: str | None = None
    retrieved: list[Passage] = Field(default_factory=list)
    latency_s: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    def is_correct(self, question: Question) -> bool | None:
        """``None`` when unscorable, so abstentions never silently count as wrong."""
        if question.answer_idx is None or self.predicted_idx is None:
            return None
        return self.predicted_idx == question.answer_idx
