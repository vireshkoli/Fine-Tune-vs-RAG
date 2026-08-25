"""Builds the supervised fine-tuning set from the MedMCQA train split.

Two decisions here shape whether the whole benchmark means anything.

**The training prompt is byte-identical to the evaluation prompt.** It is built
by the same :func:`fvr.prompts.templates.build_prompt` the arms use, not by a
parallel implementation that happens to look similar. If the two drifted, the
fine-tuned arm would gain an advantage from prompt familiarity that has nothing
to do with domain knowledge, and no amount of careful evaluation downstream
would detect it.

**The answer letter comes first, then the explanation.** Evaluation reads the
logits at the first answer position, so a target that opened with prose would
make the fine-tuned arm unscorable by the same method as every other arm.
Putting the letter first keeps scoring identical across arms while still
training on the explanation text — which matters for the parity argument: the
``rag-parity`` index serves exactly these explanations, so the fine-tune has to
see the same information for "weights versus index" to be the only difference.

``include_explanation`` is configurable because "labels only versus labels plus
rationale" is a real ablation, not a settled question.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from fvr.data.schema import OPTION_LABELS, Question
from fvr.prompts.templates import build_prompt
from fvr.retrieval.corpus import strip_answer_key


@dataclass(frozen=True)
class SFTRecord:
    """One training example, in chat form."""

    question_id: str
    messages: list[dict[str, str]]

    @property
    def target(self) -> str:
        return self.messages[-1]["content"]


@dataclass
class SFTStats:
    """What the builder actually produced, so it is reported not assumed."""

    seen: int = 0
    kept: int = 0
    dropped_unlabelled: int = 0
    dropped_leaked: int = 0
    with_explanation: int = 0
    target_chars: int = 0

    @property
    def mean_target_chars(self) -> float:
        return self.target_chars / self.kept if self.kept else 0.0

    def summary(self) -> str:
        return (
            f"{self.seen:,} seen -> {self.kept:,} kept "
            f"({self.with_explanation:,} with rationale, "
            f"mean target {self.mean_target_chars:.0f} chars) | "
            f"dropped: {self.dropped_unlabelled:,} unlabelled, {self.dropped_leaked:,} leaked"
        )


def build_sft_record(question: Question, *, include_explanation: bool) -> SFTRecord | None:
    """One ``Question`` to one chat record, or ``None`` if unusable."""
    if question.answer_idx is None:
        return None

    letter = OPTION_LABELS[question.answer_idx]
    target = letter
    if include_explanation and question.explanation:
        rationale = strip_answer_key(question.explanation)
        if rationale:
            # Letter first so first-token scoring still works; rationale after
            # so the loss covers the knowledge, not only the label.
            target = f"{letter}\n\n{rationale}"

    prompt = build_prompt(question)
    return SFTRecord(
        question_id=question.id,
        messages=[*prompt.as_messages(), {"role": "assistant", "content": target}],
    )


def build_sft_dataset(
    train_questions: Sequence[Question],
    *,
    forbidden_content_hashes: frozenset[str],
    include_explanation: bool = True,
    max_samples: int | None = None,
    seed: int = 42,
) -> tuple[list[SFTRecord], SFTStats]:
    """Build the SFT set, refusing any row that duplicates held-out content.

    Filtering is on content hash rather than id: MedMCQA repeats items across
    splits under fresh ids, so an id filter would leave test questions in the
    training data and inflate every fine-tuned arm.

    Subsampling happens *after* filtering and uses a seeded shuffle, so a
    smaller run is a random subset of the same clean pool rather than the first
    N rows in file order — which would be sorted by subject and badly skewed.
    """
    stats = SFTStats()
    records: list[SFTRecord] = []

    for question in train_questions:
        stats.seen += 1
        if question.answer_idx is None:
            stats.dropped_unlabelled += 1
            continue
        if question.content_hash() in forbidden_content_hashes:
            stats.dropped_leaked += 1
            continue
        record = build_sft_record(question, include_explanation=include_explanation)
        if record is None:  # pragma: no cover - guarded above
            continue
        records.append(record)

    if max_samples is not None and len(records) > max_samples:
        random.Random(seed).shuffle(records)
        records = records[:max_samples]

    for record in records:
        stats.kept += 1
        stats.target_chars += len(record.target)
        if "\n\n" in record.target:
            stats.with_explanation += 1

    return records, stats


def to_hf_dataset(records: Sequence[SFTRecord]) -> object:
    """Wrap records as a ``datasets.Dataset`` with the column trl expects."""
    from datasets import Dataset

    return Dataset.from_dict({"messages": [r.messages for r in records]})
