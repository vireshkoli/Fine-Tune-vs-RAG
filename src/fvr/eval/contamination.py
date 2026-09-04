"""Contamination probes.

MedMCQA predates Qwen3 and is a popular public benchmark, so assuming it is in
the pretraining corpus is the safe prior. We cannot inspect that corpus, so
these probes test for the *symptoms* of memorisation using only the model:

``permutation``
    Shuffle the answer options and re-score. A model that reasons over content
    is unaffected — the same option is still correct, merely relabelled. A model
    that memorised "the answer to this item is C" degrades. The accuracy drop is
    the contamination signal, and it is the strongest of the three because it
    holds the information constant and changes only the surface form.

``position bias``
    Falls out of the same run for free. If predictions concentrate on one letter
    regardless of content, the arm is exploiting label priors rather than
    medicine, which inflates accuracy on any benchmark with an uneven answer
    distribution.

``verbatim completion``
    Feed a truncated question stem and let the model continue. High overlap with
    the true remainder is direct evidence the item was memorised, not inferred.

None of these prove contamination; they bound it. A large permutation drop means
some of the headline accuracy is recall rather than reasoning, and that belongs
in the README even though it weakens the number.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass

from fvr.data.schema import OPTION_LABELS, Question

_WORD = re.compile(r"[a-z0-9]+")


def permute_options(question: Question, rng: random.Random) -> tuple[Question, list[int]]:
    """Return the question with shuffled options, and the permutation applied.

    ``answer_idx`` is remapped so the *same option text* remains correct. The
    item's information content is untouched; only the letter attached to the
    right answer moves.
    """
    order = list(range(len(question.options)))
    rng.shuffle(order)
    options = [question.options[i] for i in order]
    answer = None if question.answer_idx is None else order.index(question.answer_idx)
    return question.model_copy(update={"options": options, "answer_idx": answer}), order


def permute_dataset(
    questions: Sequence[Question], *, seed: int
) -> tuple[list[Question], dict[str, list[int]]]:
    """Permute every question with one seeded RNG, so the shuffle is reproducible."""
    rng = random.Random(seed)
    permuted: list[Question] = []
    orders: dict[str, list[int]] = {}
    for question in questions:
        item, order = permute_options(question, rng)
        permuted.append(item)
        orders[question.id] = order
    return permuted, orders


@dataclass(frozen=True)
class PermutationResult:
    """Accuracy before and after shuffling the option labels."""

    original_accuracy: float
    permuted_accuracy: float
    n: int

    @property
    def drop(self) -> float:
        return self.original_accuracy - self.permuted_accuracy

    @property
    def relative_drop(self) -> float:
        if self.original_accuracy <= 0:
            return 0.0
        return self.drop / self.original_accuracy

    def verdict(self, *, tolerance: float = 0.03) -> str:
        """Interpretation, stated plainly rather than left to the reader."""
        if self.drop <= tolerance:
            return (
                f"no evidence of positional memorisation "
                f"(drop {self.drop:+.3f} within +/-{tolerance:.2f} noise)"
            )
        if self.drop <= 3 * tolerance:
            return f"mild positional sensitivity (drop {self.drop:+.3f})"
        return (
            f"substantial positional sensitivity (drop {self.drop:+.3f}) — some accuracy "
            "is label recall rather than reasoning"
        )

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "n": self.n,
            "original_accuracy": self.original_accuracy,
            "permuted_accuracy": self.permuted_accuracy,
            "drop": self.drop,
            "relative_drop": self.relative_drop,
            "verdict": self.verdict(),
        }


@dataclass(frozen=True)
class PositionBias:
    """How often each answer letter is predicted, against how often it is correct."""

    predicted: dict[str, float]
    gold: dict[str, float]

    @property
    def max_excess(self) -> float:
        """Largest gap between how often a letter is chosen and how often it is right."""
        return max(abs(self.predicted[k] - self.gold.get(k, 0.0)) for k in self.predicted)

    def as_dict(self) -> dict[str, object]:
        return {
            "predicted_rate": self.predicted,
            "gold_rate": self.gold,
            "max_excess": self.max_excess,
        }


def position_bias(
    predicted_idx: Sequence[int | None], gold_idx: Sequence[int | None], n_options: int = 4
) -> PositionBias:
    labels = OPTION_LABELS[:n_options]
    total_pred = sum(1 for p in predicted_idx if p is not None) or 1
    total_gold = sum(1 for g in gold_idx if g is not None) or 1
    return PositionBias(
        predicted={
            label: sum(1 for p in predicted_idx if p == i) / total_pred
            for i, label in enumerate(labels)
        },
        gold={
            label: sum(1 for g in gold_idx if g == i) / total_gold for i, label in enumerate(labels)
        },
    )


@dataclass(frozen=True)
class VerbatimResult:
    """How closely the model reproduces the withheld remainder of a stem."""

    n: int
    mean_overlap: float
    high_overlap_rate: float
    threshold: float

    def verdict(self) -> str:
        if self.high_overlap_rate < 0.05:
            return "no verbatim reproduction detected"
        if self.high_overlap_rate < 0.15:
            return f"occasional verbatim reproduction ({self.high_overlap_rate:.1%} of items)"
        return (
            f"frequent verbatim reproduction ({self.high_overlap_rate:.1%} of items) — "
            "direct evidence these items were memorised"
        )

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "n": self.n,
            "mean_overlap": self.mean_overlap,
            "high_overlap_rate": self.high_overlap_rate,
            "threshold": self.threshold,
            "verdict": self.verdict(),
        }


def token_overlap(a: str, b: str) -> float:
    """Fraction of the reference's content words that appear in the continuation."""
    reference = set(_WORD.findall(b.lower()))
    if not reference:
        return 0.0
    return len(reference & set(_WORD.findall(a.lower()))) / len(reference)


def split_stem(question: str, *, fraction: float = 0.5) -> tuple[str, str]:
    """Split a stem into a prompt prefix and the withheld remainder."""
    words = question.split()
    cut = max(1, int(len(words) * fraction))
    return " ".join(words[:cut]), " ".join(words[cut:])


def summarise_verbatim(
    continuations: Sequence[str], references: Sequence[str], *, threshold: float = 0.75
) -> VerbatimResult:
    overlaps = [token_overlap(c, r) for c, r in zip(continuations, references, strict=True)]
    if not overlaps:
        raise ValueError("no continuations to summarise")
    return VerbatimResult(
        n=len(overlaps),
        mean_overlap=sum(overlaps) / len(overlaps),
        high_overlap_rate=sum(o >= threshold for o in overlaps) / len(overlaps),
        threshold=threshold,
    )
