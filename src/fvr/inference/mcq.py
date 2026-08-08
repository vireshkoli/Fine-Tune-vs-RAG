"""Constrained multiple-choice scoring.

The model is never asked to *generate* an answer letter. Instead the logits at
the first answer position are read directly and compared across only the A/B/C/D
tokens. Three reasons, all of which matter for the comparison:

* **No parsing.** Free generation means some arms lose points to formatting
  ("The answer is B." vs "B"), which measures instruction-following rather than
  medical knowledge and would systematically favour the fine-tuned arms, since
  fine-tuning teaches format as much as content.
* **No abstentions.** Every item gets a score, so accuracy denominators are
  identical across arms.
* **Speed.** One forward pass per item instead of autoregressive decoding —
  roughly 20x faster, which is what makes 6 arms x 3 seeds affordable.

The per-option log-probabilities are kept, not just the argmax, so calibration
and confidence analysis are possible later without re-running anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fvr.data.schema import OPTION_LABELS

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class OptionScores:
    """Log-probabilities over the answer letters for one item."""

    logprobs: tuple[float, ...]

    @property
    def predicted_idx(self) -> int:
        return max(range(len(self.logprobs)), key=lambda i: self.logprobs[i])

    @property
    def predicted_label(self) -> str:
        return OPTION_LABELS[self.predicted_idx]

    @property
    def margin(self) -> float:
        """Gap between best and runner-up — a cheap confidence signal."""
        ordered = sorted(self.logprobs, reverse=True)
        return ordered[0] - ordered[1] if len(ordered) > 1 else 0.0


def option_token_ids(tokenizer: PreTrainedTokenizerBase, n_options: int = 4) -> list[list[int]]:
    """Token ids that could begin each answer letter.

    A letter can tokenise differently with and without a leading space, and
    casing varies, so every plausible surface form is collected and the option's
    score is the max over them. Missing this makes scoring silently depend on
    tokenizer quirks rather than on the model.
    """
    variants: list[list[int]] = []
    for label in OPTION_LABELS[:n_options]:
        ids: set[int] = set()
        for surface in (label, f" {label}", label.lower(), f" {label.lower()}"):
            encoded = tokenizer.encode(surface, add_special_tokens=False)
            if encoded:
                ids.add(encoded[0])
        if not ids:  # pragma: no cover - would mean a broken tokenizer
            raise ValueError(f"tokenizer produced no ids for option {label!r}")
        variants.append(sorted(ids))
    return variants


def score_from_logits(
    logits: Any,
    option_ids: list[list[int]],
) -> OptionScores:
    """Turn next-token logits into per-option log-probabilities.

    Normalises over the full vocabulary first, then selects the option tokens,
    so the numbers are genuine log-probabilities and comparable across items —
    softmaxing over only four tokens would discard how much mass the model put
    elsewhere.
    """
    import torch

    logprobs = torch.log_softmax(logits.float(), dim=-1)
    scores = [float(torch.max(logprobs[ids])) for ids in option_ids]
    return OptionScores(logprobs=tuple(scores))
