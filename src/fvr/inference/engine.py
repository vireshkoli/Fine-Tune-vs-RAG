"""The single generate/score path, shared by every arm.

Arms differ only in which weights are loaded and whether passages are supplied.
They do not get their own decoding settings, their own batching, or their own
tokenisation — all of that lives here, so a measured difference between arms
can only come from weights or retrieval.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fvr.inference.mcq import OptionScores, option_token_ids, score_from_logits
from fvr.prompts.templates import BuiltPrompt

if TYPE_CHECKING:  # pragma: no cover
    from fvr.models.loader import LoadedModel


@dataclass(frozen=True)
class ScoredItem:
    """One item's scores plus the token accounting needed to cost it."""

    scores: OptionScores
    prompt_tokens: int


class InferenceEngine:
    """Wraps a loaded model with the project's fixed decoding policy."""

    def __init__(self, loaded: LoadedModel, n_options: int = 4) -> None:
        self.loaded = loaded
        self.tokenizer = loaded.tokenizer
        self.model = loaded.model
        self.option_ids = option_token_ids(self.tokenizer, n_options)

    def render(self, prompt: BuiltPrompt) -> str:
        """Apply the chat template, with thinking pinned off.

        ``enable_thinking`` is passed only when the template accepts it, so the
        same engine works for non-hybrid models without special-casing.
        """
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        try:
            return str(
                self.tokenizer.apply_chat_template(
                    prompt.as_messages(),
                    enable_thinking=self.loaded.config.enable_thinking,
                    **kwargs,
                )
            )
        except TypeError:
            # Template does not accept enable_thinking — nothing to disable.
            return str(self.tokenizer.apply_chat_template(prompt.as_messages(), **kwargs))

    def score_batch(self, prompts: Sequence[BuiltPrompt]) -> list[ScoredItem]:
        """Score a batch of MCQ prompts in one forward pass.

        Left padding is used so the final position of every sequence is the real
        next-token slot; with right padding that position would be a pad token
        and every score would be garbage.
        """
        import torch

        if not prompts:
            return []

        texts = [self.render(p) for p in prompts]
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.loaded.config.max_seq_length,
        ).to(self.model.device)

        with torch.inference_mode():
            logits = self.model(**encoded).logits

        attention = encoded["attention_mask"]
        results: list[ScoredItem] = []
        for i in range(len(prompts)):
            results.append(
                ScoredItem(
                    scores=score_from_logits(logits[i, -1, :], self.option_ids),
                    prompt_tokens=int(attention[i].sum()),
                )
            )
        return results

    def score_one(self, prompt: BuiltPrompt) -> ScoredItem:
        """Single-item scoring. Used for latency, where batching would cheat."""
        return self.score_batch([prompt])[0]
