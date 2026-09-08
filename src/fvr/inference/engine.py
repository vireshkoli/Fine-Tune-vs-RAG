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


@dataclass(frozen=True)
class GeneratedItem:
    """One free-text answer plus the token accounting needed to cost it."""

    text: str
    prompt_tokens: int
    completion_tokens: int


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

    def generate_batch(
        self, prompts: Sequence[BuiltPrompt], *, max_new_tokens: int = 96
    ) -> list[GeneratedItem]:
        """Generate free-text answers for a batch.

        Greedy, like every other decode in this project: sampling would add
        variance that is not the variance being studied, and would make the
        judge's seed-to-seed spread impossible to separate from the model's.

        ``max_new_tokens`` is a hard bound rather than a suggestion. The prompt
        asks for one or two sentences, but an arm that ignored that and rambled
        would inflate its own latency and cost while handing the judge more
        surface area to reward — so the cap is enforced here, identically for
        every arm, rather than trusted to the instruction.
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
            output = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        attention = encoded["attention_mask"]
        prompt_length = encoded["input_ids"].shape[1]
        results: list[GeneratedItem] = []
        for i in range(len(prompts)):
            # Left padding means every sequence's completion starts at the same
            # offset, so the prompt can be sliced off by width rather than by
            # searching for it in the decoded string.
            completion = output[i, prompt_length:]
            text = str(self.tokenizer.decode(completion, skip_special_tokens=True))
            results.append(
                GeneratedItem(
                    text=text.strip(),
                    prompt_tokens=int(attention[i].sum()),
                    completion_tokens=int((completion != self.tokenizer.pad_token_id).sum()),
                )
            )
        return results

    def generate_one(self, prompt: BuiltPrompt, *, max_new_tokens: int = 96) -> GeneratedItem:
        """Single-item generation. Used for latency, where batching would cheat."""
        return self.generate_batch([prompt], max_new_tokens=max_new_tokens)[0]
