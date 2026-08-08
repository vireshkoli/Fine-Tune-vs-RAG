"""Prompt construction — one template, shared by every arm.

This module is the fairness control. If arms were allowed to build their own
prompts, any accuracy difference would be confounded by wording, and the whole
benchmark would be worthless. So there is exactly one builder, and the *only*
permitted difference between arms is whether a retrieved-context block is
inserted. ``tests/test_prompts.py`` asserts that property directly.

``enable_thinking=False`` is pinned for every arm. Qwen3 is a hybrid reasoning
model, and variable-length thinking traces would confound both latency and
cost — an arm that happened to think longer would look slower and pricier for
reasons unrelated to fine-tuning or retrieval.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from fvr.data.schema import OPTION_LABELS, Passage, Question

SYSTEM_PROMPT = (
    "You are a medical exam assistant. Answer the multiple-choice question by "
    "selecting the single best option. Respond with only the letter of your "
    "chosen option."
)

#: Instruction appended to every question, identical across arms.
ANSWER_INSTRUCTION = "Answer with a single letter (A, B, C, or D)."

CONTEXT_HEADER = (
    "Use the following reference passages if they are relevant. "
    "If they are not relevant, rely on your own knowledge."
)


@dataclass(frozen=True)
class BuiltPrompt:
    """A prompt plus the metadata needed to score and cost it."""

    system: str
    user: str
    #: Passages injected, in the order shown. Empty for non-RAG arms.
    context: tuple[Passage, ...] = ()

    @property
    def has_context(self) -> bool:
        return bool(self.context)

    def as_messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]


def format_options(question: Question) -> str:
    return "\n".join(
        f"{label}. {text}" for label, text in zip(OPTION_LABELS, question.options, strict=False)
    )


def format_context(passages: Sequence[Passage]) -> str:
    """Numbered passages, so a groundedness check can resolve a citation."""
    blocks = [f"[{i}] {p.text.strip()}" for i, p in enumerate(passages, start=1)]
    return f"{CONTEXT_HEADER}\n\n" + "\n\n".join(blocks)


def build_prompt(question: Question, passages: Sequence[Passage] = ()) -> BuiltPrompt:
    """Build the prompt for any arm.

    Retrieval-free and retrieval-augmented arms share every token except the
    context block, which is prepended when — and only when — passages are given.
    """
    body = f"{question.question.strip()}\n\n{format_options(question)}\n\n{ANSWER_INSTRUCTION}"
    user = f"{format_context(passages)}\n\n---\n\n{body}" if passages else body
    return BuiltPrompt(system=SYSTEM_PROMPT, user=user, context=tuple(passages))


def strip_context(prompt: BuiltPrompt) -> BuiltPrompt:
    """The same prompt with its context block removed.

    Used by the parity test: stripping context from a RAG prompt must yield
    exactly the non-RAG prompt, which proves the arms differ in nothing else.
    """
    if not prompt.has_context:
        return prompt
    _, _, body = prompt.user.partition("\n\n---\n\n")
    return BuiltPrompt(system=prompt.system, user=body, context=())
