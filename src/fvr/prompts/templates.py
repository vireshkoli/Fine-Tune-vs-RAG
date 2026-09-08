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

#: The free-text arms. Same scaffolding, no options shown, and a length bound —
#: an unbounded answer would make latency and cost incomparable with the MCQ
#: arms, and would give a verbose arm more surface area for a judge to reward.
FREETEXT_SYSTEM_PROMPT = (
    "You are a medical exam assistant. Answer the question directly and "
    "concisely. State your answer; do not explain your reasoning at length."
)

FREETEXT_INSTRUCTION = "Answer in one or two sentences."

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


#: Separates the retrieved context from the question. One constant, used by
#: every builder and by :func:`strip_context`, so the round-trip cannot drift.
CONTEXT_SEPARATOR = "\n\n---\n\n"


def _with_context(body: str, passages: Sequence[Passage]) -> str:
    """Prepend the context block, or return the body unchanged.

    The single place context is inserted. Both the MCQ and free-text builders
    go through it, so the property `strip_context` asserts — that removing the
    context reproduces the non-RAG prompt byte for byte — holds for both without
    being maintained twice.
    """
    if not passages:
        return body
    return f"{format_context(passages)}{CONTEXT_SEPARATOR}{body}"


def build_prompt(question: Question, passages: Sequence[Passage] = ()) -> BuiltPrompt:
    """Build the MCQ prompt for any arm.

    Retrieval-free and retrieval-augmented arms share every token except the
    context block, which is prepended when — and only when — passages are given.
    """
    body = f"{question.question.strip()}\n\n{format_options(question)}\n\n{ANSWER_INSTRUCTION}"
    return BuiltPrompt(
        system=SYSTEM_PROMPT, user=_with_context(body, passages), context=tuple(passages)
    )


def build_freetext_prompt(question: Question, passages: Sequence[Passage] = ()) -> BuiltPrompt:
    """Build the open-ended prompt for the free-text arms.

    The answer options are deliberately *not* shown. That is the whole point of
    the arm: constrained A/B/C/D scoring measures the model's ability to rank
    four candidates, which is an easier and narrower task than producing the
    answer unaided. A model can rank correctly while being unable to generate.

    Context insertion is shared with :func:`build_prompt`, so the free-text RAG
    and non-RAG arms differ in exactly the same single block the MCQ arms do.
    """
    body = f"{question.question.strip()}\n\n{FREETEXT_INSTRUCTION}"
    return BuiltPrompt(
        system=FREETEXT_SYSTEM_PROMPT, user=_with_context(body, passages), context=tuple(passages)
    )


def strip_context(prompt: BuiltPrompt) -> BuiltPrompt:
    """The same prompt with its context block removed.

    Used by the parity test: stripping context from a RAG prompt must yield
    exactly the non-RAG prompt, which proves the arms differ in nothing else.
    """
    if not prompt.has_context:
        return prompt
    _, _, body = prompt.user.partition(CONTEXT_SEPARATOR)
    return BuiltPrompt(system=prompt.system, user=body, context=())
