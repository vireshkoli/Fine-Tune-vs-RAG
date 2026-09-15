"""The benchmark's prompt, retrieval and scoring rules, vendored for the Space.

This file is a copy of the relevant pieces of ``fvr`` (``prompts/templates.py``,
``retrieval/retriever.py``, ``inference/mcq.py``) with no dependency on the
package, so the Space installs nothing but its runtime. It is kept honest by
``tests/test_live_space.py`` in the repository, which imports this file and
asserts every constant and every built prompt matches the package byte for
byte. If the two ever diverge, the demo is no longer showing the benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field

OPTION_LABELS: tuple[str, ...] = ("A", "B", "C", "D")

SYSTEM_PROMPT = (
    "You are a medical exam assistant. Answer the multiple-choice question by "
    "selecting the single best option. Respond with only the letter of your "
    "chosen option."
)
ANSWER_INSTRUCTION = "Answer with a single letter (A, B, C, or D)."
FREETEXT_SYSTEM_PROMPT = (
    "You are a medical exam assistant. Answer the question directly and "
    "concisely. State your answer; do not explain your reasoning at length."
)
FREETEXT_INSTRUCTION = "Answer in one or two sentences."
CONTEXT_HEADER = (
    "Use the following reference passages if they are relevant. "
    "If they are not relevant, rely on your own knowledge."
)
CONTEXT_SEPARATOR = "\n\n---\n\n"

#: Retrieval settings of the benchmark's ``configs/retrieval/bge_large.yaml``.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
TOP_K = 5
MAX_CONTEXT_CHARS = 3000

#: Generation bound used for the MedMCQA free-text arms.
MAX_NEW_TOKENS = 96


@dataclass(frozen=True)
class Passage:
    id: str
    text: str
    score: float = 0.0


@dataclass(frozen=True)
class BuiltPrompt:
    system: str
    user: str
    context: tuple[Passage, ...] = field(default_factory=tuple)

    def as_messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]


def format_options(options: list[str]) -> str:
    return "\n".join(
        f"{label}. {text}" for label, text in zip(OPTION_LABELS, options, strict=False)
    )


def format_context(passages: list[Passage] | tuple[Passage, ...]) -> str:
    blocks = [f"[{i}] {p.text.strip()}" for i, p in enumerate(passages, start=1)]
    return f"{CONTEXT_HEADER}\n\n" + "\n\n".join(blocks)


def _with_context(body: str, passages: list[Passage] | tuple[Passage, ...]) -> str:
    if not passages:
        return body
    return f"{format_context(passages)}{CONTEXT_SEPARATOR}{body}"


def build_prompt(
    question: str, options: list[str], passages: list[Passage] | tuple[Passage, ...] = ()
) -> BuiltPrompt:
    """The MCQ prompt: stem, options, instruction — context prepended for RAG arms."""
    body = f"{question.strip()}\n\n{format_options(options)}\n\n{ANSWER_INSTRUCTION}"
    return BuiltPrompt(
        system=SYSTEM_PROMPT, user=_with_context(body, passages), context=tuple(passages)
    )


def build_freetext_prompt(
    question: str, passages: list[Passage] | tuple[Passage, ...] = ()
) -> BuiltPrompt:
    """The open-ended prompt: options never shown."""
    body = f"{question.strip()}\n\n{FREETEXT_INSTRUCTION}"
    return BuiltPrompt(
        system=FREETEXT_SYSTEM_PROMPT, user=_with_context(body, passages), context=tuple(passages)
    )


def retrieval_query(question: str, options: list[str], *, with_options: bool) -> str:
    """Stem plus options for the MCQ arms; stem alone for free-text arms.

    The options carry most of the retrievable signal, which is exactly why they
    must not be used when the arm is not allowed to see them.
    """
    if not with_options:
        return question
    return question + " " + " ".join(options)


def apply_context_budget(
    passages: list[Passage], max_chars: int = MAX_CONTEXT_CHARS
) -> list[Passage]:
    """Keep whole passages in rank order until the budget is spent."""
    kept: list[Passage] = []
    used = 0
    for passage in passages:
        length = len(passage.text)
        if kept and used + length > max_chars:
            break
        kept.append(passage)
        used += length
        if used >= max_chars:
            break
    return kept


def option_token_ids(tokenizer, n_options: int = 4) -> list[list[int]]:  # type: ignore[no-untyped-def]
    """Every token that could begin each answer letter; scored as the max."""
    variants: list[list[int]] = []
    for label in OPTION_LABELS[:n_options]:
        ids: set[int] = set()
        for surface in (label, f" {label}", label.lower(), f" {label.lower()}"):
            encoded = tokenizer.encode(surface, add_special_tokens=False)
            if encoded:
                ids.add(encoded[0])
        if not ids:
            raise ValueError(f"tokenizer produced no ids for option {label!r}")
        variants.append(sorted(ids))
    return variants


def option_logprobs(logits, option_ids: list[list[int]]) -> tuple[float, ...]:  # type: ignore[no-untyped-def]
    """Log-probabilities over the answer letters, normalised over the whole vocabulary."""
    import torch

    logprobs = torch.log_softmax(logits.float(), dim=-1)
    return tuple(float(torch.max(logprobs[ids])) for ids in option_ids)
