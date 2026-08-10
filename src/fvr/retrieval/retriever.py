"""Query-time retrieval, with a context budget shared by every RAG arm.

The budget is the fairness control for Phase 4. The two corpora have different
natural granularity — MedMCQA explanations run ~340 characters, chunked MIRIAD
passages ~600 — so a fixed top-k would hand the external arm roughly twice the
context of the parity arm. That extra context costs prefill tokens, which shows
up as both higher latency and higher cost per query, and more text is usually
worth some accuracy too. The comparison would then be measuring context length
rather than corpus quality.

Retrieving top-k and then truncating to a fixed **character budget** makes the
two arms cost the same and differ only in what the text says.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from fvr.data.schema import Passage, Question
from fvr.retrieval.embed import EmbedderConfig, embed_queries
from fvr.retrieval.index import VectorIndex


class RetrievalConfig(BaseModel):
    """From ``configs/retrieval/*.yaml``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    top_k: int = Field(default=5, ge=1)
    #: Shared across RAG arms. See the module docstring — this is what stops the
    #: comparison from becoming "which arm got more text".
    max_context_chars: int = Field(default=3000, ge=100)
    #: Drop hits below this cosine similarity rather than padding with noise.
    min_score: float = 0.0
    rerank: bool = False


def load_retrieval_config(path: Path | str) -> RetrievalConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return RetrievalConfig(**raw.get("retrieval", raw))


def apply_context_budget(passages: list[Passage], max_chars: int) -> list[Passage]:
    """Keep whole passages in rank order until the budget is spent.

    Whole passages only — a passage truncated mid-sentence embeds a claim the
    model cannot verify and makes the groundedness check meaningless. Better to
    inject three complete passages than three and a half.
    """
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


@dataclass
class Retriever:
    """Turns a question into the passages that will be injected into its prompt."""

    index: VectorIndex
    embedder_config: EmbedderConfig
    config: RetrievalConfig
    model: Any = None

    def embedder(self) -> Any:
        """The query encoder, loaded once and reused.

        Load-bearing. ``embed_queries`` falls back to constructing an encoder
        when handed ``None``, so leaving ``model`` unset meant a fresh
        SentenceTransformer per batch — 125 of them in a 1,000-item run. That
        exhausted 44GB of VRAM and killed the rag-external arm outright.
        """
        if self.model is None:
            from fvr.retrieval.embed import load_embedder

            self.model = load_embedder(self.embedder_config)
        return self.model

    def retrieve(self, question: Question) -> list[Passage]:
        return self.retrieve_many([question])[0]

    def retrieve_many(self, questions: list[Question]) -> list[list[Passage]]:
        """Batched retrieval — embedding one query at a time wastes the GPU."""
        if not questions:
            return []
        queries = [self._query_text(q) for q in questions]
        vectors = embed_queries(queries, self.embedder_config, model=self.embedder())
        hits = self.index.search(vectors, self.config.top_k)
        return [
            apply_context_budget(
                [p for p in row if (p.score or 0.0) >= self.config.min_score],
                self.config.max_context_chars,
            )
            for row in hits
        ]

    def _query_text(self, question: Question) -> str:
        """Question stem plus options.

        The options carry most of the retrievable signal in an exam item — the
        stem alone is often a generic vignette, while the option list names the
        specific entities worth looking up.
        """
        return question.question + " " + " ".join(question.options)


def load_retriever(
    index_dir: Path,
    embedder_config: EmbedderConfig,
    retrieval_config: RetrievalConfig,
    *,
    model: Any = None,
) -> Retriever:
    from fvr.retrieval.index import load_index

    return Retriever(
        index=load_index(index_dir),
        embedder_config=embedder_config,
        config=retrieval_config,
        model=model,
    )
