"""Builds the two retrieval corpora.

The experiment needs both because "is fine-tuning better than retrieval?" is
ambiguous until you say *retrieval over what*:

``parity``
    The ``exp`` explanation field of the MedMCQA **train** rows — precisely the
    text the fine-tune learned from. An arm retrieving this has access to the
    same information the fine-tuned weights absorbed, so the gap between them
    measures format adaptation rather than knowledge.

``external``
    Chunked MIRIAD passages (ODC-By, peer-reviewed literature). The realistic
    "we have a domain corpus" setting.

Two invariants hold for both, and are asserted in ``tests/test_retriever.py``:

1. **Only train-side text is ever indexed.** If a test question's own
   explanation were retrievable the benchmark would be measuring lookup, not
   reasoning, and every RAG number would be worthless.
2. **Chunks are size-comparable.** Raw MIRIAD passages are ~13x longer than
   MedMCQA explanations; indexing them as-is would mean the two RAG arms get
   wildly different amounts of context, confounding quality with context length.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

from fvr.data.schema import Passage, Question

#: Target chunk size in characters. Chosen to sit under the embedding model's
#: 512-token window (~2,000 chars) with headroom, while staying within a small
#: multiple of the MedMCQA explanation length so the two corpora are comparable.
DEFAULT_CHUNK_CHARS = 600
DEFAULT_CHUNK_OVERLAP = 100
#: Chunks shorter than this are dropped.
#:
#: Set from the measured distribution of MedMCQA explanations rather than
#: guessed. Below 40 characters the field is essentially always an answer-key
#: fragment carrying no knowledge — "Ans. C i.e. Mite", "A i.e. Na+", "VP",
#: "Foregut". Between 40 and 80 it is usually a real clinical fact —
#: "Suprarenal glands drain into para-aortic nodes." An 80-char cut discarded
#: 13.8% of explanations, roughly a third of which were substantive, so the
#: threshold sits at 40 and the junk is removed by content instead (see
#: :func:`strip_answer_key`).
MIN_CHUNK_CHARS = 40

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

#: Leading answer-key boilerplate: "Ans. is 'c' i.e.,", "A i.e.", "Answer is B (X)".
#: Stripped rather than kept — the letter refers to the *train* question's option
#: order, so in an index it is noise at best and actively misleading at worst.
_QUOTE = "['\"\u2018\u2019]?"  # straight or curly; MedMCQA uses both
_ANSWER_KEY_PREFIX = re.compile(
    r"^\s*(?:ans(?:wer)?\.?\s*(?:is)?\s*)?[:\-]?\s*"
    + _QUOTE
    + r"[a-eA-E]"
    + _QUOTE
    + r"\s*(?:[.):,\-]|i\.?\s*e\.?)[,.\s]*",
    re.IGNORECASE,
)
#: "Answer is A (Quinine) ..." — an explicit lead-in where the letter is followed
#: by a parenthesised option rather than punctuation, so the pattern above misses it.
_ANSWER_KEY_LEADIN = re.compile(
    r"^\s*ans(?:wer)?\.?\s*(?:is)?\s*" + _QUOTE + r"[a-eA-E]" + _QUOTE + r"\s+(?=\()",
    re.IGNORECASE,
)
#: Bibliography-only explanations: "Ref - Krishan Vij ... pg 202".
_REFERENCE_ONLY = re.compile(r"^\s*(?:ref(?:erence)?s?\b|see\s+ref)", re.IGNORECASE)


def strip_answer_key(text: str) -> str:
    """Remove answer-key boilerplate, returning "" if nothing substantive remains.

    MedMCQA explanations frequently open with the answer letter. That prefix is
    worse than useless in a retrieval index: it matches lexically against option
    text while carrying no clinical content, and the letter it names belongs to
    a different question's option ordering.
    """
    cleaned = _ANSWER_KEY_LEADIN.sub("", text.strip(), count=1)
    cleaned = _ANSWER_KEY_PREFIX.sub("", cleaned, count=1).strip()
    if _REFERENCE_ONLY.match(cleaned):
        return ""
    return cleaned


def chunk_text(
    text: str,
    *,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_chars: int = MIN_CHUNK_CHARS,
) -> list[str]:
    """Split on sentence boundaries into approximately ``chunk_chars`` windows.

    Sentence-aware rather than a fixed stride: cutting mid-sentence produces
    chunks that embed poorly and read badly when injected into a prompt. The
    overlap keeps a fact that straddles a boundary retrievable from both sides.
    """
    text = " ".join(text.split())
    if len(text) <= chunk_chars:
        return [text] if len(text) >= min_chars else []

    sentences = _SENTENCE_END.split(text)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > chunk_chars:
            chunks.append(current.strip())
            tail = current[-overlap:] if overlap else ""
            # Resume from a word boundary so the overlap is not a broken token.
            _, _, tail = tail.partition(" ")
            current = f"{tail} {sentence}".strip()
        else:
            current = f"{current} {sentence}".strip()

        # A single sentence longer than the window still has to be broken up.
        while len(current) > chunk_chars * 2:
            chunks.append(current[:chunk_chars].strip())
            current = current[chunk_chars - overlap :].strip()

    if current:
        chunks.append(current.strip())
    return [c for c in chunks if len(c) >= min_chars]


def _passage_id(prefix: str, text: str, index: int) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{index:08d}-{digest}"


@dataclass(frozen=True)
class CorpusStats:
    """Reported so corpus construction is auditable rather than assumed."""

    name: str
    source_documents: int
    chunks: int
    excluded_documents: int
    total_chars: int

    @property
    def mean_chunk_chars(self) -> float:
        return self.total_chars / self.chunks if self.chunks else 0.0

    def summary(self) -> str:
        return (
            f"{self.name}: {self.chunks:,} chunks from {self.source_documents:,} documents "
            f"(mean {self.mean_chunk_chars:.0f} chars, {self.excluded_documents:,} excluded)"
        )


def build_parity_corpus(
    train_questions: Sequence[Question],
    *,
    forbidden_content_hashes: frozenset[str],
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
) -> tuple[list[Passage], CorpusStats]:
    """Index the explanations attached to train rows.

    ``forbidden_content_hashes`` carries the content hashes of every held-out
    question. Any train row matching one is dropped — MedMCQA repeats items
    across splits under different ids, so filtering on id alone would leave a
    test item's own explanation sitting in the index.
    """
    passages: list[Passage] = []
    excluded = 0
    used_documents = 0
    total_chars = 0

    for question in train_questions:
        if not question.explanation:
            continue
        if question.content_hash() in forbidden_content_hashes:
            excluded += 1
            continue
        body = strip_answer_key(question.explanation)
        if not body:
            continue
        chunks = chunk_text(body, chunk_chars=chunk_chars)
        if not chunks:
            continue
        used_documents += 1
        for chunk in chunks:
            total_chars += len(chunk)
            passages.append(
                Passage(
                    id=_passage_id("parity", chunk, len(passages)),
                    text=chunk,
                    corpus="parity",
                    source_question_id=question.id,
                )
            )

    return passages, CorpusStats(
        name="parity",
        source_documents=used_documents,
        chunks=len(passages),
        excluded_documents=excluded,
        total_chars=total_chars,
    )


def build_external_corpus(
    documents: Iterable[tuple[str, str]],
    *,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    max_chunks: int | None = None,
) -> tuple[list[Passage], CorpusStats]:
    """Chunk ``(title, text)`` documents into the external corpus.

    Callers deduplicate upstream: MIRIAD repeats the same ``passage_text``
    across roughly 2.5 QA pairs, and indexing the duplicates would waste both
    embedding time and index space while skewing retrieval toward whichever
    passages happen to be reused most.
    """
    passages: list[Passage] = []
    used_documents = 0
    total_chars = 0

    for title, text in documents:
        chunks = chunk_text(text, chunk_chars=chunk_chars)
        if not chunks:
            continue
        used_documents += 1
        for chunk in chunks:
            total_chars += len(chunk)
            passages.append(
                Passage(
                    id=_passage_id("ext", chunk, len(passages)),
                    text=chunk,
                    corpus="external",
                    title=title or None,
                )
            )
            if max_chunks is not None and len(passages) >= max_chunks:
                return passages, CorpusStats(
                    name="external",
                    source_documents=used_documents,
                    chunks=len(passages),
                    excluded_documents=0,
                    total_chars=total_chars,
                )

    return passages, CorpusStats(
        name="external",
        source_documents=used_documents,
        chunks=len(passages),
        excluded_documents=0,
        total_chars=total_chars,
    )


def iter_passage_texts(passages: Sequence[Passage]) -> Iterator[str]:
    for passage in passages:
        yield passage.text
