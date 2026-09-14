"""MIRIAD as a training source — the airtight information-parity experiment.

The MedMCQA parity arms are close to airtight but not quite: the fine-tune saw
explanations wrapped in an exam-answer target, and the index holds the same
explanations chunked. MIRIAD lets the comparison be made exactly, in two forms:

``qa``
    Fine-tune on ``question -> answer`` pairs derived from a set of passages,
    and index those same passages. The weights saw the *answers to other
    questions about* the passage; the index holds the passage itself.

``doc``
    Fine-tune on the passage text itself, causal-LM style, and index the same
    text. The weights and the index were shown byte-identical content. This is
    the variant where "information parity" is literal rather than argued.

Held-out test questions are chosen so that their *passage is in the training
set* — otherwise neither arm could have the information and the test measures
nothing. Two leakage guards close the gaps MIRIAD's structure opens:

1. The same question text recurs across different passages (144 of 90k rows
   in shard 0). A test question whose text appears anywhere in training is
   dropped, whatever passage it came from.
2. Passages are keyed by a hash of their text, not by ``(paper_id, position)``.
   The text is what the index holds and what the fine-tune sees, so it is what
   the split must be defined on.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fvr.data.schema import Question

MIRIAD_REPO = "miriad/miriad-5.8M"
MIRIAD_SHARDS = 64
#: Shards used. Four is what the external index was built from, so the parity
#: experiment draws on the same documents the external arm already retrieves.
DEFAULT_SHARDS = 4

Variant = Literal["qa", "doc"]


def passage_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _norm(text: str) -> str:
    return " ".join(text.split()).lower()


@dataclass(frozen=True)
class MiriadQA:
    """One question-answer pair and the passage it was generated from."""

    qa_id: str
    question: str
    answer: str
    passage_key: str
    passage_text: str
    title: str
    specialty: str

    def as_question(self) -> Question:
        """Adapt to the project's ``Question`` for the free-text pipeline.

        The free-text arms never show options and grade against
        ``reference_answer`` — ``options[answer_idx]`` — so the answer goes in
        slot 0 and a placeholder fills the schema's two-option minimum. The
        placeholder is never rendered anywhere.
        """
        return Question(
            id=self.qa_id,
            question=self.question.strip(),
            options=[self.answer.strip(), "(free-text item: no options)"],
            answer_idx=0,
            subject=self.specialty or "Unknown",
            explanation=self.passage_text,
            source="miriad",
        )


@dataclass
class ParitySplit:
    """Train QA pairs, the passages they came from, and held-out test QA."""

    train: list[MiriadQA] = field(default_factory=list)
    test: list[MiriadQA] = field(default_factory=list)
    #: passage_key -> (title, text) for every passage with a train QA pair.
    #: This — and only this — is what the parity index holds.
    passages: dict[str, tuple[str, str]] = field(default_factory=dict)
    stats: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        s = self.stats
        return (
            f"{s.get('rows', 0):,} rows -> {len(self.train):,} train QA over "
            f"{len(self.passages):,} passages, {len(self.test):,} test QA | dropped: "
            f"{s.get('dropped_duplicate_question', 0):,} duplicate-question, "
            f"{s.get('dropped_duplicate_pair', 0):,} duplicate-pair, "
            f"{s.get('dropped_singleton_passage', 0):,} singleton-passage"
        )


def load_miriad_rows(n_shards: int = DEFAULT_SHARDS) -> list[dict[str, Any]]:
    """Raw rows from the first ``n_shards`` shards, via the pinned artifact cache."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    rows: list[dict[str, Any]] = []
    columns = ["qa_id", "question", "answer", "passage_text", "paper_title", "specialty"]
    for shard in range(n_shards):
        path = hf_hub_download(
            MIRIAD_REPO,
            f"data/train-{shard:05d}-of-{MIRIAD_SHARDS:05d}.parquet",
            repo_type="dataset",
        )
        frame = pd.read_parquet(path, columns=columns)
        rows.extend({str(k): v for k, v in record.items()} for record in frame.to_dict("records"))
    return rows


def build_parity_split(
    rows: Iterable[dict[str, Any]],
    *,
    n_train_qa: int,
    n_test: int,
    seed: int = 42,
) -> ParitySplit:
    """Split so every test question's passage has training QA pairs.

    Procedure, in order:

    1. Group QA pairs by passage text. Drop passages with only one pair — they
       cannot supply both a train pair and a held-out sibling.
    2. Sample passages until ``n_train_qa`` pairs are covered. From each chosen
       passage, hold out exactly one pair as a test candidate and train on the
       rest, so the test question is never among the training targets.
    3. Drop any test candidate whose question text appears anywhere in training
       (MIRIAD repeats questions across passages), or whose exact pair does.
    4. Sample ``n_test`` from what survives.
    """
    rng = random.Random(seed)
    stats: dict[str, int] = {"rows": 0}
    by_passage: dict[str, list[MiriadQA]] = {}
    seen_pairs: set[tuple[str, str]] = set()

    for row in rows:
        stats["rows"] += 1
        text = str(row["passage_text"])
        qa = MiriadQA(
            qa_id=str(row["qa_id"]),
            question=str(row["question"]),
            answer=str(row["answer"]),
            passage_key=passage_key(text),
            passage_text=text,
            title=str(row.get("paper_title") or ""),
            specialty=str(row.get("specialty") or ""),
        )
        pair = (_norm(qa.question), _norm(qa.answer))
        if pair in seen_pairs:
            stats["dropped_duplicate_pair"] = stats.get("dropped_duplicate_pair", 0) + 1
            continue
        seen_pairs.add(pair)
        by_passage.setdefault(qa.passage_key, []).append(qa)

    eligible = [k for k, qas in by_passage.items() if len(qas) >= 2]
    stats["dropped_singleton_passage"] = len(by_passage) - len(eligible)
    rng.shuffle(eligible)

    split = ParitySplit(stats=stats)
    candidates: list[MiriadQA] = []
    for key in eligible:
        if len(split.train) >= n_train_qa:
            break
        qas = list(by_passage[key])
        rng.shuffle(qas)
        held, rest = qas[0], qas[1:]
        split.train.extend(rest)
        split.passages[key] = (rest[0].title, rest[0].passage_text)
        candidates.append(held)

    train_questions = {_norm(qa.question) for qa in split.train}
    survivors: list[MiriadQA] = []
    for qa in candidates:
        if _norm(qa.question) in train_questions:
            stats["dropped_duplicate_question"] = stats.get("dropped_duplicate_question", 0) + 1
            continue
        survivors.append(qa)
    if len(survivors) < n_test:
        raise ValueError(f"only {len(survivors)} leak-free test candidates; asked for {n_test}")
    rng.shuffle(survivors)
    split.test = sorted(survivors[:n_test], key=lambda qa: qa.qa_id)
    return split


def assert_no_leakage(split: ParitySplit) -> None:
    """The invariants a test must never violate. Raises on the first breach."""
    train_q = {_norm(qa.question) for qa in split.train}
    train_pairs = {(_norm(qa.question), _norm(qa.answer)) for qa in split.train}
    train_ids = {qa.qa_id for qa in split.train}
    for qa in split.test:
        if qa.qa_id in train_ids:
            raise AssertionError(f"test item {qa.qa_id} is also a training item")
        if _norm(qa.question) in train_q:
            raise AssertionError(f"test question text of {qa.qa_id} appears in training")
        if (_norm(qa.question), _norm(qa.answer)) in train_pairs:
            raise AssertionError(f"test pair {qa.qa_id} appears in training")
        if qa.passage_key not in split.passages:
            raise AssertionError(
                f"test item {qa.qa_id}'s passage is not in the training set — "
                "parity requires it to be"
            )


def write_split(split: ParitySplit, path: Path) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stats": split.stats,
        "n_train": len(split.train),
        "n_test": len(split.test),
        "n_passages": len(split.passages),
        "train_qa_ids": [qa.qa_id for qa in split.train],
        "test_qa_ids": [qa.qa_id for qa in split.test],
        "passage_keys": sorted(split.passages),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sft_messages(qa: MiriadQA) -> list[dict[str, str]]:
    """The ``qa`` variant's training record, built by the free-text prompt.

    Same builder the free-text arms evaluate with, so the fine-tune is trained
    on the byte-identical scaffolding it is later asked to answer under.
    """
    from fvr.prompts.templates import build_freetext_prompt

    prompt = build_freetext_prompt(qa.as_question())
    return [*prompt.as_messages(), {"role": "assistant", "content": qa.answer.strip()}]


def doc_texts(split: ParitySplit, *, chunk_chars: int = 600) -> list[str]:
    """The ``doc`` variant's training texts: the training passages, chunked.

    Chunked with the *same* chunker the index uses, so the units the weights
    were trained on are the units the retriever serves — parity down to the
    chunk boundary.
    """
    from fvr.retrieval.corpus import chunk_text

    texts: list[str] = []
    for _, text in split.passages.values():
        texts.extend(chunk_text(text, chunk_chars=chunk_chars))
    return texts


def corpus_documents(split: ParitySplit) -> Sequence[tuple[str, str]]:
    """``(title, text)`` for the index builder — exactly the training passages."""
    return list(split.passages.values())
