"""The MIRIAD parity split.

MIRIAD's structure opens two leakage routes the MedMCQA split never had: the
same question text recurs across different passages, and the passage — the
thing both arms are given — is the unit the split has to be defined on. These
tests are about those routes; a split that leaks makes both parity arms look
better for a reason that has nothing to do with weights or indices.
"""

from __future__ import annotations

from typing import Any

import pytest

from fvr.data.miriad import (
    ParitySplit,
    assert_no_leakage,
    build_parity_split,
    corpus_documents,
    doc_texts,
    passage_key,
    sft_messages,
)


def rows(n_passages: int = 40, per_passage: int = 3) -> list[dict[str, Any]]:
    out = []
    for p in range(n_passages):
        text = f"Passage {p}. " + " ".join(f"Sentence {p}-{s} about topic {p}." for s in range(30))
        for q in range(per_passage):
            out.append(
                {
                    "qa_id": f"{p}_{q}",
                    "question": f"What does passage {p} say in part {q}?",
                    "answer": f"It says sentence {p}-{q}.",
                    "passage_text": text,
                    "paper_title": f"Paper {p}",
                    "specialty": "Cardiology",
                }
            )
    return out


class TestSplit:
    def test_every_test_passage_is_in_training(self) -> None:
        split = build_parity_split(rows(), n_train_qa=60, n_test=10)
        assert_no_leakage(split)
        for qa in split.test:
            assert qa.passage_key in split.passages

    def test_test_question_never_among_training_targets(self) -> None:
        split = build_parity_split(rows(), n_train_qa=60, n_test=10)
        train_ids = {qa.qa_id for qa in split.train}
        assert not any(qa.qa_id in train_ids for qa in split.test)

    def test_is_deterministic(self) -> None:
        a = build_parity_split(rows(), n_train_qa=60, n_test=10, seed=3)
        b = build_parity_split(rows(), n_train_qa=60, n_test=10, seed=3)
        assert [q.qa_id for q in a.test] == [q.qa_id for q in b.test]
        assert [q.qa_id for q in a.train] == [q.qa_id for q in b.train]

    def test_drops_singleton_passages(self) -> None:
        """One QA pair cannot supply both a training target and a held-out sibling."""
        solo = rows(n_passages=5, per_passage=1)
        for r in solo:  # distinct text, or they merge with the multi-QA passages by key
            r["passage_text"] = "Solo " + r["passage_text"]
            r["qa_id"] = "solo_" + r["qa_id"]
        data = solo + rows(n_passages=20, per_passage=3)
        split = build_parity_split(data, n_train_qa=30, n_test=5)
        assert split.stats["dropped_singleton_passage"] == 5

    def test_duplicate_question_text_across_passages_is_dropped_from_test(self) -> None:
        """MIRIAD repeats questions across passages; that is a leak, not a coincidence."""
        data = rows(n_passages=30)
        # Make passage 0's held-out candidate share its text with a training row elsewhere.
        for r in data:
            if r["qa_id"].startswith("0_"):
                r["question"] = "What is the shared question?"
            if r["qa_id"] == "1_1":
                r["question"] = "What is the shared question?"
        split = build_parity_split(data, n_train_qa=60, n_test=5, seed=1)
        assert_no_leakage(split)
        assert all(q.question != "What is the shared question?" for q in split.test)

    def test_exact_duplicate_pairs_are_dropped(self) -> None:
        data = rows(n_passages=20)
        data.append(dict(data[0], qa_id="dup"))
        split = build_parity_split(data, n_train_qa=30, n_test=5)
        assert split.stats["dropped_duplicate_pair"] == 1

    def test_refuses_when_too_few_leak_free_candidates(self) -> None:
        with pytest.raises(ValueError, match="leak-free"):
            build_parity_split(rows(n_passages=4), n_train_qa=8, n_test=10)


class TestLeakageAssertion:
    def test_catches_a_test_item_in_training(self) -> None:
        split = build_parity_split(rows(), n_train_qa=60, n_test=10)
        split.train.append(split.test[0])
        with pytest.raises(AssertionError, match="also a training item"):
            assert_no_leakage(split)

    def test_catches_a_test_passage_missing_from_the_index(self) -> None:
        split = build_parity_split(rows(), n_train_qa=60, n_test=10)
        del split.passages[split.test[0].passage_key]
        with pytest.raises(AssertionError, match="parity requires"):
            assert_no_leakage(split)


class TestTrainingViews:
    def test_qa_record_uses_the_freetext_prompt(self) -> None:
        split = build_parity_split(rows(), n_train_qa=60, n_test=10)
        messages = sft_messages(split.train[0])
        assert messages[0]["role"] == "system"
        assert "Answer in one or two sentences." in messages[1]["content"]
        assert messages[-1] == {"role": "assistant", "content": split.train[0].answer}
        assert "(free-text item" not in messages[1]["content"], "placeholder option leaked"

    def test_doc_texts_are_the_indexed_passages_chunked_identically(self) -> None:
        from fvr.retrieval.corpus import chunk_text

        split = build_parity_split(rows(), n_train_qa=60, n_test=10)
        expected = [c for _, t in split.passages.values() for c in chunk_text(t, chunk_chars=600)]
        assert doc_texts(split) == expected

    def test_corpus_documents_are_exactly_the_training_passages(self) -> None:
        split = build_parity_split(rows(), n_train_qa=60, n_test=10)
        docs = corpus_documents(split)
        assert {passage_key(t) for _, t in docs} == set(split.passages)

    def test_as_question_keeps_the_answer_as_reference(self) -> None:
        from fvr.eval.freetext import option_dependent_reason, reference_answer

        split = build_parity_split(rows(), n_train_qa=60, n_test=10)
        q = split.test[0].as_question()
        assert reference_answer(q) == split.test[0].answer
        assert option_dependent_reason(q) is None
        assert q.source == "miriad"

    def test_summary_mentions_the_counts(self) -> None:
        split = build_parity_split(rows(), n_train_qa=60, n_test=10)
        assert "train QA" in split.summary() and "test QA" in split.summary()
        assert isinstance(split, ParitySplit)
