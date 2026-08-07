"""Split tests, including the regression that pins the committed manifest.

``test_committed_manifest_is_intact`` is the one that matters most: it turns
"the test set is frozen" into something CI enforces. If a future change to
cleaning, ordering or the seed would move the split, that test fails before any
result can be quietly rebased onto different data.
"""

from __future__ import annotations

import json

import pytest

from fvr.config import PROJECT_ROOT
from fvr.data.schema import Question, Split
from fvr.data.splits import (
    SplitAssignment,
    assert_disjoint,
    build_manifest,
    find_train_leakage,
    stratified_split,
    verify_manifest,
)

MANIFEST_PATH = PROJECT_ROOT / "results" / "split_manifest.json"

# Pinned prefixes of the frozen split digests. Hoisted to constants so the
# allowlist pragma stays on the same line as the literal after formatting —
# these are SHA-256 prefixes, not credentials.
EXPECTED_TEST_SHA = "9aac1bc01a70dcb6"  # pragma: allowlist secret
EXPECTED_VAL_SHA = "e10015cda15a68ef"  # pragma: allowlist secret


def make_questions(n: int, *, subjects: int = 4, prefix: str = "q") -> list[Question]:
    return [
        Question(
            id=f"{prefix}{i:05d}",
            # The prefix is part of the text too: leakage is detected on content,
            # so two batches must differ in content, not just in id.
            question=f"{prefix} question number {i}?",
            options=["a", "b", "c", "d"],
            answer_idx=i % 4,
            subject=f"subject-{i % subjects}",
        )
        for i in range(n)
    ]


class TestStratifiedSplit:
    def test_sizes_are_exact(self) -> None:
        a = stratified_split(make_questions(600), seed=1, test_size=100, val_size=50)
        assert len(a.test_ids) == 100
        assert len(a.val_ids) == 50
        assert len(a.reserve_ids) == 450

    def test_is_deterministic_for_a_seed(self) -> None:
        qs = make_questions(600)
        first = stratified_split(qs, seed=7, test_size=100, val_size=50)
        second = stratified_split(qs, seed=7, test_size=100, val_size=50)
        assert first.test_ids == second.test_ids

    def test_input_order_does_not_matter(self) -> None:
        """A reshuffled input must not silently produce a different test set."""
        qs = make_questions(600)
        forward = stratified_split(qs, seed=7, test_size=100, val_size=50)
        backward = stratified_split(list(reversed(qs)), seed=7, test_size=100, val_size=50)
        assert forward.test_ids == backward.test_ids

    def test_different_seeds_differ(self) -> None:
        qs = make_questions(600)
        a = stratified_split(qs, seed=1, test_size=100, val_size=50)
        b = stratified_split(qs, seed=2, test_size=100, val_size=50)
        assert a.test_ids != b.test_ids

    def test_splits_are_disjoint(self) -> None:
        a = stratified_split(make_questions(600), seed=3, test_size=100, val_size=50)
        assert_disjoint(a)

    def test_every_subject_is_represented(self) -> None:
        qs = make_questions(600, subjects=10)
        a = stratified_split(qs, seed=5, test_size=100, val_size=50)
        by_id = {q.id: q for q in qs}
        assert len({by_id[i].subject for i in a.test_ids}) == 10

    def test_stratification_is_proportional(self) -> None:
        # 900 rows: subject-0 gets 3x the share of the other three.
        qs = [
            Question(
                id=f"q{i:05d}",
                question=f"q{i}?",
                options=["a", "b", "c", "d"],
                answer_idx=0,
                subject="subject-0" if i < 450 else f"subject-{i % 3 + 1}",
            )
            for i in range(900)
        ]
        a = stratified_split(qs, seed=9, test_size=200, val_size=100)
        by_id = {q.id: q for q in qs}
        n_big = sum(by_id[i].subject == "subject-0" for i in a.test_ids)
        assert 90 <= n_big <= 110, f"expected ~100 of 200 from the half-sized subject, got {n_big}"

    def test_reserve_holds_everything_unselected(self) -> None:
        a = stratified_split(make_questions(600), seed=3, test_size=100, val_size=50)
        assert len(a.test_ids) + len(a.val_ids) + len(a.reserve_ids) == 600

    def test_rejects_an_oversized_request(self) -> None:
        with pytest.raises(ValueError, match="labelled pool"):
            stratified_split(make_questions(100), seed=1, test_size=100, val_size=50)


class TestManifest:
    def _assignment(self) -> SplitAssignment:
        return stratified_split(make_questions(600), seed=1, test_size=100, val_size=50)

    def test_verify_accepts_a_matching_split(self) -> None:
        a = self._assignment()
        manifest = build_manifest(a, dataset_revision=None, pool_size=600, lexicon_size=222)
        assert verify_manifest(a, manifest) == []

    def test_verify_rejects_a_changed_split(self) -> None:
        manifest = build_manifest(
            self._assignment(), dataset_revision=None, pool_size=600, lexicon_size=222
        )
        moved = stratified_split(make_questions(600), seed=99, test_size=100, val_size=50)
        problems = verify_manifest(moved, manifest)
        assert problems and any("sha256" in p for p in problems)

    def test_verify_reports_a_missing_section(self) -> None:
        assert verify_manifest(self._assignment(), {}) == ["manifest has no `splits` section"]

    def test_digest_ignores_ordering(self) -> None:
        a = self._assignment()
        shuffled = SplitAssignment(
            test_ids=tuple(reversed(a.test_ids)),
            val_ids=a.val_ids,
            reserve_ids=a.reserve_ids,
            seed=a.seed,
            source_split=a.source_split,
        )
        assert shuffled.digest(Split.TEST) == a.digest(Split.TEST)


class TestCommittedManifest:
    """Regression guard on the real frozen split."""

    def test_committed_manifest_exists(self) -> None:
        assert MANIFEST_PATH.is_file(), "run scripts/01_build_splits.py and commit the manifest"

    def test_committed_manifest_is_intact(self) -> None:
        m = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        assert m["splits"]["test"]["n"] == 1000
        assert m["splits"]["val"]["n"] == 500
        # If cleaning, ordering or the seed ever changes, these move — and the
        # whole benchmark would silently rebase onto different data.
        assert m["splits"]["test"]["sha256"].startswith(EXPECTED_TEST_SHA)
        assert m["splits"]["val"]["sha256"].startswith(EXPECTED_VAL_SHA)

    def test_records_the_withheld_label_caveat(self) -> None:
        m = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        assert m["source_split"] == "validation"
        assert "cop == -1" in m["note"]

    def test_test_and_val_ids_are_disjoint_on_disk(self) -> None:
        ids = json.loads((PROJECT_ROOT / "results" / "split_ids.json").read_text(encoding="utf-8"))
        assert not set(ids["test"]) & set(ids["val"])
        assert len(ids["test"]) == 1000


class TestLeakage:
    def test_detects_content_duplicates_across_different_ids(self) -> None:
        held_out = [
            Question(
                id="eval-1", question="Which vessel?", options=["a", "b", "c", "d"], answer_idx=0
            )
        ]
        train = [
            Question(
                id="train-9",  # different id, identical content
                question="which vessel?",
                options=["A", "B", "C", "D"],
                answer_idx=0,
            )
        ]
        assert find_train_leakage(train, held_out) == [("train-9", "eval-1")]

    def test_reports_nothing_when_clean(self) -> None:
        assert (
            find_train_leakage(make_questions(10, prefix="t"), make_questions(10, prefix="e")) == []
        )
