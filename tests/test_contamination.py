"""Contamination and error-analysis tests. CPU only, no model."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from fvr.config import PROJECT_ROOT
from fvr.data.schema import Question
from fvr.eval.contamination import (
    PermutationResult,
    permute_dataset,
    permute_options,
    position_bias,
    split_stem,
    summarise_verbatim,
    token_overlap,
)
from fvr.eval.errors import (
    CONFIDENT_MARGIN,
    ErrorCase,
    categorise,
    category_counts,
    collect_errors,
    sample_for_review,
    write_review_csv,
)


def a_question(qid: str = "q1", answer: int = 0) -> Question:
    return Question(
        id=qid,
        question="Which vessel supplies the myocardium in a healthy adult heart?",
        options=["Coronary artery", "Portal vein", "Aorta", "Vena cava"],
        answer_idx=answer,
        subject="Anatomy",
    )


class TestPermutation:
    def test_the_same_option_text_stays_correct(self) -> None:
        """The permutation must move the label, never the information."""
        question = a_question(answer=0)
        gold_text = question.options[0]
        permuted, _ = permute_options(question, random.Random(1))
        assert permuted.answer_idx is not None
        assert permuted.options[permuted.answer_idx] == gold_text

    def test_option_set_is_preserved(self) -> None:
        permuted, _ = permute_options(a_question(), random.Random(3))
        assert sorted(permuted.options) == sorted(a_question().options)

    def test_is_deterministic_for_a_seed(self) -> None:
        first, _ = permute_dataset([a_question(f"q{i}") for i in range(20)], seed=7)
        second, _ = permute_dataset([a_question(f"q{i}") for i in range(20)], seed=7)
        assert [q.options for q in first] == [q.options for q in second]

    def test_actually_moves_labels(self) -> None:
        questions = [a_question(f"q{i}") for i in range(40)]
        permuted, _ = permute_dataset(questions, seed=5)
        moved = sum(p.answer_idx != q.answer_idx for p, q in zip(permuted, questions, strict=True))
        assert moved > 0, "a permutation that never moves the answer tests nothing"

    def test_unlabelled_question_stays_unlabelled(self) -> None:
        question = a_question().model_copy(update={"answer_idx": None})
        permuted, _ = permute_options(question, random.Random(1))
        assert permuted.answer_idx is None


class TestPermutationResult:
    def test_no_drop_reads_as_clean(self) -> None:
        result = PermutationResult(original_accuracy=0.60, permuted_accuracy=0.59, n=1000)
        assert "no evidence" in result.verdict()

    def test_large_drop_is_called_out(self) -> None:
        result = PermutationResult(original_accuracy=0.60, permuted_accuracy=0.40, n=1000)
        assert "substantial" in result.verdict()
        assert result.drop == pytest.approx(0.20)

    def test_relative_drop_handles_zero(self) -> None:
        assert PermutationResult(0.0, 0.0, 10).relative_drop == 0.0


class TestPositionBias:
    def test_uniform_predictions_have_low_excess(self) -> None:
        bias = position_bias([0, 1, 2, 3] * 25, [0, 1, 2, 3] * 25)
        assert bias.max_excess == pytest.approx(0.0, abs=1e-9)

    def test_detects_a_letter_the_model_over_picks(self) -> None:
        # Always answers C, while gold is uniform.
        bias = position_bias([2] * 100, [0, 1, 2, 3] * 25)
        assert bias.predicted["C"] == pytest.approx(1.0)
        assert bias.max_excess > 0.7


class TestVerbatim:
    def test_split_stem_returns_both_halves(self) -> None:
        prefix, rest = split_stem("one two three four five six")
        assert prefix and rest
        assert f"{prefix} {rest}".split() == ["one", "two", "three", "four", "five", "six"]

    def test_identical_text_scores_one(self) -> None:
        assert token_overlap("the heart pumps blood", "the heart pumps blood") == 1.0

    def test_unrelated_text_scores_zero(self) -> None:
        assert token_overlap("bone remodelling", "cardiac output") == 0.0

    def test_summary_flags_frequent_reproduction(self) -> None:
        result = summarise_verbatim(["a b c"] * 10, ["a b c"] * 10)
        assert result.high_overlap_rate == 1.0
        assert "frequent" in result.verdict()

    def test_summary_reports_clean_when_no_overlap(self) -> None:
        result = summarise_verbatim(["x y z"] * 10, ["a b c"] * 10)
        assert "no verbatim" in result.verdict()

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="no continuations"):
            summarise_verbatim([], [])


class TestErrorCategories:
    def test_correct_answers_are_not_errors(self) -> None:
        assert categorise(a_question(answer=0), 0, [2.0, 1.0, 0.5, 0.1]) == "correct"

    def test_confident_wrong_needs_a_wide_margin(self) -> None:
        # Chosen B by a mile while A was right — the dangerous clinical failure.
        assert categorise(a_question(answer=0), 1, [0.0, 5.0, 0.1, 0.1]) == "confident_wrong"

    def test_near_miss_when_the_runner_up_was_right(self) -> None:
        assert categorise(a_question(answer=0), 1, [1.99, 2.0, 0.1, 0.1]) == "near_miss"

    def test_margin_threshold_is_the_documented_one(self) -> None:
        assert CONFIDENT_MARGIN > 0

    def test_fixed_by_retrieval(self) -> None:
        assert (
            categorise(a_question(answer=0), 0, [2.0, 1.0, 0, 0], comparison_correct=False)
            == "fixed_by_retrieval"
        )

    def test_broken_by_retrieval(self) -> None:
        assert (
            categorise(a_question(answer=0), 1, [1.0, 2.0, 0, 0], comparison_correct=True)
            == "broken_by_retrieval"
        )

    def test_abstention_is_unscorable(self) -> None:
        assert categorise(a_question(), None, None) == "unscorable"


class TestErrorCollection:
    def _payload(self, arm: str, predicted: list[int]) -> dict[str, Any]:
        return {
            "arm": arm,
            "predictions": [
                {
                    "question_id": f"q{i}",
                    "predicted_idx": p,
                    "option_logprobs": [3.0 if j == p else 0.0 for j in range(4)],
                }
                for i, p in enumerate(predicted)
            ],
        }

    def test_only_failures_are_collected(self) -> None:
        questions = {f"q{i}": a_question(f"q{i}", answer=0) for i in range(4)}
        cases = collect_errors(self._payload("base", [0, 1, 0, 2]), questions)
        assert {c.question_id for c in cases} == {"q1", "q3"}

    def test_counts_are_sorted_by_frequency(self) -> None:
        counts = category_counts(
            [
                ErrorCase("q", "s", "t", ["a", "b", "c", "d"], "A", "B", 1.0, "wrong", "x", []),
                ErrorCase("r", "s", "t", ["a", "b", "c", "d"], "A", "B", 1.0, "wrong", "x", []),
                ErrorCase("s", "s", "t", ["a", "b", "c", "d"], "A", "B", 1.0, "near_miss", "x", []),
            ]
        )
        assert list(counts) == ["wrong", "near_miss"]

    def test_sampling_is_stratified_and_deterministic(self) -> None:
        cases = [
            ErrorCase(
                f"q{i}",
                "s",
                "t",
                ["a", "b", "c", "d"],
                "A",
                "B",
                1.0,
                "wrong" if i % 2 else "near_miss",
                "x",
                [],
            )
            for i in range(60)
        ]
        first = sample_for_review(cases, per_category=5, seed=1)
        second = sample_for_review(cases, per_category=5, seed=1)
        assert [c.question_id for c in first] == [c.question_id for c in second]
        # Both categories represented, not just the larger one.
        assert len({c.category for c in first}) == 2

    def test_review_csv_has_blank_human_columns(self, tmp_path: Path) -> None:
        cases = [
            ErrorCase(
                "q1", "Anatomy", "stem", ["a", "b", "c", "d"], "A", "B", 1.0, "wrong", "x", []
            )
        ]
        path = tmp_path / "review.csv"
        write_review_csv(cases, path)
        text = path.read_text(encoding="utf-8")
        assert "human_label" in text and "notes" in text


class TestCommittedErrorAnalysis:
    def test_summary_is_committed(self) -> None:
        path = PROJECT_ROOT / "results" / "error_analysis" / "summary.json"
        assert path.is_file(), "run scripts/06_error_analysis.py"

    def test_every_arm_has_a_review_csv(self) -> None:
        summary = json.loads(
            (PROJECT_ROOT / "results" / "error_analysis" / "summary.json").read_text("utf-8")
        )
        for arm in summary:
            assert (PROJECT_ROOT / "results" / "error_analysis" / f"{arm}_review.csv").is_file()
