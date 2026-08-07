"""Canonical schema tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fvr.data.schema import OPTION_LABELS, Passage, Prediction, Question


def a_question(**overrides: object) -> Question:
    base: dict[str, object] = {
        "id": "q1",
        "question": "Which vessel supplies the myocardium?",
        "options": ["Coronary artery", "Portal vein", "Aorta", "Vena cava"],
        "answer_idx": 0,
    }
    return Question(**{**base, **overrides})


class TestQuestion:
    def test_answer_label_is_derived(self) -> None:
        assert a_question(answer_idx=2).answer_label == "C"

    def test_unlabelled_has_no_label(self) -> None:
        q = a_question(answer_idx=None)
        assert q.answer_label is None
        assert not q.is_labelled

    def test_rejects_out_of_range_answer(self) -> None:
        with pytest.raises(ValidationError, match="out of range"):
            a_question(answer_idx=4)

    def test_rejects_empty_question(self) -> None:
        with pytest.raises(ValidationError):
            a_question(question="")

    def test_rejects_too_many_options(self) -> None:
        with pytest.raises(ValidationError):
            a_question(options=["a", "b", "c", "d", "e"])

    def test_is_immutable(self) -> None:
        with pytest.raises(ValidationError):
            a_question().answer_idx = 1  # type: ignore[misc]

    def test_option_labels_cover_every_option(self) -> None:
        q = a_question()
        assert len(OPTION_LABELS) >= len(q.options)


class TestContentHash:
    def test_ignores_id_and_metadata(self) -> None:
        assert a_question(id="x", subject="Anatomy").content_hash() == (
            a_question(id="y", subject="Physiology").content_hash()
        )

    def test_ignores_case_and_surrounding_space(self) -> None:
        assert a_question().content_hash() == (
            a_question(
                question="  WHICH VESSEL SUPPLIES THE MYOCARDIUM?  ",
                options=["coronary artery", "PORTAL VEIN", " Aorta", "Vena Cava "],
            ).content_hash()
        )

    def test_differs_on_different_options(self) -> None:
        assert (
            a_question().content_hash()
            != (
                a_question(options=["Coronary artery", "Portal vein", "Aorta", "Renal vein"])
            ).content_hash()
        )

    def test_separator_prevents_field_collisions(self) -> None:
        """Concatenating fields without a separator would collide these two."""
        first = a_question(question="ab", options=["c", "d", "e", "f"])
        second = a_question(question="abc", options=["", "d", "e", "f"])
        # The second is invalid anyway (empty option), so compare against a legal pair.
        third = a_question(question="a", options=["bc", "d", "e", "f"])
        assert first.content_hash() != third.content_hash()
        del second


class TestPrediction:
    def test_scores_a_correct_answer(self) -> None:
        q = a_question(answer_idx=1)
        assert Prediction(question_id="q1", arm="base", seed=0, predicted_idx=1).is_correct(q)

    def test_scores_a_wrong_answer(self) -> None:
        q = a_question(answer_idx=1)
        assert not Prediction(question_id="q1", arm="base", seed=0, predicted_idx=3).is_correct(q)

    def test_abstention_is_none_not_false(self) -> None:
        # An unscorable item must never be silently counted as wrong.
        q = a_question(answer_idx=1)
        assert Prediction(question_id="q1", arm="base", seed=0).is_correct(q) is None

    def test_unlabelled_question_is_unscorable(self) -> None:
        q = a_question(answer_idx=None)
        assert (
            Prediction(question_id="q1", arm="base", seed=0, predicted_idx=1).is_correct(q) is None
        )


class TestPassage:
    def test_carries_provenance(self) -> None:
        p = Passage(
            id="p1", text="the heart has four chambers", corpus="parity", source_question_id="q9"
        )
        assert p.source_question_id == "q9"

    def test_rejects_empty_text(self) -> None:
        with pytest.raises(ValidationError):
            Passage(id="p1", text="", corpus="parity")
