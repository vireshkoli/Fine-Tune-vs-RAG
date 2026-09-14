"""The free-text arm.

The arm exists because constrained A/B/C/D scoring measures whether a model can
*rank four candidates*, which is easier and narrower than producing the answer
unaided. These tests guard the two things that would make the free-text and MCQ
results incomparable: a different item set, and a different reference.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fvr.data.schema import Question
from fvr.eval.freetext import (
    MAX_NEW_TOKENS,
    FreeTextAnswer,
    FreeTextRun,
    RunNaming,
    UnlabelledQuestionError,
    option_dependent_reason,
    reference_answer,
    select_freetext_items,
)


def questions(n: int = 500, unlabelled: int = 0) -> list[Question]:
    items = [
        Question(
            id=f"q{i:04d}",
            question=f"Question {i}?",
            options=["alpha", "beta", "gamma", "delta"],
            answer_idx=i % 4,
            subject="Anatomy",
        )
        for i in range(n)
    ]
    for i in range(unlabelled):
        items[i] = items[i].model_copy(update={"answer_idx": None})
    return items


class TestReference:
    def test_is_the_gold_option_text(self) -> None:
        question = questions(1)[0]
        assert reference_answer(question) == question.options[question.answer_idx or 0]

    def test_refuses_an_unlabelled_question(self) -> None:
        """A reference of "None" would be silently graded against by the judge."""
        with pytest.raises(UnlabelledQuestionError):
            reference_answer(questions(1, unlabelled=1)[0])


class TestSelection:
    def test_is_deterministic_for_a_seed(self) -> None:
        first = [q.id for q in select_freetext_items(questions(), n=50, seed=7)]
        second = [q.id for q in select_freetext_items(questions(), n=50, seed=7)]
        assert first == second

    def test_does_not_depend_on_input_order(self) -> None:
        """Every arm must be given the same items or the comparison is unpaired.

        If selection depended on load order, two arms could silently be judged
        on different questions.
        """
        forward = questions()
        backward = list(reversed(forward))
        assert [q.id for q in select_freetext_items(forward, n=50)] == [
            q.id for q in select_freetext_items(backward, n=50)
        ]

    def test_different_seeds_select_differently(self) -> None:
        a = {q.id for q in select_freetext_items(questions(), n=50, seed=1)}
        b = {q.id for q in select_freetext_items(questions(), n=50, seed=2)}
        assert a != b

    def test_skips_unlabelled_questions(self) -> None:
        chosen = select_freetext_items(questions(unlabelled=100), n=50)
        assert all(q.answer_idx is not None for q in chosen)

    def test_refuses_to_silently_return_fewer(self) -> None:
        with pytest.raises(ValueError, match="only"):
            select_freetext_items(questions(n=10), n=50)

    def test_returns_exactly_n(self) -> None:
        assert len(select_freetext_items(questions(), n=300)) == 300


def an_answer(text: str = "Coronary artery.", **overrides: object) -> FreeTextAnswer:
    defaults = {
        "question_id": "q1",
        "subject": "Anatomy",
        "question": "Which vessel?",
        "reference": "Coronary artery",
        "answer": text,
        "prompt_tokens": 100,
        "completion_tokens": 8,
    }
    return FreeTextAnswer(**{**defaults, **overrides})  # type: ignore[arg-type]


class TestRun:
    def a_run(self, answers: list[FreeTextAnswer]) -> FreeTextRun:
        return FreeTextRun(
            arm="base",
            seed=42,
            split_sha256="abc",
            model={"repo_id": "x"},
            environment={"git_sha": "y"},
            answers=answers,
        )

    def test_counts_empty_answers_rather_than_dropping_them(self) -> None:
        """An arm that refuses is failing, not absent.

        Excluding blanks would score the arm on the subset where it spoke.
        """
        run = self.a_run([an_answer(), an_answer(""), an_answer("   ")])
        assert run.empty_answers == 2
        assert run.to_json()["n_items"] == 3

    def test_reports_mean_completion_length(self) -> None:
        run = self.a_run([an_answer(completion_tokens=10), an_answer(completion_tokens=20)])
        assert run.mean_completion_tokens == 15

    def test_round_trips_through_json(self, tmp_path: Path) -> None:
        run = self.a_run([an_answer()])
        path = tmp_path / "nested" / "base_seed42.json"
        run.write(path)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["arm"] == "base"
        assert loaded["answers"][0]["reference"] == "Coronary artery"

    def test_empty_run_does_not_divide_by_zero(self) -> None:
        assert self.a_run([]).mean_completion_tokens == 0.0

    def test_counts_answers_that_hit_the_generation_bound(self) -> None:
        """A capped answer is a fragment; the judge scores the fragment.

        The count is reported per run because a share that differs between
        arms is the bound leaking into the comparison — which is exactly what
        happened to the MIRIAD fine-tuned arms at 96 tokens.
        """
        run = self.a_run(
            [an_answer(completion_tokens=96), an_answer(completion_tokens=95), an_answer()]
        )
        run.max_new_tokens = 96
        assert run.capped_answers == 1
        payload = run.to_json()
        assert payload["max_new_tokens"] == 96
        assert payload["capped_answers"] == 1

    def test_unknown_bound_reports_no_capped_answers(self) -> None:
        run = self.a_run([an_answer(completion_tokens=1000)])
        assert run.max_new_tokens is None
        assert run.capped_answers == 0
        assert run.to_json()["capped_answers"] == 0


class TestGenerationBound:
    def test_every_dataset_has_a_bound(self) -> None:
        assert set(MAX_NEW_TOKENS) == {"medmcqa", "miriad"}

    def test_miriad_bound_clears_the_longest_reference(self) -> None:
        """The frozen MIRIAD test references top out at 212 Qwen3 tokens.

        The bound must sit above that with headroom, or an arm that writes at
        reference length is scored on where the budget ran out.
        """
        assert MAX_NEW_TOKENS["miriad"] >= 256

    def test_medmcqa_bound_is_the_original(self) -> None:
        # The six MedMCQA runs were generated at 96; changing this silently
        # would make new seeds incomparable with them.
        assert MAX_NEW_TOKENS["medmcqa"] == 96


class TestRunNaming:
    def test_medmcqa_runs_are_named_by_arm_and_seed(self) -> None:
        naming = RunNaming("medmcqa", 42)
        assert naming.stem("qlora") == "qlora_seed42"
        assert naming.arm_of("qlora_seed42") == "qlora"
        assert naming.glob == "*_seed42.json"

    def test_miriad_runs_are_named_by_tag(self) -> None:
        """Two MIRIAD runs share the arm "qlora" and differ only by adapter."""
        naming = RunNaming("miriad", 42)
        assert naming.stem("miriad-qlora-doc") == "miriad-qlora-doc"
        assert naming.arm_of("miriad-qlora-doc") == "miriad-qlora-doc"
        assert naming.glob == "miriad-*.json"

    def test_rejects_a_stem_from_the_other_dataset(self) -> None:
        with pytest.raises(ValueError):
            RunNaming("medmcqa", 42).arm_of("miriad-base")
        with pytest.raises(ValueError):
            RunNaming("miriad", 42).arm_of("base_seed42")

    def test_rejects_another_seed(self) -> None:
        with pytest.raises(ValueError):
            RunNaming("medmcqa", 42).arm_of("qlora_seed1")

    def test_globs_do_not_overlap(self, tmp_path: Path) -> None:
        """A MIRIAD judging pass must never pick up a MedMCQA run, or vice versa."""
        for name in ("base_seed42.json", "miriad-base.json", "miriad-qlora-qa.json"):
            (tmp_path / name).write_text("{}", encoding="utf-8")
        medmcqa = {p.name for p in tmp_path.glob(RunNaming("medmcqa", 42).glob)}
        miriad = {p.name for p in tmp_path.glob(RunNaming("miriad", 42).glob)}
        assert medmcqa == {"base_seed42.json"}
        assert miriad == {"miriad-base.json", "miriad-qlora-qa.json"}


def with_gold(gold: str, qid: str = "g1") -> Question:
    return Question(
        id=qid,
        question="Which of the following?",
        options=[gold, "other one", "other two", "other three"],
        answer_idx=0,
        subject="Dental",
    )


class TestOptionDependence:
    """Items whose gold answer is meaningless once the options are hidden.

    The first smoke run produced a reference of `B>A>D>C`. No free-text answer
    can agree with that, so a judge would score every arm 0 on it and the arms
    would look worse for a reason that has nothing to do with them.
    """

    @pytest.mark.parametrize(
        "gold",
        [
            "B>A>D>C",
            "All of the above",
            "All of the above.",
            "None of the above",
            "All",
            "None",
            "b,c,d true a false",
            "A. i) B. ii) C. i) D. i) E. i)",
            "1,2 & 3",
            "Both A and B",
            "Both of the above",
        ],
    )
    def test_flags_golds_that_need_the_options(self, gold: str) -> None:
        assert option_dependent_reason(with_gold(gold)) is not None

    @pytest.mark.parametrize(
        "gold",
        [
            # Real golds a first, looser rule set wrongly flagged. Pinned here
            # so a future "simplification" cannot throw away gradable items.
            "3.1",
            "3-5%",
            "Both GH and prolactin",
            "Both lateral and medial pterygoid muscle",
            "Vitamin A and D",
            "1, decreases",
            # Ordinary self-contained answers.
            "Glycogen synthesis",
            "aPTT",
            "Erythromycin",
            "Fibroblasts",
        ],
    )
    def test_keeps_golds_that_stand_alone(self, gold: str) -> None:
        assert option_dependent_reason(with_gold(gold)) is None

    def test_selection_never_returns_an_option_dependent_item(self) -> None:
        pool = questions(400) + [with_gold("All of the above", f"dep{i}") for i in range(100)]
        chosen = select_freetext_items(pool, n=300)
        assert len(chosen) == 300
        assert all(option_dependent_reason(q) is None for q in chosen)

    def test_exclusion_happens_before_sampling_so_n_is_exact(self) -> None:
        pool = questions(290) + [with_gold("All of the above", f"dep{i}") for i in range(50)]
        assert len(select_freetext_items(pool, n=290)) == 290
        with pytest.raises(ValueError, match="only 290"):
            select_freetext_items(pool, n=291)

    def test_the_run_records_how_many_were_excluded(self) -> None:
        run = FreeTextRun(
            arm="base",
            seed=42,
            split_sha256="abc",
            model={},
            environment={},
            excluded_option_dependent=23,
        )
        assert run.to_json()["excluded_option_dependent"] == 23
