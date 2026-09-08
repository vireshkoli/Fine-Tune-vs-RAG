"""Prompt-parity tests — the fairness control, asserted rather than promised.

If these pass, the only textual difference between a RAG arm and a non-RAG arm
is the inserted context block. Any accuracy gap therefore cannot be explained
by prompt wording.
"""

from __future__ import annotations

from fvr.data.schema import Passage, Question
from fvr.inference.arms import ARMS, get_arm, headline_arms
from fvr.prompts.templates import (
    ANSWER_INSTRUCTION,
    FREETEXT_INSTRUCTION,
    FREETEXT_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_freetext_prompt,
    build_prompt,
    format_options,
    strip_context,
)


def a_question() -> Question:
    return Question(
        id="q1",
        question="Which vessel supplies the myocardium?",
        options=["Coronary artery", "Portal vein", "Aorta", "Vena cava"],
        answer_idx=0,
        subject="Anatomy",
    )


def some_passages(n: int = 2) -> list[Passage]:
    return [
        Passage(id=f"p{i}", text=f"Reference passage number {i}.", corpus="external")
        for i in range(n)
    ]


class TestParity:
    def test_stripping_context_reproduces_the_plain_prompt_exactly(self) -> None:
        """The core fairness assertion."""
        question = a_question()
        plain = build_prompt(question)
        with_context = build_prompt(question, some_passages())
        assert strip_context(with_context) == plain

    def test_system_prompt_is_identical_across_arms(self) -> None:
        question = a_question()
        assert build_prompt(question).system == build_prompt(question, some_passages()).system

    def test_answer_instruction_present_in_both(self) -> None:
        question = a_question()
        assert ANSWER_INSTRUCTION in build_prompt(question).user
        assert ANSWER_INSTRUCTION in build_prompt(question, some_passages()).user

    def test_rag_prompt_is_a_strict_superset(self) -> None:
        question = a_question()
        plain = build_prompt(question)
        rag = build_prompt(question, some_passages())
        assert plain.user in rag.user
        assert len(rag.user) > len(plain.user)

    def test_context_count_does_not_change_the_question_body(self) -> None:
        question = a_question()
        one = strip_context(build_prompt(question, some_passages(1)))
        five = strip_context(build_prompt(question, some_passages(5)))
        assert one == five


class TestFormatting:
    def test_options_are_lettered_in_order(self) -> None:
        assert format_options(a_question()).splitlines() == [
            "A. Coronary artery",
            "B. Portal vein",
            "C. Aorta",
            "D. Vena cava",
        ]

    def test_passages_are_numbered_for_citation(self) -> None:
        user = build_prompt(a_question(), some_passages(3)).user
        assert "[1]" in user and "[2]" in user and "[3]" in user

    def test_messages_have_system_then_user(self) -> None:
        messages = build_prompt(a_question()).as_messages()
        assert [m["role"] for m in messages] == ["system", "user"]
        assert messages[0]["content"] == SYSTEM_PROMPT

    def test_has_context_flag_matches_reality(self) -> None:
        assert not build_prompt(a_question()).has_context
        assert build_prompt(a_question(), some_passages()).has_context

    def test_strip_context_is_a_noop_without_context(self) -> None:
        plain = build_prompt(a_question())
        assert strip_context(plain) is plain


class TestArms:
    def test_four_headline_arms(self) -> None:
        assert len(headline_arms()) == 4

    def test_six_arms_total(self) -> None:
        assert len(ARMS) == 6

    def test_arm_names_are_unique(self) -> None:
        names = [arm.name for arm in ARMS]
        assert len(names) == len(set(names))

    def test_every_combination_of_switches_is_covered(self) -> None:
        combos = {(arm.uses_adapter, arm.corpus) for arm in ARMS}
        assert combos == {
            (False, "none"),
            (False, "external"),
            (False, "parity"),
            (True, "none"),
            (True, "external"),
            (True, "parity"),
        }

    def test_parity_arms_are_diagnostics_not_headline(self) -> None:
        # They belong in REPORT.md, not the 45-second README table.
        assert not get_arm("rag-parity").headline
        assert not get_arm("qlora-rag-parity").headline

    def test_uses_retrieval_derives_from_corpus(self) -> None:
        assert not get_arm("base").uses_retrieval
        assert get_arm("rag-external").uses_retrieval

    def test_unknown_arm_raises(self) -> None:
        try:
            get_arm("nope")
        except KeyError as exc:
            assert "nope" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected KeyError")


class TestFreeTextParity:
    """The free-text arms get the same fairness control as the MCQ arms.

    They share one context-insertion helper, so these assertions are about a
    property that holds by construction — but the point of asserting it is that
    a future edit could give free text its own insertion path and quietly break
    the comparison between free-text RAG and free-text non-RAG.
    """

    def test_stripping_context_reproduces_the_plain_freetext_prompt(self) -> None:
        plain = build_freetext_prompt(a_question())
        with_context = build_freetext_prompt(a_question(), some_passages())
        assert strip_context(with_context).user == plain.user
        assert strip_context(with_context).system == plain.system

    def test_system_prompt_is_identical_across_freetext_arms(self) -> None:
        assert (
            build_freetext_prompt(a_question()).system
            == build_freetext_prompt(a_question(), some_passages()).system
            == FREETEXT_SYSTEM_PROMPT
        )

    def test_context_block_is_byte_identical_to_the_mcq_one(self) -> None:
        """Same passages must produce the same context block in both modes.

        Otherwise the free-text RAG arm would receive different evidence from
        the MCQ RAG arm and the two could not be compared.
        """
        passages = some_passages(3)
        mcq_context = build_prompt(a_question(), passages).user
        free_context = build_freetext_prompt(a_question(), passages).user
        marker = "\n\n---\n\n"
        assert mcq_context.split(marker)[0] == free_context.split(marker)[0]

    def test_options_are_never_shown(self) -> None:
        """The whole point of the arm: generate the answer, do not rank four."""
        question = a_question()
        prompt = build_freetext_prompt(question, some_passages())
        for option in question.options[1:]:
            assert option not in prompt.user
        assert format_options(question) not in prompt.user

    def test_it_does_not_ask_for_a_letter(self) -> None:
        prompt = build_freetext_prompt(a_question())
        assert ANSWER_INSTRUCTION not in prompt.user
        assert FREETEXT_INSTRUCTION in prompt.user
        assert "A, B, C" not in prompt.system

    def test_the_question_body_is_unchanged_by_context(self) -> None:
        bodies = {
            strip_context(build_freetext_prompt(a_question(), some_passages(n))).user
            for n in (0, 1, 5)
        }
        assert len(bodies) == 1
