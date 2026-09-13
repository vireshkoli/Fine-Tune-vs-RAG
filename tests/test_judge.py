"""The LLM judge.

A judge is the weakest link in any evaluation that uses one, so these tests are
about its failure modes rather than its happy path: replies it cannot parse,
preferences that flip when the answers swap places, and the skewed-scale trap
where a judge that always says "2" looks 85% accurate.

The judge is injected as a callable, so all of this runs on CPU with a scripted
stand-in and no model is ever loaded.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from fvr.eval.judge import (
    Agreement,
    ArmJudgement,
    JudgeFn,
    PairwiseResult,
    UnparseableVerdictError,
    cohens_kappa,
    compare_many,
    compare_pairwise,
    parse_score,
    parse_verdict,
    score_many,
    score_pointwise,
    summarise_pairwise,
)
from fvr.prompts.judge import (
    RUBRIC_VERSION,
    build_pairwise_prompt,
    build_pointwise_prompt,
)


def scripted(replies: list[str]) -> JudgeFn:
    """A judge that returns each reply in turn."""
    stream: Iterator[str] = iter(replies)

    def judge(_messages: list[dict[str, str]], _seed: int) -> str:
        return next(stream)

    return judge


def always(reply: str) -> JudgeFn:
    def judge(_messages: list[dict[str, str]], _seed: int) -> str:
        return reply

    return judge


class TestParsing:
    @pytest.mark.parametrize(
        ("reply", "expected"),
        [
            ("SCORE: 2", 2),
            ("SCORE:0", 0),
            ("score: 1", 1),
            ("Reasoning aside.\nSCORE: 2\n", 2),
        ],
    )
    def test_reads_a_score(self, reply: str, expected: int) -> None:
        assert parse_score(reply) == expected

    @pytest.mark.parametrize("reply", ["", "I think it is fine", "SCORE: seven", "SCORE: 5"])
    def test_refuses_an_unparseable_score(self, reply: str) -> None:
        """A missing score must not be silently counted as a zero.

        Pooling parse failures with genuine disagreements would deflate every
        arm by however often the judge went off-format.
        """
        with pytest.raises(UnparseableVerdictError):
            parse_score(reply)

    @pytest.mark.parametrize(
        ("reply", "expected"),
        [("VERDICT: A", "A"), ("VERDICT: B", "B"), ("verdict: tie", "TIE")],
    )
    def test_reads_a_verdict(self, reply: str, expected: str) -> None:
        assert parse_verdict(reply) == expected

    def test_refuses_an_unparseable_verdict(self) -> None:
        with pytest.raises(UnparseableVerdictError):
            parse_verdict("Both are good")


class TestPointwise:
    def test_averages_across_seeds(self) -> None:
        result = score_pointwise(
            "q1", "Q?", "ref", "cand", scripted(["SCORE: 2", "SCORE: 1", "SCORE: 2"])
        )
        assert result.scores == (2, 1, 2)
        assert result.mean == pytest.approx(5 / 3)
        assert result.sd > 0

    def test_reports_zero_variance_when_the_judge_is_consistent(self) -> None:
        result = score_pointwise("q1", "Q?", "ref", "cand", always("SCORE: 2"))
        assert result.sd == 0.0
        assert result.normalised == 1.0

    def test_counts_unparseable_replies_separately_from_zeros(self) -> None:
        """The distinction the whole design rests on."""
        result = score_pointwise(
            "q1", "Q?", "ref", "cand", scripted(["SCORE: 2", "no idea", "SCORE: 2"])
        )
        assert result.scores == (2, 2)
        assert result.unparseable == 1
        assert result.normalised == 1.0, "a failed parse must not drag the score down"

    def test_normalises_onto_the_accuracy_scale(self) -> None:
        assert score_pointwise("q", "Q", "r", "c", always("SCORE: 1")).normalised == 0.5

    def test_an_empty_candidate_still_reaches_the_judge(self) -> None:
        prompt = build_pointwise_prompt("Q?", "ref", "   ")
        assert "(no answer given)" in prompt.user


class TestPairwisePositionBias:
    def test_consistent_preference_names_a_winner(self) -> None:
        # Forward says A; reversed says B, which *is* A once flipped back.
        result = compare_pairwise(
            "q1",
            "Q?",
            "ref",
            "arm-a",
            "answer a",
            "arm-b",
            "answer b",
            scripted(["VERDICT: A", "VERDICT: B"]),
        )
        assert not result.inconsistent
        assert result.winner == "arm-a"

    def test_a_preference_that_flips_with_order_is_not_a_result(self) -> None:
        """A judge that prefers whichever answer came first has told us nothing."""
        result = compare_pairwise(
            "q1",
            "Q?",
            "ref",
            "arm-a",
            "answer a",
            "arm-b",
            "answer b",
            scripted(["VERDICT: A", "VERDICT: A"]),
        )
        assert result.inconsistent
        assert result.winner is None

    def test_a_tie_is_preserved_rather_than_broken(self) -> None:
        result = compare_pairwise(
            "q1",
            "Q?",
            "ref",
            "arm-a",
            "a",
            "arm-b",
            "b",
            scripted(["VERDICT: TIE", "VERDICT: TIE"]),
        )
        assert not result.inconsistent
        assert result.winner is None

    def test_summary_reports_the_judges_own_error_rate(self) -> None:
        results = [
            PairwiseResult("1", "a", "b", "A", "A", False),
            PairwiseResult("2", "a", "b", "B", "B", False),
            PairwiseResult("3", "a", "b", "A", "B", True),
            PairwiseResult("4", "a", "b", "TIE", "TIE", False),
        ]
        summary = summarise_pairwise(results)
        assert (summary.a_wins, summary.b_wins, summary.ties) == (1, 1, 1)
        assert summary.inconsistent == 1
        assert summary.position_bias_rate == 0.25
        assert summary.as_json()["rubric_version"] == RUBRIC_VERSION

    def test_both_orderings_are_actually_sent(self) -> None:
        seen: list[str] = []

        def judge(messages: list[dict[str, str]], _seed: int) -> str:
            seen.append(messages[1]["content"])
            return "VERDICT: TIE"

        compare_pairwise("q", "Q?", "ref", "a", "FIRST", "b", "SECOND", judge)
        assert len(seen) == 2
        assert seen[0].index("FIRST") < seen[0].index("SECOND")
        assert seen[1].index("SECOND") < seen[1].index("FIRST")


class TestAgreement:
    def test_perfect_agreement_on_varied_labels(self) -> None:
        agreement = cohens_kappa([2, 1, 0, 2, 1], [2, 1, 0, 2, 1])
        assert agreement.exact_agreement == 1.0
        assert agreement.kappa == pytest.approx(1.0)
        assert agreement.verdict() == "almost perfect"

    def test_a_constant_judge_scores_zero_not_high(self) -> None:
        """The trap κ exists to catch.

        A judge that always says 2 agrees with a 2-heavy human 80% of the time
        by doing nothing at all. Raw agreement rewards it; κ does not.
        """
        human = [2, 2, 2, 2, 1]
        machine = [2, 2, 2, 2, 2]
        agreement = cohens_kappa(human, machine)
        assert agreement.exact_agreement == 0.8
        assert agreement.kappa == pytest.approx(0.0, abs=1e-9)
        assert agreement.verdict() == "slight"

    def test_systematic_disagreement_goes_negative(self) -> None:
        assert cohens_kappa([2, 0, 2, 0], [0, 2, 0, 2]).kappa < 0

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            cohens_kappa([1, 2], [1])

    def test_rejects_empty_input(self) -> None:
        with pytest.raises(ValueError, match="no labels"):
            cohens_kappa([], [])

    def test_bands_are_reported_so_the_number_is_not_left_bare(self) -> None:
        assert Agreement(50, 0.9, 0.75).verdict() == "substantial"
        assert Agreement(50, 0.5, 0.35).verdict() == "fair"
        assert Agreement(50, 0.1, -0.2).verdict() == "worse than chance"


class TestArmJudgement:
    def test_aggregates_scores_and_judge_variance(self) -> None:
        judgement = ArmJudgement(arm="qlora")
        judgement.results = [
            score_pointwise("1", "Q", "r", "c", always("SCORE: 2")),
            score_pointwise("2", "Q", "r", "c", scripted(["SCORE: 0", "SCORE: 2", "SCORE: 1"])),
        ]
        assert judgement.mean_score == pytest.approx((1.0 + 0.5) / 2)
        assert judgement.judge_sd > 0
        assert judgement.as_json()["n"] == 2

    def test_empty_is_zero_rather_than_an_error(self) -> None:
        assert ArmJudgement(arm="none").mean_score == 0.0


class TestRubricIsFrozen:
    def test_both_rubrics_demand_a_parseable_format(self) -> None:
        pointwise = build_pointwise_prompt("Q?", "ref", "cand")
        pairwise = build_pairwise_prompt("Q?", "ref", "a", "b")
        assert "SCORE:" in pointwise.user
        assert "VERDICT:" in pairwise.user

    def test_the_rubric_grades_against_the_reference_not_the_judges_opinion(self) -> None:
        """The design choice: the judge reads, it does not practise medicine."""
        prompt = build_pointwise_prompt("Q?", "ref", "cand")
        assert "REFERENCE" in prompt.user
        assert "not being asked for your own medical opinion" in prompt.system

    def test_the_rubric_forbids_rewarding_style(self) -> None:
        for prompt in (
            build_pointwise_prompt("Q?", "r", "c"),
            build_pairwise_prompt("Q?", "r", "a", "b"),
        ):
            assert "length" in prompt.user


class TestSeedsReachTheJudge:
    """The bug this class exists to prevent.

    An earlier version looped over the seeds and discarded them, calling the
    judge with identical arguments each time. Against a deterministic backend
    that reports a judge SD of exactly zero — a confident claim of perfect
    self-consistency, measured by never varying anything.
    """

    def test_each_seed_is_passed_through(self) -> None:
        seen: list[int] = []

        def judge(_messages: list[dict[str, str]], seed: int) -> str:
            seen.append(seed)
            return "SCORE: 2"

        score_pointwise("q", "Q?", "ref", "cand", judge, seeds=(11, 22, 33))
        assert seen == [11, 22, 33]

    def test_a_seed_sensitive_judge_produces_nonzero_variance(self) -> None:
        def judge(_messages: list[dict[str, str]], seed: int) -> str:
            return f"SCORE: {seed % 3}"

        result = score_pointwise("q", "Q?", "ref", "cand", judge, seeds=(0, 1, 2))
        assert result.scores == (0, 1, 2)
        assert result.sd > 0

    def test_pairwise_uses_one_seed_for_both_orderings(self) -> None:
        """Otherwise position bias is confounded with sampling noise."""
        seen: list[int] = []

        def judge(_messages: list[dict[str, str]], seed: int) -> str:
            seen.append(seed)
            return "VERDICT: TIE"

        compare_pairwise("q", "Q?", "r", "a", "A", "b", "B", judge, seed=7)
        assert seen == [7, 7]


class TestConcurrentJudging:
    """Thousands of calls go to a served judge in parallel; order must survive."""

    @staticmethod
    def slow_judge(messages: list[dict[str, str]], seed: int) -> str:
        import time

        content = messages[1]["content"]
        # Later items finish *first*, so completion order is the reverse of input.
        item = int(content.split("CANDIDATE:")[1].split()[0])
        time.sleep(0.001 * (20 - item))
        return f"SCORE: {(item + seed) % 3}"

    def items(self) -> list[tuple[str, str, str, str]]:
        return [(f"q{i}", "Question?", "ref", f"{i} answer") for i in range(20)]

    def test_results_come_back_in_input_order(self) -> None:
        results = score_many(self.items(), self.slow_judge, seeds=(0,), workers=8)
        assert [r.item_id for r in results] == [f"q{i}" for i in range(20)]
        assert [r.scores[0] for r in results] == [i % 3 for i in range(20)]

    def test_concurrent_equals_sequential(self) -> None:
        serial = score_many(self.items(), self.slow_judge, seeds=(0, 1, 2), workers=1)
        pooled = score_many(self.items(), self.slow_judge, seeds=(0, 1, 2), workers=8)
        assert [r.scores for r in serial] == [r.scores for r in pooled]

    def test_a_failing_call_is_not_swallowed(self) -> None:
        def broken(_messages: list[dict[str, str]], _seed: int) -> str:
            raise RuntimeError("judge down")

        with pytest.raises(RuntimeError, match="judge down"):
            score_many(self.items(), broken, workers=4)

    def test_pairwise_many_keeps_order_and_both_orderings(self) -> None:
        items = [(f"q{i}", "Q?", "ref", "a", "A", "b", "B") for i in range(12)]
        results = compare_many(items, always("VERDICT: TIE"), workers=4)
        assert [r.item_id for r in results] == [f"q{i}" for i in range(12)]
        assert all(not r.inconsistent for r in results)
