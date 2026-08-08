"""Metrics tests, checked against hand-computable answers where possible."""

from __future__ import annotations

import numpy as np
import pytest

from fvr.eval.metrics import (
    accuracy,
    bootstrap_interval,
    mcnemar,
    minimum_detectable_effect,
    seed_variance,
)


class TestAccuracy:
    def test_known_value(self) -> None:
        assert accuracy([True, True, False, False]) == 0.5

    def test_all_correct(self) -> None:
        assert accuracy([True] * 10) == 1.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="zero items"):
            accuracy([])


class TestBootstrap:
    def test_point_estimate_matches_accuracy(self) -> None:
        data = [True] * 60 + [False] * 40
        assert bootstrap_interval(data, n_resamples=2000).point == pytest.approx(0.60)

    def test_interval_brackets_the_point(self) -> None:
        interval = bootstrap_interval([True] * 60 + [False] * 40, n_resamples=2000)
        assert interval.low < interval.point < interval.high

    def test_is_deterministic_for_a_seed(self) -> None:
        data = [True] * 60 + [False] * 40
        first = bootstrap_interval(data, n_resamples=2000, seed=7)
        second = bootstrap_interval(data, n_resamples=2000, seed=7)
        assert (first.low, first.high) == (second.low, second.high)

    def test_more_data_gives_a_tighter_interval(self) -> None:
        small = bootstrap_interval([True] * 30 + [False] * 20, n_resamples=4000, seed=1)
        large = bootstrap_interval([True] * 600 + [False] * 400, n_resamples=4000, seed=1)
        assert large.half_width < small.half_width

    def test_approximates_the_analytic_standard_error(self) -> None:
        # For p=0.5, n=1000 the normal 95% CI half-width is 1.96*sqrt(.25/1000) ~= 0.031.
        interval = bootstrap_interval([True] * 500 + [False] * 500, n_resamples=8000, seed=3)
        assert interval.half_width == pytest.approx(0.031, abs=0.006)

    def test_degenerate_all_correct_has_zero_width(self) -> None:
        interval = bootstrap_interval([True] * 50, n_resamples=1000)
        assert interval.low == interval.high == 1.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="zero items"):
            bootstrap_interval([])


class TestMcNemar:
    def test_identical_arms_are_not_significant(self) -> None:
        data = [True, False] * 50
        result = mcnemar(data, data)
        assert result.p_value == 1.0
        assert result.delta == 0.0

    def test_uses_only_discordant_pairs(self) -> None:
        # 90 agreements carry no information; the 10 disagreements do.
        a = [True] * 90 + [True] * 10
        b = [True] * 90 + [False] * 10
        result = mcnemar(a, b)
        assert (result.n_a_only, result.n_b_only) == (10, 0)

    def test_lopsided_disagreement_is_significant(self) -> None:
        a = [True] * 40 + [False] * 60
        b = [False] * 40 + [False] * 60
        assert mcnemar(a, b).is_significant()

    def test_balanced_disagreement_is_not_significant(self) -> None:
        a = [True] * 20 + [False] * 20 + [True] * 60
        b = [False] * 20 + [True] * 20 + [True] * 60
        result = mcnemar(a, b)
        assert not result.is_significant()
        assert result.delta == pytest.approx(0.0)

    def test_small_samples_use_the_exact_test(self) -> None:
        a = [True] * 5 + [False] * 95
        b = [False] * 5 + [False] * 95
        assert mcnemar(a, b).test == "mcnemar-exact"

    def test_large_samples_use_chi_square(self) -> None:
        a = [True] * 40 + [False] * 60
        b = [False] * 40 + [False] * 60
        assert mcnemar(a, b).test == "mcnemar-chi2"

    def test_verdict_is_human_readable(self) -> None:
        data = [True, False] * 50
        assert "not significant" in mcnemar(data, data).verdict()

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="same items"):
            mcnemar([True, False], [True])

    def test_a_two_point_gap_with_independent_errors_is_noise(self) -> None:
        """Two arms at 60% and 62% that disagree in both directions.

        This is the realistic case and the one the README's "2 points is noise"
        caveat refers to: the arms make largely independent mistakes, so the
        discordant pairs are near-balanced and the gap does not survive testing.
        """
        rng = np.random.default_rng(0)
        a = rng.random(1000) < 0.62
        b = rng.random(1000) < 0.60
        result = mcnemar(list(a), list(b))
        assert abs(result.delta) < 0.05
        assert not result.is_significant()

    def test_a_small_but_one_sided_gap_is_real(self) -> None:
        """The important counterexample, and why pairing beats a raw MDE.

        Here one arm gets everything the other does *plus* 15 more items, and
        never loses one. McNemar returns p < 1e-4 on a 1.5-point gap — far below
        the ~6-point minimum detectable effect an unpaired calculation predicts
        at n=1000. So "a small gap is noise" is only true when errors are
        independent; a consistent one-sided improvement is detectable much
        earlier. Reporting the discordant counts alongside p is what lets a
        reader tell the two situations apart.
        """
        base = [True] * 600 + [False] * 400
        better = [True] * 615 + [False] * 385
        result = mcnemar(better, base)
        assert result.delta == pytest.approx(0.015)
        assert (result.n_a_only, result.n_b_only) == (15, 0)
        assert result.is_significant()


class TestMinimumDetectableEffect:
    def test_shrinks_with_more_data(self) -> None:
        assert minimum_detectable_effect(4000) < minimum_detectable_effect(1000)

    def test_is_plausible_at_our_test_size(self) -> None:
        # ~6 points at n=1000, which is why a 2-point gap is not reportable.
        assert 0.04 < minimum_detectable_effect(1000) < 0.08

    def test_rejects_non_positive_n(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            minimum_detectable_effect(0)


class TestSeedVariance:
    def test_single_seed_has_zero_sd(self) -> None:
        assert seed_variance([0.6]) == (0.6, 0.0)

    def test_mean_and_sd(self) -> None:
        mean, sd = seed_variance([0.60, 0.62, 0.64])
        assert mean == pytest.approx(0.62)
        assert sd == pytest.approx(0.02)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="no accuracies"):
            seed_variance([])
