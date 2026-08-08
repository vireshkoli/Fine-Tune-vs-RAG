"""Scoring with honest uncertainty.

A 2-point accuracy gap on 1,000 examples is usually noise, so no accuracy is
ever reported here without an interval, and arm-vs-arm comparisons use a
*paired* test. The arms all see the identical frozen test set, so pairing is
both valid and much more powerful than treating the two accuracies as
independent samples — an unpaired t-test would be simply the wrong tool.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

DEFAULT_BOOTSTRAP = 10_000
DEFAULT_CONFIDENCE = 0.95


@dataclass(frozen=True)
class Interval:
    """A point estimate with a confidence interval."""

    point: float
    low: float
    high: float
    confidence: float = DEFAULT_CONFIDENCE

    @property
    def half_width(self) -> float:
        return (self.high - self.low) / 2

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.low:.3f}, {self.high:.3f}]"


@dataclass(frozen=True)
class Comparison:
    """A paired arm-vs-arm result."""

    delta: float
    p_value: float
    n_a_only: int
    n_b_only: int
    test: Literal["mcnemar-exact", "mcnemar-chi2"]

    def is_significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha

    def verdict(self, alpha: float = 0.05) -> str:
        if not self.is_significant(alpha):
            return f"not significant (p={self.p_value:.3f})"
        return f"significant (p={self.p_value:.4f})"


def accuracy(correct: Sequence[bool]) -> float:
    if not correct:
        raise ValueError("cannot compute accuracy over zero items")
    return float(np.mean(np.asarray(correct, dtype=bool)))


def bootstrap_interval(
    correct: Sequence[bool],
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap CI over the per-item correctness vector.

    Bootstrap rather than a normal approximation because accuracy near the
    floor (a zero-shot arm can sit close to the 25% chance line on 4-option
    MCQ) has a skewed sampling distribution that a Wald interval handles badly.
    """
    values = np.asarray(correct, dtype=bool)
    if values.size == 0:
        raise ValueError("cannot bootstrap over zero items")

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(n_resamples, values.size))
    means = values[draws].mean(axis=1)
    tail = (1 - confidence) / 2
    return Interval(
        point=float(values.mean()),
        low=float(np.quantile(means, tail)),
        high=float(np.quantile(means, 1 - tail)),
        confidence=confidence,
    )


def mcnemar(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> Comparison:
    """Paired significance test for two arms on the same items.

    Only the discordant pairs carry information: items both arms get right, or
    both get wrong, say nothing about which is better. The exact binomial test
    is used when discordant counts are small, where the chi-square
    approximation is unreliable.
    """
    a = np.asarray(a_correct, dtype=bool)
    b = np.asarray(b_correct, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"arms must cover the same items, got {a.shape} and {b.shape}")
    if a.size == 0:
        raise ValueError("cannot compare over zero items")

    a_only = int(np.sum(a & ~b))
    b_only = int(np.sum(~a & b))
    discordant = a_only + b_only
    delta = float(a.mean() - b.mean())

    if discordant == 0:
        return Comparison(delta, 1.0, a_only, b_only, "mcnemar-exact")

    if discordant < 25:
        p = float(stats.binomtest(a_only, discordant, 0.5).pvalue)
        return Comparison(delta, p, a_only, b_only, "mcnemar-exact")

    statistic = (abs(a_only - b_only) - 1) ** 2 / discordant
    p = float(stats.chi2.sf(statistic, df=1))
    return Comparison(delta, p, a_only, b_only, "mcnemar-chi2")


def minimum_detectable_effect(
    n: int, *, baseline: float = 0.5, alpha: float = 0.05, power: float = 0.80
) -> float:
    """Smallest accuracy gap this test-set size can resolve.

    Reported in the README so a reader can tell at a glance whether a headline
    gap is meaningful. Uses the unpaired two-proportion formula, which is
    conservative here: pairing makes the real sensitivity better, so quoting
    this number never overstates what the benchmark can see.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    z_alpha = float(stats.norm.ppf(1 - alpha / 2))
    z_power = float(stats.norm.ppf(power))
    return float((z_alpha + z_power) * np.sqrt(2 * baseline * (1 - baseline) / n))


def seed_variance(accuracies: Sequence[float]) -> tuple[float, float]:
    """Mean and sample standard deviation across training seeds."""
    values = np.asarray(accuracies, dtype=float)
    if values.size == 0:
        raise ValueError("no accuracies given")
    if values.size == 1:
        return float(values[0]), 0.0
    return float(values.mean()), float(values.std(ddof=1))
