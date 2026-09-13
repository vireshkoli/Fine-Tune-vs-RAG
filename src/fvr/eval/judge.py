"""LLM-judged scoring for the free-text arms.

The MCQ arms need no judge: constrained log-prob scoring over A/B/C/D is exact.
Free text does need one, and a judge is the weakest link in any evaluation that
uses it — so this module is built around the assumption that the judge is
*wrong some of the time* and that the interesting question is how often.

Three defences, all reported rather than assumed:

**Validated, not trusted.** :func:`cohens_kappa` compares the judge against
human labels on a hand-graded subset. The κ is published whatever it turns out
to be. An unvalidated judge is an opinion with a model number attached.

**Variance is measured.** The judge is run at several seeds and the standard
deviation of its scores is reported alongside the mean. A judge that disagrees
with itself is telling you the size of its own error bar.

**Position bias is controlled.** Pairwise comparisons are run in both orders.
Models systematically prefer whichever answer came first, so a pairwise result
from a single ordering measures the judge as much as the answers. Disagreement
between orderings is surfaced as :attr:`PairwiseResult.inconsistent` rather than
averaged into silence.

Nothing here loads a model. The judge is injected as a callable, so every rule
below is tested on CPU with a scripted judge and the vLLM server stays an
implementation detail of the caller.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Literal

from fvr.prompts.judge import (
    RUBRIC_VERSION,
    build_pairwise_prompt,
    build_pointwise_prompt,
)

#: A callable taking rendered messages *and a seed*, returning the reply text.
#: Injected so this module never imports vLLM and stays CPU-testable.
#:
#: The seed is part of the signature rather than the judge's internal state
#: because seed-to-seed spread is a reported number. A judge called three times
#: with no seed would either return the same answer three times — reporting a
#: variance of zero that means nothing — or vary uncontrollably, which is worse.
JudgeFn = Callable[[list[dict[str, str]], int], str]

Verdict = Literal["A", "B", "TIE"]

#: Pointwise scale, from POINTWISE_RUBRIC. 2 = agrees, 1 = partial, 0 = disagrees.
MAX_POINTWISE = 2

_SCORE = re.compile(r"SCORE\s*:\s*([0-2])", re.IGNORECASE)
_VERDICT = re.compile(r"VERDICT\s*:\s*(A|B|TIE)", re.IGNORECASE)


class UnparseableVerdictError(Exception):
    """The judge replied in a format the rubric did not ask for."""


def parse_score(reply: str) -> int:
    """Extract the pointwise score.

    Deliberately strict about the *value* and forgiving about the surroundings:
    a judge that wraps the line in prose is still answering, but a judge that
    returns no score at all must not be silently counted as a zero — a parse
    failure and a genuine disagreement are different events and pooling them
    would quietly deflate every arm.
    """
    match = _SCORE.search(reply)
    if match is None:
        raise UnparseableVerdictError(f"no SCORE line in judge reply: {reply[:200]!r}")
    return int(match.group(1))


def parse_verdict(reply: str) -> Verdict:
    match = _VERDICT.search(reply)
    if match is None:
        raise UnparseableVerdictError(f"no VERDICT line in judge reply: {reply[:200]!r}")
    verdict = match.group(1).upper()
    return "TIE" if verdict == "TIE" else ("A" if verdict == "A" else "B")


@dataclass(frozen=True)
class PointwiseResult:
    """One item graded at every judge seed."""

    item_id: str
    scores: tuple[int, ...]
    unparseable: int = 0

    @property
    def mean(self) -> float:
        return statistics.fmean(self.scores) if self.scores else 0.0

    @property
    def sd(self) -> float:
        """Judge disagreement with itself, across seeds."""
        return statistics.stdev(self.scores) if len(self.scores) > 1 else 0.0

    @property
    def normalised(self) -> float:
        """Mean score on 0-1, so it can sit beside an accuracy."""
        return self.mean / MAX_POINTWISE

    @property
    def majority(self) -> int:
        """The modal score — used where a single label is needed, as for κ."""
        if not self.scores:
            raise ValueError("no scores")
        return statistics.mode(self.scores)


def score_pointwise(
    item_id: str,
    question: str,
    reference: str,
    candidate: str,
    judge: JudgeFn,
    *,
    seeds: Sequence[int] = (0, 1, 2),
) -> PointwiseResult:
    """Grade one candidate answer at every seed."""
    prompt = build_pointwise_prompt(question, reference, candidate)
    scores: list[int] = []
    unparseable = 0
    for seed in seeds:
        try:
            scores.append(parse_score(judge(prompt.as_messages(), seed)))
        except UnparseableVerdictError:
            unparseable += 1
    return PointwiseResult(item_id=item_id, scores=tuple(scores), unparseable=unparseable)


@dataclass(frozen=True)
class PairwiseResult:
    """One pair, judged in both orders.

    ``winner`` is the arm the judge preferred *consistently*. When the two
    orderings disagree the result is a tie and ``inconsistent`` is set, because
    a preference that flips when the answers swap places is a measurement of the
    judge, not of the answers.
    """

    item_id: str
    arm_a: str
    arm_b: str
    forward: Verdict
    reversed_: Verdict
    inconsistent: bool

    @property
    def winner(self) -> str | None:
        if self.inconsistent or self.forward == "TIE":
            return None
        return self.arm_a if self.forward == "A" else self.arm_b


def _flip(verdict: Verdict) -> Verdict:
    """Translate a reversed-order verdict back into forward-order terms."""
    if verdict == "TIE":
        return "TIE"
    return "B" if verdict == "A" else "A"


def compare_pairwise(
    item_id: str,
    question: str,
    reference: str,
    arm_a: str,
    answer_a: str,
    arm_b: str,
    answer_b: str,
    judge: JudgeFn,
    *,
    seed: int = 0,
) -> PairwiseResult:
    """Compare two answers in both orders."""
    # Both orderings use the *same* seed. Varying it as well would confound
    # position bias with sampling noise, and position bias is the thing being
    # measured here.
    forward = parse_verdict(
        judge(build_pairwise_prompt(question, reference, answer_a, answer_b).as_messages(), seed)
    )
    backward_raw = parse_verdict(
        judge(build_pairwise_prompt(question, reference, answer_b, answer_a).as_messages(), seed)
    )
    backward = _flip(backward_raw)
    return PairwiseResult(
        item_id=item_id,
        arm_a=arm_a,
        arm_b=arm_b,
        forward=forward,
        reversed_=backward,
        inconsistent=forward != backward,
    )


@dataclass
class PairwiseSummary:
    """Aggregated pairwise outcome for one arm pair."""

    arm_a: str
    arm_b: str
    a_wins: int = 0
    b_wins: int = 0
    ties: int = 0
    inconsistent: int = 0

    @property
    def n(self) -> int:
        return self.a_wins + self.b_wins + self.ties + self.inconsistent

    @property
    def position_bias_rate(self) -> float:
        """Share of pairs whose verdict flipped when the answers swapped.

        This is the judge's own error rate on this task. A high value means the
        pairwise numbers below should not be believed, and it is reported for
        exactly that reason.
        """
        return self.inconsistent / self.n if self.n else 0.0

    def as_json(self) -> dict[str, object]:
        return {
            "arm_a": self.arm_a,
            "arm_b": self.arm_b,
            "n": self.n,
            "a_wins": self.a_wins,
            "b_wins": self.b_wins,
            "ties": self.ties,
            "inconsistent": self.inconsistent,
            "position_bias_rate": round(self.position_bias_rate, 4),
            "rubric_version": RUBRIC_VERSION,
        }


def summarise_pairwise(results: Sequence[PairwiseResult]) -> PairwiseSummary:
    if not results:
        raise ValueError("no pairwise results to summarise")
    summary = PairwiseSummary(arm_a=results[0].arm_a, arm_b=results[0].arm_b)
    for result in results:
        if result.inconsistent:
            summary.inconsistent += 1
        elif result.forward == "TIE":
            summary.ties += 1
        elif result.forward == "A":
            summary.a_wins += 1
        else:
            summary.b_wins += 1
    return summary


@dataclass(frozen=True)
class Agreement:
    """Judge-versus-human agreement on a hand-labelled subset."""

    n: int
    exact_agreement: float
    kappa: float

    def verdict(self) -> str:
        """Landis & Koch bands, stated so the number is not left to the reader."""
        if self.kappa < 0.0:
            return "worse than chance"
        if self.kappa < 0.21:
            return "slight"
        if self.kappa < 0.41:
            return "fair"
        if self.kappa < 0.61:
            return "moderate"
        if self.kappa < 0.81:
            return "substantial"
        return "almost perfect"

    def as_json(self) -> dict[str, object]:
        return {
            "n": self.n,
            "exact_agreement": round(self.exact_agreement, 4),
            "cohens_kappa": round(self.kappa, 4),
            "interpretation": self.verdict(),
            "rubric_version": RUBRIC_VERSION,
        }


def cohens_kappa(human: Sequence[int], machine: Sequence[int]) -> Agreement:
    """Cohen's κ between human and judge labels.

    Raw agreement is not enough on a skewed scale: if 85% of answers score 2,
    a judge that always says 2 looks 85% accurate and has learned nothing. κ
    subtracts the agreement expected by chance, which is exactly that failure.
    """
    if len(human) != len(machine):
        raise ValueError(f"length mismatch: {len(human)} human vs {len(machine)} machine labels")
    if not human:
        raise ValueError("no labels given")

    n = len(human)
    observed = sum(1 for h, m in zip(human, machine, strict=True) if h == m) / n

    labels = set(human) | set(machine)
    expected = sum(
        (sum(1 for h in human if h == label) / n) * (sum(1 for m in machine if m == label) / n)
        for label in labels
    )
    # Perfect agreement on a single-valued sample: κ is undefined (0/0), and
    # reporting 1.0 there would overstate a sample that proves nothing.
    kappa = 0.0 if expected >= 1.0 else (observed - expected) / (1 - expected)
    return Agreement(n=n, exact_agreement=observed, kappa=kappa)


@dataclass
class ArmJudgement:
    """Every judged item for one arm."""

    arm: str
    results: list[PointwiseResult] = field(default_factory=list)

    @property
    def mean_score(self) -> float:
        return statistics.fmean(r.normalised for r in self.results) if self.results else 0.0

    @property
    def judge_sd(self) -> float:
        """Mean within-item standard deviation across judge seeds."""
        return statistics.fmean(r.sd for r in self.results) if self.results else 0.0

    @property
    def unparseable(self) -> int:
        return sum(r.unparseable for r in self.results)

    def as_json(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "n": len(self.results),
            "mean_score": round(self.mean_score, 4),
            "judge_sd": round(self.judge_sd, 4),
            "unparseable_replies": self.unparseable,
            "rubric_version": RUBRIC_VERSION,
        }


#: ``(item_id, question, reference, candidate)``.
PointwiseItem = tuple[str, str, str, str]
#: ``(item_id, question, reference, arm_a, answer_a, arm_b, answer_b)``.
PairwiseItem = tuple[str, str, str, str, str, str, str]


def score_many(
    items: Sequence[PointwiseItem],
    judge: JudgeFn,
    *,
    seeds: Sequence[int] = (0, 1, 2),
    workers: int = 1,
    progress: Callable[[], None] | None = None,
) -> list[PointwiseResult]:
    """:func:`score_pointwise` over many items, concurrently, in input order.

    A served judge batches concurrent requests. Issuing thousands of them one at
    a time leaves most of the GPU idle between calls — for this project's 5,400
    pointwise calls that is the difference between roughly two hours and fifteen
    minutes on a shared machine. Results come back in *input* order regardless
    of completion order, so item ids and scores can never be paired wrongly.
    """
    if workers <= 1:
        results: list[PointwiseResult] = []
        for item in items:
            results.append(score_pointwise(*item, judge, seeds=seeds))
            if progress is not None:
                progress()
        return results

    slots: list[PointwiseResult | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(score_pointwise, *item, judge, seeds=seeds): index
            for index, item in enumerate(items)
        }
        for future in as_completed(futures):
            slots[futures[future]] = future.result()  # re-raises a worker's failure
            if progress is not None:
                progress()
    return [slot for slot in slots if slot is not None]


def compare_many(
    items: Sequence[PairwiseItem],
    judge: JudgeFn,
    *,
    seed: int = 0,
    workers: int = 1,
    progress: Callable[[], None] | None = None,
) -> list[PairwiseResult]:
    """:func:`compare_pairwise` over many items, concurrently, in input order."""

    def run(item: PairwiseItem) -> PairwiseResult:
        return compare_pairwise(*item, judge, seed=seed)

    if workers <= 1:
        results: list[PairwiseResult] = []
        for item in items:
            results.append(run(item))
            if progress is not None:
                progress()
        return results

    slots: list[PairwiseResult | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run, item): index for index, item in enumerate(items)}
        for future in as_completed(futures):
            slots[futures[future]] = future.result()
            if progress is not None:
                progress()
    return [slot for slot in slots if slot is not None]
