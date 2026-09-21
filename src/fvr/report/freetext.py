"""Statistics over judged free-text runs.

The MCQ arms get paired McNemar tests and bootstrap intervals from
``fvr.eval.metrics``. Judge scores are not binary, so the free-text arms get
the continuous equivalents: a percentile bootstrap over items for each arm's
mean, and for each pair of arms a **sign-flip permutation test** on the
per-item score differences. Sign-flip rather than an unpaired test because
both arms answered the same 300 questions, and item difficulty is most of the
variance; rather than a t-test because a 0/1/2 rubric averaged over three
judge seeds is not remotely normal.

Everything here reads the committed ``results/freetext/judged/*.json`` files,
so the tables in REPORT.md are regenerable and cannot drift from the data.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fvr.eval.judge import MAX_POINTWISE
from fvr.eval.metrics import DEFAULT_BOOTSTRAP, DEFAULT_CONFIDENCE, Interval

#: Enough that a p-value near 0.05 is stable to two figures; the +1 correction
#: below means the smallest reportable p is 1/(N+1).
DEFAULT_PERMUTATIONS = 20_000


@dataclass(frozen=True)
class JudgedArm:
    """One arm's judged free-text run, keyed by item."""

    name: str
    #: item_id -> mean judge score across seeds, normalised to 0-1.
    scores: dict[str, float]
    judge_sd: float
    unparseable: int
    #: The generation bound and how many answers hit it, carried from the run
    #: file so the table can say whether a low score is a fragment score.
    max_new_tokens: int | None = None
    capped_answers: int | None = None

    @property
    def n(self) -> int:
        return len(self.scores)

    @property
    def mean_score(self) -> float:
        return statistics.fmean(self.scores.values()) if self.scores else 0.0

    @property
    def capped_share(self) -> float | None:
        if self.capped_answers is None or not self.scores:
            return None
        return self.capped_answers / self.n


def load_judged(path: Path, *, strict: bool = False) -> JudgedArm:
    """Load one arm's judgement.

    ``strict`` collapses the 0/1/2 rubric to fully-correct-or-not: an item
    scores 1 when the judge's majority grade is 2, else 0. The human check
    (κ = 0.51 on the three-way scale, 0.76 on this binary one) showed the
    judge hands out partial credit more freely than a person does, so every
    paired comparison is reported under both scorings.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if strict:
        scores = {
            str(item["item_id"]): 1.0 if int(item["majority"]) == MAX_POINTWISE else 0.0
            for item in payload["items"]
            if item.get("scores")
        }
    else:
        scores = {
            str(item["item_id"]): float(item["mean"]) / MAX_POINTWISE
            for item in payload["items"]
            if item.get("scores")
        }
    return JudgedArm(
        name=str(payload["arm"]),
        scores=scores,
        judge_sd=float(payload["judge_sd"]),
        unparseable=int(payload["unparseable_replies"]),
        max_new_tokens=payload.get("max_new_tokens"),
        capped_answers=payload.get("capped_answers"),
    )


def score_interval(
    arm: JudgedArm,
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap over items of the arm's mean score."""
    values = np.fromiter(arm.scores.values(), dtype=float)
    if values.size == 0:
        raise ValueError(f"{arm.name}: no judged items")
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


@dataclass(frozen=True)
class PairedScoreDelta:
    """Arm A minus arm B on the items both answered."""

    arm_a: str
    arm_b: str
    n: int
    delta: float
    low: float
    high: float
    p_value: float
    #: Items where A scored higher / B scored higher; the rest tied.
    n_a_higher: int
    n_b_higher: int

    def is_significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha

    def as_json(self) -> dict[str, Any]:
        return {
            "arm_a": self.arm_a,
            "arm_b": self.arm_b,
            "n": self.n,
            "delta": round(self.delta, 4),
            "ci95": [round(self.low, 4), round(self.high, 4)],
            "p_value": self.p_value,
            "n_a_higher": self.n_a_higher,
            "n_b_higher": self.n_b_higher,
            "test": "sign-flip permutation on per-item score differences",
        }


def paired_delta(
    arm_a: JudgedArm,
    arm_b: JudgedArm,
    *,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    n_resamples: int = DEFAULT_BOOTSTRAP,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> PairedScoreDelta:
    """Sign-flip permutation test and bootstrap CI on per-item differences.

    Items are matched by id, not position, so two runs that list them in a
    different order are still paired correctly — and an item judged in only
    one arm is dropped from the pair rather than mismatched.
    """
    shared = sorted(set(arm_a.scores) & set(arm_b.scores))
    if not shared:
        raise ValueError(f"{arm_a.name} and {arm_b.name} share no judged items")
    diff = np.array([arm_a.scores[i] - arm_b.scores[i] for i in shared])
    observed = float(diff.mean())

    rng = np.random.default_rng(seed)
    # Under the null the sign of every item's difference is a coin flip; the
    # observed mean is extreme if few of the flipped means are as large.
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, diff.size))
    null = (signs * np.abs(diff)).mean(axis=1)
    p_value = (int(np.sum(np.abs(null) >= abs(observed))) + 1) / (n_permutations + 1)

    draws = rng.integers(0, diff.size, size=(n_resamples, diff.size))
    means = diff[draws].mean(axis=1)
    tail = (1 - confidence) / 2
    return PairedScoreDelta(
        arm_a=arm_a.name,
        arm_b=arm_b.name,
        n=diff.size,
        delta=observed,
        low=float(np.quantile(means, tail)),
        high=float(np.quantile(means, 1 - tail)),
        p_value=p_value,
        n_a_higher=int(np.sum(diff > 0)),
        n_b_higher=int(np.sum(diff < 0)),
    )


@dataclass(frozen=True)
class PairwiseRecord:
    """A both-orders pairwise judgement, as written by ``13_judge_freetext.py``."""

    arm_a: str
    arm_b: str
    a_wins: int
    b_wins: int
    ties: int
    inconsistent: int
    position_bias_rate: float

    @property
    def decided(self) -> int:
        return self.a_wins + self.b_wins

    def as_json(self) -> dict[str, Any]:
        return {
            "arm_a": self.arm_a,
            "arm_b": self.arm_b,
            "a_wins": self.a_wins,
            "b_wins": self.b_wins,
            "ties": self.ties,
            "inconsistent": self.inconsistent,
            "position_bias_rate": round(self.position_bias_rate, 4),
        }


def load_pairwise(path: Path) -> PairwiseRecord:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return PairwiseRecord(
        arm_a=str(payload["arm_a"]),
        arm_b=str(payload["arm_b"]),
        a_wins=int(payload["a_wins"]),
        b_wins=int(payload["b_wins"]),
        ties=int(payload["ties"]),
        inconsistent=int(payload["inconsistent"]),
        position_bias_rate=float(payload["position_bias_rate"]),
    )


def _fmt_p(p_value: float) -> str:
    if p_value < 0.0001:
        return f"{p_value:.0e}".replace("e-0", "e-")
    return f"{p_value:.4f}" if p_value < 0.001 else f"{p_value:.3f}"


def arms_table(
    arms: Sequence[JudgedArm],
    intervals: dict[str, Interval],
    *,
    extra: dict[str, dict[str, str]] | None = None,
) -> str:
    """Per-arm judge scores, best first.

    ``extra`` adds columns keyed by arm name — the MCQ accuracy beside the
    judge score, for the MedMCQA set. The capped column is there because a
    score is only a score of the model when the answers were not cut off.
    """
    extra = extra or {}
    extra_names = sorted({name for cols in extra.values() for name in cols})
    head = "| Arm | Judge score (0-1) | 95% CI | Judge SD | Capped |"
    head += "".join(f" {name} |" for name in extra_names)
    rows = [head, "| --- | ---: | :---: | ---: | ---: |" + " ---: |" * len(extra_names)]
    for rank, arm in enumerate(sorted(arms, key=lambda a: a.mean_score, reverse=True)):
        interval = intervals[arm.name]
        share = arm.capped_share
        capped = "—" if share is None else f"{share:.0%} @{arm.max_new_tokens}"
        bold = "**" if rank == 0 else ""
        row = (
            f"| `{arm.name}` | {bold}{arm.mean_score:.3f}{bold} | [{interval.low:.2f}, "
            f"{interval.high:.2f}] | {arm.judge_sd:.3f} | {capped} |"
        )
        row += "".join(f" {extra.get(arm.name, {}).get(name, '—')} |" for name in extra_names)
        rows.append(row)
    return "\n".join(rows)


def deltas_table(
    deltas: Sequence[PairedScoreDelta],
    pairwise: dict[tuple[str, str], PairwiseRecord] | None = None,
    *,
    extra: dict[tuple[str, str], str] | None = None,
    extra_name: str = "",
) -> str:
    """Paired deltas, with the both-orders pairwise count where one was run.

    The pairwise column is read A-B: verdicts for A, verdicts for B, after
    dropping the pairs whose verdict flipped with presentation order.
    """
    pairwise = pairwise or {}
    extra = extra or {}
    head = "| Paired comparison | Free-text Δ (pts) | 95% CI | p (permutation) |"
    head += f" {extra_name} |" if extra_name else ""
    head += " Pairwise, both orders |"
    sep = "| --- | ---: | :---: | ---: |" + (" ---: |" if extra_name else "") + " :---: |"
    rows = [head, sep]
    for d in deltas:
        record = pairwise.get((d.arm_a, d.arm_b))
        both = "—"
        if record is not None:
            both = f"{record.a_wins}-{record.b_wins} ({record.position_bias_rate:.0%} flipped)"
        bold = "**" if d.is_significant() else ""
        row = (
            f"| `{d.arm_a}` - `{d.arm_b}` | {bold}{100 * d.delta:+.1f}{bold} | "
            f"[{100 * d.low:+.1f}, {100 * d.high:+.1f}] | {_fmt_p(d.p_value)} |"
        )
        if extra_name:
            row += f" {extra.get((d.arm_a, d.arm_b), '—')} |"
        rows.append(row + f" {both} |")
    return "\n".join(rows)
