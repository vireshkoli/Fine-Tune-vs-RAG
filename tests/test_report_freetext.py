"""Paired statistics over judged free-text runs.

The sign-flip test is the only inferential statistic in the report that is not
a library call, so it gets the checks a library would: correct direction, a
calibrated null, pairing by item id rather than list position.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from fvr.report.freetext import (
    JudgedArm,
    arms_table,
    deltas_table,
    load_judged,
    load_pairwise,
    paired_delta,
    score_interval,
)


def an_arm(name: str, scores: dict[str, float], **overrides: object) -> JudgedArm:
    defaults: dict[str, object] = {"judge_sd": 0.01, "unparseable": 0}
    return JudgedArm(name=name, scores=scores, **{**defaults, **overrides})  # type: ignore[arg-type]


class TestPairedDelta:
    def test_direction_and_significance_of_a_clear_effect(self) -> None:
        rng = random.Random(0)
        ids = [f"q{i}" for i in range(200)]
        base = {i: rng.choice([0.0, 0.5, 1.0]) for i in ids}
        # B is A shifted up on a third of the items: a real, one-sided gap.
        better = {i: min(1.0, base[i] + (0.5 if rng.random() < 0.33 else 0.0)) for i in ids}
        delta = paired_delta(an_arm("b", better), an_arm("a", base), n_permutations=2000)
        assert delta.delta > 0
        assert delta.p_value < 0.001
        assert delta.n_a_higher > 0 and delta.n_b_higher == 0
        assert delta.low > 0 < delta.high

    def test_null_is_not_significant_and_p_is_bounded_below(self) -> None:
        rng = random.Random(1)
        ids = [f"q{i}" for i in range(100)]
        a = {i: rng.choice([0.0, 0.5, 1.0]) for i in ids}
        b = dict(a)
        delta = paired_delta(an_arm("a", a), an_arm("b", b), n_permutations=500)
        assert delta.delta == 0.0
        assert delta.p_value == 1.0
        # And the +1 correction: nothing can be reported as p = 0.
        assert delta.p_value >= 1 / 501

    def test_is_symmetric_up_to_sign(self) -> None:
        rng = random.Random(2)
        ids = [f"q{i}" for i in range(80)]
        a = {i: rng.choice([0.0, 0.5, 1.0]) for i in ids}
        b = {i: rng.choice([0.0, 0.5, 1.0]) for i in ids}
        ab = paired_delta(an_arm("a", a), an_arm("b", b), n_permutations=1000)
        ba = paired_delta(an_arm("b", b), an_arm("a", a), n_permutations=1000)
        assert ab.delta == pytest.approx(-ba.delta)
        assert ab.p_value == pytest.approx(ba.p_value)
        assert (ab.n_a_higher, ab.n_b_higher) == (ba.n_b_higher, ba.n_a_higher)

    def test_pairs_by_item_id_not_position(self) -> None:
        """Two runs listing items in different orders must still pair correctly."""
        a = {"q1": 1.0, "q2": 0.0, "q3": 0.5}
        b_reordered = {"q3": 0.5, "q1": 1.0, "q2": 0.0}
        delta = paired_delta(an_arm("a", a), an_arm("b", b_reordered), n_permutations=100)
        assert delta.delta == 0.0
        assert delta.n_a_higher == delta.n_b_higher == 0

    def test_drops_items_judged_in_only_one_arm(self) -> None:
        a = {"q1": 1.0, "q2": 1.0, "only_a": 0.0}
        b = {"q1": 0.0, "q2": 0.0, "only_b": 1.0}
        delta = paired_delta(an_arm("a", a), an_arm("b", b), n_permutations=100)
        assert delta.n == 2
        assert delta.delta == 1.0

    def test_refuses_disjoint_item_sets(self) -> None:
        with pytest.raises(ValueError, match="share no judged items"):
            paired_delta(an_arm("a", {"x": 1.0}), an_arm("b", {"y": 1.0}))

    def test_is_deterministic_for_a_seed(self) -> None:
        rng = random.Random(3)
        ids = [f"q{i}" for i in range(50)]
        a = {i: rng.random() for i in ids}
        b = {i: rng.random() for i in ids}
        first = paired_delta(an_arm("a", a), an_arm("b", b), seed=7)
        second = paired_delta(an_arm("a", a), an_arm("b", b), seed=7)
        assert first == second


class TestScoreInterval:
    def test_interval_brackets_the_mean(self) -> None:
        rng = random.Random(0)
        arm = an_arm("a", {f"q{i}": rng.choice([0.0, 0.5, 1.0]) for i in range(300)})
        interval = score_interval(arm, n_resamples=2000)
        assert interval.low <= interval.point <= interval.high
        assert interval.point == pytest.approx(arm.mean_score)

    def test_constant_scores_give_a_degenerate_interval(self) -> None:
        arm = an_arm("a", {f"q{i}": 0.5 for i in range(20)})
        interval = score_interval(arm, n_resamples=100)
        assert (interval.low, interval.point, interval.high) == (0.5, 0.5, 0.5)

    def test_refuses_an_empty_arm(self) -> None:
        with pytest.raises(ValueError, match="no judged items"):
            score_interval(an_arm("a", {}))


class TestLoading:
    def test_reads_judged_and_pairwise_files(self, tmp_path: Path) -> None:
        judged = tmp_path / "miriad-base.json"
        judged.write_text(
            json.dumps(
                {
                    "arm": "miriad-base",
                    "judge_sd": 0.01,
                    "unparseable_replies": 2,
                    "max_new_tokens": 256,
                    "capped_answers": 3,
                    "items": [
                        {"item_id": "q1", "scores": [2, 2, 1], "mean": 1.6667},
                        {"item_id": "q2", "scores": [0, 0, 0], "mean": 0.0},
                        {"item_id": "q3", "scores": [], "mean": 0.0},  # all unparseable
                    ],
                }
            ),
            encoding="utf-8",
        )
        arm = load_judged(judged)
        assert arm.name == "miriad-base"
        assert arm.scores == {"q1": pytest.approx(0.83335), "q2": 0.0}
        assert arm.n == 2  # the unscored item is not a zero
        assert arm.capped_share == pytest.approx(1.5)  # 3 of 2 scored: degenerate but honest
        assert arm.unparseable == 2

        pair = tmp_path / "pairwise_a__b.json"
        pair.write_text(
            json.dumps(
                {
                    "arm_a": "a",
                    "arm_b": "b",
                    "a_wins": 10,
                    "b_wins": 4,
                    "ties": 3,
                    "inconsistent": 5,
                    "position_bias_rate": 0.2273,
                }
            ),
            encoding="utf-8",
        )
        record = load_pairwise(pair)
        assert (record.a_wins, record.b_wins, record.decided) == (10, 4, 14)

    def test_strict_scoring_counts_only_a_majority_of_two(self, tmp_path: Path) -> None:
        """Partial credit is where the judge and the human disagreed; strict drops it."""
        judged = tmp_path / "base_seed42.json"
        judged.write_text(
            json.dumps(
                {
                    "arm": "base",
                    "judge_sd": 0.0,
                    "unparseable_replies": 0,
                    "items": [
                        {"item_id": "full", "scores": [2, 2, 2], "majority": 2, "mean": 2.0},
                        {"item_id": "mostly", "scores": [2, 2, 1], "majority": 2, "mean": 1.6667},
                        {"item_id": "partial", "scores": [1, 1, 1], "majority": 1, "mean": 1.0},
                        {"item_id": "wrong", "scores": [0, 0, 0], "majority": 0, "mean": 0.0},
                    ],
                }
            ),
            encoding="utf-8",
        )
        graded = load_judged(judged)
        strict = load_judged(judged, strict=True)
        assert strict.scores == {"full": 1.0, "mostly": 1.0, "partial": 0.0, "wrong": 0.0}
        assert graded.scores["partial"] == 0.5 and graded.scores["mostly"] == pytest.approx(0.83335)
        assert strict.mean_score < graded.mean_score

    def test_old_judged_files_without_a_bound_load(self, tmp_path: Path) -> None:
        """The six MedMCQA files predate max_new_tokens / capped_answers."""
        judged = tmp_path / "base_seed42.json"
        judged.write_text(
            json.dumps(
                {
                    "arm": "base",
                    "judge_sd": 0.0,
                    "unparseable_replies": 0,
                    "items": [{"item_id": "q1", "scores": [1], "mean": 1.0}],
                }
            ),
            encoding="utf-8",
        )
        arm = load_judged(judged)
        assert arm.max_new_tokens is None and arm.capped_share is None


class TestTables:
    def test_arms_table_ranks_best_first_and_bolds_only_the_top(self) -> None:
        arms = [
            an_arm("low", {"q": 0.2}),
            an_arm("high", {"q": 0.9}, max_new_tokens=256, capped_answers=0),
        ]
        intervals = {a.name: score_interval(a, n_resamples=10) for a in arms}
        text = arms_table(arms, intervals, extra={"high": {"MCQ accuracy": "70.0%"}})
        lines = text.splitlines()
        assert lines[2].startswith("| `high` | **0.900**")
        assert "0% @256" in lines[2] and "70.0%" in lines[2]
        assert lines[3].startswith("| `low` | 0.200 |")
        assert "—" in lines[3]  # no bound recorded, no MCQ column value

    def test_deltas_table_shows_pairwise_where_present(self) -> None:
        a = an_arm("a", {"q1": 1.0, "q2": 1.0, "q3": 0.0})
        b = an_arm("b", {"q1": 0.0, "q2": 0.5, "q3": 0.0})
        delta = paired_delta(a, b, n_permutations=100)
        text = deltas_table([delta])
        assert "| `a` - `b` |" in text
        assert text.count("| — |") == 1  # no pairwise record for this pair
