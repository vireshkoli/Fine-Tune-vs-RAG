"""Cost-model tests, including the crossover that is the report's headline."""

from __future__ import annotations

import pytest
import yaml

from fvr.config import PROJECT_ROOT
from fvr.eval.cost import (
    ArmCost,
    CostCurve,
    FixedCosts,
    PerQueryCosts,
    RateCard,
    crossover_volume,
)

RATES = RateCard(
    gpu_name="test",
    gpu_usd_per_hour=3600.0,  # $1 per GPU-second, so the arithmetic is checkable by hand
    cpu_usd_per_hour=360.0,  # $0.10 per CPU-second
    source_url="https://example.test",
    retrieved="2026-08-07",
)


def an_arm(
    name: str,
    *,
    train_s: float = 0.0,
    index_s: float = 0.0,
    infer_s: float = 1.0,
    retrieval_s: float = 0.0,
) -> ArmCost:
    return ArmCost(
        arm=name,
        fixed=FixedCosts(train_gpu_seconds=train_s, index_build_gpu_seconds=index_s),
        per_query=PerQueryCosts(inference_gpu_seconds=infer_s, retrieval_cpu_seconds=retrieval_s),
        rates=RATES,
    )


class TestRateCard:
    def test_per_second_conversion(self) -> None:
        assert RATES.gpu_usd_per_second == pytest.approx(1.0)
        assert RATES.cpu_usd_per_second == pytest.approx(0.1)


class TestArmCost:
    def test_zero_fixed_cost_is_just_marginal(self) -> None:
        assert an_arm("base").usd_per_query(1000) == pytest.approx(1.0)

    def test_fixed_cost_amortises(self) -> None:
        # $1000 training over 1000 queries = $1/query on top of $1 inference.
        assert an_arm("ft", train_s=1000).usd_per_query(1000) == pytest.approx(2.0)

    def test_amortised_share_shrinks_with_volume(self) -> None:
        arm = an_arm("ft", train_s=1000)
        assert arm.usd_per_query(1_000_000) == pytest.approx(1.001)

    def test_marginal_ignores_fixed(self) -> None:
        assert an_arm("ft", train_s=99999).marginal_usd_per_query() == pytest.approx(1.0)

    def test_retrieval_cpu_time_is_counted(self) -> None:
        # RAG must not be flattered by ignoring its index-serving cost.
        assert an_arm("rag", retrieval_s=2.0).marginal_usd_per_query() == pytest.approx(1.2)

    def test_usd_per_1k_scales(self) -> None:
        assert an_arm("base").usd_per_1k(1000) == pytest.approx(1000.0)

    def test_zero_volume_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            an_arm("base").usd_per_query(0)


class TestCrossover:
    def test_finds_the_break_even_volume(self) -> None:
        """Fine-tuning: pay 1000 up front, save 0.5/query. Breaks even at 2000."""
        finetuned = an_arm("qlora", train_s=1000, infer_s=1.0)
        rag = an_arm("rag", infer_s=1.5)
        assert crossover_volume(finetuned, rag) == 2000

    def test_crossover_is_symmetric(self) -> None:
        a = an_arm("qlora", train_s=1000, infer_s=1.0)
        b = an_arm("rag", infer_s=1.5)
        assert crossover_volume(a, b) == crossover_volume(b, a)

    def test_the_arms_actually_swap_around_the_crossover(self) -> None:
        finetuned = an_arm("qlora", train_s=1000, infer_s=1.0)
        rag = an_arm("rag", infer_s=1.5)
        n = crossover_volume(finetuned, rag)
        assert n is not None
        assert rag.usd_per_query(n // 2) < finetuned.usd_per_query(n // 2)
        assert finetuned.usd_per_query(n * 2) < rag.usd_per_query(n * 2)

    def test_dominated_arm_never_crosses(self) -> None:
        # Cheaper fixed AND cheaper marginal — the choice does not depend on scale.
        assert crossover_volume(an_arm("cheap", infer_s=0.5), an_arm("dear", train_s=100)) is None

    def test_equal_marginals_never_cross(self) -> None:
        assert crossover_volume(an_arm("a", train_s=100), an_arm("b", train_s=200)) is None


class TestCostCurve:
    def test_builds_a_series_per_arm(self) -> None:
        curve = CostCurve.build(
            {"base": an_arm("base"), "qlora": an_arm("qlora", train_s=1000)},
            [100, 1000, 10000],
        )
        assert set(curve.series) == {"base", "qlora"}
        assert len(curve.series["base"]) == 3

    def test_cost_decreases_monotonically_with_volume(self) -> None:
        curve = CostCurve.build({"qlora": an_arm("qlora", train_s=1000)}, [100, 1000, 10000])
        assert curve.series["qlora"] == sorted(curve.series["qlora"], reverse=True)

    def test_winner_changes_with_volume(self) -> None:
        """The whole point of the chart: the answer depends on scale."""
        curve = CostCurve.build(
            {
                "qlora": an_arm("qlora", train_s=1000, infer_s=1.0),
                "rag": an_arm("rag", infer_s=1.5),
            },
            [100, 1_000_000],
        )
        assert curve.cheapest_at(100) == "rag"
        assert curve.cheapest_at(1_000_000) == "qlora"

    def test_unknown_volume_raises(self) -> None:
        curve = CostCurve.build({"base": an_arm("base")}, [100])
        with pytest.raises(KeyError):
            curve.cheapest_at(999)

    def test_rejects_non_positive_volumes(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            CostCurve.build({"base": an_arm("base")}, [0])


class TestCommittedRateCard:
    def test_config_parses_into_a_rate_card(self) -> None:
        raw = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "eval" / "cost.yaml").read_text(encoding="utf-8")
        )
        card = RateCard(**raw["rate_card"])
        assert card.gpu_usd_per_hour > 0

    def test_rate_card_is_citable(self) -> None:
        # A cost claim without a source and a date is not defensible.
        raw = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "eval" / "cost.yaml").read_text(encoding="utf-8")
        )
        card = RateCard(**raw["rate_card"])
        assert card.source_url.startswith("http")
        assert card.retrieved

    def test_sweep_spans_several_orders_of_magnitude(self) -> None:
        raw = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "eval" / "cost.yaml").read_text(encoding="utf-8")
        )
        volumes = raw["amortization"]["sweep_volumes"]
        assert min(volumes) <= 100 and max(volumes) >= 1_000_000
