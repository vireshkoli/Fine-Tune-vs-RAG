"""Assembles the cost model from what the pipeline actually measured.

Separate from :mod:`fvr.eval.cost`, which defines the *arithmetic*. This module
supplies the *inputs*, and it does so only from files the pipeline wrote —
``build_stats.json`` for the index, the training summary for the fine-tune. An
arm that was never trained therefore contributes no imaginary training cost,
which is enforced by where the numbers come from rather than by remembering to
zero them.

Shared by ``scripts/07_make_report.py`` and ``scripts/08_push_to_hub.py`` so the
demo Space and the README cannot disagree about what an arm costs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from fvr.config import Paths
from fvr.eval.cost import ArmCost, FixedCosts, PerQueryCosts, RateCard
from fvr.inference.arms import ARMS_BY_NAME
from fvr.report.aggregate import Aggregate

TRAINING_SUMMARY = "qlora-r16.json"


def load_rate_card(configs: Path) -> tuple[RateCard, dict[str, Any]]:
    """The committed rate card and amortisation settings."""
    raw = yaml.safe_load((Path(configs) / "eval" / "cost.yaml").read_text(encoding="utf-8"))
    return RateCard(**raw["rate_card"]), dict(raw["amortization"])


def index_gpu_seconds(indices: Path, corpus: str) -> float:
    stats = Path(indices) / corpus / "build_stats.json"
    if corpus == "none" or not stats.is_file():
        return 0.0
    return float(json.loads(stats.read_text(encoding="utf-8"))["embed_gpu_seconds"])


def train_gpu_seconds(results: Path, arm_name: str) -> float:
    arm = ARMS_BY_NAME.get(arm_name)
    if arm is None or not arm.uses_adapter:
        return 0.0
    summary = Path(results) / "training" / TRAINING_SUMMARY
    if not summary.is_file():
        return 0.0
    return float(json.loads(summary.read_text(encoding="utf-8"))["train_gpu_seconds"])


def build_costs(aggregate: Aggregate, rates: RateCard, paths: Paths) -> dict[str, ArmCost]:
    """One :class:`ArmCost` per evaluated arm, from measured GPU-seconds."""
    costs: dict[str, ArmCost] = {}
    for name in aggregate.arms:
        arm = ARMS_BY_NAME.get(name)
        run = aggregate.for_arm(name)[0]
        corpus = arm.corpus if arm is not None else "none"
        costs[name] = ArmCost(
            arm=name,
            fixed=FixedCosts(
                train_gpu_seconds=train_gpu_seconds(Path(str(paths.results)), name),
                index_build_gpu_seconds=index_gpu_seconds(Path(str(paths.indices)), corpus),
            ),
            per_query=PerQueryCosts(inference_gpu_seconds=run.p50_ms / 1000),
            rates=rates,
        )
    return costs


def cost_inputs(costs: dict[str, ArmCost], rates: RateCard) -> dict[str, Any]:
    """Flatten the model into the two numbers a cost curve needs per arm.

    ``usd_per_query(N) == fixed_usd / N + marginal_usd``. Shipping the inputs
    rather than a sampled curve lets the demo compute any volume exactly,
    and keeps the formula visible instead of hiding it behind interpolation.
    """
    return {
        "rate_card": {
            "gpu_name": rates.gpu_name,
            "gpu_usd_per_hour": rates.gpu_usd_per_hour,
            "cpu_usd_per_hour": rates.cpu_usd_per_hour,
            "source_url": rates.source_url,
            "retrieved": rates.retrieved,
        },
        "arms": {
            name: {
                "fixed_usd": cost.fixed.usd(rates),
                "marginal_usd_per_query": cost.marginal_usd_per_query(),
                "train_gpu_seconds": cost.fixed.train_gpu_seconds,
                "index_build_gpu_seconds": cost.fixed.index_build_gpu_seconds,
            }
            for name, cost in costs.items()
        },
    }
