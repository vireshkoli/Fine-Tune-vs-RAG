#!/usr/bin/env python
"""Regenerate every table and figure from committed run JSONs.

    uv run python scripts/07_make_report.py

Nothing here invents a number. Each table and chart is derived from files under
``results/runs/``, so a figure in the README that no run supports is impossible
by construction rather than by review.
"""

from __future__ import annotations

from fvr.config import Paths, bootstrap_env, load_config  # isort: skip

import itertools
import json
import sys
from pathlib import Path

import yaml
from rich.console import Console

from fvr.data.loaders import load_medmcqa
from fvr.eval.cost import ArmCost, CostCurve, FixedCosts, PerQueryCosts, RateCard, crossover_volume
from fvr.inference.arms import ARMS_BY_NAME
from fvr.report.aggregate import Aggregate, load_gold
from fvr.report.charts import Theme, arm_comparison_chart, cost_crossover_chart
from fvr.report.tables import (
    comparison_table,
    cost_table,
    per_subject_table,
    provenance_note,
    results_table,
)

console = Console()


def build_costs(aggregate: Aggregate, rates: RateCard, paths: Paths) -> dict[str, ArmCost]:
    """Assemble the cost model from measured GPU-seconds.

    Inference time is the measured p50. Index build and training come from the
    artefacts those steps wrote, so an arm that was never trained contributes no
    imaginary training cost.
    """
    indices = Path(str(paths.indices))
    results = Path(str(paths.results))

    def index_seconds(corpus: str) -> float:
        stats = indices / corpus / "build_stats.json"
        if not stats.is_file():
            return 0.0
        return float(json.loads(stats.read_text(encoding="utf-8"))["embed_gpu_seconds"])

    def train_seconds(arm_name: str) -> float:
        arm = ARMS_BY_NAME.get(arm_name)
        if arm is None or not arm.uses_adapter:
            return 0.0
        summary = results / "training" / "qlora-r16.json"
        if not summary.is_file():
            return 0.0
        return float(json.loads(summary.read_text(encoding="utf-8"))["train_gpu_seconds"])

    costs: dict[str, ArmCost] = {}
    for name in aggregate.arms:
        arm = ARMS_BY_NAME.get(name)
        run = aggregate.for_arm(name)[0]
        corpus = arm.corpus if arm is not None else "none"
        costs[name] = ArmCost(
            arm=name,
            fixed=FixedCosts(
                train_gpu_seconds=train_seconds(name),
                index_build_gpu_seconds=index_seconds(corpus) if corpus != "none" else 0.0,
            ),
            per_query=PerQueryCosts(inference_gpu_seconds=run.p50_ms / 1000),
            rates=rates,
        )
    return costs


def main() -> int:
    project = load_config()
    paths = bootstrap_env(project)

    aggregate = Aggregate.load(paths.results / "runs")
    if not aggregate.runs:
        console.print("[red]No runs found under results/runs/.[/]")
        return 1
    aggregate.assert_same_split()
    console.print(f"Loaded {len(aggregate.runs)} run(s) across {len(aggregate.arms)} arm(s)")

    cost_config = yaml.safe_load((paths.configs / "eval" / "cost.yaml").read_text(encoding="utf-8"))
    rates = RateCard(**cost_config["rate_card"])
    volumes = list(cost_config["amortization"]["sweep_volumes"])

    costs = build_costs(aggregate, rates, paths)
    curve = CostCurve.build(costs, volumes)

    crossovers: dict[tuple[str, str], int] = {}
    for a, b in itertools.combinations(costs, 2):
        volume = crossover_volume(costs[a], costs[b])
        if volume is not None and min(volumes) <= volume <= max(volumes):
            crossovers[(a, b)] = volume

    pool, _ = load_medmcqa("validation")
    gold = load_gold(paths.results / "split_ids.json", pool)
    comparisons = [
        (a, b, aggregate.compare(a, b, gold)) for a, b in itertools.combinations(aggregate.arms, 2)
    ]

    figures = paths.results / "figures"
    arms = aggregate.arms
    accuracy = [aggregate.for_arm(a)[0].accuracy for a in arms]
    ci = [aggregate.for_arm(a)[0].ci for a in arms]
    p95 = [aggregate.for_arm(a)[0].p95_ms for a in arms]
    per_1k = [
        1000 * costs[a].usd_per_query(cost_config["amortization"]["default_volume"]) for a in arms
    ]

    for theme in (Theme.light(), Theme.dark()):
        suffix = "" if theme.name == "light" else "-dark"
        cost_crossover_chart(
            volumes,
            curve.series,
            figures / f"cost-crossover{suffix}.png",
            theme=theme,
            crossovers=crossovers,
        )
        arm_comparison_chart(
            arms,
            accuracy,
            ci,
            p95,
            per_1k,
            figures / f"arms{suffix}.png",
            theme=theme,
        )
    console.print(f"Wrote figures to [cyan]{figures}[/]")

    sections = {
        "results": results_table(aggregate),
        "headline": results_table(aggregate, headline_only=True),
        "comparisons": comparison_table(comparisons),
        "per_subject": per_subject_table(aggregate),
        "cost": cost_table(volumes, curve.series),
        "provenance": provenance_note(aggregate),
    }
    tables_path = paths.results / "tables.md"
    tables_path.write_text(
        "\n\n".join(f"<!-- {name} -->\n{body}" for name, body in sections.items()) + "\n",
        encoding="utf-8",
    )
    console.print(f"Wrote tables to [cyan]{tables_path}[/]")

    console.print()
    console.print(sections["results"])
    console.print()
    console.print(sections["comparisons"])
    if crossovers:
        console.print()
        for (a, b), volume in crossovers.items():
            console.print(f"  crossover: [bold]{a}[/] overtakes [bold]{b}[/] at {volume:,} queries")
    else:
        console.print(
            "\n  [yellow]No crossover inside the swept range — "
            "with no trained arm yet, no arm carries a fixed cost.[/]"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
