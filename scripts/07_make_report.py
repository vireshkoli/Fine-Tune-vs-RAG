#!/usr/bin/env python
"""Regenerate every table and figure from committed run JSONs.

    uv run python scripts/07_make_report.py

Nothing here invents a number. Each table and chart is derived from files under
``results/runs/``, so a figure in the README that no run supports is impossible
by construction rather than by review.
"""

from __future__ import annotations

from fvr.config import bootstrap_env, load_config  # isort: skip

import itertools
import sys

from rich.console import Console

from fvr.data.loaders import load_medmcqa
from fvr.eval.cost import CostCurve, crossover_volume
from fvr.report.aggregate import Aggregate, load_gold
from fvr.report.charts import Theme, arm_comparison_chart, cost_crossover_chart
from fvr.report.costs import build_costs, load_rate_card
from fvr.report.tables import (
    comparison_table,
    cost_table,
    per_subject_table,
    provenance_note,
    results_table,
)

console = Console()


def main() -> int:
    project = load_config()
    paths = bootstrap_env(project)

    aggregate = Aggregate.load(paths.results / "runs")
    if not aggregate.runs:
        console.print("[red]No runs found under results/runs/.[/]")
        return 1
    aggregate.assert_same_split()
    console.print(f"Loaded {len(aggregate.runs)} run(s) across {len(aggregate.arms)} arm(s)")

    rates, amortization = load_rate_card(paths.configs)
    volumes = list(amortization["sweep_volumes"])

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
    per_1k = [1000 * costs[a].usd_per_query(amortization["default_volume"]) for a in arms]

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
