"""Markdown tables generated from committed JSON.

Generated rather than hand-written so the README can never drift from the
results. ``make report`` regenerates them; if the output differs from what is
committed, either the results changed or someone edited a number by hand, and
both are worth catching.
"""

from __future__ import annotations

from collections.abc import Sequence

from fvr.eval.metrics import Comparison
from fvr.inference.arms import ARMS_BY_NAME
from fvr.report.aggregate import Aggregate

#: Marks a run whose timing was taken on a shared GPU, so a reader is never left
#: wondering why one arm's p95 looks odd.
CONTENDED = " ⚠"


def _fmt_pct(value: float) -> str:
    return f"{100 * value:.1f}"


def results_table(aggregate: Aggregate, *, headline_only: bool = False) -> str:
    """The main results table."""
    rows = [
        "| Arm | Accuracy | 95% CI | p50 | p95 | Prompt tokens | Retrieval hit | Grounded |",
        "| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm_name in aggregate.arms:
        arm = ARMS_BY_NAME.get(arm_name)
        if headline_only and arm is not None and not arm.headline:
            continue
        runs = aggregate.for_arm(arm_name)
        run = runs[0]
        mean, sd = aggregate.seed_summary(arm_name)
        low, high = run.ci
        accuracy = f"{_fmt_pct(mean)}%"
        if len(runs) > 1:
            accuracy += f" ± {_fmt_pct(sd)}"
        grounding = run.groundedness or {}
        hit = grounding.get("retrieval_hit_rate")
        grounded = grounding.get("grounded_rate")
        flag = "" if run.exclusive_device else CONTENDED
        rows.append(
            f"| `{arm_name}`{flag} | **{accuracy}** | [{_fmt_pct(low)}, {_fmt_pct(high)}] | "
            f"{run.p50_ms:.0f} ms | {run.p95_ms:.0f} ms | {run.mean_prompt_tokens:.0f} | "
            f"{'—' if hit is None else f'{hit:.3f}'} | "
            f"{'—' if grounded is None else f'{grounded:.3f}'} |"
        )
    return "\n".join(rows)


def comparison_table(
    comparisons: Sequence[tuple[str, str, Comparison]], *, alpha: float = 0.05
) -> str:
    """Paired McNemar results.

    Discordant counts are shown alongside p because they are what distinguish a
    true null (balanced disagreement) from an underpowered one, and a small but
    consistently one-sided gap from noise.
    """
    rows = [
        "| Comparison | Δ accuracy | Discordant (A/B) | p | Verdict |",
        "| --- | ---: | :---: | ---: | --- |",
    ]
    for arm_a, arm_b, result in comparisons:
        verdict = "**significant**" if result.is_significant(alpha) else "not significant"
        p_value = "<0.0001" if result.p_value < 0.0001 else f"{result.p_value:.4f}"
        rows.append(
            f"| `{arm_a}` vs `{arm_b}` | {result.delta:+.3f} | "
            f"{result.n_a_only}/{result.n_b_only} | {p_value} | {verdict} |"
        )
    return "\n".join(rows)


def per_subject_table(aggregate: Aggregate, *, min_items: int = 30) -> str:
    """Accuracy by subject, for arms that have runs.

    Subjects below ``min_items`` are omitted: a 4-item bucket produces
    percentages that look precise and mean nothing.
    """
    arms = aggregate.arms
    by_arm = {a: (aggregate.for_arm(a)[0].payload.get("per_subject") or {}) for a in arms}
    subjects = sorted(
        {s for stats in by_arm.values() for s, v in stats.items() if v.get("n", 0) >= min_items}
    )

    header = "| Subject | n | " + " | ".join(f"`{a}`" for a in arms) + " |"
    divider = "| --- | ---: | " + " | ".join("---:" for _ in arms) + " |"
    rows = [header, divider]
    for subject in subjects:
        counts = [by_arm[a].get(subject, {}).get("n", 0) for a in arms]
        n = max(counts)
        cells = []
        for arm in arms:
            entry = by_arm[arm].get(subject)
            cells.append("—" if entry is None else f"{_fmt_pct(entry['accuracy'])}%")
        rows.append(f"| {subject} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def cost_table(curve_volumes: Sequence[int], series: dict[str, list[float]]) -> str:
    """Cost per 1,000 queries across the swept volumes."""
    header = "| Query volume | " + " | ".join(f"`{a}`" for a in series) + " |"
    divider = "| ---: | " + " | ".join("---:" for _ in series) + " |"
    rows = [header, divider]
    for i, volume in enumerate(curve_volumes):
        cells = [f"${1000 * series[arm][i]:.3f}" for arm in series]
        rows.append(f"| {volume:,} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def provenance_note(aggregate: Aggregate) -> str:
    """One line tying the tables to the exact data and code that produced them."""
    hashes = aggregate.split_hashes()
    digest = next(iter(hashes)) if len(hashes) == 1 else "MIXED"
    run = aggregate.runs[0] if aggregate.runs else None
    env = (run.payload.get("environment") or {}) if run else {}
    n_items = run.payload.get("n_items") if run else "?"
    return (
        f"Generated from `results/runs/` by `make report`. "
        f"Test split `{digest[:16]}…` (n={n_items}), "
        f"git `{str(env.get('git_sha', '?'))[:12]}`, "
        f"{env.get('gpu', 'unknown GPU')}, torch {env.get('torch', '?')}."
    )
