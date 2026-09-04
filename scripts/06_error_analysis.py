#!/usr/bin/env python
"""Categorise every arm's failures and emit a review CSV.

    uv run python scripts/06_error_analysis.py

CPU only — it reads committed prediction JSONs, so the taxonomy is reproducible
by anyone who clones the repo without the model or the GPU.
"""

from __future__ import annotations

from fvr.config import bootstrap_env, load_config  # isort: skip

import json
import sys

from rich.console import Console
from rich.table import Table

from fvr.data.loaders import load_medmcqa
from fvr.eval.errors import (
    category_counts,
    collect_errors,
    sample_for_review,
    write_review_csv,
    write_summary,
)

console = Console()

#: Every arm is compared against the base arm, so "retrieval fixed this" and
#: "the fine-tune broke this" are computed against a common reference.
REFERENCE_ARM = "base"


def main() -> int:
    config = load_config()
    paths = bootstrap_env(config)

    runs_dir = paths.results / "runs"
    payloads = {
        json.loads(p.read_text(encoding="utf-8"))["arm"]: json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(runs_dir.glob("*.json"))
    }
    if not payloads:
        console.print("[red]No runs found.[/]")
        return 1

    split_ids = json.loads((paths.results / "split_ids.json").read_text(encoding="utf-8"))
    test_ids = set(split_ids["test"])
    pool, _ = load_medmcqa("validation")
    questions = {q.id: q for q in pool if q.id in test_ids}

    reference = payloads.get(REFERENCE_ARM)
    out_dir = paths.results / "error_analysis"
    summary: dict[str, dict[str, int]] = {}

    table = Table(title="Error taxonomy", show_lines=False)
    table.add_column("Arm")
    table.add_column("errors", justify="right")
    table.add_column("breakdown")

    for arm, payload in payloads.items():
        cases = collect_errors(
            payload, questions, reference_payload=None if arm == REFERENCE_ARM else reference
        )
        counts = category_counts(cases)
        summary[arm] = counts
        write_review_csv(sample_for_review(cases), out_dir / f"{arm}_review.csv")
        table.add_row(
            arm,
            str(len(cases)),
            ", ".join(f"{k}={v}" for k, v in counts.items()),
        )

    console.print(table)
    write_summary(summary, out_dir / "summary.json")
    console.print(f"\nReview CSVs and summary -> [cyan]{out_dir}[/]")
    console.print(
        "Each CSV carries blank `human_label` and `notes` columns: open it, "
        "read the sampled failures, and record what actually went wrong."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
