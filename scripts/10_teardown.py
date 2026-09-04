#!/usr/bin/env python
"""Reclaim this project's disk on the shared lab machine.

    uv run python scripts/10_teardown.py            # dry run, prints the manifest
    uv run python scripts/10_teardown.py --execute  # actually deletes

Deletes only ``$PROJECT_ROOT/.artifacts``. The shared ``~/.cache/huggingface``
holds unrelated prior work and is protected three ways — allowlist, denylist,
and a before/after size assertion. See ``fvr/ops/teardown.py``.
"""

from __future__ import annotations

from fvr.config import bootstrap_env, load_config  # isort: skip

import argparse
import json
import sys

from rich.console import Console

from fvr.ops.teardown import execute_teardown, missing_recoverable_artifacts, plan_teardown

console = Console()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="delete for real")
    parser.add_argument(
        "--skip-recoverability-check",
        action="store_true",
        help="not recommended: teardown normally refuses unless results are committed",
    )
    args = parser.parse_args()

    config = load_config()
    paths = bootstrap_env(config)

    missing = missing_recoverable_artifacts()
    if missing and not args.skip_recoverability_check:
        console.print(
            f"[red]Refusing to tear down: {missing} are not committed.[/]\n"
            "Deleting now would strand results that exist nowhere else. "
            "Run scripts/09_verify_recoverable.py."
        )
        return 1

    plan = plan_teardown(paths)
    console.print(plan.render())

    if not args.execute:
        console.print(
            f"\n[yellow]Dry run.[/] Would reclaim {plan.total_gib:.2f} GiB. "
            "Re-run with --execute to delete."
        )
        return 0

    console.print(f"\nDeleting {plan.total_gib:.2f} GiB…")
    outcome = execute_teardown(plan, paths, dry_run=False)
    console.print(json.dumps(outcome, indent=2))
    console.print(
        "\n[green]Done.[/] Protected caches verified unchanged. "
        "The source tree and results/ are untouched — copy them off, then remove manually."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
