#!/usr/bin/env python
"""Build and freeze the evaluation splits, and verify them against git.

Two modes:

* default — build the split and write ``results/split_manifest.json``.
* ``--verify`` — rebuild and compare against the committed manifest, failing
  if anything moved. This is what makes "the test set is frozen" a checkable
  claim rather than a promise; it runs in CI once the manifest is committed.

    uv run python scripts/01_build_splits.py
    uv run python scripts/01_build_splits.py --verify
"""

from __future__ import annotations

from fvr.config import bootstrap_env, load_config  # isort: skip

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from fvr.data.clean import load_repair_lexicon
from fvr.data.loaders import load_medmcqa, resolve_dataset_revision
from fvr.data.schema import Split
from fvr.data.splits import (
    DEFAULT_TEST_SIZE,
    DEFAULT_VAL_SIZE,
    assert_disjoint,
    build_manifest,
    find_train_leakage,
    stratified_split,
    verify_manifest,
    write_manifest,
)

console = Console()
MANIFEST_NAME = "split_manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify", action="store_true", help="check against the committed manifest"
    )
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--val-size", type=int, default=DEFAULT_VAL_SIZE)
    parser.add_argument(
        "--check-leakage",
        action="store_true",
        help="also scan the 182k train split for content-identical eval rows (slow)",
    )
    args = parser.parse_args()

    config = load_config()
    paths = bootstrap_env(config)
    manifest_path = paths.results / MANIFEST_NAME

    console.print("[bold]Loading MedMCQA validation split[/] (the only labelled pool)…")
    pool, report = load_medmcqa("validation")
    console.print(f"  cleaning: {report.summary()}")

    assignment = stratified_split(
        pool,
        seed=config.seed,
        test_size=args.test_size,
        val_size=args.val_size,
    )
    assert_disjoint(assignment)

    manifest = build_manifest(
        assignment,
        dataset_revision=resolve_dataset_revision(),
        pool_size=len(pool),
        lexicon_size=len(load_repair_lexicon()),
    )

    table = Table(title="Frozen splits", show_lines=False)
    table.add_column("Split")
    table.add_column("n", justify="right")
    table.add_column("sha256")
    for split in (Split.TEST, Split.VAL, Split.RESERVE):
        table.add_row(
            split.value, f"{len(assignment.ids_for(split)):,}", assignment.digest(split)[:16] + "…"
        )
    console.print(table)

    if args.check_leakage:
        console.print("Scanning train split for content-identical evaluation rows…")
        train, _ = load_medmcqa("train")
        by_id = {q.id: q for q in pool}
        held_out = [by_id[i] for i in (*assignment.test_ids, *assignment.val_ids)]
        leaks = find_train_leakage(train, held_out)
        if leaks:
            console.print(f"  [yellow]{len(leaks):,} train rows duplicate a held-out row[/]")
            console.print("  These are excluded from training in Phase 5.")
        else:
            console.print("  [green]no content-level leakage[/]")
        manifest["train_leakage_rows"] = len(leaks)
        manifest["train_leaked_ids"] = sorted({t for t, _ in leaks})[:2000]

    if args.verify:
        if not manifest_path.is_file():
            console.print(f"[red]{manifest_path} does not exist — run without --verify first.[/]")
            return 1
        committed = json.loads(manifest_path.read_text(encoding="utf-8"))
        problems = verify_manifest(assignment, committed)
        if problems:
            console.print("[red]Split does not reproduce the committed manifest:[/]")
            for problem in problems:
                console.print(f"  • {problem}")
            return 1
        console.print("[green]Split reproduces the committed manifest exactly.[/]")
        return 0

    write_manifest(manifest, manifest_path)
    ids_path = paths.results / "split_ids.json"
    ids_path.write_text(
        json.dumps(
            {s.value: list(assignment.ids_for(s)) for s in (Split.TEST, Split.VAL)},
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    console.print(f"Wrote [cyan]{manifest_path}[/] and [cyan]{ids_path}[/]")
    console.print("[bold]Commit both now[/], before any model exists to contaminate them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
