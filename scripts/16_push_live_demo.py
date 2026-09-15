#!/usr/bin/env python
"""Publish the live demo: the parity index as a dataset, then the ZeroGPU Space.

    uv run python scripts/16_push_live_demo.py            # dry run: prints both manifests
    uv run python scripts/16_push_live_demo.py --execute  # creates/updates both repos

Dry run by default, like ``08_push_to_hub.py``: publishing is outward-facing.
The index goes up first because the Space downloads it at startup; each repo
is written as one atomic commit so a failed upload leaves nothing half-done.
"""

from __future__ import annotations

from fvr.config import bootstrap_env, load_config  # isort: skip

import argparse
import sys

from rich.console import Console

from fvr.ops.hub import (
    LIVE_SPACE_HARDWARE,
    HubTargets,
    UploadPlan,
    plan_index_upload,
    plan_live_space_upload,
)

console = Console()


def push(plan: UploadPlan, *, space_hardware: str | None = None) -> str:
    from huggingface_hub import CommitOperationAdd, HfApi, SpaceHardware

    api = HfApi()
    kwargs = {"space_sdk": "gradio", "space_hardware": space_hardware} if space_hardware else {}
    api.create_repo(plan.repo_id, repo_type=plan.repo_type, exist_ok=True, **kwargs)
    operations = [
        CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local))
        for local, remote in plan.entries
    ]
    info = api.create_commit(
        plan.repo_id,
        repo_type=plan.repo_type,
        operations=operations,
        commit_message="Publish from scripts/16_push_live_demo.py",
    )
    if space_hardware:
        # create_repo only sets hardware on creation; an existing Space keeps
        # whatever it had, so request it explicitly.
        api.request_space_hardware(plan.repo_id, SpaceHardware(space_hardware))
    return str(info.commit_url)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="actually upload")
    parser.add_argument("--skip-index", action="store_true", help="Space only")
    parser.add_argument("--index", default=".artifacts/indices/parity")
    parser.add_argument("--app-dir", default="space_live")
    args = parser.parse_args()

    config = load_config()
    paths = bootstrap_env(config)
    from huggingface_hub import HfApi

    targets = HubTargets(namespace=HfApi().whoami()["name"])

    plans: list[tuple[UploadPlan, str | None]] = []
    if not args.skip_index:
        plans.append(
            (
                plan_index_upload(
                    paths.root / args.index, paths.root / "docs" / "index_dataset_card.md", targets
                ),
                None,
            )
        )
    plans.append((plan_live_space_upload(paths.root / args.app_dir, targets), LIVE_SPACE_HARDWARE))

    for plan, hardware in plans:
        console.print(plan.render())
        if hardware:
            console.print(f"  hardware: {hardware}")
        console.print()

    if not args.execute:
        console.print("[yellow]Dry run — nothing uploaded. Re-run with --execute to push.[/]")
        return 0

    for plan, hardware in plans:
        console.print(f"Pushing {plan.repo_type} [cyan]{plan.repo_id}[/]…")
        url = push(plan, space_hardware=hardware)
        console.print(f"  [green]done[/] {url}")
    console.print(
        f"\nSpace: https://huggingface.co/spaces/{targets.live_space_repo} — "
        "the first build takes ~10 minutes (dependencies, then the 16 GB base model)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
