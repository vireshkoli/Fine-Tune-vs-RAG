#!/usr/bin/env python
"""Prove every artifact survives the lab machine being wiped.

    uv run python scripts/09_verify_recoverable.py

Gates teardown. If this fails, deleting the artifact tree would strand a result
that exists nowhere else — which is exactly the failure a reviewer would catch
after the fact, when nothing can be recovered.
"""

from __future__ import annotations

from fvr.config import bootstrap_env, load_config  # isort: skip

import json
import subprocess
import sys

from rich.console import Console
from rich.table import Table

from fvr.ops.hub import ADAPTER_FILES, SPACE_ENTRY_POINT, HubTargets
from fvr.ops.teardown import missing_recoverable_artifacts

console = Console()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, timeout=30, check=False
    ).stdout.strip()


def _hub_checks() -> list[tuple[str, bool, str]]:
    """Ask the Hub whether the artifacts that only exist there are really there.

    The adapter cost five GPU-hours and lives in ``.artifacts/``, which is the
    directory teardown deletes. A local copy proves nothing, so this queries the
    Hub rather than the filesystem — and a network failure is reported as a
    failure, never assumed to be a pass.
    """
    from fvr.config import Secrets

    token = Secrets().hf_token
    if not token:
        return [("adapter pushed to Hub", False, "no HF_TOKEN in .env; cannot verify")]

    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        targets = HubTargets(namespace=str(api.whoami()["name"]))
    except Exception as exc:  # any failure to reach the Hub means 'unverified'
        return [("adapter pushed to Hub", False, f"could not reach the Hub: {exc}")]

    checks: list[tuple[str, bool, str]] = []

    try:
        files = set(api.list_repo_files(targets.adapter_repo, repo_type="model"))
    except Exception as exc:
        checks.append(("adapter pushed to Hub", False, f"{targets.adapter_repo}: {exc}"))
    else:
        # Weights alone are not enough: an adapter published without its card
        # is a medical model on the internet with no safety statement.
        required = {"adapter_config.json", "adapter_model.safetensors", "README.md"}
        missing = sorted(required - files)
        checks.append(
            (
                "adapter pushed to Hub",
                not missing,
                f"{targets.adapter_repo}: {len(files & set(ADAPTER_FILES)) + 1} file(s)"
                if not missing
                else f"missing {missing}",
            )
        )

    try:
        space_files = set(api.list_repo_files(targets.space_repo, repo_type="space"))
    except Exception as exc:
        checks.append(("demo Space published", False, f"{targets.space_repo}: {exc}"))
    else:
        needed = {
            SPACE_ENTRY_POINT,
            "logic.js",
            "ui.js",
            "style.css",
            "precomputed/responses.json",
        }
        missing_space = sorted(needed - space_files)
        checks.append(
            (
                "demo Space published",
                not missing_space,
                targets.space_repo if not missing_space else f"missing {missing_space}",
            )
        )

    return checks


def main() -> int:
    config = load_config()
    paths = bootstrap_env(config)
    checks: list[tuple[str, bool, str]] = []

    dirty = _git("status", "--porcelain")
    checks.append(
        (
            "working tree clean",
            not dirty,
            "clean" if not dirty else f"{len(dirty.splitlines())} uncommitted",
        )
    )

    local, remote = _git("rev-parse", "HEAD"), _git("rev-parse", "@{u}")
    checks.append(
        (
            "pushed to origin",
            bool(local) and local == remote,
            f"{local[:12]} == origin"
            if local == remote
            else f"local {local[:12]} != {remote[:12] or 'no upstream'}",
        )
    )

    missing = missing_recoverable_artifacts()
    checks.append(
        ("results committed", not missing, "all present" if not missing else f"missing {missing}")
    )

    runs = sorted((paths.results / "runs").glob("*.json"))
    hashes = {json.loads(p.read_text(encoding="utf-8"))["split_sha256"] for p in runs}
    checks.append(
        (
            "runs share one split",
            len(hashes) == 1,
            f"{len(runs)} runs, split {next(iter(hashes), '?')[:12]}…"
            if len(hashes) == 1
            else f"{len(hashes)} different splits",
        )
    )

    # Model weights are re-downloadable only if the revision is pinned.
    unpinned = [
        p.name
        for p in (paths.configs / "model").glob("*.yaml")
        if "revision:" not in p.read_text(encoding="utf-8")
    ]
    checks.append(
        (
            "model revisions pinned",
            not unpinned,
            "pinned" if not unpinned else f"unpinned: {unpinned}",
        )
    )

    checks.extend(_hub_checks())

    table = Table(title="Recoverability", show_lines=False)
    table.add_column("Check", style="bold")
    table.add_column("")
    table.add_column("Detail", overflow="fold")
    for name, ok, detail in checks:
        table.add_row(name, "[green]PASS" if ok else "[red]FAIL", detail)
    console.print(table)

    failed = [name for name, ok, _ in checks if not ok]
    if failed:
        console.print(f"\n[red]{len(failed)} check(s) failed.[/] Teardown would strand work.")
        return 1
    console.print("\n[green]Everything is reconstructible from GitHub and the Hub.[/]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
