#!/usr/bin/env python
"""Preflight check: hardware, disk, cache isolation and Hugging Face token scope.

Run this before anything downloads. It fails loudly on the three things that
would otherwise waste hours: too little disk, a read-only token discovered at
push time, and caches leaking into the shared ``~/.cache/huggingface`` on the
lab machine.

    uv run python scripts/00_setup_check.py
"""

from __future__ import annotations

# Must precede any huggingface_hub / torch import so the caches land in .artifacts.
from fvr.config import PROJECT_ROOT, bootstrap_env, load_config  # isort: skip

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from fvr.config import Secrets

console = Console()

SHARED_CACHE = Path.home() / ".cache" / "huggingface"
GIB = 1024**3


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def check_gpus() -> list[Check]:
    """Report visible GPUs via nvidia-smi, which needs no torch install."""
    query = "name,memory.total,memory.used"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return [Check("GPU", False, f"nvidia-smi unavailable: {exc}")]

    lines = [line for line in out.splitlines() if line.strip()]
    if not lines:
        return [Check("GPU", False, "nvidia-smi returned no devices")]
    return [Check(f"GPU {i}", True, line.strip()) for i, line in enumerate(lines)]


def check_disk(min_free_gib: float) -> Check:
    free_gib = shutil.disk_usage(PROJECT_ROOT).free / GIB
    ok = free_gib >= min_free_gib
    verdict = "" if ok else f" — need {min_free_gib:.0f} GiB, refusing to start"
    return Check("Disk", ok, f"{free_gib:.1f} GiB free{verdict}")


def check_cache_isolation() -> list[Check]:
    """The artifact tree must be inside the project and distinct from the shared cache."""
    import os

    checks: list[Check] = []
    hub = Path(os.environ["HF_HOME"]).resolve()

    inside = hub.is_relative_to(PROJECT_ROOT)
    checks.append(
        Check(
            "HF_HOME inside project",
            inside,
            str(hub) if inside else f"{hub} ESCAPES {PROJECT_ROOT}",
        )
    )

    collides = SHARED_CACHE.resolve() in hub.parents or hub == SHARED_CACHE.resolve()
    if SHARED_CACHE.exists():
        # Skip symlinks: the HF layout points snapshots/ at blobs/, so following
        # them counts every weight file twice and roughly doubles the reported size.
        size_gib = (
            sum(
                f.stat().st_size
                for f in SHARED_CACHE.rglob("*")
                if f.is_file() and not f.is_symlink()
            )
            / GIB
        )
        checks.append(
            Check(
                "Shared cache untouched",
                not collides,
                f"{SHARED_CACHE} holds {size_gib:.1f} GiB of unrelated prior work — "
                f"{'SEPARATE, safe' if not collides else 'COLLIDES, teardown would destroy it'}",
            )
        )
    return checks


def check_hf_token() -> Check:
    """Confirm a token exists and report whether it can write.

    Phase 8 pushes an adapter, model card and Space. Discovering a read-only
    token then costs a day; discovering it now costs a minute.
    """
    from huggingface_hub import HfApi, get_token
    from huggingface_hub.errors import HfHubHTTPError

    # Prefer .env. Fall back to the HF CLI's store so the check reports the real
    # scope rather than claiming nothing is configured — note that pinning
    # HF_HOME also relocates the token file, so the default location is tried
    # explicitly. Read-only: the shared cache is never written to.
    default_token_file = SHARED_CACHE / "token"
    token = Secrets().hf_token or get_token()
    if not token and default_token_file.is_file():
        token = default_token_file.read_text(encoding="utf-8").strip() or None
    if not token:
        return Check("HF token", False, "not set — copy .env.example to .env and add HF_TOKEN")

    try:
        info = HfApi().whoami(token=token)
    except (HfHubHTTPError, OSError) as exc:
        return Check("HF token", False, f"rejected by the Hub: {exc}")

    role = str(info.get("auth", {}).get("accessToken", {}).get("role", "unknown"))
    user = str(info.get("name", "?"))
    if role == "write":
        return Check("HF token", True, f"{user}, write scope")
    return Check(
        "HF token",
        False,
        f"{user}, {role!r} scope — Phase 8 needs WRITE; regenerate at "
        "https://huggingface.co/settings/tokens",
    )


def main() -> int:
    config = load_config()
    paths = bootstrap_env(config)

    checks: list[Check] = [
        *check_gpus(),
        check_disk(config.min_free_disk_gib),
        *check_cache_isolation(),
        check_hf_token(),
    ]

    table = Table(title="Setup check", show_lines=False)
    table.add_column("Check", style="bold")
    table.add_column("")
    table.add_column("Detail", overflow="fold")
    for check in checks:
        table.add_row(check.name, "[green]PASS" if check.ok else "[red]FAIL", check.detail)
    console.print(table)
    console.print(f"Artifact root: [cyan]{paths.artifacts}[/] (the only path teardown may delete)")

    failed = [c for c in checks if not c.ok]
    if failed:
        console.print(f"\n[red]{len(failed)} check(s) failed.[/] Fix before downloading anything.")
        return 1
    console.print("\n[green]All checks passed.[/]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
