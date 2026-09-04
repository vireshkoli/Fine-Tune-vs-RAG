"""Surgical teardown for shared infrastructure.

This project runs on a university GPU that must be vacated on completion, and
the account already holds unrelated prior work in ``~/.cache/huggingface``. A
naive ``rm -rf`` of a cache directory would destroy someone else's models.

So deletion is guarded three independent ways, because one guard is one bug away
from being no guard:

1. **Path allowlist** — the only deletable root is ``$PROJECT_ROOT/.artifacts``.
   Every target is resolved and must be a strict descendant, so symlinks and
   ``..`` cannot escape.
2. **Explicit denylist** — the shared cache, ``$HOME``, and anything outside the
   project raise :class:`RefuseToDeleteError` *even if passed directly*.
3. **Pre/post assertion** — the shared cache is measured before and after, and a
   changed size is an error.

The separation is structural rather than procedural: ``fvr/config.py`` pins
``HF_HOME`` inside ``.artifacts`` at import time, so this project's downloads
never land in the shared cache to begin with.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from fvr.config import PROJECT_ROOT, Paths


class RefuseToDeleteError(Exception):
    """Raised when a deletion target fails a safety guard."""


def _shared_caches() -> tuple[Path, ...]:
    home = Path.home()
    return (
        home / ".cache" / "huggingface",
        home / ".cache" / "torch",
        home,
        Path("/"),
    )


def assert_deletable(target: Path, paths: Paths | None = None) -> Path:
    """Raise unless ``target`` is safely inside the artifact root."""
    paths = paths or Paths()
    resolved = target.resolve()
    artifacts = paths.artifacts.resolve()

    for forbidden in _shared_caches():
        resolved_forbidden = forbidden.expanduser().resolve()
        if resolved == resolved_forbidden:
            raise RefuseToDeleteError(f"{resolved} is a protected path and is never deletable")
        if resolved_forbidden.is_relative_to(resolved):
            raise RefuseToDeleteError(
                f"deleting {resolved} would remove {resolved_forbidden}, which is not ours"
            )

    if resolved == artifacts:
        return resolved
    if not resolved.is_relative_to(artifacts):
        raise RefuseToDeleteError(
            f"{resolved} is outside {artifacts}; only the artifact root may be deleted"
        )
    return resolved


def directory_size(path: Path) -> int:
    """Bytes on disk, ignoring symlinks so HF's snapshot links are not double-counted."""
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and not f.is_symlink())


@dataclass
class DeletionPlan:
    """What teardown would remove, computed before anything is touched."""

    targets: list[tuple[Path, int]] = field(default_factory=list)
    protected: list[tuple[Path, int]] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(size for _, size in self.targets)

    @property
    def total_gib(self) -> float:
        return self.total_bytes / 2**30

    def render(self) -> str:
        lines = ["Will DELETE:"]
        for path, size in self.targets:
            lines.append(f"  {size / 2**30:8.2f} GiB  {path}")
        lines.append(f"  {'-' * 8}")
        lines.append(f"  {self.total_gib:8.2f} GiB  total")
        lines.append("")
        lines.append("Will KEEP (not ours):")
        for path, size in self.protected:
            lines.append(f"  {size / 2**30:8.2f} GiB  {path}")
        return "\n".join(lines)


def plan_teardown(paths: Paths | None = None) -> DeletionPlan:
    """Compute the deletion manifest without deleting anything."""
    paths = paths or Paths()
    plan = DeletionPlan()
    for target in paths.deletable():
        if target.exists():
            plan.targets.append((assert_deletable(target, paths), directory_size(target)))
    for shared in _shared_caches()[:2]:
        expanded = shared.expanduser()
        if expanded.exists():
            plan.protected.append((expanded, directory_size(expanded)))
    return plan


def execute_teardown(
    plan: DeletionPlan, paths: Paths | None = None, *, dry_run: bool = True
) -> dict[str, object]:
    """Delete the planned targets, verifying the shared caches are untouched."""
    paths = paths or Paths()
    before = {path: directory_size(path) for path, _ in plan.protected}

    removed: list[str] = []
    if not dry_run:
        for target, _ in plan.targets:
            assert_deletable(target, paths)  # checked again immediately before deletion
            shutil.rmtree(target, ignore_errors=False)
            removed.append(str(target))

    after = {path: directory_size(path) for path in before}
    changed = {str(p): (before[p], after[p]) for p in before if before[p] != after[p]}
    if changed:
        raise RefuseToDeleteError(
            f"a protected cache changed size during teardown: {changed}. "
            "This should be impossible; investigate before running again."
        )

    return {
        "dry_run": dry_run,
        "removed": removed,
        "reclaimed_gib": plan.total_gib if not dry_run else 0.0,
        "protected_unchanged": {str(p): before[p] for p in before},
    }


def missing_recoverable_artifacts(
    *, results_dir: Path | None = None, required: Iterable[str] = ()
) -> list[str]:
    """Files that must exist in git before the artifact tree may be deleted.

    Everything here is committed, so the benchmark survives the machine being
    wiped. If any is missing, teardown would strand the result.
    """
    results = results_dir or (PROJECT_ROOT / "results")
    wanted: Sequence[str] = tuple(required) or (
        "split_manifest.json",
        "split_ids.json",
        "tables.md",
        "figures/arms.png",
        "figures/cost-crossover.png",
        "error_analysis/summary.json",
    )
    missing = [name for name in wanted if not (results / name).is_file()]
    if not list((results / "runs").glob("*.json")):
        missing.append("runs/*.json")
    return missing
