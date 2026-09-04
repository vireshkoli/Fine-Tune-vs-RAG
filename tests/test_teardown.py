"""Teardown safety tests.

This is the code that deletes things on a shared machine, so the guards get
adversarial tests rather than happy-path ones: symlink escape, ``..`` traversal,
and passing a protected path directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fvr.config import PROJECT_ROOT, Paths
from fvr.ops.teardown import (
    DeletionPlan,
    RefuseToDeleteError,
    assert_deletable,
    directory_size,
    execute_teardown,
    missing_recoverable_artifacts,
    plan_teardown,
)


class TestAllowlist:
    def test_accepts_the_artifact_root(self) -> None:
        paths = Paths()
        assert assert_deletable(paths.artifacts, paths) == paths.artifacts.resolve()

    def test_accepts_a_child_of_the_artifact_root(self) -> None:
        paths = Paths()
        for target in paths.deletable():
            assert assert_deletable(target, paths)

    def test_rejects_a_path_outside_the_project(self, tmp_path: Path) -> None:
        with pytest.raises(RefuseToDeleteError, match="outside"):
            assert_deletable(tmp_path / "elsewhere")

    def test_rejects_the_results_directory(self) -> None:
        # results/ is committed and must survive teardown.
        with pytest.raises(RefuseToDeleteError):
            assert_deletable(Paths().results)

    def test_rejects_the_source_tree(self) -> None:
        with pytest.raises(RefuseToDeleteError):
            assert_deletable(PROJECT_ROOT / "src")


class TestDenylist:
    """The cases that would destroy someone else's work."""

    def test_rejects_the_shared_hf_cache(self) -> None:
        with pytest.raises(RefuseToDeleteError):
            assert_deletable(Path.home() / ".cache" / "huggingface")

    def test_rejects_home(self) -> None:
        with pytest.raises(RefuseToDeleteError):
            assert_deletable(Path.home())

    def test_rejects_root(self) -> None:
        with pytest.raises(RefuseToDeleteError):
            assert_deletable(Path("/"))

    def test_rejects_a_parent_of_the_shared_cache(self) -> None:
        # Deleting ~/.cache would take the shared HF cache with it.
        with pytest.raises(RefuseToDeleteError, match="would remove"):
            assert_deletable(Path.home() / ".cache")


class TestEscapes:
    def test_symlink_out_of_the_artifact_root_is_rejected(self, tmp_path: Path) -> None:
        """A symlink inside .artifacts pointing elsewhere must not widen the blast radius."""
        paths = Paths()
        outside = tmp_path / "someone_elses_data"
        outside.mkdir()
        link = paths.artifacts / "escape_test_link"
        try:
            link.symlink_to(outside)
            with pytest.raises(RefuseToDeleteError, match="outside"):
                assert_deletable(link, paths)
        finally:
            link.unlink(missing_ok=True)

    def test_dotdot_traversal_is_rejected(self) -> None:
        paths = Paths()
        with pytest.raises(RefuseToDeleteError):
            assert_deletable(paths.artifacts / ".." / ".." / "etc", paths)


class TestPlan:
    def test_plan_lists_only_artifact_paths(self) -> None:
        paths = Paths()
        plan = plan_teardown(paths)
        for target, _ in plan.targets:
            assert target.is_relative_to(paths.artifacts.resolve())

    def test_plan_lists_the_shared_cache_as_protected(self) -> None:
        """Only meaningful where a shared cache exists — a fresh runner has none.

        The environment-independent guarantee is that the cache is *undeletable*
        (TestDenylist); the plan only enumerates directories that are present.
        """
        shared = Path.home() / ".cache" / "huggingface"
        if not shared.exists():
            pytest.skip("no shared HF cache on this machine")
        protected = {str(p) for p, _ in plan_teardown().protected}
        assert any(".cache/huggingface" in p for p in protected)

    def test_render_shows_both_sections(self) -> None:
        text = plan_teardown().render()
        assert "Will DELETE" in text and "Will KEEP" in text

    def test_dry_run_deletes_nothing(self) -> None:
        plan = plan_teardown()
        before = [(p, directory_size(p)) for p, _ in plan.targets]
        outcome = execute_teardown(plan, dry_run=True)
        assert outcome["removed"] == []
        assert [(p, directory_size(p)) for p, _ in plan.targets] == before

    def test_empty_plan_totals_zero(self) -> None:
        assert DeletionPlan().total_gib == 0.0


class TestDirectorySize:
    def test_ignores_symlinks(self, tmp_path: Path) -> None:
        """HF caches link snapshots at blobs; following them double-counts."""
        real = tmp_path / "blob"
        real.write_bytes(b"x" * 1000)
        (tmp_path / "link").symlink_to(real)
        assert directory_size(tmp_path) == 1000

    def test_missing_directory_is_zero(self, tmp_path: Path) -> None:
        assert directory_size(tmp_path / "absent") == 0


class TestRecoverability:
    def test_committed_results_are_all_present(self) -> None:
        assert missing_recoverable_artifacts() == []

    def test_reports_what_is_missing(self, tmp_path: Path) -> None:
        missing = missing_recoverable_artifacts(results_dir=tmp_path)
        assert "split_manifest.json" in missing
        assert "runs/*.json" in missing
