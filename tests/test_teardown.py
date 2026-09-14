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

    def test_dry_run_deletes_nothing(self, tmp_path: Path) -> None:
        # Under tmp rather than the live tree: a training job appends to
        # .artifacts/logs while tests run, so live sizes are not stable.
        paths = tmp_paths(tmp_path)
        (paths.hub / "blob").parent.mkdir(parents=True)
        (paths.hub / "blob").write_bytes(b"x" * 64)
        (paths.artifacts / "logs").mkdir()
        (paths.artifacts / "logs" / "run.log").write_text("line\n", encoding="utf-8")
        plan = plan_teardown(paths)
        before = [(p, directory_size(p)) for p, _ in plan.targets]
        outcome = execute_teardown(plan, paths, dry_run=True)
        assert outcome["removed"] == []
        assert [(p, directory_size(p)) for p, _ in plan.targets] == before
        assert (paths.hub / "blob").is_file()

    def test_dry_run_on_the_live_tree_leaves_every_target_in_place(self) -> None:
        plan = plan_teardown()
        execute_teardown(plan, dry_run=True)
        assert all(target.exists() for target, _ in plan.targets)

    def test_empty_plan_totals_zero(self) -> None:
        assert DeletionPlan().total_gib == 0.0


def tmp_paths(tmp_path: Path) -> Paths:
    """A Paths whose artifact root is under tmp, so deletion can be exercised."""
    root = tmp_path / "project"
    artifacts = root / ".artifacts"
    return Paths(
        root=root,
        artifacts=artifacts,
        hub=artifacts / "hub",
        datasets=artifacts / "datasets",
        indices=artifacts / "indices",
        checkpoints=artifacts / "checkpoints",
        results=root / "results",
        configs=root / "configs",
    )


class TestSweep:
    """Everything under the root goes, not just the four declared directories.

    The judge's virtualenv, uv and vLLM caches, logs, locks and superseded runs
    all live under .artifacts without a ``Paths`` field. The first dry run
    would have left 11 GiB of them on the shared machine.
    """

    def test_plan_includes_undeclared_children(self, tmp_path: Path) -> None:
        paths = tmp_paths(tmp_path)
        (paths.hub / "models--x").mkdir(parents=True)
        (paths.artifacts / "judge-venv" / "bin").mkdir(parents=True)
        (paths.artifacts / "judge-venv" / "bin" / "vllm").write_bytes(b"x" * 100)
        (paths.artifacts / "stray.log").write_text("hello", encoding="utf-8")

        plan = plan_teardown(paths)
        targets = {target for target, _ in plan.targets}
        assert paths.hub.resolve() in targets
        assert (paths.artifacts / "judge-venv").resolve() in targets
        assert (paths.artifacts / "stray.log").resolve() in targets
        assert set(plan.undeclared) == {
            (paths.artifacts / "judge-venv").resolve(),
            (paths.artifacts / "stray.log").resolve(),
        }
        sizes = dict(plan.targets)
        assert sizes[(paths.artifacts / "judge-venv").resolve()] == 100
        assert sizes[(paths.artifacts / "stray.log").resolve()] == 5

    def test_render_marks_undeclared_entries(self, tmp_path: Path) -> None:
        paths = tmp_paths(tmp_path)
        paths.hub.mkdir(parents=True)
        (paths.artifacts / "uv-cache").mkdir()
        text = plan_teardown(paths).render()
        assert "uv-cache  (undeclared)" in text
        assert "hub  (undeclared)" not in text

    def test_execute_removes_everything_and_the_empty_root(self, tmp_path: Path) -> None:
        paths = tmp_paths(tmp_path)
        (paths.hub / "models--x").mkdir(parents=True)
        (paths.artifacts / "judge-venv").mkdir()
        (paths.artifacts / "stray.log").write_text("x", encoding="utf-8")
        paths.results.mkdir(parents=True)
        (paths.results / "keep.json").write_text("{}", encoding="utf-8")

        outcome = execute_teardown(plan_teardown(paths), paths, dry_run=False)

        assert not paths.artifacts.exists()
        assert (paths.results / "keep.json").is_file()
        assert str(paths.artifacts.resolve()) in outcome["removed"]  # type: ignore[operator]

    def test_symlink_child_pointing_outside_aborts_the_plan(self, tmp_path: Path) -> None:
        """A link out of the tree must stop planning, not be followed into rmtree."""
        paths = tmp_paths(tmp_path)
        paths.artifacts.mkdir(parents=True)
        elsewhere = tmp_path / "someone_elses"
        elsewhere.mkdir()
        (paths.artifacts / "link").symlink_to(elsewhere)
        with pytest.raises(RefuseToDeleteError, match="outside"):
            plan_teardown(paths)
        assert elsewhere.is_dir()


class TestDirectorySize:
    def test_ignores_symlinks(self, tmp_path: Path) -> None:
        """HF caches link snapshots at blobs; following them double-counts."""
        real = tmp_path / "blob"
        real.write_bytes(b"x" * 1000)
        (tmp_path / "link").symlink_to(real)
        assert directory_size(tmp_path) == 1000

    def test_missing_directory_is_zero(self, tmp_path: Path) -> None:
        assert directory_size(tmp_path / "absent") == 0

    def test_counts_a_hardlinked_file_once_across_targets(self, tmp_path: Path) -> None:
        """uv hard-links the judge venv to its cache; the bytes exist once."""
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir()
        b.mkdir()
        (a / "blob").write_bytes(b"x" * 1000)
        (b / "blob").hardlink_to(a / "blob")
        seen: set[tuple[int, int]] = set()
        assert directory_size(a, seen) + directory_size(b, seen) == 1000
        # Without a shared set each target is measured on its own.
        assert directory_size(a) + directory_size(b) == 2000


class TestRecoverability:
    def test_committed_results_are_all_present(self) -> None:
        assert missing_recoverable_artifacts() == []

    def test_reports_what_is_missing(self, tmp_path: Path) -> None:
        missing = missing_recoverable_artifacts(results_dir=tmp_path)
        assert "split_manifest.json" in missing
        assert "runs/*.json" in missing
