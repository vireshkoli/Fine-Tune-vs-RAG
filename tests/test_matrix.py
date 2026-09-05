"""The experiment matrix.

These are cheap tests guarding an expensive mistake. The grid is ~60 GPU-hours
on a shared machine, so an error that only surfaces at hour 40 — an eval reading
an adapter that was never trained, two jobs writing the same file, a dependency
declared after the job that needs it — costs real time that cannot be recovered.
Every one of those is checkable in milliseconds without a GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fvr.config import PROJECT_ROOT, Paths
from fvr.ops.matrix import (
    DependencyError,
    Job,
    JobResult,
    MatrixState,
    build_matrix,
    by_group,
    pending,
    total_gpu_hours,
    validate,
)


@pytest.fixture(scope="module")
def matrix() -> list[Job]:
    return build_matrix(Paths())


def runs(job: Job, script: str) -> bool:
    """Whether a job invokes a given script.

    A helper rather than ``"03_eval_arm.py" in job.command``, which is the
    obvious spelling and is always False — the command holds the full path
    ``scripts/03_eval_arm.py``. Three tests here passed vacuously that way
    before one of them happened to assert non-emptiness and caught it.
    """
    return any(argument.endswith(script) for argument in job.command)


class TestStructure:
    def test_every_job_has_a_distinct_name(self, matrix: list[Job]) -> None:
        names = [job.name for job in matrix]
        assert len(names) == len(set(names))

    def test_no_two_jobs_write_the_same_artifact(self, matrix: list[Job]) -> None:
        """Two jobs sharing an output means the second silently skips forever."""
        produced = [job.produces for job in matrix]
        duplicates = sorted({str(p) for p in produced if produced.count(p) > 1})
        assert not duplicates, f"jobs collide on: {duplicates}"

    def test_dependencies_are_declared_before_use(self, matrix: list[Job]) -> None:
        validate(matrix)

    def test_rejects_a_forward_dependency(self) -> None:
        jobs = [
            Job("second", "g", ("true",), Path("/tmp/b"), 0.1, depends_on=("first",)),
            Job("first", "g", ("true",), Path("/tmp/a"), 0.1),
        ]
        with pytest.raises(DependencyError, match="not defined before"):
            validate(jobs)

    def test_rejects_duplicate_names(self) -> None:
        jobs = [
            Job("same", "g", ("true",), Path("/tmp/a"), 0.1),
            Job("same", "g", ("true",), Path("/tmp/b"), 0.1),
        ]
        with pytest.raises(DependencyError, match="duplicate"):
            validate(jobs)


class TestCommands:
    def test_every_command_names_a_script_that_exists(self, matrix: list[Job]) -> None:
        for job in matrix:
            script = job.command[1]
            assert (PROJECT_ROOT / script).is_file(), f"{job.name}: missing {script}"

    def test_every_referenced_config_exists(self, matrix: list[Job]) -> None:
        """A typo'd config path fails 40 GPU-hours in, or worse, silently."""
        for job in matrix:
            for argument in job.command:
                if argument.startswith("configs/"):
                    assert (PROJECT_ROOT / argument).is_file(), f"{job.name}: missing {argument}"

    def test_adapter_evals_depend_on_the_training_that_makes_them(self, matrix: list[Job]) -> None:
        """The failure this prevents: evaluating a checkpoint that never existed."""
        trained = {job.name for job in matrix if runs(job, "04_train.py")}
        assert trained, "no training jobs found; the predicate is wrong"
        for job in matrix:
            if "--adapter" not in job.command:
                continue
            adapter = Path(job.command[job.command.index("--adapter") + 1])
            run_name = adapter.parent.name
            if run_name in trained:
                assert run_name in job.depends_on, (
                    f"{job.name} evaluates {run_name} without depending on it"
                )

    def test_ablations_never_write_into_the_headline_directory(self, matrix: list[Job]) -> None:
        """results/runs/ feeds the README; a k-sweep landing there corrupts it."""
        ablation_evals = [
            job for job in matrix if job.group != "seeds" and runs(job, "03_eval_arm.py")
        ]
        assert ablation_evals
        for job in ablation_evals:
            assert job.produces.parent.name == "ablations", (
                f"{job.name} writes to {job.produces.parent.name}/, not ablations/"
            )

    def test_seed_replicates_do_land_in_the_headline_directory(self, matrix: list[Job]) -> None:
        """Seed variance is a headline column, so those runs must be aggregated."""
        seed_evals = [job for job in matrix if job.group == "seeds" and runs(job, "03_eval_arm.py")]
        assert seed_evals
        for job in seed_evals:
            assert job.produces.parent.name == "runs"

    def test_each_training_seed_gets_its_own_checkpoint_directory(self, matrix: list[Job]) -> None:
        """Sharing a directory would make seed 2 resume from seed 1's checkpoint."""
        training = [job for job in matrix if runs(job, "04_train.py")]
        assert len(training) >= 3, "expected several training runs"
        directories = [job.produces.parent.parent for job in training]
        assert len(directories) == len(set(directories))


class TestBudget:
    def test_the_grid_is_within_the_planned_envelope(self, matrix: list[Job]) -> None:
        """The plan budgeted 55-75 GPU-hours. A grid far outside that is a bug."""
        assert 40 <= total_gpu_hours(matrix) <= 90

    def test_every_job_costs_something(self, matrix: list[Job]) -> None:
        assert all(job.gpu_hours > 0 for job in matrix)

    def test_groups_cover_every_job(self, matrix: list[Job]) -> None:
        grouped = by_group(matrix)
        assert sum(len(jobs) for jobs in grouped.values()) == len(matrix)

    def test_the_planned_ablations_are_all_present(self, matrix: list[Job]) -> None:
        """The plan named these explicitly; none may be quietly dropped."""
        assert set(by_group(matrix)) == {
            "seeds",
            "corpus-size",
            "rank",
            "epochs",
            "topk",
            "embedder",
            "quantization",
            "cross-base",
        }

    def test_the_rank_sweep_covers_the_planned_values(self, matrix: list[Job]) -> None:
        ranks = {job.name for job in matrix if job.group == "rank"}
        for rank in (8, 32, 64):
            assert f"qlora-r{rank}" in ranks, f"rank {rank} missing"


class TestResumability:
    def test_a_finished_job_is_skipped(self, tmp_path: Path) -> None:
        done = tmp_path / "done.json"
        done.write_text("{}", encoding="utf-8")
        jobs = [
            Job("done", "g", ("true",), done, 1.0),
            Job("todo", "g", ("true",), tmp_path / "missing.json", 1.0),
        ]
        assert [job.name for job in pending(jobs)] == ["todo"]

    def test_budget_counts_only_pending_work(self, tmp_path: Path) -> None:
        done = tmp_path / "done.json"
        done.write_text("{}", encoding="utf-8")
        jobs = [
            Job("done", "g", ("true",), done, 5.0),
            Job("todo", "g", ("true",), tmp_path / "missing.json", 2.0),
        ]
        assert total_gpu_hours(pending(jobs)) == 2.0

    def test_state_summarises_outcomes(self) -> None:
        state = MatrixState(
            results=[
                JobResult("a", "ok", seconds=3600),
                JobResult("b", "failed", seconds=60, returncode=1),
                JobResult("c", "skipped"),
            ]
        )
        summary = state.as_json()
        assert summary["completed"] == 1
        assert summary["failed"] == 1
        assert summary["skipped"] == 1
        assert summary["gpu_hours"] == pytest.approx(1.02, abs=0.01)
