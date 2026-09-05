#!/usr/bin/env python
"""Run the full experiment matrix: seeds, ablations, cross-base.

    uv run python scripts/11_run_matrix.py                    # plan + GPU budget, runs nothing
    uv run python scripts/11_run_matrix.py --only rank        # one group
    uv run python scripts/11_run_matrix.py --execute          # run everything pending
    uv run python scripts/11_run_matrix.py --execute --wait-for-gpu

Dry run by default, because the full grid is tens of GPU-hours on a machine
shared with other people's work and that is not something to start by accident.

Resumable by construction: every job names the artifact it produces, so a job
whose artifact exists is skipped. An interrupted grid — reclaimed GPU, OOM,
reboot — is resumed by re-running this command.
"""

from __future__ import annotations

from fvr.config import bootstrap_env, load_config  # isort: skip

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from fvr.ops.matrix import (
    Job,
    JobResult,
    MatrixState,
    build_matrix,
    by_group,
    pending,
    total_gpu_hours,
)

console = Console()

#: How long to wait between checks when the GPU is busy.
POLL_SECONDS = 300


def show_plan(jobs: list[Job], *, all_jobs: list[Job]) -> None:
    table = Table(title="Experiment matrix", show_lines=False)
    table.add_column("Group", style="bold")
    table.add_column("Pending", justify="right")
    table.add_column("Done", justify="right")
    table.add_column("GPU-h", justify="right")
    table.add_column("What it answers", overflow="fold")

    questions = {
        "seeds": "Is the +6.1 point gain stable across training seeds?",
        "corpus-size": "Is parity's advantage its content, or just that it is smaller?",
        "rank": "Was r=16 the right choice, or arbitrary?",
        "epochs": "Does a second and third pass over the same rows buy anything?",
        "topk": "Precision vs coverage, at a fixed context budget",
        "embedder": "Does a domain-specific embedder retrieve better medicine?",
        "quantization": "What does serving in 4-bit cost in quality and latency?",
        "cross-base": "Does the conclusion hold on a second base model?",
    }

    pending_by_group = by_group(jobs)
    for group, group_jobs in by_group(all_jobs).items():
        outstanding = pending_by_group.get(group, [])
        table.add_row(
            group,
            str(len(outstanding)),
            str(len(group_jobs) - len(outstanding)),
            f"{total_gpu_hours(outstanding):.1f}",
            questions.get(group, ""),
        )
    console.print(table)
    console.print(
        f"\n[bold]{len(jobs)} job(s) pending[/] — "
        f"[bold]{total_gpu_hours(jobs):.1f} GPU-hours[/] estimated "
        f"({total_gpu_hours(jobs) / 24:.1f} days of exclusive single-GPU time)."
    )


def wait_for_gpu(device: int) -> None:
    from fvr.eval.device import device_occupancy

    while True:
        occupancy = device_occupancy(device)
        if occupancy.is_exclusive:
            console.print(f"[green]GPU {device} is free.[/]")
            return
        console.print(
            f"GPU {device} busy ({occupancy.foreign_mib:,} MiB held by "
            f"{len(occupancy.foreign_pids)} other process(es)); "
            f"checking again in {POLL_SECONDS // 60} min."
        )
        time.sleep(POLL_SECONDS)


def run_job(job: Job, log_dir: Path) -> JobResult:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job.name}.log"
    console.print(f"[bold cyan]{job.name}[/] ({job.gpu_hours:.1f} GPU-h est.)")
    console.print(f"  {' '.join(job.command)}")

    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            ["uv", "run", *job.command],
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.perf_counter() - started

    if completed.returncode != 0:
        console.print(f"  [red]FAILED[/] after {elapsed / 60:.1f} min — see {log_path}")
        return JobResult(job.name, "failed", elapsed, completed.returncode, str(log_path))
    if not job.produces.exists():
        console.print(f"  [red]FAILED[/]: exited 0 but did not write {job.produces}")
        return JobResult(job.name, "failed", elapsed, 0, str(log_path))

    console.print(f"  [green]done[/] in {elapsed / 60:.1f} min ({elapsed / 3600:.2f} GPU-h)")
    return JobResult(job.name, "ok", elapsed, 0, str(log_path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=None, help="run one group (see the plan table)")
    parser.add_argument("--execute", action="store_true", help="actually run (default: dry run)")
    parser.add_argument(
        "--wait-for-gpu",
        action="store_true",
        help="poll until the inference GPU is exclusive instead of refusing",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="abort the grid on the first failure rather than continuing",
    )
    args = parser.parse_args()

    project = load_config()
    paths = bootstrap_env(project)

    all_jobs = build_matrix(paths)
    if args.only:
        groups = {job.group for job in all_jobs}
        if args.only not in groups:
            console.print(f"[red]Unknown group {args.only!r}; expected one of {sorted(groups)}[/]")
            return 1
        all_jobs = [job for job in all_jobs if job.group == args.only]

    todo = pending(all_jobs)
    show_plan(todo, all_jobs=all_jobs)

    if not todo:
        console.print("\n[green]Nothing pending — every artifact already exists.[/]")
        return 0

    if not args.execute:
        console.print("\n[yellow]Dry run. Re-run with --execute to start.[/]")
        return 0

    from fvr.eval.device import device_occupancy

    device = project.inference_device
    occupancy = device_occupancy(device)
    if not occupancy.is_exclusive:
        if not args.wait_for_gpu:
            console.print(
                f"\n[red]GPU {device} is not exclusive "
                f"({occupancy.foreign_mib:,} MiB held by another process).[/] "
                "Timing would be meaningless and training would contend. "
                "Pass --wait-for-gpu to poll, or wait for it to free."
            )
            return 1
        wait_for_gpu(device)

    state = MatrixState()
    failed: set[str] = set()
    log_dir = paths.artifacts / "logs"

    for job in todo:
        blocked = sorted(set(job.depends_on) & failed)
        if blocked:
            console.print(f"[yellow]{job.name}: skipped, depends on failed {blocked}[/]")
            state.results.append(JobResult(job.name, "skipped"))
            failed.add(job.name)
            continue

        result = run_job(job, log_dir)
        state.results.append(result)
        if result.status == "failed":
            failed.add(job.name)
            if args.stop_on_failure:
                console.print("[red]Stopping on first failure as requested.[/]")
                break

        status_path = paths.results / "matrix_status.json"
        status_path.write_text(json.dumps(state.as_json(), indent=2) + "\n", encoding="utf-8")

    summary = state.as_json()
    console.print(
        f"\n[bold]Matrix finished[/]: {summary['completed']} ok, "
        f"{summary['failed']} failed, {summary['skipped']} skipped, "
        f"{summary['gpu_hours']} GPU-hours spent."
    )
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
