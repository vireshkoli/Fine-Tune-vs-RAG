"""The full experiment matrix, declared as data.

Everything past the headline six arms — seed replicates, the ablation grid, the
size-matched corpus, the cross-base check — is a long sequence of GPU jobs with
dependencies between them. Declaring them as data rather than as a shell script
buys three things that matter on a shared machine:

**Resumability.** Each job names the artifact it produces. A job whose artifact
already exists is skipped, so an interrupted run — a reclaimed GPU, an OOM, a
reboot — resumes by simply being restarted. Training already resumes from its
own checkpoints; this is the same property one level up.

**An honest budget.** Every job carries a GPU-hour estimate derived from a
measured run, so the cost of the grid can be reported *before* it is launched
rather than discovered afterwards.

**Ordering that cannot be got wrong.** An eval that needs an adapter declares
that dependency, so it cannot silently run against a checkpoint that was never
trained and quietly produce a base-model number under a fine-tuned name.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from fvr.config import Paths

#: Measured: 5.06 GPU-hours for 1,875 steps at r=16, 30k samples, one epoch,
#: on an exclusive A40. Every training estimate below scales from this.
BASE_TRAIN_GPU_HOURS = 5.06

#: Measured: ~3 min of scoring for 1,000 items plus ~2 min of model loading.
EVAL_GPU_HOURS = 0.08

#: Measured from build_stats.json: 0.18 GPU-h for 218k parity chunks,
#: 0.84 for 1.6M external chunks.
PARITY_INDEX_GPU_HOURS = 0.18
EXTERNAL_INDEX_GPU_HOURS = 0.84


@dataclass(frozen=True)
class Job:
    """One unit of GPU work."""

    name: str
    group: str
    #: Argv, run from the project root.
    command: tuple[str, ...]
    #: The artifact that proves this job finished. Existence means "skip".
    produces: Path
    gpu_hours: float
    depends_on: tuple[str, ...] = ()
    note: str = ""

    def is_done(self) -> bool:
        return self.produces.exists()


def _train(
    name: str,
    group: str,
    config: str,
    paths: Paths,
    *,
    gpu_hours: float = BASE_TRAIN_GPU_HOURS,
    seed: int | None = None,
    note: str = "",
) -> Job:
    command = ["python", "scripts/04_train.py", "--config", config]
    run_name = name
    if seed is not None:
        command += ["--seed", str(seed)]
    return Job(
        name=name,
        group=group,
        command=tuple(command),
        produces=paths.checkpoints / run_name / "adapter" / "adapter_config.json",
        gpu_hours=gpu_hours,
        note=note,
    )


def _eval(
    name: str,
    group: str,
    arm: str,
    paths: Paths,
    *,
    adapter: str | None = None,
    model: str | None = None,
    corpus: str | None = None,
    retrieval_config: str | None = None,
    seed: int | None = None,
    out_dir: str = "ablations",
    tag: str | None = None,
    depends_on: tuple[str, ...] = (),
    note: str = "",
) -> Job:
    command = ["python", "scripts/03_eval_arm.py", "--arm", arm]
    if adapter:
        command += ["--adapter", adapter]
    if model:
        command += ["--model", model]
    if corpus:
        command += ["--corpus", corpus]
    if retrieval_config:
        command += ["--retrieval-config", retrieval_config]
    if seed is not None:
        command += ["--seed", str(seed)]
    stem = tag or (f"{arm}_seed{seed}" if seed is not None else arm)
    if out_dir != "runs":
        command += ["--out-dir", out_dir]
    if tag:
        command += ["--tag", tag]
    return Job(
        name=name,
        group=group,
        command=tuple(command),
        produces=paths.results / out_dir / f"{stem}.json",
        gpu_hours=EVAL_GPU_HOURS,
        depends_on=depends_on,
        note=note,
    )


def _adapter_path(paths: Paths, run_name: str) -> str:
    return str(paths.checkpoints / run_name / "adapter")


def build_matrix(paths: Paths | None = None) -> list[Job]:
    """Every remaining job, in dependency order."""
    paths = paths or Paths()
    jobs: list[Job] = []

    # --- Seed replicates -----------------------------------------------------
    # These are headline, not ablation: they go in results/runs/ so the report's
    # seed-variance column has something to compute from. Only the *trained* arms
    # have a training seed; base and rag-* are unaffected by it, which is why
    # they are not replicated here.
    for seed in (1, 2):
        run = f"qlora-r16-seed{seed}"
        jobs.append(
            _train(
                run,
                "seeds",
                "configs/train/qlora_r16.yaml",
                paths,
                seed=seed,
                note=f"training seed {seed}; the reported run is seed 42",
            )
        )
        for arm in ("qlora", "qlora-rag", "qlora-rag-parity"):
            jobs.append(
                _eval(
                    f"{arm}-seed{seed}",
                    "seeds",
                    arm,
                    paths,
                    adapter=_adapter_path(paths, run),
                    seed=seed,
                    out_dir="runs",
                    depends_on=(run,),
                )
            )

    # --- Size-matched external corpus ----------------------------------------
    # The one limitation a sharp reviewer pushes on first: parity has 218k chunks
    # and external 1.6M, so "parity beats external" conflates corpus *content*
    # with corpus *size*. Subsampling external to exactly the parity chunk count
    # separates them. Cheapest credibility in the grid.
    jobs.append(
        Job(
            name="index-external-matched",
            group="corpus-size",
            command=(
                "python",
                "scripts/02_build_index.py",
                "--corpus",
                "external",
                "--max-chunks",
                "217661",
                "--name",
                "external-matched",
            ),
            produces=paths.indices / "external-matched" / "index.faiss",
            gpu_hours=0.15,
            note="external subsampled to the parity chunk count",
        )
    )
    jobs.append(
        _eval(
            "rag-external-matched",
            "corpus-size",
            "rag-external",
            paths,
            corpus="external-matched",
            tag="rag-external-matched",
            depends_on=("index-external-matched",),
        )
    )
    jobs.append(
        _eval(
            "qlora-rag-external-matched",
            "corpus-size",
            "qlora-rag",
            paths,
            adapter=_adapter_path(paths, "qlora-r16"),
            corpus="external-matched",
            tag="qlora-rag-external-matched",
            depends_on=("index-external-matched",),
        )
    )

    # --- LoRA rank -----------------------------------------------------------
    # Answers "how did you pick r=16?" with data. Larger ranks cost slightly more
    # per step; the scaling below is from parameter count, not measured, and the
    # runner reports actuals.
    for rank, scale in ((8, 0.98), (32, 1.03), (64, 1.10)):
        run = f"qlora-r{rank}"
        jobs.append(
            _train(
                run,
                "rank",
                f"configs/train/qlora_r{rank}.yaml",
                paths,
                gpu_hours=BASE_TRAIN_GPU_HOURS * scale,
            )
        )
        jobs.append(
            _eval(
                f"qlora-r{rank}-eval",
                "rank",
                "qlora",
                paths,
                adapter=_adapter_path(paths, run),
                tag=f"qlora-r{rank}",
                depends_on=(run,),
            )
        )

    # --- Epochs --------------------------------------------------------------
    # The most expensive ablation in the grid: cost scales linearly with epochs,
    # so these two runs alone are ~25 GPU-hours. Kept because "one pass over 30k
    # unique rows beats N passes over the same rows" is a claim the config makes
    # explicitly, and an unsupported claim in a config comment is worse than no
    # comment.
    for epochs in (2, 3):
        run = f"qlora-e{epochs}"
        jobs.append(
            _train(
                run,
                "epochs",
                f"configs/train/qlora_e{epochs}.yaml",
                paths,
                gpu_hours=BASE_TRAIN_GPU_HOURS * epochs,
            )
        )
        jobs.append(
            _eval(
                f"qlora-e{epochs}-eval",
                "epochs",
                "qlora",
                paths,
                adapter=_adapter_path(paths, run),
                tag=f"qlora-e{epochs}",
                depends_on=(run,),
            )
        )

    # --- Retrieval depth -----------------------------------------------------
    # Under a fixed character budget, raising k adds competing candidates rather
    # than more context, so this measures precision-versus-coverage at constant
    # prefill cost. Run on both corpora because the answer plausibly differs: the
    # parity corpus has one ideal passage per question, the external one does not.
    for k in (1, 3, 10):
        for arm in ("rag-parity", "rag-external"):
            jobs.append(
                _eval(
                    f"{arm}-k{k}",
                    "topk",
                    arm,
                    paths,
                    retrieval_config=f"configs/retrieval/topk_{k}.yaml",
                    tag=f"{arm}-k{k}",
                )
            )

    # --- Embedder ------------------------------------------------------------
    # MedEmbed is a domain fine-tune of bge-large itself, so this isolates domain
    # adaptation from model quality. Both corpora must be re-embedded: an index
    # built with one encoder cannot be queried with another.
    jobs.append(
        Job(
            name="index-parity-medembed",
            group="embedder",
            command=(
                "python",
                "scripts/02_build_index.py",
                "--corpus",
                "parity",
                "--config",
                "configs/retrieval/medembed.yaml",
                "--name",
                "parity-medembed",
            ),
            produces=paths.indices / "parity-medembed" / "index.faiss",
            gpu_hours=PARITY_INDEX_GPU_HOURS,
        )
    )
    jobs.append(
        Job(
            name="index-external-medembed",
            group="embedder",
            command=(
                "python",
                "scripts/02_build_index.py",
                "--corpus",
                "external",
                "--config",
                "configs/retrieval/medembed.yaml",
                "--name",
                "external-medembed",
            ),
            produces=paths.indices / "external-medembed" / "index.faiss",
            gpu_hours=EXTERNAL_INDEX_GPU_HOURS,
            note="~6.5 GiB of index on disk",
        )
    )
    for arm, corpus in (("rag-parity", "parity-medembed"), ("rag-external", "external-medembed")):
        jobs.append(
            _eval(
                f"{arm}-medembed",
                "embedder",
                arm,
                paths,
                corpus=corpus,
                retrieval_config="configs/retrieval/medembed.yaml",
                tag=f"{arm}-medembed",
                depends_on=(f"index-{corpus}",),
            )
        )

    # --- Serving quantisation ------------------------------------------------
    # Only the untrained arms. Merging a bf16 adapter into an NF4 base would have
    # to dequantise first, and the loader refuses it by design, so a quantised
    # fine-tuned arm would not be measuring the configured model.
    for arm in ("base", "rag-parity"):
        jobs.append(
            _eval(
                f"{arm}-nf4",
                "quantization",
                arm,
                paths,
                model="configs/model/qwen3-8b-nf4.yaml",
                tag=f"{arm}-nf4",
            )
        )

    # --- Cross-base robustness ----------------------------------------------
    # Does the conclusion survive a different base model? The parity index is
    # reusable: it is keyed on the embedder, not on the model being evaluated.
    jobs.append(
        _train(
            "qlora-r16-llama",
            "cross-base",
            "configs/train/qlora_r16_llama.yaml",
            paths,
            note="identical recipe on Llama-3.1-8B; adapter deliberately not published",
        )
    )
    for arm in ("base", "rag-parity"):
        jobs.append(
            _eval(
                f"llama-{arm}",
                "cross-base",
                arm,
                paths,
                model="configs/model/llama31-8b.yaml",
                tag=f"llama-{arm}",
            )
        )
    for arm in ("qlora", "qlora-rag-parity"):
        jobs.append(
            _eval(
                f"llama-{arm}",
                "cross-base",
                arm,
                paths,
                model="configs/model/llama31-8b.yaml",
                adapter=_adapter_path(paths, "qlora-r16-llama"),
                tag=f"llama-{arm}",
                depends_on=("qlora-r16-llama",),
            )
        )

    validate(jobs)
    return jobs


class DependencyError(Exception):
    """Raised when the matrix declares a dependency it cannot satisfy."""


def validate(jobs: Sequence[Job]) -> None:
    """Every dependency must exist, and must come earlier in the list."""
    seen: set[str] = set()
    names = [job.name for job in jobs]
    if len(names) != len(set(names)):
        duplicates = sorted({n for n in names if names.count(n) > 1})
        raise DependencyError(f"duplicate job names: {duplicates}")
    for job in jobs:
        for dependency in job.depends_on:
            if dependency not in seen:
                raise DependencyError(
                    f"{job.name} depends on {dependency!r}, which is not defined before it"
                )
        seen.add(job.name)


def pending(jobs: Iterable[Job]) -> list[Job]:
    """Jobs whose output artifact does not yet exist."""
    return [job for job in jobs if not job.is_done()]


def total_gpu_hours(jobs: Iterable[Job]) -> float:
    return sum(job.gpu_hours for job in jobs)


def by_group(jobs: Iterable[Job]) -> dict[str, list[Job]]:
    grouped: dict[str, list[Job]] = {}
    for job in jobs:
        grouped.setdefault(job.group, []).append(job)
    return grouped


@dataclass
class JobResult:
    """What actually happened, for results/matrix_status.json."""

    name: str
    status: str
    seconds: float = 0.0
    returncode: int | None = None
    log: str | None = None


@dataclass
class MatrixState:
    results: list[JobResult] = field(default_factory=list)

    def as_json(self) -> dict[str, object]:
        return {
            "jobs": [
                {
                    "name": r.name,
                    "status": r.status,
                    "seconds": round(r.seconds, 1),
                    "returncode": r.returncode,
                    "log": r.log,
                }
                for r in self.results
            ],
            "completed": sum(1 for r in self.results if r.status == "ok"),
            "failed": sum(1 for r in self.results if r.status == "failed"),
            "skipped": sum(1 for r in self.results if r.status == "skipped"),
            "gpu_hours": round(sum(r.seconds for r in self.results) / 3600, 2),
        }
