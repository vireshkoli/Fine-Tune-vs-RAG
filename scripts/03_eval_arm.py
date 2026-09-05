#!/usr/bin/env python
"""Evaluate one arm on the frozen test set.

    uv run python scripts/03_eval_arm.py --arm base
    uv run python scripts/03_eval_arm.py --arm base --limit 50   # smoke run

Writes ``results/runs/<arm>_seed<N>.json`` containing accuracy with a bootstrap
CI, p50/p95 latency with warmup discarded, per-subject breakdown, and the model
revision and environment that produced them.
"""

from __future__ import annotations

from fvr.config import bootstrap_env, load_config  # isort: skip

import argparse
import json
import sys

from rich.console import Console
from rich.progress import Progress

from fvr.data.loaders import load_medmcqa
from fvr.eval.runner import run_arm
from fvr.inference.arms import get_arm
from fvr.models.loader import load_base_model, load_model_config
from fvr.seeding import set_all_seeds

console = Console()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="base", help="arm name (see fvr.inference.arms)")
    parser.add_argument("--model", default="configs/model/qwen3-8b.yaml")
    parser.add_argument("--seed", type=int, default=None, help="defaults to configs/base.yaml")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="defaults to configs/base.yaml"
    )
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N items")
    parser.add_argument("--adapter", default=None, help="LoRA adapter path for fine-tuned arms")
    parser.add_argument("--retrieval-config", default="configs/retrieval/bge_large.yaml")
    parser.add_argument(
        "--corpus",
        default=None,
        help="override the arm's index directory (for size-matched or re-embedded corpora)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="write under results/<dir>/ instead of results/runs/ (use for ablations)",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="output filename stem; defaults to <arm>_seed<N>",
    )
    args = parser.parse_args()

    config = load_config()
    paths = bootstrap_env(config)
    seed = args.seed if args.seed is not None else config.seed
    batch_size = args.batch_size if args.batch_size is not None else config.eval_batch_size
    set_all_seeds(seed)

    arm = get_arm(args.arm)
    if arm.uses_adapter and not args.adapter:
        console.print(f"[red]Arm {arm.name!r} needs --adapter (training lands in Phase 5).[/]")
        return 1

    corpus = args.corpus or arm.corpus
    index_dir = paths.indices / corpus if arm.uses_retrieval else None
    if index_dir is not None and not (index_dir / "index.faiss").is_file():
        console.print(
            f"[red]Arm {arm.name!r} needs the {arm.corpus!r} index. "
            f"Run: make index CORPUS={arm.corpus}[/]"
        )
        return 1

    manifest = json.loads((paths.results / "split_manifest.json").read_text(encoding="utf-8"))
    split_ids = json.loads((paths.results / "split_ids.json").read_text(encoding="utf-8"))
    test_ids = set(split_ids["test"])

    console.print("Loading the frozen test split…")
    pool, _ = load_medmcqa("validation")
    questions = [q for q in pool if q.id in test_ids]
    if len(questions) != len(test_ids):
        console.print(
            f"[red]Expected {len(test_ids)} test items, resolved {len(questions)}. "
            "The split no longer matches the data — run `make verify-splits`.[/]"
        )
        return 1
    questions.sort(key=lambda q: q.id)
    if args.limit:
        questions = questions[: args.limit]

    model_config = load_model_config(args.model)
    console.print(
        f"Loading [cyan]{model_config.repo_id}[/] @ {model_config.revision[:12]} "
        f"({model_config.dtype}, thinking={model_config.enable_thinking})…"
    )
    # Merging folds LoRA weights into the base *in place*, so a merged run must
    # never come from the shared cache — a later base-arm run would otherwise
    # receive the fine-tuned weights and be silently wrong.
    loaded = load_base_model(model_config, use_cache=not args.adapter)
    if args.adapter:
        from fvr.models.loader import attach_adapter

        loaded = attach_adapter(loaded, args.adapter)

    from fvr.inference.engine import InferenceEngine

    engine = InferenceEngine(loaded)

    retrieve = None
    retrieval_info = None
    if index_dir is not None:
        from fvr.retrieval.embed import load_embedder_config
        from fvr.retrieval.retriever import load_retrieval_config, load_retriever

        embedder_config = load_embedder_config(args.retrieval_config)
        retrieval_config = load_retrieval_config(args.retrieval_config)
        console.print(
            f"Loading the {arm.corpus!r} index with {embedder_config.name} "
            f"(top_k={retrieval_config.top_k}, budget={retrieval_config.max_context_chars} chars)…"
        )
        retriever = load_retriever(index_dir, embedder_config, retrieval_config)
        console.print(f"  {len(retriever.index):,} passages indexed")
        retrieve = retriever.retrieve_many
        retrieval_info = {
            "corpus": corpus,
            "n_passages": len(retriever.index),
            "embedder": embedder_config.name,
            "embedder_revision": embedder_config.revision,
            "top_k": retrieval_config.top_k,
            "max_context_chars": retrieval_config.max_context_chars,
        }

    console.print(f"Scoring [bold]{arm.name}[/] over {len(questions):,} items…")
    with Progress(console=console) as progress:
        task = progress.add_task("scoring", total=len(questions))

        def tick(done: int, total: int) -> None:
            progress.update(task, completed=done, total=total)

        result = run_arm(
            arm,
            engine,
            questions,
            seed=seed,
            split_sha256=manifest["splits"]["test"]["sha256"],
            batch_size=batch_size,
            retrieve=retrieve,
            retrieval_info=retrieval_info,
            device=model_config.device,
            progress=tick,
        )

    # Ablations write elsewhere on purpose: results/runs/ feeds the headline
    # tables, and a top-k sweep landing there would silently average six
    # different retrieval settings into one reported arm.
    out_dir = paths.results / (args.out_dir or "runs")
    out = out_dir / f"{args.tag or f'{arm.name}_seed{seed}'}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.write(out)

    console.print()
    console.print(f"[bold]{arm.name}[/] (seed {seed}, n={result.n_items:,})")
    console.print(
        f"  accuracy  [bold]{result.accuracy:.3f}[/]  "
        f"95% CI [{result.ci_low:.3f}, {result.ci_high:.3f}]"
    )
    console.print(f"  latency   {result.latency}")
    console.print(
        f"  MDE       {result.minimum_detectable_effect:.3f} "
        "(smallest gap this test-set size can resolve)"
    )
    if result.groundedness:
        g = result.groundedness
        console.print(
            f"  retrieval hit={g['retrieval_hit_rate']:.3f} "
            f"grounded={g['grounded_rate']:.3f} "
            f"ctx={g['mean_context_chars']:.0f} chars over {g['mean_passages']:.1f} passages"
        )
    if result.device_occupancy and not result.device_occupancy.get("exclusive"):
        console.print(
            f"  [yellow]WARNING: GPU {model_config.device} was shared during timing "
            f"({result.device_occupancy['foreign_mib']} MiB held by another process). "
            "Latency is not comparable with exclusive runs.[/]"
        )
    console.print(f"  written   [cyan]{out}[/]")

    chance = 1.0 / 4
    if result.ci_low <= chance:
        console.print(
            f"  [yellow]CI includes chance ({chance:.2f}) — this arm is not "
            "distinguishable from guessing.[/]"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
