#!/usr/bin/env python
"""Generate free-text answers for one arm.

    uv run python scripts/12_freetext_eval.py --arm base
    uv run python scripts/12_freetext_eval.py --arm qlora --adapter .artifacts/...

Writes ``results/freetext/<arm>_seed<N>.json``: one generated answer per item,
with token counts and latency. Nothing is scored here — judging is a separate
step (``13_judge_freetext.py``) so that answers are generated once and can be
re-judged under a corrected rubric without spending GPU-hours again.

The item set is a deterministic 300-item subset of the same frozen test split
the MCQ arms use, so free-text and MCQ results describe the same questions.
"""

from __future__ import annotations

from fvr.config import bootstrap_env, load_config  # isort: skip

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import Progress

from fvr.data.loaders import load_medmcqa
from fvr.data.schema import Passage
from fvr.eval.device import device_occupancy
from fvr.eval.freetext import (
    DEFAULT_N_ITEMS,
    FreeTextAnswer,
    FreeTextRun,
    option_dependent_reason,
    reference_answer,
    select_freetext_items,
)
from fvr.eval.latency import LatencyRecorder
from fvr.eval.runner import environment_fingerprint
from fvr.inference.arms import get_arm
from fvr.models.loader import load_base_model, load_model_config
from fvr.prompts.templates import build_freetext_prompt
from fvr.seeding import set_all_seeds

console = Console()

MAX_NEW_TOKENS = 96
LATENCY_SAMPLES = 60
WARMUP = 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="base")
    parser.add_argument("--model", default="configs/model/qwen3-8b.yaml")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--retrieval-config", default="configs/retrieval/bge_large.yaml")
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-items", type=int, default=DEFAULT_N_ITEMS)
    parser.add_argument("--limit", type=int, default=None, help="smoke run")
    parser.add_argument(
        "--out",
        default=None,
        help="write to this path instead of results/freetext/ — use for smoke runs, so a "
        "partial answer set can never be mistaken for the real artifact",
    )
    args = parser.parse_args()

    config = load_config()
    paths = bootstrap_env(config)
    seed = args.seed if args.seed is not None else config.seed
    batch_size = args.batch_size if args.batch_size is not None else config.eval_batch_size
    set_all_seeds(seed)

    arm = get_arm(args.arm)
    if arm.uses_adapter and not args.adapter:
        console.print(f"[red]Arm {arm.name!r} needs --adapter.[/]")
        return 1

    corpus = args.corpus or arm.corpus
    index_dir = paths.indices / corpus if arm.uses_retrieval else None
    if index_dir is not None and not (index_dir / "index.faiss").is_file():
        console.print(f"[red]Missing index {corpus!r}. Run: make index CORPUS={corpus}[/]")
        return 1

    manifest = json.loads((paths.results / "split_manifest.json").read_text(encoding="utf-8"))
    split_ids = json.loads((paths.results / "split_ids.json").read_text(encoding="utf-8"))
    test_ids = set(split_ids["test"])

    pool, _ = load_medmcqa("validation")
    test_items = [q for q in pool if q.id in test_ids]
    excluded = [q for q in test_items if q.answer_idx is not None and option_dependent_reason(q)]
    questions = select_freetext_items(test_items, n=args.n_items, seed=config.seed)
    if args.limit:
        questions = questions[: args.limit]
    console.print(
        f"Free-text set: {len(questions)} items from the frozen test split "
        f"({len(excluded)} excluded: gold answer only meaningful with the options shown)"
    )

    model_config = load_model_config(args.model)
    loaded = load_base_model(model_config, use_cache=not args.adapter)
    if args.adapter:
        from fvr.models.loader import attach_adapter

        loaded = attach_adapter(loaded, args.adapter)

    from fvr.inference.engine import InferenceEngine

    engine = InferenceEngine(loaded)

    retrieved: list[list[Passage]] = [[] for _ in questions]
    retrieval_info = None
    if index_dir is not None:
        from fvr.retrieval.embed import load_embedder_config
        from fvr.retrieval.retriever import load_retrieval_config, load_retriever

        embedder_config = load_embedder_config(args.retrieval_config)
        retrieval_config = load_retrieval_config(args.retrieval_config)
        retriever = load_retriever(index_dir, embedder_config, retrieval_config)
        console.print(f"  {len(retriever.index):,} passages indexed ({corpus})")
        retrieved = []
        for start in range(0, len(questions), batch_size):
            retrieved.extend(retriever.retrieve_many(questions[start : start + batch_size]))
        retrieval_info = {
            "corpus": corpus,
            "n_passages": len(retriever.index),
            "embedder": embedder_config.name,
            "top_k": retrieval_config.top_k,
            "max_context_chars": retrieval_config.max_context_chars,
        }

    prompts = [build_freetext_prompt(q, ctx) for q, ctx in zip(questions, retrieved, strict=True)]

    answers: list[FreeTextAnswer] = []
    console.print(f"Generating for [bold]{arm.name}[/]…")
    with Progress(console=console) as progress:
        task = progress.add_task("generating", total=len(prompts))
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start : start + batch_size]
            generated = engine.generate_batch(chunk, max_new_tokens=MAX_NEW_TOKENS)
            for question, context, item in zip(
                questions[start : start + batch_size],
                retrieved[start : start + batch_size],
                generated,
                strict=True,
            ):
                answers.append(
                    FreeTextAnswer(
                        question_id=question.id,
                        subject=question.subject,
                        question=question.question,
                        reference=reference_answer(question),
                        answer=item.text,
                        prompt_tokens=item.prompt_tokens,
                        completion_tokens=item.completion_tokens,
                        n_passages=len(context),
                    )
                )
            progress.update(task, completed=len(answers))

    # Timed one at a time, warmup discarded — a batched wall-clock divided by
    # batch size is not a latency anybody experiences. LatencyRecorder is used
    # rather than a bare perf_counter because it synchronises CUDA first; without
    # that, generation returns before the GPU has finished and every figure is
    # optimistic.
    console.print("Timing single-item generation…")
    recorder = LatencyRecorder(warmup=WARMUP)
    for prompt in prompts[: LATENCY_SAMPLES + WARMUP]:
        with recorder.measure():
            engine.generate_one(prompt, max_new_tokens=MAX_NEW_TOKENS)

    run = FreeTextRun(
        arm=arm.name,
        excluded_option_dependent=len(excluded),
        seed=seed,
        split_sha256=manifest["splits"]["test"]["sha256"],
        model=loaded.describe(),
        environment=environment_fingerprint(),
        answers=answers,
        latency=recorder.summary().as_dict(),
        retrieval=retrieval_info,
        device_occupancy=device_occupancy(model_config.device).as_dict(),
    )
    out = Path(args.out) if args.out else paths.results / "freetext" / f"{arm.name}_seed{seed}.json"
    run.write(out)

    console.print()
    console.print(f"[bold]{arm.name}[/]: {len(answers)} answers")
    console.print(f"  empty      {run.empty_answers}")
    console.print(f"  mean tokens {run.mean_completion_tokens:.1f} generated")
    console.print(f"  latency    {run.latency}")
    console.print(f"  written    [cyan]{out}[/]")
    if run.empty_answers:
        console.print(
            f"  [yellow]{run.empty_answers} empty answer(s) — these are kept and will be "
            "judged as failures, not dropped.[/]"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
