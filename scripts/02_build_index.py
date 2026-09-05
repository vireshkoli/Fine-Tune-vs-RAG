#!/usr/bin/env python
"""Build a retrieval index.

    uv run python scripts/02_build_index.py --corpus parity
    uv run python scripts/02_build_index.py --corpus external --max-chunks 1500000
    uv run python scripts/02_build_index.py --corpus external --estimate-only

Both corpora are built from **train-side text only**. The script refuses to
proceed if any held-out content hash reaches the index, because a test
question's own explanation sitting in the corpus would turn the benchmark into
a lookup task and invalidate every RAG number.
"""

from __future__ import annotations

from fvr.config import bootstrap_env, load_config  # isort: skip

import argparse
import json
import sys
import time

from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn

from fvr.data.loaders import load_medmcqa
from fvr.retrieval.corpus import build_external_corpus, build_parity_corpus
from fvr.retrieval.embed import embed_to_memmap, load_embedder, load_embedder_config
from fvr.retrieval.index import build_index, save_index

console = Console()

MIRIAD_REPO = "miriad/miriad-5.8M"
MIRIAD_SHARDS = 64


def load_miriad_documents(n_shards: int) -> list[tuple[str, str]]:
    """Unique (title, passage) pairs from the first ``n_shards`` MIRIAD shards.

    Deduplicated on passage text: MIRIAD reuses each passage across ~2.5 QA
    pairs, so indexing raw rows would waste embedding time and skew retrieval
    toward whichever passages happen to be reused most.
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download

    seen: set[str] = set()
    documents: list[tuple[str, str]] = []
    for shard in range(n_shards):
        path = hf_hub_download(
            MIRIAD_REPO,
            f"data/train-{shard:05d}-of-{MIRIAD_SHARDS:05d}.parquet",
            repo_type="dataset",
        )
        frame = pd.read_parquet(path, columns=["paper_title", "passage_text"])
        for title, text in zip(frame["paper_title"], frame["passage_text"], strict=True):
            body = str(text)
            if body in seen:
                continue
            seen.add(body)
            documents.append((str(title), body))
        console.print(f"  shard {shard}: {len(documents):,} unique passages so far")
    return documents


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=["parity", "external"], required=True)
    parser.add_argument("--config", default="configs/retrieval/bge_large.yaml")
    parser.add_argument(
        "--shards", type=int, default=4, help="MIRIAD shards for the external corpus"
    )
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument(
        "--name",
        default=None,
        help="index directory name; defaults to --corpus. Use for size-matched "
        "or re-embedded variants so they do not overwrite the originals.",
    )
    parser.add_argument(
        "--keep-embeddings",
        action="store_true",
        help="keep the raw embedding memmap after the index is built (it is a "
        "duplicate of what IndexFlatIP already stores, and twice the disk)",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="report corpus size and projected embedding time, then stop",
    )
    args = parser.parse_args()

    config = load_config()
    paths = bootstrap_env(config)
    embedder_config = load_embedder_config(args.config)

    split_ids = json.loads((paths.results / "split_ids.json").read_text(encoding="utf-8"))
    held_out_ids = set(split_ids["test"]) | set(split_ids["val"])

    console.print("Loading MedMCQA to establish the held-out content hashes…")
    pool, _ = load_medmcqa("validation")
    forbidden = frozenset(q.content_hash() for q in pool if q.id in held_out_ids)
    console.print(f"  {len(forbidden):,} held-out content hashes are banned from every index")

    if args.corpus == "parity":
        train, report = load_medmcqa("train")
        console.print(f"  train cleaning: {report.summary()}")
        passages, stats = build_parity_corpus(train, forbidden_content_hashes=forbidden)
    else:
        console.print(f"Loading {args.shards} MIRIAD shards…")
        documents = load_miriad_documents(args.shards)
        passages, stats = build_external_corpus(documents, max_chunks=args.max_chunks)

    console.print(f"[bold]{stats.summary()}[/]")

    # Leakage gate. Cheap, and the one failure that would invalidate everything.
    leaked = [p for p in passages if p.source_question_id in held_out_ids]
    if leaked:
        console.print(f"[red]ABORT: {len(leaked):,} passages derive from held-out questions.[/]")
        return 1
    console.print("[green]Leakage check passed:[/] no held-out text in the corpus.")

    if args.estimate_only:
        model = load_embedder(embedder_config)
        sample = [p.text for p in passages[:2048]]
        start = time.perf_counter()
        model.encode(sample, batch_size=embedder_config.batch_size, show_progress_bar=False)
        rate = len(sample) / (time.perf_counter() - start)
        hours = len(passages) / rate / 3600
        console.print(
            f"\n[bold]Estimate[/]: {rate:,.0f} chunks/s -> "
            f"{len(passages):,} chunks in ~{hours:.2f} GPU-hours "
            f"({len(passages) * embedder_config.dimensions * 4 / 2**30:.1f} GiB of float32)"
        )
        return 0

    out_dir = paths.indices / (args.name or args.corpus)
    memmap_path = out_dir / "embeddings.f32"
    console.print(f"Embedding {len(passages):,} chunks with {embedder_config.name}…")

    started = time.perf_counter()
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("embedding", total=len(passages))

        def tick(done: int, total: int) -> None:
            progress.update(task, completed=done, total=total)

        store = embed_to_memmap(
            [p.text for p in passages], embedder_config, memmap_path, progress=tick
        )
    elapsed = time.perf_counter() - started

    console.print(f"Building FAISS index over {store.n:,} vectors…")
    vectors = store.open(mode="r")
    index = build_index(vectors, passages, args.corpus)
    save_index(index, out_dir)
    del vectors

    (out_dir / "build_stats.json").write_text(
        json.dumps(
            {
                "corpus": args.name or args.corpus,
                "base_corpus": args.corpus,
                "chunks": stats.chunks,
                "source_documents": stats.source_documents,
                "excluded_documents": stats.excluded_documents,
                "mean_chunk_chars": stats.mean_chunk_chars,
                "embedder": embedder_config.name,
                "embedder_revision": embedder_config.revision,
                # Feeds the amortised index-build term of the cost model.
                "embed_gpu_seconds": elapsed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    # IndexFlatIP stores the vectors itself, so the memmap is an exact duplicate
    # of the index contents and is never read again — not by load_index, not by
    # the retriever. Keeping it doubles the disk cost of every index (6.5 GiB for
    # the external corpus alone), which matters on a shared box. Rebuilding from
    # scratch re-embeds, which is the documented path anyway.
    if not args.keep_embeddings:
        freed = memmap_path.stat().st_size if memmap_path.is_file() else 0
        memmap_path.unlink(missing_ok=True)
        if freed:
            console.print(f"  removed the build memmap ({freed / 2**30:.1f} GiB reclaimed)")

    console.print(f"[green]Wrote {out_dir}[/] ({elapsed / 60:.1f} min of GPU time)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
