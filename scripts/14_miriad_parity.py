#!/usr/bin/env python
"""Build the MIRIAD information-parity experiment: split, training sets, index.

    uv run python scripts/14_miriad_parity.py            # split + datasets + index
    uv run python scripts/14_miriad_parity.py --no-index # CPU only: split + datasets

One script decides the split and derives *everything* from it — the ``qa`` and
``doc`` training files, the frozen test items, and the retrieval index — so the
weights and the index are provably built from the same passages rather than
from two computations that happen to agree.

Outputs:
  results/miriad/split.json          ids + stats (committed)
  results/miriad/test_items.jsonl    300 frozen test items with references (committed)
  .artifacts/datasets/miriad/qa.jsonl   chat records, last 200 = validation
  .artifacts/datasets/miriad/doc.jsonl  passage chunks, last 200 = validation
  .artifacts/indices/miriad-parity/     FAISS index over exactly the training passages
"""

from __future__ import annotations

from fvr.config import bootstrap_env, load_config  # isort: skip

import argparse
import json
import sys
import time

from rich.console import Console

from fvr.data.miriad import (
    DEFAULT_SHARDS,
    assert_no_leakage,
    build_parity_split,
    corpus_documents,
    doc_texts,
    load_miriad_rows,
    sft_messages,
    write_split,
)
from fvr.retrieval.corpus import build_external_corpus

console = Console()

#: Matches the MedMCQA fine-tune's 30k rows, so the two parity experiments cost
#: the same to train and the adapters are comparable in what they absorbed.
N_TRAIN_QA = 30_000
N_TEST = 300


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=int, default=DEFAULT_SHARDS)
    parser.add_argument("--n-train", type=int, default=N_TRAIN_QA)
    parser.add_argument("--n-test", type=int, default=N_TEST)
    parser.add_argument("--config", default="configs/retrieval/bge_large.yaml")
    parser.add_argument("--no-index", action="store_true")
    args = parser.parse_args()

    project = load_config()
    paths = bootstrap_env(project)

    console.print(f"Loading {args.shards} MIRIAD shards…")
    rows = load_miriad_rows(args.shards)
    split = build_parity_split(rows, n_train_qa=args.n_train, n_test=args.n_test, seed=project.seed)
    assert_no_leakage(split)
    console.print(f"  {split.summary()}")

    out = paths.results / "miriad"
    write_split(split, out / "split.json")
    with (out / "test_items.jsonl").open("w", encoding="utf-8") as handle:
        for qa in split.test:
            handle.write(json.dumps(qa.as_question().model_dump(), ensure_ascii=False) + "\n")
    console.print(f"  split -> [cyan]{out}[/]")

    data_dir = paths.datasets / "miriad"
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / "qa.jsonl").open("w", encoding="utf-8") as handle:
        for qa in split.train:
            handle.write(json.dumps({"qa_id": qa.qa_id, "messages": sft_messages(qa)}) + "\n")
    chunks = doc_texts(split)
    with (data_dir / "doc.jsonl").open("w", encoding="utf-8") as handle:
        for text in chunks:
            handle.write(json.dumps({"text": text}) + "\n")
    console.print(
        f"  qa.jsonl: {len(split.train):,} records | doc.jsonl: {len(chunks):,} chunks "
        f"-> [cyan]{data_dir}[/]"
    )

    if args.no_index:
        return 0

    from fvr.retrieval.embed import embed_to_memmap, load_embedder_config
    from fvr.retrieval.index import build_index, save_index

    embedder_config = load_embedder_config(args.config)
    passages, stats = build_external_corpus(corpus_documents(split), corpus="miriad-parity")
    console.print(f"  {stats.summary()}")
    if len(passages) != len(chunks):
        console.print(
            f"[red]index has {len(passages):,} chunks but doc.jsonl has {len(chunks):,}; "
            "the doc variant would not be trained on what the index serves[/]"
        )
        return 1

    index_dir = paths.indices / "miriad-parity"
    memmap = index_dir / "embeddings.f32"
    console.print(f"Embedding {len(passages):,} chunks with {embedder_config.name}…")
    started = time.perf_counter()
    store = embed_to_memmap([p.text for p in passages], embedder_config, memmap)
    elapsed = time.perf_counter() - started
    vectors = store.open(mode="r")
    save_index(build_index(vectors, passages, "miriad-parity"), index_dir)
    del vectors
    memmap.unlink(missing_ok=True)
    (index_dir / "build_stats.json").write_text(
        json.dumps(
            {
                "corpus": "miriad-parity",
                "chunks": len(passages),
                "source_documents": len(split.passages),
                "embedder": embedder_config.name,
                "embedder_revision": embedder_config.revision,
                "embed_gpu_seconds": elapsed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    console.print(f"[green]Wrote {index_dir}[/] ({elapsed / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
