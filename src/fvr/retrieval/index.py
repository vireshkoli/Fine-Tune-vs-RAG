"""FAISS index build, save, load and search.

``IndexFlatIP`` — exact inner-product search over L2-normalised vectors, which
is cosine similarity. Exact rather than HNSW or IVF on purpose: at a few million
chunks the approximate variants buy latency this benchmark does not need, and
they introduce a recall parameter that would have to be defended as a possible
explanation for any RAG result. "Retrieval is exact" removes a variable from the
comparison, which is worth more here than milliseconds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from fvr.data.schema import Passage

if TYPE_CHECKING:  # pragma: no cover
    pass

#: Vectors added per call. Keeps the transient copy FAISS makes bounded, which
#: matters far more for the 31GB of system RAM than for the 46GB of VRAM.
ADD_BATCH = 50_000


@dataclass
class VectorIndex:
    """A FAISS index plus the passages its row ids refer to."""

    index: Any
    passages: list[Passage]
    corpus: str

    def __len__(self) -> int:
        return len(self.passages)

    def search(self, queries: np.ndarray, k: int) -> list[list[Passage]]:
        """Top-k passages per query, each carrying its similarity score."""
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        k = min(k, len(self.passages))
        scores, indices = self.index.search(np.ascontiguousarray(queries, dtype=np.float32), k)

        results: list[list[Passage]] = []
        for row_scores, row_indices in zip(scores, indices, strict=True):
            hits: list[Passage] = []
            for score, idx in zip(row_scores, row_indices, strict=True):
                if idx < 0:  # FAISS pads with -1 when fewer than k exist
                    continue
                hits.append(self.passages[int(idx)].model_copy(update={"score": float(score)}))
            results.append(hits)
        return results


def build_index(vectors: np.ndarray, passages: list[Passage], corpus: str) -> VectorIndex:
    """Build an exact inner-product index over pre-normalised vectors."""
    import faiss

    if len(vectors) != len(passages):
        raise ValueError(f"{len(vectors)} vectors but {len(passages)} passages")
    if len(passages) == 0:
        raise ValueError("cannot build an index over zero passages")

    index = faiss.IndexFlatIP(int(vectors.shape[1]))
    for start in range(0, len(vectors), ADD_BATCH):
        index.add(np.ascontiguousarray(vectors[start : start + ADD_BATCH], dtype=np.float32))
    return VectorIndex(index=index, passages=passages, corpus=corpus)


def save_index(index: VectorIndex, directory: Path) -> None:
    """Persist the index and its passages side by side."""
    import faiss

    directory.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index.index, str(directory / "index.faiss"))
    with (directory / "passages.jsonl").open("w", encoding="utf-8") as handle:
        for passage in index.passages:
            handle.write(
                json.dumps(
                    {
                        "id": passage.id,
                        "text": passage.text,
                        "corpus": passage.corpus,
                        "source_question_id": passage.source_question_id,
                        "title": passage.title,
                    }
                )
                + "\n"
            )
    (directory / "meta.json").write_text(
        json.dumps({"corpus": index.corpus, "n": len(index.passages)}, indent=2) + "\n",
        encoding="utf-8",
    )


def load_index(directory: Path) -> VectorIndex:
    import faiss

    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    passages: list[Passage] = []
    with (directory / "passages.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            passages.append(Passage(**json.loads(line)))

    if len(passages) != int(meta["n"]):
        raise ValueError(f"index says {meta['n']} passages but {len(passages)} were loaded")

    return VectorIndex(
        index=faiss.read_index(str(directory / "index.faiss")),
        passages=passages,
        corpus=str(meta["corpus"]),
    )
