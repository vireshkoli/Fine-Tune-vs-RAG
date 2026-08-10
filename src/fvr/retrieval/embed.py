"""Batched embedding, written straight to a memory-mapped array.

The memmap is not premature optimisation. This box has 31GB of system RAM and
the corpus runs to millions of chunks: holding an (N, 1024) float32 array in
RAM alongside the embedding model and the chunk list is exactly how the CPU
side gets OOM-killed two thirds of the way through an hour-long job. Writing
each batch through to disk keeps peak RAM flat regardless of corpus size.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field


class EmbedderConfig(BaseModel):
    """From ``configs/retrieval/*.yaml``."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    name: str
    repo_id: str
    revision: str = Field(min_length=7)
    dimensions: int
    max_seq_length: int = 512
    batch_size: int = 256
    device: int = 0
    #: SentenceTransformer defaults to float32, which on an A40 runs ~4.3x slower
    #: than fp16 (248 vs 1,054 chunks/s measured). Verified equivalent for
    #: retrieval before adopting: per-vector cosine >= 0.9995 against fp32,
    #: top-1 agreement 1.000, top-5 set overlap 0.998 over 3,000 real chunks.
    dtype: Literal["float16", "float32"] = "float16"
    #: Texts handed to the encoder per call. Larger lets it sort by length across
    #: a wider window and cuts padding waste; bounded so RAM stays flat.
    encode_window: int = 8192
    normalize: bool = True
    #: bge-family models expect this prefix on *queries* only, not on documents.
    query_prefix: str = ""


def load_embedder_config(path: Path | str) -> EmbedderConfig:
    """Read the ``embedder:`` section, or the whole file if it has no sections.

    Retrieval configs carry both an ``embedder`` and a ``retrieval`` block so
    the two stay in one reviewable file; accepting a flat mapping too keeps
    small test fixtures simple.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return EmbedderConfig(**raw.get("embedder", raw))


@dataclass
class EmbeddingStore:
    """An (N, D) float32 memmap plus the metadata needed to reopen it."""

    path: Path
    n: int
    dimensions: int

    @property
    def sidecar(self) -> Path:
        return self.path.with_suffix(".meta.json")

    def open(self, mode: Any = "r") -> Any:
        """Reopen the array. ``mode`` is a numpy memmap mode literal."""
        return np.memmap(self.path, dtype=np.float32, mode=mode, shape=(self.n, self.dimensions))

    def write_meta(self) -> None:
        import json

        self.sidecar.write_text(
            json.dumps({"n": self.n, "dimensions": self.dimensions}, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> EmbeddingStore:
        import json

        meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        return cls(path=path, n=int(meta["n"]), dimensions=int(meta["dimensions"]))


def load_embedder(config: EmbedderConfig) -> Any:
    # Returns a SentenceTransformer. Typed loosely because the package ships no
    # annotations, so a precise return type would be a fiction mypy cannot check.
    from fvr.config import bootstrap_env

    bootstrap_env()

    import torch
    from sentence_transformers import SentenceTransformer

    device = f"cuda:{config.device}" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(config.repo_id, revision=config.revision, device=device)
    model.max_seq_length = config.max_seq_length
    # fp16 only on GPU; on CPU it is slower than float32, not faster.
    if config.dtype == "float16" and torch.cuda.is_available():
        model = model.half()
    return model


def embed_to_memmap(
    texts: Sequence[str],
    config: EmbedderConfig,
    out_path: Path,
    *,
    model: Any = None,
    progress: Any = None,
) -> EmbeddingStore:
    """Embed ``texts`` into a float32 memmap at ``out_path``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    store = EmbeddingStore(path=out_path, n=len(texts), dimensions=config.dimensions)
    if not texts:
        raise ValueError("nothing to embed")

    encoder = model if model is not None else load_embedder(config)
    array = store.open(mode="w+")

    # Hand the encoder a large window at a time rather than one GPU batch.
    # sentence-transformers sorts by length inside each call to minimise padding,
    # so a 256-text window wastes most of that: measured 341 chunks/s at 256
    # against ~1,000 on uniform-length text. The window is still bounded so peak
    # RAM stays flat — the whole point of writing through a memmap.
    window = max(config.batch_size, config.encode_window)
    for start in range(0, len(texts), window):
        batch = list(texts[start : start + window])
        vectors = encoder.encode(
            batch,
            batch_size=config.batch_size,
            normalize_embeddings=config.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        array[start : start + len(batch)] = vectors.astype(np.float32)
        if progress is not None:
            progress(min(start + len(batch), len(texts)), len(texts))

    array.flush()
    del array
    store.write_meta()
    return store


def embed_queries(texts: Sequence[str], config: EmbedderConfig, *, model: Any = None) -> np.ndarray:
    """Embed queries, applying the model's query prefix if it has one.

    bge asymmetric retrieval expects an instruction prefix on the query side and
    none on the document side; applying it to both, or neither, measurably
    degrades recall.
    """
    encoder = model if model is not None else load_embedder(config)
    prefixed = [f"{config.query_prefix}{t}" for t in texts] if config.query_prefix else list(texts)
    vectors = encoder.encode(
        prefixed,
        batch_size=config.batch_size,
        normalize_embeddings=config.normalize,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype=np.float32)
