---
license: apache-2.0
language:
  - en
tags:
  - medical
  - retrieval
  - faiss
  - medmcqa
  - benchmark
pretty_name: Fine-Tune vs RAG — parity retrieval index
size_categories:
  - 100K<n<1M
---

# Fine-Tune vs RAG — parity retrieval index

The exact FAISS index the
[Fine-Tune vs RAG benchmark](https://github.com/vireshkoli/Fine-Tune-vs-RAG)
used for its `rag-parity` arm, published so the
[live demo](https://huggingface.co/spaces/vireshk/fine-tune-vs-rag-live)
retrieves over the same passages the report did.

> **Not for clinical use. Not medical advice. Not a medical device.** This is
> exam-explanation text from a public benchmark dataset, chunked for retrieval
> research. It contains the textual noise and errors documented in the report.

## What is in it

| File | Contents |
| --- | --- |
| `index.faiss` | `IndexFlatIP` over 217,661 L2-normalised 1024-d vectors — exact cosine search |
| `passages.jsonl` | one chunk per row: `id`, `text`, `corpus`, `source_question_id` (the MedMCQA train item it came from), `title` |
| `meta.json` | corpus name and row count, checked on load |
| `build_stats.json` | chunking and embedding statistics from the build |

- **Source:** the `exp` (explanation) field of the **MedMCQA training split**
  only — 139,856 documents. No text derived from the benchmark's frozen test
  items is present; the repository's leakage test proves this against the
  committed split ids.
- **Chunking:** as in `fvr.retrieval.corpus`, mean 403 characters per chunk.
- **Embedder:** `BAAI/bge-large-en-v1.5` at revision
  `d4aa6901d3a41ba39fb536a557fa166f842b0e09`, float16, normalised, with the
  document side embedded *without* the query instruction. Queries must be
  prefixed with `Represent this sentence for searching relevant passages: `.

The name "parity" is the benchmark's: this corpus holds the same explanations
the fine-tuned arm was trained on, so retrieval and fine-tuning are compared
on identical information.

## Loading

```python
import faiss, json
from huggingface_hub import hf_hub_download

repo = "vireshk/fine-tune-vs-rag-parity-index"
index = faiss.read_index(hf_hub_download(repo, "index.faiss", repo_type="dataset"))
with open(hf_hub_download(repo, "passages.jsonl", repo_type="dataset")) as f:
    passages = [json.loads(line) for line in f]
```

## Licence

MedMCQA is released under Apache 2.0 (Pal, Umapathi & Sankarasubbu, 2022);
these chunks are a derived work under the same licence.
