# Fine-Tune vs. Retrieve

**When is fine-tuning worth it versus just retrieving?** A controlled head-to-head benchmark in
clinical QA — four arms, one frozen test set, measured on quality, latency and cost per query.

> **Status: under construction (Phase 1 of 10).** No results yet. This README is rewritten in
> Phase 8 with the results table, charts and quickstart. Nothing below is a claim about performance.
>
> **Not for clinical use. Not medical advice. Not validated on real patients.**

## The question

Teams argue about this constantly and usually resolve it by intuition. The deliverable here is
evidence, not a checkpoint:

| Arm | Weights | Retrieval |
| --- | --- | --- |
| `base` | frozen | none |
| `rag` | frozen | domain corpus |
| `qlora` | fine-tuned | none |
| `qlora-rag` | fine-tuned | domain corpus |

All four share the same base weights, prompt scaffolding, decoding settings, hardware and test set.
Two extra diagnostic arms isolate whether fine-tuning wins by absorbing *information* or merely by
learning the answer *format*.

The headline output is a **cost-per-query vs. query-volume curve**. The point where the arms cross
is the actual answer to "when is fine-tuning worth it," and unlike a single accuracy number it
generalises past this dataset.

## Planned stack

Python 3.12 · uv · QLoRA (`peft` + `bitsandbytes` + `trl`) · FAISS · Weights & Biases ·
ruff + mypy(strict) + pytest · Docker · GitHub Actions

## Quickstart

```bash
uv sync --group dev     # CPU-only: lint, type-check, tests
make check              # ruff + mypy + pytest
make setup-check        # GPU, disk, cache isolation and HF token scope
```

Full reproduction instructions land in Phase 8, once there are numbers to reproduce.

## Notes on how this is built

- **Nothing lives only on disk.** Every artifact is reconstructible from GitHub and the Hugging Face
  Hub, because the GPU used to build this is shared university hardware that gets vacated on
  completion. Model revision SHAs are pinned for the same reason.
- **Caches are project-local.** `HF_HOME` and friends point into `.artifacts/`, so this project never
  writes into a shared cache it does not own.
- **Weights and datasets never enter git.** Adapters go to the Hub.

## Licence

MIT for the code. Datasets and models retain their own licences, recorded in `REPORT.md`.
