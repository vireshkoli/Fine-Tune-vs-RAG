# Handover: resurrecting this project on another machine

This benchmark ran on a shared university A40 box that gets **fully vacated on
completion** — `.artifacts/` is deleted and the disk returned to its prior
state. Everything below is therefore written on the assumption that *the
original machine no longer exists*.

Nothing here needs the lab GPU, a support ticket, or anyone's permission. If a
step below cannot be completed from a clean clone, that is a bug in the repo,
not a gap in this document.

---

## 1. What lives where

| Artifact | Where it survives | How to get it back |
| --- | --- | --- |
| Source, configs, every results JSON, figures, split manifests, error CSVs | **GitHub** — committed | `git clone` |
| The trained LoRA adapter (166 MB) | **HF Hub** — `vireshk/qwen3-8b-medmcqa-qlora` | `huggingface-cli download` |
| Model card | Same Hub repo, as `README.md`; source of truth is `docs/model_card.md` | in git |
| Demo Space | **HF Hub** — `vireshk/fine-tune-vs-rag` | in git under `app/` |
| Base model, embedder | **HF Hub**, public, pinned to commit SHAs in `configs/model/*.yaml` | re-downloaded on first run |
| MedMCQA / MIRIAD | **HF Hub**, public; dataset revision pinned in `results/split_manifest.json` | re-downloaded on first run |
| FAISS indices (~6 GB) | **Not stored** — deliberately | `make index CORPUS=parity` rebuilds them (~11 min) |
| Training checkpoints, optimizer state | **Not stored** — deliberately | not needed; the adapter is the artifact |

Two things are intentionally *not* preserved. Optimizer state only matters to a
resumed run, and the indices are cheaper to rebuild than to host — the build is
deterministic given the pinned embedder revision, and `make index` writes a
`build_stats.json` whose hash can be compared against the committed one.

## 2. Cold start on a new machine

```bash
git clone https://github.com/vireshkoli/Fine-Tune-vs-RAG && cd Fine-Tune-vs-RAG
uv sync --group dev          # CPU only
make check                   # ruff + mypy(strict) + the full test suite
make verify-splits           # proves the frozen test set still hashes identically
make report                  # regenerates every table and figure from committed JSON
```

That much requires **no GPU and no secrets**, and it reproduces every number in
the README. If `make report` produces a diff against `results/tables.md`, the
committed results and the committed code have diverged — investigate before
trusting anything downstream.

With a GPU:

```bash
cp .env.example .env         # then paste an HF token (write scope, for pushing)
uv sync --group dev --extra gpu
make setup-check             # GPUs, free disk, cache isolation, token scope
make index CORPUS=parity     # ~11 min
make eval ARM=base           # ~8 min
```

## 3. Things that will bite you

**Pin the GPU before importing torch.** HF's `Trainer` places the model on
`cuda:0` regardless of `device_map`, so `scripts/04_train.py` sets
`CUDA_VISIBLE_DEVICES` *before* any torch import. On a single-GPU machine this
is invisible; on a shared box it is the difference between a clean run and
OOM-ing somebody else's job.

**`bootstrap_env()` must run before `huggingface_hub` is imported.**
`huggingface_hub` freezes its cache path at import time. `src/fvr/__init__.py`
calls `bootstrap_env()` at package import for exactly this reason — moving that
call leaks the entire model cache into `~/.cache/huggingface`. There is a
subprocess regression test that reproduces the bad import order; do not delete
it.

**torch comes from the cu128 index.** The lab driver was 570.133.07 (CUDA
12.8) and PyPI's default wheel is built against CUDA 13.0. On a newer driver you
can drop the `[[tool.uv.index]]` block in `pyproject.toml`; on a 12.x driver you
need it.

**`make check` can pass while CI fails.** CI installs the CPU dependency set
only, which changes what mypy sees (absent `transformers` makes
`TrainerCallback` resolve to `Any`). Use **`make check-ci`**, which reproduces
CI in a throwaway CPU-only venv. This bit twice.

**Merge the adapter before timing it.** An unmerged PEFT wrapper runs ~4.9x
slower on identical prompts. `attach_adapter()` merges by default and refuses to
merge into a quantised base; a merged model also bypasses the loader cache,
because merging mutates weights in place and a later `base` run would otherwise
silently receive fine-tuned weights.

**Batch size is not inert.** It changes padding, and in bf16 that moves a
couple of borderline items (0.566 at 16 vs 0.568 at 8). It is pinned in
`configs/base.yaml` and every arm must use the same value.

## 4. Reproducing the fine-tune from scratch

```bash
make train-estimate CONFIG=configs/train/qlora_r16.yaml   # times a few steps, projects total
make train CONFIG=configs/train/qlora_r16.yaml            # ~5.1 GPU-hours on one A40
```

Resumption is built in, not bolted on: kill the run and re-issue the same
command — it resumes from the latest checkpoint and the loss curve continues.
Checkpoints are selected on the **validation** split; a test asserts the trainer
never loads the test split.

## 5. Publishing

```bash
make push-dry     # prints the exact upload manifest, uploads nothing
make push         # pushes the adapter and the Space
```

`make push-dry` is the default for a reason. Publishing is outward-facing, and
`src/fvr/ops/hub.py` enforces two things the reviewer should not have to
remember: the model card must carry the not-for-clinical-use statement (it
refuses otherwise), and optimizer state, RNG state and `training_args.bin` are
never uploaded. The card that ships is `docs/model_card.md`, uploaded *as*
`README.md` — the adapter directory's own `README.md` is a `trl`-generated stub
full of "[More Information Needed]" and is excluded by allowlist.

## 6. Teardown, if you are the one vacating a shared machine

```bash
make verify-recoverable   # gates the next step; fails if anything exists only locally
make teardown             # dry run: prints the byte-exact deletion manifest
make teardown EXECUTE=1   # reclaims ~33 GiB
du -sh ~/.cache/huggingface   # must be unchanged — proof nothing outside scope was touched
```

The only deletable root is `$PROJECT_ROOT/.artifacts`. `~/.cache/huggingface`
held **19 GB of a labmate's unrelated prior work** (Llama-3.1-8B, CLIP, dinov2,
InLegalBERT), so it is on an explicit denylist *and* its size is asserted
unchanged across the operation. The separation is structural rather than
procedural: `HF_HOME` points inside `.artifacts/` from the first phase, so this
project's downloads never entered the shared cache to begin with.

## 7. Known gaps, in priority order

1. **Seeds 2 and 3** (~10 GPU-hours). Everything is single-seed (42). The
   headline comparisons are paired within-seed, which is the stronger test, but
   training-seed variance is unmeasured.
2. **The ablation grid** (~25 GPU-hours): LoRA rank {8,16,32,64}, epochs, top-k,
   `bge` vs `MedEmbed`, bf16 vs NF4. The chosen config is defensible but not
   shown to be optimal.
3. **A size-matched external corpus.** The two indices differ 7.3x in size, so
   the parity-vs-external contrast conflates corpus *content* with corpus
   *size*. Subsampling `external` to 218k chunks separates them.
4. **Hand-annotation of `results/error_analysis/*_review.csv`.** The rows are
   stratified and pre-filled; the `human_label` and `notes` columns are blank.
5. **Live inference in the Space** (v0.1.1), once there is quota to test it
   against.
