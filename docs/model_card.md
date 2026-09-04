---
license: apache-2.0
base_model: Qwen/Qwen3-8B
library_name: peft
tags:
  - medical
  - question-answering
  - qlora
  - lora
  - benchmark
datasets:
  - openlifescienceai/medmcqa
language:
  - en
pipeline_tag: text-generation
---

# Qwen3-8B QLoRA — MedMCQA (research artifact)

> ## ⚠️ Not for clinical use
>
> **This adapter is not a medical device and must not be used for medical
> advice, diagnosis, or treatment.** It has never been validated on real
> patients, real clinical notes, or any prospective data. It was trained on
> multiple-choice exam questions and is evaluated only on multiple-choice exam
> questions.
>
> It is confidently wrong a great deal of the time. In the benchmark below,
> **86% of the base model's errors were high-confidence errors** — the model
> does not hedge when it is mistaken. Any clinical deployment would require
> validation this artifact does not have and was never intended to support.
>
> It exists to answer a research question — *when is fine-tuning worth it versus
> just retrieving?* — and its value is the comparison, not the checkpoint.

## Model summary

A LoRA adapter for `Qwen/Qwen3-8B`, trained with QLoRA (4-bit NF4 base) on
MedMCQA training questions. Released as **one arm of a controlled six-arm
benchmark**, not as a general medical assistant.

| | |
| --- | --- |
| Base model | [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) @ `b968826d9c46dd6066d109eabc6255188de91218` |
| Method | QLoRA — LoRA r=16, α=32, dropout 0.05, over 4-bit NF4 |
| Target modules | `q,k,v,o,gate,up,down_proj` (attention **and** MLP) |
| Trainable params | 43,646,976 of 4,761,498,624 (**0.92%**) |
| Training data | 30,000 MedMCQA train rows (Apache 2.0) |
| Compute | 1,875 steps, 1 epoch, **5h04m on one NVIDIA A40** |
| Final loss | train 1.287, best eval 1.201 (on a held-out validation split) |
| Precision | trained in 4-bit NF4; **evaluated merged into bf16** |

## Intended use

**In scope.** Reproducing the benchmark in
[the accompanying repository](https://github.com/vireshkoli/Fine-Tune-vs-RAG);
research on fine-tuning versus retrieval trade-offs; a baseline for further
medical-QA experiments.

**Out of scope.** Any clinical, diagnostic, triage, or treatment context.
Patient-facing applications. Generating medical content for consumption without
expert review. Any use where being confidently wrong carries a cost.

## Results

Scored on a frozen, subject-stratified 1,000-item test set carved from MedMCQA's
labelled validation split. (MedMCQA's own `test` split has `cop == -1` —
labels are withheld — so it cannot be used for evaluation.)

| Arm | Accuracy | 95% CI | p50 latency | Prompt tokens |
| --- | ---: | :---: | ---: | ---: |
| `base` — Qwen3-8B zero-shot | 56.8% | [53.8, 59.9] | 103 ms | 113 |
| **`qlora` — this adapter** | **62.9%** | [60.0, 65.9] | 102 ms | 113 |
| `rag-parity` — retrieval, no fine-tune | 67.0% | [64.1, 69.9] | 171 ms | 656 |

**+6.1 points over base**, paired McNemar p < 0.0001 over identical items.

**The headline finding is not flattering to this adapter.** Retrieving the same
training explanations at inference time scores **higher** (67.0%) than absorbing
them into these weights (62.9%), and the difference is significant (p = 0.016).
Same base model, same prompt, same information — one in an index, one in the
weights, and the index wins.

The adapter's advantage is cost, not accuracy: merged, it has identical
inference latency to the base model while serving **113-token prompts instead of
656**. It repays its training cost at roughly **200,000 queries**.

## Training data

**[MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa)**, Apache
2.0 — Indian medical entrance exam questions (AIIMS/NEET-PG). No PHI: exam
questions with de-identified vignettes.

Preprocessing, all documented and reproducible:

- **OCR repair.** MedMCQA's text has systematic extractor corruption in which
  the `rt` bigram is dropped, and *the corrupted spelling is more common than
  the correct one* (`aery` 11,203 vs `artery` 7,009; `hea` 7,626 vs `heart`
  4,957). Repaired with a 225-entry human-reviewed lexicon covering 93,174
  tokens.
- **Answer-key stripping.** Many explanations open with the answer letter
  ("Ans. is 'd' i.e., ..."). Training on those teaches the letter, not the
  medicine, so the boilerplate is removed and explanations reduced to nothing
  are dropped.
- **Leakage gate.** Rows are filtered on content hash, not id — MedMCQA repeats
  items across splits under different ids. **0 leaked rows** reached training.
- **Subsampling.** A seeded random 30,000 of the ~179,600 clean rows.

Targets are the answer letter followed by the explanation. The letter comes
first so that first-token scoring grades this arm identically to every other
arm; the explanation is included so the fine-tune sees the same text the
retrieval arm serves.

## Limitations

1. **Contamination is unquantified.** MedMCQA predates Qwen3 and is a widely
   mirrored public benchmark, so assuming it appears in pretraining is the safe
   prior. Probes are implemented but **not yet run**. Until they are, some
   portion of the accuracy above may be recall rather than reasoning. This is
   the largest known threat to these numbers.
2. **Single seed.** All results are seed 42. Training-seed variance is
   unmeasured.
3. **Narrow evaluation.** 4-option multiple choice only, scored by constrained
   log-probability. No free-text generation, no citation, no calibration
   assessment beyond confidence margins.
4. **31.3% of the evaluation pool is dentistry**, not general medicine. Results
   excluding it are meaningfully different (base 59.8%, this adapter 66.5%).
5. **English only**, and the questions reflect the Indian medical curriculum.
6. **Merged evaluation.** Latency figures assume the adapter is merged into the
   base. Served unmerged it is ~4.9× slower.

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen3-8B"
REV = "b968826d9c46dd6066d109eabc6255188de91218"  # pragma: allowlist secret

tok = AutoTokenizer.from_pretrained(BASE, revision=REV)
model = AutoModelForCausalLM.from_pretrained(BASE, revision=REV, dtype="bfloat16")
model = PeftModel.from_pretrained(model, "<this-repo>").merge_and_unload()
```

The benchmark pins `enable_thinking=False` in the chat template for every arm,
because variable-length reasoning traces confound latency and cost measurement.
Reproducing the numbers above requires the same setting.

## Licences

- **This adapter:** Apache 2.0, inherited from the Qwen3-8B base.
- **Training data:** MedMCQA, Apache 2.0.
- **Benchmark code:** MIT.

MedQA was deliberately not used anywhere in this project: `bigbio/med_qa`
declares its licence "unknown", and a re-uploader's `cc-by-4.0` tag does not
launder upstream copyright on USMLE board-prep material.

## Citation

```bibtex
@software{koli_finetune_vs_rag_2026,
  author = {Koli, Viresh},
  title  = {Fine-Tune vs. Retrieve: A Controlled Benchmark in Clinical QA},
  year   = {2026},
  url    = {https://github.com/vireshkoli/Fine-Tune-vs-RAG}
}
```
