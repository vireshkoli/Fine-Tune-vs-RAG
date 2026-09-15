---
title: Fine-Tune vs RAG — Live
emoji: ⚖️
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.27.0
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
suggested_hardware: zero-a10g
short_description: Ask a medical question; four benchmark arms answer live
models:
  - Qwen/Qwen3-8B
  - vireshk/qwen3-8b-medmcqa-qlora
  - BAAI/bge-large-en-v1.5
datasets:
  - vireshk/fine-tune-vs-rag-parity-index
---

# Fine-Tune vs RAG — live

The input→output companion to the
[Fine-Tune vs RAG benchmark](https://github.com/vireshkoli/Fine-Tune-vs-RAG):
type a medical question and four arms answer it with the benchmark's exact
prompts, retrieval settings and greedy decoding.

| Arm | What it is |
| --- | --- |
| base | Qwen3-8B as released, thinking off |
| base + RAG | + top-5 chunks from the parity index (139,856 MedMCQA explanations, bge-large-en-v1.5) |
| fine-tuned | + the QLoRA adapter trained on MedMCQA (`vireshk/qwen3-8b-medmcqa-qlora`) |
| fine-tuned + RAG | both |

Free text is always generated with the options hidden from the model *and*
from the retriever — the benchmark's free-text arm. Supply options and each
arm also picks a letter by constrained log-probability over A–D, which is how
the headline multiple-choice numbers were scored. The thing to watch for is
the report's §2.4 finding: the fine-tuned arm picks the right letter more
often than base and writes a worse answer.

`bench.py` is a vendored copy of the benchmark's prompt, retrieval and scoring
rules; a test in the repository asserts it matches the package byte for byte.

**Latency shown here is ZeroGPU with an unmerged adapter and is not the
report's.** The benchmark merged the adapter and timed on an exclusive A40.

> **Not for clinical use. Not medical advice. Not a medical device.** Every
> arm can be confidently wrong; the base model's free-text score against the
> gold answer was 0.45 on a 0–1 scale.

Licences: Qwen3-8B Apache 2.0; MedMCQA Apache 2.0; bge-large-en-v1.5 MIT.
