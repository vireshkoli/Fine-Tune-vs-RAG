---
title: Fine-Tune vs Retrieve
emoji: ⚖️
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
short_description: When is fine-tuning worth it versus just retrieving?
---

# Fine-Tune vs. Retrieve

A controlled six-arm benchmark in clinical multiple-choice QA, asking one
question: **for a given domain, when is fine-tuning worth it versus just
retrieving?**

This Space is **precomputed**. Every answer it shows was produced by a real
evaluated run over a frozen, SHA-256-pinned 1,000-item test set and is read
from `precomputed/responses.json` — the same committed JSON that generates the
repository's README tables. It therefore needs no GPU quota, never queues, and
cannot drift from the published numbers.

- **Source and full methodology:** https://github.com/vireshkoli/Fine-Tune-vs-RAG
- **Adapter:** https://huggingface.co/vireshk/qwen3-8b-medmcqa-qlora

> ⚠️ **Not for clinical use. Not medical advice. Not validated on real patients.**

`LIVE_INFERENCE_ENABLED` is read at start-up and defaults to `false`. Live
free-text inference lands in v0.1.1; shipping an untested GPU path here would
be a claim this Space cannot back up.
