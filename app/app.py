"""Fine-Tune vs. Retrieve — the demo Space.

Precomputed-first by design. Every answer shown here comes from
``precomputed/responses.json``, written from the same committed run JSONs that
produce the README tables, so the demo works with zero GPU quota, costs nothing
to host, and cannot drift from the published numbers.

``LIVE_INFERENCE_ENABLED`` is read at start-up and is off by default. Live
free-text inference lands in v0.1.1, when there is quota to test it against;
shipping an untested GPU path here would be a claim this Space cannot back up.
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from demo import (
    ANSWER_TABLE_HEADERS,
    ARMS_TABLE_HEADERS,
    BUCKET_ORDER,
    COST_TABLE_HEADERS,
    arms_table,
    cost_summary,
    cost_table,
    find_item,
    item_choices,
    load_payload,
    provenance_note,
    render_answers,
    render_question,
)

HERE = Path(__file__).parent
PAYLOAD = load_payload()
LIVE_INFERENCE_ENABLED = os.environ.get("LIVE_INFERENCE_ENABLED", "false").lower() == "true"

HEADLINE = """
# Fine-Tune vs. Retrieve

### Retrieval beat fine-tuning on identical information — and the same retrieval over a different corpus was worth nothing.

A controlled six-arm benchmark in clinical multiple-choice QA. One base model,
one frozen 1,000-item test set, one prompt, one GPU.

> ⚠️ **Not for clinical use. Not medical advice. Not validated on real patients.**
> This is a measurement artifact about *fine-tuning versus retrieval*, not a medical tool.
"""

FINDINGS = """
**1. On identical information, the index beat the weights.** `rag-parity` retrieves the
same MedMCQA explanations the fine-tune trained on — same base model, same prompt, same
facts. Retrieval gained +10.2 points, fine-tuning +6.1, and the gap between them is
significant (p = 0.016).

**2. Retrieval over the wrong corpus was worth nothing.** `rag-external` reads 1.6M chunks
of peer-reviewed literature and scores **+0.001 against base** (p = 1.000) — a true null.
Yet its retrieval *works*: it surfaces the gold answer in 45.4% of items. Lexical presence
of an answer is not usable evidence. **The corpus mattered more than the technique.**

**3. Fine-tuning repays its training cost at ~200,000 queries.** A *merged* adapter has
identical inference cost to the base, so its advantage is purely prompt length: 113 tokens
versus 738. See the Cost tab.
"""

METHOD = """
## How the arms are kept comparable

Asserted in tests in the source repository, not promised in prose:

- **One `from_pretrained`** in the codebase; arms share the same model object.
- **One prompt builder** — stripping a RAG prompt's context must reproduce the non-RAG
  prompt byte for byte, and the *training* prompt is built by the same function.
- **Constrained A/B/C/D log-prob scoring**, never free generation — otherwise fine-tuned
  arms gain from learned formatting rather than knowledge.
- **Shared 3,000-character context budget** across RAG arms, so corpus quality is not
  confounded with context length.
- **Indices built from train-side text only**, gated on content hash.
- **Pinned model revisions**, fixed batch size, one frozen SHA-256-committed test set.

## Honest limitations

- **Single seed.** All results are seed 42. Headline comparisons are paired within-seed —
  the stronger test — but training-seed variance is unmeasured.
- **Contamination is present but measured.** ~9% of test stems are reproduced verbatim
  above a shuffled-reference chance baseline. But shuffling the answer options changes the
  base model's score by only -1.6 points, so it is not relying on memorised labels, and
  contamination is roughly constant across arms — the between-arm comparisons are largely
  unaffected. The fine-tune did pick up mild positional sensitivity (+3.2) the base lacks.
- **31.3% of the evaluation pool is dentistry**, the subject where retrieval helps least.
- **The two indices differ 7.3x in size**, so parity-vs-external conflates corpus content
  with corpus size.
- **No free-text evaluation.** All results are 4-option MCQ.
- **This demo shows no retrieved passages.** The committed run JSONs store predictions and
  token counts, not the retrieved text; showing it would mean re-running retrieval.

Full methodology: [the repository](https://github.com/vireshkoli/Fine-Tune-vs-RAG).
"""


def on_bucket_change(bucket: str) -> gr.Dropdown:
    choices = item_choices(PAYLOAD, bucket)
    return gr.Dropdown(choices=choices, value=choices[0][1] if choices else None)


def on_item_change(item_id: str | None) -> tuple[str, list[list[str]]]:
    item = find_item(PAYLOAD, item_id) if item_id else None
    if item is None:
        return "*Pick an item.*", []
    return render_question(item), render_answers(item, PAYLOAD)


def on_volume_change(exponent: float) -> tuple[list[list[str]], str]:
    volume = round(10**exponent)
    return cost_table(PAYLOAD, volume), cost_summary(PAYLOAD, volume)


with gr.Blocks(title="Fine-Tune vs. Retrieve", theme=gr.themes.Soft()) as app:
    gr.Markdown(HEADLINE)

    with gr.Tab("The finding"):
        gr.Markdown(FINDINGS)
        gr.Dataframe(
            value=arms_table(PAYLOAD),
            headers=ARMS_TABLE_HEADERS,
            interactive=False,
            wrap=True,
        )
        with gr.Row():
            gr.Image(
                str(HERE / "assets" / "arms.png"),
                label="Accuracy, p95 latency and cost",
                show_label=True,
            )
            gr.Image(str(HERE / "assets" / "cost-crossover.png"), label="Cost against query volume")
        gr.Markdown(provenance_note(PAYLOAD))

    with gr.Tab("Browse the benchmark"):
        gr.Markdown(
            "60 items, stratified by **which approach got them right** rather than sampled "
            "at random — a random draw is mostly items where every arm agrees, which shows "
            "nothing. Start with *index_only* and *weights_only*: those are the cases the "
            "whole benchmark exists to separate."
        )
        bucket = gr.Dropdown(
            choices=[("all buckets", "all"), *[(b, b) for b in BUCKET_ORDER]],
            value=BUCKET_ORDER[0],
            label="Disagreement pattern",
        )
        initial = item_choices(PAYLOAD, BUCKET_ORDER[0])
        item = gr.Dropdown(
            choices=initial,
            value=initial[0][1] if initial else None,
            label="Question",
        )
        question_md = gr.Markdown()
        answers_df = gr.Dataframe(headers=ANSWER_TABLE_HEADERS, interactive=False, wrap=True)

        bucket.change(on_bucket_change, inputs=bucket, outputs=item)
        item.change(on_item_change, inputs=item, outputs=[question_md, answers_df])
        app.load(on_item_change, inputs=item, outputs=[question_md, answers_df])

    with gr.Tab("Cost"):
        gr.Markdown(
            "Cost per query is meaningless without a volume assumption, so it is swept "
            "rather than asserted. Each arm costs "
            "`fixed_cost / volume + marginal_cost`; fine-tuning is a large one-off with "
            "cheap short-prompt inference, retrieval is nearly free up front but inflates "
            "prefill tokens on every query, forever. **The crossover is the deliverable.**"
        )
        volume_slider = gr.Slider(
            minimum=2,
            maximum=8,
            step=0.25,
            value=5,
            label="Lifetime query volume (log₁₀)",
            info="5 = 100,000 queries",
        )
        cost_df = gr.Dataframe(headers=COST_TABLE_HEADERS, interactive=False, wrap=True)
        cost_md = gr.Markdown()
        volume_slider.change(on_volume_change, inputs=volume_slider, outputs=[cost_df, cost_md])
        app.load(on_volume_change, inputs=volume_slider, outputs=[cost_df, cost_md])

    with gr.Tab("Method & limitations"):
        gr.Markdown(METHOD)

    if not LIVE_INFERENCE_ENABLED:
        gr.Markdown(
            "---\n*Live inference is disabled. Every result here is precomputed from the "
            "committed benchmark runs, which is why this Space needs no GPU quota and "
            "never shows you a queue.*"
        )

if __name__ == "__main__":
    app.launch()
