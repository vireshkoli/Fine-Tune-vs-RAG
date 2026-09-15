"""Fine-Tune vs RAG — live demo.

Type a medical question; four arms answer it with the benchmark's exact
prompts, retrieval settings and decoding policy:

    base            Qwen3-8B, as released
    base + RAG      + top-5 passages from the parity index (MedMCQA explanations)
    fine-tuned      + the QLoRA adapter trained on MedMCQA
    fine-tuned + RAG

Free text is always generated (options hidden from the model and from the
retriever, as in the benchmark's free-text arm). If you also supply options,
each arm additionally picks a letter by constrained log-probability, which is
how the headline MCQ numbers were scored — so you can watch an arm pick the
right letter and write the wrong answer, which is the report's §2.4 finding.

Not for clinical use. Not medical advice.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

try:
    import spaces  # ZeroGPU: must be imported before torch
except ImportError:  # local smoke test on a real GPU

    class _Spaces:
        @staticmethod
        def GPU(*_args, **_kwargs):  # type: ignore[no-untyped-def]  # noqa: N802 - spaces' name
            def wrap(fn):  # type: ignore[no-untyped-def]
                return fn

            return wrap

    spaces = _Spaces()  # type: ignore[assignment]

import bench
import faiss
import gradio as gr
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_REPO = "Qwen/Qwen3-8B"
BASE_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"  # pragma: allowlist secret
ADAPTER_REPO = "vireshk/qwen3-8b-medmcqa-qlora"
EMBEDDER_REPO = "BAAI/bge-large-en-v1.5"
EMBEDDER_REVISION = "d4aa6901d3a41ba39fb536a557fa166f842b0e09"  # pragma: allowlist secret
INDEX_REPO = "vireshk/fine-tune-vs-rag-parity-index"
REPORT_URL = "https://github.com/vireshkoli/Fine-Tune-vs-RAG/blob/main/REPORT.md"

ARMS = (
    ("base", False, False),
    ("base + RAG", False, True),
    ("fine-tuned", True, False),
    ("fine-tuned + RAG", True, True),
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------- loading

tokenizer = AutoTokenizer.from_pretrained(BASE_REPO, revision=BASE_REVISION)
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

_base = AutoModelForCausalLM.from_pretrained(
    BASE_REPO, revision=BASE_REVISION, dtype=torch.bfloat16
)
# Unmerged on purpose: one copy of the weights serves both the base and the
# fine-tuned arms through disable_adapter(). The benchmark merged the adapter
# before timing it, so the latencies shown here are not the report's.
# torch_device="cpu": on ZeroGPU the adapter weights must not be loaded onto
# CUDA at import; the single .to() below is the sanctioned move.
model = PeftModel.from_pretrained(_base, ADAPTER_REPO, torch_device="cpu")
model.eval()
model.to(DEVICE)
OPTION_IDS = bench.option_token_ids(tokenizer)

embedder = SentenceTransformer(EMBEDDER_REPO, revision=EMBEDDER_REVISION, device="cpu")
embedder.max_seq_length = 512


def _download(name: str) -> Path:
    # A local copy of the index short-circuits the Hub download during the
    # smoke test on the training machine; the Space never sets this.
    local = os.environ.get("FVR_INDEX_DIR")
    if local:
        return Path(local) / name
    return Path(hf_hub_download(INDEX_REPO, name, repo_type="dataset"))


index = faiss.read_index(str(_download("index.faiss")))
passages: list[bench.Passage] = []
with _download("passages.jsonl").open(encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        passages.append(bench.Passage(id=row["id"], text=row["text"]))
meta = json.loads(_download("meta.json").read_text(encoding="utf-8"))
assert len(passages) == int(meta["n"]), "index and passages disagree"


# ---------------------------------------------------------------- inference


def retrieve(query: str) -> list[bench.Passage]:
    vector = embedder.encode(
        [bench.QUERY_PREFIX + query], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)
    scores, ids = index.search(np.ascontiguousarray(vector), bench.TOP_K)
    hits = [
        bench.Passage(id=passages[i].id, text=passages[i].text, score=float(s))
        for s, i in zip(scores[0], ids[0], strict=True)
        if i >= 0
    ]
    return bench.apply_context_budget(hits)


def render(prompt: bench.BuiltPrompt) -> str:
    return str(
        tokenizer.apply_chat_template(
            prompt.as_messages(),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )


def _generate(prompt: bench.BuiltPrompt, max_new_tokens: int) -> tuple[str, int, int]:
    encoded = tokenizer(render(prompt), return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id,
        )
    completion = output[0, encoded["input_ids"].shape[1] :]
    text = tokenizer.decode(completion, skip_special_tokens=True).strip()
    return text, int(encoded["input_ids"].shape[1]), int(completion.numel())


def _score(prompt: bench.BuiltPrompt) -> tuple[float, ...]:
    encoded = tokenizer(render(prompt), return_tensors="pt").to(model.device)
    with torch.inference_mode():
        logits = model(**encoded).logits[0, -1, :]
    return bench.option_logprobs(logits, OPTION_IDS)


def parse_options(text: str) -> list[str]:
    """One option per line; a leading 'A.' / 'B)' label is stripped."""
    options: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if len(line) > 2 and line[0].upper() in "ABCD" and line[1] in ".):":
            line = line[2:].strip()
        options.append(line)
    return options[:4]


@spaces.GPU(duration=120)
def answer(question: str, options_text: str, max_new_tokens: int) -> list[str]:
    question = (question or "").strip()
    if not question:
        return ["Enter a question."] * 4 + ["", ""]
    options = parse_options(options_text or "")
    mcq = len(options) >= 2
    max_new_tokens = int(max_new_tokens)

    # Free-text arms retrieve on the stem alone; MCQ arms may use the options.
    stem_hits = retrieve(bench.retrieval_query(question, options, with_options=False))
    mcq_hits = retrieve(bench.retrieval_query(question, options, with_options=True)) if mcq else []

    panels: list[str] = []
    for name, adapter, rag in ARMS:
        ctx = torch.no_grad() if adapter else model.disable_adapter()
        with ctx:
            t0 = time.perf_counter()
            text, p_tok, c_tok = _generate(
                bench.build_freetext_prompt(question, stem_hits if rag else ()), max_new_tokens
            )
            gen_s = time.perf_counter() - t0
            pick = ""
            if mcq:
                t0 = time.perf_counter()
                logprobs = _score(bench.build_prompt(question, options, mcq_hits if rag else ()))
                score_s = time.perf_counter() - t0
                best = max(range(len(options)), key=lambda i: logprobs[i])
                ordered = sorted(logprobs[: len(options)], reverse=True)
                margin = ordered[0] - ordered[1]
                pick = (
                    f"\n\n**MCQ pick: {bench.OPTION_LABELS[best]}. {options[best]}**  "
                    f"(margin {margin:.2f} nats, {score_s * 1000:.0f} ms)"
                )
        panels.append(
            f"### {name}\n\n{text or '_(empty)_'}{pick}\n\n"
            f"<sub>{c_tok} tokens generated from a {p_tok}-token prompt in {gen_s:.1f} s"
            f"{' · ' + str(len(stem_hits if rag else [])) + ' passages in context' if rag else ''}</sub>"
        )

    shown = "\n\n".join(
        f"**[{i}]** <sub>score {p.score:.3f}</sub>\n\n{p.text}" for i, p in enumerate(stem_hits, 1)
    )
    passages_md = (
        f"Top-{bench.TOP_K} over {len(passages):,} MedMCQA explanation chunks, "
        f"{bench.MAX_CONTEXT_CHARS:,}-character budget, query = question stem only "
        f"(the free-text arms never see the options).\n\n{shown}"
    )
    if mcq:
        passages_md += (
            "\n\n<sub>The MCQ picks used a second retrieval over stem + options, as the "
            "benchmark's MCQ arms do; those passages are not shown.</sub>"
        )
    status = (
        f"Free text at {max_new_tokens} tokens max, greedy. "
        + ("MCQ picks by constrained log-probability over A-D. " if mcq else "")
        + "Latency is ZeroGPU with an unmerged adapter — see the report for A40 numbers."
    )
    return [*panels, passages_md, status]


# ---------------------------------------------------------------- UI

EXAMPLES = [
    [
        "Preferred drug for the treatment of uncomplicated grade 2 hypertension in a "
        "48 year old man is",
        "Chlorthalidone\nTriamterene\nSpironolactone\nFurosemide",
    ],
    [
        "Which of the heart valves is most likely to be involved by infective "
        "endocarditis following a septic abortion?",
        "Aortic valve\nTricuspid valve\nPulmonary valve\nMitral valve",
    ],
    [
        "On applying pressure on the angle of the jaw while maintaining a patent "
        "airway, which nerve is likely to be damaged?",
        "6th\n7th\n4th\n9th",
    ],
    ["What is the mechanism of action of metformin?", ""],
]

INTRO = f"""
# Fine-Tune vs RAG — live

One base model, one prompt, one retrieval setting — the same ones the
[benchmark]({REPORT_URL}) used. Ask a medical question and watch the four arms answer it.
Add options to also see each arm's multiple-choice pick.

The report's finding to look for: **the fine-tuned arm picks the right letter more often
than base, and writes a worse answer** — it learned to rank candidates, not the facts
to produce one. Retrieval over the right corpus does both.

> **Not for clinical use. Not medical advice.** Every arm here can be confidently wrong.
"""

with gr.Blocks(title="Fine-Tune vs RAG — live") as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        with gr.Column(scale=3):
            question = gr.Textbox(label="Question", lines=3, placeholder="A clinical question…")
            options = gr.Textbox(
                label="Options (optional, one per line, up to four)",
                lines=4,
                placeholder="A. …\nB. …\nC. …\nD. …",
            )
        with gr.Column(scale=1):
            max_new = gr.Slider(
                32, 256, value=bench.MAX_NEW_TOKENS, step=16, label="Max new tokens"
            )
            run = gr.Button("Ask all four arms", variant="primary")
    gr.Examples(EXAMPLES, inputs=[question, options], label="Try one from the test set")
    status = gr.Markdown()
    with gr.Row():
        out_base = gr.Markdown()
        out_base_rag = gr.Markdown()
    with gr.Row():
        out_ft = gr.Markdown()
        out_ft_rag = gr.Markdown()
    with gr.Accordion("Retrieved passages (free-text arms)", open=False):
        out_passages = gr.Markdown()
    gr.Markdown(
        f"Adapter: [`{ADAPTER_REPO}`](https://huggingface.co/{ADAPTER_REPO}) · "
        f"index: [`{INDEX_REPO}`](https://huggingface.co/datasets/{INDEX_REPO}) · "
        f"[report]({REPORT_URL}) · "
        "[results explorer](https://huggingface.co/spaces/vireshk/fine-tune-vs-rag)"
    )
    run.click(
        answer,
        inputs=[question, options, max_new],
        outputs=[out_base, out_base_rag, out_ft, out_ft_rag, out_passages, status],
    )

if __name__ == "__main__":
    demo.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"))
