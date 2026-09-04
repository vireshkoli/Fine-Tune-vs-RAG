# Fine-Tune vs. Retrieve — Evaluation Report

**A controlled benchmark in clinical multiple-choice QA.** Six arms, one frozen
test set, one base model, one prompt. The question is not "does fine-tuning
work" but **when it is worth it versus just retrieving** — and the answer turns
out to depend far more on *what you retrieve over* than on which technique you
pick.

> **Not for clinical use. Not medical advice. Not validated on real patients.**
> Every number below is a benchmark score on exam questions, and MedMCQA
> contains textual noise and answer-key artefacts documented in §2.

---

## 1. Headline results

All six arms scored on the same frozen 1,000-item test set, timed back to back
on an exclusive NVIDIA A40.

| Arm | Accuracy | 95% CI | p50 | p95 | Prompt tokens | Retrieval hit | Grounded |
| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| `qlora-rag-parity` ◆ | **71.1%** | [68.3, 73.9] | 171 ms | 193 ms | 656 | 0.570 | 0.723 |
| `rag-parity` ◆ | **67.0%** | [64.1, 69.9] | 171 ms | 193 ms | 656 | 0.570 | 0.725 |
| `qlora` | **62.9%** | [60.0, 65.9] | 102 ms | 112 ms | 113 | — | — |
| `qlora-rag` | 61.4% | [58.4, 64.4] | 179 ms | 195 ms | 738 | 0.454 | 0.628 |
| `base` | 56.8% | [53.8, 59.9] | 103 ms | 118 ms | 113 | — | — |
| `rag-external` | 56.7% | [53.6, 59.8] | 179 ms | 195 ms | 738 | 0.454 | 0.635 |

◆ = diagnostic arm, not part of the four-arm headline comparison. See §3.

Paired McNemar over identical items, which is the correct test here because
every arm sees the same questions:

| Comparison | Δ accuracy | Discordant (A/B) | p | Verdict |
| --- | ---: | :---: | ---: | --- |
| `rag-parity` vs `base` | **+0.102** | 190/88 | <0.0001 | **significant** |
| `qlora` vs `base` | **+0.061** | 119/58 | <0.0001 | **significant** |
| `rag-parity` vs `qlora` | **+0.041** | 158/117 | 0.0159 | **significant** |
| `rag-external` vs `base` | +0.001 | 121/120 | 1.0000 | **not significant** |
| `qlora-rag` vs `qlora` | −0.015 | 97/112 | 0.3328 | not significant |
| `qlora-rag-parity` vs `rag-parity` | +0.041 | 98/57 | 0.0013 | **significant** |

**Minimum detectable effect at n=1,000 is 0.063.** Any gap smaller than that,
reported without a paired test, would be noise. This is why the table above
carries discordant counts rather than only p-values — see §6.

---

## 2. Three findings

### 2.1 Retrieval beats absorption, on identical information

`rag-parity` retrieves the *same* MedMCQA training explanations that the QLoRA
arm was trained on. Same base weights, same prompt, same information — the only
difference is whether that information lives in an index or in the weights.

**The index wins: +10.2 points versus +6.1 (p = 0.016 between them).**

This is the comparison the parity corpus exists to make, and a conventional
four-arm design cannot produce it. With only `base`, `rag`, `qlora` and
`qlora+rag` you can say "retrieval helped more than fine-tuning here", but you
cannot separate *the technique* from *the corpus*, because the RAG arm would be
reading a different body of text from the one the fine-tune saw.

### 2.2 Retrieval over the wrong corpus is worth exactly nothing

`rag-external` retrieves over 1,598,753 chunks of peer-reviewed biomedical
literature (MIRIAD). Against `base` it scores **+0.001, p = 1.000, with 121
discordant items each way.**

That balance is what makes this a *true* null rather than an underpowered one: a
small real effect would show lopsided discordance. Here retrieval changes 241
answers and improves the score by one item.

It is not that retrieval failed to run. The groundedness check shows the
external arm surfaces the gold answer text in **45.4%** of items — retrieval is
working as retrieval. **Lexical presence of the answer simply is not usable
evidence for the model.** Accuracy alone could not tell that apart from a broken
retriever; that is the entire reason groundedness is measured.

**So the corpus matters more than the technique.** Same model, same retriever,
same context budget, same prompt: one corpus is worth ten points and the other
is worth zero.

### 2.3 Fine-tuning and retrieval compose — but only over a corpus that works

`qlora-rag-parity` (71.1%) beats `rag-parity` (67.0%) by +4.1 points
(p = 0.0013), so the two techniques are not redundant when retrieval is useful.

Over the external corpus they do not compose: `qlora-rag` (61.4%) is *below*
`qlora` alone (62.9%), though not significantly (p = 0.33). Injecting 625 tokens
of irrelevant context is at best free and at worst mildly harmful.

---

## 3. What the arms are, and why six

| Arm | Weights | Retrieval corpus | Role |
| --- | --- | --- | --- |
| `base` | frozen | — | the floor |
| `rag-external` | frozen | MIRIAD literature, 1.6M chunks | realistic "we have a domain corpus" |
| `qlora` | fine-tuned | — | the fine-tune |
| `qlora-rag` | fine-tuned | MIRIAD literature | the combination most write-ups skip |
| `rag-parity` ◆ | frozen | MedMCQA train explanations, 218k chunks | **diagnostic** |
| `qlora-rag-parity` ◆ | fine-tuned | MedMCQA train explanations | **diagnostic** |

The parity arms retrieve exactly the text the fine-tune trained on. Their whole
purpose is to hold information constant so that "weights versus index" is the
only variable. They are diagnostics rather than headline results because their
corpus is not something you would deploy — it is the training set.

### Fairness controls

These are enforced in code and asserted in tests, not promised in prose:

- **One `from_pretrained`.** `fvr/models/loader.py` is the only place weights are
  loaded, with a cache keyed on repo+revision, so arms share the *same model
  object* rather than merely equal ones.
- **One prompt builder.** Stripping the context block from a RAG prompt must
  reproduce the non-RAG prompt byte for byte (`tests/test_prompts.py`). The
  training prompt is built by the same function (`tests/test_train.py`), so the
  fine-tuned arm gains nothing from prompt familiarity.
- **Constrained A/B/C/D log-prob scoring**, never free generation. Parsing
  generated answers would penalise arms for formatting, and fine-tuning teaches
  format as much as content — that alone would have flattered the QLoRA arms.
- **`enable_thinking=False`** pinned everywhere. Qwen3 is a hybrid reasoning
  model and variable-length traces would confound both latency and cost.
- **Shared 3,000-character context budget** across both RAG arms. Parity chunks
  average 403 characters and external chunks 496; a fixed top-k would hand one
  arm more prefill than the other and confound corpus quality with context
  length.
- **Both indices built from train-side text only**, gated on content hash. A
  test item's own explanation cannot enter an index under a different id.
- **Pinned model revisions**, never `main` — the lab GPU gets wiped, and a
  moving reference would silently resolve to different weights on a rebuild.
- **Fixed batch size.** Not inert: padding differs with batch size and in bf16
  that moves a couple of borderline items (0.566 at 16 versus 0.568 at 8).

---

## 4. Cost

One currency, one published rate card, applied identically to every arm. Local
GPU-seconds are never mixed with hosted API pricing.

Four components: measured inference GPU-seconds; amortised training GPU-seconds;
amortised index-build GPU-seconds; index-serving CPU time.

| Fixed cost | GPU-seconds |
| --- | ---: |
| QLoRA training (1,875 steps) | 18,214 |
| Parity index build (218k chunks) | 647 |
| External index build (1.6M chunks) | 3,027 |

**Crossover volumes** — the volume at which the fine-tune's one-off cost is
repaid by its cheaper per-query inference:

| | crossover |
| --- | ---: |
| `qlora` overtakes `rag-external` | **198,138 queries** |
| `qlora` overtakes `rag-parity` | **255,332 queries** |
| `qlora-rag-parity` overtakes `rag-external` | 1,979,418 queries |

This is the number that generalises past this dataset. An accuracy delta is
specific to MedMCQA; "fine-tuning repays its training cost at roughly 200k
queries, given a 5-hour training run and a 6× prompt-token reduction" is a
statement a team can apply to their own workload.

**A measurement error nearly inverted this.** Fine-tuning's cost advantage comes
entirely from prompt length — 113 tokens versus 738 — because a *merged* adapter
has identical inference cost to the base model (102 ms vs 103 ms, measured). Two
bugs each hid that:

1. The adapter was initially evaluated **unmerged**, costing 4.9× base latency
   on identical prompts. That is wrapper overhead, not knowledge; anyone
   deploying a fine-tune merges it.
2. Even merged, `qlora` first appeared 2.2× slower than `base` — impossible for
   an architecturally identical model. Re-measuring `base` in the same session
   gave 103 ms against 102 ms. The original 47 ms was session drift.

Uncorrected, the report would have claimed fine-tuning triples inference cost.
It does not change it at all. All six arms are now timed back to back on an
exclusive GPU.

---

## 5. The dataset is dirtier than its reputation

Findings verified against the live data, not inherited from the dataset card.

### 5.1 MedMCQA's test split has no labels

`cop == -1` throughout. The only labelled evaluation pool is the **4,183-row
validation split**. Work reporting "MedMCQA test accuracy" from the Hub is
reporting validation. Our test/val/reserve splits are carved from that pool and
the manifest says so.

### 5.2 Systematic text corruption

The PDF extractor drops the `rt` bigram, and **the corrupted spelling is more
common than the correct one**:

| corrupted | count | correct | count |
| --- | ---: | --- | ---: |
| `aery` | 11,203 | `artery` | 7,009 |
| `hea` | 7,626 | `heart` | 4,957 |
| `impoant` | 4,506 | `important` | 3,733 |
| `hypeension` | 4,060 | `hypertension` | 2,519 |

Repaired via a generated, human-reviewed lexicon of 225 entries covering 93,174
tokens, committed to git rather than applied as a runtime heuristic so the
mapping is auditable and identical on every machine. Two entries were rejected
by manual context review: `pos → ports` (it is `posterior` split by a stray
space) and `baer → barter` (BAER is Brainstem Auditory Evoked Response).

### 5.3 A third of "clinical QA" is dentistry

**31.3% of the labelled pool is Dental** — and it is the subject where retrieval
helps least (+0.010). Excluding it:

| Arm | All items (n=1,000) | Excluding Dental (n=687) |
| --- | ---: | ---: |
| `base` | 56.8% | 59.8% |
| `rag-external` | 56.7% | 60.4% |
| `qlora` | 62.9% | 66.5% |
| `qlora-rag` | 61.4% | 64.3% |
| `rag-parity` | 67.0% | **74.2%** |
| `qlora-rag-parity` | 71.1% | **76.4%** |

The overall figure **understates** the effect on genuinely medical subjects by
about four points. Both cuts are reported; showing only the flattering one would
be dishonest, and showing only the overall one would hide a real signal.

---

## 5.5 Contamination: measured, not assumed

MedMCQA predates Qwen3 and is a widely mirrored public benchmark, so the safe
prior is that it appears in pretraining. Three probes, on the frozen test set:

### Permutation sensitivity — the strongest probe

Re-score with the answer options shuffled. Same information, different labels. A
model reasoning over content is unaffected; a model that memorised "this item's
answer is C" degrades.

| Arm | Original | Permuted | Drop | Verdict |
| --- | ---: | ---: | ---: | --- |
| `base` | 0.568 | **0.584** | **−0.016** | no positional memorisation |
| `qlora` | 0.629 | 0.597 | **+0.032** | mild positional sensitivity |

**The base model does not rely on memorised answer positions at all** — shuffling
the options made it marginally *better*, well inside noise. Whatever the base
arm's 56.8% is made of, it is not label recall.

**Fine-tuning made it slightly worse.** `qlora` loses 3.2 points under
permutation, right at the boundary of the ±3-point noise band. Training on
30,000 MedMCQA items appears to have introduced a mild positional shortcut that
the base model did not have. This is a cost of fine-tuning that an accuracy
table alone would never show, and it slightly deflates the +6.1-point gain.

### Position bias

Gold answers are unevenly distributed (A 32.5%, B 25.2%, C 23.5%, D 18.8%) —
itself worth knowing about this benchmark. Predictions track it closely:

| Arm | A | B | C | D | Max excess over gold |
| --- | ---: | ---: | ---: | ---: | ---: |
| gold | 0.325 | 0.252 | 0.235 | 0.188 | — |
| `base` | 0.340 | 0.232 | 0.211 | 0.217 | 0.029 |
| `qlora` | 0.348 | 0.252 | 0.190 | 0.210 | 0.045 |

Neither arm is exploiting the label prior to any meaningful degree.

### Verbatim reproduction — with a chance baseline

Feed half a question stem and let the model continue, then measure content-word
overlap with the withheld half. **A bare overlap figure is uninterpretable**:
exam stems are formulaic ("A 45-year-old man presents with…"), so shared
phrasing alone produces overlap. Each continuation is therefore *also* scored
against a **different** item's remainder, and only the excess counts.

| Arm | High overlap | Shuffled-reference control | **Excess** |
| --- | ---: | ---: | ---: |
| `base` | 10.0% | 1.3% | **+8.7 pts** |
| `qlora` | 8.7% | 2.0% | **+6.7 pts** |

**This is real.** Roughly **9% of test stems are reproduced verbatim well above
chance**, which is direct evidence those items were seen in pretraining. The
control is what makes the claim defensible — without it, "10% overlap" could
have been entirely genre effect.

### What this means for the headline numbers

Taken together the probes say something more specific than "contaminated" or
"clean":

- **~9% of items show memorisation of the question text.**
- **But the model is not using memorised answer labels** — the permutation probe,
  which is the direct test of that, is flat for `base`.

So some absolute accuracy is inflated by recall of specific items, and every
arm is inflated by roughly the same amount because they share a base model and a
test set. **The comparisons between arms — which is what this study is about —
are largely unaffected**, because contamination is a constant across arms.

The exception is `qlora`, whose mild positional sensitivity is *caused by* the
fine-tuning rather than inherited. That is a genuine, if small, mark against the
fine-tuned arm, and it is reported here rather than omitted.

---

## 6. Statistical honesty

- **Bootstrap 95% CIs** (10,000 resamples) on every accuracy.
- **Paired McNemar** for arm comparisons. The arms see identical items, so
  pairing is both valid and far more powerful; an unpaired t-test would be the
  wrong tool.
- **Minimum detectable effect reported** (0.063 at n=1,000).
- **Discordant counts shown alongside p.** This matters more than it looks: the
  common rule "a 2-point gap on 1,000 items is noise" is only true when errors
  are *independent*. A consistently one-sided 1.5-point gap reaches p<1e-4 — far
  below the unpaired MDE. Both cases are encoded as tests.

### Known limitations

1. **Single seed.** Every result is seed 42. The headline comparisons are paired
   within-seed, which is the stronger test, but training-seed variance for the
   QLoRA arms is unmeasured. Two further seeds are budgeted at ~10 GPU-hours.
2. **Contamination is present but bounded** (§5.5). ~9% of test stems are
   reproduced verbatim above a chance baseline, so some absolute accuracy is
   recall. The permutation probe shows the base model does *not* use memorised
   answer positions, and contamination is roughly constant across arms, so the
   between-arm comparisons this study is about are largely unaffected. The
   fine-tuned arm did acquire mild positional sensitivity (+0.032) that the base
   model lacks.
3. **The parity and external indices differ 7.3× in size** (218k vs 1.6M
   chunks). The context budget equalises what reaches the model but not
   retrieval difficulty, so §2.2 conflates corpus *content* with corpus *size*.
   A size-matched external subsample would separate them.
4. **No free-text evaluation.** All results are 4-option MCQ scored by
   constrained log-prob. An LLM-judged free-text arm is designed but unrun.
5. **One epoch, one LoRA rank.** Hyperparameters were chosen from measurement
   (§7) rather than swept; the ablation grid is not yet run.

---

## 7. Choices made by measurement, not assumption

| Choice | Evidence |
| --- | --- |
| Gradient checkpointing **on** | Turning it off was *slower* (33.4 vs 9.8 s/step) and OOM'd at step 9. The activation memory it saves matters more than the recomputation it costs. |
| fp16 embeddings | float32 ran 4.3× slower on an A40 (248 vs 1,054 chunks/s). Verified equivalent first: cosine ≥ 0.9995, top-1 agreement 1.000, top-5 overlap 0.998 over 3,000 real chunks. |
| Chunking is mandatory | Raw MIRIAD passages average 4,509 characters against bge-large's 512-token window — over half of each would have been silently unsearchable. |
| 1 epoch over 30k rows | Same compute as 2 epochs over 15k; more unique data beats repetition for knowledge acquisition. |
| Minimum chunk length 40, not 80 | An 80-char cut discarded 13.8% of explanations, a third of them substantive. Below 40 the field is answer-key boilerplate; the junk is removed by content instead. |

---

## 8. A decision framework

Generalising past this dataset, in the order the questions actually arise:

1. **Do you have a corpus that contains your answers?** Measure retrieval hit
   rate before building anything. `rag-external` had a 45.4% hit rate and gained
   nothing — presence of the answer is not the same as usable evidence. If your
   hit rate is low, neither retrieval nor a fine-tune trained on that corpus
   will help.
2. **Is your corpus distributionally close to your task?** The single largest
   effect in this study is corpus choice: +10.2 points versus +0.1, same
   technique. Literature was near-useless for exam questions; exam explanations
   were transformative.
3. **Then estimate volume.** Fine-tuning is a fixed cost repaid by cheaper
   inference — here ~200k queries at a 6× prompt-token reduction. Below your
   crossover, retrieve; above it, fine-tune. Compute your own crossover; do not
   inherit this one.
4. **Consider both.** They composed here (+4.1 over the better single technique)
   when retrieval was useful, and did not when it was not.
5. **Measure quality, latency and cost together.** Ranked by accuracy alone,
   `rag-parity` beats `qlora`. Ranked by cost at high volume, `qlora` wins — it
   serves 113-token prompts instead of 656.

---

## 9. Reproducing

```bash
git clone https://github.com/vireshkoli/Fine-Tune-vs-RAG && cd Fine-Tune-vs-RAG
uv sync --group dev
make check                      # ruff + mypy(strict) + 298 tests, CPU only
make verify-splits              # proves the frozen split still hashes identically
make setup-gpu                  # GPU extras
make index CORPUS=parity        # ~11 min
make index CORPUS=external      # ~50 min
make train CONFIG=configs/train/qlora_r16.yaml   # ~5.1 GPU-hours
make eval ARM=base
make report                     # regenerates every table and figure
```

Every number in this report traces to a JSON file under `results/runs/`.
`make report` regenerates the tables and figures from those files; if the output
differs from what is committed, either the results changed or a number was
edited by hand.

**Environment.** 2× NVIDIA A40 (46 GB), driver 570.133.07 / CUDA 12.8, Python
3.12, torch 2.11.0+cu128 (the cu128 index is pinned because the default wheel
needs CUDA 13.0 and this is shared hardware whose driver is not ours to update).

**Licences.** MedMCQA is Apache 2.0. MIRIAD is ODC-By 1.0. Qwen3-8B is Apache
2.0 and ungated; `bge-large-en-v1.5` is MIT. MedQA was deliberately *not* used —
`bigbio/med_qa` declares its licence "unknown", and a re-uploader's `cc-by-4.0`
tag does not launder upstream copyright on USMLE board-prep material.
