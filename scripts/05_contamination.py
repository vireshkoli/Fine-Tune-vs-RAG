#!/usr/bin/env python
"""Quantify how much of the benchmark's accuracy is recall rather than reasoning.

    uv run python scripts/05_contamination.py --arm base
    uv run python scripts/05_contamination.py --arm base --skip-verbatim

MedMCQA predates Qwen3 and is a widely mirrored public benchmark, so assuming it
appears in pretraining is the safe prior. We cannot inspect that corpus, so this
measures the *symptoms*:

* **permutation** — re-score with the answer options shuffled. Same information,
  different labels. A reasoning model is unaffected; a model that memorised
  "this item's answer is C" degrades.
* **position bias** — free from the same run: does the arm favour a letter
  regardless of content?
* **verbatim completion** — feed half a question stem and let the model
  continue. High overlap with the withheld half is direct memorisation evidence.

Results are written to ``results/contamination/<arm>.json`` and reported in the
README even where they weaken the headline.
"""

from __future__ import annotations

from fvr.config import bootstrap_env, load_config  # isort: skip

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from fvr.data.loaders import load_medmcqa
from fvr.data.schema import Question
from fvr.eval.contamination import (
    PermutationResult,
    permute_dataset,
    position_bias,
    split_stem,
    summarise_verbatim,
)
from fvr.eval.device import assert_device_exclusive
from fvr.inference.arms import get_arm
from fvr.models.loader import load_base_model, load_model_config
from fvr.prompts.templates import build_prompt
from fvr.seeding import set_all_seeds

console = Console()

#: Items used for the verbatim probe. Generation is far slower than scoring, and
#: memorisation shows up clearly well before the full test set.
VERBATIM_ITEMS = 150


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="base")
    parser.add_argument("--model", default="configs/model/qwen3-8b.yaml")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-verbatim", action="store_true")
    args = parser.parse_args()

    config = load_config()
    paths = bootstrap_env(config)
    seed = args.seed if args.seed is not None else config.seed
    set_all_seeds(seed)

    arm = get_arm(args.arm)
    if arm.uses_retrieval:
        console.print("[red]Contamination probes target the model, not the retriever.[/]")
        return 1

    split_ids = json.loads((paths.results / "split_ids.json").read_text(encoding="utf-8"))
    test_ids = set(split_ids["test"])
    pool, _ = load_medmcqa("validation")
    questions = sorted((q for q in pool if q.id in test_ids), key=lambda q: q.id)
    console.print(f"Loaded {len(questions):,} frozen test items")

    model_config = load_model_config(args.model)
    occupancy = assert_device_exclusive(model_config.device)
    console.print(f"Device: {occupancy}")

    loaded = load_base_model(model_config, use_cache=not args.adapter)
    if args.adapter:
        from fvr.models.loader import attach_adapter

        loaded = attach_adapter(loaded, args.adapter)

    from fvr.inference.engine import InferenceEngine

    engine = InferenceEngine(loaded)
    batch = config.eval_batch_size

    def score(items: list[Question]) -> list[int]:
        predicted: list[int] = []
        for start in range(0, len(items), batch):
            prompts = [build_prompt(q) for q in items[start : start + batch]]
            predicted.extend(s.scores.predicted_idx for s in engine.score_batch(prompts))
        return predicted

    console.print("Scoring in the original option order…")
    original = score(questions)
    console.print("Scoring with options permuted…")
    permuted_questions, _ = permute_dataset(questions, seed=seed)
    permuted = score(permuted_questions)

    gold = [q.answer_idx for q in questions]
    permuted_gold = [q.answer_idx for q in permuted_questions]
    original_acc = sum(p == g for p, g in zip(original, gold, strict=True)) / len(gold)
    permuted_acc = sum(p == g for p, g in zip(permuted, permuted_gold, strict=True)) / len(gold)

    permutation = PermutationResult(original_acc, permuted_acc, len(gold))
    bias = position_bias(original, gold)
    permuted_bias = position_bias(permuted, permuted_gold)

    payload: dict[str, object] = {
        "arm": arm.name,
        "seed": seed,
        "model": loaded.describe(),
        "permutation": permutation.as_dict(),
        "position_bias_original": bias.as_dict(),
        "position_bias_permuted": permuted_bias.as_dict(),
        "device_occupancy": occupancy.as_dict(),
    }

    if not args.skip_verbatim:
        console.print(f"Verbatim-completion probe over {VERBATIM_ITEMS} items…")
        import torch

        prefixes, references = zip(
            *(split_stem(q.question) for q in questions[:VERBATIM_ITEMS]), strict=True
        )
        continuations: list[str] = []
        for start in range(0, len(prefixes), batch):
            chunk = list(prefixes[start : start + batch])
            encoded = engine.tokenizer(chunk, return_tensors="pt", padding=True).to(
                engine.model.device
            )
            with torch.inference_mode():
                out = engine.model.generate(
                    **encoded,
                    max_new_tokens=48,
                    do_sample=False,
                    pad_token_id=engine.tokenizer.pad_token_id,
                )
            for i in range(len(chunk)):
                generated = out[i][encoded["input_ids"].shape[1] :]
                decoded = engine.tokenizer.decode(generated, skip_special_tokens=True)
                continuations.append(decoded if isinstance(decoded, str) else " ".join(decoded))
        payload["verbatim"] = summarise_verbatim(continuations, list(references)).as_dict()

        # Chance baseline. Exam stems are formulaic ("A 45-year-old man presents
        # with..."), so a bare overlap figure cannot distinguish memorisation
        # from shared phrasing. Scoring each continuation against a *different*
        # item's remainder gives the overlap attributable to genre alone; only
        # the excess over this baseline is evidence of anything.
        shuffled = [*references[1:], references[0]]
        payload["verbatim_shuffled_control"] = summarise_verbatim(continuations, shuffled).as_dict()

    out = paths.results / "contamination" / f"{arm.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    table = Table(title=f"Contamination probes — {arm.name}", show_lines=False)
    table.add_column("Probe", style="bold")
    table.add_column("Result", overflow="fold")
    table.add_row("original accuracy", f"{original_acc:.3f}")
    table.add_row("permuted accuracy", f"{permuted_acc:.3f}")
    table.add_row("drop", f"{permutation.drop:+.3f}")
    table.add_row("verdict", permutation.verdict())
    table.add_row(
        "position bias",
        f"predicted {bias.predicted}, max excess {bias.max_excess:.3f}",
    )
    if "verbatim" in payload:
        verbatim = payload["verbatim"]
        assert isinstance(verbatim, dict)
        table.add_row("verbatim", str(verbatim["verdict"]))
    console.print(table)
    console.print(f"Written to [cyan]{out}[/]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
