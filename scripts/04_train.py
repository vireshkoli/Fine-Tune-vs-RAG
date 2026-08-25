#!/usr/bin/env python
"""QLoRA fine-tuning.

    uv run python scripts/04_train.py --config configs/train/smoke.yaml
    uv run python scripts/04_train.py --config configs/train/qlora_r16.yaml
    uv run python scripts/04_train.py --config configs/train/qlora_r16.yaml --estimate-only

Checkpoint selection uses the **validation** split. The test split is never
loaded by this script — asserted in ``tests/test_train.py`` — so no amount of
hyperparameter fiddling can leak into the reported numbers.

Interrupted runs resume automatically from the newest checkpoint; pass
``--no-resume`` to start over.
"""

from __future__ import annotations

# Pin the visible GPU before anything can initialise CUDA.
#
# HuggingFace Trainer places the model on `cuda:0` of its own accord, ignoring
# the `device_map` the loader passes. On a shared box that silently trains on
# whichever GPU happens to be device 0 — here that meant landing on a neighbour's
# 34GB job and OOMing at step 0. Restricting visibility is the only reliable fix,
# and it has to happen before torch initialises, hence the manual config read.
import os as _os  # isort: skip
import sys as _sys  # isort: skip
from pathlib import Path as _Path  # isort: skip

import yaml as _yaml  # isort: skip


def _pin_visible_device() -> None:
    if "CUDA_VISIBLE_DEVICES" in _os.environ:
        return
    argv = _sys.argv
    config_path = "configs/train/qlora_r16.yaml"
    if "--config" in argv:
        config_path = argv[argv.index("--config") + 1]
    try:
        train_raw = _yaml.safe_load(_Path(config_path).read_text(encoding="utf-8"))
        model_path = train_raw.get("model_config_path", "configs/model/qwen3-8b.yaml")
        model_raw = _yaml.safe_load(_Path(model_path).read_text(encoding="utf-8"))
        _os.environ["CUDA_VISIBLE_DEVICES"] = str(model_raw.get("device", 0))
    except (OSError, KeyError, TypeError, ValueError, IndexError):
        pass  # fall through to whatever torch defaults to


_pin_visible_device()

from fvr.config import bootstrap_env, load_config  # isort: skip

import argparse
import json
import sys
import time

from rich.console import Console

from fvr.data.loaders import load_medmcqa
from fvr.data.sft import build_sft_dataset, to_hf_dataset
from fvr.eval.device import assert_device_exclusive
from fvr.models.loader import load_model_config
from fvr.seeding import set_all_seeds
from fvr.train.callbacks import latest_checkpoint
from fvr.train.config import load_train_config

console = Console()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train/qlora_r16.yaml")
    parser.add_argument("--no-resume", action="store_true", help="ignore existing checkpoints")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="time a few steps, project total GPU-hours, then stop",
    )
    args = parser.parse_args()

    project = load_config()
    paths = bootstrap_env(project)
    train_config = load_train_config(args.config)
    model_config = load_model_config(train_config.model_config_path)
    physical_device = model_config.device
    # Renumbered: with CUDA_VISIBLE_DEVICES set, the target GPU is index 0.
    model_config = model_config.model_copy(update={"device": 0})
    set_all_seeds(train_config.seed)

    occupancy = assert_device_exclusive(physical_device)
    console.print(f"Device: {occupancy}")
    if not occupancy.is_exclusive:
        console.print(
            "[yellow]Training on a shared device. It will still be correct, but slower, "
            "and the recorded GPU-seconds will overstate the true training cost.[/]"
        )

    split_ids = json.loads((paths.results / "split_ids.json").read_text(encoding="utf-8"))
    held_out_ids = set(split_ids["test"]) | set(split_ids["val"])
    val_ids = set(split_ids["val"])

    console.print("Loading MedMCQA…")
    pool, _ = load_medmcqa("validation")
    forbidden = frozenset(q.content_hash() for q in pool if q.id in held_out_ids)
    train_rows, clean = load_medmcqa("train")
    console.print(f"  train cleaning: {clean.summary()}")

    records, stats = build_sft_dataset(
        train_rows,
        forbidden_content_hashes=forbidden,
        include_explanation=train_config.include_explanation,
        max_samples=train_config.max_train_samples,
        seed=train_config.seed,
    )
    console.print(f"  SFT: {stats.summary()}")
    if stats.dropped_leaked:
        console.print(f"  [yellow]{stats.dropped_leaked} leaked rows excluded from training[/]")

    val_records, val_stats = build_sft_dataset(
        [q for q in pool if q.id in val_ids],
        forbidden_content_hashes=frozenset(),
        include_explanation=train_config.include_explanation,
    )
    console.print(f"  val: {val_stats.summary()}")

    train_dataset = to_hf_dataset(records)
    eval_dataset = to_hf_dataset(val_records)

    output_dir = paths.checkpoints / train_config.name
    existing = latest_checkpoint(output_dir)
    if existing and not args.no_resume:
        console.print(f"[cyan]Resuming from {existing}[/]")

    steps_per_epoch = max(1, len(records) // train_config.effective_batch_size)
    total_steps = int(steps_per_epoch * train_config.epochs)
    console.print(
        f"\n[bold]{train_config.name}[/]: LoRA r={train_config.lora.r}, "
        f"{train_config.quantization}, effective batch {train_config.effective_batch_size}, "
        f"~{total_steps:,} steps over {train_config.epochs} epochs"
    )

    from fvr.train.qlora import train

    if args.estimate_only:
        probe = 12
        console.print(f"Timing {probe} steps to project total cost…")
        started = time.perf_counter()
        train(
            train_config,
            model_config,
            train_dataset,
            eval_dataset,
            output_dir.with_name(output_dir.name + "-probe"),
            resume=False,
            max_steps=probe,
        )
        per_step = (time.perf_counter() - started) / probe
        console.print(
            f"\n[bold]Estimate[/]: {per_step:.2f}s/step -> {total_steps:,} steps "
            f"= [bold]{total_steps * per_step / 3600:.2f} GPU-hours[/]"
        )
        return 0

    result = train(
        train_config,
        model_config,
        train_dataset,
        eval_dataset,
        output_dir,
        resume=not args.no_resume,
        max_steps=args.max_steps,
    )

    summary = paths.results / "training" / f"{train_config.name}.json"
    result.write(summary)

    console.print(f"\n[bold]{train_config.name}[/] finished at step {result.final_step:,}")
    console.print(
        f"  trainable {result.trainable_params:,} / {result.total_params:,} "
        f"({result.trainable_pct:.3f}%)"
    )
    console.print(f"  train loss {result.train_loss:.4f}" if result.train_loss else "")
    console.print(f"  best eval loss {result.best_eval_loss:.4f}" if result.best_eval_loss else "")
    console.print(f"  {result.train_gpu_seconds / 3600:.2f} GPU-hours")
    console.print(f"  adapter -> [cyan]{result.output_dir / 'adapter'}[/]")
    console.print(f"  summary -> [cyan]{summary}[/]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
