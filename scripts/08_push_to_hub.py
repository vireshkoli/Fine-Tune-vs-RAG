#!/usr/bin/env python
"""Publish the adapter and the demo Space to the Hugging Face Hub.

    uv run python scripts/08_push_to_hub.py                     # dry run, prints the manifest
    uv run python scripts/08_push_to_hub.py --build-data        # regenerate the Space payload
    uv run python scripts/08_push_to_hub.py --execute           # actually push

Dry run is the default, matching ``10_teardown.py``: publishing is outward-facing
and irreversible in the sense that matters — once a file is on the Hub it may
have been fetched — so the manifest is printed and reviewed before anything is
uploaded.

This is what makes teardown safe. The adapter cost five GPU-hours and lives in
``.artifacts/``, which is the directory that gets deleted; the Hub copy is the
one that survives the lab machine being wiped.
"""

from __future__ import annotations

from fvr.config import Paths, Secrets, bootstrap_env, load_config  # isort: skip

import argparse
import itertools
import json
import sys
from pathlib import Path

from rich.console import Console

from fvr.ops.hub import HubTargets, UploadPlan, plan_adapter_upload, plan_space_upload

console = Console()

ADAPTER_DIR = "checkpoints/qlora-r16/adapter"
CARD_PATH = "docs/model_card.md"
APP_DIR = "app"


def build_space_data(paths: Paths) -> Path:
    """Regenerate ``app/precomputed/responses.json`` from committed run JSONs."""
    from fvr.data.loaders import load_medmcqa
    from fvr.eval.cost import crossover_volume
    from fvr.ops.space import build_payload, write_payload
    from fvr.report.aggregate import Aggregate
    from fvr.report.costs import build_costs, cost_inputs, load_rate_card

    aggregate = Aggregate.load(paths.results / "runs")
    if not aggregate.runs:
        raise SystemExit("no runs under results/runs/ — nothing to build a demo from")

    console.print("Loading MedMCQA to attach question text to the committed predictions…")
    pool, _ = load_medmcqa("validation")

    test_ids = set(
        json.loads((paths.results / "split_ids.json").read_text(encoding="utf-8"))["test"]
    )
    questions = {q.id: q for q in pool if q.id in test_ids}

    # The Space computes cost curves itself from these two numbers per arm, so
    # the demo and the README cannot disagree about what an arm costs.
    rates, amortization = load_rate_card(paths.configs)
    costs = build_costs(aggregate, rates, paths)
    cost = cost_inputs(costs, rates)
    cost["default_volume"] = amortization["default_volume"]
    cost["crossovers"] = {
        f"{a}|{b}": volume
        for a, b in itertools.combinations(costs, 2)
        if (volume := crossover_volume(costs[a], costs[b])) is not None
    }

    payload = build_payload(aggregate, questions, extra={"cost": cost})
    out = write_payload(payload, paths.root / APP_DIR / "precomputed" / "responses.json")
    buckets: dict[str, int] = {}
    for item in payload["items"]:
        buckets[item["bucket"]] = buckets.get(item["bucket"], 0) + 1
    console.print(
        f"Wrote [cyan]{out}[/] — {len(payload['items'])} items across {len(buckets)} buckets"
    )
    for name, count in sorted(buckets.items()):
        console.print(f"    {count:3d}  {name}")
    return out


def push(plan: UploadPlan, *, token: str | None, message: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(plan.repo_id, repo_type=plan.repo_type, exist_ok=True, private=False)
    for local, remote in sorted(plan.entries, key=lambda e: e[1]):
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=remote,
            repo_id=plan.repo_id,
            repo_type=plan.repo_type,
            commit_message=message,
        )
    prefix = "spaces/" if plan.repo_type == "space" else ""
    return f"https://huggingface.co/{prefix}{plan.repo_id}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["adapter", "space", "all"], default="all")
    parser.add_argument("--namespace", default=None, help="defaults to the token's own account")
    parser.add_argument(
        "--build-data", action="store_true", help="regenerate the Space payload first"
    )
    parser.add_argument("--execute", action="store_true", help="actually upload (default: dry run)")
    args = parser.parse_args()

    config = load_config()
    paths = bootstrap_env(config)

    token = Secrets().hf_token
    if not token:
        console.print("[red]No HF_TOKEN in .env — see .env.example. A write token is required.[/]")
        return 1

    from huggingface_hub import HfApi

    who = HfApi(token=token).whoami()
    namespace = args.namespace or str(who["name"])
    role = who.get("auth", {}).get("accessToken", {}).get("role")
    console.print(f"Authenticated as [bold]{namespace}[/] (token role: {role})")
    if args.execute and role != "write":
        console.print("[red]Token is not a write token; generate one at hf.co/settings/tokens.[/]")
        return 1

    if args.build_data:
        build_space_data(paths)

    targets = HubTargets(namespace=namespace)
    plans: list[UploadPlan] = []
    if args.target in {"adapter", "all"}:
        plans.append(
            plan_adapter_upload(paths.artifacts / ADAPTER_DIR, paths.root / CARD_PATH, targets)
        )
    if args.target in {"space", "all"}:
        plans.append(plan_space_upload(paths.root / APP_DIR, targets))

    console.print()
    for plan in plans:
        console.print(plan.render())
        console.print()

    if not args.execute:
        console.print("[yellow]Dry run — nothing uploaded. Re-run with --execute to push.[/]")
        return 0

    message = "Publish benchmark artifact from Fine-Tune-vs-RAG"
    for plan in plans:
        console.print(f"Pushing {plan.repo_type} [bold]{plan.repo_id}[/]…")
        url = push(plan, token=token, message=message)
        console.print(f"  [green]{url}[/]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
