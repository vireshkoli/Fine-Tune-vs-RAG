#!/usr/bin/env python
"""Score the free-text answers with the LLM judge.

    make judge-server                                   # print the serve command
    uv run python scripts/13_judge_freetext.py --check   # is the judge reachable?
    uv run python scripts/13_judge_freetext.py --arm base
    uv run python scripts/13_judge_freetext.py --all
    uv run python scripts/13_judge_freetext.py --kappa   # judge vs your labels

Reads ``results/freetext/<arm>_seed<N>.json`` and writes
``results/freetext/judged/<arm>_seed<N>.json``.

Separate from generation on purpose: answers cost GPU-hours, judging does not
re-run the model, so a rubric correction means re-judging rather than
regenerating.
"""

from __future__ import annotations

from fvr.config import bootstrap_env, load_config  # isort: skip

import argparse
import csv
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from fvr.eval.judge import ArmJudgement, JudgeFn, cohens_kappa, score_pointwise
from fvr.eval.judge_client import HttpJudge, JudgeUnavailableError, load_judge_config
from fvr.prompts.judge import RUBRIC_VERSION

console = Console()

#: How many items you hand-label. Enough for a usable kappa, small enough to do
#: in one sitting — this is the highest-value-per-hour item in the project.
KAPPA_SAMPLE = 50


def judged_path(results: Path, arm: str, seed: int) -> Path:
    return results / "freetext" / "judged" / f"{arm}_seed{seed}.json"


def judge_one_run(run_path: Path, judge: JudgeFn, seeds: tuple[int, ...]) -> ArmJudgement:
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    judgement = ArmJudgement(arm=str(payload["arm"]))
    with Progress(console=console) as progress:
        task = progress.add_task(f"judging {payload['arm']}", total=len(payload["answers"]))
        for answer in payload["answers"]:
            judgement.results.append(
                score_pointwise(
                    answer["question_id"],
                    answer["question"],
                    answer["reference"],
                    answer["answer"],
                    judge,
                    seeds=seeds,
                )
            )
            progress.advance(task)
    return judgement


def write_kappa_sheet(run_path: Path, judged: ArmJudgement, out: Path, n: int) -> Path:
    """A CSV for a human to grade, with the judge's own score withheld.

    The judge's label is deliberately *not* in the sheet. Showing it would
    anchor the grader, and an agreement statistic between a human and a number
    they were shown is not a measurement of anything.
    """
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    by_id = {a["question_id"]: a for a in payload["answers"]}
    # Stratified over the judge's own score so the sheet is not 90% easy
    # agreements: disagreements are where kappa is actually decided.
    buckets: dict[int, list[str]] = {}
    for result in judged.results:
        if result.scores:
            buckets.setdefault(result.majority, []).append(result.item_id)

    chosen: list[str] = []
    per_bucket = max(1, n // max(1, len(buckets)))
    for score in sorted(buckets):
        chosen.extend(sorted(buckets[score])[:per_bucket])
    for score in sorted(buckets):
        for item_id in sorted(buckets[score]):
            if len(chosen) >= n:
                break
            if item_id not in chosen:
                chosen.append(item_id)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "question_id",
                "question",
                "reference",
                "candidate",
                "human_score",
                "notes",
            ],
        )
        writer.writeheader()
        for item_id in chosen[:n]:
            answer = by_id[item_id]
            writer.writerow(
                {
                    "question_id": item_id,
                    "question": answer["question"],
                    "reference": answer["reference"],
                    "candidate": answer["answer"],
                    "human_score": "",
                    "notes": "",
                }
            )
    return out


def compute_kappa(sheet: Path, judged_file: Path) -> None:
    """Compare a filled-in sheet against the judge's labels."""
    judged = json.loads(judged_file.read_text(encoding="utf-8"))
    machine = {r["item_id"]: r["majority"] for r in judged["items"]}

    human_labels: list[int] = []
    machine_labels: list[int] = []
    skipped = 0
    with sheet.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw = (row.get("human_score") or "").strip()
            if not raw:
                skipped += 1
                continue
            if row["question_id"] not in machine:
                skipped += 1
                continue
            human_labels.append(int(raw))
            machine_labels.append(machine[row["question_id"]])

    if not human_labels:
        console.print(
            f"[yellow]No rows in {sheet} have a human_score yet.[/] "
            "Fill the column with 0, 1 or 2 per the rubric in src/fvr/prompts/judge.py."
        )
        return

    agreement = cohens_kappa(human_labels, machine_labels)
    console.print()
    console.print(f"[bold]Judge vs human[/] on {agreement.n} items ({skipped} unlabelled)")
    console.print(f"  exact agreement  {agreement.exact_agreement:.1%}")
    console.print(f"  Cohen's kappa    {agreement.kappa:.3f}  ([bold]{agreement.verdict()}[/])")

    out = judged_file.parent / "human_agreement.json"
    out.write_text(json.dumps(agreement.as_json(), indent=2) + "\n", encoding="utf-8")
    console.print(f"  written          [cyan]{out}[/]")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default=None)
    parser.add_argument("--all", action="store_true", help="judge every generated run")
    parser.add_argument("--config", default="configs/eval/judge.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--check", action="store_true", help="probe the judge and stop")
    parser.add_argument("--kappa", action="store_true", help="score a filled-in labelling sheet")
    parser.add_argument("--kappa-arm", default="base", help="which arm's sheet to use")
    args = parser.parse_args()

    project = load_config()
    paths = bootstrap_env(project)
    seed = args.seed if args.seed is not None else project.seed
    judge_config = load_judge_config(args.config)
    freetext = paths.results / "freetext"

    if args.kappa:
        sheet = freetext / "judged" / f"{args.kappa_arm}_kappa_sheet.csv"
        judged_file = judged_path(paths.results, args.kappa_arm, seed)
        if not sheet.is_file() or not judged_file.is_file():
            console.print(f"[red]Need both {sheet} and {judged_file}. Judge an arm first.[/]")
            return 1
        compute_kappa(sheet, judged_file)
        return 0

    judge = HttpJudge(judge_config)

    if args.check:
        console.print(f"Probing [cyan]{judge_config.base_url}[/] ({judge_config.repo_id})…")
        try:
            reply = judge([{"role": "user", "content": "Reply with exactly: SCORE: 2"}], 0)
        except JudgeUnavailableError as exc:
            console.print(f"[red]{exc}[/]")
            return 1
        console.print(f"[green]Judge is up.[/] Replied: {reply!r}")
        return 0

    runs = sorted(freetext.glob(f"*_seed{seed}.json"))
    if args.arm:
        runs = [r for r in runs if r.stem == f"{args.arm}_seed{seed}"]
    if not runs:
        console.print(
            f"[red]No generated runs under {freetext}.[/] Run scripts/12_freetext_eval.py first."
        )
        return 1
    if not args.all and not args.arm:
        console.print("[yellow]Pass --arm NAME or --all.[/]")
        return 1

    table = Table(title=f"Judged (rubric {RUBRIC_VERSION})")
    table.add_column("Arm", style="bold")
    table.add_column("n", justify="right")
    table.add_column("Mean score", justify="right")
    table.add_column("Judge SD", justify="right")
    table.add_column("Unparseable", justify="right")

    for run_path in runs:
        try:
            judgement = judge_one_run(run_path, judge, tuple(judge_config.seeds))
        except JudgeUnavailableError as exc:
            console.print(f"[red]{exc}[/]")
            return 1

        out = judged_path(paths.results, judgement.arm, seed)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    **judgement.as_json(),
                    "judge": judge_config.describe(),
                    "judge_calls": judge.calls,
                    "items": [
                        {
                            "item_id": r.item_id,
                            "scores": list(r.scores),
                            "majority": r.majority if r.scores else None,
                            "mean": round(r.mean, 4),
                            "sd": round(r.sd, 4),
                            "unparseable": r.unparseable,
                        }
                        for r in judgement.results
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        table.add_row(
            judgement.arm,
            str(len(judgement.results)),
            f"{judgement.mean_score:.3f}",
            f"{judgement.judge_sd:.3f}",
            str(judgement.unparseable),
        )

        sheet = write_kappa_sheet(
            run_path, judgement, out.parent / f"{judgement.arm}_kappa_sheet.csv", KAPPA_SAMPLE
        )
        console.print(f"  {judgement.arm}: judged -> {out}")
        console.print(f"  {judgement.arm}: labelling sheet -> [cyan]{sheet}[/]")

    console.print()
    console.print(table)
    console.print(
        "\nNext: fill [cyan]human_score[/] (0/1/2 per the rubric) in a sheet, then "
        "`uv run python scripts/13_judge_freetext.py --kappa --kappa-arm <arm>`."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
