"""Pure rendering logic for the demo Space.

Deliberately free of any ``gradio`` import so it can be unit-tested in the main
repo's CPU-only test suite — ``app.py`` is a thin shell over these functions.
Nothing here computes a model output: every answer is read from
``precomputed/responses.json``, which was written from the committed run JSONs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
PAYLOAD_PATH = HERE / "precomputed" / "responses.json"

OPTION_LABELS = ("A", "B", "C", "D")

#: What each disagreement bucket demonstrates. The demo is stratified over these
#: rather than sampled at random, because a random draw is mostly items where
#: every arm agrees — which shows nothing.
BUCKET_BLURBS: dict[str, str] = {
    "index_only": (
        "**Retrieval got it, the fine-tune did not.** The same explanation is in "
        "the index and in the weights — only the index could use it."
    ),
    "weights_only": (
        "**The fine-tune got it, retrieval did not.** Either the retriever missed "
        "the passage, or the answer needed knowledge no single passage carries."
    ),
    "both_fixed_it": (
        "**Both approaches rescued an item the base model missed.** The kind of "
        "question where adding domain information of any form helps."
    ),
    "both_broke_it": (
        "**The base model was right and both approaches broke it.** Retrieved "
        "context and fine-tuning can each talk a correct model out of an answer."
    ),
    "everyone_right": "**Every arm answered correctly.** The base model already knew this.",
    "everyone_wrong": (
        "**No arm answered correctly.** The ceiling on this benchmark is not information access."
    ),
}

BUCKET_ORDER = (
    "index_only",
    "weights_only",
    "both_fixed_it",
    "both_broke_it",
    "everyone_right",
    "everyone_wrong",
)


def load_payload(path: Path = PAYLOAD_PATH) -> dict[str, Any]:
    return dict(json.loads(Path(path).read_text(encoding="utf-8")))


def arms_table(payload: dict[str, Any]) -> list[list[str]]:
    """The results table, ordered by accuracy."""
    rows = []
    for arm in sorted(payload["arms"], key=lambda a: -a["accuracy"]):
        low, high = arm["ci_95"]
        rows.append(
            [
                arm["name"],
                f"{arm['accuracy']:.1%}",
                f"[{low:.1%}, {high:.1%}]",
                f"{arm['p50_ms']:.0f} ms",
                f"{arm['p95_ms']:.0f} ms",
                f"{arm['prompt_tokens']:.0f}",
                "yes" if arm["headline"] else "diagnostic",
            ]
        )
    return rows


ARMS_TABLE_HEADERS = ["Arm", "Accuracy", "95% CI", "p50", "p95", "Prompt tokens", "Headline"]


def item_choices(payload: dict[str, Any], bucket: str) -> list[tuple[str, str]]:
    """``(label, id)`` pairs for the item picker, filtered to one bucket."""
    choices = []
    for item in payload["items"]:
        if bucket not in ("all", item["bucket"]):
            continue
        stem = item["question"].strip().replace("\n", " ")
        label = stem if len(stem) <= 90 else stem[:87] + "…"
        choices.append((f"[{item['subject']}] {label}", item["id"]))
    return choices


def find_item(payload: dict[str, Any], item_id: str) -> dict[str, Any] | None:
    return next((i for i in payload["items"] if i["id"] == item_id), None)


def render_question(item: dict[str, Any]) -> str:
    """The question, with the correct option marked."""
    lines = [f"### {item['question']}", "", f"*{item['subject']}*", ""]
    for idx, option in enumerate(item["options"]):
        marker = " ← **correct**" if idx == item["answer_idx"] else ""
        lines.append(f"- **{OPTION_LABELS[idx]}.** {option}{marker}")
    lines += ["", BUCKET_BLURBS.get(item["bucket"], "")]
    return "\n".join(lines)


def render_answers(item: dict[str, Any], payload: dict[str, Any]) -> list[list[str]]:
    """Each arm's precomputed answer for one item."""
    order = [arm["name"] for arm in payload["arms"]]
    rows = []
    for name in order:
        answer = item["answers"].get(name)
        if answer is None:
            continue
        correct = answer["predicted_idx"] == item["answer_idx"]
        confidence = answer.get("confidence")
        rows.append(
            [
                name,
                f"{answer['predicted_label']}. {item['options'][answer['predicted_idx']]}"
                if answer["predicted_idx"] is not None
                else "—",
                "correct" if correct else "wrong",
                f"{confidence:.0%}" if confidence is not None else "—",
                str(answer.get("prompt_tokens") or "—"),
            ]
        )
    return rows


ANSWER_TABLE_HEADERS = ["Arm", "Answered", "Verdict", "Confidence", "Prompt tokens"]


def usd_per_1k(arm_cost: dict[str, float], volume: int) -> float:
    """``fixed / N + marginal``, in dollars per 1,000 queries.

    The formula is deliberately here in full rather than interpolated from a
    sampled curve: the whole point of the cost section is that the shape is
    knowable, not that a chart says so.
    """
    if volume <= 0:
        raise ValueError("volume must be positive")
    return 1000 * (arm_cost["fixed_usd"] / volume + arm_cost["marginal_usd_per_query"])


def cost_table(payload: dict[str, Any], volume: int) -> list[list[str]]:
    """Cost per 1,000 queries for every arm at one lifetime volume."""
    arms = payload["cost"]["arms"]
    ranked = sorted(arms, key=lambda name: usd_per_1k(arms[name], volume))
    rows = []
    for rank, name in enumerate(ranked, start=1):
        cost = arms[name]
        fixed = cost["fixed_usd"]
        rows.append(
            [
                f"{rank}",
                name,
                f"${usd_per_1k(cost, volume):.4f}",
                f"${fixed:.2f}",
                f"${1000 * cost['marginal_usd_per_query']:.4f}",
            ]
        )
    return rows


COST_TABLE_HEADERS = [
    "#",
    "Arm",
    "$ / 1k queries",
    "Fixed cost (one-off)",
    "$ / 1k marginal",
]


def cost_summary(payload: dict[str, Any], volume: int) -> str:
    arms = payload["cost"]["arms"]
    cheapest = min(arms, key=lambda name: usd_per_1k(arms[name], volume))
    rate = payload["cost"]["rate_card"]
    lines = [
        f"At **{volume:,} lifetime queries**, the cheapest arm is **`{cheapest}`** "
        f"at ${usd_per_1k(arms[cheapest], volume):.4f} per 1,000 queries.",
        "",
        "Crossover volumes — where one arm overtakes another:",
        "",
    ]
    for pair, crossing in sorted(payload["cost"]["crossovers"].items(), key=lambda kv: kv[1]):
        a, b = pair.split("|")
        lines.append(f"- `{a}` becomes cheaper than `{b}` at **{crossing:,}** queries")
    lines += [
        "",
        f"Rates: {rate['gpu_name']} at ${rate['gpu_usd_per_hour']:.2f}/GPU-hour "
        f"([source]({rate['source_url']}), retrieved {rate['retrieved']}). "
        "Local GPU-seconds are never mixed with hosted API pricing.",
    ]
    return "\n".join(lines)


def provenance_note(payload: dict[str, Any]) -> str:
    p = payload["provenance"]
    return (
        f"Every answer shown was produced by a real evaluated run over a frozen "
        f"{p['n_test_items']:,}-item test set "
        f"(SHA-256 `{p['split_sha256'][:16]}…`), using "
        f"[`{p['model']}`](https://huggingface.co/{p['model']}) "
        f"at revision `{p['model_revision'][:12]}`, seed {p['seed']}, "
        f"from commit `{p['git_sha'][:12]}`. Nothing is generated at demo time."
    )
