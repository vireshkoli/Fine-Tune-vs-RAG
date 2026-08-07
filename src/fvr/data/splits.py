"""Frozen evaluation splits.

The test set is chosen once, hashed, and committed **before any model exists**.
Everything about this module is designed so that a later change cannot quietly
move the goalposts: selection is deterministic from a seed, the manifest records
a SHA-256 over the chosen ids, and :func:`verify_manifest` fails loudly if a
rebuild disagrees with what is in git.

A note on which rows are usable. MedMCQA ships three splits, but its ``test``
split has ``cop == -1`` — the answers are withheld for a leaderboard. The only
labelled evaluation pool is therefore the 4,183-row ``validation`` split, and
work that reports "MedMCQA test accuracy" from the Hub is really reporting
validation. We carve test and val out of that pool and say so.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from fvr.data.schema import Question, Split

#: Held out for the final report. Never used for checkpoint selection.
DEFAULT_TEST_SIZE = 1000
#: Used for checkpoint selection during training. Never reported as a headline.
DEFAULT_VAL_SIZE = 500

MANIFEST_VERSION = 1


@dataclass(frozen=True)
class SplitAssignment:
    """Which question ids landed in which split, plus the hashes that pin them."""

    test_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    reserve_ids: tuple[str, ...]
    seed: int
    source_split: str

    def digest(self, split: Split) -> str:
        return _hash_ids(self.ids_for(split))

    def ids_for(self, split: Split) -> tuple[str, ...]:
        match split:
            case Split.TEST:
                return self.test_ids
            case Split.VAL:
                return self.val_ids
            case Split.RESERVE:
                return self.reserve_ids
            case _:
                raise KeyError(f"{split} is not carved from the labelled pool")


def _hash_ids(ids: Sequence[str]) -> str:
    """Order-independent hash, so a reordering is not mistaken for a change."""
    joined = "\n".join(sorted(ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def stratified_split(
    questions: Sequence[Question],
    *,
    seed: int,
    test_size: int = DEFAULT_TEST_SIZE,
    val_size: int = DEFAULT_VAL_SIZE,
    source_split: str = "validation",
) -> SplitAssignment:
    """Deterministically carve test/val/reserve, stratified by subject.

    Stratification matters here: MedMCQA spans 21 subjects with very uneven
    counts, and an unstratified 1,000-row sample can miss small subjects
    entirely, which would make per-subject error analysis meaningless.

    Rows not selected become ``RESERVE`` — deliberately unused rather than
    folded into training, so the labelled pool stays clean and the test set
    cannot be contaminated by a later decision to train on the leftovers.
    """
    if test_size + val_size > len(questions):
        raise ValueError(
            f"need {test_size + val_size} rows but the labelled pool has {len(questions)}"
        )

    by_subject: dict[str, list[Question]] = defaultdict(list)
    for q in sorted(questions, key=lambda x: x.id):  # sort first: dict order must not matter
        by_subject[q.subject].append(q)

    rng = random.Random(seed)
    total = len(questions)
    test: list[str] = []
    val: list[str] = []

    def _quota(want: int) -> dict[str, int]:
        """Largest-remainder allocation, so quotas sum exactly to ``want``
        instead of drifting by a few rows through repeated rounding down."""
        exact = {s: want * len(qs) / total for s, qs in by_subject.items()}
        quota = {s: int(v) for s, v in exact.items()}
        for subject in sorted(exact, key=lambda s: (-(exact[s] - quota[s]), s)):
            if sum(quota.values()) >= want:
                break
            quota[subject] += 1
        return quota

    test_quota, val_quota = _quota(test_size), _quota(val_size)

    for subject in sorted(by_subject):
        pool = [q.id for q in by_subject[subject]]
        rng.shuffle(pool)
        n_test = min(test_quota.get(subject, 0), len(pool))
        n_val = min(val_quota.get(subject, 0), len(pool) - n_test)
        test.extend(pool[:n_test])
        val.extend(pool[n_test : n_test + n_val])

    chosen = set(test) | set(val)
    reserve = [q.id for q in sorted(questions, key=lambda x: x.id) if q.id not in chosen]

    return SplitAssignment(
        test_ids=tuple(sorted(test)),
        val_ids=tuple(sorted(val)),
        reserve_ids=tuple(reserve),
        seed=seed,
        source_split=source_split,
    )


def build_manifest(
    assignment: SplitAssignment,
    *,
    dataset_revision: str | None,
    pool_size: int,
    lexicon_size: int,
) -> dict[str, object]:
    """The committed record of what the frozen split is."""
    return {
        "manifest_version": MANIFEST_VERSION,
        "dataset": "openlifescienceai/medmcqa",
        "dataset_revision": dataset_revision,
        "source_split": assignment.source_split,
        "note": (
            "MedMCQA's own `test` split withholds labels (cop == -1), so the labelled "
            "`validation` split is the only usable evaluation pool. test/val/reserve "
            "below are carved from it."
        ),
        "seed": assignment.seed,
        "ocr_lexicon_entries": lexicon_size,
        "labelled_pool_size": pool_size,
        "splits": {
            "test": {"n": len(assignment.test_ids), "sha256": assignment.digest(Split.TEST)},
            "val": {"n": len(assignment.val_ids), "sha256": assignment.digest(Split.VAL)},
            "reserve": {
                "n": len(assignment.reserve_ids),
                "sha256": assignment.digest(Split.RESERVE),
            },
        },
    }


def write_manifest(manifest: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_manifest(assignment: SplitAssignment, manifest: Mapping[str, object]) -> list[str]:
    """Compare a freshly built assignment against the committed manifest.

    Returns a list of human-readable mismatches; empty means the split
    reproduced exactly. This is what makes "frozen" verifiable rather than
    aspirational.
    """
    problems: list[str] = []
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        return ["manifest has no `splits` section"]

    for split in (Split.TEST, Split.VAL, Split.RESERVE):
        recorded = splits.get(split.value)
        if not isinstance(recorded, dict):
            problems.append(f"{split.value}: missing from manifest")
            continue
        actual = assignment.digest(split)
        if recorded.get("sha256") != actual:
            problems.append(
                f"{split.value}: sha256 {actual[:12]}… does not match committed "
                f"{str(recorded.get('sha256'))[:12]}…"
            )
        if recorded.get("n") != len(assignment.ids_for(split)):
            problems.append(
                f"{split.value}: size {len(assignment.ids_for(split))} != "
                f"committed {recorded.get('n')}"
            )
    return problems


def assert_disjoint(assignment: SplitAssignment) -> None:
    """Test, val and reserve must not overlap. Cheap to check, fatal to miss."""
    test, val, reserve = (
        set(assignment.test_ids),
        set(assignment.val_ids),
        set(assignment.reserve_ids),
    )
    for a, b, name in (
        (test, val, "test/val"),
        (test, reserve, "test/reserve"),
        (val, reserve, "val/reserve"),
    ):
        if overlap := a & b:
            raise AssertionError(
                f"{name} overlap on {len(overlap)} ids, e.g. {sorted(overlap)[:3]}"
            )


def find_train_leakage(
    train: Sequence[Question], evaluation: Sequence[Question]
) -> list[tuple[str, str]]:
    """Train rows whose content matches an evaluation row.

    Matches on :meth:`Question.content_hash`, not id — MedMCQA repeats items
    across splits under different ids, and those repeats are exactly what would
    inflate the fine-tuned arms.
    """
    eval_hashes = {q.content_hash(): q.id for q in evaluation}
    return [
        (q.id, eval_hashes[digest]) for q in train if (digest := q.content_hash()) in eval_hashes
    ]
