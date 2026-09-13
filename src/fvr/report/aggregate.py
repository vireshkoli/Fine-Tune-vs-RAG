"""Loads committed run JSONs into one aggregate view.

Every number that reaches the README passes through here, so the README cannot
contain a figure that is not backed by a file under ``results/runs/``. That is
the property that makes "every number is reproducible" checkable rather than
aspirational.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fvr.eval.metrics import Comparison, mcnemar, seed_variance

#: The training seed every headline number is reported at. Mirrors
#: ``configs/base.yaml``; entry points pass the loaded config value explicitly.
DEFAULT_REFERENCE_SEED = 42


@dataclass(frozen=True)
class RunRecord:
    """One committed run JSON, typed."""

    arm: str
    seed: int
    path: Path
    payload: dict[str, Any]

    @property
    def accuracy(self) -> float:
        return float(self.payload["accuracy"])

    @property
    def ci(self) -> tuple[float, float]:
        low, high = self.payload["ci_95"]
        return float(low), float(high)

    @property
    def p50_ms(self) -> float:
        return float(self.payload["latency"]["p50_s"]) * 1000

    @property
    def p95_ms(self) -> float:
        return float(self.payload["latency"]["p95_s"]) * 1000

    @property
    def mean_prompt_tokens(self) -> float:
        tokens = [p["prompt_tokens"] for p in self.payload["predictions"]]
        return sum(tokens) / len(tokens) if tokens else 0.0

    @property
    def groundedness(self) -> dict[str, float] | None:
        return self.payload.get("groundedness")

    @property
    def exclusive_device(self) -> bool:
        occupancy = self.payload.get("device_occupancy") or {}
        return bool(occupancy.get("exclusive", True))

    def correctness(self, gold: dict[str, int | None]) -> dict[str, bool]:
        return {
            p["question_id"]: p["predicted_idx"] == gold[p["question_id"]]
            for p in self.payload["predictions"]
        }


@dataclass
class Aggregate:
    """All runs, grouped by arm."""

    runs: list[RunRecord] = field(default_factory=list)
    #: Which seed's run is *the* number for an arm. Everything else for that
    #: arm is a replicate and feeds only the seed-variance column.
    reference_seed: int = DEFAULT_REFERENCE_SEED

    @classmethod
    def load(cls, runs_dir: Path, *, reference_seed: int = DEFAULT_REFERENCE_SEED) -> Aggregate:
        records: list[RunRecord] = []
        for path in sorted(Path(runs_dir).glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            records.append(
                RunRecord(
                    arm=str(payload["arm"]),
                    seed=int(payload["seed"]),
                    path=path,
                    payload=payload,
                )
            )
        return cls(runs=records, reference_seed=reference_seed)

    @property
    def arms(self) -> list[str]:
        seen: list[str] = []
        for run in self.runs:
            if run.arm not in seen:
                seen.append(run.arm)
        return seen

    def for_arm(self, arm: str) -> list[RunRecord]:
        return [r for r in self.runs if r.arm == arm]

    def primary(self, arm: str) -> RunRecord:
        """The run reported as this arm's number: the reference seed.

        Never ``for_arm(arm)[0]``. Runs load in filename order, and
        ``qlora_seed1.json`` sorts before ``qlora_seed42.json`` — so the moment
        seed replicates were added, "the first run" silently became seed 1 for
        every trained arm while base stayed at seed 42. Every headline table,
        every paired McNemar test and the cost model would have been rebuilt on
        a different training run, pairing seed-1 fine-tunes against seed-42
        baselines, with all tests still passing. Selecting by seed makes the
        choice explicit and makes the ambiguous case an error.
        """
        runs = self.for_arm(arm)
        if not runs:
            raise KeyError(f"no runs for arm {arm!r}")
        reference = [r for r in runs if r.seed == self.reference_seed]
        if reference:
            return reference[0]
        if len(runs) == 1:
            # One run is unambiguous whatever its seed.
            return runs[0]
        raise ValueError(
            f"arm {arm!r} has runs at seeds {sorted(r.seed for r in runs)} but none at the "
            f"reference seed {self.reference_seed}; refusing to guess which one to report"
        )

    def seeds(self, arm: str) -> list[int]:
        return sorted(r.seed for r in self.for_arm(arm))

    def split_hashes(self) -> set[str]:
        return {str(r.payload["split_sha256"]) for r in self.runs}

    def assert_same_split(self) -> None:
        """Every arm must have been scored on the identical frozen test set.

        Cheap to check and fatal to miss: arms evaluated on different data are
        not comparable, and nothing downstream would notice.
        """
        hashes = self.split_hashes()
        if len(hashes) > 1:
            raise ValueError(
                f"runs span {len(hashes)} different test splits: {sorted(hashes)}. "
                "These results are not comparable."
            )

    def seed_summary(self, arm: str) -> tuple[float, float]:
        return seed_variance([r.accuracy for r in self.for_arm(arm)])

    def compare(self, arm_a: str, arm_b: str, gold: dict[str, int | None]) -> Comparison:
        """Paired comparison on the items both arms scored, at the reference seed."""
        a_correct = self.primary(arm_a).correctness(gold)
        b_correct = self.primary(arm_b).correctness(gold)
        shared = sorted(set(a_correct) & set(b_correct))
        return mcnemar([a_correct[i] for i in shared], [b_correct[i] for i in shared])


def load_gold(split_ids_path: Path, questions: Sequence[Any]) -> dict[str, int | None]:
    """Gold answers for the frozen test ids."""
    ids = set(json.loads(Path(split_ids_path).read_text(encoding="utf-8"))["test"])
    return {q.id: q.answer_idx for q in questions if q.id in ids}
