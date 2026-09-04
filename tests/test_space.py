"""The demo Space: payload construction, and the rendering that reads it.

Two concerns, tested together because they are two halves of one contract —
``fvr.ops.space`` writes ``app/precomputed/responses.json`` and ``app/logic.js``
reads it. A field renamed on one side and not the other would break the live
Space silently.

The Space is static, so its rendering logic is JavaScript. Rather than leave it
untested or duplicate it in Python, the last class shells out to ``node --test``
and surfaces the result here — one ``make test`` still covers the whole demo.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from fvr.config import PROJECT_ROOT
from fvr.data.schema import Question
from fvr.ops.hub import SPACE_ENTRY_POINT
from fvr.ops.space import bucket_for, build_payload, select_items, softmax, write_payload
from fvr.report.aggregate import Aggregate, RunRecord

APP = PROJECT_ROOT / "app"


def _question(qid: str, answer_idx: int) -> Question:
    return Question(
        id=qid,
        question=f"Question {qid}?",
        options=["one", "two", "three", "four"],
        answer_idx=answer_idx,
        subject="Anatomy",
    )


def _run(arm: str, correct_ids: set[str], all_ids: list[str]) -> RunRecord:
    return RunRecord(
        arm=arm,
        seed=42,
        path=Path(f"{arm}.json"),
        payload={
            "arm": arm,
            "seed": 42,
            "split_sha256": "abc123",
            "n_items": len(all_ids),
            "accuracy": len(correct_ids) / len(all_ids),
            "ci_95": [0.0, 1.0],
            "latency": {"p50_s": 0.1, "p95_s": 0.12},
            "model": {"repo_id": "Qwen/Qwen3-8B", "revision": "deadbeef"},
            "environment": {"git_sha": "cafef00d"},
            "predictions": [
                {
                    "question_id": qid,
                    "predicted_idx": 0 if qid in correct_ids else 1,
                    "option_logprobs": [-0.1, -2.0, -3.0, -4.0],
                    "prompt_tokens": 113,
                }
                for qid in all_ids
            ],
        },
    )


class TestSoftmax:
    def test_normalises_to_one(self) -> None:
        assert sum(softmax([-0.1, -2.0, -3.0, -4.0])) == pytest.approx(1.0)

    def test_preserves_the_argmax(self) -> None:
        probs = softmax([-5.0, -0.2, -3.0, -9.0])
        assert probs.index(max(probs)) == 1

    def test_survives_large_negative_logprobs(self) -> None:
        """Shifting by the max is what stops this underflowing to all-zero."""
        probs = softmax([-900.0, -901.0, -902.0, -903.0])
        assert sum(probs) == pytest.approx(1.0)
        assert all(math.isfinite(p) for p in probs)

    def test_empty_is_empty(self) -> None:
        assert softmax([]) == []


class TestBuckets:
    def test_index_only_when_retrieval_alone_succeeds(self) -> None:
        assert bucket_for(base_ok=False, finetune_ok=False, retrieval_ok=True) == "index_only"

    def test_weights_only_when_finetune_alone_succeeds(self) -> None:
        assert bucket_for(base_ok=False, finetune_ok=True, retrieval_ok=False) == "weights_only"

    def test_both_fixed_it(self) -> None:
        assert bucket_for(base_ok=False, finetune_ok=True, retrieval_ok=True) == "both_fixed_it"

    def test_both_broke_it(self) -> None:
        assert bucket_for(base_ok=True, finetune_ok=False, retrieval_ok=False) == "both_broke_it"

    def test_everyone_right(self) -> None:
        assert bucket_for(base_ok=True, finetune_ok=True, retrieval_ok=True) == "everyone_right"

    def test_everyone_wrong(self) -> None:
        assert bucket_for(base_ok=False, finetune_ok=False, retrieval_ok=False) == "everyone_wrong"


class TestSelection:
    @pytest.fixture
    def aggregate(self) -> Aggregate:
        ids = [f"q{i:03d}" for i in range(40)]
        return Aggregate(
            runs=[
                _run("base", set(ids[:20]), ids),
                _run("qlora", set(ids[10:30]), ids),
                _run("rag-parity", set(ids[20:]), ids),
            ]
        )

    @pytest.fixture
    def questions(self) -> dict[str, Question]:
        return {f"q{i:03d}": _question(f"q{i:03d}", 0) for i in range(40)}

    def test_is_deterministic_for_a_seed(
        self, aggregate: Aggregate, questions: dict[str, Question]
    ) -> None:
        first = [i.question.id for i in select_items(aggregate, questions, per_bucket=3, seed=7)]
        second = [i.question.id for i in select_items(aggregate, questions, per_bucket=3, seed=7)]
        assert first == second

    def test_spreads_across_buckets(
        self, aggregate: Aggregate, questions: dict[str, Question]
    ) -> None:
        items = select_items(aggregate, questions, per_bucket=3)
        buckets = {item.bucket for item in items}
        assert len(buckets) > 1, "a stratified pick that lands in one bucket is not stratified"

    def test_respects_per_bucket(
        self, aggregate: Aggregate, questions: dict[str, Question]
    ) -> None:
        counts: dict[str, int] = {}
        for item in select_items(aggregate, questions, per_bucket=2):
            counts[item.bucket] = counts.get(item.bucket, 0) + 1
        assert all(count <= 2 for count in counts.values())

    def test_refuses_without_the_reference_arms(self, questions: dict[str, Question]) -> None:
        ids = list(questions)
        lonely = Aggregate(runs=[_run("base", set(ids[:5]), ids)])
        with pytest.raises(KeyError, match="qlora"):
            select_items(lonely, questions)


class TestPayload:
    def test_records_provenance_and_round_trips(
        self, tmp_path: Path, aggregate_and_questions: tuple[Aggregate, dict[str, Question]]
    ) -> None:
        aggregate, questions = aggregate_and_questions
        payload = build_payload(aggregate, questions, per_bucket=2)
        assert payload["provenance"]["split_sha256"] == "abc123"
        assert payload["provenance"]["model_revision"] == "deadbeef"
        assert payload["items"]

        path = write_payload(payload, tmp_path / "nested" / "responses.json")
        assert json.loads(path.read_text(encoding="utf-8")) == payload

    def test_refuses_runs_from_different_splits(
        self, aggregate_and_questions: tuple[Aggregate, dict[str, Question]]
    ) -> None:
        aggregate, questions = aggregate_and_questions
        aggregate.runs[-1].payload["split_sha256"] = "a-different-split"
        with pytest.raises(ValueError, match="not comparable"):
            build_payload(aggregate, questions)


@pytest.fixture
def aggregate_and_questions() -> tuple[Aggregate, dict[str, Question]]:
    ids = [f"q{i:03d}" for i in range(40)]
    questions = {qid: _question(qid, 0) for qid in ids}
    aggregate = Aggregate(
        runs=[
            _run("base", set(ids[:20]), ids),
            _run("qlora", set(ids[10:30]), ids),
            _run("rag-parity", set(ids[20:]), ids),
        ]
    )
    return aggregate, questions


class TestStaticSpace:
    """The shipped Space, and the JavaScript that renders it."""

    def test_declares_a_static_sdk_matching_its_entry_point(self) -> None:
        """A Space whose `sdk:` disagrees with reality builds the wrong runtime."""
        card = (APP / "README.md").read_text(encoding="utf-8")
        assert "sdk: static" in card
        assert "app_file: index.html" in card
        assert (APP / SPACE_ENTRY_POINT).is_file()

    def test_the_page_loads_only_files_that_exist(self) -> None:
        """A broken asset path is invisible in review and obvious to a visitor."""
        html = (APP / "index.html").read_text(encoding="utf-8")
        referenced = set(re.findall(r'(?:src|href)="([^"#:]+)"', html))
        for reference in referenced:
            assert (APP / reference).is_file(), f"index.html references missing {reference}"

    def test_the_payload_is_committed_and_current(self) -> None:
        """The Space ships the payload; a stale one would show stale answers."""
        payload = json.loads((APP / "precomputed" / "responses.json").read_text(encoding="utf-8"))
        runs = Aggregate.load(PROJECT_ROOT / "results" / "runs")
        assert {arm["name"] for arm in payload["arms"]} == set(runs.arms)
        assert payload["provenance"]["split_sha256"] in runs.split_hashes()

    def test_the_figures_match_the_generated_ones(self) -> None:
        """`make space-data` copies these; a drifted copy shows an old chart."""
        for figure in ("arms.png", "cost-crossover.png"):
            shipped = (APP / "assets" / figure).read_bytes()
            generated = (PROJECT_ROOT / "results" / "figures" / figure).read_bytes()
            assert shipped == generated, f"{figure} is stale; run `make space-data`"

    @pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
    def test_the_javascript_suite_passes(self) -> None:
        """Run app/logic.test.js so one `make test` covers the whole demo."""
        result = subprocess.run(
            ["node", "--test"],
            cwd=APP,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
