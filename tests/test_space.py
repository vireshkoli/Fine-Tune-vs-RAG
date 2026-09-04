"""The demo Space: payload construction and the rendering it feeds.

Two concerns, tested together because they are two halves of one contract —
``fvr.ops.space`` writes ``app/precomputed/responses.json`` and ``app/demo.py``
reads it. A field renamed on one side and not the other would break the live
Space silently, so the last class exercises the real committed payload through
the real rendering functions.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from fvr.config import PROJECT_ROOT
from fvr.data.schema import Question
from fvr.ops.space import bucket_for, build_payload, select_items, softmax, write_payload
from fvr.report.aggregate import Aggregate, RunRecord


def _load_demo() -> ModuleType:
    """Import ``app/demo.py`` by path — ``app/`` ships to the Space, not as a package."""
    path = PROJECT_ROOT / "app" / "demo.py"
    spec = importlib.util.spec_from_file_location("space_demo", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["space_demo"] = module
    spec.loader.exec_module(module)
    return module


demo = _load_demo()


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

    def test_every_bucket_has_a_blurb(self) -> None:
        """A bucket the demo cannot explain should not reach the demo."""
        assert set(demo.BUCKET_ORDER) <= set(demo.BUCKET_BLURBS)


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
        assert demo.load_payload(path) == payload

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


class TestCostRendering:
    @pytest.fixture
    def payload(self) -> dict[str, Any]:
        return {
            "cost": {
                "rate_card": {
                    "gpu_name": "A40",
                    "gpu_usd_per_hour": 0.4,
                    "cpu_usd_per_hour": 0.03,
                    "source_url": "https://example.test",
                    "retrieved": "2026-08-07",
                },
                "arms": {
                    "trained": {"fixed_usd": 2.0, "marginal_usd_per_query": 0.00001},
                    "retrieval": {"fixed_usd": 0.0, "marginal_usd_per_query": 0.00002},
                },
                "crossovers": {"trained|retrieval": 200000},
                "default_volume": 100000,
            }
        }

    def test_amortisation_shrinks_with_volume(self, payload: dict[str, Any]) -> None:
        arm = {"fixed_usd": 2.0, "marginal_usd_per_query": 0.00001}
        assert demo.usd_per_1k(arm, 1) > demo.usd_per_1k(arm, 1_000_000)

    def test_approaches_the_marginal_rate(self) -> None:
        arm = {"fixed_usd": 2.0, "marginal_usd_per_query": 0.00001}
        assert demo.usd_per_1k(arm, 10**12) == pytest.approx(0.01, rel=1e-3)

    def test_rejects_zero_volume(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            demo.usd_per_1k({"fixed_usd": 1.0, "marginal_usd_per_query": 0.0}, 0)

    def test_the_crossover_actually_crosses(self, payload: dict[str, Any]) -> None:
        """The published crossover must be where the ranking really flips."""
        arms = payload["cost"]["arms"]
        before = demo.cost_table(payload, 100)[0][1]
        after = demo.cost_table(payload, 10_000_000)[0][1]
        assert before == "retrieval", "the low-fixed-cost arm should win at low volume"
        assert after == "trained", "the low-marginal arm should win at high volume"
        assert set(arms) == {"trained", "retrieval"}

    def test_summary_names_the_winner(self, payload: dict[str, Any]) -> None:
        assert "`retrieval`" in demo.cost_summary(payload, 100)
        assert "200,000" in demo.cost_summary(payload, 100)


@pytest.fixture(scope="module")
def committed_payload() -> dict[str, Any]:
    """The real ``app/precomputed/responses.json``, as the Space will read it."""
    payload: dict[str, Any] = demo.load_payload()
    return payload


class TestAgainstTheCommittedPayload:
    """The real ``responses.json`` through the real rendering functions."""

    def test_every_item_renders(self, committed_payload: dict[str, Any]) -> None:
        for item in committed_payload["items"]:
            rendered = demo.render_question(item)
            assert item["question"][:40] in rendered
            assert "correct" in rendered, "the gold option must be marked"

    def test_answer_verdicts_agree_with_the_gold_label(
        self, committed_payload: dict[str, Any]
    ) -> None:
        for item in committed_payload["items"]:
            for row in demo.render_answers(item, committed_payload):
                predicted = row[1].split(".")[0]
                expected = "correct" if predicted == item["answer_label"] else "wrong"
                assert row[2] == expected, f"{item['id']} / {row[0]}"

    def test_arms_table_is_ordered_by_accuracy(self, committed_payload: dict[str, Any]) -> None:
        accuracies = [float(row[1].rstrip("%")) for row in demo.arms_table(committed_payload)]
        assert accuracies == sorted(accuracies, reverse=True)

    def test_bucket_filter_narrows_the_picker(self, committed_payload: dict[str, Any]) -> None:
        everything = demo.item_choices(committed_payload, "all")
        one = demo.item_choices(committed_payload, "index_only")
        assert 0 < len(one) < len(everything)

    def test_lookup_by_id_round_trips(self, committed_payload: dict[str, Any]) -> None:
        for _, item_id in demo.item_choices(committed_payload, "all"):
            assert demo.find_item(committed_payload, item_id) is not None
        assert demo.find_item(committed_payload, "no-such-item") is None

    def test_cost_table_covers_every_evaluated_arm(self, committed_payload: dict[str, Any]) -> None:
        named = {row[1] for row in demo.cost_table(committed_payload, 100_000)}
        assert named == {arm["name"] for arm in committed_payload["arms"]}

    def test_provenance_cites_the_frozen_split(self, committed_payload: dict[str, Any]) -> None:
        note = demo.provenance_note(committed_payload)
        assert committed_payload["provenance"]["split_sha256"][:16] in note
        assert "Nothing is generated at demo time" in note
