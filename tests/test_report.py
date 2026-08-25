"""Report-layer tests.

The load-bearing one is ``test_refuses_to_mix_test_splits``: arms scored on
different data are not comparable, and nothing else in the pipeline would
notice. The rest keep the generated tables honest about what they contain.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fvr.config import PROJECT_ROOT
from fvr.eval.metrics import mcnemar
from fvr.report.aggregate import Aggregate, RunRecord
from fvr.report.charts import SERIES_DARK, SERIES_LIGHT, Theme
from fvr.report.tables import comparison_table, cost_table, provenance_note, results_table

RESULTS = PROJECT_ROOT / "results"


def a_payload(arm: str = "base", *, accuracy: float = 0.6, split: str = "abc123") -> dict[str, Any]:
    return {
        "arm": arm,
        "seed": 42,
        "split_sha256": split,
        "n_items": 3,
        "accuracy": accuracy,
        "ci_95": [accuracy - 0.03, accuracy + 0.03],
        "minimum_detectable_effect": 0.06,
        "latency": {"p50_s": 0.05, "p95_s": 0.06},
        "per_subject": {"Anatomy": {"n": 40, "accuracy": accuracy}},
        "groundedness": None,
        "retrieval": None,
        "device_occupancy": {"exclusive": True},
        "model": {"revision": "deadbeef"},
        "environment": {"git_sha": "abc", "gpu": "A40", "torch": "2.11"},
        "predictions": [
            {"question_id": "q1", "predicted_idx": 0, "prompt_tokens": 100},
            {"question_id": "q2", "predicted_idx": 1, "prompt_tokens": 110},
            {"question_id": "q3", "predicted_idx": 2, "prompt_tokens": 120},
        ],
    }


def an_aggregate(*payloads: dict[str, Any]) -> Aggregate:
    return Aggregate(
        runs=[
            RunRecord(arm=p["arm"], seed=p["seed"], path=Path(f"{p['arm']}.json"), payload=p)
            for p in payloads
        ]
    )


class TestAggregate:
    def test_refuses_to_mix_test_splits(self) -> None:
        """Arms scored on different data are not comparable."""
        aggregate = an_aggregate(a_payload("base", split="aaa"), a_payload("qlora", split="bbb"))
        with pytest.raises(ValueError, match="not comparable"):
            aggregate.assert_same_split()

    def test_accepts_a_single_split(self) -> None:
        an_aggregate(a_payload("base"), a_payload("qlora")).assert_same_split()

    def test_arms_preserve_first_seen_order(self) -> None:
        aggregate = an_aggregate(a_payload("qlora"), a_payload("base"))
        assert aggregate.arms == ["qlora", "base"]

    def test_seed_summary_reports_spread(self) -> None:
        aggregate = an_aggregate(a_payload("base", accuracy=0.60), a_payload("base", accuracy=0.64))
        mean, sd = aggregate.seed_summary("base")
        assert mean == pytest.approx(0.62)
        assert sd > 0

    def test_mean_prompt_tokens(self) -> None:
        assert an_aggregate(a_payload()).for_arm("base")[0].mean_prompt_tokens == pytest.approx(110)

    def test_compare_pairs_on_shared_items(self) -> None:
        aggregate = an_aggregate(a_payload("base"), a_payload("qlora"))
        gold: dict[str, int | None] = {"q1": 0, "q2": 1, "q3": 9}
        result = aggregate.compare("base", "qlora", gold)
        assert result.delta == 0.0

    def test_compare_raises_for_a_missing_arm(self) -> None:
        with pytest.raises(KeyError):
            an_aggregate(a_payload("base")).compare("base", "absent", {})


class TestTables:
    def test_results_table_has_a_row_per_arm(self) -> None:
        table = results_table(an_aggregate(a_payload("base"), a_payload("rag-external")))
        assert "`base`" in table and "`rag-external`" in table

    def test_headline_filter_drops_diagnostic_arms(self) -> None:
        aggregate = an_aggregate(a_payload("base"), a_payload("rag-parity"))
        table = results_table(aggregate, headline_only=True)
        assert "`base`" in table
        assert "`rag-parity`" not in table, "diagnostics belong in REPORT, not the README table"

    def test_contended_runs_are_flagged(self) -> None:
        payload = a_payload("base")
        payload["device_occupancy"] = {"exclusive": False, "foreign_mib": 35670}
        assert "⚠" in results_table(an_aggregate(payload))

    def test_comparison_table_shows_discordant_counts(self) -> None:
        """Discordant counts separate a true null from an underpowered one."""
        result = mcnemar([True] * 60 + [False] * 40, [True] * 55 + [False] * 45)
        table = comparison_table([("a", "b", result)])
        assert "Discordant" in table
        assert f"{result.n_a_only}/{result.n_b_only}" in table

    def test_cost_table_formats_dollars(self) -> None:
        table = cost_table([100, 1000], {"base": [0.001, 0.0005]})
        assert "$1.000" in table and "100" in table

    def test_provenance_names_the_split_and_commit(self) -> None:
        note = provenance_note(an_aggregate(a_payload()))
        assert "abc123" in note and "make report" in note

    def test_provenance_flags_mixed_splits(self) -> None:
        note = provenance_note(an_aggregate(a_payload(split="a"), a_payload(split="b")))
        assert "MIXED" in note


class TestThemes:
    def test_light_and_dark_use_different_steps(self) -> None:
        # Dark is a selected set for the dark surface, not an inverted light one.
        assert Theme.light().series != Theme.dark().series

    def test_palettes_are_the_documented_order(self) -> None:
        assert Theme.light().series == SERIES_LIGHT
        assert Theme.dark().series == SERIES_DARK

    def test_enough_slots_for_every_arm(self) -> None:
        from fvr.inference.arms import ARMS

        assert len(SERIES_LIGHT) >= len(ARMS)


class TestCommittedArtifacts:
    def test_tables_are_committed(self) -> None:
        assert (RESULTS / "tables.md").is_file(), "run `make report`"

    def test_figures_exist_for_both_themes(self) -> None:
        for name in ("cost-crossover", "arms"):
            assert (RESULTS / "figures" / f"{name}.png").is_file()
            assert (RESULTS / "figures" / f"{name}-dark.png").is_file()

    def test_every_committed_run_shares_the_frozen_split(self) -> None:
        aggregate = Aggregate.load(RESULTS / "runs")
        if aggregate.runs:
            aggregate.assert_same_split()

    def test_committed_runs_match_the_split_manifest(self) -> None:
        manifest = json.loads((RESULTS / "split_manifest.json").read_text(encoding="utf-8"))
        aggregate = Aggregate.load(RESULTS / "runs")
        if aggregate.runs:
            assert aggregate.split_hashes() == {manifest["splits"]["test"]["sha256"]}
