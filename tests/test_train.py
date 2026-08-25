"""Training-pipeline tests.

The important ones assert two properties that no amount of downstream care
could recover if broken: the test split is unreachable from training, and the
training prompt is byte-identical to the evaluation prompt.

All CPU, no GPU, no network — the model is never constructed here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fvr.config import PROJECT_ROOT
from fvr.data.schema import OPTION_LABELS, Question
from fvr.data.sft import SFTStats, build_sft_dataset, build_sft_record
from fvr.prompts.templates import build_prompt
from fvr.train.callbacks import OOMGuardCallback, latest_checkpoint
from fvr.train.config import TrainConfig, load_train_config

CONFIGS = PROJECT_ROOT / "configs" / "train"


def a_question(qid: str = "q1", *, answer: int = 1, explanation: str | None = None) -> Question:
    return Question(
        id=qid,
        question="Which vessel supplies the myocardium?",
        options=["Portal vein", "Coronary artery", "Aorta", "Vena cava"],
        answer_idx=answer,
        explanation=explanation,
        subject="Anatomy",
    )


class TestSFTRecord:
    def test_target_starts_with_the_answer_letter(self) -> None:
        """First-token scoring is how every arm is graded; prose first breaks it."""
        record = build_sft_record(a_question(answer=1), include_explanation=False)
        assert record is not None
        assert record.target == "B"
        assert record.target[0] in OPTION_LABELS

    def test_letter_still_leads_when_a_rationale_is_attached(self) -> None:
        record = build_sft_record(
            a_question(answer=2, explanation="The aorta is the largest artery in the body."),
            include_explanation=True,
        )
        assert record is not None
        assert record.target.startswith("C")
        assert "largest artery" in record.target

    def test_rationale_is_omitted_when_disabled(self) -> None:
        record = build_sft_record(
            a_question(explanation="Some explanation of adequate length for a chunk."),
            include_explanation=False,
        )
        assert record is not None
        assert record.target == "B"

    def test_answer_key_boilerplate_is_stripped_from_the_rationale(self) -> None:
        # Training on "Ans. is 'b'" teaches the letter, not the medicine.
        record = build_sft_record(
            a_question(explanation="Ans. is 'b' i.e., The coronary artery supplies the heart."),
            include_explanation=True,
        )
        assert record is not None
        assert "Ans." not in record.target

    def test_unlabelled_questions_produce_nothing(self) -> None:
        assert build_sft_record(a_question(answer=None), include_explanation=True) is None  # type: ignore[arg-type]

    def test_prompt_is_byte_identical_to_the_evaluation_prompt(self) -> None:
        """The fairness control: no separate training-prompt implementation.

        If these drifted, the fine-tuned arm would gain an edge from prompt
        familiarity that has nothing to do with domain knowledge.
        """
        question = a_question()
        record = build_sft_record(question, include_explanation=False)
        assert record is not None
        expected = build_prompt(question).as_messages()
        assert record.messages[:-1] == expected

    def test_the_record_ends_with_an_assistant_turn(self) -> None:
        record = build_sft_record(a_question(), include_explanation=False)
        assert record is not None
        assert record.messages[-1]["role"] == "assistant"


class TestSFTDataset:
    def test_excludes_held_out_content_under_a_different_id(self) -> None:
        """The leak that an id filter would miss."""
        held_out = a_question("test-1")
        duplicate = a_question("train-9", explanation="A perfectly usable explanation here.")
        records, stats = build_sft_dataset(
            [duplicate], forbidden_content_hashes=frozenset({held_out.content_hash()})
        )
        assert records == []
        assert stats.dropped_leaked == 1

    def test_keeps_unrelated_rows(self) -> None:
        other = Question(
            id="train-2",
            question="Which nerve innervates the diaphragm?",
            options=["Phrenic", "Vagus", "Ulnar", "Radial"],
            answer_idx=0,
        )
        records, stats = build_sft_dataset(
            [other], forbidden_content_hashes=frozenset({a_question().content_hash()})
        )
        assert len(records) == 1
        assert stats.kept == 1

    def test_subsampling_is_deterministic(self) -> None:
        rows = [a_question(f"t{i}") for i in range(100)]
        first, _ = build_sft_dataset(
            rows, forbidden_content_hashes=frozenset(), max_samples=10, seed=7
        )
        second, _ = build_sft_dataset(
            rows, forbidden_content_hashes=frozenset(), max_samples=10, seed=7
        )
        assert [r.question_id for r in first] == [r.question_id for r in second]

    def test_subsampling_shuffles_rather_than_truncating(self) -> None:
        """Taking the first N would follow file order, which is subject-sorted."""
        rows = [a_question(f"t{i:03d}") for i in range(200)]
        picked, _ = build_sft_dataset(
            rows, forbidden_content_hashes=frozenset(), max_samples=20, seed=1
        )
        assert [r.question_id for r in picked] != [f"t{i:03d}" for i in range(20)]

    def test_no_subsampling_keeps_everything(self) -> None:
        rows = [a_question(f"t{i}") for i in range(30)]
        records, _ = build_sft_dataset(rows, forbidden_content_hashes=frozenset())
        assert len(records) == 30

    def test_stats_account_for_every_row(self) -> None:
        rows = [a_question(f"t{i}") for i in range(10)] + [a_question("dup")]
        _, stats = build_sft_dataset(
            rows, forbidden_content_hashes=frozenset({a_question().content_hash()})
        )
        assert stats.seen == 11
        assert stats.dropped_leaked == 11  # every row shares the same content


class TestNoTestSplitInTraining:
    """The single most important guarantee in this phase."""

    def test_the_training_script_never_reads_the_test_ids(self) -> None:
        source = (PROJECT_ROOT / "scripts" / "04_train.py").read_text(encoding="utf-8")
        # It may read split_ids to *exclude* test content, but must never build
        # a dataset from it. Assert the only use of "test" is in the ban list.
        assert 'split_ids["test"]' in source, "test ids must be loaded, to be excluded"
        assert "held_out_ids" in source
        assert 'val_ids = set(split_ids["val"])' in source
        # No dataset is ever constructed from test ids.
        assert "test_ids" not in source

    def test_committed_split_ids_are_disjoint(self) -> None:
        ids = json.loads((PROJECT_ROOT / "results" / "split_ids.json").read_text(encoding="utf-8"))
        assert not set(ids["test"]) & set(ids["val"])


class TestTrainConfig:
    @pytest.mark.parametrize("name", ["qlora_r16", "smoke"])
    def test_committed_configs_parse(self, name: str) -> None:
        assert load_train_config(CONFIGS / f"{name}.yaml").name

    def test_effective_batch_size(self) -> None:
        config = TrainConfig(name="t", per_device_batch_size=4, gradient_accumulation_steps=4)
        assert config.effective_batch_size == 16

    def test_checkpoint_selection_never_uses_test(self) -> None:
        assert load_train_config(CONFIGS / "qlora_r16.yaml").metric_for_best_model == "eval_loss"

    def test_gradient_checkpointing_is_on_by_default(self) -> None:
        # Mandatory below 24GB; kept on so the recipe is portable.
        assert load_train_config(CONFIGS / "qlora_r16.yaml").gradient_checkpointing

    def test_quantisation_is_four_bit(self) -> None:
        assert load_train_config(CONFIGS / "qlora_r16.yaml").quantization == "nf4"

    def test_adapters_cover_mlp_not_just_attention(self) -> None:
        modules = load_train_config(CONFIGS / "qlora_r16.yaml").lora.target_modules
        assert {"gate_proj", "up_proj", "down_proj"} <= set(modules)

    def test_save_steps_bound_the_loss_from_an_interruption(self) -> None:
        config = load_train_config(CONFIGS / "qlora_r16.yaml")
        assert config.save_steps <= 250, "an interruption should cost minutes, not hours"

    def test_smoke_config_is_actually_small(self) -> None:
        config = load_train_config(CONFIGS / "smoke.yaml")
        assert config.max_train_samples is not None
        assert config.max_train_samples <= 256
        assert config.report_to == "none", "the smoke run must not spam W&B"

    def test_rejects_unknown_keys(self) -> None:
        with pytest.raises(ValueError, match="typo"):
            TrainConfig(name="t", typo=1)  # type: ignore[call-arg]


class TestCheckpointResumption:
    def test_finds_the_newest_checkpoint_numerically(self, tmp_path: Path) -> None:
        """`checkpoint-1000` sorts before `checkpoint-200` as a string."""
        for step in (200, 1000, 400):
            (tmp_path / f"checkpoint-{step}").mkdir()
        found = latest_checkpoint(tmp_path)
        assert found is not None and found.name == "checkpoint-1000"

    def test_returns_none_when_empty(self, tmp_path: Path) -> None:
        assert latest_checkpoint(tmp_path) is None

    def test_returns_none_for_a_missing_directory(self, tmp_path: Path) -> None:
        assert latest_checkpoint(tmp_path / "absent") is None

    def test_ignores_non_checkpoint_directories(self, tmp_path: Path) -> None:
        (tmp_path / "adapter").mkdir()
        (tmp_path / "checkpoint-notanumber").mkdir()
        assert latest_checkpoint(tmp_path) is None


class TestOOMGuard:
    def test_marker_records_the_step_and_how_to_resume(self, tmp_path: Path) -> None:
        guard = OOMGuardCallback(tmp_path, per_device_batch_size=8, seq_length=1024)

        class State:
            global_step = 437

        marker = guard.write_marker(State(), RuntimeError("CUDA out of memory"))
        payload = json.loads(marker.read_text(encoding="utf-8"))
        assert payload["global_step"] == 437
        assert "--resume" in payload["resume_with"]

    def test_marker_suggests_halving_the_batch(self, tmp_path: Path) -> None:
        guard = OOMGuardCallback(tmp_path, per_device_batch_size=8, seq_length=1024)

        class State:
            global_step = 1

        payload = json.loads(
            guard.write_marker(State(), RuntimeError("oom")).read_text(encoding="utf-8")
        )
        assert any("halve per_device_batch_size to 4" in s for s in payload["suggestions"])
        assert any("gradient_checkpointing" in s for s in payload["suggestions"])

    def test_batch_size_suggestion_never_reaches_zero(self, tmp_path: Path) -> None:
        guard = OOMGuardCallback(tmp_path, per_device_batch_size=1, seq_length=512)

        class State:
            global_step = 1

        payload = json.loads(
            guard.write_marker(State(), RuntimeError("oom")).read_text(encoding="utf-8")
        )
        assert any("to 1" in s for s in payload["suggestions"])


class TestStats:
    def test_summary_is_populated(self) -> None:
        stats = SFTStats(seen=100, kept=90, with_explanation=80, target_chars=27000)
        text = stats.summary()
        assert "90" in text and "300 chars" in text

    def test_mean_target_chars_handles_empty(self) -> None:
        assert SFTStats().mean_target_chars == 0.0
