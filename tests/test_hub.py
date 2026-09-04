"""Publishing guards.

This code puts files on the public internet under the author's name, so the
tests are adversarial about the two ways that goes wrong: shipping a medical
model card that lost its safety warning, and shipping training scratch.

The last test is a drift guard rather than a unit test — it proves the numbers
argued in ``docs/model_card.md`` still match ``results/runs/*.json``. The card
is hand-written on purpose (it argues, and a generated table cannot), so drift
is caught instead of prevented.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fvr.config import PROJECT_ROOT
from fvr.ops.hub import (
    HubTargets,
    UnsafeCardError,
    UnsafeUploadError,
    UploadPlan,
    assert_card_safe,
    card_results,
    plan_adapter_upload,
    plan_space_upload,
)

SAFE_CARD = """# A model
This adapter is not a medical device and is **not for clinical use**.
"""


@pytest.fixture
def fake_adapter(tmp_path: Path) -> Path:
    """A PEFT output directory, including the files that must not be published."""
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"\x00" * 128)
    (adapter / "tokenizer.json").write_text("{}", encoding="utf-8")
    # trl writes its own stub card, and training leaves optimizer state behind.
    (adapter / "README.md").write_text("# Model Card\n[More Information Needed]", encoding="utf-8")
    (adapter / "optimizer.pt").write_bytes(b"\x00" * 4096)
    (adapter / "training_args.bin").write_bytes(b"\x00" * 512)
    return adapter


@pytest.fixture
def card(tmp_path: Path) -> Path:
    path = tmp_path / "model_card.md"
    path.write_text(SAFE_CARD, encoding="utf-8")
    return path


class TestTargets:
    def test_repo_ids_are_namespaced(self) -> None:
        targets = HubTargets(namespace="someone")
        assert targets.adapter_repo == "someone/qwen3-8b-medmcqa-qlora"
        assert targets.space_repo == "someone/fine-tune-vs-rag"


class TestCardSafety:
    def test_accepts_a_card_with_both_statements(self) -> None:
        assert_card_safe(SAFE_CARD)

    def test_survives_reflowing(self) -> None:
        """A line break mid-phrase must not defeat the check."""
        assert_card_safe("not a medical device and is not for\n   clinical use")

    @pytest.mark.parametrize(
        "text",
        [
            "# A model\nA fine adapter for medicine.",
            "This is not a medical device.",  # missing the clinical-use statement
            "Not for clinical use.",  # missing the device statement
        ],
    )
    def test_rejects_a_card_missing_a_statement(self, text: str) -> None:
        with pytest.raises(UnsafeCardError, match="safety statement"):
            assert_card_safe(text)

    def test_the_real_card_passes(self) -> None:
        assert_card_safe((PROJECT_ROOT / "docs" / "model_card.md").read_text(encoding="utf-8"))


class TestAdapterPlan:
    def test_publishes_the_reviewed_card_as_readme(self, fake_adapter: Path, card: Path) -> None:
        plan = plan_adapter_upload(fake_adapter, card, HubTargets(namespace="someone"))
        remote = {str(r): local for local, r in plan.entries}
        assert remote["README.md"] == card, "the Hub renders README.md; it must be our card"

    def test_never_publishes_training_scratch(self, fake_adapter: Path, card: Path) -> None:
        plan = plan_adapter_upload(fake_adapter, card, HubTargets(namespace="someone"))
        published = {remote for _, remote in plan.entries}
        assert "optimizer.pt" not in published
        assert "training_args.bin" not in published

    def test_never_publishes_the_library_stub_card(self, fake_adapter: Path, card: Path) -> None:
        plan = plan_adapter_upload(fake_adapter, card, HubTargets(namespace="someone"))
        assert not any(local == fake_adapter / "README.md" for local, _ in plan.entries)

    def test_publishes_the_weights(self, fake_adapter: Path, card: Path) -> None:
        plan = plan_adapter_upload(fake_adapter, card, HubTargets(namespace="someone"))
        published = {remote for _, remote in plan.entries}
        assert {"adapter_config.json", "adapter_model.safetensors"} <= published

    def test_refuses_an_unsafe_card(self, fake_adapter: Path, tmp_path: Path) -> None:
        bad = tmp_path / "bad.md"
        bad.write_text("# Great medical model\nUse it freely.", encoding="utf-8")
        with pytest.raises(UnsafeCardError):
            plan_adapter_upload(fake_adapter, bad, HubTargets(namespace="someone"))

    def test_refuses_a_directory_that_is_not_an_adapter(self, tmp_path: Path, card: Path) -> None:
        with pytest.raises(FileNotFoundError, match="adapter_config"):
            plan_adapter_upload(tmp_path, card, HubTargets(namespace="someone"))

    def test_a_scratch_file_slipping_into_the_allowlist_is_caught(
        self, fake_adapter: Path, card: Path
    ) -> None:
        """The second guard: even if someone allowlists it, the plan is rejected."""
        with pytest.raises(UnsafeUploadError, match="training scratch"):
            plan_adapter_upload(
                fake_adapter,
                card,
                HubTargets(namespace="someone"),
                allowed=("adapter_config.json", "optimizer.pt"),
            )


class TestSpacePlan:
    def test_preserves_nested_paths(self, tmp_path: Path) -> None:
        app = tmp_path / "app"
        (app / "precomputed").mkdir(parents=True)
        (app / "app.py").write_text("print('hi')", encoding="utf-8")
        (app / "precomputed" / "responses.json").write_text("{}", encoding="utf-8")
        plan = plan_space_upload(app, HubTargets(namespace="someone"))
        assert {remote for _, remote in plan.entries} == {
            "app.py",
            "precomputed/responses.json",
        }

    def test_requires_an_entry_point(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match=r"app\.py"):
            plan_space_upload(tmp_path, HubTargets(namespace="someone"))

    def test_the_real_space_is_publishable(self) -> None:
        plan = plan_space_upload(PROJECT_ROOT / "app", HubTargets(namespace="someone"))
        published = {remote for _, remote in plan.entries}
        assert {"app.py", "demo.py", "requirements.txt"} <= published
        assert "precomputed/responses.json" in published


class TestRenderPlan:
    def test_render_lists_every_entry(self, fake_adapter: Path, card: Path) -> None:
        plan = plan_adapter_upload(fake_adapter, card, HubTargets(namespace="someone"))
        rendered = plan.render()
        assert "someone/qwen3-8b-medmcqa-qlora" in rendered
        for _, remote in plan.entries:
            assert remote in rendered

    def test_total_bytes_sums_the_files(self, tmp_path: Path) -> None:
        a, b = tmp_path / "a.txt", tmp_path / "b.txt"
        a.write_bytes(b"x" * 10)
        b.write_bytes(b"y" * 5)
        plan = UploadPlan(repo_id="x/y", repo_type="model", entries=[(a, "a"), (b, "b")])
        assert plan.total_bytes == 15


class TestCardMatchesResults:
    """The published card must not drift from the committed run JSONs."""

    def test_every_card_row_matches_its_run(self) -> None:
        card_text = (PROJECT_ROOT / "docs" / "model_card.md").read_text(encoding="utf-8")
        rows = card_results(card_text)
        assert rows, "no results rows parsed out of the model card"

        for arm, claimed in rows.items():
            run_path = PROJECT_ROOT / "results" / "runs" / f"{arm}_seed42.json"
            assert run_path.is_file(), f"card cites {arm}, which has no committed run"
            payload = json.loads(run_path.read_text(encoding="utf-8"))

            assert claimed["accuracy"] == pytest.approx(payload["accuracy"], abs=0.0005), arm
            low, high = payload["ci_95"]
            assert claimed["ci_low"] == pytest.approx(low, abs=0.0005), arm
            assert claimed["ci_high"] == pytest.approx(high, abs=0.0005), arm

            p50_ms = payload["latency"]["p50_s"] * 1000
            assert claimed["p50_ms"] == pytest.approx(p50_ms, abs=0.5), arm

            tokens = [p["prompt_tokens"] for p in payload["predictions"]]
            assert claimed["prompt_tokens"] == pytest.approx(sum(tokens) / len(tokens), abs=0.5)
