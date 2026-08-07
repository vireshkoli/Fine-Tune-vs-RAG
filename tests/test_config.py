"""Config and cache-isolation tests.

The cache-isolation cases are the important ones: they encode the promise that
this project never writes into the shared ``~/.cache/huggingface`` on the lab
GPU, and that teardown can only ever reach inside ``.artifacts/``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from fvr.config import (
    ARTIFACTS_DIRNAME,
    PROJECT_ROOT,
    Paths,
    ProjectConfig,
    Secrets,
    bootstrap_env,
    load_config,
)


class TestPaths:
    def test_every_path_is_absolute(self) -> None:
        for name, value in Paths():
            assert isinstance(value, Path)
            assert value.is_absolute(), f"{name} is relative: {value}"

    def test_all_paths_live_inside_the_project(self) -> None:
        for name, value in Paths():
            assert value.is_relative_to(PROJECT_ROOT), f"{name} escapes the project: {value}"

    def test_deletable_paths_are_confined_to_artifacts(self) -> None:
        paths = Paths()
        for target in paths.deletable():
            assert target.is_relative_to(paths.artifacts), f"{target} is outside .artifacts/"

    def test_results_are_not_deletable(self) -> None:
        # results/ is committed to git and must survive teardown.
        paths = Paths()
        assert paths.results not in paths.deletable()
        assert not paths.results.is_relative_to(paths.artifacts)

    def test_shared_hf_cache_is_never_a_deletable_target(self) -> None:
        shared = (Path.home() / ".cache" / "huggingface").resolve()
        paths = Paths()
        for target in paths.deletable():
            assert target != shared
            assert not shared.is_relative_to(target), f"deleting {target} would remove {shared}"


class TestProjectConfig:
    def test_defaults(self) -> None:
        config = ProjectConfig()
        assert config.seed == 42
        assert config.inference_device != config.judge_device

    def test_rejects_unknown_keys(self) -> None:
        # extra="forbid" turns a typo in YAML into an error instead of a silent no-op.
        with pytest.raises(ValueError, match="typo_key"):
            ProjectConfig(typo_key=1)  # type: ignore[call-arg]

    def test_is_immutable(self) -> None:
        config = ProjectConfig()
        with pytest.raises(ValueError):
            config.seed = 7  # type: ignore[misc]

    def test_committed_base_yaml_parses(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
        assert config.min_free_disk_gib >= 74.0, "must reserve room for the ~74 GiB footprint"

    def test_missing_file_falls_back_to_defaults(self, tmp_path: Path) -> None:
        assert load_config(tmp_path / "absent.yaml") == ProjectConfig()

    def test_non_mapping_yaml_is_rejected(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("- just\n- a list\n", encoding="utf-8")
        with pytest.raises(TypeError, match="mapping"):
            load_config(bad)

    def test_yaml_values_override_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump({"seed": 1234}), encoding="utf-8")
        assert load_config(path).seed == 1234


class TestSecrets:
    def test_absent_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setattr(Secrets, "model_config", {**Secrets.model_config, "env_file": None})
        assert Secrets().hf_token is None

    def test_read_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HF_TOKEN", "hf_test_value")
        assert Secrets().hf_token == "hf_test_value"

    def test_base_yaml_contains_no_secret_fields(self) -> None:
        # Secrets must be unreachable from YAML, so they cannot be committed.
        raw = yaml.safe_load((PROJECT_ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
        assert set(raw) & {"hf_token", "wandb_api_key", "token", "api_key"} == set()


class TestBootstrapEnv:
    def test_points_every_cache_into_artifacts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE"):
            monkeypatch.delenv(var, raising=False)

        paths = bootstrap_env()

        for var in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE"):
            resolved = Path(os.environ[var]).resolve()
            assert resolved.is_relative_to(paths.artifacts), f"{var} leaks to {resolved}"

    def test_never_points_at_the_shared_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HF_HOME", raising=False)
        shared = (Path.home() / ".cache" / "huggingface").resolve()
        assert (
            not Path(os.environ.get("HF_HOME") or bootstrap_env().hub)
            .resolve()
            .is_relative_to(shared)
        )

    def test_respects_a_preset_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # setdefault, not overwrite — Docker and CI set these themselves.
        monkeypatch.setenv("HF_HOME", "/tmp/preset-hf-home")
        bootstrap_env()
        assert os.environ["HF_HOME"] == "/tmp/preset-hf-home"

    def test_artifacts_directory_is_gitignored(self) -> None:
        ignored = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
        assert f"{ARTIFACTS_DIRNAME}/" in ignored, "the artifact tree must never be committed"
