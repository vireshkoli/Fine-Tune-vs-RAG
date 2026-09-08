"""The HTTP judge client.

The client exists so vLLM never enters this project's venv — it pins torch
hard, this project pins torch 2.11.0+cu128 for a driver that is not ours to
update, and a sync that rewrote torch would break an in-flight experiment grid.
So the judge is an OpenAI-compatible endpoint and this module is stdlib-only.

These tests inject a transport, so nothing here opens a socket.
"""

from __future__ import annotations

import urllib.error
from typing import Any

import pytest

from fvr.config import PROJECT_ROOT
from fvr.eval.judge import score_pointwise
from fvr.eval.judge_client import (
    HttpJudge,
    JudgeConfig,
    JudgeUnavailableError,
    extract_reply,
    load_judge_config,
    server_command,
)


def a_config(**overrides: Any) -> JudgeConfig:
    defaults: dict[str, Any] = {
        "name": "test-judge",
        "repo_id": "org/model",
        "revision": "0123456789abcdef",
        "base_url": "http://127.0.0.1:8000/v1",
        "max_retries": 2,
        "backoff_s": 0.0,
    }
    return JudgeConfig(**{**defaults, **overrides})


def reply(text: str) -> dict[str, Any]:
    return {"choices": [{"message": {"content": text}}]}


class TestRequestShape:
    def test_sends_the_seed_so_variance_is_controlled(self) -> None:
        seen: list[dict[str, Any]] = []

        def transport(_url: str, payload: dict[str, Any]) -> dict[str, Any]:
            seen.append(payload)
            return reply("SCORE: 2")

        judge = HttpJudge(a_config(), transport=transport)
        judge([{"role": "user", "content": "hi"}], seed=7)
        assert seen[0]["seed"] == 7

    def test_posts_to_the_chat_completions_path(self) -> None:
        urls: list[str] = []

        def transport(url: str, _payload: dict[str, Any]) -> dict[str, Any]:
            urls.append(url)
            return reply("SCORE: 1")

        HttpJudge(a_config(base_url="http://host:9/v1/"), transport=transport)([], 0)
        assert urls == ["http://host:9/v1/chat/completions"]

    def test_carries_the_configured_sampling_settings(self) -> None:
        seen: list[dict[str, Any]] = []

        def transport(_url: str, payload: dict[str, Any]) -> dict[str, Any]:
            seen.append(payload)
            return reply("SCORE: 0")

        HttpJudge(a_config(temperature=0.3, max_tokens=16), transport=transport)([], 0)
        assert seen[0]["temperature"] == 0.3
        assert seen[0]["max_tokens"] == 16

    def test_counts_its_calls(self) -> None:
        judge = HttpJudge(a_config(), transport=lambda _u, _p: reply("SCORE: 2"))
        for seed in range(3):
            judge([], seed)
        assert judge.calls == 3


class TestFailureHandling:
    def test_retries_a_transient_failure_then_succeeds(self) -> None:
        attempts = {"n": 0}

        def transport(_url: str, _payload: dict[str, Any]) -> dict[str, Any]:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise urllib.error.URLError("connection refused")
            return reply("SCORE: 2")

        judge = HttpJudge(a_config(), transport=transport)
        assert judge([], 0) == "SCORE: 2"
        assert attempts["n"] == 2

    def test_gives_up_with_an_actionable_error(self) -> None:
        def transport(_url: str, _payload: dict[str, Any]) -> dict[str, Any]:
            raise urllib.error.URLError("connection refused")

        with pytest.raises(JudgeUnavailableError, match="judge-server"):
            HttpJudge(a_config(), transport=transport)([], 0)

    def test_a_dead_endpoint_is_not_counted_as_a_call(self) -> None:
        def transport(_url: str, _payload: dict[str, Any]) -> dict[str, Any]:
            raise TimeoutError

        judge = HttpJudge(a_config(), transport=transport)
        with pytest.raises(JudgeUnavailableError):
            judge([], 0)
        assert judge.calls == 0


class TestResponseParsing:
    def test_extracts_the_assistant_text(self) -> None:
        assert extract_reply(reply("VERDICT: A")) == "VERDICT: A"

    @pytest.mark.parametrize(
        "response",
        [{}, {"choices": []}, {"choices": [{}]}, {"choices": [{"message": {}}]}],
    )
    def test_a_malformed_response_raises_rather_than_returning_empty(
        self, response: dict[str, Any]
    ) -> None:
        """An empty string would be scored as an unparseable verdict.

        That would file a broken endpoint under "judge flakiness" — a plausible
        statistic hiding an outage.
        """
        with pytest.raises(JudgeUnavailableError):
            extract_reply(response)

    def test_null_content_raises(self) -> None:
        with pytest.raises(JudgeUnavailableError, match="null"):
            extract_reply({"choices": [{"message": {"content": None}}]})


class TestItSatisfiesTheJudgeProtocol:
    def test_plugs_straight_into_score_pointwise(self) -> None:
        """The contract that matters: HttpJudge *is* a JudgeFn."""
        scores = {0: "SCORE: 2", 1: "SCORE: 1", 2: "SCORE: 2"}

        def transport(_url: str, payload: dict[str, Any]) -> dict[str, Any]:
            return reply(scores[payload["seed"]])

        judge = HttpJudge(a_config(), transport=transport)
        result = score_pointwise("q1", "Q?", "ref", "cand", judge, seeds=(0, 1, 2))
        assert result.scores == (2, 1, 2)
        assert result.sd > 0


class TestShippedConfig:
    def test_the_committed_config_loads_and_is_pinned(self) -> None:
        config = load_judge_config(PROJECT_ROOT / "configs" / "eval" / "judge.yaml")
        assert len(config.revision) >= 7
        assert config.revision != "main"

    def test_the_judge_is_cross_family_from_every_arm(self) -> None:
        """Self-preference bias: a judge must not share lineage with an arm."""
        config = load_judge_config(PROJECT_ROOT / "configs" / "eval" / "judge.yaml")
        assert "qwen" not in config.repo_id.lower()

    def test_temperature_is_nonzero_so_variance_is_measurable(self) -> None:
        """At temperature 0 the reported judge SD would be a meaningless zero."""
        config = load_judge_config(PROJECT_ROOT / "configs" / "eval" / "judge.yaml")
        assert config.temperature > 0
        assert len(config.seeds) > 1

    def test_server_command_pins_the_revision(self) -> None:
        config = load_judge_config(PROJECT_ROOT / "configs" / "eval" / "judge.yaml")
        command = server_command(config, device=1)
        assert config.revision in command
        assert "CUDA_VISIBLE_DEVICES=1" in command
