"""Talks to the judge over HTTP, so vLLM never enters this project's venv.

The obvious implementation imports vLLM and loads the 70B judge in-process.
That was rejected for a concrete reason: vLLM pins torch hard, this project
pins torch 2.11.0+cu128 against a CUDA 12.8 driver that is not ours to update,
and a ``uv sync`` that pulls vLLM in would rewrite torch underneath a running
60-GPU-hour experiment grid. A dependency that can break an in-flight run is
not worth the convenience.

So the judge is a *server*, spoken to over the OpenAI-compatible chat API that
vLLM, llama.cpp, Ollama, TGI and every hosted provider implement. Consequences,
all good:

* This module needs no third-party package at all — stdlib ``urllib`` only.
* The judge is swappable. Offline, point it at local vLLM; if someone later has
  an API key, point it at a hosted model and change nothing else.
* The judge can run in its own venv, or on another machine, or not at all —
  the MCQ arms never touch it.

``make judge-server`` prints the exact command; the judge deliberately is not
launched from inside this process, because a subprocess owning 37 GiB of GPU
memory that outlives its parent is how a shared machine gets wedged.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

#: Injected in tests. Takes (url, payload) and returns the decoded response.
Transport = Callable[[str, dict[str, Any]], dict[str, Any]]


class JudgeUnavailableError(Exception):
    """The judge endpoint could not be reached or kept failing."""


class JudgeConfig(BaseModel):
    """From ``configs/eval/judge.yaml``."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    name: str
    repo_id: str
    #: Pinned like every other model here: the benchmark must still reproduce
    #: once this machine is wiped, and "latest" is not a reproducible judge.
    revision: str = Field(min_length=7)
    base_url: str = "http://127.0.0.1:8000/v1"
    #: Non-zero on purpose. The judge's seed-to-seed spread is a reported
    #: number, and at temperature 0 it would be identically zero — which says
    #: nothing about how stable the judge actually is.
    temperature: float = 0.3
    max_tokens: int = 16
    seeds: tuple[int, ...] = (0, 1, 2)
    timeout_s: float = 120.0
    max_retries: int = 3
    #: Seconds between retries, doubled each attempt.
    backoff_s: float = 2.0

    def describe(self) -> dict[str, Any]:
        return {
            "judge": self.name,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "temperature": self.temperature,
            "seeds": list(self.seeds),
            "max_tokens": self.max_tokens,
        }


def load_judge_config(path: Path | str) -> JudgeConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return JudgeConfig(**raw)


def _urllib_transport(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return dict(json.loads(response.read().decode("utf-8")))


@dataclass
class HttpJudge:
    """An OpenAI-compatible chat endpoint, exposed as a :data:`JudgeFn`.

    Callable as ``judge(messages, seed)``, which is what
    :mod:`fvr.eval.judge` expects.
    """

    config: JudgeConfig
    transport: Transport | None = None
    #: Requests actually issued. Judging is thousands of calls, so this is worth
    #: recording next to the results rather than estimating afterwards.
    calls: int = 0

    def __post_init__(self) -> None:
        # Judging runs across a thread pool; an unguarded `+=` drops counts.
        self._lock = threading.Lock()

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        if self.transport is not None:
            return self.transport(url, payload)
        return _urllib_transport(url, payload, self.config.timeout_s)

    def __call__(self, messages: list[dict[str, str]], seed: int) -> str:
        payload: dict[str, Any] = {
            "model": self.config.repo_id,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "seed": seed,
        }

        last: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                response = self._post(payload)
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                # Transient: a busy server, a dropped connection. Retried.
                last = exc
                time.sleep(self.config.backoff_s * (2**attempt))
                continue
            with self._lock:
                self.calls += 1
            return extract_reply(response)

        raise JudgeUnavailableError(
            f"judge at {self.config.base_url} failed {self.config.max_retries} times: {last}. "
            "Is the server running? See `make judge-server`."
        ) from last


def extract_reply(response: dict[str, Any]) -> str:
    """Pull the assistant text out of an OpenAI-compatible response.

    Raises rather than returning "" on a malformed response: an empty string
    would be scored as an unparseable verdict and quietly counted as judge
    flakiness, hiding a broken endpoint behind a plausible-looking statistic.
    """
    try:
        choices = response["choices"]
        content = choices[0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise JudgeUnavailableError(f"malformed judge response: {str(response)[:300]}") from exc
    if content is None:
        raise JudgeUnavailableError("judge returned a null message content")
    return str(content)


def server_command(
    config: JudgeConfig,
    *,
    device: int = 1,
    port: int = 8000,
    vllm_bin: str = "vllm",
    hf_home: str | None = None,
) -> str:
    """The exact command to serve this judge.

    Printed rather than executed. A subprocess holding 37 GiB of GPU memory that
    outlives its parent is how a shared machine gets wedged, so starting and
    stopping it stays a deliberate step.

    ``hf_home`` points vLLM at the project's artifact tree, where the pinned
    snapshot was downloaded, and turns the Hub off: the weights are already on
    disk, and on a flaky connection a revision check at start-up is a way for the
    server to fail for no reason. The tokenizer is pinned to the same revision as
    the weights, since vLLM otherwise resolves it separately.
    """
    env = f"CUDA_VISIBLE_DEVICES={device}"
    if hf_home is not None:
        # This project pins HUGGINGFACE_HUB_CACHE *directly* at .artifacts/hub, so
        # snapshots live at .artifacts/hub/models--* rather than under a hub/
        # subfolder. HF_HOME alone would send vLLM to .artifacts/hub/hub/ and it
        # would report the fully-downloaded judge as missing — which it did.
        env += f" HF_HOME={hf_home} HF_HUB_CACHE={hf_home} HF_HUB_OFFLINE=1"
        # vLLM's torch.compile cache defaults to ~/.cache/vllm — outside the one
        # directory teardown may delete, on a machine shared with other people's
        # vLLM installs. Keep it beside the weights.
        env += f" VLLM_CACHE_ROOT={Path(hf_home).parent / 'vllm-cache'}"
    return (
        f"{env} {vllm_bin} serve {config.repo_id} "
        f"--revision {config.revision} --tokenizer-revision {config.revision} "
        f"--served-model-name {config.repo_id} "
        # 2048, not 4096: the 38 GiB AWQ weights leave ~0.75 GiB of KV cache on a
        # 46 GiB card at 0.90, and vLLM refuses to start unless one full-length
        # request fits (4096 needs 1.25 GiB). Rubric prompts are ~500 tokens, so
        # 2048 costs nothing; 0.95 buys back ~2 GiB of KV for concurrency.
        f"--port {port} --max-model-len 2048 --gpu-memory-utilization 0.95 "
        "--max-num-seqs 16 --quantization awq_marlin"
    )
