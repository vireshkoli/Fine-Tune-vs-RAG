"""The live Space vendors the benchmark's rules; this asserts the copy is exact.

``space_live/bench.py`` has no dependency on ``fvr`` so the Space installs
only its runtime. The price is duplication, and the guard against duplication
drifting is here: every constant, every built prompt, the retrieval query, the
context budget and the option-token logic are compared against the package.
"""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType

import pytest

from fvr.config import PROJECT_ROOT
from fvr.data.schema import OPTION_LABELS, Passage, Question
from fvr.inference import mcq
from fvr.prompts import templates
from fvr.retrieval.embed import load_embedder_config
from fvr.retrieval.retriever import apply_context_budget, load_retrieval_config


@pytest.fixture(scope="module")
def bench() -> ModuleType:
    path = PROJECT_ROOT / "space_live" / "bench.py"
    spec = importlib.util.spec_from_file_location("space_live_bench", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["space_live_bench"] = module
    spec.loader.exec_module(module)
    return module


def a_question() -> Question:
    return Question(
        id="q1",
        question="  Which vessel supplies the SA node in most people?  ",
        options=["Right coronary artery", "Left anterior descending", "Circumflex", "PDA"],
        answer_idx=0,
        subject="Anatomy",
    )


def some_passages() -> list[Passage]:
    return [
        Passage(
            id=f"p{i}",
            text=f"Passage {i} text. " * (10 * i + 1),
            corpus="parity",
            score=1.0 - i / 10,
        )
        for i in range(1, 6)
    ]


class TestConstants:
    def test_prompt_strings_match(self, bench: ModuleType) -> None:
        assert bench.SYSTEM_PROMPT == templates.SYSTEM_PROMPT
        assert bench.ANSWER_INSTRUCTION == templates.ANSWER_INSTRUCTION
        assert bench.FREETEXT_SYSTEM_PROMPT == templates.FREETEXT_SYSTEM_PROMPT
        assert bench.FREETEXT_INSTRUCTION == templates.FREETEXT_INSTRUCTION
        assert bench.CONTEXT_HEADER == templates.CONTEXT_HEADER
        assert bench.CONTEXT_SEPARATOR == templates.CONTEXT_SEPARATOR
        assert bench.OPTION_LABELS == OPTION_LABELS

    def test_retrieval_settings_match_the_default_config(self, bench: ModuleType) -> None:
        embedder = load_embedder_config("configs/retrieval/bge_large.yaml")
        retrieval = load_retrieval_config("configs/retrieval/bge_large.yaml")
        assert embedder.query_prefix == bench.QUERY_PREFIX
        assert retrieval.top_k == bench.TOP_K
        assert retrieval.max_context_chars == bench.MAX_CONTEXT_CHARS
        assert bench.MAX_NEW_TOKENS == 96  # the MedMCQA free-text bound

    def test_app_pins_the_same_model_revisions(self) -> None:
        app = (PROJECT_ROOT / "space_live" / "app.py").read_text(encoding="utf-8")
        from fvr.models.loader import load_model_config

        base = load_model_config("configs/model/qwen3-8b.yaml")
        embedder = load_embedder_config("configs/retrieval/bge_large.yaml")
        assert f'BASE_REVISION = "{base.revision}"' in app
        assert f'EMBEDDER_REVISION = "{embedder.revision}"' in app
        assert f'BASE_REPO = "{base.repo_id}"' in app


class TestPrompts:
    @pytest.mark.parametrize("n_passages", [0, 3])
    def test_freetext_prompt_is_byte_identical(self, bench: ModuleType, n_passages: int) -> None:
        q = a_question()
        passages = some_passages()[:n_passages]
        ours = templates.build_freetext_prompt(q, passages)
        theirs = bench.build_freetext_prompt(
            q.question, [bench.Passage(id=p.id, text=p.text) for p in passages]
        )
        assert theirs.system == ours.system
        assert theirs.user == ours.user
        assert theirs.as_messages() == ours.as_messages()

    @pytest.mark.parametrize("n_passages", [0, 3])
    def test_mcq_prompt_is_byte_identical(self, bench: ModuleType, n_passages: int) -> None:
        q = a_question()
        passages = some_passages()[:n_passages]
        ours = templates.build_prompt(q, passages)
        theirs = bench.build_prompt(
            q.question, list(q.options), [bench.Passage(id=p.id, text=p.text) for p in passages]
        )
        assert theirs.system == ours.system
        assert theirs.user == ours.user


class TestRetrieval:
    def test_query_text_matches_both_modes(self, bench: ModuleType) -> None:
        from fvr.retrieval.retriever import Retriever

        q = a_question()
        for with_options in (True, False):
            ours = Retriever._query_text(None, q, with_options=with_options)  # type: ignore[arg-type]
            theirs = bench.retrieval_query(q.question, list(q.options), with_options=with_options)
            assert theirs == ours

    def test_context_budget_matches(self, bench: ModuleType) -> None:
        passages = some_passages()
        ours = [p.id for p in apply_context_budget(passages, 800)]
        theirs = [
            p.id
            for p in bench.apply_context_budget(
                [bench.Passage(id=p.id, text=p.text, score=p.score or 0.0) for p in passages], 800
            )
        ]
        assert theirs == ours
        assert 0 < len(ours) < len(passages)  # the budget actually bit


class FakeTokenizer:
    """Encodes each character to its code point; a leading space is its own token."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]


class TestScoring:
    def test_option_token_surfaces_match(self, bench: ModuleType) -> None:
        tok = FakeTokenizer()
        assert bench.option_token_ids(tok) == mcq.option_token_ids(tok)  # type: ignore[arg-type]

    def test_option_logprobs_match(self, bench: ModuleType) -> None:
        torch = pytest.importorskip("torch")
        logits = torch.randn(200)
        ids = [[65, 97], [66, 98], [67, 99], [68, 100]]
        ours = mcq.score_from_logits(logits, ids).logprobs
        theirs = bench.option_logprobs(logits, ids)
        assert theirs == pytest.approx(ours)


class TestSpaceFiles:
    def test_readme_declares_zero_gpu_and_the_disclaimer(self) -> None:
        readme = (PROJECT_ROOT / "space_live" / "README.md").read_text(encoding="utf-8")
        assert "sdk: gradio" in readme
        assert "suggested_hardware: zero-a10g" in readme
        assert "Not for clinical use" in readme

    def test_requirements_pin_the_benchmark_versions(self) -> None:
        import peft
        import sentence_transformers
        import transformers

        pins = (PROJECT_ROOT / "space_live" / "requirements.txt").read_text(encoding="utf-8")
        assert f"transformers=={transformers.__version__}" in pins
        assert f"peft=={peft.__version__}" in pins
        assert f"sentence-transformers=={sentence_transformers.__version__}" in pins
