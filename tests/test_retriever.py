"""Retrieval tests.

The leakage cases are the ones that matter. If held-out text can reach an
index, the RAG arms are doing lookup rather than reasoning and every retrieval
number in the report is meaningless — so that property is asserted here rather
than left to the build script's runtime check.

Everything runs on CPU with a tiny synthetic index; no model or network needed.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from fvr.config import PROJECT_ROOT
from fvr.data.schema import Passage, Question
from fvr.eval.groundedness import GroundednessSummary, score_groundedness
from fvr.retrieval.corpus import (
    DEFAULT_CHUNK_CHARS,
    MIN_CHUNK_CHARS,
    build_external_corpus,
    build_parity_corpus,
    chunk_text,
    strip_answer_key,
)
from fvr.retrieval.embed import load_embedder_config
from fvr.retrieval.retriever import RetrievalConfig, apply_context_budget, load_retrieval_config

CONFIG_PATH = PROJECT_ROOT / "configs" / "retrieval" / "bge_large.yaml"


def a_question(qid: str = "q1", *, explanation: str | None = None, answer: int = 0) -> Question:
    return Question(
        id=qid,
        question="Which vessel supplies the myocardium?",
        options=["Coronary artery", "Portal vein", "Aorta", "Vena cava"],
        answer_idx=answer,
        explanation=explanation,
        subject="Anatomy",
    )


def passages(n: int, *, chars: int = 100) -> list[Passage]:
    """``n`` passages whose text is exactly ``chars`` characters long."""
    out = []
    for i in range(n):
        head = f"passage {i} "
        out.append(
            Passage(id=f"p{i}", text=(head + "x" * (chars - len(head)))[:chars], corpus="external")
        )
    return out


class TestChunking:
    def test_short_text_is_one_chunk(self) -> None:
        text = (
            "The coronary arteries supply the myocardium and arise from the "
            "aortic sinuses located just above the aortic valve cusps."
        )
        assert len(text) > MIN_CHUNK_CHARS
        assert chunk_text(text) == [text]

    def test_very_short_text_is_dropped(self) -> None:
        # Fragments below MIN_CHUNK_CHARS retrieve noisily, so they never enter
        # an index. Reported in the corpus stats rather than dropped silently.
        assert chunk_text("Too short.") == []
        assert len("Too short.") < MIN_CHUNK_CHARS

    def test_long_text_is_split(self) -> None:
        text = " ".join(f"Sentence number {i} about the cardiovascular system." for i in range(80))
        chunks = chunk_text(text)
        assert len(chunks) > 1

    def test_chunks_respect_the_size_target(self) -> None:
        text = " ".join(f"Sentence number {i} about the cardiovascular system." for i in range(200))
        # Allow slack: chunks end on sentence boundaries rather than mid-word.
        assert all(len(c) <= DEFAULT_CHUNK_CHARS * 2 for c in chunk_text(text))

    def test_no_content_is_lost(self) -> None:
        text = " ".join(f"Fact {i} matters." for i in range(60))
        joined = " ".join(chunk_text(text))
        assert "Fact 0 matters." in joined
        assert "Fact 59 matters." in joined

    def test_a_single_enormous_sentence_still_terminates(self) -> None:
        # No sentence boundaries at all — the fallback split must not loop forever.
        chunks = chunk_text("word " * 2000)
        assert len(chunks) > 1

    def test_whitespace_is_normalised(self) -> None:
        messy = (
            "The   heart\n\n  has\tfour chambers and   sits inside\nthe mediastinum, "
            "bounded  laterally   by the two pleural cavities."
        )
        assert chunk_text(messy) == [
            "The heart has four chambers and sits inside the mediastinum, "
            "bounded laterally by the two pleural cavities."
        ]


class TestAnswerKeyStripping:
    """MedMCQA explanations often open with the answer letter.

    Those fragments carry no clinical content, match option text lexically, and
    name a letter belonging to a different question's option order — so they are
    removed by content rather than by a blunt length cut.
    """

    @pytest.mark.parametrize(
        "junk",
        [
            "Ans. C i.e. Mite",
            "Ans. is 'd' i.e., Corneal ulceration",
            "A i.e. Na+",
            "D i.e. All",
            "Foregut",
            "VP",
        ],
    )
    def test_pure_answer_key_fragments_are_discarded(self, junk: str) -> None:
        assert chunk_text(strip_answer_key(junk)) == []

    @pytest.mark.parametrize(
        "citation",
        [
            "Ref - Krishan Vij textbook of forensic medicine and toxicology 5e pg - 202",
            "Reference: Harpers illustrated biochemistry 30th edition page 275",
        ],
    )
    def test_bibliography_only_explanations_are_discarded(self, citation: str) -> None:
        assert strip_answer_key(citation) == ""

    def test_strips_the_prefix_but_keeps_the_fact(self) -> None:
        assert strip_answer_key("Ans. B: True positives Sensitivity denotes true positives") == (
            "True positives Sensitivity denotes true positives"
        )

    def test_strips_a_parenthesised_lead_in(self) -> None:
        assert (
            strip_answer_key(
                "Answer is A (Quinine) Quinine is not associated with discoloured urine."
            )
            == "(Quinine) Quinine is not associated with discoloured urine."
        )

    def test_leaves_a_clean_explanation_untouched(self) -> None:
        fact = "Suprarenal glands drain into para-aortic nodes."
        assert strip_answer_key(fact) == fact

    def test_does_not_eat_a_sentence_starting_with_a_word(self) -> None:
        # "Dysplasia is reversible" must not lose its first token to the letter rule.
        fact = "Dysplasia is reversible whereas anaplasia is irreversible."
        assert strip_answer_key(fact) == fact


class TestParityCorpus:
    def test_indexes_train_explanations(self) -> None:
        train = [
            a_question(
                "t1",
                explanation=(
                    "The coronary arteries supply the myocardium and arise from "
                    "the aortic sinuses just above the aortic valve."
                ),
            )
        ]
        built, stats = build_parity_corpus(train, forbidden_content_hashes=frozenset())
        assert len(built) == 1
        assert built[0].corpus == "parity"
        assert built[0].source_question_id == "t1"
        assert stats.chunks == 1

    def test_skips_rows_without_an_explanation(self) -> None:
        built, _ = build_parity_corpus([a_question("t1")], forbidden_content_hashes=frozenset())
        assert built == []

    def test_excludes_held_out_content_even_under_a_different_id(self) -> None:
        """The leakage case that an id-based filter would miss.

        MedMCQA repeats items across splits with fresh ids, so a test question's
        own explanation would otherwise land in the index.
        """
        held_out = a_question("test-1")
        # Same content, different id, and it carries an explanation.
        duplicate = a_question(
            "train-9",
            explanation=(
                "The coronary arteries supply the myocardium and arise from "
                "the aortic sinuses just above the aortic valve."
            ),
        )
        built, stats = build_parity_corpus(
            [duplicate], forbidden_content_hashes=frozenset({held_out.content_hash()})
        )
        assert built == []
        assert stats.excluded_documents == 1

    def test_keeps_unrelated_train_rows(self) -> None:
        held_out = a_question("test-1")
        other = Question(
            id="train-2",
            question="Which nerve innervates the diaphragm?",
            options=["Phrenic", "Vagus", "Ulnar", "Radial"],
            answer_idx=0,
            explanation=(
                "The phrenic nerve arises from cervical roots three to five and "
                "provides the sole motor supply to the diaphragm."
            ),
        )
        built, _ = build_parity_corpus(
            [other], forbidden_content_hashes=frozenset({held_out.content_hash()})
        )
        assert len(built) == 1

    def test_no_parity_passage_references_a_held_out_id(self) -> None:
        train = [
            a_question(
                f"t{i}",
                explanation=(
                    f"Explanation number {i} concerning anatomy, physiology and the "
                    "clinical relevance of the vessel in question."
                ),
            )
            for i in range(20)
        ]
        built, _ = build_parity_corpus(train, forbidden_content_hashes=frozenset())
        held_out_ids = {"test-1", "test-2"}
        assert not [p for p in built if p.source_question_id in held_out_ids]


class TestExternalCorpus:
    def test_chunks_documents(self) -> None:
        long_text = " ".join(f"Finding {i} was reported in the cohort." for i in range(120))
        built, stats = build_external_corpus([("A paper", long_text)])
        assert len(built) > 1
        assert stats.source_documents == 1
        assert all(p.corpus == "external" for p in built)

    def test_title_is_retained_for_citation(self) -> None:
        built, _ = build_external_corpus([("Relapsing Polychondritis", "A" * 300 + " end here.")])
        assert built[0].title == "Relapsing Polychondritis"

    def test_max_chunks_caps_the_corpus(self) -> None:
        long_text = " ".join(f"Finding {i} was reported in the cohort." for i in range(400))
        built, _ = build_external_corpus([("A", long_text), ("B", long_text)], max_chunks=5)
        assert len(built) == 5

    def test_external_passages_carry_no_question_provenance(self) -> None:
        built, _ = build_external_corpus(
            [
                (
                    "A paper",
                    "Some finding was reported in this cohort study of two hundred "
                    "adults recruited across four tertiary centres.",
                )
            ]
        )
        assert built[0].source_question_id is None

    def test_ids_are_unique(self) -> None:
        long_text = " ".join(f"Finding {i} was reported in the cohort." for i in range(200))
        built, _ = build_external_corpus([("A", long_text)])
        assert len({p.id for p in built}) == len(built)


class TestContextBudget:
    def test_keeps_passages_until_the_budget_is_spent(self) -> None:
        # Three fit in 350 chars; a fourth would overshoot, so it is left out.
        assert len(apply_context_budget(passages(10, chars=100), 350)) == 3

    def test_never_truncates_a_passage_mid_text(self) -> None:
        # A partial passage would embed a claim the model cannot verify.
        kept = apply_context_budget(passages(10, chars=100), 350)
        assert all(len(p.text) == 100 for p in kept)

    def test_always_keeps_at_least_one_passage(self) -> None:
        # Even an over-budget single passage beats sending no context at all.
        assert len(apply_context_budget(passages(3, chars=5000), 100)) == 1

    def test_equalises_context_across_differently_sized_corpora(self) -> None:
        """The fairness control: both RAG arms get the same character budget.

        Parity chunks are short and external chunks long, so a fixed top-k would
        hand one arm far more prefill than the other.
        """
        budget = 1200
        short = apply_context_budget(passages(20, chars=300), budget)
        long = apply_context_budget(passages(20, chars=600), budget)
        assert sum(len(p.text) for p in short) <= budget
        assert sum(len(p.text) for p in long) <= budget
        assert len(short) > len(long)

    def test_empty_input(self) -> None:
        assert apply_context_budget([], 1000) == []


class TestConfig:
    def test_committed_config_parses(self) -> None:
        embedder = load_embedder_config(CONFIG_PATH)
        assert embedder.dimensions == 1024
        assert embedder.repo_id == "BAAI/bge-large-en-v1.5"

    def test_chunks_fit_inside_the_embedder_window(self) -> None:
        """Chunks must not silently overflow the 512-token limit.

        Raw MIRIAD passages average ~4,500 chars (~1,100 tokens); indexing them
        unchunked would leave over half of each passage unsearchable.
        """
        embedder = load_embedder_config(CONFIG_PATH)
        approx_tokens = DEFAULT_CHUNK_CHARS / 4
        assert approx_tokens < embedder.max_seq_length

    def test_query_prefix_is_set_for_bge(self) -> None:
        # bge is asymmetric; dropping the query instruction measurably hurts recall.
        assert load_embedder_config(CONFIG_PATH).query_prefix.strip()

    def test_retrieval_section_parses(self) -> None:
        retrieval = load_retrieval_config(CONFIG_PATH)
        assert retrieval.top_k >= 1
        assert retrieval.max_context_chars >= 1000

    def test_both_rag_arms_share_one_budget(self) -> None:
        raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        assert "max_context_chars" in raw["retrieval"]

    def test_rejects_a_zero_top_k(self) -> None:
        with pytest.raises(ValueError):
            RetrievalConfig(top_k=0)


class TestGroundedness:
    def test_detects_a_supported_answer(self) -> None:
        question = a_question(answer=0)
        context = [
            Passage(id="p1", text="The coronary artery supplies the myocardium.", corpus="x")
        ]
        score = score_groundedness(question, context, predicted_idx=0)
        assert score.gold_coverage == pytest.approx(1.0)
        assert score.retrieval_hit
        assert score.is_grounded

    def test_detects_irrelevant_retrieval(self) -> None:
        """A right answer with unrelated context — retrieval contributed nothing."""
        question = a_question(answer=0)
        context = [Passage(id="p1", text="Bone remodelling involves osteoclasts.", corpus="x")]
        score = score_groundedness(question, context, predicted_idx=0)
        assert score.gold_coverage == 0.0
        assert not score.retrieval_hit

    def test_stopwords_do_not_manufacture_support(self) -> None:
        question = Question(
            id="q", question="Which?", options=["the and of it", "b", "c", "d"], answer_idx=0
        )
        context = [Passage(id="p1", text="the and of it was in on at", corpus="x")]
        assert score_groundedness(question, context, 0).gold_coverage == 0.0

    def test_handles_empty_context(self) -> None:
        score = score_groundedness(a_question(), [], predicted_idx=0)
        assert score.n_passages == 0
        assert score.top_score is None

    def test_summary_aggregates(self) -> None:
        question = a_question(answer=0)
        hit = Passage(id="p1", text="The coronary artery supplies the myocardium.", corpus="x")
        miss = Passage(id="p2", text="Unrelated text about bone.", corpus="x")
        scores = [
            score_groundedness(question, [hit], 0),
            score_groundedness(question, [miss], 0),
        ]
        summary = GroundednessSummary.from_scores(scores)
        assert summary.n == 2
        assert summary.retrieval_hit_rate == 0.5

    def test_summary_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="no groundedness"):
            GroundednessSummary.from_scores([])


class TestVectorIndex:
    def test_search_returns_nearest_first(self) -> None:
        faiss = pytest.importorskip("faiss")
        assert faiss is not None
        from fvr.retrieval.index import build_index

        vectors = np.eye(4, dtype=np.float32)
        items = [Passage(id=f"p{i}", text=f"passage {i} text", corpus="t") for i in range(4)]
        index = build_index(vectors, items, "t")

        hits = index.search(np.array([[0, 1, 0, 0]], dtype=np.float32), k=2)[0]
        assert hits[0].id == "p1"
        assert hits[0].score == pytest.approx(1.0)

    def test_k_larger_than_the_index_is_clamped(self) -> None:
        pytest.importorskip("faiss")
        from fvr.retrieval.index import build_index

        index = build_index(
            np.eye(3, dtype=np.float32),
            [Passage(id=f"p{i}", text=f"text {i} here", corpus="t") for i in range(3)],
            "t",
        )
        assert len(index.search(np.eye(1, 3, dtype=np.float32), k=99)[0]) == 3

    def test_mismatched_lengths_raise(self) -> None:
        pytest.importorskip("faiss")
        from fvr.retrieval.index import build_index

        with pytest.raises(ValueError, match="vectors but"):
            build_index(np.eye(3, dtype=np.float32), [Passage(id="p", text="t t", corpus="t")], "t")


class TestRetrieverReusesItsEncoder:
    """Regression guard for the OOM that killed the rag-external arm.

    ``embed_queries`` constructs an encoder when passed ``None``, so a Retriever
    that never caches one builds a fresh SentenceTransformer per batch. Over 125
    batches that exhausted 44GB of VRAM and the arm died outright.
    """

    def test_embedder_is_constructed_once_across_many_batches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("faiss")
        from fvr.retrieval.index import build_index
        from fvr.retrieval.retriever import Retriever

        calls = 0

        def fake_loader(_config: object) -> object:
            nonlocal calls
            calls += 1
            return object()

        monkeypatch.setattr("fvr.retrieval.embed.load_embedder", fake_loader)
        monkeypatch.setattr(
            "fvr.retrieval.retriever.embed_queries",
            lambda texts, cfg, *, model=None: np.zeros((len(texts), 4), dtype=np.float32),
        )

        index = build_index(
            np.eye(4, dtype=np.float32),
            [Passage(id=f"p{i}", text=f"passage {i} text here", corpus="t") for i in range(4)],
            "t",
        )
        retriever = Retriever(
            index=index,
            embedder_config=load_embedder_config(CONFIG_PATH),
            config=RetrievalConfig(),
        )
        for _ in range(10):
            retriever.retrieve_many([a_question()])
        assert calls == 1, f"encoder rebuilt {calls} times instead of once"

    def test_an_injected_encoder_is_never_replaced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("faiss")
        from fvr.retrieval.index import build_index
        from fvr.retrieval.retriever import Retriever

        def explode(_config: object) -> object:  # pragma: no cover - must not run
            raise AssertionError("load_embedder called despite an injected model")

        monkeypatch.setattr("fvr.retrieval.embed.load_embedder", explode)
        monkeypatch.setattr(
            "fvr.retrieval.retriever.embed_queries",
            lambda texts, cfg, *, model=None: np.zeros((len(texts), 4), dtype=np.float32),
        )
        sentinel = object()
        retriever = Retriever(
            index=build_index(
                np.eye(4, dtype=np.float32),
                [Passage(id=f"p{i}", text=f"passage {i} text here", corpus="t") for i in range(4)],
                "t",
            ),
            embedder_config=load_embedder_config(CONFIG_PATH),
            config=RetrievalConfig(),
            model=sentinel,
        )
        retriever.retrieve_many([a_question()])
        assert retriever.embedder() is sentinel
