"""Tests for the OCR repair lexicon and its generator.

These run in CI with no network: the lexicon is a committed resource, so the
applier is fully testable offline. The generator is tested against a synthetic
corpus rather than the real 182k-row dataset.
"""

from __future__ import annotations

import json
from collections import Counter
from importlib.resources import files

import pytest

from fvr.data.clean import (
    ACCEPTED_BIGRAMS,
    MANUALLY_REJECTED,
    CleaningReport,
    build_lexicon,
    load_repair_lexicon,
    load_system_dictionary,
    propose_repairs,
    repair_text,
)


class TestLexiconResource:
    def test_is_committed_and_populated(self) -> None:
        lexicon = load_repair_lexicon()
        assert len(lexicon) > 200, "the shipped lexicon looks truncated"

    def test_known_corruptions_are_present(self) -> None:
        # These are the highest-frequency victims measured on the train split.
        lexicon = load_repair_lexicon()
        for corrupt, correct in [
            ("hea", "heart"),
            ("impoant", "important"),
            ("hypeension", "hypertension"),
            ("bih", "birth"),
            ("coex", "cortex"),
            ("ahritis", "arthritis"),
        ]:
            assert lexicon.get(corrupt) == correct

    @pytest.mark.parametrize("rejected", sorted(MANUALLY_REJECTED))
    def test_manually_rejected_entries_never_ship(self, rejected: str) -> None:
        # `pos` is `posterior` split by a space; `baer` is Brainstem Auditory
        # Evoked Response. Both would corrupt clinical text.
        assert rejected not in load_repair_lexicon()

    def test_no_entry_maps_to_itself(self) -> None:
        assert not [k for k, v in load_repair_lexicon().items() if k == v]

    def test_every_repair_inserts_an_accepted_bigram(self) -> None:
        """Each entry must be explainable as one bigram insertion, not a free edit."""
        for corrupt, correct in load_repair_lexicon().items():
            assert any(
                any(corrupt[:i] + bg + corrupt[i:] == correct for i in range(len(corrupt) + 1))
                for bg in ACCEPTED_BIGRAMS
            ), f"{corrupt!r} -> {correct!r} is not a single accepted-bigram insertion"

    def test_no_two_letter_keys(self) -> None:
        # Two-letter candidates are almost all abbreviations (`hu` -> Hounsfield).
        assert not [k for k in load_repair_lexicon() if len(k) < 3]

    def test_resource_is_valid_sorted_json(self) -> None:
        raw = files("fvr.data.resources").joinpath("ocr_repairs.json").read_text(encoding="utf-8")
        data = json.loads(raw)
        assert list(data) == sorted(data), "keep the lexicon sorted so diffs stay readable"


class TestRepairText:
    def test_repairs_a_known_token(self) -> None:
        out, fired = repair_text("the left coronary aery supplies the hea")
        assert out == "the left coronary artery supplies the heart"
        assert fired == Counter({"aery": 1, "hea": 1})

    def test_leaves_clean_text_untouched(self) -> None:
        clean = "the left coronary artery supplies the heart"
        out, fired = repair_text(clean)
        assert out == clean
        assert not fired

    def test_is_idempotent(self) -> None:
        once, _ = repair_text("hypeension of the coex")
        twice, fired = repair_text(once)
        assert once == twice
        assert not fired

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("hea", "heart"),
            ("Hea", "Heart"),
            ("HEA", "HEART"),
        ],
    )
    def test_preserves_capitalisation(self, text: str, expected: str) -> None:
        assert repair_text(text)[0] == expected

    def test_respects_word_boundaries(self) -> None:
        # `hea` must not fire inside `header` or `heals`.
        for word in ("header", "heals", "heading"):
            assert repair_text(word)[0] == word

    def test_handles_punctuation_adjacency(self) -> None:
        out, _ = repair_text("(hea), aery; coex.")
        assert out == "(heart), artery; cortex."


class TestProposeRepairs:
    DICT = frozenset({"heart", "artery", "important", "port", "pot", "colour"})

    def test_proposes_a_clear_corruption(self) -> None:
        freq = Counter({"hea": 100, "heart": 80})
        out = propose_repairs(freq, self.DICT, bigrams=("rt",), min_count=10)
        assert out["hea"]["repair"] == "heart"

    def test_skips_words_already_in_the_dictionary(self) -> None:
        # `pot` is a real word and must never become `port`.
        freq = Counter({"pot": 500, "port": 400})
        assert propose_repairs(freq, self.DICT, bigrams=("rt",), min_count=10) == {}

    def test_skips_when_the_correct_form_is_absent(self) -> None:
        """The rule that rejects abbreviations like `ans` -> `antis`."""
        freq = Counter({"hea": 5000, "heart": 3})
        assert propose_repairs(freq, self.DICT, bigrams=("rt",), min_count=10) == {}

    def test_skips_rare_tokens(self) -> None:
        # A systematic extractor bug is frequent; one-offs are ordinary typos.
        freq = Counter({"hea": 2, "heart": 80})
        assert propose_repairs(freq, self.DICT, bigrams=("rt",), min_count=20) == {}

    def test_skips_ambiguous_insertions(self) -> None:
        # "abc" admits `rt` at index 1 ("artbc") and index 2 ("abrtc"); with both
        # in the dictionary there is no single safe repair, so propose nothing.
        dictionary = frozenset({"artbc", "abrtc"})
        freq = Counter({"abc": 100, "artbc": 90, "abrtc": 90})
        assert propose_repairs(freq, dictionary, bigrams=("rt",), min_count=10) == {}

    def test_accepts_when_only_one_insertion_point_is_valid(self) -> None:
        """Companion to the above: exactly one dictionary hit is unambiguous."""
        dictionary = frozenset({"artbc"})
        freq = Counter({"abc": 100, "artbc": 90})
        assert (
            propose_repairs(freq, dictionary, bigrams=("rt",), min_count=10)["abc"]["repair"]
            == "artbc"
        )

    def test_manual_rejections_are_filtered(self) -> None:
        dictionary = frozenset({"ports"})
        freq = Counter({"pos": 100, "ports": 90})
        assert propose_repairs(freq, dictionary, bigrams=("rt",), min_count=10) == {}

    @pytest.mark.skipif(
        not load_system_dictionary(),
        reason="needs /usr/share/dict/american-english, absent on CI runners",
    )
    def test_build_lexicon_round_trips(self) -> None:
        texts = ["the hea pumps blood"] * 30 + ["the heart pumps blood"] * 25
        lexicon, provenance = build_lexicon(texts, min_count=10)
        assert lexicon["hea"] == "heart"
        assert provenance["hea"]["count"] == 30

    def test_build_lexicon_always_includes_overrides(self) -> None:
        """Overrides are hand-verified, so they must survive an absent wordlist."""
        lexicon, _ = build_lexicon(["irrelevant text"], min_count=10)
        assert lexicon["aery"] == "artery"


class TestCleaningReport:
    def test_summary_is_populated(self) -> None:
        report = CleaningReport(rows_seen=10, rows_modified=4, tokens_repaired=7)
        report.repairs_by_token.update({"hea": 5, "aery": 2})
        text = report.summary()
        assert "40.0%" in text
        assert "hea->5" in text

    def test_rows_kept_accounts_for_every_drop(self) -> None:
        report = CleaningReport(
            rows_seen=100, duplicates_dropped=5, unlabelled_dropped=3, malformed_dropped=2
        )
        assert report.rows_kept == 90

    def test_summary_on_empty_report_does_not_divide_by_zero(self) -> None:
        assert "0.0%" in CleaningReport().summary()
