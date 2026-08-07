"""Text repair for MedMCQA's systematic extractor corruption.

MedMCQA's text was extracted from PDFs with a bug that silently deletes certain
letter bigrams. It is not a rare blemish: measured over the 182,822-row train
split, the corrupted spelling is *more common than the correct one* for many
everyday clinical words.

===============  =======  ==============  =======
corrupted          count  correct           count
===============  =======  ==============  =======
``aery``          11,203  ``artery``        7,009
``hea``            7,626  ``heart``         4,957
``pa``             7,379  ``part``          4,947
``impoant``        4,506  ``important``     3,733
``hypeension``     4,060  ``hypertension``  2,519
===============  =======  ==============  =======

The repair is a **generated, human-reviewed lexicon** committed to git
(``resources/ocr_repairs.json``) rather than a heuristic applied at runtime.
That choice matters: the mapping is auditable, diffable, and identical on every
machine, so cleaning cannot silently change between runs. The generator lives
here so it can be re-run and re-reviewed if the source dataset ever changes.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

#: Bigrams the extractor is known to drop, in descending order of impact.
#: Only ``rt`` is used to build the shipped lexicon — see ``ACCEPTED_BIGRAMS``.
KNOWN_DROPPED_BIGRAMS: tuple[str, ...] = ("rt", "ti", "ol", "id", "iv", "it", "at")

#: The bigram actually repaired. ``rt`` accounts for 216 of the 255 candidate
#: entries and the overwhelming majority of affected tokens.
#:
#: The other bigrams are deliberately excluded. Their candidates are dominated
#: by real medical vocabulary that simply is not in a general English wordlist,
#: so rule 2 of :func:`propose_repairs` misfires: ``cor`` (cor pulmonale) would
#: become "color", ``conization`` (cervical conization) "colonization", ``anomic``
#: (anomic aphasia) "anatomic", ``acs`` (acute coronary syndrome) "acids", and the
#: ``adeno-``/``fibro-`` prefixes would be mangled. Under-repairing is much safer
#: than silently corrupting clinical terms, so the long tail is left alone and
#: reported as a known limitation.
ACCEPTED_BIGRAMS: tuple[str, ...] = ("rt",)

#: Entries the generator proposed that manual context review rejected.
#: Every length<=4 candidate was inspected against real occurrences; these two
#: failed. Kept as data rather than deleted so the review is auditable and a
#: regenerated lexicon cannot silently reintroduce them.
MANUALLY_REJECTED: Mapping[str, str] = {
    # Contexts read "a pos terior dislocation", "pos tmenopausal" — this is
    # `posterior`/`postmenopausal` split by a stray space, not `ports`.
    "pos": "ports",
    # BAER is Brainstem Auditory Evoked Response, a genuine acronym.
    "baer": "barter",
    # `pas` really is `parts` in context ("other pas of the body", 1,706 hits),
    # but PAS is the Periodic Acid-Schiff stain and matching is case-insensitive,
    # so accepting this would rewrite "PAS stain" to "PARTS stain". Not worth it.
    "pas": "parts",
}

#: Verified corruptions that the dictionary guard wrongly rejects.
#:
#: Rule 1 of :func:`propose_repairs` skips any token that is itself a dictionary
#: word, which is normally what stops ``pot`` becoming ``port``. But the single
#: largest corruption in the corpus defeats it: the extractor turns ``arteri-``
#: into ``aeri-``, and ``aery`` (a variant of *eyrie*) and ``aerial`` are both
#: real English words. Together these are 15,495 tokens — more than the entire
#: automatically generated lexicon covers for any other stem.
#:
#: Each was confirmed by reading real contexts ("lodged in the aery supplying
#: the optic nerve", "mean aerial pressure", "systemic aerial blood"). None is a
#: medical acronym, so case-insensitive matching is safe for them.
DICTIONARY_OVERRIDES: Mapping[str, str] = {
    "aery": "artery",
    "aeries": "arteries",
    "aerial": "arterial",
}

_WORD_RE = re.compile(r"[A-Za-z]+")
_SYSTEM_DICTIONARIES = (
    Path("/usr/share/dict/american-english"),
    Path("/usr/share/dict/words"),
)


@dataclass
class CleaningReport:
    """What the cleaning pass actually did, so it can be reported not assumed."""

    rows_seen: int = 0
    rows_modified: int = 0
    tokens_repaired: int = 0
    repairs_by_token: Counter[str] = field(default_factory=Counter)
    duplicates_dropped: int = 0
    unlabelled_dropped: int = 0
    #: Rows discarded for an empty question stem or a blank option.
    malformed_dropped: int = 0
    choice_type_normalised: int = 0

    @property
    def rows_modified_pct(self) -> float:
        return 100.0 * self.rows_modified / self.rows_seen if self.rows_seen else 0.0

    @property
    def rows_kept(self) -> int:
        return (
            self.rows_seen
            - self.duplicates_dropped
            - self.unlabelled_dropped
            - self.malformed_dropped
        )

    def summary(self) -> str:
        top = ", ".join(f"{w}->{n}" for w, n in self.repairs_by_token.most_common(5)) or "none"
        return (
            f"{self.rows_seen:,} seen -> {self.rows_kept:,} kept | "
            f"{self.rows_modified:,} rows repaired ({self.rows_modified_pct:.1f}%), "
            f"{self.tokens_repaired:,} tokens | dropped: {self.duplicates_dropped:,} dup, "
            f"{self.unlabelled_dropped:,} unlabelled, {self.malformed_dropped:,} malformed | "
            f"top repairs: {top}"
        )


@lru_cache(maxsize=1)
def load_repair_lexicon() -> Mapping[str, str]:
    """The committed corrupted->correct mapping, lowercase keys."""
    resource = files("fvr.data.resources").joinpath("ocr_repairs.json")
    data: dict[str, str] = json.loads(resource.read_text(encoding="utf-8"))
    return data


@lru_cache(maxsize=1)
def _compiled_lexicon() -> tuple[re.Pattern[str], Mapping[str, str]]:
    """One alternation over all keys — far faster than a pass per entry."""
    lexicon = load_repair_lexicon()
    if not lexicon:
        return re.compile(r"(?!)"), lexicon
    # Longest first so `aeries` wins over `aery` when both could match a prefix.
    alternation = "|".join(sorted(map(re.escape, lexicon), key=len, reverse=True))
    return re.compile(rf"\b({alternation})\b", re.IGNORECASE), lexicon


def _match_case(source: str, replacement: str) -> str:
    """Carry the original capitalisation onto the repaired token."""
    if source.isupper() and len(source) > 1:
        return replacement.upper()
    if source[:1].isupper():
        return replacement.capitalize()
    return replacement


def repair_text(text: str) -> tuple[str, Counter[str]]:
    """Apply the committed lexicon.

    Returns the repaired text and a per-token tally, so the cleaning report can
    show *which* repairs fired rather than only how many.
    """
    pattern, lexicon = _compiled_lexicon()
    fired: Counter[str] = Counter()

    def _sub(match: re.Match[str]) -> str:
        original = match.group(0)
        replacement = lexicon.get(original.lower())
        if replacement is None:  # pragma: no cover - pattern is built from the keys
            return original
        fired[original.lower()] += 1
        return _match_case(original, replacement)

    return pattern.sub(_sub, text), fired


# --------------------------------------------------------------------------- generator


def load_system_dictionary() -> frozenset[str]:
    """Lowercase English wordlist, used as the external check on a proposed repair.

    External rather than corpus-derived on purpose: corpus frequency is polluted
    by the very corruption being detected, so ``aery`` looks like a real word if
    you only count occurrences.
    """
    for path in _SYSTEM_DICTIONARIES:
        if path.is_file():
            words = path.read_text(encoding="utf-8", errors="ignore").split()
            return frozenset(w.lower() for w in words if w.isalpha())
    return frozenset()


def token_frequencies(texts: Iterable[str]) -> Counter[str]:
    freq: Counter[str] = Counter()
    for text in texts:
        freq.update(w.lower() for w in _WORD_RE.findall(text))
    return freq


def propose_repairs(
    freq: Counter[str],
    dictionary: frozenset[str],
    *,
    bigrams: Iterable[str] = KNOWN_DROPPED_BIGRAMS,
    min_count: int = 20,
    min_length: int = 2,
    min_correct_ratio: float = 0.30,
) -> dict[str, dict[str, object]]:
    """Propose corrupted->correct repairs for human review.

    A candidate is proposed only when every one of these holds, which is what
    keeps the false-positive rate low enough to review by hand:

    1. the observed token is **not** a dictionary word (so ``pot`` is never
       "repaired" into ``port``);
    2. inserting the bigram yields a token that **is** a dictionary word;
    3. exactly one insertion position does so, so ambiguous cases are dropped
       rather than guessed;
    4. the observed token occurs at least ``min_count`` times, because a
       systematic extractor bug is frequent by nature and one-offs are typos;
    5. the *correct* form also occurs, at ``min_correct_ratio`` of the corrupted
       form's frequency.

    Rule 5 is what separates real corruption from medical abbreviations, and it
    was added after the first generated table proposed ``ans -> antis`` (47,196
    hits — it is short for "answer"), ``ml -> moll`` and ``lh -> lath``. The
    extractor corrupts only *some* occurrences of a word, so a genuine victim
    like ``hea``/``heart`` shows both spellings at comparable frequency, while
    an abbreviation's "correct" form is essentially absent from the corpus.

    Output is a review table, not a lexicon — a human promotes entries.
    """
    proposals: dict[str, dict[str, object]] = {}
    for token, count in freq.items():
        if count < min_count or len(token) < min_length or token in dictionary:
            continue
        hits: list[tuple[str, str]] = []
        for bigram in bigrams:
            for i in range(1, len(token) + 1):
                candidate = token[:i] + bigram + token[i:]
                if candidate in dictionary:
                    hits.append((bigram, candidate))
        unique = {candidate for _, candidate in hits}
        if len(unique) != 1:
            continue  # zero matches, or ambiguous — leave the text alone
        bigram, candidate = hits[0]
        correct_count = freq.get(candidate, 0)
        if correct_count < min_correct_ratio * count:
            continue  # an abbreviation, not a corruption
        if MANUALLY_REJECTED.get(token) == candidate:
            continue  # failed human context review; see MANUALLY_REJECTED
        proposals[token] = {
            "repair": candidate,
            "bigram": bigram,
            "count": count,
            "correct_form_count": correct_count,
        }
    return proposals


def build_lexicon(
    texts: Iterable[str],
    *,
    bigrams: Iterable[str] = ACCEPTED_BIGRAMS,
    min_count: int = 20,
    min_length: int = 3,
) -> tuple[dict[str, str], dict[str, dict[str, object]]]:
    """Regenerate the shipped lexicon. Returns ``(lexicon, provenance)``.

    ``min_length=3`` because two-letter candidates are almost all abbreviations
    (``hu`` would otherwise become "hurt", destroying Hounsfield Units).
    """
    freq = token_frequencies(texts)
    proposals = propose_repairs(
        freq,
        load_system_dictionary(),
        bigrams=bigrams,
        min_count=min_count,
        min_length=min_length,
    )
    lexicon = {token: str(info["repair"]) for token, info in proposals.items()}
    for token, repair in DICTIONARY_OVERRIDES.items():
        lexicon[token] = repair
        proposals.setdefault(
            token,
            {
                "repair": repair,
                "bigram": "rt",
                "count": freq.get(token, 0),
                "correct_form_count": freq.get(repair, 0),
                "source": "DICTIONARY_OVERRIDES (manually verified)",
            },
        )
    return dict(sorted(lexicon.items())), proposals
