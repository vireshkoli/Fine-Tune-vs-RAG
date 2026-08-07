"""Dataset loading, normalised into :mod:`fvr.data.schema` types.

Parquet is fetched directly rather than through ``datasets.load_dataset``:
the loader path adds a script-resolution round trip that fails on a transient
connection reset, and going straight to the parquet is both faster and one
fewer moving part.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import pandas as pd
from huggingface_hub import hf_hub_download

from fvr.data.clean import CleaningReport, repair_text
from fvr.data.schema import Question

MEDMCQA_REPO = "openlifescienceai/medmcqa"
MEDMCQA_FILES = {
    "train": "data/train-00000-of-00001.parquet",
    "validation": "data/validation-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}
#: MedMCQA marks withheld answers with this sentinel rather than a null.
WITHHELD_ANSWER = -1


def resolve_dataset_revision(repo: str = MEDMCQA_REPO) -> str:
    """The dataset's current commit SHA.

    Recorded in the split manifest so the benchmark still reproduces after the
    lab machine is wiped — "latest" is not a reproducible pin once the local
    copy is gone.
    """
    from huggingface_hub import HfApi

    return str(HfApi().dataset_info(repo).sha)


def fetch_medmcqa_frame(split: str, *, revision: str | None = None) -> pd.DataFrame:
    """Download one MedMCQA split as a DataFrame, into the project artifact tree."""
    if split not in MEDMCQA_FILES:
        raise KeyError(f"unknown split {split!r}; expected one of {sorted(MEDMCQA_FILES)}")
    path = hf_hub_download(
        MEDMCQA_REPO,
        MEDMCQA_FILES[split],
        repo_type="dataset",
        revision=revision,
    )
    return pd.read_parquet(path)


def _clean(value: object, report: CleaningReport) -> str:
    # Missing cells arrive as None or as a float NaN depending on the column
    # dtype; `pd.isna` is not typed for a bare object, so check both directly.
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    if not text or text.lower() == "none":
        return ""
    repaired, fired = repair_text(text)
    report.tokens_repaired += sum(fired.values())
    report.repairs_by_token.update(fired)
    return repaired


def frame_to_questions(
    frame: pd.DataFrame,
    *,
    report: CleaningReport | None = None,
    drop_unlabelled: bool = True,
    drop_duplicates: bool = True,
) -> tuple[list[Question], CleaningReport]:
    """Normalise a MedMCQA frame into ``Question`` objects, repairing text.

    Duplicate detection uses :meth:`Question.content_hash`, which ignores ids —
    MedMCQA contains the same item under different ids, and leaving those in
    would let a test question also appear in training.
    """
    rep = report or CleaningReport()
    questions: list[Question] = []
    seen: set[str] = set()

    for row in frame.itertuples(index=False):
        rep.rows_seen += 1
        before = rep.tokens_repaired

        options = [_clean(getattr(row, f), rep) for f in ("opa", "opb", "opc", "opd")]
        question_text = _clean(row.question, rep)
        explanation = _clean(getattr(row, "exp", None), rep) or None

        if not question_text or not all(options):
            rep.malformed_dropped += 1
            continue

        cop = int(getattr(row, "cop", WITHHELD_ANSWER))
        answer_idx = None if cop == WITHHELD_ANSWER else cop
        if answer_idx is None and drop_unlabelled:
            rep.unlabelled_dropped += 1
            continue

        # `choice_type` is unreliable: ~37% of rows say "multi" while carrying a
        # single `cop`. The field is dropped rather than trusted.
        if str(getattr(row, "choice_type", "")) == "multi":
            rep.choice_type_normalised += 1

        item = Question(
            id=str(row.id),
            question=question_text,
            options=options,
            answer_idx=answer_idx,
            subject=str(getattr(row, "subject_name", "") or "Unknown"),
            explanation=explanation,
            source="medmcqa",
        )

        if drop_duplicates:
            digest = item.content_hash()
            if digest in seen:
                rep.duplicates_dropped += 1
                continue
            seen.add(digest)

        if rep.tokens_repaired > before:
            rep.rows_modified += 1
        questions.append(item)

    return questions, rep


def load_medmcqa(
    split: str,
    *,
    revision: str | None = None,
    drop_unlabelled: bool = True,
) -> tuple[list[Question], CleaningReport]:
    """Fetch, clean and normalise one MedMCQA split."""
    frame = fetch_medmcqa_frame(split, revision=revision)
    return frame_to_questions(frame, drop_unlabelled=drop_unlabelled)


def iter_corpus_text(questions: Iterator[Question] | list[Question]) -> Iterator[str]:
    """All human-readable text, for frequency counting when rebuilding the lexicon."""
    for q in questions:
        yield q.question
        yield from q.options
        if q.explanation:
            yield q.explanation
