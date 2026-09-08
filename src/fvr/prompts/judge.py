"""The judge rubric — frozen text, committed before any answer was scored.

An LLM judge is measurement apparatus. Apparatus that can be adjusted after
seeing the results is not measurement, so the rubric lives here in git rather
than in a notebook, and REPORT.md quotes it verbatim. If it is ever changed,
the change is a commit with a date and every affected number is re-run.

Two scoring modes, for two different questions:

**Pointwise** grades one answer against the reference. It is what produces a
per-arm score, and it is graded against a *reference answer*, never against the
judge's own medical opinion — a judge asked "is this correct?" is being asked to
be a doctor, while a judge asked "does this agree with the reference?" is being
asked to read, which is a far more reliable thing to ask of a model.

**Pairwise** compares two arms' answers to the same question. It is more
sensitive than pointwise for close calls, and it has a known failure mode:
models prefer whichever answer came first. So every pair is scored twice, in
both orders, and disagreement between the two orderings is reported rather than
averaged away.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Bumped whenever the rubric text changes. Recorded in every judged result, so
#: scores produced under different rubrics can never be silently pooled.
RUBRIC_VERSION = "1.0.0"

JUDGE_SYSTEM_PROMPT = (
    "You are a careful grader of medical question answering. You compare a "
    "candidate answer against a reference answer and report whether they agree. "
    "You are not being asked for your own medical opinion, and you must not "
    "reward an answer for being longer, better written, or more confident. "
    "Respond in the exact format requested and nothing else."
)

POINTWISE_RUBRIC = """\
Grade the CANDIDATE answer against the REFERENCE answer for the QUESTION below.

Score on this scale:

2 — AGREES. The candidate states the same clinical conclusion as the reference.
    Different wording, extra correct detail, or a missing non-essential detail
    are all still a 2.
1 — PARTIAL. The candidate contains the reference's conclusion but also states
    something that contradicts it, or hedges between the reference answer and a
    conflicting one, or answers only part of a multi-part question.
0 — DISAGREES. The candidate states a different clinical conclusion, refuses,
    is empty, or does not address the question.

Judge only agreement with the reference. Do not reward length, fluency, or
confidence. Do not penalise a terse answer that is correct. If the candidate is
correct but the reference is arguably wrong, still score agreement with the
reference — disagreements with the reference are handled elsewhere.

QUESTION:
{question}

REFERENCE:
{reference}

CANDIDATE:
{candidate}

Reply with exactly one line:
SCORE: <0, 1, or 2>"""

PAIRWISE_RUBRIC = """\
Two candidate answers, A and B, were given to the QUESTION below. Decide which
agrees better with the REFERENCE answer.

Judge only agreement with the reference. Ignore length, fluency, formatting and
confidence. If both agree equally well, or both fail equally, answer TIE — do
not break a genuine tie arbitrarily.

QUESTION:
{question}

REFERENCE:
{reference}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Reply with exactly one line:
VERDICT: <A, B, or TIE>"""


@dataclass(frozen=True)
class JudgePrompt:
    """A rendered judge request, ready for the chat template."""

    system: str
    user: str

    def as_messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]


def build_pointwise_prompt(question: str, reference: str, candidate: str) -> JudgePrompt:
    return JudgePrompt(
        system=JUDGE_SYSTEM_PROMPT,
        user=POINTWISE_RUBRIC.format(
            question=question.strip(),
            reference=reference.strip(),
            candidate=candidate.strip() or "(no answer given)",
        ),
    )


def build_pairwise_prompt(
    question: str, reference: str, answer_a: str, answer_b: str
) -> JudgePrompt:
    return JudgePrompt(
        system=JUDGE_SYSTEM_PROMPT,
        user=PAIRWISE_RUBRIC.format(
            question=question.strip(),
            reference=reference.strip(),
            answer_a=answer_a.strip() or "(no answer given)",
            answer_b=answer_b.strip() or "(no answer given)",
        ),
    )
