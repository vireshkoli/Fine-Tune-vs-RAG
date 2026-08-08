"""The experimental arms.

Six configurations, composed from two independent switches: whether an adapter
is attached, and which retrieval index (if any) is used. Defining them as data
rather than as six code paths is deliberate — it makes it impossible for one
arm to quietly acquire a different prompt or decoding setting.

Four arms are the headline. ``rag-parity`` and ``qlora-rag-parity`` are
diagnostics: they retrieve from the *training explanations*, i.e. exactly the
text the fine-tune learned from. Comparing them against the external-corpus
arms separates "fine-tuning absorbed knowledge" from "fine-tuning learned the
answer format", which a four-arm design cannot distinguish.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

CorpusName = Literal["none", "external", "parity"]


@dataclass(frozen=True)
class Arm:
    """One experimental configuration."""

    name: str
    uses_adapter: bool
    corpus: CorpusName
    headline: bool
    description: str

    @property
    def uses_retrieval(self) -> bool:
        return self.corpus != "none"


ARMS: tuple[Arm, ...] = (
    Arm(
        name="base",
        uses_adapter=False,
        corpus="none",
        headline=True,
        description="Base model, zero-shot. The floor.",
    ),
    Arm(
        name="rag-external",
        uses_adapter=False,
        corpus="external",
        headline=True,
        description="Retrieval over the external corpus, base weights unchanged.",
    ),
    Arm(
        name="qlora",
        uses_adapter=True,
        corpus="none",
        headline=True,
        description="QLoRA fine-tune, no retrieval.",
    ),
    Arm(
        name="qlora-rag",
        uses_adapter=True,
        corpus="external",
        headline=True,
        description="Fine-tune plus retrieval — the combination most write-ups skip.",
    ),
    Arm(
        name="rag-parity",
        uses_adapter=False,
        corpus="parity",
        headline=False,
        description="Diagnostic: retrieves the same text the fine-tune trained on.",
    ),
    Arm(
        name="qlora-rag-parity",
        uses_adapter=True,
        corpus="parity",
        headline=False,
        description="Diagnostic: fine-tune plus the parity index.",
    ),
)

ARMS_BY_NAME: dict[str, Arm] = {arm.name: arm for arm in ARMS}


def get_arm(name: str) -> Arm:
    try:
        return ARMS_BY_NAME[name]
    except KeyError:
        raise KeyError(f"unknown arm {name!r}; expected one of {sorted(ARMS_BY_NAME)}") from None


def headline_arms() -> Sequence[Arm]:
    """The four arms that appear in the README table."""
    return tuple(arm for arm in ARMS if arm.headline)
