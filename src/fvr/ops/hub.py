"""Publishing the adapter and its card to the Hugging Face Hub.

This is the step that makes teardown safe. The lab machine gets wiped on
completion, so an adapter that exists only in ``.artifacts/checkpoints`` is an
adapter that is about to stop existing — ``scripts/09_verify_recoverable.py``
gates teardown on the Hub copy being real.

Three things are enforced here rather than remembered:

1. **The card that ships is the reviewed one.** ``trl`` writes its own stub
   ``README.md`` into the adapter directory, full of "[More Information
   Needed]". Uploading the directory wholesale would publish that stub as the
   model card and silently discard ``docs/model_card.md``.
2. **The not-for-clinical-use statement is present.** A medical adapter without
   it does not get pushed — :func:`assert_card_safe` raises instead.
3. **Optimizer and RNG state never leave the box.** They are training scratch,
   they are pickles, and they are ~30x the size of the weights anyone wants.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

#: Files worth publishing from a PEFT adapter directory. An allowlist rather
#: than a denylist: a future library version adding a new file should require a
#: decision here, not get published because nobody thought to exclude it.
ADAPTER_FILES: tuple[str, ...] = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
)

#: Never published. ``training_args.bin`` is a pickle of a ``TrainingArguments``
#: object carrying absolute paths from this machine; the reproducible source of
#: those settings is ``configs/train/qlora_r16.yaml``, which is in git. The rest
#: is optimizer scratch that only means anything to a resumed run.
NEVER_PUBLISH: tuple[str, ...] = (
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "trainer_state.json",
    "training_args.bin",
)

#: The card must carry this, in substance. Checked case-insensitively on the
#: normalised text so reformatting the card cannot silently drop it.
REQUIRED_CARD_PHRASES: tuple[str, ...] = (
    "not for clinical use",
    "not a medical device",
)


class UnsafeCardError(Exception):
    """Raised when a model card is missing a required safety statement."""


class UnsafeUploadError(Exception):
    """Raised when an upload plan contains a file that must never be published."""


@dataclass(frozen=True)
class HubTargets:
    """Where this project's artifacts live on the Hub."""

    namespace: str
    adapter_name: str = "qwen3-8b-medmcqa-qlora"
    space_name: str = "fine-tune-vs-rag"

    @property
    def adapter_repo(self) -> str:
        return f"{self.namespace}/{self.adapter_name}"

    @property
    def space_repo(self) -> str:
        return f"{self.namespace}/{self.space_name}"


@dataclass
class UploadPlan:
    """What would be pushed, resolved before anything touches the network."""

    repo_id: str
    repo_type: str
    #: ``(local file, path inside the repo)``.
    entries: list[tuple[Path, str]] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(local.stat().st_size for local, _ in self.entries if local.is_file())

    def render(self) -> str:
        lines = [f"{self.repo_type}: {self.repo_id}"]
        for local, remote in sorted(self.entries, key=lambda e: e[1]):
            size = local.stat().st_size / 2**20 if local.is_file() else 0.0
            rename = f"   (from {local.name})" if local.name != Path(remote).name else ""
            lines.append(f"  {size:8.1f} MiB  {remote}{rename}")
        lines.append(f"  {'-' * 8}")
        lines.append(f"  {self.total_bytes / 2**20:8.1f} MiB  total")
        return "\n".join(lines)


def normalise(text: str) -> str:
    """Collapse whitespace and case so a phrase check survives reflowing."""
    return re.sub(r"\s+", " ", text).lower()


def assert_card_safe(card_text: str) -> None:
    """Raise unless the card carries every required safety statement.

    The user made the not-for-clinical-use statement non-negotiable, so it is a
    precondition of publishing rather than a review item. A card that loses the
    warning in an edit cannot reach the Hub.
    """
    flat = normalise(card_text)
    missing = [phrase for phrase in REQUIRED_CARD_PHRASES if phrase not in flat]
    if missing:
        raise UnsafeCardError(
            f"model card is missing required safety statement(s): {missing}. "
            "A medical adapter is not published without them."
        )


def assert_upload_safe(plan: UploadPlan) -> None:
    """Raise if the plan would publish training scratch or a placeholder card."""
    for local, remote in plan.entries:
        if Path(remote).name in NEVER_PUBLISH:
            raise UnsafeUploadError(f"{remote} is training scratch and is never published")
        if "[More Information Needed]" in _peek(local):
            raise UnsafeUploadError(
                f"{local} still contains the library's placeholder card text; "
                "publish docs/model_card.md instead"
            )


def _peek(path: Path, limit: int = 4096) -> str:
    """First few KiB as text, or empty for binary. Used only for the card check."""
    if not path.is_file() or path.suffix not in {".md", ".txt", ".json"}:
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:limit]
    except OSError:  # pragma: no cover - unreadable file is caught at upload
        return ""


def plan_adapter_upload(
    adapter_dir: Path,
    card_path: Path,
    targets: HubTargets,
    *,
    allowed: Iterable[str] = ADAPTER_FILES,
) -> UploadPlan:
    """Resolve exactly which files would be pushed for the adapter.

    The card is uploaded *as* ``README.md``, which is what the Hub renders. The
    adapter directory's own ``README.md`` is deliberately not in ``allowed``.
    """
    adapter_dir = Path(adapter_dir)
    if not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"{adapter_dir} is not a PEFT adapter directory (no adapter_config.json)"
        )
    card_path = Path(card_path)
    if not card_path.is_file():
        raise FileNotFoundError(f"model card not found at {card_path}")

    assert_card_safe(card_path.read_text(encoding="utf-8"))

    plan = UploadPlan(repo_id=targets.adapter_repo, repo_type="model")
    plan.entries.append((card_path, "README.md"))
    for name in allowed:
        candidate = adapter_dir / name
        if candidate.is_file():
            plan.entries.append((candidate, name))

    assert_upload_safe(plan)
    return plan


def plan_space_upload(app_dir: Path, targets: HubTargets) -> UploadPlan:
    """Resolve the Space upload: every file under ``app/``, paths preserved."""
    app_dir = Path(app_dir)
    if not (app_dir / "app.py").is_file():
        raise FileNotFoundError(f"{app_dir} has no app.py")

    plan = UploadPlan(repo_id=targets.space_repo, repo_type="space")
    for path in sorted(app_dir.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            plan.entries.append((path, str(path.relative_to(app_dir))))
    assert_upload_safe(plan)
    return plan


#: Matches a results-table row in the model card: ``| `arm` — … | 56.8% | …``.
_CARD_ROW = re.compile(
    r"^\|\s*\**`(?P<arm>[a-z0-9-]+)`.*?\|\s*\**(?P<accuracy>\d+\.\d)%\**\s*\|"
    r"\s*\[(?P<ci_low>\d+\.\d), (?P<ci_high>\d+\.\d)\]\s*\|"
    r"\s*(?P<p50>\d+) ms\s*\|\s*(?P<tokens>\d+)\s*\|",
    re.MULTILINE,
)


def card_results(card_text: str) -> dict[str, dict[str, float]]:
    """Parse the card's results table back out.

    Exists so a test can prove the published card still matches
    ``results/runs/*.json``. The card is written by hand — it argues, and a
    generated table cannot — so drift is caught rather than prevented.
    """
    return {
        m.group("arm"): {
            "accuracy": float(m.group("accuracy")) / 100,
            "ci_low": float(m.group("ci_low")) / 100,
            "ci_high": float(m.group("ci_high")) / 100,
            "p50_ms": float(m.group("p50")),
            "prompt_tokens": float(m.group("tokens")),
        }
        for m in _CARD_ROW.finditer(card_text)
    }
