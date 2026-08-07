"""Shared fixtures.

CI has no GPU and no network, so tests must not reach for either. Anything that
needs them carries a marker and is deselected by default in CI.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from fvr.config import PROJECT_ROOT


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def isolated_artifacts(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """Redirect the artifact tree into tmp so tests never write real caches."""
    root = Path(str(tmp_path)) / ".artifacts"
    for var in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE"):
        monkeypatch.setenv(var, str(root / "hub"))
    root.mkdir(parents=True, exist_ok=True)
    yield root
