"""Seeding and determinism.

Every number in the README has to be reproducible from a stated seed, so this
is the only place seeds are set. ``set_all_seeds`` is safe to call before torch
is installed — the GPU libraries are seeded only if importable, which keeps the
CPU-only CI path working.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SeedReport:
    """What was actually seeded, so a run can record it rather than assume it."""

    seed: int
    torch_seeded: bool
    cuda_seeded: bool
    deterministic_algorithms: bool


def set_all_seeds(seed: int, *, deterministic: bool = True) -> SeedReport:
    """Seed Python, NumPy and torch; return what took effect.

    ``deterministic`` trades a little throughput for run-to-run reproducibility.
    It stays on for measured runs and is turned off only for the ablation sweep,
    where wall-clock matters more than bit-exactness.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is present wherever it matters
        pass

    torch_seeded = cuda_seeded = deterministic_on = False
    try:
        import torch
    except ImportError:
        return SeedReport(seed, torch_seeded, cuda_seeded, deterministic_on)

    torch.manual_seed(seed)
    torch_seeded = True

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        cuda_seeded = True

    if deterministic:
        # cuBLAS needs this set before the first CUDA context to honour
        # deterministic reductions; harmless if the context already exists.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            deterministic_on = True
        except (RuntimeError, AttributeError):  # pragma: no cover - version dependent
            deterministic_on = False

    return SeedReport(seed, torch_seeded, cuda_seeded, deterministic_on)
