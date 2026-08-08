"""Latency measurement.

Two things make the difference between a real number and a misleading one, and
both are enforced here rather than left to discipline:

* **Warmup runs are discarded.** The first few calls pay for CUDA context
  creation, kernel autotuning and cache population. Including them inflates the
  mean and destroys p95.
* **The GPU is synchronised before stopping the clock.** CUDA kernels are
  launched asynchronously, so timing without a sync measures how fast Python
  can queue work, not how fast the model runs.

p50 and p95 are reported rather than the mean: the mean of a right-skewed
latency distribution is dominated by its tail and describes no actual request.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TypeVar

import numpy as np

T = TypeVar("T")

DEFAULT_WARMUP = 3


def synchronize() -> None:
    """Block until queued GPU work has finished. No-op without CUDA."""
    try:
        import torch
    except ImportError:  # pragma: no cover - CPU-only CI
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass
class LatencyRecorder:
    """Collects per-call timings and summarises them.

    The first ``warmup`` samples are recorded but excluded from every statistic,
    so a run can still show what warmup cost without letting it pollute results.
    """

    warmup: int = DEFAULT_WARMUP
    samples: list[float] = field(default_factory=list)

    @contextmanager
    def measure(self) -> Iterator[None]:
        synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            synchronize()
            self.samples.append(time.perf_counter() - start)

    def time_call(self, fn: Callable[[], T]) -> T:
        with self.measure():
            return fn()

    @property
    def measured(self) -> list[float]:
        """Samples after discarding warmup."""
        return self.samples[self.warmup :]

    @property
    def warmup_samples(self) -> list[float]:
        return self.samples[: self.warmup]

    def summary(self) -> LatencySummary:
        return LatencySummary.from_samples(self.measured, discarded=len(self.warmup_samples))


@dataclass(frozen=True)
class LatencySummary:
    n: int
    discarded_warmup: int
    p50: float
    p95: float
    p99: float
    mean: float
    minimum: float
    maximum: float

    @classmethod
    def from_samples(cls, samples: list[float], *, discarded: int = 0) -> LatencySummary:
        if not samples:
            raise ValueError("no samples left after discarding warmup")
        values = np.asarray(samples, dtype=float)
        return cls(
            n=values.size,
            discarded_warmup=discarded,
            p50=float(np.percentile(values, 50)),
            p95=float(np.percentile(values, 95)),
            p99=float(np.percentile(values, 99)),
            mean=float(values.mean()),
            minimum=float(values.min()),
            maximum=float(values.max()),
        )

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n": self.n,
            "discarded_warmup": self.discarded_warmup,
            "p50_s": self.p50,
            "p95_s": self.p95,
            "p99_s": self.p99,
            "mean_s": self.mean,
            "min_s": self.minimum,
            "max_s": self.maximum,
        }

    def __str__(self) -> str:
        return f"p50={self.p50 * 1000:.0f}ms p95={self.p95 * 1000:.0f}ms (n={self.n})"
