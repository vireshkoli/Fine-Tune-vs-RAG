"""Cost accounting, in one currency.

The rule this module exists to enforce: **never mix local GPU-seconds with
hosted API pricing**. Every arm is costed the same way, from one published
rental rate for the hardware the benchmark actually ran on.

A single "cost per query" number is close to meaningless on its own, because
the arms have very different *shapes*:

* fine-tuning is a large one-off cost with cheap short-prompt inference;
* RAG has almost no fixed cost but inflates prefill tokens on every query,
  forever.

So the deliverable is not a scalar but :func:`crossover_volume` — the query
volume at which one arm overtakes another. That number is the actual answer to
"when is fine-tuning worth it", and unlike an accuracy delta it generalises
beyond this dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class RateCard:
    """Published hardware prices. Committed in ``configs/eval/cost.yaml``.

    ``source_url`` and ``retrieved`` are part of the data, not documentation:
    a cost claim without a citable rate and date is not defensible.
    """

    gpu_name: str
    gpu_usd_per_hour: float
    cpu_usd_per_hour: float
    source_url: str
    retrieved: str

    @property
    def gpu_usd_per_second(self) -> float:
        return self.gpu_usd_per_hour / SECONDS_PER_HOUR

    @property
    def cpu_usd_per_second(self) -> float:
        return self.cpu_usd_per_hour / SECONDS_PER_HOUR


@dataclass(frozen=True)
class FixedCosts:
    """One-off costs, amortised over the assumed lifetime query volume."""

    #: QLoRA training, summed across the seeds actually used for the reported arm.
    train_gpu_seconds: float = 0.0
    #: Embedding the corpus and building the FAISS index.
    index_build_gpu_seconds: float = 0.0

    def usd(self, rates: RateCard) -> float:
        return (self.train_gpu_seconds + self.index_build_gpu_seconds) * rates.gpu_usd_per_second


@dataclass(frozen=True)
class PerQueryCosts:
    """Marginal costs, measured per query."""

    #: Wall-clock GPU time for prefill + decode.
    inference_gpu_seconds: float
    #: CPU time for retrieval. Small, but counted so RAG is not flattered.
    retrieval_cpu_seconds: float = 0.0

    def usd(self, rates: RateCard) -> float:
        return (
            self.inference_gpu_seconds * rates.gpu_usd_per_second
            + self.retrieval_cpu_seconds * rates.cpu_usd_per_second
        )


@dataclass(frozen=True)
class ArmCost:
    """The complete cost model for one arm."""

    arm: str
    fixed: FixedCosts
    per_query: PerQueryCosts
    rates: RateCard

    def usd_per_query(self, volume: int) -> float:
        """Average cost per query once ``volume`` queries have been served."""
        if volume <= 0:
            raise ValueError("volume must be positive")
        return self.fixed.usd(self.rates) / volume + self.per_query.usd(self.rates)

    def usd_per_1k(self, volume: int) -> float:
        return 1000 * self.usd_per_query(volume)

    def marginal_usd_per_query(self) -> float:
        """Cost per query ignoring amortisation — the asymptote as volume grows."""
        return self.per_query.usd(self.rates)


def crossover_volume(a: ArmCost, b: ArmCost) -> int | None:
    """Query volume at which ``a`` becomes cheaper than ``b``.

    Solving ``F_a/N + m_a = F_b/N + m_b`` gives ``N = (F_a - F_b)/(m_b - m_a)``.

    Returns ``None`` when they never cross: either one arm dominates at every
    volume (cheaper fixed *and* cheaper marginal), or the marginal rates are
    equal so the gap never closes. A ``None`` is a real finding, not a failure —
    it means the choice does not depend on scale.
    """
    fixed_gap = a.fixed.usd(a.rates) - b.fixed.usd(b.rates)
    marginal_gap = b.marginal_usd_per_query() - a.marginal_usd_per_query()

    if marginal_gap == 0:
        return None
    volume = fixed_gap / marginal_gap
    if volume <= 0:
        return None
    return round(volume)


@dataclass
class CostCurve:
    """Cost-per-query against query volume — the headline chart."""

    volumes: list[int]
    series: dict[str, list[float]] = field(default_factory=dict)

    @classmethod
    def build(cls, arms: dict[str, ArmCost], volumes: list[int]) -> CostCurve:
        if any(v <= 0 for v in volumes):
            raise ValueError("volumes must all be positive")
        return cls(
            volumes=list(volumes),
            series={name: [arm.usd_per_query(v) for v in volumes] for name, arm in arms.items()},
        )

    def cheapest_at(self, volume: int) -> str:
        """Which arm wins at a given volume."""
        if volume not in self.volumes:
            raise KeyError(f"{volume} is not on the curve")
        i = self.volumes.index(volume)
        return min(self.series, key=lambda name: self.series[name][i])
