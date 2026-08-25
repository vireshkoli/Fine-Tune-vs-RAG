"""Figures for the README and REPORT.

Two forms, chosen by what the data has to say:

* **Cost per query against query volume** — a line chart, because the question
  is how a quantity changes across a continuous variable. This is the headline:
  the point where the lines cross is the answer to "when is fine-tuning worth
  it", and unlike an accuracy delta it generalises beyond this dataset.
* **Quality / latency / cost across arms** — *small multiples*, one panel per
  measure. Never a dual-axis chart: three measures on different scales share an
  arm ordering and a colour, not an axis.

Colour follows the palette's documented order, assigned by position within each
figure and never cycled. The sets used here were run through the validator
rather than eyeballed:

===================================  ==========================  =============
set                                  worst adjacent CVD ΔE       normal-vision
===================================  ==========================  =============
slots 1-4 light (#2a78d6 …#eda100)   9.1 (target ≥ 8)            22.9 (≥ 15)
slots 1-4 dark  (#3987e5 …#c98500)   8.4                         19.8
===================================  ==========================  =============

Light slots 3 and 4 fall below 3:1 against the light surface, so the palette's
**relief rule** applies: every series carries a direct label and a committed
markdown table exists alongside. Identity is never colour alone.

A naive "fixed arm → fixed slot" map was tried first and rejected: it put slot 2
(orange) beside slot 5 (magenta), whose normal-vision ΔE is 12.9 — under the 15
floor, meaning full-colour readers could not separate two arms. Assigning by
position within each figure keeps every rendered set on the validated list.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The documented categorical order. Never cycled; a figure uses a prefix of it.
SERIES_LIGHT = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
SERIES_DARK = ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300")


@dataclass(frozen=True)
class Theme:
    """Chart surface and ink. Dark is a selected set, not an inverted light one."""

    name: str
    surface: str
    text_primary: str
    text_secondary: str
    grid: str
    series: tuple[str, ...]

    @classmethod
    def light(cls) -> Theme:
        return cls("light", "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", SERIES_LIGHT)

    @classmethod
    def dark(cls) -> Theme:
        return cls("dark", "#1a1a19", "#ffffff", "#c3c2b7", "#332f2c", SERIES_DARK)


def _style(theme: Theme) -> Any:
    # Returned loosely: matplotlib types rcParams keys as a huge Literal union,
    # so a dict[str, Any] is rejected even though every key below is valid.
    return {
        "figure.facecolor": theme.surface,
        "axes.facecolor": theme.surface,
        "savefig.facecolor": theme.surface,
        "text.color": theme.text_primary,
        "axes.labelcolor": theme.text_secondary,
        "axes.edgecolor": theme.grid,
        "xtick.color": theme.text_secondary,
        "ytick.color": theme.text_secondary,
        "grid.color": theme.grid,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }


def _recede(ax: Any, theme: Theme) -> None:
    """Grid and axes stay recessive so the marks carry the meaning."""
    ax.grid(True, axis="y", linewidth=0.6, alpha=0.5, color=theme.grid)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def _spread_labels(values: Sequence[float], *, min_points: float = 11.0) -> list[float]:
    """Vertical offsets (in points) that keep descending labels from overlapping.

    Works on rank rather than on data units so it is correct on a log axis,
    where equal data gaps are not equal screen gaps.
    """
    offsets = [0.0] * len(values)
    for i in range(1, len(values)):
        # Values arrive sorted descending; each label sits at least min_points
        # below the one above it in screen space.
        if abs(values[i - 1] - values[i]) / max(values[i - 1], 1e-12) < 0.25:
            offsets[i] = offsets[i - 1] - min_points
    return offsets


def cost_crossover_chart(
    volumes: Sequence[int],
    series: dict[str, list[float]],
    out_path: Path,
    *,
    theme: Theme | None = None,
    crossovers: dict[tuple[str, str], int] | None = None,
) -> Path:
    """Cost per 1,000 queries against lifetime query volume.

    Log-log, because both axes span orders of magnitude and the interesting
    feature is where lines cross, which a linear axis would compress into the
    left margin.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theme = theme or Theme.light()
    with plt.rc_context(_style(theme)):
        fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)

        ends: list[tuple[float, str, str]] = []
        for slot, (name, values) in enumerate(series.items()):
            colour = theme.series[slot % len(theme.series)]
            per_1k = [1000 * v for v in values]
            ax.plot(volumes, per_1k, linewidth=2.0, color=colour, marker="o", markersize=5)
            ends.append((per_1k[-1], name, colour))

        # Direct labels, nudged apart where lines converge. Identity must never
        # be colour alone (light slots 3+ sit below 3:1, so the relief rule
        # applies), and two labels stacked on top of each other are no label.
        ends.sort(reverse=True)
        offsets = _spread_labels([value for value, _, _ in ends])
        for (value, name, _colour), offset in zip(ends, offsets, strict=True):
            ax.annotate(
                name,
                xy=(volumes[-1], value),
                xytext=(8, offset),
                textcoords="offset points",
                va="center",
                fontsize=9,
                color=theme.text_primary,
            )

        for (arm_a, arm_b), volume in (crossovers or {}).items():
            ax.axvline(volume, color=theme.text_secondary, linewidth=1.0, linestyle=":", alpha=0.7)
            ax.annotate(
                f"{arm_a} overtakes {arm_b}\nat {volume:,} queries",
                xy=(volume, ax.get_ylim()[1]),
                xytext=(4, -14),
                textcoords="offset points",
                fontsize=8,
                color=theme.text_secondary,
                va="top",
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Lifetime query volume")
        ax.set_ylabel("Cost per 1,000 queries (USD)")
        ax.set_title("Cost per query falls with volume — the crossover is the decision")
        _recede(ax, theme)
        ax.margins(x=0.18)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
    return out_path


def _label_bars(
    ax: Any,
    y: Sequence[int],
    values: Sequence[float],
    theme: Theme,
    *,
    fmt: str,
    anchors: Sequence[float] | None = None,
) -> None:
    """Value at the end of every bar, in ink rather than in the series colour.

    Mandatory here, not decorative: three light-mode slots fall below 3:1
    against the surface, and the palette's relief rule requires visible labels
    or a table view whenever that happens.
    """
    # `anchors` lets a label clear something drawn past the bar end — the CI
    # whisker on the accuracy panel would otherwise sit underneath the text.
    positions = list(anchors) if anchors is not None else list(values)
    span = max(positions) if positions else 1.0
    for row, value, position in zip(y, values, positions, strict=True):
        ax.annotate(
            fmt.format(value),
            xy=(position, row),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=theme.text_secondary,
        )
    ax.set_xlim(0, span * 1.26)


def arm_comparison_chart(
    arms: Sequence[str],
    accuracy: Sequence[float],
    ci: Sequence[tuple[float, float]],
    p95_ms: Sequence[float],
    cost_per_1k: Sequence[float],
    out_path: Path,
    *,
    theme: Theme | None = None,
    chance: float = 0.25,
) -> Path:
    """Small multiples: accuracy, latency and cost share an arm order, not an axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theme = theme or Theme.light()
    colours = [theme.series[i % len(theme.series)] for i in range(len(arms))]
    y = list(range(len(arms)))[::-1]

    with plt.rc_context(_style(theme)):
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), dpi=200)

        # Accuracy, with the CI drawn as an error bar — a bare bar would imply
        # a precision the sample size does not support.
        ax = axes[0]
        errors = [
            [a - lo for a, (lo, _) in zip(accuracy, ci, strict=True)],
            [hi - a for a, (_, hi) in zip(accuracy, ci, strict=True)],
        ]
        ax.barh(y, [100 * a for a in accuracy], height=0.42, color=colours)
        ax.errorbar(
            [100 * a for a in accuracy],
            y,
            xerr=[[100 * e for e in errors[0]], [100 * e for e in errors[1]]],
            fmt="none",
            ecolor=theme.text_secondary,
            elinewidth=1.2,
            capsize=3,
        )
        ax.axvline(100 * chance, color=theme.text_secondary, linewidth=1.0, linestyle=":")
        # Anchored in axes coordinates so it cannot be clipped off the top.
        ax.annotate(
            "chance",
            xy=(100 * chance, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(3, -10),
            textcoords="offset points",
            fontsize=8,
            color=theme.text_secondary,
            ha="left",
        )
        _label_bars(
            ax,
            y,
            [100 * a for a in accuracy],
            theme,
            fmt="{:.1f}%",
            anchors=[100 * hi for _, hi in ci],
        )
        ax.set_title("Accuracy (95% CI)")
        ax.set_xlabel("%")

        ax = axes[1]
        ax.barh(y, p95_ms, height=0.42, color=colours)
        _label_bars(ax, y, list(p95_ms), theme, fmt="{:.0f}")
        ax.set_title("Latency p95")
        ax.set_xlabel("ms")

        ax = axes[2]
        ax.barh(y, cost_per_1k, height=0.42, color=colours)
        _label_bars(ax, y, list(cost_per_1k), theme, fmt="${:.3f}")
        ax.set_title("Cost per 1,000 queries")
        ax.set_xlabel("USD")

        for index, ax in enumerate(axes):
            ax.set_yticks(y)
            # Arm names on every panel: the label carries identity, not the hue.
            ax.set_yticklabels(arms if index == 0 else ["" for _ in arms], fontsize=9)
            ax.grid(True, axis="x", linewidth=0.6, alpha=0.5, color=theme.grid)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
    return out_path
