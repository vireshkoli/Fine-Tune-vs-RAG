"""Seeding tests. Reproducibility is a headline claim, so it gets asserted."""

from __future__ import annotations

import os
import random

from fvr.seeding import SeedReport, set_all_seeds


def test_reports_the_seed_it_was_given() -> None:
    assert set_all_seeds(123).seed == 123


def test_python_rng_is_reproducible() -> None:
    set_all_seeds(7)
    first = [random.random() for _ in range(5)]
    set_all_seeds(7)
    assert [random.random() for _ in range(5)] == first


def test_different_seeds_diverge() -> None:
    set_all_seeds(1)
    first = [random.random() for _ in range(5)]
    set_all_seeds(2)
    assert [random.random() for _ in range(5)] != first


def test_sets_pythonhashseed() -> None:
    set_all_seeds(99)
    assert os.environ["PYTHONHASHSEED"] == "99"


def test_works_without_torch_installed() -> None:
    # CI is CPU-only and has no torch; seeding must degrade rather than raise.
    report = set_all_seeds(5)
    assert isinstance(report, SeedReport)
    if not report.torch_seeded:
        assert not report.cuda_seeded


def test_numpy_is_reproducible() -> None:
    numpy = __import__("numpy")
    set_all_seeds(11)
    first = numpy.random.rand(4).tolist()
    set_all_seeds(11)
    assert numpy.random.rand(4).tolist() == first
