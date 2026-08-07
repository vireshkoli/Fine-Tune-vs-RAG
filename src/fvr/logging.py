"""Console logging. One configuration call, used by every script."""

from __future__ import annotations

import logging
from typing import Literal

from rich.console import Console
from rich.logging import RichHandler

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]

console = Console(stderr=True)


def setup_logging(level: LogLevel = "INFO") -> logging.Logger:
    """Configure root logging and return the project logger.

    Idempotent — scripts and tests both call it without stacking handlers.
    """
    root = logging.getLogger()
    for handler in tuple(root.handlers):
        root.removeHandler(handler)

    root.setLevel(level)
    root.addHandler(
        RichHandler(
            console=console,
            rich_tracebacks=True,
            show_path=False,
            omit_repeated_times=False,
        )
    )
    # These are chatty at INFO and drown out anything useful during downloads.
    for noisy in ("urllib3", "filelock", "huggingface_hub.file_download"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logging.getLogger("fvr")


def get_logger(name: str) -> logging.Logger:
    """Child logger under the ``fvr`` namespace."""
    return logging.getLogger(f"fvr.{name}")
