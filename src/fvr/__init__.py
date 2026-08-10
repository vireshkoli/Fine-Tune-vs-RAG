"""Fine-tune vs. retrieve: a controlled benchmark in clinical QA.

Importing this package pins every ML cache into ``$PROJECT_ROOT/.artifacts``
**as an import side effect**. That is deliberate and load-bearing rather than
convenient.

``huggingface_hub`` reads its cache location into module constants at import
time, so ``HF_HOME`` has to be set before it is imported — not before it is
*used*. Calling ``bootstrap_env()`` at the top of each script is not enough,
because a script imports ``fvr.data.loaders`` first and that module imports
``huggingface_hub`` transitively, freezing the default path before the script
body ever runs.

This is not hypothetical. It leaked roughly 18GB — Qwen3-8B, bge-large, MIRIAD
and MedMCQA — into the shared ``~/.cache/huggingface`` on the lab machine,
which this project has no business writing to, and downloaded Qwen3-8B twice.
Doing it here means any ``import fvr.anything`` isolates the caches first, and
``tests/test_config.py`` asserts the property against exactly the import order
that broke it.
"""

from __future__ import annotations

from fvr.config import bootstrap_env as _bootstrap_env

_bootstrap_env()

__version__ = "0.1.0"

__all__ = ["__version__"]
