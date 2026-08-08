"""The only place this project calls ``from_pretrained``.

Centralised for the sake of the comparison. If arms loaded their own weights,
"same base model" would be an assumption; here it is a fact enforced by a
single cache, so arms 1/2/2b literally share one model object and arms 3/4/4b
attach an adapter to that same object.

Heavy imports (torch, transformers, peft) are deferred into the functions that
need them, so ``fvr.models`` can be imported — and its config validated — in
CPU-only CI where those packages are not installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedTokenizerBase

Quantization = Literal["none", "nf4", "int8"]


class ModelConfig(BaseModel):
    """Everything needed to load a model, from ``configs/model/*.yaml``."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    name: str
    repo_id: str
    #: A commit SHA, never "main". The benchmark must still reproduce after the
    #: lab machine is wiped, and a moving reference would break that silently.
    revision: str = Field(min_length=7)
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    device: int = 0
    enable_thinking: bool = False
    max_seq_length: int = 4096
    max_new_tokens: int = 8
    temperature: float = 0.0
    quantization: Quantization = "none"

    @property
    def torch_dtype(self) -> Any:
        import torch

        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.dtype]


def load_model_config(path: Path | str) -> ModelConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return ModelConfig(**raw)


@dataclass
class LoadedModel:
    """A model, its tokenizer, and the config it was built from."""

    #: ``PreTrainedModel`` or a ``PeftModel`` wrapping one. PeftModel is not a
    #: subclass, and the two share no protocol, so this stays deliberately loose
    #: rather than pretending a union that neither library declares.
    model: Any
    tokenizer: PreTrainedTokenizerBase
    config: ModelConfig
    #: Adapter path if one is attached, else None. Recorded in results so a run
    #: can never be ambiguous about which weights produced it.
    adapter_path: str | None = None

    @property
    def device(self) -> Any:
        return self.model.device

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.config.name,
            "repo_id": self.config.repo_id,
            "revision": self.config.revision,
            "dtype": self.config.dtype,
            "quantization": self.config.quantization,
            "adapter": self.adapter_path,
            "enable_thinking": self.config.enable_thinking,
        }


def _quantization_config(quantization: Quantization, dtype: Any) -> Any:
    if quantization == "none":
        return None
    from transformers import BitsAndBytesConfig

    if quantization == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )


#: Cache keyed by (repo, revision, dtype, quantization, device). Two arms with
#: the same base therefore get the *same object*, not two equal ones.
_CACHE: dict[tuple[str, ...], LoadedModel] = {}


def load_base_model(config: ModelConfig, *, use_cache: bool = True) -> LoadedModel:
    """Load base weights and tokenizer. Repeated calls return the same object."""
    from fvr.config import bootstrap_env

    bootstrap_env()  # pin caches before transformers resolves anything

    key = (
        config.repo_id,
        config.revision,
        config.dtype,
        config.quantization,
        str(config.device),
    )
    if use_cache and key in _CACHE:
        return _CACHE[key]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.repo_id, revision=config.revision, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.repo_id,
        revision=config.revision,
        dtype=config.torch_dtype,
        quantization_config=_quantization_config(config.quantization, config.torch_dtype),
        device_map={"": config.device} if torch.cuda.is_available() else None,
    )
    model.eval()

    loaded = LoadedModel(model=model, tokenizer=tokenizer, config=config)
    if use_cache:
        _CACHE[key] = loaded
    return loaded


def attach_adapter(base: LoadedModel, adapter_path: str | Path) -> LoadedModel:
    """Attach a LoRA adapter to an already-loaded base.

    Deliberately wraps the *same* base object rather than reloading, so a
    fine-tuned arm is provably the identical starting weights plus an adapter.
    """
    from peft import PeftModel

    model = PeftModel.from_pretrained(base.model, str(adapter_path))
    model.eval()
    return LoadedModel(
        model=model,
        tokenizer=base.tokenizer,
        config=base.config,
        adapter_path=str(adapter_path),
    )


def clear_cache() -> None:
    """Drop cached models and free VRAM. Used between arms in a sweep."""
    _CACHE.clear()
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover - CPU-only CI
        pass
