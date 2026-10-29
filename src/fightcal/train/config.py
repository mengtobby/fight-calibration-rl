"""Validated, portable experiment configurations."""

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Self


@dataclass(frozen=True)
class _RunConfig:
    arm: str
    train_path: str
    val_path: str
    output_dir: str
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    seed: int = 42
    anonymize: bool = True
    include_odds: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 5e-6
    max_steps: int = 300
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    logging_steps: int = 10
    save_steps: int = 100
    max_train_bouts: int | None = None

    def __post_init__(self) -> None:
        for name in ("arm", "train_path", "val_path", "output_dir", "model_name"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        for name in ("anonymize", "include_odds"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in (
            "lora_r",
            "lora_alpha",
            "max_steps",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "logging_steps",
            "save_steps",
            "max_train_bouts",
            "num_generations",
            "max_prompt_length",
            "max_completion_length",
            "max_seq_length",
        ):
            value = getattr(self, name, None)
            if value is None and name == "max_train_bouts":
                continue
            if hasattr(self, name) and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "temperature", "num_epochs"):
            if hasattr(self, name):
                self._number(name, positive=True)
        for name in ("lora_dropout", "beta"):
            if hasattr(self, name):
                self._number(name, positive=False)
        if self.lora_dropout >= 1:
            raise ValueError("lora_dropout must be in [0, 1)")

    def _number(self, name: str, *, positive: bool) -> None:
        value = getattr(self, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or (value <= 0 if positive else value < 0)
        ):
            raise ValueError(
                f"{name} must be finite and {'positive' if positive else 'nonnegative'}"
            )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        if not isinstance(d, dict):
            raise ValueError("config must be an object")
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        try:
            return cls(**d)
        except TypeError as exc:
            raise ValueError(f"Invalid config: {exc}") from exc

    @classmethod
    def from_json(cls, path: str | Path) -> Self:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GRPORunConfig(_RunConfig):
    kind: str = "grpo"
    reward: str = "brier"
    num_generations: int = 8
    max_prompt_length: int = 768
    max_completion_length: int = 384
    temperature: float = 0.9
    beta: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind != "grpo":
            raise ValueError("kind must be grpo")
        if self.reward not in ("brier", "log", "correct"):
            raise ValueError("reward must be brier, log, or correct")
        batch = self.per_device_train_batch_size * self.gradient_accumulation_steps
        if batch % self.num_generations:
            raise ValueError("effective batch size must be divisible by num_generations")


@dataclass(frozen=True)
class SFTRunConfig(_RunConfig):
    kind: str = "sft"
    learning_rate: float = 1e-4
    max_seq_length: int = 1024
    num_epochs: float = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind != "sft":
            raise ValueError("kind must be sft")


def load_config(path: str | Path) -> GRPORunConfig | SFTRunConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("kind") not in ("grpo", "sft"):
        raise ValueError("config kind must be grpo or sft")
    cls = GRPORunConfig if data["kind"] == "grpo" else SFTRunConfig
    return cls.from_dict(data)
