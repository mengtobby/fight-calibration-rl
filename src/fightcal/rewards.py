"""Proper scoring rewards and lightweight TRL adapters."""

import math
from collections.abc import Callable
from typing import Any

from fightcal.prompts import parse_probability

PENALTIES: dict[str, float] = {"brier": -0.5, "correct": -0.5, "log": math.log(0.01) - 1}


def completion_text(completion: str | list[dict[str, str]]) -> str:
    if isinstance(completion, str):
        return completion
    if not isinstance(completion, list) or any(
        not isinstance(message, dict) or not isinstance(message.get("content"), str)
        for message in completion
    ):
        raise ValueError("completion must be text or a list of messages with string content")
    return "\n".join(m["content"] for m in completion if m.get("role") == "assistant")


def _validate(p: float | None, a_won: bool) -> None:
    if a_won not in (True, False):
        raise ValueError("a_won must be a binary outcome")
    if p is not None and (not math.isfinite(p) or not 0 <= p <= 1):
        raise ValueError("probability must be finite and in [0, 1]")


def brier_reward(p: float | None, a_won: bool) -> float:
    _validate(p, a_won)
    return PENALTIES["brier"] if p is None else 1 - (p - a_won) ** 2


def log_reward(p: float | None, a_won: bool, clip: float = 0.01) -> float:
    _validate(p, a_won)
    if not math.isfinite(clip) or not 0 < clip < 0.5:
        raise ValueError("clip must be finite and in (0, 0.5)")
    if p is None:
        return math.log(clip) - 1
    clipped = min(max(p, clip), 1 - clip)
    return math.log(clipped) if a_won else math.log1p(-clipped)


def correctness_reward(p: float | None, a_won: bool) -> float:
    _validate(p, a_won)
    if p is None:
        return PENALTIES["correct"]
    return 0.5 if p == 0.5 else float((p > 0.5) == a_won)


REWARD_FNS: dict[str, Callable[..., float]] = {
    "brier": brier_reward,
    "log": log_reward,
    "correct": correctness_reward,
}


def make_trl_reward(
    kind: str,
    *,
    label_key: str = "a_won",
    on_batch: Callable[[list[float | None], list[bool]], None] | None = None,
) -> Callable[..., list[float]]:
    if kind not in REWARD_FNS:
        raise ValueError(f"Unknown reward kind: {kind}")

    def reward(completions: list[Any], **kwargs: Any) -> list[float]:
        raw_labels = kwargs.get(label_key)
        if raw_labels is None:
            raise ValueError(f"Missing labels: {label_key}")
        try:
            labels = list(raw_labels)
        except TypeError as exc:
            raise ValueError("labels must be a sequence") from exc
        if len(labels) != len(completions):
            raise ValueError("labels and completions must have equal lengths")
        for label in labels:
            _validate(None, label)
        labels = [bool(label) for label in labels]
        parsed = [parse_probability(completion_text(c)) for c in completions]
        scores = [REWARD_FNS[kind](p, y) for p, y in zip(parsed, labels, strict=True)]
        if on_batch is not None:
            on_batch(parsed, labels)
        return scores

    reward.__name__ = f"{kind}_reward"
    return reward
