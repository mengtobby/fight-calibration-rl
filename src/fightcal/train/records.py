"""Pure conversational dataset preparation and probability logging."""

import json
import math
import random
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from fightcal.data.odds import market_prob_a
from fightcal.data.schema import Bout
from fightcal.prompts import ANSWER_PREFIX, PromptOptions, build_messages


def grpo_records(bouts: Sequence[Bout], opts: PromptOptions) -> list[dict[str, Any]]:
    return [
        {
            "prompt": build_messages(b, replace(opts, mode="reason_prob")),
            "a_won": b.a_won,
            "bout_id": b.bout_id,
            "market_p_a": market_prob_a(b),
        }
        for b in bouts
    ]


def sft_records(bouts: Sequence[Bout], opts: PromptOptions) -> list[dict[str, Any]]:
    return [
        {
            "prompt": build_messages(b, replace(opts, mode="pick")),
            "completion": [
                {
                    "role": "assistant",
                    "content": ANSWER_PREFIX + ("A" if b.a_won else "B") + "</answer>",
                }
            ],
        }
        for b in bouts
    ]


def select_subset(bouts: Sequence[Bout], n: int | None, seed: int) -> list[Bout]:
    if n is not None and (type(n) is not int or n <= 0):
        raise ValueError("n must be a positive integer or None")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return (
        list(bouts) if n is None or n >= len(bouts) else random.Random(seed).sample(list(bouts), n)
    )


class ProbabilityLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.step = 0

    def set_step(self, step: int) -> None:
        if type(step) is not int or step < 0:
            raise ValueError("step must be a nonnegative integer")
        self.step = step

    def __call__(self, parsed: Sequence[float | None], labels: Sequence[bool]) -> None:
        if len(parsed) != len(labels):
            raise ValueError("parsed and labels must have equal lengths")
        rows = []
        for p, label in zip(parsed, labels, strict=True):
            if p is not None and (
                isinstance(p, bool)
                or not isinstance(p, (float, int))
                or not math.isfinite(p)
                or not 0 <= p <= 1
            ):
                raise ValueError("probabilities must be None or finite values in [0, 1]")
            if label not in (True, False):
                raise ValueError("labels must be binary outcomes")
            rows.append({"step": self.step, "p": p, "a_won": bool(label)})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
