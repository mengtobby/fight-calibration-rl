"""Prediction persistence and aggregation.

Swapped predictions retain ORIGINAL A's perspective: producers store 1 - P(new A
wins), and retain the original outcome. Thus coherent base and swapped records
agree; reports call symmetry_gap(p_base, 1 - p_swapped).
"""

import json
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

VARIANTS = ("base", "odds_in_prompt", "odds_perturbed", "swapped", "reasoning_swap")


@dataclass(frozen=True)
class PredictionRecord:
    bout_id: str
    arm: str
    variant: str
    sample_idx: int
    p_a: float | None
    a_won: bool
    market_p_a: float | None
    shown_market_p_a: float | None = None
    completion: str | None = None
    checkpoint: str | None = None

    def __post_init__(self) -> None:
        for name in ("bout_id", "arm"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty string")
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if type(self.sample_idx) is not int or self.sample_idx < -1:
            raise ValueError("sample_idx must be an integer >= -1")
        if type(self.a_won) is not bool:
            raise ValueError("a_won must be boolean")
        for name in ("p_a", "market_p_a", "shown_market_p_a"):
            p = getattr(self, name)
            if p is not None and (
                isinstance(p, bool)
                or not isinstance(p, (int, float))
                or not math.isfinite(p)
                or not 0 <= p <= 1
            ):
                raise ValueError(f"{name} must be null or a finite probability in [0, 1]")
        for name in ("completion", "checkpoint"):
            if getattr(self, name) is not None and not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be null or a string")


def write_predictions(
    path: str | Path, records: Iterable[PredictionRecord], *, append: bool = False
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a" if append else "w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(asdict(record), allow_nan=False) + "\n")


def read_predictions(path: str | Path) -> list[PredictionRecord]:
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                records.append(PredictionRecord(**json.loads(line)))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid prediction on line {number}: {exc}") from exc
    return records


def aggregate(records: Iterable[PredictionRecord]) -> list[PredictionRecord]:
    groups: dict[tuple, list[PredictionRecord]] = defaultdict(list)
    for record in records:
        groups[(record.arm, record.variant, record.checkpoint, record.bout_id)].append(record)
    result = []
    for samples in groups.values():
        first = samples[0]
        for sample in samples[1:]:
            if (sample.a_won, sample.market_p_a, sample.shown_market_p_a) != (
                first.a_won,
                first.market_p_a,
                first.shown_market_p_a,
            ):
                raise ValueError(f"Inconsistent sample metadata for bout {first.bout_id}")
        parsed = [s.p_a for s in samples if s.p_a is not None]
        result.append(
            replace(first, sample_idx=-1, p_a=sum(parsed) / len(parsed) if parsed else None)
        )
    return result


def parse_rate(records: Iterable[PredictionRecord]) -> float:
    samples = list(records)
    return sum(r.p_a is not None for r in samples) / len(samples) if samples else 0.0
