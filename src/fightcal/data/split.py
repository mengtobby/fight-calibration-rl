"""Stable corner assignment and chronological partitions."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from hashlib import sha256

from fightcal.data.schema import Bout


def randomize_corners(bouts: Iterable[Bout], seed: int) -> list[Bout]:
    return [
        b.swapped() if sha256(f"{seed}:{b.bout_id}".encode()).digest()[0] < 128 else b
        for b in bouts
    ]


@dataclass(frozen=True)
class Split:
    train: tuple[Bout, ...]
    val: tuple[Bout, ...]
    test: tuple[Bout, ...]


def temporal_split(bouts: Iterable[Bout], *, val_start: date, test_start: date) -> Split:
    if val_start >= test_start:
        raise ValueError("val_start must precede test_start")
    ordered = sorted(bouts, key=lambda b: (b.date, b.bout_id))
    return Split(
        tuple(b for b in ordered if b.date < val_start),
        tuple(b for b in ordered if val_start <= b.date < test_start),
        tuple(b for b in ordered if b.date >= test_start),
    )


def a_won_rate(bouts: Iterable[Bout]) -> float:
    records = list(bouts)
    if not records:
        raise ValueError("Cannot compute win rate for empty bouts")
    return sum(b.a_won for b in records) / len(records)
