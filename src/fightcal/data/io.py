"""Portable JSONL persistence for bout records."""

import json
from collections.abc import Iterable
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

from fightcal.data.schema import Bout, FighterProfile


def bout_to_dict(b: Bout) -> dict[str, Any]:
    result = asdict(b)
    result["date"] = b.date.isoformat()
    return result


def bout_from_dict(d: dict[str, Any]) -> Bout:
    values = dict(d)
    try:
        values["date"] = date.fromisoformat(values["date"])
        for corner in ("fighter_a", "fighter_b"):
            values[corner] = FighterProfile(**values[corner])
        return Bout(**values)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid bout record: {exc}") from exc


def save_bouts_jsonl(path: str | Path, bouts: Iterable[Bout]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        for b in bouts:
            stream.write(json.dumps(bout_to_dict(b), allow_nan=False) + "\n")


def load_bouts_jsonl(path: str | Path) -> list[Bout]:
    with Path(path).open(encoding="utf-8") as stream:
        return [bout_from_dict(json.loads(line)) for line in stream if line.strip()]
