"""Immutable pre-fight records."""

from dataclasses import dataclass, replace
from datetime import date
from hashlib import sha1


@dataclass(frozen=True)
class FighterProfile:
    name: str
    age: float | None = None
    height_cm: float | None = None
    reach_cm: float | None = None
    weight_lbs: float | None = None
    stance: str | None = None
    wins: int | None = None
    losses: int | None = None
    draws: int | None = None
    current_win_streak: int | None = None
    current_lose_streak: int | None = None
    longest_win_streak: int | None = None
    total_rounds_fought: int | None = None
    total_title_bouts: int | None = None
    wins_ko: int | None = None
    wins_sub: int | None = None
    wins_dec: int | None = None
    sig_str_landed_pm: float | None = None
    sig_str_pct: float | None = None
    td_landed_per15: float | None = None
    td_pct: float | None = None
    sub_att_per15: float | None = None
    odds_american: float | None = None


@dataclass(frozen=True)
class Bout:
    bout_id: str
    date: date
    weight_class: str
    fighter_a: FighterProfile
    fighter_b: FighterProfile
    a_won: bool
    gender: str | None = None
    title_bout: bool = False
    scheduled_rounds: int = 3

    def swapped(self) -> "Bout":
        return replace(
            self, fighter_a=self.fighter_b, fighter_b=self.fighter_a, a_won=not self.a_won
        )


def make_bout_id(date: date, name1: str, name2: str) -> str:
    import json

    key = [date.isoformat(), *sorted([name1.strip().lower(), name2.strip().lower()])]
    return sha1(json.dumps(key).encode()).hexdigest()[:16]
