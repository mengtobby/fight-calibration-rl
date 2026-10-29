"""Load the legacy and camel-case Ultimate UFC Dataset layouts."""

import logging
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from fightcal.data.odds import american_to_decimal
from fightcal.data.schema import Bout, FighterProfile, make_bout_id

logger = logging.getLogger(__name__)

_FIELDS = {
    "name": ("fighter", "Fighter"),
    "age": ("age", "Age"),
    "height_cm": ("Height_cms", "HeightCms"),
    "reach_cm": ("Reach_cms", "ReachCms"),
    "weight_lbs": ("Weight_lbs", "WeightLbs"),
    "stance": ("Stance", "Stance"),
    "wins": ("wins", "Wins"),
    "losses": ("losses", "Losses"),
    "draws": ("draw", "Draws"),
    "current_win_streak": ("current_win_streak", "CurrentWinStreak"),
    "current_lose_streak": ("current_lose_streak", "CurrentLoseStreak"),
    "longest_win_streak": ("longest_win_streak", "LongestWinStreak"),
    "total_rounds_fought": ("total_rounds_fought", "TotalRoundsFought"),
    "total_title_bouts": ("total_title_bouts", "TotalTitleBouts"),
    "wins_ko": ("win_by_KO/TKO", "WinsByKO"),
    "wins_sub": ("win_by_Submission", "WinsBySubmission"),
    "decision_unanimous": ("win_by_Decision_Unanimous", "WinsByDecisionUnanimous"),
    "decision_split": ("win_by_Decision_Split", "WinsByDecisionSplit"),
    "decision_majority": ("win_by_Decision_Majority", "WinsByDecisionMajority"),
    "sig_str_landed_pm": ("avg_SIG_STR_landed", "AvgSigStrLanded"),
    "sig_str_pct": ("avg_SIG_STR_pct", "AvgSigStrPct"),
    "td_landed_per15": ("avg_TD_landed", "AvgTDLanded"),
    "td_pct": ("avg_TD_pct", "AvgTDPct"),
    "sub_att_per15": ("avg_SUB_ATT", "AvgSubAtt"),
    "odds_american": ("odds", "Odds"),
}
_COMMON = {
    "date": ("date", "Date"),
    "winner": ("Winner", "Winner"),
    "title_bout": ("title_bout", "TitleBout"),
    "weight_class": ("weight_class", "WeightClass"),
    "gender": ("gender", "Gender"),
    "scheduled_rounds": ("no_of_rounds", "NumberOfRounds"),
}
COLUMN_MAPS: dict[str, dict[str, str]] = {}
for _index, _layout in enumerate(("legacy", "camel")):
    COLUMN_MAPS[_layout] = {key: names[_index] for key, names in _COMMON.items()}
    for _corner, _prefix in zip(
        ("red", "blue"), (("R_", "B_"), ("Red", "Blue"))[_index], strict=True
    ):
        COLUMN_MAPS[_layout].update(
            {f"{_corner}_{key}": _prefix + names[_index] for key, names in _FIELDS.items()}
        )
_REQUIRED = ("red_name", "blue_name", "date", "winner")
_INT_FIELDS = {
    "wins",
    "losses",
    "draws",
    "current_win_streak",
    "current_lose_streak",
    "longest_win_streak",
    "total_rounds_fought",
    "total_title_bouts",
    "wins_ko",
    "wins_sub",
    "decision_unanimous",
    "decision_split",
    "decision_majority",
}


def detect_layout(columns: Iterable[str]) -> str:
    available = set(columns)
    missing = {}
    for layout, mapping in COLUMN_MAPS.items():
        missing[layout] = [mapping[k] for k in _REQUIRED if mapping[k] not in available]
        if not missing[layout]:
            return layout
    raise ValueError(f"Missing required columns: {missing}")


def _number(value: Any, field: str, integer: bool = False) -> float | int | None:
    if value is None:
        return None
    try:
        number = float(value)
        if not math.isfinite(number) or (integer and not number.is_integer()):
            raise ValueError
        return int(number) if integer else number
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value for {field}: {value!r}") from exc


def bouts_from_frame(
    df: pd.DataFrame,
    *,
    layout: str | None = None,
    drop_draws: bool = True,
    require_odds: bool = False,
) -> list[Bout]:
    layout = detect_layout(df.columns) if layout is None else layout
    if layout not in COLUMN_MAPS:
        raise ValueError(f"Unknown column layout: {layout}")
    mapping = COLUMN_MAPS[layout]
    missing = [mapping[k] for k in _REQUIRED if mapping[k] not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    bouts: dict[str, Bout] = {}
    dropped = 0
    for _, row in df.iterrows():

        def get(key: str, row: pd.Series = row) -> Any:
            value = row.get(mapping[key])
            return None if value is None or pd.isna(value) else value

        winner = str(get("winner")).strip().lower()
        if winner not in ("red", "blue"):
            if drop_draws:
                dropped += 1
                continue
            raise ValueError("Draws and no-contests cannot be represented by a binary Bout outcome")

        def fighter(corner: str) -> FighterProfile:
            values: dict[str, Any] = {}
            for field in _FIELDS:
                value = get(f"{corner}_{field}")
                if field in ("name", "stance"):
                    values[field] = str(value).strip() if value is not None else None
                elif field == "odds_american":
                    try:
                        odds = float(value) if value is not None else None
                        if odds is not None:
                            american_to_decimal(odds)
                        values[field] = odds
                    except (ValueError, TypeError):
                        logger.warning("Invalid odds %r for %s; treating as missing", value, corner)
                        values[field] = None
                else:
                    values[field] = _number(value, field, field in _INT_FIELDS)
            if not values["name"]:
                raise ValueError(f"Missing required fighter name: {mapping[corner + '_name']}")
            decisions = [
                values.pop(f"decision_{kind}") for kind in ("unanimous", "split", "majority")
            ]
            values["wins_dec"] = (
                sum(v for v in decisions if v is not None)
                if any(v is not None for v in decisions)
                else None
            )
            return FighterProfile(**values)

        a, b = fighter("red"), fighter("blue")
        if require_odds and (a.odds_american is None or b.odds_american is None):
            continue
        raw_date = get("date")
        if raw_date is None:
            raise ValueError("Missing required date")
        try:
            bout_date = pd.to_datetime(raw_date, errors="raise").date()
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid date: {raw_date!r}") from exc
        title = get("title_bout")
        if title is None:
            title = False
        elif str(title).strip().lower() in ("true", "1", "1.0", "yes"):
            title = True
        elif str(title).strip().lower() in ("false", "0", "0.0", "no"):
            title = False
        else:
            raise ValueError(f"Invalid title_bout: {title!r}")
        rounds = _number(get("scheduled_rounds"), "scheduled_rounds", True)
        if rounds is not None and rounds <= 0:
            raise ValueError("scheduled_rounds must be positive")
        bout_id = make_bout_id(bout_date, a.name, b.name)
        bouts.setdefault(
            bout_id,
            Bout(
                bout_id,
                bout_date,
                str(get("weight_class") or ""),
                a,
                b,
                winner == "red",
                str(get("gender")) if get("gender") is not None else None,
                title,
                int(rounds) if rounds is not None else 3,
            ),
        )
    if dropped:
        logger.info("Dropped %d draws/no-contests", dropped)
    return sorted(bouts.values(), key=lambda b: (b.date, b.bout_id))


def load_bouts(path: str | Path, **kw: Any) -> list[Bout]:
    return bouts_from_frame(pd.read_csv(path), **kw)
