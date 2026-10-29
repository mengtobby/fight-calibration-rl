"""Pre-fight numeric differences and directional stance matchups."""

from collections.abc import Iterable

import numpy as np

from fightcal.data.schema import Bout

_NUMERIC = (
    "age",
    "height_cm",
    "reach_cm",
    "wins",
    "losses",
    "current_win_streak",
    "current_lose_streak",
    "total_rounds_fought",
    "wins_ko",
    "wins_sub",
    "wins_dec",
    "sig_str_landed_pm",
    "sig_str_pct",
    "td_landed_per15",
    "td_pct",
    "sub_att_per15",
)
FEATURE_NAMES = (*_NUMERIC, "a_southpaw_b_orthodox", "a_orthodox_b_southpaw")


def bout_features(bout: Bout) -> np.ndarray:
    a, b = bout.fighter_a, bout.fighter_b
    values = [
        float(getattr(a, name) - getattr(b, name))
        if getattr(a, name) is not None and getattr(b, name) is not None
        else np.nan
        for name in _NUMERIC
    ]
    sa, sb = (s.strip().lower() if s is not None else None for s in (a.stance, b.stance))
    values.extend(
        [np.nan, np.nan]
        if sa is None or sb is None
        else [
            float(sa == "southpaw" and sb == "orthodox"),
            float(sa == "orthodox" and sb == "southpaw"),
        ]
    )
    return np.asarray(values, dtype=float)


def feature_matrix(bouts: Iterable[Bout]) -> np.ndarray:
    rows = [bout_features(b) for b in bouts]
    return np.vstack(rows) if rows else np.empty((0, len(FEATURE_NAMES)), dtype=float)
