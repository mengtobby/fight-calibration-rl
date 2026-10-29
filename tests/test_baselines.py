from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from fightcal.baselines import (
    base_rate_baseline,
    constant_baseline,
    logistic_baseline,
    market_baseline,
)
from fightcal.data.schema import Bout, FighterProfile
from fightcal.features import FEATURE_NAMES, bout_features, feature_matrix
from fightcal.metrics import brier


def bouts(n=120):
    return [
        Bout(
            str(i),
            date(2020, 1, 1),
            "lightweight",
            FighterProfile("A", reach_cm=170 + i % 20, stance="Southpaw"),
            FighterProfile("B", reach_cm=180, stance="Orthodox"),
            i % 20 > 10,
        )
        for i in range(n)
    ]


def test_features():
    bout = bouts()[0]
    values = bout_features(bout)
    assert len(values) == len(FEATURE_NAMES) == 18
    assert values[2] == -10
    assert np.isnan(values[0])
    assert list(values[-2:]) == [1, 0]
    assert feature_matrix([]).shape == (0, 18)
    missing = replace(bout, fighter_a=FighterProfile("a"))
    assert np.isnan(bout_features(missing)[-1])


def test_baselines():
    train, test = bouts(), bouts(40)
    y = [b.a_won for b in test]
    p = [r.p_a for r in logistic_baseline(train, test)]
    assert brier(p, y) < brier([0.5] * len(test), y)
    assert logistic_baseline(train, []) == []
    assert base_rate_baseline(train, test)[0].p_a == 0.45
    assert market_baseline(test)[0].p_a is None
    assert constant_baseline(test)[0].p_a == 0.5
    with_odds = replace(
        test[0],
        fighter_a=replace(test[0].fighter_a, odds_american=-150),
        fighter_b=replace(test[0].fighter_b, odds_american=150),
    )
    assert market_baseline([with_odds])[0].p_a == pytest.approx(0.6)
    with pytest.raises(ValueError):
        base_rate_baseline([], test)
    with pytest.raises(ValueError):
        logistic_baseline(train[:1], test)
    with pytest.raises(ValueError):
        logistic_baseline(train, test, C=0)
    with pytest.raises(ValueError):
        constant_baseline(test, p=2)
