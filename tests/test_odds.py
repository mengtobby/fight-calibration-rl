from datetime import date

import pytest

from fightcal.data.odds import (
    american_to_decimal,
    american_to_implied,
    devig_pair,
    market_prob_a,
)
from fightcal.data.schema import Bout, FighterProfile


@pytest.mark.parametrize("odds", [-99, 0, 99, float("nan"), float("inf")])
def test_invalid_odds(odds):
    with pytest.raises(ValueError, match="American odds"):
        american_to_decimal(odds)


@pytest.mark.parametrize("method", ["multiplicative", "power"])
def test_devig(method):
    a, b = devig_pair(american_to_implied(-150), american_to_implied(130), method)
    assert a + b == pytest.approx(1)
    assert a > 0.5
    assert american_to_decimal(100) == 2
    assert american_to_decimal(-100) == 2
    assert devig_pair(0.2, 0.3, method)[0] < 0.5
    bout = Bout(
        "id",
        date(2020, 1, 1),
        "",
        FighterProfile("A", odds_american=-150),
        FighterProfile("B", odds_american=130),
        True,
    )
    assert market_prob_a(bout, method) == a
    assert market_prob_a(bout.swapped(), method) == pytest.approx(b)
    assert (
        market_prob_a(Bout("x", bout.date, "", FighterProfile("A"), bout.fighter_b, True)) is None
    )


@pytest.mark.parametrize(
    "a,b,method",
    [(0, 0.5, "power"), (1, 0.5, "power"), (float("nan"), 0.5, "power"), (0.5, 0.5, "bad")],
)
def test_bad_devig(a, b, method):
    with pytest.raises(ValueError):
        devig_pair(a, b, method)
