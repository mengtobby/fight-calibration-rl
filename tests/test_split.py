from datetime import date, timedelta

import pytest

from fightcal.data.schema import Bout, FighterProfile
from fightcal.data.split import a_won_rate, randomize_corners, temporal_split


def test_splits_and_corners():
    bouts = [
        Bout(
            str(i),
            date(2020, 1, 1) + timedelta(days=i),
            "",
            FighterProfile("A"),
            FighterProfile("B"),
            True,
        )
        for i in range(1000)
    ]
    randomized = randomize_corners(bouts, 42)
    assert randomized == randomize_corners(bouts, 42)
    assert randomized == list(reversed(randomize_corners(reversed(bouts), 42)))
    assert 0.4 < a_won_rate(randomized) < 0.6
    for bout in randomized:
        assert (bout.fighter_a if bout.a_won else bout.fighter_b).name == "A"
    split = temporal_split(bouts, val_start=bouts[600].date, test_start=bouts[800].date)
    assert (len(split.train), len(split.val), len(split.test)) == (600, 200, 200)
    assert a_won_rate(split.train) == 1
    with pytest.raises(ValueError):
        temporal_split(bouts, val_start=bouts[800].date, test_start=bouts[600].date)
    with pytest.raises(ValueError):
        a_won_rate([])
