from datetime import date

import pytest

from fightcal.data.io import (
    bout_from_dict,
    bout_to_dict,
    load_bouts_jsonl,
    save_bouts_jsonl,
)
from fightcal.data.schema import Bout, FighterProfile, make_bout_id


def test_round_trip(tmp_path):
    day = date(2020, 1, 2)
    a = FighterProfile("A", age=25, wins=3, odds_american=-150)
    b = FighterProfile("B", stance="Southpaw", odds_american=130)
    bout = Bout(make_bout_id(day, a.name, b.name), day, "Lightweight", a, b, True)
    assert make_bout_id(day, "B", "a") == bout.bout_id
    assert len(bout.bout_id) == 16
    assert bout.swapped().swapped() == bout
    assert bout.swapped().bout_id == bout.bout_id
    assert bout_from_dict(bout_to_dict(bout)) == bout
    path = tmp_path / "nested" / "bouts.jsonl"
    save_bouts_jsonl(path, [bout, bout.swapped()])
    assert load_bouts_jsonl(path) == [bout, bout.swapped()]
    with pytest.raises(ValueError, match="Invalid bout"):
        bout_from_dict({"date": "bad"})
