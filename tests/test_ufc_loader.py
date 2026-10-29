import pandas as pd
import pytest

from fightcal.data.ufc import COLUMN_MAPS, bouts_from_frame, detect_layout, load_bouts


def row(layout, **updates):
    logical = {"red_name": "Alpha", "blue_name": "Beta", "date": "2020-01-01", "winner": "Red"}
    logical.update(updates)
    return {COLUMN_MAPS[layout][key]: value for key, value in logical.items()}


@pytest.mark.parametrize("layout", ["legacy", "camel"])
def test_layouts(layout, tmp_path, caplog):
    record = row(
        layout,
        red_odds_american=-150,
        blue_odds_american=130,
        red_decision_unanimous=2,
        red_decision_split=1,
        red_decision_majority=0,
        title_bout="True",
        scheduled_rounds=5,
        red_age=30,
        red_stance="Orthodox",
    )
    frame = pd.DataFrame(
        [
            record,
            record,
            row(layout, winner="Draw"),
            row(layout, date="2019-01-01", winner="bLuE", red_name="Other", red_odds_american=50),
        ]
    )
    assert detect_layout(frame.columns) == layout
    bouts = bouts_from_frame(frame)
    assert len(bouts) == 2
    assert not bouts[0].a_won
    assert bouts[0].fighter_a.odds_american is None
    assert "Invalid odds" in caplog.text
    bout = bouts[1]
    assert bout.fighter_a.name == "Alpha" and bout.a_won
    assert bout.fighter_a.wins_dec == 3
    assert bout.fighter_a.height_cm is None
    assert bout.fighter_a.age == 30
    assert bout.title_bout and bout.scheduled_rounds == 5
    assert len(bouts_from_frame(frame, require_odds=True)) == 1
    path = tmp_path / "ufc.csv"
    frame.to_csv(path, index=False)
    assert load_bouts(path) == bouts
    with pytest.raises(ValueError, match="binary"):
        bouts_from_frame(frame, drop_draws=False)


@pytest.mark.parametrize("layout", ["legacy", "camel"])
def test_minimal_and_errors(layout):
    frame = pd.DataFrame([row(layout)])
    bout = bouts_from_frame(frame)[0]
    assert bout.fighter_b.wins_dec is None
    assert bout.scheduled_rounds == 3 and not bout.title_bout
    assert bout.gender is None
    with pytest.raises(ValueError, match=COLUMN_MAPS[layout]["red_name"]):
        bouts_from_frame(frame.drop(columns=COLUMN_MAPS[layout]["red_name"]), layout=layout)
    with pytest.raises(ValueError, match="Missing required columns"):
        detect_layout([])
    with pytest.raises(ValueError, match="Unknown"):
        bouts_from_frame(frame, layout="other")
    for update, message in [
        ({"red_name": None}, "name"),
        ({"date": None}, "date"),
        ({"date": "bad"}, "date"),
        ({"title_bout": "bad"}, "title"),
        ({"scheduled_rounds": 0}, "positive"),
        ({"red_wins": 1.5}, "numeric"),
        ({"red_age": "bad"}, "numeric"),
    ]:
        with pytest.raises(ValueError, match=message):
            bouts_from_frame(pd.DataFrame([row(layout, **update)]))
