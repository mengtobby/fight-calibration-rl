from dataclasses import replace

import pytest

from fightcal.predictions import (
    PredictionRecord,
    aggregate,
    parse_rate,
    read_predictions,
    write_predictions,
)


def test_roundtrip_and_aggregation(tmp_path):
    record = PredictionRecord("bout", "model", "base", 0, 0.6, True, 0.7)
    rows = [record, replace(record, sample_idx=1, p_a=0.8), replace(record, sample_idx=2, p_a=None)]
    path = tmp_path / "predictions.jsonl"
    write_predictions(path, rows[:1])
    write_predictions(path, rows[1:], append=True)
    assert read_predictions(path) == rows
    assert aggregate(rows)[0].p_a == pytest.approx(0.7)
    assert aggregate(rows)[0].sample_idx == -1
    assert parse_rate(rows) == 2 / 3
    assert parse_rate([]) == 0
    assert aggregate([rows[-1]])[0].p_a is None
    with pytest.raises(ValueError, match="Inconsistent"):
        aggregate([record, replace(record, a_won=False)])


@pytest.mark.parametrize("line", ["no json", "{}", "[]", '{"p_a": NaN}', ""])
def test_invalid_json(tmp_path, line):
    path = tmp_path / "bad.jsonl"
    path.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        read_predictions(path)


@pytest.mark.parametrize(
    "changes",
    [
        {"bout_id": ""},
        {"arm": 3},
        {"variant": "bad"},
        {"sample_idx": True},
        {"sample_idx": -2},
        {"a_won": 1},
        {"p_a": float("nan")},
        {"p_a": 2},
        {"market_p_a": True},
        {"completion": 1},
        {"checkpoint": 2},
    ],
)
def test_validation(changes):
    record = PredictionRecord("b", "m", "base", -1, 0.5, True, None)
    with pytest.raises(ValueError):
        replace(record, **changes)
