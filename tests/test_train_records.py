"""Conversational training records and append-only probability traces."""

import json
from datetime import date

import pytest

from fightcal.data.schema import Bout, FighterProfile
from fightcal.prompts import PromptOptions
from fightcal.train.records import ProbabilityLogger, grpo_records, select_subset, sft_records


def bouts():
    return [
        Bout(
            str(i),
            date(2020, 1, 1),
            "Lightweight",
            FighterProfile("Secret Alice", odds_american=-150),
            FighterProfile("Secret Bob", odds_american=130),
            bool(i % 2),
        )
        for i in range(10)
    ]


def test_records():
    source = bouts()
    opts = PromptOptions(mode="pick")
    grpo = grpo_records(source, opts)
    sft = sft_records(source, opts)
    assert opts.mode == "pick"
    for b, g, s in zip(source, grpo, sft, strict=True):
        assert set(g) == {"prompt", "a_won", "bout_id", "market_p_a"}
        assert g["a_won"] == b.a_won and g["bout_id"] == b.bout_id
        assert 0 < g["market_p_a"] < 1
        assert [m["role"] for m in g["prompt"]] == ["system", "user"]
        assert "Secret" not in str(g) + str(s)
        assert "<think>" in g["prompt"][-1]["content"]
        assert s["completion"] == [
            {"role": "assistant", "content": "<answer>" + ("A" if b.a_won else "B") + "</answer>"}
        ]
    assert "Secret Alice" in str(grpo_records(source, PromptOptions(anonymize=False)))


def test_subset():
    source = bouts()
    assert select_subset(source, None, 42) == source
    assert select_subset(source, 100, 42) == source
    assert select_subset(source, None, 42) is not source
    assert select_subset(source, 3, 42) == select_subset(source, 3, 42)
    assert len(select_subset(source, 3, 42)) == 3
    for n, seed in ((0, 1), (-1, 1), (True, 1), (1, -1)):
        with pytest.raises(ValueError):
            select_subset(source, n, seed)


def test_logger(tmp_path):
    path = tmp_path / "nested" / "trace.jsonl"
    logger = ProbabilityLogger(path)
    parsed, labels = [0.7, None], [True, False]
    logger(parsed, labels)
    logger.set_step(12)
    logger([0], [True])
    assert parsed == [0.7, None] and labels == [True, False]
    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        {"step": 0, "p": 0.7, "a_won": True},
        {"step": 0, "p": None, "a_won": False},
        {"step": 12, "p": 0, "a_won": True},
    ]
    for p, y in (([1], []), ([float("nan")], [True]), ([2], [True]), ([0.5], [2])):
        with pytest.raises(ValueError):
            logger(p, y)
    with pytest.raises(ValueError):
        logger.set_step(-1)
