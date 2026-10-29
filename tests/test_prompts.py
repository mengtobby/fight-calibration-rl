from datetime import date

import pytest

from fightcal.data.odds import american_to_implied
from fightcal.data.schema import Bout, FighterProfile
from fightcal.prompts import (
    PromptOptions,
    build_messages,
    extract_reasoning,
    parse_pick,
    parse_probability,
    prob_to_american,
    render_matchup,
)


def bout() -> Bout:
    return Bout(
        "test",
        date(2020, 1, 1),
        "Lightweight",
        FighterProfile(
            "Alice Example", wins=10, losses=2, draws=0, sig_str_pct=0.45, odds_american=-150
        ),
        FighterProfile("Bob Example", odds_american=130),
        True,
    )


def test_render() -> None:
    card = render_matchup(bout())
    assert "Alice Example" not in card and "Bob Example" not in card
    assert "Betting market" not in card
    assert "10-2-0" in card and "45%" in card and "n/a" in card
    assert "Alice Example" in render_matchup(bout(), PromptOptions(anonymize=False))
    assert "Betting market" in render_matchup(bout(), PromptOptions(include_odds=True))
    override = render_matchup(bout(), PromptOptions(include_odds=True, odds_override_prob_a=0.8))
    assert "P(A)=0.8" in override and "-400" in override and "400" in override
    assert "single letter" in render_matchup(bout(), PromptOptions(mode="pick"))
    assert [m["role"] for m in build_messages(bout())] == ["system", "user"]


@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_odds(p: float) -> None:
    assert american_to_implied(prob_to_american(p)) == pytest.approx(p)


@pytest.mark.parametrize("p", [0, 1, -0.1, float("nan")])
def test_bad_odds(p: float) -> None:
    with pytest.raises(ValueError):
        prob_to_american(p)
    with pytest.raises(ValueError):
        PromptOptions(odds_override_prob_a=p)


@pytest.mark.parametrize("answer", ["0.63", ".63", "63%", "P(A)=0.63", "0.63 (Fighter A)"])
def test_probability(answer: str) -> None:
    assert parse_probability(f"<answer>0.1</answer><answer>{answer}</answer>") == 0.63
    assert parse_probability(f"<answer>{answer}") == 0.63


@pytest.mark.parametrize(
    "text", ["0.6", "<answer>no", "<answer>1.1", "<answer>-1%", "<answer>0.4 or 0.6", "<answer>NaN"]
)
def test_invalid_probability(text: str) -> None:
    assert parse_probability(text) is None


def test_other_parsers() -> None:
    assert parse_pick("<answer>A</answer><answer>b") == "B"
    assert parse_pick("<answer>A or B</answer>") is None
    assert parse_pick("A") is None
    assert (
        extract_reasoning("<think>first\nthought</think><think>second</think>") == "first\nthought"
    )
    assert extract_reasoning("none") == ""
    with pytest.raises(ValueError):
        PromptOptions(mode="invalid")
