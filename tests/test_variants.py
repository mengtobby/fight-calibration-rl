"""Pure inference intervention contracts."""

from datetime import date

import pytest

from fightcal.data.schema import Bout, FighterProfile
from fightcal.variants import (
    build_forced_answer_prefix,
    pair_for_reasoning_swap,
    perturb_market,
    plan_variants,
    to_original_perspective,
)


def bouts() -> list[Bout]:
    return [
        Bout(
            str(i),
            date(2020, 1, 1),
            "Lightweight",
            FighterProfile("a", odds_american=-150),
            FighterProfile("b", odds_american=130),
            True,
        )
        for i in range(3)
    ]


@pytest.mark.parametrize("p", [0, 0.05, 0.5, 0.95, 1])
def test_perturb(p: float) -> None:
    result = perturb_market(p, 42, "bout")
    assert result == perturb_market(p, 42, "bout")
    assert 0.05 <= result <= 0.95
    assert result != p


@pytest.mark.parametrize("p,shift", [(float("nan"), 0.1), (-0.1, 0.1), (0.5, 0)])
def test_invalid_perturb(p: float, shift: float) -> None:
    with pytest.raises(ValueError):
        perturb_market(p, 0, "x", shift)


def test_pairs() -> None:
    ids = ["a", "b", "c", "d"]
    pairs = pair_for_reasoning_swap(ids, 4)
    assert pairs == pair_for_reasoning_swap(reversed(ids), 4)
    assert set(pairs.values()) == set(ids)
    assert all(k != v for k, v in pairs.items())
    for invalid in ([], ["a"], ["a", "a"]):
        with pytest.raises(ValueError):
            pair_for_reasoning_swap(invalid, 0)


def test_plan() -> None:
    records = bouts()
    base = {b.bout_id: f"<think>reason {b.bout_id}</think><answer>.5</answer>" for b in records}
    variants = ["base", "odds_in_prompt", "odds_perturbed", "swapped", "reasoning_swap"]
    plan = plan_variants(records, variants, base_completions=base)
    assert len(plan) == 15
    pairs = pair_for_reasoning_swap(base, 0)
    for request in plan:
        assert request.swapped == (request.variant == "swapped")
        assert request.opts.include_odds == (request.variant in variants[1:3])
        if request.variant == "odds_perturbed":
            assert request.opts.odds_override_prob_a is not None
        if request.variant == "reasoning_swap":
            assert request.forced_prefix == build_forced_answer_prefix(
                f"reason {pairs[request.bout_id]}"
            )
        else:
            assert request.forced_prefix is None
    assert records[0].fighter_a.name == "a"
    for names, completions in [
        (["reasoning_swap"], None),
        (["reasoning_swap"], {}),
        (["unknown"], base),
    ]:
        with pytest.raises(ValueError):
            plan_variants(records, names, base_completions=completions)
    missing = Bout("missing", date(2020, 1, 1), "", FighterProfile("a"), FighterProfile("b"), True)
    with pytest.raises(ValueError, match="requires odds"):
        plan_variants([missing], ["odds_perturbed"])


def test_perspective() -> None:
    assert to_original_perspective(None, True) is None
    assert to_original_perspective(0.25, True) == 0.75
    assert to_original_perspective(0.25, False) == 0.25
    with pytest.raises(ValueError):
        to_original_perspective(2, False)
    assert build_forced_answer_prefix("abc") == "<think>abc</think>\n<answer>"


def test_odds_arm_diagnostics_keep_market_condition() -> None:
    from fightcal.data.odds import market_prob_a
    from fightcal.prompts import render_matchup

    records = bouts()
    base = {b.bout_id: "<think>reason</think><answer>.5</answer>" for b in records}
    requests = plan_variants(
        records,
        ["base", "swapped", "reasoning_swap"],
        base_completions=base,
        base_include_odds=True,
    )
    for request in requests:
        assert request.opts.include_odds
        assert request.opts.odds_override_prob_a is None
        bout = next(b for b in records if b.bout_id == request.bout_id)
        shown_bout = bout.swapped() if request.swapped else bout
        card = render_matchup(shown_bout, request.opts)
        assert f"de-vigged P(A)={market_prob_a(shown_bout):g}" in card


def test_pick_context_uses_joint_answer_tokenization() -> None:
    from fightcal.infer import _pick_token_context

    class FusedTokenizer:
        def apply_chat_template(self, messages: list, **kwargs: object) -> str:
            return "chat:"

        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert not add_special_tokens
            # The letter is fused with '>'; standalone A/B ids are never requested.
            assert text in ("chat:<answer>A</answer>", "chat:<answer>B</answer>")
            return [10, 11, 12, 101 if ">A" in text else 102, 13, 14]

    context, ids = _pick_token_context(FusedTokenizer(), [])
    assert context == [10, 11, 12]
    assert ids == (101, 102)
