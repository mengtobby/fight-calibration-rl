import math

import numpy as np
import pytest

from fightcal.rewards import (
    PENALTIES,
    REWARD_FNS,
    brier_reward,
    completion_text,
    correctness_reward,
    log_reward,
    make_trl_reward,
)


@pytest.mark.parametrize("kind", REWARD_FNS)
def test_penalties(kind: str) -> None:
    reward = REWARD_FNS[kind]
    assert reward(None, True) == PENALTIES[kind]
    assert reward(None, True) < min(reward(p, y) for p in [0, 0.5, 1] for y in [True, False])


@pytest.mark.parametrize("reward", [brier_reward, log_reward])
@pytest.mark.parametrize("q", [0.2, 0.5, 0.8])
def test_properness(reward, q: float) -> None:
    grid = np.linspace(0, 1, 101)
    scores = [q * reward(p, True) + (1 - q) * reward(p, False) for p in grid]
    assert grid[np.argmax(scores)] == pytest.approx(q)


def test_correctness() -> None:
    assert correctness_reward(0.51, True) == correctness_reward(1, True) == 1
    assert correctness_reward(0, True) == correctness_reward(0.49, True) == 0
    assert correctness_reward(0.5, False) == 0.5
    assert brier_reward(0.8, True) == pytest.approx(0.96)
    assert log_reward(0, True, clip=0.1) == pytest.approx(math.log(0.1))
    assert log_reward(None, False, clip=0.1) < math.log(0.1)


def test_adapter() -> None:
    batches = []
    reward = make_trl_reward(
        "brier", label_key="labels", on_batch=lambda p, y: batches.append((p, y))
    )
    messages = [
        {"role": "user", "content": "ignored"},
        {"role": "assistant", "content": "<answer>0.8</answer>"},
    ]
    assert reward([messages, "invalid"], labels=[True, False]) == pytest.approx([0.96, -0.5])
    assert batches == [([0.8, None], [True, False])]
    assert reward.__name__ == "brier_reward"
    assert completion_text("text") == "text"
    for kwargs in [{}, {"labels": []}, {"labels": 1}, {"labels": [2]}]:
        with pytest.raises(ValueError):
            reward(["x"], **kwargs)
    with pytest.raises(ValueError):
        make_trl_reward("unknown")
    with pytest.raises(ValueError):
        completion_text([{"content": 1}])


@pytest.mark.parametrize("reward", [brier_reward, log_reward, correctness_reward])
def test_invalid_values(reward) -> None:
    for p in [-0.1, 1.1, float("nan")]:
        with pytest.raises(ValueError):
            reward(p, True)
    with pytest.raises(ValueError):
        reward(0.5, 2)


@pytest.mark.parametrize("clip", [0, 0.5, float("nan")])
def test_invalid_clip(clip: float) -> None:
    with pytest.raises(ValueError):
        log_reward(0.5, True, clip)
