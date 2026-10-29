import numpy as np
import pytest

from fightcal.metrics import (
    accuracy,
    brier,
    ece,
    log_loss,
    murphy_decomposition,
    paired_bootstrap,
    reliability_table,
    sharpness,
    summarize,
)


def test_half_and_perfect() -> None:
    assert brier([0.5, 0.5], [0, 1]) == 0.25
    assert log_loss([0.5, 0.5], [0, 1]) == pytest.approx(np.log(2))
    assert accuracy([0.5, 0.5], [0, 1]) == 0.5
    assert accuracy([0, 1], [0, 1]) == 1
    assert sharpness([0.5]) == 0
    assert np.isfinite(log_loss([0, 1], [1, 0]))
    assert set(summarize([0, 1], [0, 1])) == {
        "n",
        "brier",
        "log_loss",
        "accuracy",
        "ece",
        "ece_quantile",
        "sharpness",
        "reliability",
        "resolution",
        "uncertainty",
    }


def test_bins_and_decomposition() -> None:
    p = np.repeat([0.1, 0.5, 0.9], 10)
    y = np.array([1] + [0] * 9 + [1] * 5 + [0] * 5 + [1] * 9 + [0])
    d = murphy_decomposition(p, y)
    assert d.reliability - d.resolution + d.uncertainty == pytest.approx(brier(p, y))
    assert len(reliability_table([0.5] * 4, [0, 1, 0, 1], strategy="quantile")) == 1
    bins = reliability_table([0, 0.5, 1], [0, 0, 1], n_bins=2)
    assert [b.count for b in bins] == [1, 2]


def test_synthetic_calibration() -> None:
    rng = np.random.default_rng(42)
    p = rng.uniform(0.1, 0.9, 30000)
    y = rng.binomial(1, p)
    assert ece(p, y) < 0.03
    assert ece(p, y, strategy="quantile") < 0.03
    assert ece(np.where(p > 0.5, 0.99, 0.01), y) > ece(p, y)


@pytest.mark.parametrize("metric", ["brier", "log_loss", "accuracy"])
def test_bootstrap(metric: str) -> None:
    result = paired_bootstrap([0, 1], [1, 0], [0, 1], metric=metric, n_boot=30)
    assert result == paired_bootstrap([0, 1], [1, 0], [0, 1], metric=metric, n_boot=30)
    assert result.ci_low <= result.mean_diff <= result.ci_high
    assert result.p_value == 0
    tied = paired_bootstrap([0.5], [0.5], [1], n_boot=5)
    assert tied.p_value == 1


@pytest.mark.parametrize(
    "p,y",
    [
        ([], []),
        ([0], [0, 1]),
        ([1.1], [1]),
        ([np.nan], [1]),
        ([0.5], [0.2]),
        ([[0]], [0]),
        (["bad"], [0]),
    ],
)
def test_invalid_arrays(p: list, y: list) -> None:
    with pytest.raises(ValueError):
        brier(p, y)


def test_invalid_options() -> None:
    for bins in [0, 1.5, True]:
        with pytest.raises(ValueError):
            reliability_table([0], [0], n_bins=bins)
    with pytest.raises(ValueError):
        ece([0], [0], strategy="bad")
    with pytest.raises(ValueError):
        log_loss([0], [0], eps=0)
    with pytest.raises(ValueError):
        paired_bootstrap([0], [0], [0], metric="bad")
    with pytest.raises(ValueError):
        paired_bootstrap([0], [0], [0], n_boot=0)
