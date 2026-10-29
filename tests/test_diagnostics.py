import numpy as np
import pytest
from scipy.special import expit

from fightcal.diagnostics import (
    collapse_stats,
    encompassing_test,
    market_copy_stats,
    odds_sensitivity,
    reasoning_ablation,
    reasoning_consistency,
    reasoning_lean,
    symmetry_gap,
)


def test_collapse() -> None:
    c = collapse_stats([0.5] * 5)
    assert c.frac_near_half == 1
    assert c.mean_abs_dev == c.std == 0
    assert c.mean_entropy_bits == 1
    assert collapse_stats([0, 1]).mean_entropy_bits == 0
    assert collapse_stats([0.45, 0.55]).frac_near_half == 1
    with pytest.raises(ValueError):
        collapse_stats([0.5], band=-1)


def test_market_copy() -> None:
    p = [0, 0.2, 0.5, 0.9, 1]
    c = market_copy_stats(p, p)
    assert c.pearson == pytest.approx(1)
    assert c.spearman == pytest.approx(1)
    assert c.logit_slope == pytest.approx(1)
    assert c.logit_intercept == pytest.approx(0)
    assert c.mean_abs_diff == 0
    assert c.frac_within_002 == 1
    constant = market_copy_stats([0.5] * 3, [0.5] * 3)
    assert np.isnan(constant.pearson)
    assert np.isnan(constant.logit_slope)


def test_counterfactuals() -> None:
    o = odds_sensitivity([0.2, 0.6], [0.3, 0.4], [0.2, 0.6], [0.3, 0.4])
    assert o.mean_abs_dp == pytest.approx(0.15)
    assert o.slope == pytest.approx(1)
    assert o.frac_follow == 1
    o = odds_sensitivity([0.2], [0.3], [0.5], [0.5])
    assert np.isnan(o.slope) and np.isnan(o.frac_follow)
    assert symmetry_gap([0.2, 0.9], [0.8, 0.1]) == pytest.approx(0)
    r = reasoning_ablation([0.2, 0.8], [0.8, 0.2])
    assert r.mean_abs_dp == pytest.approx(0.6)
    assert r.pearson == pytest.approx(-1)


@pytest.mark.parametrize(
    "text,lean",
    [
        ("I favor Fighter A.", 1),
        ("Fighter B has the edge.", -1),
        ("Fighter A is likely to win.", 1),
        ("No clear conclusion.", 0),
        ("Fighter A has no advantage.", 0),
        ("Fighter A has an edge. Fighter B has an edge.", 0),
        ("I favor Fighter A." + " " * 401, 0),
    ],
)
def test_reasoning_heuristic(text: str, lean: int) -> None:
    assert reasoning_lean(text) == lean


def test_consistency() -> None:
    assert (
        reasoning_consistency(["Favor Fighter A", "Pick Fighter B", "unclear"], [0.8, 0.7, 0.5])
        == 0.5
    )
    assert np.isnan(reasoning_consistency(["unclear"], [0.5]))
    assert reasoning_consistency(["Pick Fighter A"], [0.5]) == 0
    for texts in [["a", "b"], "text", [None]]:
        with pytest.raises(ValueError):
            reasoning_consistency(texts, [0.5])


def test_encompassing_synthetic() -> None:
    rng = np.random.default_rng(12)
    market = expit(rng.normal(size=20000))
    model = expit(rng.normal(size=20000))
    y = rng.binomial(1, market)
    fit = encompassing_test(model, market, y)
    assert fit.coef_market == pytest.approx(1, abs=0.08)
    assert fit.coef_model == pytest.approx(0, abs=0.06)
    assert fit.se_market > 0 and fit.se_model > 0
    assert fit.n == len(y)
    with pytest.raises(ValueError, match="rank"):
        encompassing_test(market, market, y)
    with pytest.raises(ValueError):
        encompassing_test(model, market, np.zeros(len(y)))


@pytest.mark.parametrize("p,q", [([0], [0, 1]), ([np.nan], [0]), ([1.2], [0])])
def test_invalid(p: list, q: list) -> None:
    with pytest.raises(ValueError):
        market_copy_stats(p, q)
