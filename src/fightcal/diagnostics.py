"""Array-only checks for probability collapse, market copying and reasoning use."""

import re
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit, logit, xlogy
from scipy.stats import rankdata

from fightcal.metrics import _pair, _probabilities


def _aligned(*arrays: ArrayLike) -> list[NDArray[np.float64]]:
    values = [_probabilities(a) for a in arrays]
    if any(a.shape != values[0].shape for a in values):
        raise ValueError("arrays must have equal lengths")
    return values


def _correlation(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _logits(p: NDArray[np.float64]) -> NDArray[np.float64]:
    return logit(np.clip(p, 1e-4, 1 - 1e-4))


@dataclass(frozen=True)
class CollapseStats:
    frac_near_half: float
    mean_abs_dev: float
    std: float
    mean_entropy_bits: float


def collapse_stats(p: ArrayLike, band: float = 0.05) -> CollapseStats:
    a = _probabilities(p)
    if not np.isfinite(band) or not 0 <= band <= 0.5:
        raise ValueError("band must be finite and in [0, 0.5]")
    dev = abs(a - 0.5)
    return CollapseStats(
        float(np.mean(dev <= band + 1e-15)),
        float(dev.mean()),
        float(a.std()),
        float(np.mean(-xlogy(a, a) - xlogy(1 - a, 1 - a)) / np.log(2)),
    )


@dataclass(frozen=True)
class MarketCopyStats:
    pearson: float
    spearman: float
    mean_abs_diff: float
    frac_within_002: float
    logit_slope: float
    logit_intercept: float


def market_copy_stats(p_model: ArrayLike, p_market: ArrayLike) -> MarketCopyStats:
    a, b = _aligned(p_model, p_market)
    x, z = _logits(b), _logits(a)
    if np.ptp(x) == 0:
        slope, intercept = float("nan"), float("nan")
    else:
        slope = float(np.dot(x - x.mean(), z - z.mean()) / np.sum((x - x.mean()) ** 2))
        intercept = float(z.mean() - slope * x.mean())
    return MarketCopyStats(
        _correlation(a, b),
        _correlation(rankdata(a), rankdata(b)),
        float(np.mean(abs(a - b))),
        float(np.mean(abs(a - b) <= 0.02 + 1e-15)),
        slope,
        intercept,
    )


@dataclass(frozen=True)
class OddsSensitivity:
    mean_abs_dp: float
    slope: float
    frac_follow: float


def odds_sensitivity(
    p_base: ArrayLike,
    p_perturbed: ArrayLike,
    market_shown_base: ArrayLike,
    market_shown_perturbed: ArrayLike,
) -> OddsSensitivity:
    a, b, c, d = _aligned(p_base, p_perturbed, market_shown_base, market_shown_perturbed)
    dp, dm = b - a, d - c
    changed = dm != 0
    denom = float(np.dot(dm, dm))
    return OddsSensitivity(
        float(np.mean(abs(dp))),
        float(np.dot(dm, dp) / denom) if denom else float("nan"),
        float(np.mean(np.sign(dp[changed]) == np.sign(dm[changed])))
        if np.any(changed)
        else float("nan"),
    )


def symmetry_gap(p_ab: ArrayLike, p_ba: ArrayLike) -> float:
    a, b = _aligned(p_ab, p_ba)
    return float(np.mean(abs(a + b - 1)))


@dataclass(frozen=True)
class ReasoningAblation:
    mean_abs_dp: float
    pearson: float


def reasoning_ablation(p_own: ArrayLike, p_swapped: ArrayLike) -> ReasoningAblation:
    a, b = _aligned(p_own, p_swapped)
    return ReasoningAblation(float(np.mean(abs(a - b))), _correlation(a, b))


def reasoning_lean(text: str) -> int:
    """Scan the last 400 characters for fighter mentions near positive cue words.

    A cue within six words in the same clause votes for the closest fighter mention
    (ties ignored). Negated cues within three preceding words are ignored. Majority
    wins; tied or absent evidence returns zero. This is a lexical proxy, not NLP.
    """
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    votes: list[int] = []
    for clause in re.split(r"[.!?;\n]", text[-400:].lower()):
        tokens = re.findall(r"[a-z]+", clause)
        mentions = [
            (i, 1 if tokens[i + 1] == "a" else -1)
            for i in range(len(tokens) - 1)
            if tokens[i] == "fighter" and tokens[i + 1] in ("a", "b")
        ]
        for i, word in enumerate(tokens):
            if not re.fullmatch(r"favors?|favou?red|edges?|advantages?|wins?|likely|picks?", word):
                continue
            if set(tokens[max(0, i - 3) : i]) & {"not", "no", "never", "unlikely"}:
                continue
            distances = [(min(abs(i - j), abs(i - j - 1)), side) for j, side in mentions]
            if distances:
                nearest = min(d for d, _ in distances)
                sides = {side for d, side in distances if d == nearest}
                if nearest <= 6 and len(sides) == 1:
                    votes.append(sides.pop())
    return int(np.sign(sum(votes)))


def reasoning_consistency(texts: Iterable[str], p: ArrayLike) -> float:
    a = _probabilities(p)
    if isinstance(texts, str):
        raise ValueError("texts must be a sequence of strings")
    leans = np.asarray([reasoning_lean(t) for t in texts])
    if leans.shape != a.shape:
        raise ValueError("texts and probabilities must have equal lengths")
    mask = leans != 0
    return float(np.mean(leans[mask] == np.sign(a[mask] - 0.5))) if mask.any() else float("nan")


@dataclass(frozen=True)
class EncompassingResult:
    coef_market: float
    coef_model: float
    se_market: float
    se_model: float
    n: int


def encompassing_test(p_model: ArrayLike, p_market: ArrayLike, y: ArrayLike) -> EncompassingResult:
    """Unregularized logistic IRLS; singular or separated fits raise ValueError."""
    a, b = _aligned(p_model, p_market)
    _, outcomes = _pair(a, y)
    x = np.column_stack((np.ones(a.size), _logits(b), _logits(a)))
    beta = np.zeros(3)
    try:
        if np.linalg.matrix_rank(x) < 3:
            raise ValueError("encompassing design is rank deficient")
        for _ in range(50):
            fitted = expit(x @ beta)
            fisher = x.T @ ((fitted * (1 - fitted))[:, None] * x)
            step = np.linalg.solve(fisher, x.T @ (outcomes - fitted))
            beta = beta + step
            if not np.all(np.isfinite(beta)):
                break
            if np.max(abs(step)) < 1e-8:
                fitted = expit(x @ beta)
                fisher = x.T @ ((fitted * (1 - fitted))[:, None] * x)
                se = np.sqrt(np.diag(np.linalg.inv(fisher)))
                if not np.all(np.isfinite(se)):
                    break
                return EncompassingResult(
                    float(beta[1]), float(beta[2]), float(se[1]), float(se[2]), a.size
                )
    except np.linalg.LinAlgError as exc:
        raise ValueError("encompassing fit has singular Fisher information") from exc
    raise ValueError("encompassing logistic regression did not converge in 50 iterations")
