"""Array-only forecast scores and calibration estimates."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _probabilities(p: ArrayLike) -> NDArray[np.float64]:
    try:
        a = np.asarray(p, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("probabilities must be numeric") from exc
    if a.ndim != 1 or not a.size or not np.all(np.isfinite(a)):
        raise ValueError("probabilities must be a non-empty, finite one-dimensional array")
    if np.any((a < 0) | (a > 1)):
        raise ValueError("probabilities must be in [0, 1]")
    return a


def _pair(p: ArrayLike, y: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    a, b = _probabilities(p), _probabilities(y)
    if a.shape != b.shape:
        raise ValueError("arrays must have equal lengths")
    if np.any((b != 0) & (b != 1)):
        raise ValueError("outcomes must be binary (0 or 1)")
    return a, b


def brier(p: ArrayLike, y: ArrayLike) -> float:
    a, b = _pair(p, y)
    return float(np.mean((a - b) ** 2))


def log_loss(p: ArrayLike, y: ArrayLike, eps: float = 1e-6) -> float:
    a, b = _pair(p, y)
    if not np.isfinite(eps) or not 0 < eps < 0.5:
        raise ValueError("eps must be finite and in (0, 0.5)")
    a = np.clip(a, eps, 1 - eps)
    return float(-np.mean(b * np.log(a) + (1 - b) * np.log1p(-a)))


def accuracy(p: ArrayLike, y: ArrayLike) -> float:
    a, b = _pair(p, y)
    return float(np.mean(np.where(a == 0.5, 0.5, (a > 0.5) == b)))


@dataclass(frozen=True)
class ReliabilityBin:
    lo: float
    hi: float
    count: int
    mean_p: float
    frac_pos: float


def reliability_table(
    p: ArrayLike,
    y: ArrayLike,
    n_bins: int = 10,
    strategy: Literal["uniform", "quantile"] = "uniform",
) -> list[ReliabilityBin]:
    """Return occupied bins, left-closed/right-open except the final closed bin.

    Quantile edges are deduplicated, so tied predictions stay together.
    """
    a, b = _pair(p, y)
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    if strategy not in ("uniform", "quantile"):
        raise ValueError("strategy must be 'uniform' or 'quantile'")
    edges = (
        np.linspace(0, 1, n_bins + 1)
        if strategy == "uniform"
        else np.unique(np.quantile(a, np.linspace(0, 1, n_bins + 1)))
    )
    if edges.size == 1:
        edges = np.repeat(edges, 2)
    ids = np.searchsorted(edges[1:-1], a, side="right")
    return [
        ReliabilityBin(
            float(edges[i]),
            float(edges[i + 1]),
            int(np.sum(ids == i)),
            float(a[ids == i].mean()),
            float(b[ids == i].mean()),
        )
        for i in range(len(edges) - 1)
        if np.any(ids == i)
    ]


def ece(
    p: ArrayLike,
    y: ArrayLike,
    n_bins: int = 10,
    strategy: Literal["uniform", "quantile"] = "uniform",
) -> float:
    bins = reliability_table(p, y, n_bins, strategy)
    return sum(v.count * abs(v.mean_p - v.frac_pos) for v in bins) / sum(v.count for v in bins)


@dataclass(frozen=True)
class BrierDecomposition:
    reliability: float
    resolution: float
    uncertainty: float


def murphy_decomposition(p: ArrayLike, y: ArrayLike, n_bins: int = 10) -> BrierDecomposition:
    """Binned Murphy terms; rel - res + uncertainty only approximates raw Brier.

    The identity is exact for forecasts constant within each occupied bin.
    """
    a, b = _pair(p, y)
    bins = reliability_table(a, b, n_bins)
    rate = float(b.mean())
    return BrierDecomposition(
        sum(v.count * (v.mean_p - v.frac_pos) ** 2 for v in bins) / a.size,
        sum(v.count * (v.frac_pos - rate) ** 2 for v in bins) / a.size,
        rate * (1 - rate),
    )


def sharpness(p: ArrayLike) -> float:
    return float(np.mean(abs(_probabilities(p) - 0.5)) * 2)


def summarize(p: ArrayLike, y: ArrayLike) -> dict[str, float]:
    a, b = _pair(p, y)
    d = murphy_decomposition(a, b)
    return dict(
        n=float(a.size),
        brier=brier(a, b),
        log_loss=log_loss(a, b),
        accuracy=accuracy(a, b),
        ece=ece(a, b),
        ece_quantile=ece(a, b, strategy="quantile"),
        sharpness=sharpness(a),
        reliability=d.reliability,
        resolution=d.resolution,
        uncertainty=d.uncertainty,
    )


@dataclass(frozen=True)
class BootstrapResult:
    mean_diff: float
    ci_low: float
    ci_high: float
    p_value: float


def paired_bootstrap(
    p1: ArrayLike,
    p2: ArrayLike,
    y: ArrayLike,
    metric: Literal["brier", "log_loss", "accuracy"] = "brier",
    n_boot: int = 2000,
    seed: int = 0,
) -> BootstrapResult:
    """Resample paired bouts; mean_diff is the observed score difference.

    The two-sided tail estimate counts ties in both tails and is capped at one.
    """
    a, b = _pair(p1, y)
    c, _ = _pair(p2, b)
    if metric not in ("brier", "log_loss", "accuracy"):
        raise ValueError("metric must be brier, log_loss or accuracy")
    if isinstance(n_boot, bool) or not isinstance(n_boot, (int, np.integer)) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    score = {"brier": brier, "log_loss": log_loss, "accuracy": accuracy}[metric]
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        ix = rng.integers(0, a.size, a.size)
        diffs[i] = score(a[ix], b[ix]) - score(c[ix], b[ix])
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return BootstrapResult(
        score(a, b) - score(c, b),
        float(lo),
        float(hi),
        float(min(1, 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0)))),
    )
