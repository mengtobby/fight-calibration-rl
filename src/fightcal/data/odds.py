"""American odds conversion and two-outcome market normalization."""

import math

from fightcal.data.schema import Bout


def american_to_decimal(odds: float) -> float:
    if not math.isfinite(odds) or -100 < odds < 100:
        raise ValueError("American odds must be finite and <= -100 or >= 100")
    return 1 + (odds / 100 if odds > 0 else 100 / -odds)


def american_to_implied(odds: float) -> float:
    return 1 / american_to_decimal(odds)


def devig_pair(p_a: float, p_b: float, method: str = "multiplicative") -> tuple[float, float]:
    if any(not math.isfinite(p) or not 0 < p < 1 for p in (p_a, p_b)):
        raise ValueError("Implied probabilities must be finite and strictly between 0 and 1")
    if method == "multiplicative":
        q = p_a / (p_a + p_b)
    elif method == "power":
        low, high = 0.0, 1.0
        while p_a**high + p_b**high > 1:
            high *= 2
        for _ in range(100):
            mid = (low + high) / 2
            if p_a**mid + p_b**mid > 1:
                low = mid
            else:
                high = mid
        q = p_a ** ((low + high) / 2)
    else:
        raise ValueError(f"Unknown de-vig method: {method}")
    return q, 1 - q


def market_prob_a(bout: Bout, method: str = "multiplicative") -> float | None:
    a, b = bout.fighter_a.odds_american, bout.fighter_b.odds_american
    if a is None or b is None:
        return None
    return devig_pair(american_to_implied(a), american_to_implied(b), method)[0]
