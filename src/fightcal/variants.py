"""Deterministic inference interventions, without model dependencies."""

import math
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256

from fightcal.data.odds import market_prob_a
from fightcal.data.schema import Bout
from fightcal.prompts import ANSWER_PREFIX, PromptOptions, extract_reasoning


def perturb_market(p: float, seed: int, bout_id: str, shift: float = 0.15) -> float:
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("p must be a finite probability in [0, 1]")
    if not math.isfinite(shift) or shift <= 0:
        raise ValueError("shift must be finite and positive")
    # Salted so the direction is independent of randomize_corners, which hashes seed:bout_id.
    sign = 1 if sha256(f"perturb:{seed}:{bout_id}".encode()).digest()[0] < 128 else -1
    result = min(0.95, max(0.05, p + sign * shift))
    if result == p:
        result = min(0.95, max(0.05, p - sign * shift))
    return result


def pair_for_reasoning_swap(bout_ids: Iterable[str], seed: int) -> dict[str, str]:
    ids = list(bout_ids)
    if len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValueError("reasoning swap requires at least two unique bout IDs")
    ordered = sorted(ids, key=lambda key: (sha256(f"{seed}:{key}".encode()).digest(), key))
    return dict(zip(ordered, ordered[1:] + ordered[:1], strict=True))


def build_forced_answer_prefix(reasoning: str) -> str:
    return "<think>" + reasoning + "</think>\n" + ANSWER_PREFIX


@dataclass(frozen=True)
class VariantRequest:
    bout_id: str
    variant: str
    opts: PromptOptions
    swapped: bool
    forced_prefix: str | None


def plan_variants(
    bouts: Iterable[Bout],
    variants: Iterable[str],
    *,
    base_completions: dict[str, str] | None = None,
    seed: int = 0,
    base_include_odds: bool = False,
) -> list[VariantRequest]:
    records, names = list(bouts), list(variants)
    allowed = {"base", "odds_in_prompt", "odds_perturbed", "swapped", "reasoning_swap"}
    if any(name not in allowed for name in names):
        raise ValueError("unknown inference variant")
    partners = {}
    if "reasoning_swap" in names:
        if base_completions is None or any(b.bout_id not in base_completions for b in records):
            raise ValueError("reasoning_swap requires base completions for every bout")
        partners = pair_for_reasoning_swap([b.bout_id for b in records], seed)
    result = []
    for name in names:
        for bout in records:
            override = None
            if name == "odds_perturbed":
                market = market_prob_a(bout)
                if market is None:
                    raise ValueError(f"odds_perturbed requires odds for {bout.bout_id}")
                override = perturb_market(market, seed, bout.bout_id)
            prefix = None
            if name == "reasoning_swap":
                assert base_completions is not None
                prefix = build_forced_answer_prefix(
                    extract_reasoning(base_completions[partners[bout.bout_id]])
                )
            result.append(
                VariantRequest(
                    bout.bout_id,
                    name,
                    PromptOptions(
                        include_odds=base_include_odds
                        or name in {"odds_in_prompt", "odds_perturbed"},
                        odds_override_prob_a=override,
                    ),
                    name == "swapped",
                    prefix,
                )
            )
    return result


def to_original_perspective(p: float | None, swapped: bool) -> float | None:
    if p is None:
        return None
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("p must be a finite probability in [0, 1]")
    return 1 - p if swapped else p
