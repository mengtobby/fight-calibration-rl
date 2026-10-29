"""Pre-fight matchup cards and conservative answer parsing."""

import math
import re
from dataclasses import dataclass
from typing import Literal

from fightcal.data.odds import market_prob_a
from fightcal.data.schema import Bout, FighterProfile

SYSTEM_PROMPT = (
    "You are an expert MMA analyst. Weigh reach, styles and stance, recent form and streaks, "
    "striking versus grappling rates, cardio proxies such as rounds fought and decision wins, "
    "and age. Be calibrated: acknowledge uncertainty and avoid unsupported confidence. "
    "Use only the pre-fight information provided."
)
ANSWER_PREFIX = "<answer>"


@dataclass(frozen=True)
class PromptOptions:
    anonymize: bool = True
    include_odds: bool = False
    odds_override_prob_a: float | None = None
    mode: Literal["reason_prob", "pick"] = "reason_prob"

    def __post_init__(self) -> None:
        if self.mode not in ("reason_prob", "pick"):
            raise ValueError("mode must be reason_prob or pick")
        if self.odds_override_prob_a is not None:
            prob_to_american(self.odds_override_prob_a)


def prob_to_american(p: float) -> float:
    """Convert an interior probability to finite, no-vig American odds."""
    if not math.isfinite(p) or not 0 < p < 1:
        raise ValueError("probability must be finite and strictly between 0 and 1")
    return -100 * p / (1 - p) if p >= 0.5 else 100 * (1 - p) / p


def _value(value: str | float | None, unit: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:g}{unit}" if isinstance(value, (int, float)) else value


def _card(fighter: FighterProfile, side: str, anonymize: bool) -> str:
    heading = f"Fighter {side}" + ("" if anonymize else f": {fighter.name}")
    fields = [
        ("Age", fighter.age, " years"),
        ("Height", fighter.height_cm, " cm"),
        ("Reach", fighter.reach_cm, " cm"),
        ("Weight", fighter.weight_lbs, " lb"),
        ("Stance", fighter.stance, ""),
        ("Current win streak", fighter.current_win_streak, ""),
        ("Current loss streak", fighter.current_lose_streak, ""),
        ("Longest win streak", fighter.longest_win_streak, ""),
        ("Total rounds fought", fighter.total_rounds_fought, ""),
        ("Total title bouts", fighter.total_title_bouts, ""),
        ("KO wins", fighter.wins_ko, ""),
        ("Submission wins", fighter.wins_sub, ""),
        ("Decision wins", fighter.wins_dec, ""),
        ("Significant strikes landed", fighter.sig_str_landed_pm, " / minute"),
        ("Significant strike accuracy", fighter.sig_str_pct, ""),
        ("Takedowns landed", fighter.td_landed_per15, " / 15 minutes"),
        ("Takedown accuracy", fighter.td_pct, ""),
        ("Submission attempts", fighter.sub_att_per15, " / 15 minutes"),
    ]
    lines = [
        heading,
        "Record (W-L-D): "
        + "-".join(_value(v) for v in (fighter.wins, fighter.losses, fighter.draws)),
    ]
    for label, value, unit in fields:
        if label.endswith("accuracy") and value is not None:
            lines.append(f"{label}: {float(value) * 100:g}%")
        else:
            lines.append(f"{label}: {_value(value, unit)}")
    return "\n".join(lines)


_DEFAULT_OPTIONS = PromptOptions()


def render_matchup(bout: Bout, opts: PromptOptions = _DEFAULT_OPTIONS) -> str:
    parts = [
        f"Weight class: {bout.weight_class}\nScheduled rounds: {bout.scheduled_rounds}\n"
        f"Title bout: {'yes' if bout.title_bout else 'no'}\nGender: {_value(bout.gender)}",
        _card(bout.fighter_a, "A", opts.anonymize),
        _card(bout.fighter_b, "B", opts.anonymize),
    ]
    if opts.include_odds:
        p = opts.odds_override_prob_a
        if p is None:
            p = market_prob_a(bout)
            a, b = bout.fighter_a.odds_american, bout.fighter_b.odds_american
        else:
            a, b = prob_to_american(p), prob_to_american(1 - p)
        parts.append(
            f"Betting market: American odds A {_value(a)}, B {_value(b)}; "
            f"de-vigged P(A)={_value(p)}, P(B)={_value(None if p is None else 1 - p)}"
        )
    if opts.mode == "reason_prob":
        parts.append(
            "Think step by step inside <think></think>, then give P(Fighter A wins) as a "
            "decimal in [0,1] inside <answer></answer>."
        )
    else:
        parts.append("Answer with the single letter A or B inside <answer></answer>.")
    return "\n\n".join(parts)


def build_messages(bout: Bout, opts: PromptOptions = _DEFAULT_OPTIONS) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_matchup(bout, opts)},
    ]


def _last_answer(text: str) -> str | None:
    starts = list(re.finditer(r"<answer>", text, re.IGNORECASE))
    if not starts:
        return None
    tail = text[starts[-1].end() :]
    return re.split(r"</answer>", tail, maxsplit=1, flags=re.IGNORECASE)[0].strip()


def parse_probability(text: str) -> float | None:
    answer = _last_answer(text)
    if answer is None:
        return None
    numbers = re.findall(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\s*%?", answer)
    if not numbers:
        return None
    values = [float(n.rstrip("% ")) / (100 if n.endswith("%") else 1) for n in numbers]
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in values):
        return None
    return values[0] if len(set(values)) == 1 else None


def parse_pick(text: str) -> str | None:
    answer = _last_answer(text)
    return answer.upper() if answer is not None and answer.upper() in ("A", "B") else None


def extract_reasoning(text: str) -> str:
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""
