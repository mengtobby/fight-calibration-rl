"""Matched-bout evaluation and JSON-safe research reports."""

import math
import re
from collections.abc import Iterable
from dataclasses import asdict
from typing import Any

from fightcal import diagnostics as d
from fightcal import metrics as m
from fightcal.predictions import PredictionRecord, aggregate, parse_rate


def _safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return value


def _checkpoint_key(value: str | None) -> tuple:
    # Natural ordering makes checkpoint-100 later than checkpoint-20.
    return (
        value is not None,
        tuple(
            (1, int(part)) if part.isdigit() else (0, part)
            for part in re.split(r"(\d+)", value or "")
        ),
    )


def _pairs(a: list[PredictionRecord], b: list[PredictionRecord]) -> list[tuple]:
    lookup = {r.bout_id: r for r in b if r.p_a is not None}
    return [
        (r, lookup[r.bout_id])
        for r in a
        if r.p_a is not None and r.bout_id in lookup and r.a_won == lookup[r.bout_id].a_won
    ]


def build_report(
    records: Iterable[PredictionRecord],
    *,
    reference_arm: str = "market",
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    raw = list(records)
    combined = aggregate(raw)
    notes: list[str] = []
    arms: dict[str, Any] = {}
    latest: dict[str, list[PredictionRecord]] = {}
    for arm in sorted({r.arm for r in combined}):
        rows = [r for r in combined if r.arm == arm and r.variant == "base"]
        if not rows:
            notes.append(f"{arm}: no base records")
            arms[arm] = {"n": 0, "notes": ["No base records"], "diagnostics": {}}
            continue
        checkpoint = max({r.checkpoint for r in rows}, key=_checkpoint_key)
        latest[arm] = [r for r in combined if r.arm == arm and r.checkpoint == checkpoint]
        base = [r for r in rows if r.checkpoint == checkpoint]
        valid = [r for r in base if r.p_a is not None]
        entry: dict[str, Any] = {
            "checkpoint": checkpoint,
            "n": len(valid),
            "parse_rate": parse_rate(
                r
                for r in raw
                if r.arm == arm and r.variant == "base" and r.checkpoint == checkpoint
            ),
            "notes": [],
            "diagnostics": {},
            "trajectory": [],
        }
        arms[arm] = entry
        if len(valid) < len(base):
            entry["notes"].append(f"Skipped {len(base) - len(valid)} unparsed base bouts")
        if valid:
            p, y = [r.p_a for r in valid], [r.a_won for r in valid]
            entry["metrics"] = m.summarize(p, y)
            entry["collapse_stats"] = asdict(d.collapse_stats(p))
            entry["reliability_table"] = [asdict(b) for b in m.reliability_table(p, y)]
        else:
            entry["notes"].append("No parsed base predictions")
        for cp in sorted({r.checkpoint for r in rows}, key=_checkpoint_key):
            scored = [r for r in rows if r.checkpoint == cp and r.p_a is not None]
            if scored:
                p, y = [r.p_a for r in scored], [r.a_won for r in scored]
                entry["trajectory"].append(
                    {
                        "checkpoint": cp,
                        "brier": m.brier(p, y),
                        "sharpness": m.sharpness(p),
                        "frac_near_half": d.collapse_stats(p).frac_near_half,
                    }
                )
    reference = [r for r in latest.get(reference_arm, []) if r.variant == "base"]
    for arm, rows in latest.items():
        entry = arms[arm]
        base = [r for r in rows if r.variant == "base"]
        paired = _pairs(base, reference)
        if len(paired) < len([r for r in base if r.p_a is not None]):
            entry["notes"].append("Paired comparison excludes bouts without a scored reference")
        if paired:
            a, b = zip(*paired, strict=True)
            entry["paired_bootstrap"] = {
                score: asdict(
                    m.paired_bootstrap(
                        [r.p_a for r in a],
                        [r.p_a for r in b],
                        [r.a_won for r in a],
                        metric=score,
                        n_boot=n_boot,
                        seed=seed,
                    )
                )
                for score in ("brier", "log_loss")
            }
            entry["paired_n"] = len(paired)
        else:
            entry["notes"].append(f"No scored bouts shared with {reference_arm}")
        market = [r for r in base if r.p_a is not None and r.market_p_a is not None]
        if market:
            try:
                entry["encompassing_test"] = asdict(
                    d.encompassing_test(
                        [r.p_a for r in market],
                        [r.market_p_a for r in market],
                        [r.a_won for r in market],
                    )
                )
            except ValueError as exc:
                entry["notes"].append(f"Encompassing test unavailable: {exc}")
        else:
            entry["notes"].append("No market odds for encompassing test")
        variants = {
            v: [r for r in rows if r.variant == v]
            for v in ("base", "odds_in_prompt", "odds_perturbed", "swapped", "reasoning_swap")
        }
        diag = entry["diagnostics"]
        odds = [
            r for r in variants["odds_in_prompt"] if r.p_a is not None and r.market_p_a is not None
        ]
        if odds:
            diag["market_copy_stats"] = asdict(
                d.market_copy_stats([r.p_a for r in odds], [r.market_p_a for r in odds])
            )
        else:
            entry["notes"].append("Market copying: no parsed odds-in-prompt records with odds")
        pairs = [
            (a, b)
            for a, b in _pairs(variants["odds_in_prompt"], variants["odds_perturbed"])
            if a.shown_market_p_a is not None and b.shown_market_p_a is not None
        ]
        if pairs:
            diag["odds_sensitivity"] = asdict(
                d.odds_sensitivity(
                    [a.p_a for a, b in pairs],
                    [b.p_a for a, b in pairs],
                    [a.shown_market_p_a for a, b in pairs],
                    [b.shown_market_p_a for a, b in pairs],
                )
            )
        else:
            entry["notes"].append("Odds sensitivity: no matched predictions with shown odds")
        for variant, name in [
            ("swapped", "symmetry_gap"),
            ("reasoning_swap", "reasoning_ablation"),
        ]:
            pairs = _pairs(base, variants[variant])
            if pairs:
                pa, pb = [a.p_a for a, b in pairs], [b.p_a for a, b in pairs]
                diag[name] = (
                    d.symmetry_gap(pa, [1 - p for p in pb])
                    if variant == "swapped"
                    else asdict(d.reasoning_ablation(pa, pb))
                )
            else:
                entry["notes"].append(f"{name}: no matched parsed variants")
        texts = [
            r
            for r in raw
            if r.arm == arm
            and r.checkpoint == entry["checkpoint"]
            and r.variant == "base"
            and r.p_a is not None
            and r.completion is not None
        ]
        if texts:
            diag["reasoning_consistency"] = d.reasoning_consistency(
                [r.completion for r in texts], [r.p_a for r in texts]
            )
        else:
            entry["notes"].append("Reasoning consistency: no parsed base completions")
    return _safe({"reference_arm": reference_arm, "arms": arms, "notes": notes})


def render_markdown(report: dict) -> str:
    lines = [
        "# Forecast evaluation",
        "",
        "| Arm | n | Parse rate | Brier | Log loss | Accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, entry in report["arms"].items():
        metrics = entry.get("metrics", {})
        values = [
            entry.get("n"),
            entry.get("parse_rate"),
            *[metrics.get(k) for k in ("brier", "log_loss", "accuracy")],
        ]
        cells = ["—" if v is None else f"{v:.4g}" for v in values]
        lines.append("| " + arm.replace("|", "\\|") + " | " + " | ".join(cells) + " |")
    for arm, entry in report["arms"].items():
        lines.extend(["", f"## {arm} diagnostics", ""])
        for key in ("collapse_stats", "paired_bootstrap", "encompassing_test", "trajectory"):
            if key in entry:
                lines.append(f"- {key}: {entry[key]}")
        for key, value in entry["diagnostics"].items():
            lines.append(f"- {key}: {value}")
        lines.extend(f"- {note}" for note in entry.get("notes", []))
    lines.extend(["", *report.get("notes", [])])
    return "\n".join(lines) + "\n"
