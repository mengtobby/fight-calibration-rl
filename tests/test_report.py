import json
from dataclasses import replace

import pytest

from fightcal.predictions import VARIANTS, PredictionRecord
from fightcal.report import build_report, render_markdown


def test_all_variants():
    rows = []
    for i in range(12):
        p = 0.3 if i % 2 else 0.7
        for arm in ("market", "model"):
            for variant in VARIANTS:
                rows.append(
                    PredictionRecord(
                        str(i),
                        arm,
                        variant,
                        0,
                        p,
                        bool(i % 3),
                        p,
                        shown_market_p_a=0.6 if variant == "odds_perturbed" else 0.5,
                        completion="Fighter A has the advantage.",
                        checkpoint="step-20",
                    )
                )
    rows.append(replace(rows[0], checkpoint="step-9"))
    report = build_report(rows, n_boot=10)
    assert report["arms"]["model"]["checkpoint"] == "step-20"
    assert report["arms"]["model"]["diagnostics"]["symmetry_gap"] == pytest.approx(0)
    assert len(report["arms"]["market"]["trajectory"]) == 2
    assert report["arms"]["model"]["paired_n"] == 12
    assert set(report["arms"]["model"]["diagnostics"]) == {
        "market_copy_stats",
        "odds_sensitivity",
        "symmetry_gap",
        "reasoning_ablation",
        "reasoning_consistency",
    }
    assert "NaN" not in json.dumps(report, allow_nan=False)
    markdown = render_markdown(report)
    assert "market" in markdown and "model" in markdown


def test_missing_data():
    rows = [
        PredictionRecord("b", "failed", "base", 0, None, True, None),
        PredictionRecord("b", "no_variants", "base", 0, 0.5, True, None),
        PredictionRecord("b", "no_base", "swapped", 0, 0.5, True, None),
    ]
    report = build_report(rows, n_boot=2)
    assert report["arms"]["failed"]["n"] == 0
    assert report["notes"]
    assert all(arm in render_markdown(report) for arm in ("failed", "no_variants", "no_base"))
    json.dumps(report, allow_nan=False)
    assert build_report([])["arms"] == {}
    with pytest.raises(ValueError):
        build_report([], n_boot=0)
