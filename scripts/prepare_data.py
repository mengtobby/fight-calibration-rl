"""Prepare randomized, temporally partitioned UFC data."""

import argparse
import json
from datetime import date
from pathlib import Path

from fightcal.data.io import save_bouts_jsonl
from fightcal.data.odds import market_prob_a
from fightcal.data.split import randomize_corners, temporal_split
from fightcal.data.ufc import load_bouts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--val-start", type=date.fromisoformat, default=date(2022, 1, 1))
    parser.add_argument("--test-start", type=date.fromisoformat, default=date(2023, 7, 1))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--require-odds", action="store_true")
    args = parser.parse_args(argv)
    bouts = randomize_corners(load_bouts(args.csv, require_odds=args.require_odds), args.seed)
    split = temporal_split(bouts, val_start=args.val_start, test_start=args.test_start)
    target = Path(args.out_dir)
    target.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in ("train", "val", "test"):
        records = getattr(split, name)
        save_bouts_jsonl(target / f"{name}.jsonl", records)
        summary[name] = {
            "count": len(records),
            "date_min": min(b.date for b in records).isoformat() if records else None,
            "date_max": max(b.date for b in records).isoformat() if records else None,
            "a_win_rate": sum(b.a_won for b in records) / len(records) if records else None,
            "share_with_odds": sum(market_prob_a(b) is not None for b in records) / len(records)
            if records
            else None,
        }
    (target / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
