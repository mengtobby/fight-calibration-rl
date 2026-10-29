"""Generate non-LM baseline predictions."""

import argparse

from fightcal.baselines import (
    base_rate_baseline,
    constant_baseline,
    logistic_baseline,
    market_baseline,
)
from fightcal.data.io import load_bouts_jsonl
from fightcal.predictions import write_predictions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    train, test = load_bouts_jsonl(args.train), load_bouts_jsonl(args.test)
    records = (
        market_baseline(test)
        + constant_baseline(test)
        + base_rate_baseline(train, test)
        + logistic_baseline(train, test)
    )
    write_predictions(args.out, records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
