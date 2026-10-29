"""Run a configured SFT experiment."""

import argparse
from dataclasses import replace

from fightcal.train.config import SFTRunConfig
from fightcal.train.sft import run_sft


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    cfg = SFTRunConfig.from_json(args.config)
    overrides = {}
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    run_sft(replace(cfg, **overrides))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
