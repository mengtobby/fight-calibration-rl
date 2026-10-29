"""Build evaluation reports."""

import argparse
import json
from pathlib import Path

from fightcal.predictions import read_predictions
from fightcal.report import build_report, render_markdown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    report = build_report([r for path in args.predictions for r in read_predictions(path)])
    target = Path(args.out_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (target / "report.md").write_text(render_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
