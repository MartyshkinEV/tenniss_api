try:
    import _bootstrap
except ModuleNotFoundError:
    from scripts import _bootstrap

import argparse
import json

import pandas as pd

from config import settings
from src.betting.analytics import backtest_market_predictions, build_profitability_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV or JSONL file with model_prob/odds/stake/actual columns")
    parser.add_argument("--output", default=str(settings.market_backtest_path))
    parser.add_argument("--analytics-output", default=str(settings.market_analytics_path))
    args = parser.parse_args()

    if args.input.endswith(".jsonl"):
        frame = pd.read_json(args.input, lines=True)
    else:
        frame = pd.read_csv(args.input)

    backtest = backtest_market_predictions(frame)
    report = build_profitability_report(backtest)

    backtest.to_json(args.output, orient="records", indent=2)
    with open(args.analytics_output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=True, indent=2)
    print(args.output)
    print(args.analytics_output)


if __name__ == "__main__":
    main()
