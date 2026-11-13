"""Run BullsEye's development fusion pipeline with real or synthetic news forecasts."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fusion.combine import (ELIGIBLE, FUSION_MODEL, KEYS, PREDICTION_COLUMNS, combine,
                            export_predictions, read_coverage, read_predictions)
from fusion.evaluate import evaluate, read_targets

PRICE = Path("results/price_refinement/predictions_2020_2021.csv")
TARGETS = Path("data_local/price_current_50_2017_2023.csv")
COVERAGE = Path("data/qualitative_dataset/company_day_news_window_5session.csv")
UNIVERSE = Path("final_dataset/dataset_summary.csv")
TREND = "price_trend_shrunk"
DEMO_SEED = 42


def synthetic_news(price):
    """Use only company-date keys and a seed. Omit about a quarter of forecasts."""
    keys = price[KEYS].sort_values(KEYS).reset_index(drop=True)
    rng = np.random.default_rng(DEMO_SEED)
    keys = keys.loc[rng.random(len(keys)) >= .25]
    return keys.assign(model_name="synthetic_news", predicted_excess_return=rng.normal(0, .002, len(keys)))[
        PREDICTION_COLUMNS]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(args):
    universe = set(pd.read_csv(UNIVERSE, usecols=["ticker"]).ticker)
    price = read_predictions(args.price, args.price_model, universe)
    trend = price if args.price_model == TREND else read_predictions(args.trend, TREND, universe)
    coverage = read_coverage(args.coverage, args.coverage_flag, universe)
    news = synthetic_news(price) if args.demo else read_predictions(args.news, args.news_model, universe)
    joined = combine(price, trend, news, coverage)
    targets, excluded = read_targets(args.targets, universe)
    before_validation = joined.prediction_date.between("2021-01-01", args.validation_start, inclusive="left")
    joined = joined.loc[~before_validation]
    targets = targets.loc[~targets.prediction_date.between("2021-01-01", args.validation_start, inclusive="left")]
    metrics, coverage_metrics = evaluate(joined, targets, args.price_model == TREND)

    args.output.mkdir(parents=True, exist_ok=True)
    predictions = export_predictions(joined)
    if args.demo:
        predictions["model_name"] = "synthetic_demo_" + FUSION_MODEL
        news.to_csv(args.output / "synthetic_news_predictions.csv", index=False)
    predictions.to_csv(args.output / "combined_predictions.csv", index=False)
    joined[[*KEYS, "news_status"]].to_csv(args.output / "news_usage_by_company_day.csv", index=False)
    metrics.insert(0, "run_type", "synthetic_demo" if args.demo else "development")
    metrics.to_csv(args.output / "comparison_metrics.csv", index=False)
    coverage_metrics.to_csv(args.output / "coverage.csv", index=False)

    paths = {"price": args.price, "coverage": args.coverage, "targets": args.targets, "universe": UNIVERSE}
    if args.price_model != TREND:
        paths["trend"] = args.trend
    if not args.demo:
        paths["news"] = args.news
    manifest = {
        "run_type": "synthetic_demo" if args.demo else "development",
        "years": [2020, 2021],
        "validation_start": args.validation_start,
        "price_rows_excluded_before_validation_start": int(before_validation.sum()),
        "price_model": args.price_model,
        "trend_model": TREND,
        "news_model": "synthetic_news" if args.demo else args.news_model,
        "combined_model": predictions.model_name.iloc[0],
        "blend_weights": {"price": .5, "news": .5},
        "coverage_flag": args.coverage_flag,
        "cross_year_outcomes_excluded": excluded,
        "synthetic_news_seed": DEMO_SEED if args.demo else None,
        "inputs": {name: {"path": str(path), "sha256": digest(path)} for name, path in paths.items()},
    }
    (args.output / "run.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"{manifest['run_type']}: wrote {len(predictions):,} combined predictions to {args.output}")
    print(coverage_metrics.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true", help="Generate target-independent synthetic news forecasts")
    source.add_argument("--news", type=Path, help="News forecasts in the shared prediction format")
    parser.add_argument("--news-model", help="Model name in the news prediction CSV")
    parser.add_argument("--price", type=Path, default=PRICE)
    parser.add_argument("--price-model", default=TREND)
    parser.add_argument("--trend", type=Path, default=PRICE, help="CSV containing the scaled trend benchmark")
    parser.add_argument("--targets", type=Path, default=TARGETS)
    parser.add_argument("--coverage", type=Path, default=COVERAGE)
    parser.add_argument("--coverage-flag", default=ELIGIBLE)
    parser.add_argument("--validation-start", default="2021-01-01",
                        help="First allowed validation date in 2021 (YYYY-MM-DD)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.news_model) != bool(args.news):
        parser.error("--news-model is required with --news and unavailable with --demo")
    validation_start = pd.to_datetime(args.validation_start, format="%Y-%m-%d", errors="coerce")
    if (pd.isna(validation_start) or validation_start.year != 2021
            or validation_start.strftime("%Y-%m-%d") != args.validation_start):
        parser.error("--validation-start must be a 2021 date in YYYY-MM-DD format")
    run(args)


if __name__ == "__main__":
    main()
