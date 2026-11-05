"""Audit the qualitative company-day grid without reading price outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--news", type=Path, required=True,
                        help="Materialized qualitative company_day_news_window_5session.csv")
    parser.add_argument("--price-grid", type=Path,
                        default=Path("final_dataset/quant_dataset_30day.csv"))
    parser.add_argument("--summary", type=Path,
                        default=Path("final_dataset/dataset_summary.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/price_only_current/news_coverage_audit.json"))
    args = parser.parse_args()
    columns = ["ticker", "prediction_date", "source_coverage_available",
               "news_count", "full_text_records_in_month"]
    news = pd.read_csv(args.news, usecols=columns)
    price = pd.read_csv(args.price_grid, usecols=["ticker", "prediction_date"])
    tickers = set(pd.read_csv(args.summary).ticker)
    if set(news.ticker) != tickers or set(price.ticker) != tickers:
        raise ValueError("News, price, and summary ticker lists disagree")
    if news.duplicated(["ticker", "prediction_date"]).any() or \
            price.duplicated(["ticker", "prediction_date"]).any():
        raise ValueError("Duplicate company-day key")
    joined = price.merge(news, on=["ticker", "prediction_date"],
                         how="outer", indicator=True, validate="one_to_one")
    if not (joined._merge == "both").all():
        raise ValueError("News and price company-day grids differ")
    news = news.loc[news.prediction_date.between("2020-01-01", "2023-12-31")].copy()
    news["year"] = news.prediction_date.str[:4]
    summary = news.groupby(["ticker", "year"]).agg(
        rows=("ticker", "size"),
        coverage_days=("source_coverage_available", "sum"),
        days_with_articles=("news_count", lambda s: int((s > 0).sum())),
        total_article_links=("news_count", "sum"),
    ).reset_index()
    summary.to_csv(args.output.with_name("news_coverage_by_ticker_year.csv"), index=False)
    by_year = news.groupby("year").agg(
        rows=("ticker", "size"),
        covered_rows=("source_coverage_available", "sum"),
        days_with_articles=("news_count", lambda s: int((s > 0).sum())),
    )
    any_coverage = summary.groupby("ticker").coverage_days.sum()
    coverage_2021 = summary.loc[summary.year == "2021"]
    digest = hashlib.sha256(args.news.read_bytes()).hexdigest()
    report = {
        "news_table_sha256": digest,
        "company_count": len(tickers),
        "exact_price_news_company_day_grid": True,
        "full_text_coverage_definition": "Source has at least one nonempty full-text Nasdaq record for that ticker-month",
        "by_year": {year: {key: int(value) for key, value in row.items()}
                    for year, row in by_year.to_dict(orient="index").items()},
        "tickers_with_any_full_text_coverage_2020_2023": int((any_coverage > 0).sum()),
        "tickers_never_covered_2020_2023": sorted(any_coverage.index[any_coverage == 0].tolist()),
        "tickers_with_any_full_text_coverage_2021": int((coverage_2021.coverage_days > 0).sum()),
        "tickers_not_covered_in_2021": sorted(coverage_2021.loc[
            coverage_2021.coverage_days == 0, "ticker"].tolist()),
        "article_days_flagged_uncovered": int(((news.news_count > 0) &
                                               ~news.source_coverage_available).sum()),
        "note": "Ticker rows in the news grid are not evidence of usable full-text coverage. Resolve uncovered article days and choose a common eligible comparison mask before training fusion models. No price target values were loaded.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Exact grid for {len(tickers)} tickers; 2021 covered tickers "
          f"{report['tickers_with_any_full_text_coverage_2021']}; 2020-23 "
          f"covered tickers {report['tickers_with_any_full_text_coverage_2020_2023']}")


if __name__ == "__main__":
    main()
