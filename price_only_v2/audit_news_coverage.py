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
    parser.add_argument("--links", type=Path,
                        default=Path("data/qualitative_dataset/nasdaq_article_company_links_2020_2023.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/price_only_current/news_coverage_audit.json"))
    args = parser.parse_args()
    columns = ["ticker", "prediction_date", "article_ids",
               "source_coverage_available", "news_count", "full_text_records_in_month"]
    news = pd.read_csv(args.news, usecols=columns,
                       dtype={"ticker": str, "prediction_date": str, "article_ids": str})
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
    news["article_ids"] = news.article_ids.fillna("")
    listed_count = news.article_ids.map(lambda value: 0 if not value else len(value.split(",")))
    if not (listed_count == news.news_count).all():
        raise ValueError("Company-day news_count disagrees with article_ids")
    args.output.parent.mkdir(parents=True, exist_ok=True)
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
    if args.links.exists():
        links = pd.read_csv(args.links, usecols=["article_id", "ticker", "Date", "match_status"],
                            dtype={"article_id": str, "ticker": str, "Date": str, "match_status": str})
        links["year"] = links.Date.str[:4]
        accepted = links.loc[links.match_status == "accepted", ["article_id", "ticker", "year"]].drop_duplicates()
        accepted_pairs = accepted[["article_id", "ticker"]].drop_duplicates()
        references = news.loc[news.article_ids != "", ["ticker", "article_ids"]].copy()
        references["article_id"] = references.article_ids.str.split(",")
        references = references.explode("article_id")[["ticker", "article_id"]]
        matched = references.merge(accepted_pairs, on=["ticker", "article_id"],
                                   how="left", indicator=True, validate="many_to_one")
        if not (matched._merge == "both").all():
            raise ValueError(f"News grid references {int((matched._merge != 'both').sum())} unaccepted article-company links")
        counts = accepted.groupby(["ticker", "year"]).size().reindex(
            pd.MultiIndex.from_product([sorted(tickers), ["2020", "2021", "2022", "2023"]],
                                       names=["ticker", "year"]), fill_value=0)
        counts.rename("accepted_article_links").reset_index().to_csv(
            args.output.with_name("accepted_news_links_by_ticker_year.csv"), index=False)
        report["article_link_check"] = {
            "links_sha256": hashlib.sha256(args.links.read_bytes()).hexdigest(),
            "accepted_article_company_pairs": len(accepted_pairs),
            "referenced_article_company_occurrences": len(references),
            "all_references_accepted": True,
            "tickers_with_any_accepted_link_2020_2023": int(accepted.ticker.nunique()),
            "tickers_without_accepted_links": sorted(tickers-set(accepted.ticker)),
            "tickers_with_any_accepted_link_2021": int(accepted.loc[accepted.year == "2021", "ticker"].nunique()),
        }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Exact grid for {len(tickers)} tickers; 2021 covered tickers "
          f"{report['tickers_with_any_full_text_coverage_2021']}; 2020-23 "
          f"covered tickers {report['tickers_with_any_full_text_coverage_2020_2023']}")


if __name__ == "__main__":
    main()
