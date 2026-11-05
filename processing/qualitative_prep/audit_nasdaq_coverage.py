"""Audit FNSPID Nasdaq news coverage for the modeling tickers.

For each ticker in the quantitative dataset and each year from 2020 through
2023, count source-tagged Nasdaq records and records with nonempty article text.

Output:
    data/processed/qualitative/nasdaq_news_coverage_2020_2023.csv
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from audit_ticker_mappings import (
    NASDAQ_NEWS_PATH,
    OUTPUT_DIR,
    candidate_symbols,
    load_model_tickers,
    sql_string,
)


YEARS = ["2020", "2021", "2022", "2023"]


def build_candidate_table(model_tickers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker in model_tickers["ticker"]:
        for nasdaq_symbol, match_type in candidate_symbols(ticker):
            rows.append(
                {
                    "ticker": ticker,
                    "nasdaq_symbol": nasdaq_symbol,
                    "match_type": match_type,
                }
            )
    return pd.DataFrame(rows)


def count_nasdaq_coverage(symbols: list[str]) -> pd.DataFrame:
    symbol_list = ", ".join(sql_string(symbol) for symbol in symbols)
    year_list = ", ".join(sql_string(year) for year in YEARS)

    query = f"""
    SELECT
        upper(trim(Stock_symbol)) AS nasdaq_symbol,
        substr(Date, 1, 4) AS year,
        count(*) AS source_records,
        sum(
            CASE
                WHEN Article IS NOT NULL
                    AND length(trim(Article)) > 0
                    AND lower(trim(Article)) NOT IN ('nan', 'none', 'null', '0')
                THEN 1
                ELSE 0
            END
        ) AS full_text_records
    FROM read_csv_auto(
        {sql_string(NASDAQ_NEWS_PATH.as_posix())},
        header=true,
        all_varchar=true,
        strict_mode=false,
        ignore_errors=true
    )
    WHERE upper(trim(Stock_symbol)) IN ({symbol_list})
      AND substr(Date, 1, 4) IN ({year_list})
    GROUP BY nasdaq_symbol, year
    """
    return duckdb.sql(query).df()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_tickers = load_model_tickers()
    candidates = build_candidate_table(model_tickers)
    counts = count_nasdaq_coverage(sorted(candidates["nasdaq_symbol"].unique()))

    candidate_counts = candidates.merge(counts, on="nasdaq_symbol", how="left")
    candidate_counts["year"] = candidate_counts["year"].fillna("")
    candidate_counts[["source_records", "full_text_records"]] = candidate_counts[
        ["source_records", "full_text_records"]
    ].fillna(0).astype(int)

    ticker_year_grid = pd.MultiIndex.from_product(
        [model_tickers["ticker"], YEARS],
        names=["ticker", "year"],
    ).to_frame(index=False)

    coverage = (
        candidate_counts[candidate_counts["year"].isin(YEARS)]
        .groupby(["ticker", "year"])
        .agg(
            source_records=("source_records", "sum"),
            full_text_records=("full_text_records", "sum"),
            source_symbols_counted=(
                "nasdaq_symbol",
                lambda values: ",".join(sorted(set(values))),
            ),
        )
        .reset_index()
    )

    coverage = ticker_year_grid.merge(coverage, on=["ticker", "year"], how="left")
    coverage[["source_records", "full_text_records"]] = coverage[
        ["source_records", "full_text_records"]
    ].fillna(0).astype(int)
    coverage["source_symbols_counted"] = coverage["source_symbols_counted"].fillna("")
    coverage["has_source_records"] = coverage["source_records"] > 0
    coverage["has_full_text_records"] = coverage["full_text_records"] > 0

    output_path = OUTPUT_DIR / "nasdaq_news_coverage_2020_2023.csv"
    coverage.to_csv(output_path, index=False)

    print(f"Wrote Nasdaq coverage audit: {output_path}")
    print("\nTicker-years without source records:")
    print(
        coverage.loc[
            ~coverage["has_source_records"],
            ["ticker", "year"],
        ].to_string(index=False)
    )
    print("\nTicker-years with source records but no full article text:")
    print(
        coverage.loc[
            coverage["has_source_records"] & ~coverage["has_full_text_records"],
            ["ticker", "year", "source_records"],
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
