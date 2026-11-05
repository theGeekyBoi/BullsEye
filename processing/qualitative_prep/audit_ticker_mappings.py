"""Audit ticker-symbol mappings between the quantitative dataset and FNSPID news.

This checks whether each modeling ticker appears exactly in FNSPID's Nasdaq
news file, and also checks common alternate forms such as BRK-B vs BRK.B.

Outputs:
    data/processed/qualitative/ticker_symbol_mapping_summary.csv
    data/processed/qualitative/ticker_symbol_mapping_candidates.csv
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
QUANT_DATASET_PATH = REPO_ROOT / "data/quantitative_dataset/quant_dataset_30day.csv"
NASDAQ_NEWS_PATH = REPO_ROOT / "data/raw/nasdaq_exteral_data.csv"
OUTPUT_DIR = REPO_ROOT / "data/qualitative_dataset"


KNOWN_ALTERNATES = {
    "BRK-B": ["BRK.B", "BRKB", "BRK/A", "BRK.A"],
    "GOOG": ["GOOGL"],
    "GOOGL": ["GOOG"],
    "META": ["FB"],
    "FB": ["META"],
    "RTX": ["RTN", "UTX"],
}


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def candidate_symbols(ticker: str) -> list[tuple[str, str]]:
    ticker = ticker.upper().strip()
    candidates = [(ticker, "exact")]

    if "-" in ticker:
        candidates.append((ticker.replace("-", "."), "hyphen_to_dot"))
        candidates.append((ticker.replace("-", ""), "hyphen_removed"))
        candidates.append((ticker.replace("-", "/"), "hyphen_to_slash"))

    if "." in ticker:
        candidates.append((ticker.replace(".", "-"), "dot_to_hyphen"))
        candidates.append((ticker.replace(".", ""), "dot_removed"))

    for alternate in KNOWN_ALTERNATES.get(ticker, []):
        candidates.append((alternate, "known_alternate"))

    deduped = []
    seen = set()
    for symbol, match_type in candidates:
        if symbol not in seen:
            deduped.append((symbol, match_type))
            seen.add(symbol)
    return deduped


def load_model_tickers() -> pd.DataFrame:
    quantitative_df = pd.read_csv(QUANT_DATASET_PATH, usecols=["ticker"])
    tickers = sorted(quantitative_df["ticker"].dropna().str.upper().str.strip().unique())
    return pd.DataFrame({"ticker": tickers})


def count_nasdaq_symbols(symbols: list[str]) -> pd.DataFrame:
    symbol_list = ", ".join(sql_string(symbol) for symbol in symbols)
    query = f"""
    SELECT
        upper(trim(Stock_symbol)) AS nasdaq_symbol,
        count(*) AS nasdaq_total_rows,
        sum(CASE WHEN substr(Date, 1, 4) BETWEEN '2017' AND '2020' THEN 1 ELSE 0 END) AS rows_2017_2020,
        sum(CASE WHEN substr(Date, 1, 4) = '2021' THEN 1 ELSE 0 END) AS rows_2021,
        sum(CASE WHEN substr(Date, 1, 4) BETWEEN '2022' AND '2023' THEN 1 ELSE 0 END) AS rows_2022_2023
    FROM read_csv_auto(
        {sql_string(NASDAQ_NEWS_PATH.as_posix())},
        header=true,
        all_varchar=true,
        strict_mode=false,
        ignore_errors=true
    )
    WHERE upper(trim(Stock_symbol)) IN ({symbol_list})
    GROUP BY nasdaq_symbol
    """
    return duckdb.sql(query).df()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_tickers = load_model_tickers()
    candidate_rows = []
    for ticker in model_tickers["ticker"]:
        for nasdaq_symbol, match_type in candidate_symbols(ticker):
            candidate_rows.append(
                {
                    "ticker": ticker,
                    "nasdaq_symbol": nasdaq_symbol,
                    "match_type": match_type,
                }
            )

    candidates = pd.DataFrame(candidate_rows)
    counts = count_nasdaq_symbols(sorted(candidates["nasdaq_symbol"].unique()))

    candidates = candidates.merge(counts, on="nasdaq_symbol", how="left")
    count_columns = ["nasdaq_total_rows", "rows_2017_2020", "rows_2021", "rows_2022_2023"]
    candidates[count_columns] = candidates[count_columns].fillna(0).astype(int)
    candidates["present_in_nasdaq"] = candidates["nasdaq_total_rows"] > 0

    summary = (
        candidates[candidates["present_in_nasdaq"]]
        .sort_values(["ticker", "match_type", "nasdaq_total_rows"], ascending=[True, True, False])
        .groupby("ticker")
        .agg(
            matched_nasdaq_symbols=("nasdaq_symbol", lambda values: ",".join(values)),
            match_types=("match_type", lambda values: ",".join(values)),
            total_rows=("nasdaq_total_rows", "sum"),
            rows_2017_2020=("rows_2017_2020", "sum"),
            rows_2021=("rows_2021", "sum"),
            rows_2022_2023=("rows_2022_2023", "sum"),
        )
        .reset_index()
    )
    summary = model_tickers.merge(summary, on="ticker", how="left")
    summary["matched_nasdaq_symbols"] = summary["matched_nasdaq_symbols"].fillna("")
    summary["match_types"] = summary["match_types"].fillna("")
    summary[["total_rows", "rows_2017_2020", "rows_2021", "rows_2022_2023"]] = summary[
        ["total_rows", "rows_2017_2020", "rows_2021", "rows_2022_2023"]
    ].fillna(0).astype(int)
    summary["has_any_nasdaq_symbol"] = summary["total_rows"] > 0
    summary["needs_symbol_mapping"] = summary["match_types"].str.contains("known_alternate|hyphen|dot", regex=True)

    candidates_path = OUTPUT_DIR / "ticker_symbol_mapping_candidates.csv"
    summary_path = OUTPUT_DIR / "ticker_symbol_mapping_summary.csv"
    candidates.to_csv(candidates_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Wrote candidate mapping audit: {candidates_path}")
    print(f"Wrote mapping summary: {summary_path}")
    print("\nTickers without any matching Nasdaq symbol candidate:")
    print(summary.loc[~summary["has_any_nasdaq_symbol"], ["ticker"]].to_string(index=False))
    print("\nTickers with non-exact mappings present:")
    print(
        summary.loc[
            summary["needs_symbol_mapping"],
            ["ticker", "matched_nasdaq_symbols", "match_types", "total_rows"],
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
