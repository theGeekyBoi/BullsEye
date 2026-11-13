"""Export daily replay prices in split-adjusted share units for BullsEye."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf

from price_only_v2.build_dataset import sha256

SOURCE = Path("data_local/replay_source_2016_2026.csv")
OUTPUT = Path("data_local/daily_prices_replay.csv")
UNIVERSE = Path("final_dataset/dataset_summary.csv")
SOURCE_COLUMNS = {
    "Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close",
    "Adj Close": "adjusted_close", "Volume": "volume", "Dividends": "dividend",
    "Capital Gains": "capital_gain", "Stock Splits": "split_ratio",
}
# Yahoo reports these spin-offs or distributions as price adjustments.
NON_STANDARD_ACTIONS = {
    ("GE", "2019-02-26"), ("GE", "2023-01-04"), ("GE", "2024-04-02"),
    ("IBM", "2021-11-04"), ("MRK", "2021-06-03"), ("RTX", "2020-04-03"),
}


def fetch(tickers, source):
    frames = []
    for offset in range(0, len(tickers), 8):
        batch = tickers[offset:offset + 8]
        data = yf.download(batch, start="2016-11-01", end="2026-11-12",
                           auto_adjust=False, actions=True, group_by="ticker", progress=False)
        for ticker in batch:
            if ticker not in data.columns.get_level_values(0):
                raise ValueError(f"No daily data for {ticker}")
            frame = data[ticker].dropna(subset=["Close"]).rename_axis("Date").reset_index()
            if frame.empty:
                raise ValueError(f"No daily data for {ticker}")
            if "Capital Gains" not in frame:
                frame["Capital Gains"] = 0.0
            frames.append(frame[list(SOURCE_COLUMNS)].assign(ticker=ticker))
    source.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(source, index=False, float_format="%.12g")


def export(source, output):
    tickers = sorted(pd.read_csv(UNIVERSE).ticker.unique()) + ["SPY"]
    if not source.exists():
        fetch(tickers, source)
    table = pd.read_csv(source).rename(columns=SOURCE_COLUMNS)
    table = table[["ticker", *SOURCE_COLUMNS.values()]].copy()
    if set(table.ticker) != set(tickers) or table.duplicated(["ticker", "date"]).any():
        raise ValueError("Unexpected tickers or duplicate daily prices")
    numeric = table[list(SOURCE_COLUMNS.values())[1:]]
    price_columns = ["open", "high", "low", "close", "adjusted_close"]
    if (not np.isfinite(numeric).all().all() or (numeric < 0).any().any()
            or (table[price_columns] <= 0).any().any()
            or (table.high < table[["open", "low", "close"]].max(axis=1)).any()
            or (table.low > table[["open", "close"]].min(axis=1)).any()):
        raise ValueError("Nonfinite, negative, or inconsistent daily source data")

    # Cover the shared project period and the final five-session exits.
    table = table.loc[table.date.between("2017-01-01", "2024-01-09")].copy()
    sessions = mcal.get_calendar("NYSE").valid_days("2017-01-01", "2024-01-10").strftime("%Y-%m-%d")
    next_sessions = dict(zip(sessions[:-1], sessions[1:]))
    if not set(table.date).issubset(next_sessions):
        raise ValueError("Source price on a non-session")
    missing = {}
    for ticker, group in table.groupby("ticker"):
        dates = sessions[(sessions >= group.date.min()) & (sessions <= "2024-01-09")]
        gaps = sorted(set(dates) - set(group.date))
        if gaps:
            missing[ticker] = gaps
        if ticker != "PLTR" and group.date.min() != sessions[0]:
            raise ValueError(f"Unexpected missing initial dates for {ticker}")
    if missing:
        raise ValueError(f"Missing sessions after listing: {missing}")

    table["next_trading_date"] = table.date.map(next_sessions)
    table["adjusted_open"] = table.open * table.adjusted_close / table.close
    table["cash_distribution"] = table.dividend + table.capital_gain
    table["non_standard_action"] = [
        (ticker, day) in NON_STANDARD_ACTIONS or (0.5 < ratio < 2 and ratio != 1)
        for ticker, day, ratio in zip(table.ticker, table.date, table.split_ratio)
    ]
    table = table.sort_values(["date", "ticker"])
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False, float_format="%.12g")
    manifest = {
        "source_sha256": sha256(source), "output_sha256": sha256(output),
        "rows": len(table), "symbols_including_spy": len(tickers),
        "first_date": table.date.min(), "last_date": table.date.max(),
        "pltr_first_date": table.loc[table.ticker == "PLTR", "date"].min(),
        "missing_sessions_after_listing": missing,
        "ordinary_split_rows": int(((table.split_ratio > 0) & ~table.non_standard_action).sum()),
        "non_standard_action_rows": int(table.non_standard_action.sum()),
        "cash_distribution_rows": int((table.cash_distribution > 0).sum()),
        "replay_basis": {
            "total_return_units": "Fill at adjusted_open and mark at adjusted_close. Distribution and split effects are embedded.",
            "cash_distribution_units": "Use open and close with cash_distribution. Prices are already split-adjusted. Review special actions separately.",
        },
        "source": "Yahoo Finance via yfinance download with auto_adjust=False and actions=True.",
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    export(args.source, args.output)
