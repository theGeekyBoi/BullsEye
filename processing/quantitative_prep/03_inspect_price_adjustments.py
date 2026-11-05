
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

YAHOO_FILE = ROOT / "prototype_yfinance" / "daily_prices.csv"
ACTIONS_FILE = ROOT / "prototype_yfinance" / "corporate_actions.csv"
FNSPID_DIR = ROOT / "raw_fnspid" / "prices" / "full_history"

SAMPLE_TICKERS = ["AAPL", "NVDA", "MSFT", "SPY", "HD"]

def load_fnspid(ticker):
    # META's historical ticker was FB.
    filename = "FB.csv" if ticker == "META" else f"{ticker}.csv"
    path = FNSPID_DIR / filename

    if not path.exists():
        return None

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for col in ["open", "high", "low", "close", "adj close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["date"]).sort_values("date")

def main():
    yahoo = pd.read_csv(YAHOO_FILE)
    yahoo["date"] = pd.to_datetime(yahoo["date"], errors="coerce")
    yahoo["ticker"] = yahoo["ticker"].astype(str).str.upper()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        yahoo[col] = pd.to_numeric(yahoo[col], errors="coerce")

    print("=" * 75)
    print("PRICE ADJUSTMENT COMPARISON")
    print("=" * 75)

    for ticker in SAMPLE_TICKERS:
        print(f"\n\n--- {ticker} ---")

        y = yahoo[yahoo["ticker"] == ticker].copy()
        f = load_fnspid(ticker)

        if f is None:
            print("No matching FNSPID file.")
            continue

        merged = y.merge(
            f[["date", "open", "high", "low", "close", "adj close"]],
            on="date",
            how="inner",
            suffixes=("_yahoo", "_fnspid")
        )

        print("Yahoo rows:", len(y))
        print("FNSPID rows:", len(f))
        print("Overlapping dates:", len(merged))

        if len(merged) == 0:
            continue

        # Compare Yahoo prototype Close against FNSPID Close
        # and FNSPID Adjusted Close.
        merged["diff_vs_close"] = (
            merged["Close"] / merged["close"] - 1
        ).abs()

        merged["diff_vs_adj_close"] = (
            merged["Close"] / merged["adj close"] - 1
        ).abs()

        print(
            "Median absolute difference vs FNSPID Close:",
            merged["diff_vs_close"].median()
        )
        print(
            "Median absolute difference vs FNSPID Adj Close:",
            merged["diff_vs_adj_close"].median()
        )

        print("\nFirst 5 overlapping records:")
        print(
            merged[
                [
                    "date", "Open", "Close",
                    "open", "close", "adj close"
                ]
            ].head(5).to_string(index=False)
        )

        print("\nLargest differences from FNSPID Adj Close:")
        print(
            merged.nlargest(3, "diff_vs_adj_close")[
                [
                    "date", "Close", "adj close",
                    "diff_vs_adj_close"
                ]
            ].to_string(index=False)
        )

    # Inspect invalid HD rows.
    print("\n\n" + "=" * 75)
    print("INVALID HD OHLC RECORDS")
    print("=" * 75)

    hd = yahoo[yahoo["ticker"] == "HD"].copy()

    invalid = (
        (hd["Open"] <= 0)
        | (hd["High"] <= 0)
        | (hd["Low"] <= 0)
        | (hd["Close"] <= 0)
        | (hd["Low"] > hd["Open"])
        | (hd["Low"] > hd["Close"])
        | (hd["High"] < hd["Open"])
        | (hd["High"] < hd["Close"])
        | (hd["High"] < hd["Low"])
    )

    bad_rows = hd[invalid]
    print(bad_rows.to_string(index=False))

    # Compare the HD anomaly with FNSPID, if the date exists.
    if len(bad_rows):
        f = load_fnspid("HD")

        for _, row in bad_rows.iterrows():
            print("\nFNSPID comparison for:", row["date"])

            match = f[f["date"] == row["date"]]

            if len(match):
                print(match.to_string(index=False))
            else:
                print("No FNSPID record for this date.")

    # Inspect corporate actions for the sample tickers.
    print("\n\n" + "=" * 75)
    print("CORPORATE ACTIONS")
    print("=" * 75)

    if ACTIONS_FILE.exists():
        actions = pd.read_csv(ACTIONS_FILE)
        actions["date"] = pd.to_datetime(
            actions["date"], errors="coerce"
        )

        actions = actions[
            actions["ticker"].isin(SAMPLE_TICKERS)
            & (
                (actions["dividends"] != 0)
                | (actions["stock_splits"] != 0)
            )
        ]

        print(actions.to_string(index=False))
    else:
        print("Corporate actions file not found.")

    print("\nINSPECTION COMPLETE")

if __name__ == "__main__":
    main()
