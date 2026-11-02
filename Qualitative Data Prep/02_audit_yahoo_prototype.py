
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PRICE_FILE = ROOT / "prototype_yfinance" / "daily_prices.csv"
OUTPUT_DIR = ROOT / "processed_fnspid"

OUTPUT_DIR.mkdir(exist_ok=True)

EXPECTED_TICKERS = [
    "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN",
    "AVGO", "META", "TSLA", "BRK-B", "LLY",
    "MU", "AMD", "JPM", "WMT", "V", "MA",
    "ORCL", "COST", "NFLX", "XOM", "JNJ",
    "PLTR", "ABBV", "CVX", "BAC", "CAT",
    "GE", "AXP", "KO", "CSCO", "CRM", "WFC",
    "LIN", "RTX", "MRK", "TMUS", "TMO", "DIS",
    "ABT", "ACN", "QCOM", "TXN", "MCD", "AMGN",
    "PEP", "ADBE", "HD", "IBM", "NOW", "GS",
    "SPY"
]

def main():
    print("=" * 70)
    print("YAHOO PROTOTYPE DAILY PRICE AUDIT")
    print("=" * 70)

    if not PRICE_FILE.exists():
        print("ERROR: File not found:", PRICE_FILE)
        return

    df = pd.read_csv(PRICE_FILE)

    required = [
        "date", "Open", "High", "Low",
        "Close", "Volume", "ticker"
    ]

    missing_columns = [
        col for col in required if col not in df.columns
    ]

    if missing_columns:
        print("ERROR: Missing columns:", missing_columns)
        print("Actual columns:", list(df.columns))
        return

    print("\nTotal rows:", len(df))
    print("Original columns:", list(df.columns))

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    numeric_columns = ["Open", "High", "Low", "Close", "Volume"]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print("\nDate parsing failures:", df["date"].isna().sum())

    print("\nDuplicate ticker/date records:",
          df.duplicated(["ticker", "date"]).sum())

    print("\nTicker coverage:")
    actual_tickers = set(df["ticker"].dropna())
    expected_tickers = set(EXPECTED_TICKERS)

    print("Expected tickers:", len(expected_tickers))
    print("Found tickers:", len(actual_tickers))

    print("\nMissing expected tickers:")
    for ticker in sorted(expected_tickers - actual_tickers):
        print(" ", ticker)

    print("\nUnexpected tickers:")
    for ticker in sorted(actual_tickers - expected_tickers):
        print(" ", ticker)

    print("\nPer-ticker summary:")

    summaries = []

    for ticker, group in df.groupby("ticker"):
        valid_dates = group["date"].dropna()

        summary = {
            "ticker": ticker,
            "rows": len(group),
            "first_date": (
                valid_dates.min().date()
                if len(valid_dates) else None
            ),
            "last_date": (
                valid_dates.max().date()
                if len(valid_dates) else None
            ),
            "duplicate_dates": group.duplicated("date").sum(),
            "missing_open": group["Open"].isna().sum(),
            "missing_high": group["High"].isna().sum(),
            "missing_low": group["Low"].isna().sum(),
            "missing_close": group["Close"].isna().sum(),
            "missing_volume": group["Volume"].isna().sum(),
        }

        # Basic OHLC and volume validation
        valid_ohlc = (
            group[["Open", "High", "Low", "Close"]].notna().all(axis=1)
            & (group["Open"] > 0)
            & (group["High"] > 0)
            & (group["Low"] > 0)
            & (group["Close"] > 0)
            & (group["Low"] <= group["Open"])
            & (group["Low"] <= group["Close"])
            & (group["High"] >= group["Open"])
            & (group["High"] >= group["Close"])
            & (group["High"] >= group["Low"])
        )

        summary["invalid_ohlc_rows"] = int((~valid_ohlc).sum())
        summary["invalid_volume_rows"] = int(
            (group["Volume"].isna() | (group["Volume"] < 0)).sum()
        )

        summaries.append(summary)

    report = pd.DataFrame(summaries).sort_values("ticker")

    print(report.to_string(index=False))

    report_path = OUTPUT_DIR / "yahoo_prototype_audit.csv"
    report.to_csv(report_path, index=False)

    print("\nSaved audit report:")
    print(report_path)

    # Check the prototype's price scale on a sample row.
    print("\nSample records:")
    print(
        df[df["ticker"].isin(["NVDA", "AAPL", "SPY"])]
        .sort_values(["ticker", "date"])
        .groupby("ticker")
        .head(2)
        .to_string(index=False)
    )

    print("\nAUDIT COMPLETE")

if __name__ == "__main__":
    main()
