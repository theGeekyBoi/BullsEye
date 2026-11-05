from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

PRICE_FILE = ROOT / "prototype_yfinance" / "daily_prices.csv"
CALENDAR_FILE = ROOT / "trading_calendar.csv"

OUTPUT_DIR = ROOT / "processed_fnspid"
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    prices = pd.read_csv(PRICE_FILE)
    calendar = pd.read_csv(CALENDAR_FILE)

    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce")

    prices["ticker"] = prices["ticker"].astype(str).str.upper()

    numeric = ["Open", "High", "Low", "Close", "Volume"]

    for col in numeric:
        prices[col] = pd.to_numeric(prices[col], errors="coerce")

    # Only validate dates through the last available price date.
    last_price_date = prices["date"].max()

    calendar_dates = sorted(
        calendar.loc[
            calendar["date"].notna()
            & (calendar["date"] <= last_price_date),
            "date"
        ].unique()
    )

    results = []

    for ticker, group in prices.groupby("ticker"):
        group = group.sort_values("date").copy()

        first_date = group["date"].min()
        last_date = group["date"].max()

        # Only expect sessions during this ticker's observed date range.
        expected_dates = {
            d for d in calendar_dates
            if first_date <= d <= last_date
        }

        actual_dates = set(group["date"].dropna())

        missing_dates = sorted(expected_dates - actual_dates)
        unexpected_dates = sorted(actual_dates - set(calendar_dates))

        finite = np.isfinite(
            group[["Open", "High", "Low", "Close"]]
        ).all(axis=1)

        positive = (
            (group["Open"] > 0)
            & (group["High"] > 0)
            & (group["Low"] > 0)
            & (group["Close"] > 0)
        )

        low_open_valid = group["Low"] <= group["Open"]
        low_close_valid = group["Low"] <= group["Close"]
        high_open_valid = group["High"] >= group["Open"]
        high_close_valid = group["High"] >= group["Close"]
        high_low_valid = group["High"] >= group["Low"]

        ohlc_valid = (
            low_open_valid
            & low_close_valid
            & high_open_valid
            & high_close_valid
            & high_low_valid
        )

        volume_valid = (
            np.isfinite(group["Volume"])
            & (group["Volume"] >= 0)
        )

        valid = finite & positive & ohlc_valid & volume_valid
        invalid_rows = group.loc[~valid].copy()

        results.append({
            "ticker": ticker,
            "rows": len(group),
            "expected_sessions": len(expected_dates),
            "missing_sessions": len(missing_dates),
            "unexpected_sessions": len(unexpected_dates),
            "duplicate_dates": int(group["date"].duplicated().sum()),
            "invalid_rows": int((~valid).sum()),
            "first_date": first_date,
            "last_date": last_date
        })

        if len(invalid_rows):
            print(f"\nINVALID RECORDS: {ticker}")

            for idx in invalid_rows.index:
                print(group.loc[idx, numeric + ["date"]].to_dict())
                print("finite:", bool(finite.loc[idx]))
                print("positive:", bool(positive.loc[idx]))
                print("low <= open:", bool(low_open_valid.loc[idx]))
                print("low <= close:", bool(low_close_valid.loc[idx]))
                print("high >= open:", bool(high_open_valid.loc[idx]))
                print("high >= close:", bool(high_close_valid.loc[idx]))
                print("high >= low:", bool(high_low_valid.loc[idx]))
                print("volume valid:", bool(volume_valid.loc[idx]))

        if missing_dates:
            print(f"\nMISSING DATES: {ticker}")
            print([
                pd.Timestamp(d).strftime("%Y-%m-%d")
                for d in missing_dates[:15]
            ])

        if unexpected_dates:
            print(f"\nUNEXPECTED DATES: {ticker}")
            print([
                pd.Timestamp(d).strftime("%Y-%m-%d")
                for d in unexpected_dates[:15]
            ])

    report = pd.DataFrame(results).sort_values("ticker")

    print("\n" + "=" * 75)
    print("CORRECTED NYSE CALENDAR VALIDATION SUMMARY")
    print("=" * 75)
    print(report.to_string(index=False))

    report_path = OUTPUT_DIR / "yahoo_calendar_validation.csv"
    report.to_csv(report_path, index=False)

    print("\nSaved:", report_path)
    print("Calendar sessions through last price date:", len(calendar_dates))
    print("Last available price date:", last_price_date)
    print("Unique Yahoo tickers:", prices["ticker"].nunique())
    print("Total Yahoo records:", len(prices))


if __name__ == "__main__":
    main()