from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

PRICE_FILE = ROOT / "prototype_yfinance" / "daily_prices.csv"
ACTIONS_FILE = ROOT / "prototype_yfinance" / "corporate_actions.csv"

OUTPUT_DIR = ROOT / "processed_fnspid"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "split_volume_audit.csv"


def main():
    prices = pd.read_csv(PRICE_FILE)
    actions = pd.read_csv(ACTIONS_FILE)

    prices["date"] = pd.to_datetime(prices["date"])
    actions["date"] = pd.to_datetime(actions["date"])

    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    actions["ticker"] = actions["ticker"].astype(str).str.upper()

    prices["Volume"] = pd.to_numeric(
        prices["Volume"], errors="coerce"
    )

    actions["stock_splits"] = pd.to_numeric(
        actions["stock_splits"], errors="coerce"
    )

    splits = actions[
        actions["stock_splits"].notna()
        & (actions["stock_splits"] != 0)
        & (actions["stock_splits"] != 1)
    ].copy()

    results = []

    print("Auditing historical volume around stock splits...")
    print(f"Split events found: {len(splits)}")

    for _, split in splits.iterrows():
        ticker = split["ticker"]
        split_date = split["date"]
        split_ratio = split["stock_splits"]

        company = (
            prices[prices["ticker"] == ticker]
            .sort_values("date")
            .reset_index(drop=True)
        )

        # Find the first available trading session on or after
        # the corporate-action date.
        positions = np.where(
            company["date"].to_numpy() >= np.datetime64(split_date)
        )[0]

        if len(positions) == 0:
            continue

        i = positions[0]

        # Five trading sessions before and after the event.
        window = company.iloc[
            max(0, i - 5):min(len(company), i + 6)
        ].copy()

        print("\n" + "=" * 75)
        print(f"{ticker} | Split date: {split_date.date()}")
        print(f"Split ratio recorded: {split_ratio}")

        print(
            window[
                ["date", "Open", "Close", "Volume"]
            ].to_string(index=False)
        )

        # Compare median volume before and after the event.
        before = company.iloc[max(0, i - 5):i]
        after = company.iloc[i:min(len(company), i + 5)]

        median_before = before["Volume"].median()
        median_after = after["Volume"].median()

        volume_ratio = (
            median_after / median_before
            if pd.notna(median_before) and median_before != 0
            else np.nan
        )

        results.append({
            "ticker": ticker,
            "split_date": split_date,
            "split_ratio": split_ratio,
            "median_volume_before": median_before,
            "median_volume_after": median_after,
            "after_to_before_volume_ratio": volume_ratio,
            "before_observations": before["Volume"].notna().sum(),
            "after_observations": after["Volume"].notna().sum()
        })

        print(f"Median volume before: {median_before:,.0f}")
        print(f"Median volume after:  {median_after:,.0f}")
        print(f"After/before ratio:   {volume_ratio:.4f}")

    report = pd.DataFrame(results)
    report.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 75)
    print("SPLIT VOLUME AUDIT COMPLETE")
    print("=" * 75)
    print(f"Events audited: {len(report)}")
    print(f"Report saved: {OUTPUT_FILE}")

    if not report.empty:
        print("\nSummary:")
        print(report.to_string(index=False))


if __name__ == "__main__":
    main()
