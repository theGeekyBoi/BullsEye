
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

PRICE_FILE = ROOT / "prototype_yfinance" / "daily_prices.csv"
CALENDAR_FILE = ROOT / "trading_calendar.csv"

OUTPUT_DIR = ROOT / "final_dataset"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "quant_dataset_30day.csv"
REPORT_FILE = OUTPUT_DIR / "dataset_summary.csv"

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def main():
    print("Loading price data and NYSE calendar...")

    prices = pd.read_csv(PRICE_FILE)
    calendar = pd.read_csv(CALENDAR_FILE)

    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce")
    prices["ticker"] = prices["ticker"].astype(str).str.upper()

    for col in PRICE_COLUMNS:
        prices[col] = pd.to_numeric(prices[col], errors="coerce")

    # Restrict the calendar to dates for which prices exist.
    last_price_date = prices["date"].max()

    calendar_dates = pd.DatetimeIndex(
        sorted(
            calendar.loc[
                calendar["date"].notna()
                & (calendar["date"] <= last_price_date),
                "date"
            ].unique()
        )
    )

    if prices.duplicated(["ticker", "date"]).any():
        raise ValueError("Duplicate ticker/date records found.")

    # SPY is used as the benchmark, not as a prediction ticker.
    if "SPY" not in prices["ticker"].unique():
        raise ValueError("SPY data is missing.")

    spy = (
        prices[prices["ticker"] == "SPY"]
        .set_index("date")
        .reindex(calendar_dates)
    )

    all_rows = []
    summary = []

    company_tickers = sorted(
        ticker for ticker in prices["ticker"].unique()
        if ticker != "SPY"
    )

    print(f"Found {len(company_tickers)} company tickers.")
    print(f"Calendar sessions: {len(calendar_dates)}")

    # Process each company independently.
    for ticker in company_tickers:

        company = (
            prices[prices["ticker"] == ticker]
            .set_index("date")
            .reindex(calendar_dates)
        )

        ticker_rows = []
        skipped_history = 0
        skipped_target = 0

        # i is prediction session t.
        # We need 29 earlier sessions and 5 future sessions.
        for i in range(29, len(calendar_dates) - 5):

            prediction_date = calendar_dates[i]
            entry_date = calendar_dates[i + 1]
            exit_date = calendar_dates[i + 5]

            # The 30 input sessions end at prediction day t.
            historical = company.iloc[i - 29:i + 1]

            if historical[PRICE_COLUMNS].isna().any().any():
                skipped_history += 1
                continue

            # Company forward prices.
            company_entry_open = company.iloc[i + 1]["Open"]
            company_exit_close = company.iloc[i + 5]["Close"]

            # SPY forward prices on exactly the same dates.
            spy_entry_open = spy.iloc[i + 1]["Open"]
            spy_exit_close = spy.iloc[i + 5]["Close"]

            target_values = [
                company_entry_open,
                company_exit_close,
                spy_entry_open,
                spy_exit_close
            ]

            if not np.isfinite(target_values).all():
                skipped_target += 1
                continue

            if min(target_values) <= 0:
                skipped_target += 1
                continue

            company_return = (
                company_exit_close / company_entry_open
            ) - 1

            spy_return = (
                spy_exit_close / spy_entry_open
            ) - 1

            excess_return = company_return - spy_return

            row = {
                "ticker": ticker,
                "prediction_date": prediction_date,
                "entry_date": entry_date,
                "exit_date": exit_date
            }

            # lag_1 = prediction date t.
            # lag_30 = oldest historical session.
            for lag in range(30, 0, -1):

                historical_position = i - (lag - 1)
                historical_row = company.iloc[historical_position]

                for col in PRICE_COLUMNS:
                    row[f"{col.lower()}_lag_{lag}"] = (
                        historical_row[col]
                    )

            # Target information is stored separately from features.
            row["target_company_return"] = company_return
            row["target_spy_return"] = spy_return
            row["target_excess_return"] = excess_return

            ticker_rows.append(row)

        all_rows.extend(ticker_rows)

        summary.append({
            "ticker": ticker,
            "usable_examples": len(ticker_rows),
            "skipped_missing_history": skipped_history,
            "skipped_missing_target": skipped_target,
            "first_prediction_date": (
                ticker_rows[0]["prediction_date"]
                if ticker_rows else None
            ),
            "last_prediction_date": (
                ticker_rows[-1]["prediction_date"]
                if ticker_rows else None
            )
        })

        print(
            f"{ticker}: {len(ticker_rows)} examples | "
            f"history skipped: {skipped_history} | "
            f"target skipped: {skipped_target}"
        )

    # Convert rows to a DataFrame.
    dataset = pd.DataFrame(all_rows)

    if dataset.empty:
        raise ValueError("No training examples were generated.")

    dataset = dataset.sort_values(
        ["prediction_date", "ticker"]
    ).reset_index(drop=True)

    # Confirm unique ticker/date combinations.
    if dataset.duplicated(["ticker", "prediction_date"]).any():
        raise ValueError("Duplicate dataset examples found.")

    # Confirm no missing feature or target values.
    feature_columns = [
        f"{col.lower()}_lag_{lag}"
        for lag in range(1, 31)
        for col in PRICE_COLUMNS
    ]

    required_columns = (
        feature_columns
        + [
            "target_company_return",
            "target_spy_return",
            "target_excess_return"
        ]
    )

    if dataset[required_columns].isna().any().any():
        raise ValueError("Missing values found in final dataset.")

    # Save final dataset and per-ticker summary.
    dataset.to_csv(OUTPUT_FILE, index=False)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(REPORT_FILE, index=False)

    print("\n" + "=" * 70)
    print("30-SESSION DATASET BUILD COMPLETE")
    print("=" * 70)

    print("Total examples:", len(dataset))
    print("Company tickers:", dataset["ticker"].nunique())
    print("Input features:", len(feature_columns))
    print("Target columns: 3")
    print("Total columns:", len(dataset.columns))

    print("\nExamples per ticker:")
    print(summary_df.to_string(index=False))

    print("\nDataset columns:")
    print("Metadata: ticker, prediction_date, entry_date, exit_date")
    print("Features:", len(feature_columns))
    print("Targets: target_company_return, target_spy_return, "
          "target_excess_return")

    print("\nSaved dataset:", OUTPUT_FILE)
    print("Saved summary:", REPORT_FILE)

    print("\nFirst 5 examples:")
    print(
        dataset[
            [
                "ticker",
                "prediction_date",
                "entry_date",
                "exit_date",
                "target_company_return",
                "target_spy_return",
                "target_excess_return"
            ]
        ].head().to_string(index=False)
    )


if __name__ == "__main__":
    main()
