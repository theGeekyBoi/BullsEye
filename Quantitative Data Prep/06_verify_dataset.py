from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

PRICE_FILE = ROOT / "prototype_yfinance" / "daily_prices.csv"
CALENDAR_FILE = ROOT / "trading_calendar.csv"
DATASET_FILE = ROOT / "final_dataset" / "quant_dataset_30day.csv"

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
TOLERANCE = 1e-9


def close_enough(actual, expected):
    return np.isclose(
        float(actual),
        float(expected),
        rtol=TOLERANCE,
        atol=TOLERANCE
    )


def main():
    print("Loading original prices, calendar, and final dataset...")

    prices = pd.read_csv(PRICE_FILE)
    calendar = pd.read_csv(CALENDAR_FILE)
    dataset = pd.read_csv(DATASET_FILE)

    prices["date"] = pd.to_datetime(prices["date"])
    calendar["date"] = pd.to_datetime(calendar["date"])
    dataset["prediction_date"] = pd.to_datetime(dataset["prediction_date"])
    dataset["entry_date"] = pd.to_datetime(dataset["entry_date"])
    dataset["exit_date"] = pd.to_datetime(dataset["exit_date"])

    prices["ticker"] = prices["ticker"].astype(str).str.upper()

    for col in PRICE_COLUMNS:
        prices[col] = pd.to_numeric(prices[col], errors="coerce")

    calendar_dates = pd.DatetimeIndex(
        sorted(calendar["date"].dropna().unique())
    )

    # Original source prices indexed by ticker and date.
    price_tables = {}

    for ticker in prices["ticker"].unique():
        price_tables[ticker] = (
            prices[prices["ticker"] == ticker]
            .set_index("date")
            .sort_index()
        )

    # Select representative examples.
    samples = []

    def add_sample(ticker, requested_date, description):
        requested_date = pd.Timestamp(requested_date)

        matches = dataset[
            (dataset["ticker"] == ticker)
            & (dataset["prediction_date"] == requested_date)
        ]

        if matches.empty:
            print(
                f"WARNING: Could not find {ticker} on "
                f"{requested_date.date()} ({description})."
            )
            return

        samples.append((matches.iloc[0], description))

    add_sample("AAPL", "2020-02-13", "First AAPL example")
    add_sample("MSFT", "2023-06-15", "Middle-period MSFT example")
    add_sample("NVDA", "2024-06-07", "NVDA split-period example")
    add_sample("PLTR", "2020-11-10", "First PLTR example")
    add_sample("ADBE", "2025-05-15", "Later-period ADBE example")

    if len(samples) == 0:
        raise ValueError("No sample examples could be found.")

    print(f"\nVerifying {len(samples)} examples...\n")

    for row, description in samples:
        ticker = row["ticker"]
        prediction_date = row["prediction_date"]

        print("=" * 70)
        print(description)
        print(f"Ticker: {ticker}")
        print(f"Prediction date: {prediction_date.date()}")

        # Find prediction date in the NYSE calendar.
        positions = np.where(calendar_dates == prediction_date)[0]

        if len(positions) != 1:
            raise ValueError(
                f"{ticker}: prediction date missing or duplicated "
                "in the calendar."
            )

        i = positions[0]

        if i < 29 or i + 5 >= len(calendar_dates):
            raise ValueError("Insufficient calendar history or future data.")

        expected_entry_date = calendar_dates[i + 1]
        expected_exit_date = calendar_dates[i + 5]

        assert row["entry_date"] == expected_entry_date
        assert row["exit_date"] == expected_exit_date

        company = price_tables[ticker]
        spy = price_tables["SPY"]

        # Verify all 150 historical input features.
        feature_checks = 0

        for lag in range(1, 31):
            historical_position = i - (lag - 1)
            historical_date = calendar_dates[historical_position]

            if historical_date not in company.index:
                raise ValueError(
                    f"{ticker}: missing original price on "
                    f"{historical_date.date()}."
                )

            original = company.loc[historical_date]

            for col in PRICE_COLUMNS:
                feature_name = f"{col.lower()}_lag_{lag}"
                actual = row[feature_name]
                expected = original[col]

                if not close_enough(actual, expected):
                    raise AssertionError(
                        f"Feature mismatch: {ticker}, "
                        f"{prediction_date.date()}, {feature_name}. "
                        f"Dataset={actual}, original={expected}"
                    )

                feature_checks += 1

        # Verify the forward target using the original source.
        company_entry = company.loc[expected_entry_date, "Open"]
        company_exit = company.loc[expected_exit_date, "Close"]

        spy_entry = spy.loc[expected_entry_date, "Open"]
        spy_exit = spy.loc[expected_exit_date, "Close"]

        expected_company_return = company_exit / company_entry - 1
        expected_spy_return = spy_exit / spy_entry - 1
        expected_excess_return = (
            expected_company_return - expected_spy_return
        )

        target_checks = {
            "target_company_return": expected_company_return,
            "target_spy_return": expected_spy_return,
            "target_excess_return": expected_excess_return
        }

        for target_name, expected in target_checks.items():
            actual = row[target_name]

            if not close_enough(actual, expected):
                raise AssertionError(
                    f"Target mismatch: {ticker}, "
                    f"{prediction_date.date()}, {target_name}. "
                    f"Dataset={actual}, recalculated={expected}"
                )

        print(f"Entry date verified: {expected_entry_date.date()}")
        print(f"Exit date verified:  {expected_exit_date.date()}")
        print(f"Historical features verified: {feature_checks}/150")

        print("\nTarget comparison:")
        print(
            f"Company return: dataset={row['target_company_return']:.10f} "
            f"| recalculated={expected_company_return:.10f}"
        )
        print(
            f"SPY return:     dataset={row['target_spy_return']:.10f} "
            f"| recalculated={expected_spy_return:.10f}"
        )
        print(
            f"Excess return:  dataset={row['target_excess_return']:.10f} "
            f"| recalculated={expected_excess_return:.10f}"
        )

        print("\nPASS: This example matches the original price data.")

    print("\n" + "=" * 70)
    print("ALL SAMPLE VERIFICATION CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()