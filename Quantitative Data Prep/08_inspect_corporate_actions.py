from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

ACTION_FILE = ROOT / "prototype_yfinance" / "corporate_actions.csv"

actions = pd.read_csv(ACTION_FILE)

actions["date"] = pd.to_datetime(actions["date"])
actions["ticker"] = actions["ticker"].astype(str).str.upper()
actions["stock_splits"] = pd.to_numeric(
    actions["stock_splits"], errors="coerce"
)

splits = actions[
    actions["stock_splits"].notna()
    & (actions["stock_splits"] != 0)
    & (actions["stock_splits"] != 1)
].copy()

splits = splits.sort_values(["ticker", "date"])

print("\nALL RECORDED NON-UNIT SPLIT EVENTS")
print("=" * 75)

for ticker, group in splits.groupby("ticker"):
    print(f"\n{ticker}")

    for _, row in group.iterrows():
        print(
            f"  Date: {row['date'].date()} | "
            f"Ratio: {row['stock_splits']}"
        )

print("\n" + "=" * 75)
print("COMPLETE CORPORATE ACTION RECORDS")
print("=" * 75)

print(
    actions[
        actions["ticker"].isin(splits["ticker"].unique())
    ]
    .sort_values(["ticker", "date"])
    .to_string(index=False)
)