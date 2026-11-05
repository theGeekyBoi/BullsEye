
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[2]

FNSPID_DIR = ROOT / "raw_fnspid" / "prices" / "full_history"

SEARCH_DIRS = [
    ROOT / "prototype_yfinance",
    ROOT / "raw_yahoo",
    ROOT
]

TICKERS = [
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

def inspect_csv(path):
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames
            first = next(reader, None)

            if first is None:
                return columns, 0, None, None

            return columns, 1, first, None

    except Exception as e:
        return None, 0, None, str(e)


def main():
    print("=" * 65)
    print("HYBRID DATASET SOURCE INVENTORY")
    print("=" * 65)

    print("\nFNSPID DIRECTORY:")
    print(FNSPID_DIR)
    print("Exists:", FNSPID_DIR.exists())

    if FNSPID_DIR.exists():
        for ticker in TICKERS:
            path = FNSPID_DIR / f"{ticker}.csv"

            if ticker == "META":
                path = FNSPID_DIR / "FB.csv"

            print(f"{ticker:8} {'FOUND' if path.exists() else 'MISSING'}")

    print("\n" + "=" * 65)
    print("POTENTIAL YAHOO FINANCE FILES")
    print("=" * 65)

    seen = set()
    yahoo_files = []

    for folder in SEARCH_DIRS:
        if not folder.exists():
            continue

        for path in folder.glob("*.csv"):
            resolved = path.resolve()

            if resolved in seen:
                continue

            seen.add(resolved)

            # Ignore the FNSPID archive
            if FNSPID_DIR in path.parents:
                continue

            yahoo_files.append(path)

    if not yahoo_files:
        print("No CSV files found in the searched folders.")

    for path in sorted(yahoo_files):
        columns, count, first, error = inspect_csv(path)

        print(f"\nFILE: {path.relative_to(ROOT)}")

        if error:
            print("ERROR:", error)
            continue

        print("COLUMNS:", columns)

        if first:
            print("FIRST RECORD:", first)

    print("\n" + "=" * 65)
    print("INVENTORY COMPLETE")
    print("CSV files inspected:", len(yahoo_files))
    print("=" * 65)


if __name__ == "__main__":
    main()
