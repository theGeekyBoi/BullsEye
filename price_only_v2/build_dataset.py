"""Build chronological 30-session examples for the original BullsEye universe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

FIELDS = ("Open", "High", "Low", "Close", "Volume")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def build(snapshot: Path, companies: Path, output: Path) -> dict:
    prices = pd.read_csv(snapshot, parse_dates=["date"])
    universe = pd.read_csv(companies)[["ticker", "sector"]]
    universe["ticker"] = universe["ticker"].str.upper()
    expected = set(universe.ticker)
    found = set(prices.ticker) - {"SPY"}
    if expected != found or len(expected) != 50:
        raise ValueError(f"Company list mismatch: missing={sorted(expected-found)}, extra={sorted(found-expected)}")
    if prices.duplicated(["ticker", "date"]).any():
        raise ValueError("Duplicate source ticker/date")
    calendar = pd.DatetimeIndex(mcal.get_calendar("NYSE").valid_days("2016-11-01", "2024-01-09").tz_localize(None))
    spy = prices.loc[prices.ticker == "SPY"].set_index("date").reindex(calendar)
    records = []
    skipped = {}
    indices = np.arange(29, len(calendar)-5)
    indices = indices[(calendar[indices].year >= 2017) & (calendar[indices].year <= 2023)]
    feature_columns = [
        f"{prefix}{field.lower()}_lag_{lag}"
        for lag in range(30, 0, -1)
        for prefix in ("", "spy_")
        for field in FIELDS
    ]
    market_values = spy[list(FIELDS)].to_numpy(dtype=np.float64)
    market_ok = np.isfinite(market_values).all(axis=1) & (market_values[:, :4] > 0).all(axis=1) & (market_values[:, 4] >= 0)
    for ticker, sector in universe.itertuples(index=False, name=None):
        company = prices.loc[prices.ticker == ticker].set_index("date").reindex(calendar)
        values = company[list(FIELDS)].to_numpy(dtype=np.float64)
        valid = np.isfinite(values).all(axis=1) & (values[:, :4] > 0).all(axis=1) & (values[:, 4] >= 0)
        history_valid = np.convolve(valid.astype(int), np.ones(30, dtype=int), "valid") == 30
        market_history_valid = np.convolve(market_ok.astype(int), np.ones(30, dtype=int), "valid") == 30
        good = (history_valid[indices-29] & market_history_valid[indices-29]
                & valid[indices+1] & valid[indices+5]
                & market_ok[indices+1] & market_ok[indices+5])
        chosen = indices[good]
        skipped[ticker] = int((~good).sum())
        if not len(chosen):
            continue
        company_window = np.stack([values[chosen-lag+1] for lag in range(30, 0, -1)], axis=1)
        spy_window = np.stack([market_values[chosen-lag+1] for lag in range(30, 0, -1)], axis=1)
        flat = np.stack((company_window, spy_window), axis=2).reshape(len(chosen), -1)
        one = pd.DataFrame(flat, columns=feature_columns)
        one.insert(0, "ticker", ticker)
        one.insert(1, "sector", sector)
        one.insert(2, "prediction_date", calendar[chosen])
        one.insert(3, "entry_date", calendar[chosen+1])
        one.insert(4, "exit_date", calendar[chosen+5])
        one["target_company_return"] = values[chosen+5, 3]/values[chosen+1, 0]-1
        one["target_spy_return"] = market_values[chosen+5, 3]/market_values[chosen+1, 0]-1
        one["target_excess_return"] = one["target_company_return"]-one["target_spy_return"]
        records.append(one)
    frame = pd.concat(records, ignore_index=True).sort_values(["prediction_date", "ticker"])
    if frame.duplicated(["ticker", "prediction_date"]).any():
        raise ValueError("Duplicate output company/date")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, float_format="%.12g")
    manifest = {"source_snapshot": str(snapshot), "source_sha256": sha256(snapshot),
                "companies_file": str(companies), "companies_sha256": sha256(companies),
                "output": str(output), "output_sha256": sha256(output),
                "rows": len(frame), "companies": len(expected),
                "rows_by_year": {str(k): int(v) for k, v in frame.groupby(frame.prediction_date.dt.year).size().items()},
                "earliest_date": str(frame.prediction_date.min().date()),
                "latest_date": str(frame.prediction_date.max().date()),
                "skipped_rows_by_ticker": skipped,
                "price_policy": "Yahoo auto-adjusted OHLC for 49 companies and SPY; EA adjusted FNSPID fallback; unmodified source volume",
                "test_policy": "2022-2023 labels are built but never loaded by development training/evaluation"}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Built {len(frame):,} examples for {len(expected)} companies: {manifest['earliest_date']} to {manifest['latest_date']}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("data_local/yahoo_adjusted_2016_2024.csv"))
    parser.add_argument("--companies", type=Path, default=Path("COMPANIES.csv"))
    parser.add_argument("--output", type=Path, default=Path("data_local/price_v2_original_50_2017_2023.csv"))
    args = parser.parse_args()
    build(args.snapshot, args.companies, args.output)


if __name__ == "__main__":
    main()
