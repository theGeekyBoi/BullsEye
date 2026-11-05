"""Check the news-covered price rebuild against the existing 2020-21 grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from price_only_v2.build_dataset import sha256


KEY = ["ticker", "prediction_date"]
OUTCOMES = ["target_company_return", "target_spy_return", "target_excess_return"]


def development_rows(path: Path) -> pd.DataFrame:
    columns = KEY + ["entry_date", "exit_date", "open_lag_30", "close_lag_1", "volume_lag_1"] + OUTCOMES
    # Determine the cutoff using only dates, then load outcome columns only
    # through 2021. Both inputs are sorted by prediction date.
    count = 0
    for dates in pd.read_csv(path, usecols=["prediction_date"], chunksize=10_000):
        before_2022 = dates.prediction_date < "2022-01-01"
        count += int(before_2022.sum())
        if not before_2022.all():
            break
    result = pd.read_csv(path, usecols=columns, nrows=count)
    result = result.loc[result.prediction_date >= "2020-01-01"].copy()
    if result.duplicated(KEY).any():
        raise ValueError(f"Duplicate company-day key in {path}")
    if not np.isfinite(result[OUTCOMES + ["open_lag_30", "close_lag_1", "volume_lag_1"]]).all().all():
        raise ValueError(f"Nonfinite price or outcome in {path}")
    if not np.allclose(result.target_company_return-result.target_spy_return,
                       result.target_excess_return, rtol=0, atol=1e-10):
        raise ValueError(f"Inconsistent excess-return arithmetic in {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prototype", type=Path, default=Path("final_dataset/quant_dataset_30day.csv"))
    parser.add_argument("--rebuilt", type=Path, default=Path("data_local/price_current_50_2017_2023.csv"))
    parser.add_argument("--companies", type=Path, default=Path("CURRENT_COMPANIES.csv"))
    parser.add_argument("--summary", type=Path, default=Path("final_dataset/dataset_summary.csv"))
    parser.add_argument("--snapshot", type=Path, default=Path("data_local/yahoo_adjusted_current_2016_2024.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/price_only_current/data_audit.json"))
    args = parser.parse_args()

    actual = set(pd.read_csv(args.summary).ticker)
    chosen = set(pd.read_csv(args.companies).ticker)
    if len(actual) != 50 or chosen != actual:
        raise ValueError("Current company list does not match final_dataset summary")
    old = development_rows(args.prototype)
    new = development_rows(args.rebuilt)
    if set(old.ticker) != actual or set(new.ticker) != actual:
        raise ValueError("Development price data do not contain exact current 50")
    both = old.merge(new, on=KEY, how="inner", suffixes=("_prototype", "_rebuilt"), validate="one_to_one")
    if not len(both):
        raise ValueError("No common company-date keys")
    differences = {}
    for field in OUTCOMES:
        delta = (both[f"{field}_prototype"]-both[f"{field}_rebuilt"]).abs()
        differences[field] = {"max_abs_decimal": float(delta.max()),
                              "rows_over_0.0001": int((delta > 0.0001).sum()),
                              "rows_over_0.001": int((delta > 0.001).sum())}
    if (both.entry_date_prototype != both.entry_date_rebuilt).any() or \
            (both.exit_date_prototype != both.exit_date_rebuilt).any():
        raise ValueError("Entry or exit date differs on a shared key")
    prices = pd.read_csv(args.snapshot)
    invalid = ~np.isfinite(prices[["Open", "High", "Low", "Close", "Volume"]]).all(axis=1)
    invalid |= (prices[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
    invalid |= prices.Volume < 0
    if invalid.any():
        raise ValueError(f"Invalid daily source OHLCV rows: {int(invalid.sum())}")
    split_rows = prices.loc[(prices["Stock Splits"] > 0) &
                            prices.date.between("2020-01-01", "2021-12-31")]
    split_checks = []
    for ticker, day in [("AAPL", "2020-08-31"), ("TSLA", "2020-08-31"),
                        ("NVDA", "2021-07-20"), ("GE", "2021-08-02")]:
        one = prices.loc[prices.ticker == ticker].sort_values("date").reset_index(drop=True)
        idx = one.index[one.date == day]
        if len(idx) != 1 or int(idx[0]) == 0:
            continue
        before, after = one.iloc[int(idx[0])-1], one.iloc[int(idx[0])]
        split_checks.append({"ticker": ticker, "date": day,
                             "reported_split_ratio": float(after["Stock Splits"]),
                             "adjusted_close_to_open_change": float(after.Open/before.Close-1),
                             "volume_before": float(before.Volume),
                             "volume_after": float(after.Volume)})
    report = {"current_company_count": len(actual),
              "current_company_list_sha256": sha256(args.companies),
              "prototype_development_rows": len(old),
              "rebuilt_development_rows": len(new),
              "common_2020_2021_keys": len(both),
              "prototype_only_keys": len(old)-len(both),
              "rebuilt_only_keys": len(new)-len(both),
              "outcome_differences": differences,
              "invalid_daily_source_rows": int(invalid.sum()),
              "reported_split_events_2020_2021": len(split_rows),
              "selected_split_checks": split_checks,
              "volume_policy": "Yahoo-reported Volume retained unchanged; models with and without volume are compared. Event samples do not fully establish historical volume adjustment semantics.",
              "reserved_test_outcomes_read": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Audited {len(both):,} shared 2020-21 keys; maximum target difference "
          f"{differences['target_excess_return']['max_abs_decimal']:.6g}; "
          f"{len(split_checks)} selected split checks", flush=True)


if __name__ == "__main__":
    main()
