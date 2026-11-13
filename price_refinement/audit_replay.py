"""Check replay inputs against existing development labels and ordinary splits."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from price_refinement.daily_prices import OUTPUT
from price_refinement.experiment import DATA, OUT, read_research


def audit(data, replay):
    prices = pd.read_csv(replay, parse_dates=["date", "next_trading_date"])
    prices = prices.loc[prices.date.dt.year <= 2021]
    keys = read_research(data)
    keys = keys.loc[keys.prediction_date.dt.year.between(2020, 2021)
                    & (keys.prediction_date.dt.year == keys.exit_date.dt.year)].copy()
    keys["entry_date"] = pd.to_datetime(keys.entry_date)
    index = prices.set_index(["ticker", "date"])
    def series(tickers, dates, field):
        return index[field].reindex(pd.MultiIndex.from_arrays([tickers, dates])).to_numpy()
    next_date = series(keys.ticker, keys.prediction_date, "next_trading_date")
    if not np.array_equal(next_date, keys.entry_date.to_numpy()):
        raise ValueError("Replay calendar differs from saved next-open entry dates")
    stock = (series(keys.ticker, keys.exit_date, "adjusted_close")
             / series(keys.ticker, keys.entry_date, "adjusted_open") - 1)
    benchmark = np.repeat("SPY", len(keys))
    spy = (series(benchmark, keys.exit_date, "adjusted_close")
           / series(benchmark, keys.entry_date, "adjusted_open") - 1)
    delta = np.abs(stock - spy - keys.target_excess_return.to_numpy()) * 100
    if not np.isfinite(delta).all() or np.max(delta) > .03:
        raise ValueError("Replay snapshot does not reproduce development returns to 0.03 pp")
    prices = prices.sort_values(["ticker", "date"])
    previous_close = prices.groupby("ticker").close.shift()
    splits = prices.loc[(prices.split_ratio > 0) & ~prices.non_standard_action,
                       ["ticker", "date", "split_ratio"]].copy()
    splits["open_over_previous_close"] = prices.loc[splits.index, "open"] / previous_close[splits.index]
    if not splits.open_over_previous_close.between(.65, 1.35).all():
        raise ValueError("Unexpected discontinuity at ordinary split")
    splits.date = splits.date.dt.strftime("%Y-%m-%d")
    return {
        "development_rows_checked": len(keys),
        "entry_date_mismatches": 0,
        "max_abs_excess_return_delta_pp": float(delta.max()),
        "median_abs_excess_return_delta_pp": float(np.median(delta)),
        "ordinary_splits_checked": splits.to_dict(orient="records"),
        "reserved_outcomes_compared": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--replay", type=Path, default=OUTPUT)
    parser.add_argument("--output", type=Path, default=OUT / "replay_audit.json")
    args = parser.parse_args()
    result = audit(args.data, args.replay)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
