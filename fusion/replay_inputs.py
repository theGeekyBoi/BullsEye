"""Prepare 2021 development inputs for the simulator on the quant_model branch."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fusion.combine import KEYS, RETURN, read_predictions
from fusion.run import UNIVERSE

PRICE_COLUMNS = ["ticker", "date", "next_trading_date", "adjusted_open", "adjusted_close"]


def export_replay_inputs(prediction_path, model, price_path, output, start="2021-01-01"):
    first_day = pd.to_datetime(start, format="%Y-%m-%d", errors="coerce")
    if pd.isna(first_day) or first_day.year != 2021 or first_day.strftime("%Y-%m-%d") != start:
        raise ValueError("Replay start must be a 2021 date in YYYY-MM-DD format")
    universe = set(pd.read_csv(UNIVERSE, usecols=["ticker"]).ticker)
    forecasts = read_predictions(prediction_path, model, universe)
    forecasts = forecasts.loc[forecasts.prediction_date.between(start, "2021-12-31")].copy()
    if forecasts.empty or not forecasts.groupby("prediction_date").ticker.nunique().eq(len(universe)).all():
        raise ValueError("Expected complete project rankings on each exported 2021 forecast date")

    prices = pd.read_csv(price_path, usecols=PRICE_COLUMNS, dtype={"date": str, "next_trading_date": str})
    prices = prices.loc[prices.date.str.startswith("2021-")].copy()
    if set(prices.ticker) != universe | {"SPY"} or prices.duplicated(["ticker", "date"]).any():
        raise ValueError("Expected unique daily prices for the project universe plus SPY")
    adjusted = prices[["adjusted_open", "adjusted_close"]]
    if not np.isfinite(adjusted).all().all() or (adjusted <= 0).any().any():
        raise ValueError("Adjusted prices must be positive and finite")

    sessions = prices.loc[prices.ticker.eq("SPY"), "date"].sort_values()
    session_next = dict(zip(sessions.iloc[:-1], sessions.iloc[1:]))
    marks = prices.set_index(["ticker", "date"]).sort_index()
    signal_index = pd.MultiIndex.from_frame(forecasts[KEYS])
    next_date = forecasts.prediction_date.map(session_next)
    marked_next = marks.next_trading_date.reindex(signal_index)
    if (next_date.isna().any()
            or not np.array_equal(marked_next.to_numpy(), next_date.to_numpy())
            or not pd.MultiIndex.from_arrays([forecasts.ticker, next_date]).isin(marks.index).all()):
        raise ValueError("Missing price or incorrect next-session open for a forecast")

    converted = forecasts.rename(columns={"prediction_date": "date", "model_name": "model"})
    converted["pred_diff_pp"] = converted[RETURN] * 100
    replay_prices = prices.rename(columns={"adjusted_open": "open", "adjusted_close": "close"})
    output.mkdir(parents=True, exist_ok=True)
    converted[["date", "ticker", "pred_diff_pp", "model"]].to_csv(
        output / "sim_predictions.csv", index=False)
    replay_prices[["date", "ticker", "open", "close"]].to_csv(
        output / "sim_daily_prices.csv", index=False)
    print(f"Wrote {len(converted):,} forecasts and {len(replay_prices):,} adjusted 2021 prices to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prices", type=Path, default=Path("data_local/daily_prices_replay.csv"))
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    export_replay_inputs(args.predictions, args.model, args.prices, args.output, args.start)
