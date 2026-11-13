"""Score the selected study model from daily adjusted price data."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from price_refinement.experiment import pack_predictions
from price_refinement.study import MODEL, add_interactions, study_prediction
from price_refinement.study_features import project_panel

OHLCV = ["open", "high", "low", "close", "volume"]


def score_daily(prices, groups, bundle, start, end):
    """Require a panel of peer prices available through the last forecast close."""
    if "adjusted_close" in prices.columns:
        raise ValueError("Pass dividend- and split-adjusted OHLCV from the model price snapshot")
    table = prices[["ticker", "date", *OHLCV]].copy()
    table = table.assign(date=pd.to_datetime(table.date))
    table = table.loc[table.date <= end]
    expected = set(groups.index) | {"SPY"}
    if pd.Timestamp(end) < pd.Timestamp("2020-09-30"):
        expected.discard("PLTR")
    if table.duplicated(["ticker", "date"]).any() or set(table.ticker) != expected:
        raise ValueError("Daily input must have unique ticker/date keys for the project companies and SPY")
    if pd.Timestamp(start) <= bundle["last_training_exit"] or pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError("Requested dates must follow the last training outcome")
    calendar = mcal.get_calendar("NYSE").valid_days(table.date.min(), end).tz_localize(None)
    histories = {ticker: data.set_index("date").sort_index()[OHLCV] for ticker, data in table.groupby("ticker")}
    spy = histories.pop("SPY").reindex(calendar)
    if not np.isfinite(spy).all().all() or (spy[OHLCV[:4]] <= 0).any().any():
        raise ValueError("Missing or invalid SPY bars")
    frame = add_interactions(project_panel(histories, spy, groups.loc[list(histories)], with_target=False))
    frame = frame.loc[frame.prediction_date.between(start, end)]
    if frame.empty:
        raise ValueError("No eligible company windows in requested dates")
    features = frame[bundle["columns"]].fillna(0)
    values = study_prediction(bundle["model"], features)
    if not np.isfinite(values).all():
        raise ValueError("Invalid forecasts")
    return pack_predictions(frame, {"price_study_candidate": values})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Daily adjusted OHLCV, with columns ticker,date,open,high,low,close,volume")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--model", type=Path, default=MODEL / "candidate_2021.joblib")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    groups = pd.read_csv("CURRENT_COMPANIES.csv").set_index("ticker").sector
    scores = score_daily(pd.read_csv(args.input), groups, joblib.load(args.model), args.start, args.end)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output, index=False, float_format="%.12g")
