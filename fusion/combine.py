"""Read the shared prediction format and combine price and news forecasts."""

import numpy as np
import pandas as pd

KEYS = ["ticker", "prediction_date"]
RETURN = "predicted_excess_return"
PREDICTION_COLUMNS = [*KEYS, "model_name", RETURN]
ELIGIBLE = "include_in_multimodal_comparison"
FUSION_MODEL = "fusion_equal_weight"


def development_rows(frame, universe, label):
    valid = pd.to_datetime(frame.prediction_date, format="%Y-%m-%d", errors="coerce")
    if not frame.prediction_date.eq(valid.dt.strftime("%Y-%m-%d")).all():
        raise ValueError(f"{label}: expected YYYY-MM-DD prediction dates")
    frame = frame.loc[valid.dt.year.between(2020, 2021)].copy()
    if not frame.ticker.isin(universe).all():
        raise ValueError(f"{label}: ticker outside the BullsEye universe")
    if frame.duplicated(KEYS).any():
        raise ValueError(f"{label}: duplicate ticker/date")
    return frame


def read_predictions(path, model, universe):
    frame = pd.read_csv(path, dtype={**dict.fromkeys(KEYS, str), "model_name": str})
    if set(frame.columns) != set(PREDICTION_COLUMNS):
        raise ValueError(f"{path}: expected exactly the four shared prediction columns")
    frame = development_rows(frame.loc[frame.model_name == model], universe, str(path))
    frame[RETURN] = pd.to_numeric(frame[RETURN], errors="raise")
    if frame.empty or not np.isfinite(frame[RETURN]).all():
        raise ValueError(f"{path}: {model!r} must have finite development predictions")
    return frame[PREDICTION_COLUMNS]


def read_coverage(path, flag, universe):
    columns = [*KEYS, flag]
    frame = pd.read_csv(path, usecols=columns, dtype=dict.fromkeys(columns, str))
    frame = development_rows(frame, universe, str(path))
    eligible = frame[flag].str.strip().str.lower().map({"true": True, "false": False, "1": True, "0": False})
    if eligible.isna().any():
        raise ValueError(f"{path}: {flag} must be true or false for every development row")
    return frame[KEYS].assign(news_eligible=eligible)


def combine(price, trend, news, coverage):
    """Preserve the price forecast grid and return the blend and audit columns."""
    price_grid = pd.MultiIndex.from_frame(price[KEYS])
    trend_grid = pd.MultiIndex.from_frame(trend[KEYS])
    if not price_grid.sort_values().equals(trend_grid.sort_values()):
        raise ValueError("Price and trend forecasts must cover the same company-dates")

    joined = price[KEYS + [RETURN]].rename(columns={RETURN: "price"})
    for inputs, column in ((trend, "trend"), (news, "news")):
        joined = joined.merge(inputs[KEYS + [RETURN]].rename(columns={RETURN: column}),
                              on=KEYS, how="left", validate="one_to_one")
    joined = joined.merge(coverage, on=KEYS, how="left", validate="one_to_one")
    has_coverage_row = joined.news_eligible.notna()
    eligible = joined.news_eligible.eq(True)
    has_news = joined.news.notna()
    joined["news_status"] = np.select(
        [~has_coverage_row, ~eligible, ~has_news],
        ["missing_coverage_row", "ineligible", "missing_news_forecast"],
        default="blended",
    )
    blended = joined.news_status.eq("blended")
    joined["combined"] = joined.price
    joined.loc[blended, "combined"] = .5 * joined.loc[blended, "price"] + .5 * joined.loc[blended, "news"]
    return joined.sort_values(["prediction_date", "ticker"]).reset_index(drop=True)


def export_predictions(joined):
    return joined.assign(model_name=FUSION_MODEL, predicted_excess_return=joined.combined)[PREDICTION_COLUMNS]
