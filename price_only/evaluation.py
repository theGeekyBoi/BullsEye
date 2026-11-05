"""Common company-day metrics for price, news, and future fusion predictions."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, spearmanr


def evaluate_predictions(
    keys: pd.DataFrame, actual: np.ndarray, predicted: np.ndarray
) -> dict[str, float | int | None]:
    """Average rank correlation by date; calculate remaining metrics by row."""
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if len(keys) != len(actual) or len(actual) != len(predicted):
        raise ValueError("Keys, actual values, and predictions must have equal lengths.")
    if not np.isfinite(actual).all():
        raise ValueError("Actual targets must be finite.")
    valid = np.isfinite(predicted)
    total = len(actual)
    if not valid.any():
        return {
            "rows": total,
            "predicted_rows": 0,
            "coverage": 0.0,
            "mae_percentage_points": None,
            "sign_accuracy": None,
            "mean_daily_spearman": None,
            "ranked_days": 0,
            "eligible_days": int(keys["prediction_date"].nunique()),
        }

    actual_valid = actual[valid]
    predicted_valid = predicted[valid]
    dates = pd.to_datetime(keys.loc[valid, "prediction_date"]).to_numpy()
    rank_values = []
    for date in np.unique(dates):
        same_day = dates == date
        if same_day.sum() < 2:
            continue
        if np.ptp(actual_valid[same_day]) == 0 or np.ptp(predicted_valid[same_day]) == 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConstantInputWarning)
            score = float(spearmanr(predicted_valid[same_day], actual_valid[same_day]).statistic)
        if np.isfinite(score):
            rank_values.append(score)

    return {
        "rows": total,
        "predicted_rows": int(valid.sum()),
        "coverage": float(valid.mean()),
        "mae_percentage_points": float(np.mean(np.abs(actual_valid - predicted_valid)) * 100),
        "sign_accuracy": float(np.mean((predicted_valid > 0) == (actual_valid > 0))),
        "mean_daily_spearman": float(np.mean(rank_values)) if rank_values else None,
        "ranked_days": len(rank_values),
        "eligible_days": int(keys["prediction_date"].nunique()),
    }
