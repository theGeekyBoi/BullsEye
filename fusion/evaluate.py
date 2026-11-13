"""Evaluate the blend and baselines on the same development company-days."""

import numpy as np
import pandas as pd

from fusion.combine import KEYS, development_rows
from price_only.evaluation import evaluate_predictions

TARGET = "target_excess_return"


def read_targets(path, universe):
    """Read target values only for 2020–2021 rows whose exits stay in the same year."""
    dates = pd.read_csv(path, usecols=["prediction_date", "exit_date"], dtype=str)
    prediction = pd.to_datetime(dates.prediction_date, format="%Y-%m-%d", errors="raise")
    exit_date = pd.to_datetime(dates.exit_date, format="%Y-%m-%d", errors="raise")
    relevant = prediction.dt.year.between(2020, 2021)
    if (relevant & (exit_date.isna() | (exit_date <= prediction))).any():
        raise ValueError("Development targets must have a valid exit after the forecast date")
    selected = relevant & prediction.dt.year.eq(exit_date.dt.year)
    included = set(np.flatnonzero(selected) + 1)
    targets = pd.read_csv(path, usecols=[*KEYS, TARGET], dtype=dict.fromkeys(KEYS, str),
                          skiprows=lambda i: i != 0 and i not in included)
    targets = development_rows(targets, universe, str(path))
    targets[TARGET] = pd.to_numeric(targets[TARGET], errors="raise")
    if targets.empty or not np.isfinite(targets[TARGET]).all():
        raise ValueError("Expected finite development targets")
    return targets, int((relevant & ~selected).sum())


def top_vs_middle(frame, prediction):
    """Return average daily top-five spread, giving tied positions equal credit."""
    spreads = []
    for _, day in frame.groupby("prediction_date"):
        if len(day) < 20 or day[prediction].nunique() < 2:
            continue
        tied_mean = day.groupby(prediction)[TARGET].transform("mean")
        actual = tied_mean.loc[day.sort_values(prediction, ascending=False).index]
        middle = (len(actual) - 10) // 2
        spreads.append(float((actual.iloc[:5].mean() - actual.iloc[middle:middle+10].mean()) * 100))
    return (float(np.mean(spreads)) if spreads else None), len(spreads)


def evaluate(joined, targets, same_as_trend):
    joined = joined.merge(targets, on=KEYS, how="outer", validate="one_to_one", indicator=True)
    if joined._merge.ne("both").any():
        counts = joined._merge.value_counts()
        raise ValueError("Price predictions must match the eligible outcome grid. "
                         f"Forecast only: {counts['left_only']}; outcome only: {counts['right_only']}")
    joined["zero"] = 0.0
    joined["year"] = joined.prediction_date.str[:4].astype(int)
    summary, coverage = [], []
    statuses = ["blended", "missing_coverage_row", "ineligible", "missing_news_forecast"]
    for year, all_rows in joined.groupby("year"):
        matched = all_rows.loc[all_rows.news_status.eq("blended")]
        counts = all_rows.news_status.value_counts()
        coverage.append({"year": year, "company_days": len(all_rows),
                         **{status: int(counts.get(status, 0)) for status in statuses},
                         "fallback_rate": 1 - len(matched)/len(all_rows)})
        baselines = ["price", "zero"] if same_as_trend else ["price", "trend", "zero"]
        for scope, rows, models in (
            ("matched_news", matched, [*baselines, "news", "combined"]),
            ("full_price_grid", all_rows, [*baselines, "combined"]),
        ):
            for model in models:
                metrics = evaluate_predictions(rows.reset_index(drop=True), rows[TARGET], rows[model])
                spread, spread_days = top_vs_middle(rows, model)
                summary.append({"year": year, "scope": scope, "model": model, **metrics,
                                "top5_minus_middle10_pp": spread, "top5_scored_days": spread_days})
    return pd.DataFrame(summary), pd.DataFrame(coverage)
