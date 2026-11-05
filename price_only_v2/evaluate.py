"""Check 2021 rankings, uncertainty, and an illustrative long-only weekly rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from price_only_v2.features import read_development, split_mask


ROOT = Path("results/price_only_v2")


def bootstrap_interval(values: np.ndarray, seed: int = 42) -> list[float]:
    """Resample five-date blocks because five-session outcomes overlap."""
    blocks = [values[i:i+5] for i in range(0, len(values), 5)]
    generator = np.random.default_rng(seed)
    samples = []
    for _ in range(2000):
        chosen = generator.integers(0, len(blocks), len(blocks))
        samples.append(float(np.nanmean(np.concatenate([blocks[i] for i in chosen]))))
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def weekly_portfolio(group: pd.DataFrame, cost_bps_per_side: int) -> dict:
    days = sorted(group.prediction_date.unique())
    selected_days = days[::5]
    weekly = []
    spy = []
    equal = []
    trades = 0
    for day in selected_days:
        one = group.loc[group.prediction_date == day].sort_values(
            ["predicted_excess_return", "ticker"], ascending=[False, True])
        positive = one.loc[one.predicted_excess_return > 0].head(5)
        # Rank weights are fixed: 5/15, 4/15, ..., 1/15. Unused weight stays cash.
        weights = np.arange(5, 5-len(positive), -1)/15
        gross = float(np.dot(weights, positive.actual_company_return))
        cost = float(weights.sum()*2*cost_bps_per_side/10_000)
        weekly.append(gross-cost)
        trades += len(positive)
        spy.append(float(one.actual_spy_return.iloc[0]))
        equal.append(float(one.actual_company_return.mean()))
    return {"weeks": len(selected_days), "stock_positions_bought": trades,
            "cost_bps_per_side": cost_bps_per_side,
            "model_compounded_return": float(np.prod(1+np.array(weekly))-1),
            "spy_compounded_return": float(np.prod(1+np.array(spy))-1),
            "equal_group_compounded_return": float(np.prod(1+np.array(equal))-1),
            "model_mean_weekly_return": float(np.mean(weekly))}


def main() -> None:
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data_local/price_v2_original_50_2017_2023.csv"))
    parser.add_argument("--results-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    ROOT = args.results_dir
    files = [ROOT / "tabular/predictions_2021.csv", ROOT / "lstm/predictions_2021.csv"]
    frame = pd.concat([pd.read_csv(path, parse_dates=["prediction_date"]) for path in files], ignore_index=True)
    development = read_development(args.data)
    _, valid = split_mask(development, 2017, 2020, 2021)
    labels = development.loc[valid, ["ticker", "prediction_date", "target_excess_return",
                                      "target_company_return", "target_spy_return"]]
    frame = frame.merge(labels, on=["ticker", "prediction_date"], how="left", validate="many_to_one")
    frame = frame.rename(columns={"target_excess_return": "actual_excess_return",
                                  "target_company_return": "actual_company_return",
                                  "target_spy_return": "actual_spy_return"})
    if frame.actual_excess_return.isna().any():
        raise ValueError("Prediction without local label")
    if frame.duplicated(["model_name", "ticker", "prediction_date"]).any():
        raise ValueError("Duplicate company-date-model prediction")
    keys = None
    report = {"period": "2021 development validation only", "test_years_not_opened": [2022, 2023],
              "weekly_rule": "Every fifth prediction date, buy up to five positive-forecast companies with rank weights 5/15 to 1/15; unused weight stays cash; close all held shares at the end of their five-session outcome; no shorts or calls.",
              "models": {}}
    monthly_rows = []
    company_rows = []
    for name, model in frame.groupby("model_name", sort=True):
        model = model.sort_values(["prediction_date", "ticker"])
        if keys is None:
            keys = list(zip(model.ticker, model.prediction_date))
        elif keys != list(zip(model.ticker, model.prediction_date)):
            raise ValueError(f"Different validation company/date keys for {name}")
        daily = []
        for day, one in model.groupby("prediction_date", sort=True):
            rank = (float(spearmanr(one.predicted_excess_return, one.actual_excess_return).statistic)
                    if one.predicted_excess_return.nunique() > 1 else np.nan)
            ordered = one.sort_values("predicted_excess_return", ascending=False)
            top = float(ordered.actual_excess_return.head(5).mean()) if np.isfinite(rank) else np.nan
            middle = float(ordered.actual_excess_return.iloc[20:30].mean()) if np.isfinite(rank) else np.nan
            daily.append((day, rank, top, middle))
        daily_frame = pd.DataFrame(daily, columns=["date", "rank", "top5", "middle10"])
        daily_frame["month"] = daily_frame.date.dt.to_period("M").astype(str)
        monthly = daily_frame.groupby("month").agg(mean_rank=("rank", "mean")).reset_index()
        for _, row in monthly.iterrows():
            monthly_rows.append({"model_name": name, "month": row.month,
                                 "mean_rank": row.mean_rank})
        for ticker, one in model.groupby("ticker"):
            company_rows.append({"model_name": name, "ticker": ticker,
                "rows": len(one),
                "mae_pp": float(np.abs(one.actual_excess_return-one.predicted_excess_return).mean()*100),
                "sign_accuracy": float(((one.actual_excess_return>0)==(one.predicted_excess_return>0)).mean()),
                "time_rank_correlation": (float(spearmanr(one.predicted_excess_return, one.actual_excess_return).statistic)
                                          if one.predicted_excess_return.nunique()>1 else None)})
        spread = (daily_frame.top5-daily_frame.middle10).to_numpy()
        report["models"][name] = {
            "daily_rank_mean": (float(np.nanmean(daily_frame['rank'])) if daily_frame['rank'].notna().any() else None),
            "daily_rank_95pct_block_interval": (bootstrap_interval(daily_frame['rank'].to_numpy()) if daily_frame['rank'].notna().any() else None),
            "months_positive_rank": int((monthly.mean_rank > 0).sum()) if daily_frame['rank'].notna().any() else None,
            "top5_minus_middle10_mean_pp": (float(np.nanmean(spread)*100) if np.isfinite(spread).any() else None),
            "top5_minus_middle10_95pct_block_interval_pp": ([x*100 for x in bootstrap_interval(spread)] if np.isfinite(spread).any() else None),
            "positive_prediction_share": float((model.predicted_excess_return > 0).mean()),
            "weekly_cost_0bps": weekly_portfolio(model, 0),
            "weekly_cost_10bps": weekly_portfolio(model, 10),
            "weekly_cost_20bps": weekly_portfolio(model, 20),
        }
    pd.DataFrame(monthly_rows).to_csv(ROOT / "monthly_ranking_2021.csv", index=False)
    pd.DataFrame(company_rows).to_csv(ROOT / "by_company_2021.csv", index=False)
    (ROOT / "evaluation_2021.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, item in report["models"].items():
        print(f"{name:24s} rank {item['daily_rank_mean']!s:>7}  top-mid {item['top5_minus_middle10_mean_pp']!s:>8} pp  net weekly {item['weekly_cost_10bps']['model_compounded_return']:+.3f}")


if __name__ == "__main__":
    main()
