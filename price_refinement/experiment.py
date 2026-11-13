"""Run chronological price-only experiments for the current BullsEye universe."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from price_only.evaluation import evaluate_predictions
from price_only_v2.build_dataset import sha256
from price_only_v2.features import split_mask, windows
from price_refinement.features import make_features

DATA = Path("data_local/price_current_50_2017_2023.csv")
OUT = Path("results/price_refinement")
MODEL = Path("models/price_refinement/candidate_2021.joblib")
COLUMNS = ["ticker", "prediction_date", "model_name", "predicted_excess_return"]
TICKERS = Path("final_dataset/dataset_summary.csv")
CONFIGS = (
    ("ridge_base", "ridge", "base", 42),
    ("ridge_peer", "ridge", "peer", 42),
    ("ridge_peer_volume", "ridge", "peer_volume", 42),
    ("huber_peer", "huber", "peer", 42),
    ("ridge_recency", "ridge_recency", "peer", 42),
    ("tree_peer", "tree", "peer", 42),
    ("tree_peer_seed123", "tree", "peer", 123),
    ("rank_peer", "rank", "peer", 42),
    ("rank_peer_seed123", "rank", "peer", 123),
    ("ridge_monthly_1y", "monthly_1y", "base", 42),
    ("ridge_monthly_2y", "monthly_2y", "base", 42),
    ("ridge_rank_target", "ridge_rank_target", "base", 42),
)


def calibrate(score, target):
    centered = score - score.mean()
    slope = max(float(np.dot(centered, target - target.mean())
                      / max(np.dot(centered, centered), 1e-16)), 1e-8)
    return float(target.mean() - slope * score.mean()), slope


def fit_model(kind, x, y, keys, seed):
    if kind in ("ridge", "ridge_recency", "huber"):
        estimator = HuberRegressor(max_iter=500) if kind == "huber" else Ridge(alpha=100)
        model = make_pipeline(StandardScaler(), estimator)
        fit_args = {}
        if kind == "ridge_recency":
            age = (keys.prediction_date.max() - keys.prediction_date).dt.days
            fit_args["ridge__sample_weight"] = np.power(0.5, age / 365.25)
        model.fit(x, y * 100, **fit_args)
        return {"kind": "linear", "model": model}
    params = {"tree_method": "hist", "max_depth": 2, "eta": 0.05, "min_child_weight": 20,
              "subsample": 0.8, "colsample_bytree": 0.8, "lambda": 5,
              "seed": seed, "nthread": 4, "objective": "reg:squarederror"}
    if kind == "tree":
        return {"kind": "tree", "model": xgb.train(params, xgb.DMatrix(x, label=y*100), 50)}
    # Reserve the last 63 sessions within the training period to calibrate the rank score.
    cutoff = keys.prediction_date.drop_duplicates().iloc[-63]
    train = keys.exit_date < cutoff
    calibration = keys.prediction_date >= cutoff
    ranks = pd.Series(y[train], index=keys.index[train]).groupby(keys.prediction_date[train]).rank(pct=True)
    if kind == "ridge_rank_target":
        model = make_pipeline(StandardScaler(), Ridge(alpha=100))
        model.fit(x.loc[train], ranks - 0.5)
        scores = model.predict(x.loc[calibration])
        return {"kind": "linear_rank", "model": model, "calibration": calibrate(scores, y[calibration])}
    matrix = xgb.DMatrix(x.loc[train], label=np.minimum(np.floor(ranks * 5), 4))
    matrix.set_group(keys.loc[train].groupby("prediction_date", sort=False).size().to_numpy())
    params.update(objective="rank:pairwise", lambdarank_pair_method="mean", lambdarank_num_pair_per_sample=8)
    model = xgb.train(params, matrix, 50)
    scores = model.predict(xgb.DMatrix(x.loc[calibration]))
    return {"kind": "rank", "model": model, "calibration": calibrate(scores, y[calibration])}


def predict_model(bundle, x):
    kind, model = bundle["kind"], bundle["model"]
    if kind == "linear":
        return model.predict(x) / 100
    pred = (model.predict(x) if kind == "linear_rank"
            else model.predict(xgb.DMatrix(x))).astype("float64")
    if kind in ("rank", "linear_rank"):
        offset, slope = bundle["calibration"]
        return offset + slope * pred
    return pred / 100


def price_features(frame):
    return make_features(frame, *windows(frame))


def read_research(data):
    manifest = json.loads(data.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    nrows = sum(manifest["rows_by_year"][str(year)] for year in range(2017, 2022))
    dates = pd.read_csv(data, nrows=nrows, usecols=["prediction_date", "exit_date"])
    complete = dates.exit_date < "2022-01-01"
    if (not dates.prediction_date.is_monotonic_increasing
            or dates.prediction_date.min()[:4] != "2017"
            or dates.prediction_date.max()[:4] != "2021"
            or not complete.is_monotonic_decreasing):
        raise ValueError("Development rows must be sorted with all cross-boundary outcomes at the end")
    frame = pd.read_csv(data, nrows=int(complete.sum()), parse_dates=["prediction_date", "exit_date"])
    if frame.duplicated(["prediction_date", "ticker"]).any():
        raise ValueError("Duplicate company/date in development data")
    return frame.sort_values(["prediction_date", "ticker"]).reset_index(drop=True)


def pack_predictions(keys, predictions):
    return pd.concat([keys[["ticker", "prediction_date"]].assign(
        model_name=name, predicted_excess_return=values)
        for name, values in predictions.items()], ignore_index=True)[COLUMNS]


def monthly_ridge(frame, x, target, valid, lookback_years):
    """Refit at each month's first close using only already completed targets."""
    forecast = pd.Series(index=frame.index[valid], dtype=float)
    dates = frame.loc[valid, "prediction_date"]
    for _, month in dates.groupby(dates.dt.to_period("M")):
        cutoff = month.min()
        train = ((frame.prediction_date >= cutoff - pd.DateOffset(years=lookback_years))
                 & (frame.exit_date < cutoff))
        bundle = fit_model("ridge", x.loc[train], target[train], frame.loc[train], 42)
        forecast.loc[month.index] = predict_model(bundle, x.loc[month.index])
    return forecast.to_numpy()


def fit_fold(frame, features, simple, year, configs):
    train, valid = split_mask(frame, 2017, year - 1, year)
    target = frame.target_excess_return.to_numpy()
    predictions = {"zero": np.zeros(valid.sum()), "trend_raw": simple["trend"][valid],
                   "trend_shrunk": 0.05 * simple["trend"][valid]}
    models = {"trend_shrunk": {"kind": "signal", "feature": "trend", "calibration": (0.0, 0.05)}}
    for name, signal in simple.items():
        offset, slope = calibrate(signal[train], target[train])
        predictions[name] = offset + slope * signal[valid]
        models[name] = {"kind": "signal", "feature": name, "calibration": (offset, slope)}
    for name, kind, feature_set, seed in configs:
        x = features[feature_set]
        if kind.startswith("monthly_"):
            predictions[name] = monthly_ridge(frame, x, target, valid, int(kind[-2]))
            continue
        bundle = fit_model(kind, x.loc[train], target[train], frame.loc[train], seed)
        predictions[name] = predict_model(bundle, x.loc[valid])
        models[name] = dict(bundle, feature=feature_set, seed=seed)
    keys = frame.loc[valid].reset_index(drop=True)
    metrics = {name: evaluate_predictions(keys, target[valid], pred)
               for name, pred in predictions.items()}
    return keys, predictions, models, metrics


def select_candidate(folds, years):
    """Require mean improvement over the raw trend before selecting a challenger."""
    names = [name for name in folds[years[0]][3] if name not in ("zero", "trend_raw")]
    rows = {name: [folds[year][3][name]["mean_daily_spearman"] for year in years] for name in names}
    baseline = np.mean([folds[year][3]["trend_raw"]["mean_daily_spearman"] for year in years])
    # Alternate seeds are sensitivity checks, not extra selection opportunities.
    eligible = [name for name, scores in rows.items()
                if not name.endswith("seed123") and not name.startswith("ridge_monthly")
                and all(score is not None for score in scores)]
    chosen = max(eligible, key=lambda name: np.mean(rows[name]))
    return chosen if np.mean(rows[chosen]) > baseline else "trend_shrunk"


def paired_interval(values):
    """Moving block bootstrap over 10 adjacent sessions."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(42)
    block_size = 10
    starts = rng.integers(0, len(values)-block_size+1, size=(2000, int(np.ceil(len(values)/block_size))))
    draws = (starts[:, :, None] + np.arange(block_size)).reshape(2000, -1)[:, :len(values)]
    return [float(q) for q in np.quantile(values[draws].mean(axis=1), [.025, .975])]


def diagnose(keys, predictions):
    detail = keys[["ticker", "sector", "prediction_date", "target_excess_return"]].copy()
    daily, by_company, by_sector = [], [], []
    for name, pred in predictions.items():
        detail["prediction"] = pred
        detail["abs_error_pp"] = np.abs(pred - detail.target_excess_return) * 100
        for day, group in detail.groupby("prediction_date", sort=True):
            ranked = group.prediction.nunique() > 1
            ordered = group.sort_values(["prediction", "ticker"], ascending=[False, True])
            actual = ordered.target_excess_return
            middle = (len(group) - 10) // 2
            daily.append({"model_name": name, "prediction_date": day,
                          "rank": spearmanr(group.prediction, group.target_excess_return).statistic if ranked else np.nan,
                          "top5_minus_middle10_pp": (actual.iloc[:5].mean()-actual.iloc[middle:middle+10].mean())*100 if ranked else np.nan,
                          "prediction_spread_pp": group.prediction.std() * 100,
                          "realized_spread_pp": group.target_excess_return.std() * 100,
                          "positive_share": (group.prediction > 0).mean()})
        for field, output in (("ticker", by_company), ("sector", by_sector)):
            for key, group in detail.groupby(field):
                output.append({"model_name": name, field: key, "mae_pp": group.abs_error_pp.mean(),
                               "forecast_mean_pp": group.prediction.mean()*100,
                               "actual_mean_pp": group.target_excess_return.mean()*100,
                               "rows": len(group)})
    return pd.DataFrame(daily), pd.DataFrame(by_company), pd.DataFrame(by_sector)


def run(data, output, model_path):
    frame = read_research(data)
    universe = set(pd.read_csv(TICKERS).ticker)
    if len(universe) != 50 or set(frame.ticker) != universe:
        raise ValueError("Input must contain the current 50 BullsEye tickers")
    features, simple = price_features(frame)
    folds = {}
    selections = {}
    for year in (2018, 2019, 2020, 2021):
        if year >= 2020:
            selections[year] = select_candidate(folds, list(range(2018, year)))
        folds[year] = fit_fold(frame, features, simple, year, CONFIGS)
        print(f"Completed {year}: {selections.get(year, 'model selection fold')}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = []
    for year, (_, _, _, scores) in folds.items():
        metrics.extend({"year": year, "model_name": name, **score} for name, score in scores.items())
    pd.DataFrame(metrics).to_csv(output / "fold_metrics.csv", index=False)
    selected_name = "price_refinement_candidate"
    handoff = []
    for year, selected in selections.items():
        keys, preds, models, _ = folds[year]
        if year == 2021:
            train, _ = split_mask(frame, 2017, 2020, 2021)
            joblib.dump({"model": models[selected], "name": selected,
                         "last_training_exit": frame.loc[train, "exit_date"].max()}, model_path)
            pack_predictions(keys, preds).to_csv(model_path.parent / "candidate_comparison_2021.csv", index=False)
        handoff.append(pack_predictions(keys, {selected_name: preds[selected],
                                               "price_trend_shrunk": preds["trend_shrunk"],
                                               "price_zero": preds["zero"]}))
    pd.concat(handoff, ignore_index=True).to_csv(output / "predictions_2020_2021.csv", index=False)

    keys, predictions, _, _ = folds[2021]
    selected = selections[2021]
    prior = pd.read_csv("results/price_only_current/tabular/predictions_2021.csv",
                        parse_dates=["prediction_date"])
    prior_lstm = pd.read_csv("results/price_only_current/lstm/predictions_2021.csv",
                             parse_dates=["prediction_date"])
    prior = pd.concat([prior, prior_lstm], ignore_index=True).pivot(
        index=["prediction_date", "ticker"], columns="model_name", values="predicted_excess_return")
    joined = keys.join(prior, on=["prediction_date", "ticker"], validate="one_to_one")
    if joined[prior.columns].isna().any().any():
        raise ValueError("Old predictions and 2021 refinement keys differ")
    daily, by_company, by_sector = diagnose(keys, {
        **{f"prior_{col}": joined[col].to_numpy() for col in prior.columns},
        **predictions,
    })
    daily["month"] = daily.prediction_date.dt.to_period("M").astype(str)
    daily.groupby(["model_name", "month"], as_index=False)[
        ["rank", "top5_minus_middle10_pp", "prediction_spread_pp", "realized_spread_pp", "positive_share"]
    ].mean().to_csv(output / "monthly_diagnostics_2021.csv", index=False)
    by_company.to_csv(output / "company_diagnostics_2021.csv", index=False)
    by_sector.to_csv(output / "sector_diagnostics_2021.csv", index=False)
    actual = keys.target_excess_return.to_numpy()
    daily_rank = daily.pivot(index="prediction_date", columns="model_name", values="rank")
    delta = daily_rank[selected] - daily_rank["trend_raw"]
    spread = daily.loc[daily.model_name == selected, "top5_minus_middle10_pp"]
    mae_delta = pd.Series(100*(np.abs(predictions[selected]-actual)-np.abs(actual))).groupby(
        keys.prediction_date).mean()
    manifest = {
        "data_sha256": sha256(data), "universe_sha256": sha256(TICKERS),
        "versions": {"pandas": pd.__version__, "scikit_learn": sklearn.__version__, "xgboost": xgb.__version__},
        "development_rows": len(frame), "selection_by_prediction_year": selections,
        "configuration": CONFIGS, "fit_through_year": {"2020": 2019, "2021": 2020},
        "selection": "Highest average daily rank correlation over earlier folds only. Seed 123 and monthly refits are diagnostics. Use 5% scaled trend if the best fixed trial does not beat raw trend.",
        "validation_2021": {
            "selected": folds[2021][3][selected],
            "trend": folds[2021][3]["trend_raw"],
            "shrunk_trend": folds[2021][3]["trend_shrunk"],
            "trend_rank_95pct_interval": paired_interval(daily_rank["trend_raw"]),
            "zero": folds[2021][3]["zero"],
            "paired_daily_rank_delta_vs_trend": delta.mean(),
            "paired_rank_delta_95pct_interval": paired_interval(delta),
            "top5_minus_middle10_pp": spread.mean(),
            "top5_minus_middle10_95pct_interval_pp": paired_interval(spread),
            "paired_mae_delta_vs_zero_pp": mae_delta.mean(),
            "paired_mae_delta_95pct_interval_pp": paired_interval(mae_delta),
        },
        "reserved_outcomes_loaded": False,
    }
    (output / "run.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(manifest["validation_2021"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--model", type=Path, default=MODEL)
    args = parser.parse_args()
    run(args.data, args.output, args.model)
