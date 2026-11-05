"""Tune simple price models on 2018-20 rolling folds; report 2021 once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from price_only.evaluation import evaluate_predictions
from price_only_v2.build_dataset import sha256
from price_only_v2.features import feature_matrix, read_development, split_mask, windows

ROUNDS = (10, 20, 40, 70, 110)


def score(frame: pd.DataFrame, mask: np.ndarray, truth: np.ndarray,
          prediction: np.ndarray) -> dict:
    keys = frame.loc[mask, ["ticker", "prediction_date"]].reset_index(drop=True)
    return evaluate_predictions(keys, truth[mask], prediction)


def xgb_fit(x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray,
            depth: int, rounds: int, seed: int) -> tuple[xgb.Booster, np.ndarray]:
    params = {"objective": "reg:squarederror", "tree_method": "hist", "max_depth": depth,
              "eta": 0.05, "min_child_weight": 20, "subsample": 0.8,
              "colsample_bytree": 0.8, "lambda": 5.0, "seed": seed, "nthread": 4}
    model = xgb.train(params, xgb.DMatrix(x_train, label=y_train*100),
                      num_boost_round=rounds, verbose_eval=False)
    return model, xgb.DMatrix(x_valid)


def tune(frame: pd.DataFrame, company: np.ndarray, market: np.ndarray,
         y: np.ndarray, output: Path, model_dir: Path, data_path: Path,
         companies_path: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    configs = []
    for lookback in (5, 10, 30):
        for alpha in (1.0, 10.0, 100.0):
            configs.append({"family": "ridge", "lookback": lookback,
                            "alpha": alpha, "market": True, "volume": False})
        for depth in (2, 3):
            configs.append({"family": "xgboost", "lookback": lookback,
                            "depth": depth, "market": True, "volume": False})
    # Predeclared ablations for market context and volume, using a short window.
    configs += [
        {"family": "xgboost", "lookback": 10, "depth": 2, "market": False, "volume": False},
        {"family": "xgboost", "lookback": 10, "depth": 2, "market": True, "volume": True},
    ]
    cache = {}
    for config in configs:
        key = (config["lookback"], config["market"], config["volume"])
        if key not in cache:
            cache[key] = feature_matrix(company, market, frame.sector, *key)
    folds = [(2017, 2018), (2018, 2019), (2019, 2020)]
    trials = []
    for j, config in enumerate(configs, 1):
        X, names = cache[(config["lookback"], config["market"], config["volume"])]
        fold_metrics = []
        best_rounds = []
        for train_end, valid_year in folds:
            tr, va = split_mask(frame, 2017, train_end, valid_year)
            if config["family"] == "ridge":
                model = make_pipeline(StandardScaler(), Ridge(alpha=config["alpha"]))
                model.fit(X[tr], y[tr]*100)
                pred = model.predict(X[va])/100
                fold_metrics.append(score(frame, va, y, pred))
            else:
                model, valid_matrix = xgb_fit(X[tr], y[tr], X[va], config["depth"], max(ROUNDS), 42)
                rounds_and_scores = []
                for count in ROUNDS:
                    pred = model.predict(valid_matrix, iteration_range=(0, count))/100
                    metric = score(frame, va, y, pred)
                    rounds_and_scores.append((count, metric))
                count, metric = max(rounds_and_scores, key=lambda x: (x[1]["mean_daily_spearman"], -x[1]["mae_percentage_points"]))
                best_rounds.append(count)
                fold_metrics.append(metric)
        trial = {**config, "fold_years": [2018, 2019, 2020],
                 "fold_ranking": [x["mean_daily_spearman"] for x in fold_metrics],
                 "fold_mae_pp": [x["mae_percentage_points"] for x in fold_metrics],
                 "mean_ranking": float(np.mean([x["mean_daily_spearman"] for x in fold_metrics])),
                 "mean_mae_pp": float(np.mean([x["mae_percentage_points"] for x in fold_metrics]))}
        if best_rounds:
            trial["best_rounds_by_fold"] = best_rounds
            trial["fixed_final_rounds"] = int(np.median(best_rounds))
        trials.append(trial)
        print(f"{j:02d}/{len(configs)} {config}: rolling rank {trial['mean_ranking']:+.4f}", flush=True)
    chosen = {}
    for family in ("ridge", "xgboost"):
        subset = [x for x in trials if x["family"] == family]
        chosen[family] = max(subset, key=lambda x: (x["mean_ranking"], -x["mean_mae_pp"]))
    tr, va = split_mask(frame, 2017, 2020, 2021)
    predictions = {
        "zero": np.zeros(int(va.sum())),
        "recent_trend_relative": (company[va, -1, 3]/company[va, -5, 0]
                                  - market[va, -1, 3]/market[va, -5, 0]).astype(np.float64),
    }
    settings = {}
    for family, selected in chosen.items():
        X, names = cache[(selected["lookback"], selected["market"], selected["volume"])]
        if family == "ridge":
            model = make_pipeline(StandardScaler(), Ridge(alpha=selected["alpha"]))
            model.fit(X[tr], y[tr]*100)
            predictions[family] = model.predict(X[va])/100
            joblib.dump(model, model_dir / "ridge.joblib")
        else:
            # A second seed measures training sensitivity. Settings and round
            # count were fixed from 2018-20, before looking at 2021 outcomes.
            for seed in (42, 123):
                model, valid_matrix = xgb_fit(X[tr], y[tr], X[va], selected["depth"], selected["fixed_final_rounds"], seed)
                name = f"xgboost_seed{seed}"
                predictions[name] = model.predict(valid_matrix)/100
                model.save_model(model_dir / f"{name}.json")
        settings[family] = {"selected_from": "2018-2020 rolling validation", **selected,
                            "feature_names": names}
    keys = frame.loc[va, ["ticker", "prediction_date"]].reset_index(drop=True)
    valid_y = y[va]
    rows = []
    metrics = {}
    for name, pred in predictions.items():
        item = keys.copy()
        item["model_name"] = name
        item["predicted_excess_return"] = pred
        rows.append(item)
        metrics[name] = evaluate_predictions(keys, valid_y, pred)
    pd.concat(rows, ignore_index=True).to_csv(output / "predictions_2021.csv", index=False, float_format="%.12g")
    pd.DataFrame.from_dict(metrics, orient="index").rename_axis("model_name").to_csv(output / "metrics_2021.csv")
    (output / "rolling_trials.json").write_text(json.dumps(trials, indent=2), encoding="utf-8")
    (output / "run_manifest.json").write_text(json.dumps({"chosen": settings, "seeds": [42,123],
        "development_data_sha256": sha256(data_path),
        "company_list_sha256": sha256(companies_path),
        "company_list_path": str(companies_path),
        "training_years": [2017,2018,2019,2020], "validation_year": 2021,
        "reserved_test_years": [2022,2023], "train_rows": int(tr.sum()),
        "validation_rows": int(va.sum()), "xgboost_version": xgb.__version__,
        "sklearn_version": sklearn.__version__}, indent=2), encoding="utf-8")
    print(pd.DataFrame.from_dict(metrics, orient="index").to_string(), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data_local/price_v2_original_50_2017_2023.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/price_only_v2/tabular"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/price_only_v2/tabular"))
    parser.add_argument("--companies", type=Path, default=Path("COMPANIES.csv"))
    args = parser.parse_args()
    frame = read_development(args.data)
    company, market = windows(frame)
    y = frame.target_excess_return.to_numpy(dtype=np.float64)
    tune(frame, company, market, y, args.output, args.model_dir, args.data, args.companies)


if __name__ == "__main__":
    main()
