"""Run saved price-only models on company-day inputs without outcome columns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from price_only_v2.features import feature_matrix, sequence_matrix, windows
from price_only_v2.train_lstm import LSTM

MODEL_NAMES = ("zero", "recent_trend_relative", "ridge", "xgboost_seed42",
               "xgboost_seed123", "lstm_ohlc_seed42", "lstm_ohlc_seed123",
               "lstm_ohlcv_seed42", "lstm_ohlcv_seed123")


def predict(frame: pd.DataFrame, names: list[str], model_dir: Path,
            results_dir: Path) -> pd.DataFrame:
    if frame.duplicated(["ticker", "prediction_date"]).any():
        raise ValueError("Duplicate company/date input")
    if not set(names).issubset(MODEL_NAMES):
        raise ValueError("Unknown model requested")
    company, market = windows(frame)
    tabular = json.loads((results_dir / "tabular/run_manifest.json").read_text())["chosen"]
    outputs = []
    for name in names:
        if name == "zero":
            values = np.zeros(len(frame))
        elif name == "recent_trend_relative":
            values = company[:, -1, 3]/company[:, -5, 0] - market[:, -1, 3]/market[:, -5, 0]
        elif name == "ridge":
            config = tabular["ridge"]
            features, _ = feature_matrix(company, market, frame.sector, config["lookback"],
                                         config["market"], config["volume"])
            values = joblib.load(model_dir / "tabular/ridge.joblib").predict(features)/100
        elif name.startswith("xgboost"):
            config = tabular["xgboost"]
            features, _ = feature_matrix(company, market, frame.sector, config["lookback"],
                                         config["market"], config["volume"])
            model = xgb.Booster()
            model.load_model(model_dir / "tabular" / f"{name}.json")
            values = model.predict(xgb.DMatrix(features))/100
        else:
            include_volume = "ohlcv" in name
            sequence = sequence_matrix(company, market, include_volume)
            with np.load(model_dir / "lstm" / f"{name}_standardizer.npz") as scaler:
                normalized = ((sequence - scaler["mean"])/scaler["std"]).astype(np.float32)
            model = LSTM(normalized.shape[2])
            model.load_state_dict(torch.load(model_dir / "lstm" / f"{name}.pt", map_location="cpu", weights_only=True))
            model.eval()
            with torch.no_grad():
                values = model(torch.from_numpy(normalized)).numpy()/100
        item = frame[["ticker", "prediction_date"]].copy()
        item["model_name"] = name
        item["predicted_excess_return"] = values
        outputs.append(item)
    return pd.concat(outputs, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES))
    parser.add_argument("--model-dir", type=Path, default=Path("models/price_only_v2"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/price_only_v2"))
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    result = predict(frame, args.models, args.model_dir, args.results_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, float_format="%.12g")
    print(f"Saved {len(result):,} predictions to {args.output}")


if __name__ == "__main__":
    main()
