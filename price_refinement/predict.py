"""Generate the fixed price baselines and the saved Ridge candidate from input windows."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from price_refinement.experiment import MODEL, pack_predictions, predict_model, price_features


def predict(frame, model):
    columns = ["ticker", "prediction_date"]
    if frame.empty or frame.duplicated(columns).any():
        raise ValueError("Expected unique, nonempty company/date keys")
    frame = frame.sort_values(["prediction_date", "ticker"]).reset_index(drop=True)
    frame["prediction_date"] = pd.to_datetime(frame["prediction_date"])
    features, signals = price_features(frame)
    candidate = model["model"]
    if frame.prediction_date.min() <= model["last_training_exit"]:
        raise ValueError("Candidate was trained on outcomes too recent for these dates")
    if candidate["kind"] == "signal":
        offset, slope = candidate["calibration"]
        forecast = offset + slope * signals[candidate["feature"]]
    else:
        forecast = predict_model(candidate, features[candidate["feature"]])
    if not np.isfinite(forecast).all():
        raise ValueError("Nonfinite price prediction")
    return pack_predictions(frame, {
        "price_refinement_candidate": forecast,
        "price_trend_shrunk": 0.05 * signals["trend"],
        "price_zero": np.zeros(len(frame)),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Company-day price windows with no outcome columns needed")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=MODEL, help="Locally trained model bundle")
    args = parser.parse_args()
    result = predict(pd.read_csv(args.input), joblib.load(args.model))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, float_format="%.12g")
