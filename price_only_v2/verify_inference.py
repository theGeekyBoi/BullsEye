"""Check that saved model weights reproduce shareable validation forecasts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from price_only_v2.features import read_development
from price_only_v2.predict import MODEL_NAMES, predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data_local/price_v2_original_50_2017_2023.csv"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/price_only_v2"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/price_only_v2"))
    args = parser.parse_args()
    frame = read_development(args.data)
    one = frame.loc[(frame.prediction_date == "2021-06-01") & (frame.ticker == "AMD")].copy()
    one["prediction_date"] = one.prediction_date.dt.strftime("%Y-%m-%d")
    one = one.drop(columns=[column for column in one.columns if column.startswith("target_") or column in ("entry_date", "exit_date")])
    got = predict(one, list(MODEL_NAMES), args.model_dir, args.results_dir)
    prior = pd.concat([pd.read_csv(args.results_dir / group / "predictions_2021.csv")
                       for group in ("tabular", "lstm")], ignore_index=True)
    expected = prior.loc[(prior.prediction_date == "2021-06-01") & (prior.ticker == "AMD")]
    comparison = got.merge(expected, on=["ticker", "prediction_date", "model_name"], validate="one_to_one")
    if len(comparison) != len(MODEL_NAMES):
        raise ValueError("Missing model in inference check")
    delta = np.abs(comparison.predicted_excess_return_x-comparison.predicted_excess_return_y)
    if delta.max() > 1e-7:
        raise ValueError(f"Saved models fail to reproduce predictions: maximum difference {delta.max()}")
    print(f"All {len(comparison)} saved models reproduced their AMD 2021-06-01 forecast, maximum difference {delta.max():.2g}")


if __name__ == "__main__":
    main()
