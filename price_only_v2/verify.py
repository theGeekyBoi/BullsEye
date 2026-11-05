"""Independently verify saved 2021 development forecasts and source identity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from price_only.evaluation import evaluate_predictions
from price_only_v2.build_dataset import sha256
from price_only_v2.features import read_development, split_mask


def main() -> None:
    root = Path("results/price_only_v2")
    source = Path("data_local/price_v2_original_50_2017_2023.csv")
    data_hash = sha256(source)
    for name in ("tabular", "lstm"):
        manifest = json.loads((root / name / "run_manifest.json").read_text())
        if manifest["development_data_sha256"] != data_hash:
            raise ValueError(f"{name}: source hash differs from training run")
    frame = read_development(source)
    _, valid = split_mask(frame, 2017, 2020, 2021)
    expected = frame.loc[valid, ["ticker", "prediction_date", "target_excess_return"]].copy()
    expected["prediction_date"] = expected.prediction_date.dt.strftime("%Y-%m-%d")
    expected = expected.set_index(["ticker", "prediction_date"])
    original = set(pd.read_csv("COMPANIES.csv").ticker)
    if set(expected.index.get_level_values("ticker")) != original:
        raise ValueError("Validation universe is not original 50")
    checked = 0
    model_count = 0
    for family in ("tabular", "lstm"):
        pred = pd.read_csv(root / family / "predictions_2021.csv")
        saved = pd.read_csv(root / family / "metrics_2021.csv").set_index("model_name")
        if pred.duplicated(["model_name", "ticker", "prediction_date"]).any():
            raise ValueError("Duplicate prediction")
        if not (pred.prediction_date.str.startswith("2021-")).all():
            raise ValueError("Non-2021 prediction was saved")
        for name, one in pred.groupby("model_name"):
            one = one.set_index(["ticker", "prediction_date"]).sort_index()
            actual = expected.sort_index().target_excess_return
            if not one.index.equals(actual.index):
                raise ValueError(f"{name}: missing or extra company/date")
            if not np.isfinite(one.predicted_excess_return).all():
                raise ValueError(f"{name}: nonfinite prediction")
            metric = evaluate_predictions(one.reset_index()[["ticker", "prediction_date"]],
                                          actual.to_numpy(), one.predicted_excess_return.to_numpy())
            for key, value in metric.items():
                saved_value = saved.loc[name, key]
                if value is None and pd.isna(saved_value):
                    continue
                if not np.isclose(value, saved_value, rtol=0, atol=1e-8):
                    raise ValueError(f"{name}: saved {key} differs from recomputation")
            checked += len(one)
            model_count += 1
    print(f"Verified {checked:,} 2021 predictions across {model_count} models, exact original-50 keys, labels, metrics, and source hash")


if __name__ == "__main__":
    main()
