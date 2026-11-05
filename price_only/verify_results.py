"""Verify saved price-only validation predictions against the source labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from price_only.data import KEY_COLUMNS, load_development_rows, split_development_rows
from price_only.evaluation import evaluate_predictions


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("final_dataset/quant_dataset_30day.csv"))
    parser.add_argument("--results", type=Path, default=Path("results/price_only/dev_2020_2021_seed42"))
    args = parser.parse_args()

    manifest = json.loads((args.results / "run_manifest.json").read_text(encoding="utf-8"))
    saved_metrics = json.loads(
        (args.results / f"metrics_{manifest['validation_year']}.json").read_text(encoding="utf-8")
    )["models"]
    if sha256(args.data) != manifest["data_sha256"]:
        raise ValueError("Source dataset hash differs from the training run.")
    frame = load_development_rows(
        args.data, manifest["training_year"], manifest["validation_year"]
    )
    _, validation_mask = split_development_rows(
        frame, manifest["training_year"], manifest["validation_year"]
    )
    actual = frame.loc[
        validation_mask, [*KEY_COLUMNS, "target_excess_return"]
    ].copy()
    actual["prediction_date"] = actual["prediction_date"].dt.strftime("%Y-%m-%d")
    predictions = pd.read_csv(
        args.results / f"predictions_{manifest['validation_year']}.csv",
        dtype={"ticker": str, "prediction_date": str, "model_name": str},
    )
    expected_columns = {
        "ticker", "prediction_date", "model_name", "predicted_excess_return", "run_id"
    }
    if set(predictions.columns) != expected_columns:
        raise ValueError("Saved prediction columns do not match the agreed interface.")
    if predictions.duplicated(["ticker", "prediction_date", "model_name"]).any():
        raise ValueError("Duplicate saved model/company/date predictions.")
    if not (predictions["run_id"] == manifest["run_id"]).all():
        raise ValueError("Prediction run IDs differ from the manifest.")
    if set(predictions["model_name"]) != set(saved_metrics):
        raise ValueError("Saved model names differ from the metrics report.")
    if not np.isfinite(predictions["predicted_excess_return"]).all():
        raise ValueError("A saved prediction is non-finite.")

    for name, group in predictions.groupby("model_name", sort=True):
        merged = group.merge(actual, on=list(KEY_COLUMNS), how="left", validate="one_to_one")
        if len(merged) != len(actual) or merged["target_excess_return"].isna().any():
            raise ValueError(f"{name} lacks a prediction for a validation company-day.")
        measured = evaluate_predictions(
            merged.loc[:, list(KEY_COLUMNS)],
            merged["target_excess_return"].to_numpy(),
            merged["predicted_excess_return"].to_numpy(),
        )
        for key, saved in saved_metrics[name].items():
            computed = measured[key]
            if saved is None or computed is None:
                if saved != computed:
                    raise ValueError(f"{name} metric {key} differs: {saved} vs {computed}")
            elif not np.isclose(saved, computed, rtol=0, atol=1e-9):
                raise ValueError(f"{name} metric {key} differs: {saved} vs {computed}")

    print(
        f"Verified {len(predictions):,} predictions for {len(saved_metrics)} models "
        f"on {len(actual):,} validation company-days. All saved metrics match."
    )


if __name__ == "__main__":
    main()
