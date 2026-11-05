"""Train price-only development models and save 2021 validation predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn as nn
import xgboost as xgb
from torch.utils.data import DataLoader, TensorDataset

from price_only.data import (
    KEY_COLUMNS,
    fit_standardizer,
    load_development_rows,
    ordered_windows,
    recent_trend_prediction,
    sequence_features,
    split_development_rows,
    tree_features,
)
from price_only.evaluation import evaluate_predictions


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(sequences)
        return self.head(hidden[-1]).squeeze(-1)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_xgboost(
    windows_train: np.ndarray,
    windows_validation: np.ndarray,
    targets_train_pp: np.ndarray,
    targets_validation_pp: np.ndarray,
    seed: int,
    threads: int,
    artifact_dir: Path,
) -> tuple[np.ndarray, dict, dict]:
    features_train, names = tree_features(windows_train)
    features_validation, _ = tree_features(windows_validation)
    train_matrix = xgb.DMatrix(features_train, label=targets_train_pp, feature_names=names)
    validation_matrix = xgb.DMatrix(
        features_validation, label=targets_validation_pp, feature_names=names
    )
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "tree_method": "hist",
        "max_depth": 4,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "lambda": 1.0,
        "seed": seed,
        "nthread": threads,
    }
    history: dict = {}
    model = xgb.train(
        params,
        train_matrix,
        num_boost_round=400,
        evals=[(train_matrix, "train"), (validation_matrix, "validation")],
        early_stopping_rounds=30,
        evals_result=history,
        verbose_eval=False,
    )
    best_iteration = int(model.best_iteration)
    predicted_pp = model.predict(
        validation_matrix, iteration_range=(0, best_iteration + 1)
    )
    model.save_model(artifact_dir / "xgboost.json")
    settings = {
        "params": params,
        "max_rounds": 400,
        "early_stopping_rounds": 30,
        "best_iteration": best_iteration,
        "feature_count": len(names),
        "feature_names": names,
        "target_unit_during_training": "percentage_points",
    }
    return predicted_pp.astype(np.float64) / 100.0, settings, history


def train_lstm(
    name: str,
    windows_train: np.ndarray,
    windows_validation: np.ndarray,
    targets_train_pp: np.ndarray,
    targets_validation_pp: np.ndarray,
    include_volume: bool,
    seed: int,
    threads: int,
    device: str,
    artifact_dir: Path,
    max_epochs: int,
) -> tuple[np.ndarray, dict, list[dict]]:
    seed_everything(seed, threads)
    raw_train = sequence_features(windows_train, include_volume)
    raw_validation = sequence_features(windows_validation, include_volume)
    mean, std = fit_standardizer(raw_train)
    train_sequences = ((raw_train - mean) / std).astype(np.float32)
    validation_sequences = ((raw_validation - mean) / std).astype(np.float32)
    np.savez(artifact_dir / f"{name}_standardizer.npz", mean=mean, std=std)

    train_dataset = TensorDataset(
        torch.from_numpy(train_sequences),
        torch.from_numpy(targets_train_pp.astype(np.float32)),
    )
    validation_dataset = TensorDataset(torch.from_numpy(validation_sequences))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation_dataset, batch_size=1024, shuffle=False)

    model = LSTMRegressor(input_size=train_sequences.shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_function = nn.SmoothL1Loss(beta=1.0)
    best_mae = float("inf")
    best_epoch = 0
    best_state = None
    patience = 4
    stale_epochs = 0
    history = []

    def validation_prediction() -> np.ndarray:
        model.eval()
        parts = []
        with torch.no_grad():
            for (inputs,) in validation_loader:
                parts.append(model(inputs.to(device)).cpu().numpy())
        return np.concatenate(parts)

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(inputs), targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(inputs)
        validation_pp = validation_prediction()
        mae_pp = float(np.mean(np.abs(validation_pp - targets_validation_pp)))
        history.append({
            "epoch": epoch,
            "train_huber_loss": total_loss / len(train_dataset),
            "validation_mae_percentage_points": mae_pp,
        })
        print(f"{name}: epoch {epoch:02d} | validation MAE {mae_pp:.4f} pp", flush=True)
        if mae_pp < best_mae - 1e-5:
            best_mae = mae_pp
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("LSTM training did not produce a model state.")
    model.load_state_dict(best_state)
    model.to(device)
    predicted_pp = validation_prediction()
    torch.save(best_state, artifact_dir / f"{name}.pt")
    settings = {
        "input_features": ["open", "high", "low", "close"] + (["volume"] if include_volume else []),
        "window_order": "lag_30_to_lag_1",
        "hidden_size": 32,
        "batch_size": 256,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "loss": "SmoothL1Loss(beta=1.0)",
        "max_epochs": max_epochs,
        "patience": patience,
        "best_epoch": best_epoch,
        "best_validation_mae_percentage_points": best_mae,
        "standardizer_fit_on": "training rows only",
        "target_unit_during_training": "percentage_points",
        "device": device,
    }
    return predicted_pp.astype(np.float64) / 100.0, settings, history


def choose_model(metrics: dict[str, dict]) -> str:
    """Use validation mean daily rank correlation, then MAE, to select a model."""
    ranked = [
        (name, value)
        for name, value in metrics.items()
        if value["mean_daily_spearman"] is not None
    ]
    if not ranked:
        raise ValueError("No model has a defined ranking score.")
    return max(
        ranked,
        key=lambda item: (
            item[1]["mean_daily_spearman"],
            -item[1]["mae_percentage_points"],
        ),
    )[0]


def validate_prediction_set(keys: pd.DataFrame, predictions: dict[str, np.ndarray]) -> None:
    if keys.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("Validation keys contain duplicate company-date pairs.")
    for name, values in predictions.items():
        if len(values) != len(keys):
            raise ValueError(f"{name} produced {len(values)} predictions for {len(keys)} rows.")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} produced a non-finite prediction.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("final_dataset/quant_dataset_30day.csv"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--train-year", type=int, default=2020)
    parser.add_argument("--validation-year", type=int, default=2021)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-epochs", type=int, default=18)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    if args.threads < 1 or args.max_epochs < 1:
        raise ValueError("Threads and maximum epochs must be positive.")
    run_id = f"dev_{args.train_year}_{args.validation_year}_seed{args.seed}"
    args.output = args.output or Path("results/price_only") / run_id
    args.model_dir = args.model_dir or Path("models/price_only") / run_id
    seed_everything(args.seed, args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    print("Loading development-year examples...", flush=True)
    frame = load_development_rows(args.data, args.train_year, args.validation_year)
    train_mask, validation_mask = split_development_rows(
        frame, args.train_year, args.validation_year
    )
    windows = ordered_windows(frame)
    windows_train = windows[train_mask]
    windows_validation = windows[validation_mask]
    targets_train = frame.loc[train_mask, "target_excess_return"].to_numpy(dtype=np.float32)
    targets_validation = frame.loc[validation_mask, "target_excess_return"].to_numpy(dtype=np.float64)
    keys_validation = frame.loc[validation_mask, list(KEY_COLUMNS)].reset_index(drop=True)
    print(
        f"Training rows: {len(windows_train):,}; validation rows: {len(windows_validation):,}",
        flush=True,
    )

    predictions = {
        "zero": np.zeros(len(windows_validation), dtype=np.float64),
        "recent_trend": recent_trend_prediction(windows_validation),
    }
    model_settings: dict[str, dict] = {
        "zero": {"description": "Constant zero excess-return prediction."},
        "recent_trend": {
            "description": "Last five completed sessions' stock return; no SPY history is in the input CSV.",
            "formula": "close_lag_1 / open_lag_5 - 1",
        },
    }
    training_history: dict = {}

    print("Training XGBoost...", flush=True)
    predictions["xgboost"], model_settings["xgboost"], training_history["xgboost"] = train_xgboost(
        windows_train,
        windows_validation,
        targets_train * 100,
        targets_validation * 100,
        args.seed,
        args.threads,
        args.model_dir,
    )

    for name, include_volume in (("lstm_ohlc", False), ("lstm_ohlcv", True)):
        print(f"Training {name}...", flush=True)
        predictions[name], model_settings[name], training_history[name] = train_lstm(
            name,
            windows_train,
            windows_validation,
            targets_train * 100,
            targets_validation * 100,
            include_volume,
            args.seed,
            args.threads,
            args.device,
            args.model_dir,
            args.max_epochs,
        )

    validate_prediction_set(keys_validation, predictions)
    metrics = {
        name: evaluate_predictions(keys_validation, targets_validation, prediction)
        for name, prediction in predictions.items()
    }
    selected = choose_model(metrics)
    prediction_rows = []
    for name, values in predictions.items():
        result = keys_validation.copy()
        result["model_name"] = name
        result["predicted_excess_return"] = values
        result["run_id"] = run_id
        prediction_rows.append(result)
    output_predictions = pd.concat(prediction_rows, ignore_index=True)
    output_predictions.to_csv(
        args.output / f"predictions_{args.validation_year}.csv",
        index=False,
        float_format="%.12g",
    )
    pd.DataFrame.from_dict(metrics, orient="index").rename_axis("model_name").to_csv(
        args.output / f"metrics_{args.validation_year}.csv"
    )
    (args.output / f"metrics_{args.validation_year}.json").write_text(
        json.dumps({"selected_model": selected, "models": metrics}, indent=2), encoding="utf-8"
    )
    (args.output / "training_history.json").write_text(
        json.dumps(training_history, indent=2), encoding="utf-8"
    )

    manifest = {
        "run_id": run_id,
        "data_file": str(args.data).replace("\\", "/"),
        "data_sha256": file_sha256(args.data),
        "training_year": args.train_year,
        "validation_year": args.validation_year,
        "split_rule": "Prediction and exit must be in the same assigned calendar year; later years are not read into model inputs.",
        "training_rows": int(train_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "training_date_min": str(frame.loc[train_mask, "prediction_date"].min().date()),
        "training_date_max": str(frame.loc[train_mask, "prediction_date"].max().date()),
        "validation_date_min": str(frame.loc[validation_mask, "prediction_date"].min().date()),
        "validation_date_max": str(frame.loc[validation_mask, "prediction_date"].max().date()),
        "excluded_boundary_rows": int(len(frame) - train_mask.sum() - validation_mask.sum()),
        "input_window": "30 sessions, lag_30 oldest through lag_1 prediction date",
        "saved_prediction_unit": "decimal excess return (0.01 = 1 percentage point)",
        "target_column": "target_excess_return",
        "seed": args.seed,
        "threads": args.threads,
        "device": args.device,
        "model_selection": "Highest 2021 mean daily Spearman; lower MAE breaks ties.",
        "selected_model": selected,
        "models": model_settings,
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "xgboost": xgb.__version__,
        },
    }
    (args.output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("\n2021 validation results:", flush=True)
    print(pd.DataFrame.from_dict(metrics, orient="index").to_string(), flush=True)
    print(f"Selected by ranking metric: {selected}", flush=True)
    print(f"Saved predictions and metrics to {args.output}", flush=True)
    print(f"Saved model weights locally to {args.model_dir} (gitignored)", flush=True)


if __name__ == "__main__":
    main()
