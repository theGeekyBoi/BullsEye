"""Train small OHLC and OHLCV LSTMs with market context and earlier-year selection."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from price_only.evaluation import evaluate_predictions
from price_only_v2.build_dataset import sha256
from price_only_v2.features import read_development, sequence_matrix, split_mask, windows


class LSTM(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.encoder = nn.LSTM(channels, 24, batch_first=True)
        self.head = nn.Sequential(nn.Linear(24, 12), nn.ReLU(), nn.Linear(12, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(x)
        return self.head(hidden[-1]).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)


def fit(sequences: np.ndarray, targets: np.ndarray, train: np.ndarray,
        valid: np.ndarray, dates: pd.DataFrame, seed: int, max_epochs: int,
        choose_epoch: bool) -> tuple[np.ndarray, int, list[dict], dict, tuple[np.ndarray, np.ndarray]]:
    set_seed(seed)
    mean = sequences[train].mean(axis=(0, 1), keepdims=True)
    std = np.maximum(sequences[train].std(axis=(0, 1), keepdims=True), 1e-6)
    train_x = torch.from_numpy(((sequences[train]-mean)/std).astype(np.float32))
    valid_x = torch.from_numpy(((sequences[valid]-mean)/std).astype(np.float32))
    train_y = torch.from_numpy((targets[train]*100).astype(np.float32))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=512,
                        shuffle=True, generator=generator)
    valid_loader = DataLoader(TensorDataset(valid_x), batch_size=1024)
    model = LSTM(sequences.shape[2])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss = nn.SmoothL1Loss(beta=1.0)
    best_rank, best_epoch, best_state, stale = -np.inf, 0, None, 0
    history = []
    valid_keys = dates.loc[valid, ["ticker", "prediction_date"]].reset_index(drop=True)
    for epoch in range(1, max_epochs+1):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            error = loss(model(x), y)
            error.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(error.item())*len(x)
        model.eval()
        with torch.no_grad():
            pred = np.concatenate([model(x).numpy() for (x,) in valid_loader])/100
        metric = evaluate_predictions(valid_keys, targets[valid], pred)
        history.append({"epoch": epoch, "train_loss": total_loss/len(train_x),
                        "rank": metric["mean_daily_spearman"], "mae_pp": metric["mae_percentage_points"]})
        if choose_epoch and metric["mean_daily_spearman"] > best_rank + 0.0001:
            best_rank, best_epoch, stale = metric["mean_daily_spearman"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        elif choose_epoch:
            stale += 1
            if stale >= 3:
                break
    if choose_epoch:
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pred = np.concatenate([model(x).numpy() for (x,) in valid_loader])/100
    else:
        best_epoch = max_epochs
    return pred, best_epoch, history, model.state_dict(), (mean, std)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data_local/price_v2_original_50_2017_2023.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/price_only_v2/lstm"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/price_only_v2/lstm"))
    parser.add_argument("--companies", type=Path, default=Path("COMPANIES.csv"))
    parser.add_argument("--max-tuning-epochs", type=int, default=10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    frame = read_development(args.data)
    company, market = windows(frame)
    targets = frame.target_excess_return.to_numpy(dtype=np.float64)
    inner_train, inner_valid = split_mask(frame, 2017, 2019, 2020)
    final_train, final_valid = split_mask(frame, 2017, 2020, 2021)
    predictions, metrics, runs = [], {}, {}
    keys = frame.loc[final_valid, ["ticker", "prediction_date"]].reset_index(drop=True)
    for include_volume in (False, True):
        name = "ohlcv" if include_volume else "ohlc"
        sequences = sequence_matrix(company, market, include_volume)
        _, best_epoch, history, _, _ = fit(sequences, targets, inner_train, inner_valid,
                                            frame, 42, args.max_tuning_epochs, True)
        print(f"{name}: best 2020 rank checkpoint epoch {best_epoch}", flush=True)
        for seed in (42, 123):
            pred, _, final_history, state, scaler = fit(
                sequences, targets, final_train, final_valid, frame,
                seed, best_epoch, False)
            model_name = f"lstm_{name}_seed{seed}"
            torch.save(state, args.model_dir / f"{model_name}.pt")
            np.savez(args.model_dir / f"{model_name}_standardizer.npz", mean=scaler[0], std=scaler[1])
            metric = evaluate_predictions(keys, targets[final_valid], pred)
            metrics[model_name] = metric
            result = keys.copy()
            result["model_name"] = model_name
            result["predicted_excess_return"] = pred
            predictions.append(result)
            runs[model_name] = {"seed": seed, "include_volume": include_volume,
                                "market_context": True, "selected_epoch_on": 2020,
                                "selected_epoch": best_epoch, "tuning_history": history,
                                "final_training_history": final_history}
            print(f"{model_name}: 2021 rank {metric['mean_daily_spearman']:+.4f}, MAE {metric['mae_percentage_points']:.4f} pp", flush=True)
    pd.concat(predictions, ignore_index=True).to_csv(args.output / "predictions_2021.csv", index=False, float_format="%.12g")
    pd.DataFrame.from_dict(metrics, orient="index").rename_axis("model_name").to_csv(args.output / "metrics_2021.csv")
    (args.output / "run_manifest.json").write_text(json.dumps({
        "development_data_sha256": sha256(args.data),
        "company_list_sha256": sha256(args.companies),
        "company_list_path": str(args.companies),
        "training_years": [2017, 2018, 2019, 2020], "checkpoint_year": 2020,
        "validation_year": 2021, "reserved_test_years": [2022, 2023],
        "runs": runs}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
