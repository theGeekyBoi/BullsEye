"""Load the quantitative CSV without exposing future targets as model features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PRICE_FIELDS = ("open", "high", "low", "close", "volume")
KEY_COLUMNS = ("ticker", "prediction_date")
TARGET_COLUMNS = (
    "target_company_return",
    "target_spy_return",
    "target_excess_return",
)
METADATA_COLUMNS = ("ticker", "prediction_date", "entry_date", "exit_date")
WINDOW_COLUMNS = tuple(
    f"{field}_lag_{lag}"
    for lag in range(30, 0, -1)
    for field in PRICE_FIELDS
)
REQUIRED_COLUMNS = METADATA_COLUMNS + WINDOW_COLUMNS + TARGET_COLUMNS


def load_development_rows(path: str | Path, train_year: int, validation_year: int) -> pd.DataFrame:
    """Read only the requested development years from the full CSV."""
    path = Path(path)
    header = pd.read_csv(path, nrows=0).columns
    missing = sorted(set(REQUIRED_COLUMNS) - set(header))
    if missing:
        raise ValueError(f"Quantitative dataset is missing columns: {missing}")
    if validation_year != train_year + 1:
        raise ValueError("The development split currently expects adjacent calendar years.")

    pieces = []
    for chunk in pd.read_csv(path, usecols=list(REQUIRED_COLUMNS), chunksize=10_000):
        dates = pd.to_datetime(chunk["prediction_date"], errors="raise")
        keep = dates.dt.year.isin((train_year, validation_year))
        if keep.any():
            pieces.append(chunk.loc[keep].copy())
    if not pieces:
        raise ValueError("No examples found for the requested development years.")

    frame = pd.concat(pieces, ignore_index=True)
    for column in ("prediction_date", "entry_date", "exit_date"):
        frame[column] = pd.to_datetime(frame[column], errors="raise")
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    frame = frame.sort_values(list(KEY_COLUMNS), kind="stable").reset_index(drop=True)

    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("Duplicate ticker/prediction_date pairs found.")
    if frame[list(WINDOW_COLUMNS) + list(TARGET_COLUMNS)].isna().any().any():
        raise ValueError("Missing model input or target value found.")
    numeric = frame[list(WINDOW_COLUMNS) + list(TARGET_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("Non-finite model input or target value found.")
    if not ((frame["prediction_date"] < frame["entry_date"]) &
            (frame["entry_date"] < frame["exit_date"])).all():
        raise ValueError("Prediction, entry, and exit dates are out of order.")
    if not np.allclose(
        frame["target_company_return"] - frame["target_spy_return"],
        frame["target_excess_return"],
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError("Excess-return targets disagree with company minus SPY returns.")
    return frame


def split_development_rows(
    frame: pd.DataFrame, train_year: int, validation_year: int
) -> tuple[np.ndarray, np.ndarray]:
    """Exclude rows whose five-session outcome reaches the next year."""
    train = (
        (frame["prediction_date"].dt.year == train_year)
        & (frame["exit_date"].dt.year == train_year)
    ).to_numpy()
    validation = (
        (frame["prediction_date"].dt.year == validation_year)
        & (frame["exit_date"].dt.year == validation_year)
    ).to_numpy()
    if not train.any() or not validation.any():
        raise ValueError("Training or validation split has no eligible examples.")
    if frame.loc[train, "exit_date"].max() >= frame.loc[validation, "prediction_date"].min():
        raise ValueError("Training outcomes overlap the validation prediction period.")
    return train, validation


def ordered_windows(frame: pd.DataFrame) -> np.ndarray:
    """Return N x 30 x 5 arrays, with lag_30 first and lag_1 last."""
    values = frame.loc[:, WINDOW_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    windows = values.reshape(len(frame), 30, len(PRICE_FIELDS))
    if not np.isfinite(windows).all():
        raise ValueError("Window contains non-finite values.")
    if (windows[:, :, :4] <= 0).any() or (windows[:, :, 4] < 0).any():
        raise ValueError("Window contains a non-positive price or negative volume.")
    return windows


def recent_trend_prediction(windows: np.ndarray) -> np.ndarray:
    """Past five-session stock return, used as a simple excess-return proxy."""
    return windows[:, -1, 3].astype(np.float64) / windows[:, -5, 0] - 1.0


def tree_features(windows: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Scale each window by its own known closing price and add simple summaries."""
    prices = windows[:, :, :4].astype(np.float64)
    volume = windows[:, :, 4].astype(np.float64)
    last_close = prices[:, -1, 3:4]
    relative_prices = prices / last_close[:, None, :] - 1.0
    log_volume = np.log1p(volume)
    relative_volume = log_volume - log_volume.mean(axis=1, keepdims=True)

    flat_prices = relative_prices.reshape(len(windows), -1)
    flat_volume = relative_volume
    names = [
        f"{field}_relative_lag_{lag}"
        for lag in range(30, 0, -1)
        for field in PRICE_FIELDS[:4]
    ] + [f"log_volume_relative_lag_{lag}" for lag in range(30, 0, -1)]

    closes = prices[:, :, 3]
    opens = prices[:, :, 0]
    highs = prices[:, :, 1]
    lows = prices[:, :, 2]
    log_returns = np.diff(np.log(closes), axis=1)
    summaries: list[np.ndarray] = []
    for periods in (1, 5, 10, 20, 29):
        summaries.append(closes[:, -1] / closes[:, -(periods + 1)] - 1.0)
        names.append(f"close_return_{periods}_intervals")
    for periods in (5, 10, 29):
        summaries.append(log_returns[:, -periods:].std(axis=1))
        names.append(f"log_return_volatility_{periods}")
    for periods in (5, 30):
        summaries.append(((closes - opens) / opens)[:, -periods:].mean(axis=1))
        names.append(f"mean_intraday_change_{periods}")
        summaries.append(((highs - lows) / opens)[:, -periods:].mean(axis=1))
        names.append(f"mean_daily_range_{periods}")
    summaries.append(volume[:, -5:].mean(axis=1) / np.maximum(volume.mean(axis=1), 1.0) - 1.0)
    names.append("recent_to_window_volume_ratio")

    features = np.column_stack([flat_prices, flat_volume, *summaries]).astype(np.float32)
    if features.shape[1] != len(names) or not np.isfinite(features).all():
        raise ValueError("Tree feature creation failed.")
    return features, names


def sequence_features(windows: np.ndarray, include_volume: bool) -> np.ndarray:
    """Produce per-window scaled OHLC(V) sequences, before train-only standardization."""
    prices = windows[:, :, :4].astype(np.float32)
    reference = prices[:, -1, 3].reshape(-1, 1, 1)
    scaled_prices = prices / reference - 1.0
    if not include_volume:
        return scaled_prices
    log_volume = np.log1p(windows[:, :, 4:5].astype(np.float32))
    return np.concatenate((scaled_prices, log_volume), axis=2)


def fit_standardizer(train_sequences: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit one mean and standard deviation per feature using training rows only."""
    mean = train_sequences.mean(axis=(0, 1), keepdims=True)
    std = train_sequences.std(axis=(0, 1), keepdims=True)
    std = np.maximum(std, 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)
