"""Features and chronological development rows for the original 50."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

FIELDS = ("open", "high", "low", "close", "volume")
SECTORS = tuple(sorted(pd.read_csv(Path(__file__).resolve().parent.parent / "COMPANIES.csv").sector.unique()))


def read_development(path: Path) -> pd.DataFrame:
    """Read 2017-2021 only, stopping before the reserved 2022-23 period."""
    manifest_path = path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    count = sum(int(manifest["rows_by_year"][str(year)]) for year in range(2017, 2022))
    frame = pd.read_csv(path, nrows=count)
    frame["prediction_date"] = pd.to_datetime(frame.prediction_date)
    frame["exit_date"] = pd.to_datetime(frame.exit_date)
    if frame.prediction_date.max().year != 2021 or frame.prediction_date.min().year != 2017:
        raise ValueError("Expected 2017-2021 development rows")
    if frame.duplicated(["ticker", "prediction_date"]).any():
        raise ValueError("Duplicate company/date")
    return frame


def windows(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    def extract(prefix: str) -> np.ndarray:
        cols = [f"{prefix}{field}_lag_{lag}" for lag in range(30, 0, -1) for field in FIELDS]
        return frame[cols].to_numpy(dtype=np.float32).reshape(len(frame), 30, 5)
    return extract(""), extract("spy_")


def feature_matrix(company: np.ndarray, market: np.ndarray, sector: pd.Series,
                   lookback: int, include_market: bool = True,
                   include_volume: bool = False) -> tuple[np.ndarray, list[str]]:
    if lookback not in (5, 10, 30):
        raise ValueError("Lookback must be 5, 10, or 30")
    c = company[:, -lookback:, :].astype(np.float64)
    s = market[:, -lookback:, :].astype(np.float64)
    names, arrays = [], []
    def add(name: str, value: np.ndarray) -> None:
        names.append(name)
        arrays.append(value)
    for label, window in (("company", c), ("spy", s)):
        if label == "spy" and not include_market:
            continue
        close, open_, high, low = window[:, :, 3], window[:, :, 0], window[:, :, 1], window[:, :, 2]
        log_returns = np.diff(np.log(close), axis=1)
        for period in (1, 3, 4, 5, 9, 10, 20, 29):
            if period < lookback:
                add(f"{label}_return_{period}", close[:, -1]/close[:, -(period+1)]-1)
        for period in (5, 10, 30):
            if period <= lookback:
                add(f"{label}_volatility_{period}", log_returns[:, -(period-1):].std(axis=1))
                add(f"{label}_intraday_{period}", ((close-open_)/open_)[:, -period:].mean(axis=1))
                add(f"{label}_range_{period}", ((high-low)/open_)[:, -period:].mean(axis=1))
        add(f"{label}_last_intraday", close[:, -1]/open_[:, -1]-1)
        add(f"{label}_last_range", (high[:, -1]-low[:, -1])/open_[:, -1])
        if include_volume:
            vol = window[:, :, 4]
            add(f"{label}_volume_recent_vs_window", vol[:, -min(5,lookback):].mean(axis=1)/np.maximum(vol.mean(axis=1), 1)-1)
            add(f"{label}_log_volume_last", np.log1p(vol[:, -1]))
    if include_market:
        for period in (1, 3, 4, 5, 9, 10, 20, 29):
            if period < lookback:
                add(f"relative_return_{period}", c[:, -1, 3]/c[:, -(period+1), 3]
                    - s[:, -1, 3]/s[:, -(period+1), 3])
    if not set(sector.unique()).issubset(SECTORS):
        raise ValueError("Unknown sector in inference input")
    for name in SECTORS:
        add(f"sector_{name}", (sector.to_numpy() == name).astype(float))
    result = np.column_stack(arrays).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite feature")
    return result, names


def sequence_matrix(company: np.ndarray, market: np.ndarray, include_volume: bool) -> np.ndarray:
    def scaled(window: np.ndarray) -> np.ndarray:
        prices = window[:, :, :4] / window[:, -1:, 3:4] - 1
        if include_volume:
            log_volume = np.log1p(window[:, :, 4:5])
            return np.concatenate((prices, log_volume), axis=2)
        return prices
    return np.concatenate((scaled(company), scaled(market)), axis=2).astype(np.float32)


def split_mask(frame: pd.DataFrame, first_train_year: int, last_train_year: int,
               validation_year: int) -> tuple[np.ndarray, np.ndarray]:
    train = ((frame.prediction_date.dt.year >= first_train_year)
             & (frame.prediction_date.dt.year <= last_train_year)
             & (frame.exit_date.dt.year <= last_train_year)).to_numpy()
    valid = ((frame.prediction_date.dt.year == validation_year)
             & (frame.exit_date.dt.year == validation_year)).to_numpy()
    if not train.any() or not valid.any():
        raise ValueError("Empty temporal split")
    if frame.loc[train, "exit_date"].max() >= frame.loc[valid, "prediction_date"].min():
        raise ValueError("Overlapping label intervals at split")
    return train, valid
