import pandas as pd


def _read(path):
    return pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)


def load_prices(path) -> pd.DataFrame:
    """Daily price table: date, ticker, open, close (one row per company-session, plus SPY).
    Opens and closes must be on the SAME split/dividend adjustment basis."""
    df = _read(path)
    # tolerate common header variants (Yahoo/FNSPID exports): Date, Symbol, Open, Close ...
    df = df.rename(columns={c: c.strip().lower().replace(" ", "_") for c in df.columns})
    df = df.rename(columns={"symbol": "ticker", "prediction_date": "date"})
    missing = {"date", "ticker", "open", "close"} - set(df.columns)
    if missing:
        raise ValueError(f"price table missing columns: {missing}")
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_predictions(path, model=None) -> pd.DataFrame:
    """Saved predictions: date, ticker, pred_diff_pp [, model]. `date` is the close of day t."""
    df = _read(path)
    missing = {"date", "ticker", "pred_diff_pp"} - set(df.columns)
    if missing:
        raise ValueError(f"prediction file missing columns: {missing}")
    if "model" in df.columns:
        if model is None:
            if df["model"].nunique() > 1:
                raise ValueError(f"multiple models {sorted(df['model'].unique())}; pass model=")
        else:
            df = df[df["model"] == model]
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_period_benchmarks(path) -> pd.DataFrame:
    """date (signal day t), spy_ret, ew_ret: SPY and equal-weight returns for open(t+1) -> close(t+5), as decimals."""
    df = _read(path)
    missing = {"date", "spy_ret", "ew_ret"} - set(df.columns)
    if missing:
        raise ValueError(f"benchmark file missing columns: {missing}")
    df["date"] = pd.to_datetime(df["date"])
    return df
