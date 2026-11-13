"""Prepare development-only price panels for the BullsEye follow-up study."""
import argparse
import io
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf

from price_refinement.experiment import TICKERS

LOCAL = Path("data_local")
FROZEN = LOCAL / "yahoo_adjusted_current_2016_2024.csv"
EARLY = LOCAL / "price_study_early.csv"
MEMBERS = LOCAL / "sp500_membership.csv"
ARCHIVE = LOCAL / "full_history.zip"
MEMBERS_REV = "a2430f2af0c79ddf0748e91de11bdeb1616ab5a7"
MEMBERS_URL = (f"https://raw.githubusercontent.com/fja05680/sp500/{MEMBERS_REV}/"
               "S%26P%20500%20Historical%20Components%20%26%20Changes%20%28Updated%29.csv")
ARCHIVE_URL = ("https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/"
               "Stock_price/full_history.zip")
FIELDS = ["open", "high", "low", "close", "volume"]
END = pd.Timestamp("2021-12-31")
SESSIONS = mcal.get_calendar("NYSE").valid_days("2009-01-01", END).tz_localize(None)


def download_early(path):
    tickers = sorted(set(pd.read_csv(TICKERS).ticker) - {"PLTR"}) + ["SPY"]
    chunks = []
    for start in range(0, len(tickers), 8):
        batch = tickers[start:start + 8]
        prices = yf.download(batch, start="2009-01-01", end="2017-01-03",
                             auto_adjust=True, group_by="ticker", progress=False)
        for ticker in batch:
            one = prices[ticker].dropna(subset=["Close"]).rename_axis("date").reset_index()
            if one.empty:
                raise ValueError(f"Missing price history for {ticker}")
            chunks.append(one.assign(ticker=ticker))
    pd.concat(chunks).to_csv(path, index=False, float_format="%.12g")


def project_prices():
    if not EARLY.exists():
        download_early(EARLY)
    old = pd.read_csv(EARLY, parse_dates=["date"]).rename(columns=str.lower)
    new = pd.read_csv(FROZEN, parse_dates=["date"]).rename(columns=str.lower)
    new = new.loc[new.date <= END]
    both = old.merge(new, on=["ticker", "date"], suffixes=("_old", "_frozen"))
    both["ratio"] = both.close_frozen / both.close_old
    scale = both.groupby("ticker").ratio.agg(["min", "max", "median", "size"])
    if set(scale.index) != set(old.ticker) or (scale["size"] < 20).any():
        raise ValueError("Insufficient overlap to join historical snapshots")
    discrepancy = ((scale["max"] - scale["min"]) / scale["median"]).max()
    if discrepancy > .001:
        raise ValueError(f"Inconsistent adjustment bases in overlap: {discrepancy}")
    old = old.loc[old.date < new.date.min()].copy()
    old[FIELDS[:4]] = old[FIELDS[:4]].mul(old.ticker.map(scale["median"]), axis=0)
    return pd.concat([old, new])[["ticker", "date", *FIELDS]], float(discrepancy)


def membership():
    if not MEMBERS.exists():
        urlretrieve(MEMBERS_URL, MEMBERS)
    history = pd.read_csv(MEMBERS, parse_dates=["date"]).sort_values("date")
    # Snapshots are effective as of their date. Restrict the broad cohort to training years.
    snapshots = history.set_index("date").tickers.loc[:"2020-12-31"]
    active = snapshots.reindex(SESSIONS[SESSIONS.year < 2021], method="ffill")
    by_ticker = {}
    for day, tickers in active.items():
        for ticker in tickers.split(","):
            by_ticker.setdefault(ticker, []).append(day)
    return {ticker: pd.DatetimeIndex(dates) for ticker, dates in by_ticker.items()}


def archive_prices(ticker, archive, names):
    path = f"full_history/{ticker}.csv"
    if path not in names:
        return None
    data = pd.read_csv(io.BytesIO(archive.read(path)), parse_dates=["date"])
    data = data.loc[data.date.between(SESSIONS.min(), "2020-12-31")].sort_values("date").copy()
    if data.empty or data.date.duplicated().any():
        return None
    ratio = data["adj close"] / data.close
    # Unexpected downward steps in a retrospective adjustment factor can create false returns.
    # Suppress the affected bar so no feature window or outcome crosses that break.
    valid = ratio.gt(0) & np.isfinite(ratio) & ~(ratio.pct_change() < -.001)
    data[FIELDS[:4]] = data[FIELDS[:4]].mul(ratio.where(valid), axis=0)
    return data.set_index("date")[FIELDS]


def prepare_prices():
    LOCAL.mkdir(exist_ok=True)
    fixed, discrepancy = project_prices()
    if not ARCHIVE.exists():
        urlretrieve(ARCHIVE_URL, ARCHIVE)
    return fixed, discrepancy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    fixed, discrepancy = prepare_prices()
    print(f"Price history: {len(fixed):,} rows. Max relative adjustment spread: {discrepancy:.6g}")
    print(f"Historical membership: {len(membership()):,} ticker strings.")
