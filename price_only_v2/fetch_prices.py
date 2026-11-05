"""Fetch a local, dated Yahoo snapshot for the agreed companies and SPY.

The snapshot is intentionally kept out of Git. Re-running a public data service
later may produce revised historical adjustments, so the build records a hash.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


FIELDS = ("Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ea_from_fnspid(archive: Path, start: str, end: str) -> pd.DataFrame:
    """Use the archived EA history after Yahoo stopped serving delisted EA.

    FNSPID's adjusted-close field is spuriously equal to raw close through
    2020-07-02, then suddenly drops to an adjustment factor below one on
    2020-07-06 without a split or dividend. Carry that factor backward to
    remove the false discontinuity.
    This is checked in audit.py and must be disclosed as a source limitation.
    """
    with zipfile.ZipFile(archive) as bundle:
        table = pd.read_csv(io.BytesIO(bundle.read("full_history/EA.csv")))
    table["date"] = pd.to_datetime(table["date"])
    table = table.sort_values("date").reset_index(drop=True)
    ratio = table["adj close"] / table["close"]
    drops = np.flatnonzero(np.diff(ratio.to_numpy()) < -1e-4) + 1
    drops = [i for i in drops if table.loc[i, "date"] >= pd.Timestamp("2017-01-01")]
    if len(drops) != 1 or str(table.loc[drops[0], "date"].date()) != "2020-07-06":
        raise ValueError(f"EA adjustment discontinuity changed: {drops}")
    ratio.iloc[:drops[0]] = float(ratio.iloc[drops[0]])
    if (np.diff(ratio.to_numpy()[drops[0]:]) < -1e-4).any():
        raise ValueError("EA adjustment factor is not monotone after correction")
    for field in ("open", "high", "low", "close"):
        table[field] = table[field] * ratio
    table = table.loc[table["date"].between(start, end, inclusive="left")]
    result = table[["date", "open", "high", "low", "close", "volume"]].copy()
    result.columns = ["date", "Open", "High", "Low", "Close", "Volume"]
    result["Dividends"] = np.nan  # event amounts are unavailable in this archive
    result["Stock Splits"] = 0.0
    result["date"] = result["date"].dt.strftime("%Y-%m-%d")
    result.insert(0, "ticker", "EA")
    return result


def fetch(tickers: list[str], start: str, end: str, output: Path, archive: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    failures = []
    yahoo_tickers = [ticker for ticker in tickers if ticker != "EA"]
    for offset in range(0, len(yahoo_tickers), 8):
        batch = yahoo_tickers[offset:offset + 8]
        query_batch = ["BNY" if ticker == "BK" else ticker for ticker in batch]
        print(f"Fetching {offset + 1}-{min(offset + 8, len(tickers))}/{len(tickers)}: {', '.join(batch)}", flush=True)
        table = yf.download(
            query_batch, start=start, end=end, interval="1d", auto_adjust=True,
            actions=True, repair=False, keepna=False, group_by="ticker",
            threads=True, progress=False,
        )
        for ticker in batch:
            query_ticker = "BNY" if ticker == "BK" else ticker
            if query_ticker not in table.columns.get_level_values(0):
                failures.append(ticker)
                continue
            one = table[query_ticker].copy()
            if not set(FIELDS).issubset(one.columns):
                failures.append(ticker)
                continue
            one = one.loc[one["Close"].notna(), list(FIELDS)].reset_index()
            one = one.rename(columns={one.columns[0]: "date"})
            one["date"] = pd.to_datetime(one["date"]).dt.strftime("%Y-%m-%d")
            one.insert(0, "ticker", ticker)
            records.append(one)
    if failures:
        raise RuntimeError(f"Missing Yahoo data for {failures}; no partial snapshot written")
    records.append(ea_from_fnspid(archive, start, end))
    frame = pd.concat(records, ignore_index=True).sort_values(["ticker", "date"])
    if frame.duplicated(["ticker", "date"]).any():
        raise ValueError("Duplicate ticker/date in downloaded snapshot")
    frame.to_csv(output, index=False, float_format="%.12g")
    meta = {
        "source": "Yahoo Finance via yfinance; EA from FNSPID price archive",
        "source_url": "https://finance.yahoo.com/",
        "fnspid_url": "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_price/full_history.zip",
        "fnspid_archive_sha256": sha256(archive),
        "ticker_aliases": {"BK": "BNY"},
        "ea_adjustment": "FNSPID adjusted-close/close factor; its first downward jump on 2020-07-06 is a source discontinuity, so that factor is carried backward through prior dates. EA dividend events unavailable for separate checking.",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "yfinance_version": yf.__version__, "start_inclusive": start,
        "end_exclusive": end, "auto_adjust": True, "repair": False,
        "volume_policy": "Yahoo-reported Volume is retained unchanged by auto_adjust; audit split consistency separately.",
        "tickers": tickers, "rows": len(frame), "sha256": sha256(output),
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {len(frame):,} daily rows to {output}; SHA-256 {meta['sha256']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--companies", type=Path, default=Path("COMPANIES.csv"))
    parser.add_argument("--start", default="2016-11-01")
    parser.add_argument("--end", default="2024-01-10")
    parser.add_argument("--output", type=Path, default=Path("data_local/yahoo_adjusted_2016_2024.csv"))
    parser.add_argument("--fnspid-archive", type=Path, default=Path("data_local/full_history.zip"))
    args = parser.parse_args()
    tickers = sorted(set(pd.read_csv(args.companies)["ticker"].astype(str).str.upper())) + ["SPY"]
    fetch(tickers, args.start, args.end, args.output, args.fnspid_archive)


if __name__ == "__main__":
    main()
