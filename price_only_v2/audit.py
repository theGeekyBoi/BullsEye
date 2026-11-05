"""Audit source coverage, selected corporate actions, and rebuilt labels."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

FIELDS = ("open", "high", "low", "close", "volume")
CASES = [
    ("AMD", "2017-05-01", "ordinary training day"),
    ("CMCSA", "2017-02-16", "2017 two-for-one split during holding"),
    ("TJX", "2018-11-05", "2018 two-for-one split during holding"),
    ("NEE", "2020-10-23", "2020 four-for-one split during holding"),
    ("GE", "2021-07-29", "2021 one-for-eight reverse split during holding"),
    ("MRK", "2021-03-10", "2021-03-12 dividend in outcome"),
    ("EA", "2020-07-01", "EA source adjustment discontinuity during holding"),
    ("EA", "2021-04-01", "archived ticker fallback"),
    ("SPY", "2020-03-13", "market benchmark shock"),
]


def audit(snapshot: Path, built: Path, original: Path, archive: Path, output: Path) -> dict:
    source = pd.read_csv(snapshot, parse_dates=["date"])
    target_columns = ["ticker", "sector", "prediction_date", "entry_date", "exit_date",
                      "target_company_return", "target_spy_return", "target_excess_return"]
    sample_tickers = {ticker for ticker, _, _ in CASES if ticker != "SPY"}
    samples = []
    for chunk in pd.read_csv(built, chunksize=2000):
        chosen = chunk.ticker.isin(sample_tickers) & chunk.prediction_date.isin([d for _, d, _ in CASES])
        samples.extend(chunk.loc[chosen].to_dict("records"))
    lookup = {(row["ticker"], row["prediction_date"]): row for row in samples}
    original_tickers = set(pd.read_csv(original).ticker)
    current = pd.read_csv("final_dataset/quant_dataset_30day.csv", usecols=["ticker"])
    current_tickers = set(current.ticker)
    source["ticker"] = source.ticker.astype(str)
    if source.duplicated(["ticker", "date"]).any():
        raise ValueError("Duplicate source company/date")
    required = [x.title() for x in FIELDS]
    invalid = (~np.isfinite(source[required].to_numpy(dtype=float)).all(axis=1)
               | (source[["Open", "High", "Low", "Close"]].to_numpy(dtype=float) <= 0).any(axis=1)
               | (source.Volume < 0)
               | (source.High + 1e-3 < source[["Open", "Close", "Low"]].max(axis=1))
               | (source.Low - 1e-3 > source[["Open", "Close", "High"]].min(axis=1)))
    dates = pd.DatetimeIndex(mcal.get_calendar("NYSE").valid_days("2017-01-01", "2023-12-31").tz_localize(None))
    coverage = {}
    for ticker in sorted(original_tickers | {"SPY"}):
        observed = pd.DatetimeIndex(source.loc[source.ticker == ticker, "date"])
        coverage[ticker] = {"sessions": int(len(observed.intersection(dates))),
                            "missing_dates": [str(x.date()) for x in dates.difference(observed)[:10]],
                            "missing_count": int(len(dates.difference(observed)))}
    by_key = {(ticker, date): row for (ticker, date), row in source.set_index(["ticker", "date"]).iterrows()}
    checks = []
    for ticker, date, description in CASES:
        if ticker == "SPY":
            continue
        row = lookup.get((ticker, date))
        if row is None:
            raise ValueError(f"Audit case {ticker} {date} absent")
        t = pd.Timestamp(date)
        prior = dates[dates <= t][-30:]
        if len(prior) != 30:
            raise ValueError("Insufficient prior history")
        max_input_error = 0.0
        for lag, source_date in zip(range(30, 0, -1), prior):
            for field in FIELDS:
                for prefix, source_ticker in (("", ticker), ("spy_", "SPY")):
                    actual = float(by_key[(source_ticker, source_date)][field.title()])
                    saved = float(row[f"{prefix}{field}_lag_{lag}"])
                    max_input_error = max(max_input_error, abs(saved-actual))
        entry = pd.Timestamp(row["entry_date"])
        exit_ = pd.Timestamp(row["exit_date"])
        expected_company = by_key[(ticker, exit_)]["Close"]/by_key[(ticker, entry)]["Open"]-1
        expected_spy = by_key[("SPY", exit_)]["Close"]/by_key[("SPY", entry)]["Open"]-1
        max_target_error = max(abs(row["target_company_return"]-expected_company),
                               abs(row["target_spy_return"]-expected_spy),
                               abs(row["target_excess_return"]-(expected_company-expected_spy)))
        if max_input_error > 1e-5 or max_target_error > 1e-9:
            raise ValueError(f"Audit case {ticker} {date} disagrees with source")
        checks.append({"ticker": ticker, "prediction_date": date, "case": description,
                       "entry": str(entry.date()), "exit": str(exit_.date()),
                       "dividend_events_in_outcome": source.loc[
                           (source.ticker == ticker) & source.date.between(entry, exit_)
                           & (source.Dividends > 0), ["date", "Dividends"]
                       ].assign(date=lambda x: x.date.dt.strftime("%Y-%m-%d")).to_dict("records"),
                       "inputs_checked": 300, "maximum_input_error": max_input_error,
                       "maximum_target_error": max_target_error,
                       "company_return": float(expected_company), "spy_return": float(expected_spy),
                       "excess_return": float(expected_company-expected_spy)})
    # Inspect conventional splits without using 2022-23 test outcomes.
    split_cases = []
    with zipfile.ZipFile(archive) as bundle:
        ea_raw = pd.read_csv(io.BytesIO(bundle.read("full_history/EA.csv"))).sort_values("date")
        ea_raw = ea_raw.loc[ea_raw.date.between("2017-01-01", "2023-12-31")].copy()
        ea_ratio = ea_raw["adj close"]/ea_raw.close
        jumps = ea_ratio.diff()
        ea_dividend_steps = []
        for row, change, old_factor in zip(ea_raw.loc[jumps > 1e-4].itertuples(),
                                           jumps[jumps > 1e-4], ea_ratio.shift(1)[jumps > 1e-4]):
            ea_dividend_steps.append({"date": row.date,
                                      "approximate_cash_from_adjustment": float(change/old_factor*row.close)})
        for ticker, day in (("CMCSA", "2017-02-21"), ("TJX", "2018-11-07"),
                            ("NEE", "2020-10-27"), ("GE", "2021-08-02")):
            one = source[source.ticker == ticker].sort_values("date")
            after = one.loc[one.date == day].iloc[0]
            before = one.loc[one.date < day].iloc[-1]
            fnspid = pd.read_csv(io.BytesIO(bundle.read(f"full_history/{ticker}.csv")))
            fnspid_pre = fnspid.loc[fnspid.date == before.date.strftime("%Y-%m-%d")].iloc[0]
            split_cases.append({"ticker": ticker, "date": day, "reported_split_ratio": float(after["Stock Splits"]),
                                "adjusted_close_change_pct": float((after.Close/before.Close-1)*100),
                                "yahoo_volume_before": float(before.Volume), "yahoo_volume_after": float(after.Volume),
                                "fnspid_volume_before": float(fnspid_pre.volume),
                                "yahoo_to_fnspid_volume_before": float(before.Volume/fnspid_pre.volume)})
    report = {"original_company_count": len(original_tickers), "current_company_count": len(current_tickers),
              "overlap_count": len(original_tickers & current_tickers),
              "original_only": sorted(original_tickers-current_tickers),
              "prototype_only": sorted(current_tickers-original_tickers),
              "source_invalid_ohlcv_rows": int(invalid.sum()), "coverage": coverage,
              "sample_checks": checks, "split_cases": split_cases,
              "ea_dividend_adjustment_steps": ea_dividend_steps,
              "ea_caveat": "EA uses FNSPID archived adjusted-close factors because Yahoo no longer serves delisted EA. Its factor had a false downward step on 2020-07-06; the post-step factor was carried backward. Thirteen later upward steps imply approximately 17-19 cent dividends, consistent with company announcements, but a complete independent daily price source is unavailable.",
              "volume_conclusion": "Yahoo auto_adjust changes OHLC but does not apply another multiplier to Volume. Ordinary split dates show no split-sized price discontinuity and TJX/NEE volumes match the 2023 FNSPID snapshot. This supports using reported Volume without another adjustment, but unusual spin-offs are not independently resolved. Keep volume as an ablation, not a required primary input."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Audited {len(checks)} examples (300 inputs plus 3 targets each), {len(split_cases)} split events; invalid source rows: {report['source_invalid_ohlcv_rows']}")
    print(f"Original/current company overlap: {report['overlap_count']}; source missing counts: {[ (k,v['missing_count']) for k,v in coverage.items() if v['missing_count'] ]}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("data_local/yahoo_adjusted_2016_2024.csv"))
    parser.add_argument("--built", type=Path, default=Path("data_local/price_v2_original_50_2017_2023.csv"))
    parser.add_argument("--archive", type=Path, default=Path("data_local/full_history.zip"))
    parser.add_argument("--output", type=Path, default=Path("results/price_only_v2/data_audit.json"))
    args = parser.parse_args()
    audit(args.snapshot, args.built, Path("COMPANIES.csv"), args.archive, args.output)


if __name__ == "__main__":
    main()
