"""Rebuild a daily price table (date, ticker, open, close) from the quant's wide 30-session dataset,
then check it against the quant's own target returns.

Each wide row's open_lag_1 / close_lag_1 are the prices of the prediction session itself (day t),
so one row per company-session gives one price row per company-session.

Limits: SPY is not in the wide file, and the last 5 sessions of the data have no row.
If the quant can hand over the consolidated Yahoo daily file (50 companies + SPY), use it instead:
load_prices() accepts Date/Symbol/Open/Close style headers.
"""
import argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--wide", required=True, help="path to quant_dataset_30day.csv")
ap.add_argument("--out", default="daily_prices.csv")
ap.add_argument("--bench-out", default="period_benchmarks.csv")
a = ap.parse_args()

cols = ["ticker", "prediction_date", "entry_date", "exit_date", "open_lag_1", "close_lag_1",
        "target_company_return", "target_spy_return"]
df = pd.read_csv(a.wide, usecols=cols, parse_dates=["prediction_date", "entry_date", "exit_date"])
px = df.rename(columns={"prediction_date": "date", "open_lag_1": "open", "close_lag_1": "close"})
px = px[["date", "ticker", "open", "close"]]
if px.duplicated(["date", "ticker"]).any():
    raise SystemExit("duplicate ticker/date rows in the wide file")
px = px.sort_values(["ticker", "date"])
px.to_csv(a.out, index=False)
print(f"wrote {a.out}: {len(px):,} rows, {px.ticker.nunique()} tickers, "
      f"{px.date.min().date()} to {px.date.max().date()}")

# ---- check: do the rebuilt prices reproduce the quant's own target_company_return? ----
o = px.set_index(["ticker", "date"])["open"]
c = px.set_index(["ticker", "date"])["close"]
entry_open = o.reindex(pd.MultiIndex.from_arrays([df.ticker, df.entry_date])).to_numpy()
exit_close = c.reindex(pd.MultiIndex.from_arrays([df.ticker, df.exit_date])).to_numpy()
calc = exit_close / entry_open - 1
diff = pd.Series(abs(calc - df.target_company_return.to_numpy())).dropna()
print(f"checkable rows: {len(diff):,} of {len(df):,}")
print(f"max |rebuilt target - quant target|: {diff.max():.2e}")
print(f"rows differing by more than 1e-6: {(diff > 1e-6).sum():,}")
if len(diff) == 0 or (diff > 1e-6).any():
    print("WARNING: rebuilt prices do not fully match the quant's targets. Do not trust results until resolved.")
else:
    print("OK: rebuilt daily prices reproduce the quant's targets.")

# ---- SPY and equal-weight 5-session benchmarks, taken from the quant's own target columns ----
spy_chk = df.groupby("prediction_date")["target_spy_return"].nunique()
spread = df.groupby("prediction_date")["target_spy_return"].agg(lambda s: s.max() - s.min()).max()
print(f"SPY target identical across companies on each date: max spread {spread:.2e}")
if spread > 1e-9:
    print("WARNING: target_spy_return differs across companies on the same date; check with the quant.")
bm = df.groupby("prediction_date").agg(spy_ret=("target_spy_return", "first"),
                                       ew_ret=("target_company_return", "mean")).reset_index()
bm = bm.rename(columns={"prediction_date": "date"})
bm.to_csv(a.bench_out, index=False)
print(f"wrote {a.bench_out}: {len(bm):,} dates (return of open t+1 -> close t+5, decimals)")
