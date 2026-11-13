"""Create saved predictions (date, ticker, pred_diff_pp, model) from the quant's wide dataset.

modes
  trend_spy  company return over open(t-4) -> close(t) minus SPY over the same window. The SPY window comes from the
             target_spy_return of the row whose exit_date is t (finished by the close of t, so nothing from the future).
             Predicts continuation. --reverse flips it.
  trend   recent-trend rule: trailing `--lookback`-session return, minus that day's average across companies
          (a stand-in for "relative to SPY", since SPY isn't in the wide file). Uses only data up to the close of t.
          --reverse flips it into a mean-reversion rule.
  zero    predicts 0 for everything (the "zero-difference" rule). The simulator should never trade.
  oracle  LEAKY: uses the true future excess return. Only for checking the simulator mechanics (a ceiling).
          Never report it as a model result.
"""
import argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--wide", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--mode", choices=["trend_spy", "trend", "zero", "oracle"], default="trend_spy")
ap.add_argument("--lookback", type=int, default=5, help="sessions, 1-29")
ap.add_argument("--reverse", action="store_true")
a = ap.parse_args()

need = ["ticker", "prediction_date", "close_lag_1"]
if a.mode == "trend_spy":
    need += ["open_lag_5", "exit_date", "target_spy_return"]
if a.mode == "trend":
    assert 1 <= a.lookback <= 29
    need.append(f"close_lag_{1 + a.lookback}")
if a.mode == "oracle":
    need.append("target_excess_return")
df = pd.read_csv(a.wide, usecols=need, parse_dates=[c for c in ("prediction_date", "exit_date") if c in need])
out = df[["prediction_date", "ticker"]].rename(columns={"prediction_date": "date"})

if a.mode == "trend_spy":
    spy_by_exit = df.drop_duplicates("exit_date").set_index("exit_date")["target_spy_return"]
    spy_win = df["prediction_date"].map(spy_by_exit)  # NaN until the first SPY window has finished
    ret = (df["close_lag_1"] / df["open_lag_5"] - 1 - spy_win) * 100
    out["pred_diff_pp"] = -ret if a.reverse else ret
    out["model"] = f"{'reversal' if a.reverse else 'trend'}_vs_spy_5d"
    out = out[spy_win.notna()]
elif a.mode == "trend":
    ret = (df["close_lag_1"] / df[f"close_lag_{1 + a.lookback}"] - 1) * 100
    ret = ret - ret.groupby(df["prediction_date"]).transform("mean")
    out["pred_diff_pp"] = -ret if a.reverse else ret
    out["model"] = f"{'reversal' if a.reverse else 'trend'}_{a.lookback}d"
elif a.mode == "zero":
    out["pred_diff_pp"] = 0.0
    out["model"] = "zero"
else:
    out["pred_diff_pp"] = df["target_excess_return"] * 100
    out["model"] = "ORACLE_LEAKY"
out.to_csv(a.out, index=False)
print(f"wrote {a.out}: {len(out):,} rows, model={out.model.iloc[0]}")
