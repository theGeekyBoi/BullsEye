"""Generate made-up prices and predictions so the simulator can be tested before real model output exists."""
import argparse
import numpy as np
import pandas as pd


def make_prices(n_tickers=20, start="2020-12-01", end="2023-12-31", seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    rows = []
    for t in ["SPY"] + [f"T{i:02d}" for i in range(n_tickers)]:
        ret = rng.normal(0.0004, 0.012, len(dates))
        close = 100 * np.exp(np.cumsum(ret))
        open_ = np.roll(close, 1) * (1 + rng.normal(0, 0.003, len(dates)))
        open_[0] = 100
        rows.append(pd.DataFrame(dict(date=dates, ticker=t, open=open_, close=close)))
    return pd.concat(rows, ignore_index=True)


def make_predictions(prices, signal=0.15, seed=1):
    """pred = signal * realized 5-session excess (open t+1 -> close t+5) + noise, in pp. Weak, so results stay realistic."""
    rng = np.random.default_rng(seed)
    o = prices.pivot(index="date", columns="ticker", values="open")
    c = prices.pivot(index="date", columns="ticker", values="close")
    fwd = (c.shift(-5) / o.shift(-1) - 1) * 100
    excess = fwd.sub(fwd["SPY"], axis=0).drop(columns="SPY")
    pred = signal * excess + rng.normal(0, 2.0, excess.shape)
    out = pred.stack().rename("pred_diff_pp").reset_index()
    out["model"] = "synthetic"
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="demo_data")
    a = ap.parse_args()
    import os
    os.makedirs(a.out_dir, exist_ok=True)
    px = make_prices()
    px.to_csv(f"{a.out_dir}/daily_prices.csv", index=False)
    make_predictions(px).to_csv(f"{a.out_dir}/predictions.csv", index=False)
    print("wrote", a.out_dir)
