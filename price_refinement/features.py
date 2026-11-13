"""Close-time features for the price refinement experiments."""
import numpy as np
import pandas as pd


def make_features(keys, company, market):
    """Use complete same-session eligible rows for the peer comparisons."""
    date, sector = keys["prediction_date"], keys["sector"]
    c, s = company.astype("float64"), market.astype("float64")
    op, high, low, close, volume = c.transpose(2, 0, 1)
    spy = s[:, :, 3]
    daily = np.diff(np.log(close), axis=1)
    spy_daily = np.diff(np.log(spy), axis=1)
    vol = daily[:, -20:].std(axis=1)
    risk = np.maximum(vol, 0.002)
    base = {}
    for days in (1, 4, 9, 20, 29):
        stock_return = np.log(close[:, -1] / close[:, -1-days])
        spy_return = np.log(spy[:, -1] / spy[:, -1-days])
        base[f"relative_{days}"] = stock_return - spy_return
    base["intraday_5"] = np.log(close[:, -5:] / op[:, -5:]).mean(axis=1)
    base["overnight_5"] = np.log(op[:, -5:] / close[:, -6:-1]).mean(axis=1)
    base["volatility_20"] = vol
    base["range_5"] = np.log(high[:, -5:] / low[:, -5:]).mean(axis=1)
    base["drawdown_20"] = np.log(close[:, -1] / close[:, -20:].max(axis=1))
    base["risk_scaled_relative_4"] = base["relative_4"] / risk
    base["spy_volatility_20"] = spy_daily[:, -20:].std(axis=1)
    base["spy_return_20"] = np.log(spy[:, -1] / spy[:, -21])
    frame = pd.DataFrame(base, index=keys.index)
    peer = frame.copy()
    for col in ("relative_1", "relative_4", "relative_9", "relative_20",
                "volatility_20", "drawdown_20", "risk_scaled_relative_4"):
        peer[f"daily_rank_{col}"] = frame[col].groupby(date).rank(pct=True) - 0.5
        peer[f"sector_relative_{col}"] = frame[col] - frame[col].groupby([date, sector]).transform("median")
    peer_volume = peer.assign(relative_volume=np.log((volume[:, -5:].mean(axis=1) + 1)
                                                     / (volume[:, -20:].mean(axis=1) + 1)))
    trend = close[:, -1] / op[:, -5] - spy[:, -1] / s[:, -5, 0]
    simple = {
        "trend": trend,
        "short_reversal": -base["relative_1"],
        "risk_adjusted_trend": trend / risk,
        "low_volatility": -vol,
    }
    return {"base": frame, "peer": peer, "peer_volume": peer_volume}, simple
