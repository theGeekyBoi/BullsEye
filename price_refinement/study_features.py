"""Compute price features and the shared five-session target on a daily calendar."""
from zipfile import ZipFile

import numpy as np
import pandas as pd

from price_refinement.study_data import ARCHIVE, FIELDS, SESSIONS, archive_prices


def log_return(close):
    return np.log(close).diff()


def residual(stock, factors, lookback):
    """OLS with coefficients fit on sessions strictly before the scored return."""
    x = [part for _, part in factors.items()]
    lag = lambda a: a.shift().rolling(lookback, min_periods=lookback)
    s = lag(stock).mean()
    m = [lag(a).mean() for a in x]
    cov = [lag(stock).cov(a.shift()) for a in x]
    var = [lag(a).var() + 1e-8 for a in x]
    if len(x) == 1:
        return (stock - s) - (cov[0] / var[0]) * (x[0] - m[0])
    xy = lag(x[0]).cov(x[1].shift())
    det = var[0] * var[1] - xy**2
    beta = [(cov[0] * var[1] - cov[1] * xy) / det,
            (cov[1] * var[0] - cov[0] * xy) / det]
    return (stock - s) - beta[0] * (x[0] - m[0]) - beta[1] * (x[1] - m[1])


def stock_rows(ticker, data, spy, sector_factor=None, with_target=True):
    sessions = spy.index
    a = data.reindex(sessions)
    close, op = a.close, a.open
    daily, market = log_return(close), log_return(spy.close)
    vol = daily.rolling(20).std(ddof=0)
    result = {}
    for n in (1, 4, 9, 20, 29):
        result[f"relative_{n}"] = np.log(close / close.shift(n)) - np.log(spy.close / spy.close.shift(n))
    result["intraday_5"] = np.log(close / op).rolling(5).mean()
    result["overnight_5"] = np.log(op / close.shift()).rolling(5).mean()
    result["volatility_20"] = vol
    result["range_5"] = np.log(a.high / a.low).rolling(5).mean()
    result["drawdown_20"] = np.log(close / close.rolling(20).max())
    result["risk_scaled_relative_4"] = result["relative_4"] / vol.clip(lower=.002)
    result["spy_volatility_20"] = market.rolling(20).std(ddof=0)
    result["spy_return_20"] = np.log(spy.close / spy.close.shift(20))
    result["trend"] = close / op.shift(4) - spy.close / spy.open.shift(4)
    for label, factors in [("market", {"spy": market}), ("sector", {"spy": market, "peer": sector_factor})]:
        if label == "sector" and sector_factor is None:
            continue
        for window in (60, 120):
            moves = residual(daily, factors, window)
            for n in (1, 5, 20):
                result[f"{label}_{window}_{n}"] = moves.rolling(n).sum()
    for n in (60, 120, 240):
        result[f"trend_{n}"] = np.log(close / close.shift(n)) - np.log(spy.close / spy.close.shift(n))
    result["market_volatility_relative"] = (market.rolling(20).std() / market.rolling(120).std()).clip(.3, 3) - 1
    valid = np.isfinite(a[FIELDS]).all(axis=1) & (a[FIELDS[:4]] > 0).all(axis=1) & (a.volume >= 0)
    history = valid.rolling(30).sum() == 30
    if with_target:
        result["target_excess_return"] = (close.shift(-5) / op.shift(-1)
                                          - spy.close.shift(-5) / spy.open.shift(-1))
        result["exit_date"] = pd.Series(sessions, index=sessions).shift(-5)
        history &= valid.shift(-1, fill_value=False) & valid.shift(-5, fill_value=False)
    frame = pd.DataFrame(result).loc[history]
    return frame.rename_axis("prediction_date").reset_index().assign(ticker=ticker)


def project_panel(histories, spy, groups, with_target=True):
    returns = pd.DataFrame({ticker: log_return(histories[ticker].close.reindex(spy.index))
                            for ticker in histories})
    project = []
    for ticker, sector in groups.items():
        others = [t for t, group in groups.items() if group == sector and t != ticker]
        peer = returns[others].mean(axis=1).where(returns[others].count(axis=1) >= 1)
        project.append(stock_rows(ticker, histories[ticker], spy, peer, with_target))
    return pd.concat(project, ignore_index=True).sort_values(["prediction_date", "ticker"])


def build_panels(prices, groups, members):
    histories = {ticker: a.set_index("date").sort_index()[FIELDS]
                 for ticker, a in prices.groupby("ticker")}
    spy = histories.pop("SPY").reindex(SESSIONS)
    fixed = project_panel(histories, spy, groups)

    broad, unavailable, excluded_reused = [], [], []
    known_reused = {"AAL", "C", "GM", "K", "META", "WELL", "WTW"}
    with ZipFile(ARCHIVE) as archive:
        names = set(archive.namelist())
        for ticker, dates in members.items():
            if ticker.replace(".", "-") in histories or ticker == "FB":
                continue
            if ticker in known_reused:
                excluded_reused.append(ticker)
                continue
            data = archive_prices(ticker, archive, names)
            if data is None:
                unavailable.append(ticker)
                continue
            rows = stock_rows(ticker, data, spy)
            rows = rows.loc[rows.prediction_date.isin(dates) & (rows.exit_date < "2021-01-01")]
            if len(rows):
                broad.append(rows)
            else:
                unavailable.append(ticker)
    extra = pd.concat(broad, ignore_index=True)
    return fixed, extra, unavailable, excluded_reused
