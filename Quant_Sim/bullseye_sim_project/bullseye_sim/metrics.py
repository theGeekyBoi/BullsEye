import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _daily_returns(eq: pd.Series, initial: float) -> pd.Series:
    prev = eq.shift(1)
    prev.iloc[0] = initial  # first day measured from starting capital (opened at that session's open)
    return eq / prev - 1


def compute_metrics(daily: pd.DataFrame, trades: pd.DataFrame, initial: float) -> dict:
    eq = daily.set_index("date")["equity"]
    r = _daily_returns(eq, initial)
    n = len(r)
    total = eq.iloc[-1] / initial - 1
    ann = (1 + total) ** (TRADING_DAYS / n) - 1 if n > 0 else np.nan
    sd = r.std(ddof=1)
    sharpe = r.mean() / sd * np.sqrt(TRADING_DAYS) if sd and sd > 0 else np.nan
    path = pd.concat([pd.Series([initial]), eq.reset_index(drop=True)])
    mdd = (path / path.cummax() - 1).min()
    traded = trades["notional"].sum() if len(trades) else 0.0
    costs = trades["cost"].sum() if len(trades) else 0.0
    return dict(net_return=total, annualized_return=ann, sharpe=sharpe, max_drawdown=mdd,
                turnover=traded / eq.mean(), total_costs=costs, cost_drag=costs / initial,
                n_trades=len(trades), avg_positions=daily["n_positions"].mean(), n_sessions=n)


def benchmark_curves(o, c, dates, initial, bench, tickers) -> dict:
    """SPY buy-and-hold and daily-rebalanced equal-weight basket, same first-day convention
    (open of first session -> close) as the portfolio."""
    dates = pd.DatetimeIndex(dates)
    cc, oo = c.loc[dates], o.loc[dates]
    out = {}
    if bench in c:
        out["spy"] = initial * cc[bench] / oo[bench].iloc[0]
    tk = [t for t in tickers if t in c]
    if tk:
        r = cc[tk].pct_change()
        r.iloc[0] = cc[tk].iloc[0] / oo[tk].iloc[0] - 1
        out["equal_weight"] = initial * (1 + r.mean(axis=1)).cumprod()
    return out


def period_metrics(per: pd.DataFrame) -> dict:
    """per: one row per 5-session holding period with port_ret and optional spy_ret / ew_ret (decimals).
    Periods are open(t+1) -> close(t+5), the same window as the quant's targets, so they compare like-for-like."""
    out = {"n_periods": len(per)}
    if per.empty:
        return out
    comp = lambda s: float((1 + s).prod() - 1)
    out["period_compounded_port"] = comp(per["port_ret"])
    for name in ("spy", "ew"):
        col = f"{name}_ret"
        if col in per and per[col].notna().any():
            sub = per[per[col].notna()]  # score only periods that have a benchmark value
            ex = sub["port_ret"] - sub[col]
            sd = ex.std(ddof=1)
            out[f"period_compounded_{name}"] = comp(sub[col])
            out[f"mean_excess_vs_{name}"] = float(ex.mean())
            out[f"hit_rate_vs_{name}"] = float((ex > 0).mean())
            out[f"info_ratio_vs_{name}"] = float(ex.mean() / sd * np.sqrt(TRADING_DAYS / 5)) if sd > 0 else np.nan
    return out
