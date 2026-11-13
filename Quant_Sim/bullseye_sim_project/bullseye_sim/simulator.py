"""Long-only rank-based portfolio simulator.

Timeline for one rebalance: prediction made at the CLOSE of session t  ->  trades at the OPEN of t+1.
Sells execute before buys so sale proceeds can fund purchases at the same open.
"""
import math
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .config import SimConfig
from .metrics import compute_metrics, benchmark_curves, period_metrics


@dataclass
class SimResult:
    trades: pd.DataFrame
    daily: pd.DataFrame      # date, cash, holdings_value, equity, n_positions
    holdings: pd.DataFrame   # date, ticker, shares, value (close)
    log: pd.DataFrame        # skipped trades and notable non-events
    periods: pd.DataFrame    # one row per 5-session holding period (portfolio vs SPY / equal-weight)
    metrics: dict
    benchmarks: dict


def sell_fraction(k: int, m: int, min_frac: float) -> float:
    """Fraction of a position sold for the k-th worst (k=1 is worst) of m sell candidates.
    Linear from 100% (worst) down to min_frac (closest to the threshold)."""
    if m <= 1:
        return 1.0
    return 1.0 - (1.0 - min_frac) * (k - 1) / (m - 1)


def buy_weights(m: int) -> np.ndarray:
    """Linear rank weights for m buy candidates (best first): m, m-1, ..., 1, normalized."""
    w = np.arange(m, 0, -1, dtype=float)
    return w / w.sum()


def _equity_at_open(state, o, cfp, d):
    """Portfolio value at session d's open before any trades (previous close if an open is missing)."""
    v = state["cash"]
    for t, q in state["shares"].items():
        if q > 0:
            px = o.at[d, t] if not pd.isna(o.at[d, t]) else cfp.at[d, t]
            if not pd.isna(px):
                v += q * px
    return v


def _period_table(periods, daily_df, sessions, cfg, bench):
    """Portfolio return for each full 5-session window open(t+1) -> close(t+5), joined to SPY / equal-weight.
    Only built when rebalancing every 5 sessions, so windows match the target horizon and never overlap."""
    if cfg.rebalance_every_sessions != 5 or not periods:
        return pd.DataFrame(columns=["signal_date", "trade_date", "end_date", "port_ret"])
    pos = {s: i for i, s in enumerate(sessions)}
    eq = daily_df.set_index("date")["equity"]
    rows = []
    for p in periods:
        end_i = pos[p["trade_date"]] + 4
        if end_i >= len(sessions):
            continue  # incomplete final window
        end = sessions[end_i]
        rows.append(dict(signal_date=p["signal_date"], trade_date=p["trade_date"], end_date=end,
                         port_ret=eq[end] / p["start_equity"] - 1))
    per = pd.DataFrame(rows, columns=["signal_date", "trade_date", "end_date", "port_ret"])
    if bench is not None and not per.empty:
        b = bench.rename(columns={"date": "signal_date"})
        per = per.merge(b[["signal_date", "spy_ret", "ew_ret"]], on="signal_date", how="left")
    return per


def _rebalance(cfg, d, sig, day_preds, o, cfp, state, trades, log):
    rate = cfg.cost_rate
    thr = cfg.neutral_threshold_pp
    cash, shares = state["cash"], state["shares"]

    p = day_preds.sort_values(["pred_diff_pp", "ticker"], ascending=[False, True]).reset_index(drop=True)
    p["rank"] = np.arange(1, len(p) + 1)  # 1 = best predicted
    p = p[p["ticker"].isin(o.columns) & (p["ticker"] != cfg.benchmark_ticker)]

    def rec(t, side, q, px, row):
        nonlocal cash
        notional = q * px
        cost = notional * rate
        cash += (notional - cost) if side == "SELL" else -(notional + cost)
        shares[t] = shares.get(t, 0) + (q if side == "BUY" else -q)
        trades.append(dict(signal_date=sig, trade_date=d, ticker=t, side=side, shares=int(q),
                           price=px, notional=notional, cost=cost, rank=int(row["rank"]),
                           pred_diff_pp=row["pred_diff_pp"]))

    # ---- SELLS: sell zone = predicted below -thr; k=1 is the worst-ranked ----
    sell_zone = p[p["pred_diff_pp"] < -thr].sort_values("rank", ascending=False)
    m = len(sell_zone)
    for k, row in enumerate(sell_zone.itertuples(index=False), 1):
        r = row._asdict()
        t = r["ticker"]
        held = shares.get(t, 0)
        if held <= 0:
            continue  # nothing owned -> no sell trade recorded
        px = o.at[d, t]
        if pd.isna(px):
            log.append(dict(trade_date=d, ticker=t, event="skip_sell_missing_open"))
            continue
        q = min(held, math.ceil(sell_fraction(k, m, cfg.min_sell_fraction) * held - 1e-9))
        if q > 0:
            rec(t, "SELL", q, px, r)

    # ---- BUYS: buy zone = predicted above +thr; best-ranked first ----
    buy_zone = p[p["pred_diff_pp"] > thr].sort_values("rank")
    if cfg.max_buy_names:
        buy_zone = buy_zone.head(cfg.max_buy_names)
    if buy_zone.empty:
        log.append(dict(trade_date=d, ticker="", event="no_buy_candidates"))
    elif cash <= 0:
        log.append(dict(trade_date=d, ticker="", event="no_cash"))
    else:
        px_of = lambda t: o.at[d, t] if not pd.isna(o.at[d, t]) else cfp.at[d, t]
        equity = cash + sum(q * px_of(t) for t, q in shares.items() if q > 0 and not pd.isna(px_of(t)))
        cap_value = cfg.max_position_pct * equity
        budget = cash  # allocate cash available after sells
        w = buy_weights(len(buy_zone))
        for wt, row in zip(w, buy_zone.itertuples(index=False)):
            r = row._asdict()
            t = r["ticker"]
            px = o.at[d, t]
            if pd.isna(px) or px <= 0:
                log.append(dict(trade_date=d, ticker=t, event="skip_buy_missing_open"))
                continue
            room = max(cap_value - shares.get(t, 0) * px, 0.0)
            target = min(budget * wt, room)
            unit = px * (1 + rate)  # price including cost
            q = min(math.floor(target / unit + 1e-9), math.floor(cash / unit + 1e-9))
            if q > 0:
                rec(t, "BUY", q, px, r)
    state["cash"] = cash
    assert state["cash"] >= -1e-6 and all(v >= 0 for v in shares.values()), "invariant broken"


def run_simulation(preds: pd.DataFrame, prices: pd.DataFrame, cfg: SimConfig,
                   start=None, end=None, period_benchmarks=None) -> SimResult:
    cfg.validate()
    start = pd.Timestamp(start) if start is not None else None
    end = pd.Timestamp(end) if end is not None else None
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    preds = preds.copy()
    preds["date"] = pd.to_datetime(preds["date"])
    if prices.duplicated(["date", "ticker"]).any():
        raise ValueError("duplicate date/ticker rows in price table")
    if preds.duplicated(["date", "ticker"]).any():
        raise ValueError("duplicate date/ticker rows in predictions (filter to one model)")

    o = prices.pivot(index="date", columns="ticker", values="open").sort_index()
    c = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    cf = c.ffill()
    cfp = cf.shift(1).reindex(o.index)  # previous close: fallback valuation when an open is missing
    bench = cfg.benchmark_ticker
    sessions = list(c[bench].dropna().index) if bench in c else list(c.index)
    sessions = [s for s in sessions if (start is None or s >= start) and (end is None or s <= end)]

    pred_dates = set(preds["date"])
    first = next((i for i, s in enumerate(sessions) if s in pred_dates), None)
    if first is None or first + 1 >= len(sessions):
        raise ValueError("no usable prediction dates inside the requested window")
    trade_map = {sessions[p + 1]: sessions[p]
                 for p in range(first, len(sessions) - 1, cfg.rebalance_every_sessions)}
    by_date = {d: g for d, g in preds.groupby("date")}

    state = {"cash": cfg.initial_capital, "shares": {}}
    trades, log, daily, hold_rows, periods = [], [], [], [], []
    for d in sessions[first + 1:]:
        if d in trade_map:
            sig = trade_map[d]
            periods.append(dict(signal_date=sig, trade_date=d, start_equity=_equity_at_open(state, o, cfp, d)))
            if sig not in by_date:
                log.append(dict(trade_date=d, ticker="", event="no_predictions_for_signal_date"))
            else:
                _rebalance(cfg, d, sig, by_date[sig], o, cfp, state, trades, log)
        hv = 0.0
        npos = 0
        for t, q in state["shares"].items():
            if q > 0:
                v = q * cf.at[d, t]
                hv += v
                npos += 1
                hold_rows.append(dict(date=d, ticker=t, shares=q, value=v))
        daily.append(dict(date=d, cash=state["cash"], holdings_value=hv,
                          equity=state["cash"] + hv, n_positions=npos))

    tcols = ["signal_date", "trade_date", "ticker", "side", "shares", "price", "notional",
             "cost", "rank", "pred_diff_pp"]
    trades_df = pd.DataFrame(trades, columns=tcols)
    daily_df = pd.DataFrame(daily)
    log_df = pd.DataFrame(log, columns=["trade_date", "ticker", "event"])
    hold_df = pd.DataFrame(hold_rows, columns=["date", "ticker", "shares", "value"])
    first_trade_day = sessions[first + 1]
    per_df = _period_table(periods, daily_df, sessions, cfg, period_benchmarks)
    bm = benchmark_curves(o, c, list(daily_df["date"]), cfg.initial_capital, bench,
                          [t for t in c.columns if t != bench])
    metrics = compute_metrics(daily_df, trades_df, cfg.initial_capital)
    metrics.update(period_metrics(per_df))
    metrics["start_date"], metrics["end_date"] = str(first_trade_day.date()), str(sessions[-1].date())
    for name, eq in bm.items():
        metrics[f"{name}_net_return"] = eq.iloc[-1] / cfg.initial_capital - 1
    return SimResult(trades_df, daily_df, hold_df, log_df, per_df, metrics, bm)
