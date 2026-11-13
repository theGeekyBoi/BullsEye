import math
import pandas as pd
import pytest
from dataclasses import replace

from bullseye_sim import SimConfig, run_simulation, freeze_config, verify_frozen
from bullseye_sim.simulator import sell_fraction, buy_weights

TICKERS = ["A", "B", "C"]
DATES = pd.bdate_range("2023-01-02", periods=30)


def make_prices(tickers=TICKERS, opens=None):
    rows = []
    for t in tickers + ["SPY"]:
        for d in DATES:
            px = 100.0
            if opens and (t, d) in opens:
                px = opens[(t, d)]
            rows.append(dict(date=d, ticker=t, open=px, close=100.0))
    return pd.DataFrame(rows)


def preds_on(day_idx, values):
    d = DATES[day_idx]
    return pd.DataFrame([dict(date=d, ticker=t, pred_diff_pp=v) for t, v in values.items()])


def cfg(**kw):
    base = dict(neutral_threshold_pp=0.5, rebalance_every_sessions=5, max_position_pct=0.5,
                min_sell_fraction=0.25, commission_bps=0, slippage_bps=0, initial_capital=100_000)
    base.update(kw)
    return SimConfig(**base)


def check_invariants(res):
    assert (res.daily["cash"] >= -1e-6).all()
    assert (res.holdings["shares"] > 0).all() if len(res.holdings) else True


def test_formulas():
    assert sell_fraction(1, 4, 0.25) == 1.0
    assert sell_fraction(4, 4, 0.25) == pytest.approx(0.25)
    assert sell_fraction(1, 1, 0.25) == 1.0
    assert buy_weights(3).tolist() == pytest.approx([0.5, 1 / 3, 1 / 6])


def test_empty_portfolio_sell_signal_records_no_trade():
    p = preds_on(0, {"A": -3, "B": -2, "C": -1})
    res = run_simulation(p, make_prices(), cfg())
    assert res.trades.empty
    assert (res.daily["equity"] == 100_000).all()


def test_no_buy_candidates_keeps_cash_and_logs():
    p = preds_on(0, {"A": 0.2, "B": -0.1, "C": 0.0})
    res = run_simulation(p, make_prices(), cfg())
    assert res.trades.empty
    assert "no_buy_candidates" in set(res.log["event"])


def test_no_available_cash_buys_nothing():
    p = preds_on(0, {"A": 3, "B": 2, "C": 1})
    res = run_simulation(p, make_prices(), cfg(initial_capital=50))  # < one share
    assert res.trades.empty
    check_invariants(res)


def test_fully_invested_then_no_cash_for_new_buys():
    p = pd.concat([preds_on(0, {"A": 3, "B": 2, "C": 1}),
                   preds_on(5, {"A": 3, "B": 2, "C": 1})])
    res = run_simulation(p, make_prices(), cfg(max_position_pct=1.0))
    check_invariants(res)
    assert res.daily["cash"].min() >= 0


def test_buy_to_neutral_holds_then_sell_reduces():
    p = pd.concat([preds_on(0, {"A": 3, "B": -3, "C": 0}),
                   preds_on(5, {"A": 0.1, "B": 0, "C": 0}),    # A crosses buy -> neutral
                   preds_on(10, {"A": -2, "B": 0, "C": 0})])   # A crosses neutral -> sell
    res = run_simulation(p, make_prices(), cfg(max_position_pct=1.0))
    t = res.trades
    a = t[t.ticker == "A"]
    assert list(a["side"]) == ["BUY", "SELL"]            # nothing traded at the neutral rebalance
    assert a.iloc[0]["trade_date"] == DATES[1]
    assert a.iloc[1]["trade_date"] == DATES[11]
    assert a.iloc[1]["shares"] == a.iloc[0]["shares"]    # lone sell candidate -> 100% sold
    assert "B" not in set(t.ticker)                       # B never owned -> no sell trade
    check_invariants(res)


def test_trades_price_at_next_session_open_with_costs():
    opens = {("A", DATES[1]): 110.0}
    p = preds_on(0, {"A": 3, "B": 0, "C": 0})
    res = run_simulation(p, make_prices(opens=opens), cfg(commission_bps=5, slippage_bps=5))
    tr = res.trades.iloc[0]
    assert tr.price == 110.0 and tr.trade_date == DATES[1] and tr.signal_date == DATES[0]
    assert tr.cost == pytest.approx(tr.notional * 0.001)


def test_position_cap_and_whole_shares():
    p = preds_on(0, {"A": 3, "B": 2, "C": 1})
    res = run_simulation(p, make_prices(), cfg(max_position_pct=0.2))
    assert (res.trades["shares"] == res.trades["shares"].astype(int)).all()
    assert (res.trades["notional"] <= 0.2 * 100_000 + 1e-6).all()


def test_missing_open_skips_and_logs():
    opens = {("A", DATES[1]): float("nan")}
    p = preds_on(0, {"A": 3, "B": 2, "C": 0})
    res = run_simulation(p, make_prices(opens=opens), cfg())
    assert "A" not in set(res.trades.ticker)
    assert "skip_buy_missing_open" in set(res.log["event"])


def test_top_n_option():
    p = preds_on(0, {"A": 3, "B": 2, "C": 1})
    res = run_simulation(p, make_prices(), cfg(max_buy_names=1))
    assert set(res.trades.ticker) == {"A"}


def test_frozen_config_detects_edit(tmp_path):
    f = str(tmp_path / "final.yaml")
    freeze_config(cfg(), f)
    verify_frozen(f)
    open(f, "a").write("\n# tampered\n")
    with pytest.raises(RuntimeError):
        verify_frozen(f)


def test_period_metrics_and_benchmarks():
    from bullseye_sim.metrics import period_metrics
    per = pd.DataFrame(dict(port_ret=[0.02, -0.01, 0.03], spy_ret=[0.01, 0.0, 0.01], ew_ret=[0.0, 0.0, 0.0]))
    m = period_metrics(per)
    assert m["n_periods"] == 3
    assert m["mean_excess_vs_spy"] == pytest.approx((0.01 - 0.01 + 0.02) / 3)
    assert m["hit_rate_vs_spy"] == pytest.approx(2 / 3)
    assert m["period_compounded_port"] == pytest.approx(1.02 * 0.99 * 1.03 - 1)


def test_simulation_builds_period_table_with_benchmarks():
    p = pd.concat([preds_on(0, {"A": 3, "B": 0, "C": 0}), preds_on(5, {"A": 3, "B": 0, "C": 0})])
    bm = pd.DataFrame(dict(date=[DATES[0], DATES[5]], spy_ret=[0.01, 0.02], ew_ret=[0.0, 0.0]))
    res = run_simulation(p, make_prices(), cfg(), period_benchmarks=bm)
    assert len(res.periods) >= 2
    assert list(res.periods["spy_ret"][:2]) == [0.01, 0.02]
    assert res.periods["trade_date"].iloc[0] == DATES[1]
    assert res.periods["end_date"].iloc[0] == DATES[5]  # open(t+1) -> close(t+5)
    assert "info_ratio_vs_spy" in res.metrics
