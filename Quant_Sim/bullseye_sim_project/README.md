# BullsEye long-only portfolio simulator

Turns saved model predictions into a reproducible buy/hold/sell replay. Works on made-up data first.

## Run it
```
pip install -r requirements.txt
python -m pytest                      # 11 checks, incl. the four step-6 edge cases
python run_demo.py                    # made-up prices + predictions, whole range
python make_synthetic.py --out-dir demo_data
python tune_validation.py --prices demo_data/daily_prices.csv --preds demo_data/predictions.csv   # validation (2021) grid -> frozen config
python run_heldout_test.py --prices demo_data/daily_prices.csv --preds demo_data/predictions.csv  # test (2022-23), once only
```
Real data: pass your quant's daily price table and each model's saved predictions (`--model name` if the file holds several).

## Input formats
- Predictions: `date, ticker, pred_diff_pp [, model]`. `date` = close of day t; value = predicted % return vs SPY (pp).
- Daily prices: `date, ticker, open, close`, one row per company-session plus SPY. **Opens and closes must share one adjustment basis.**

## The rules and why
| Choice | Setting | Reason |
|---|---|---|
| Signal vs. SPY | Buy if prediction > +threshold, sell if < -threshold, otherwise neutral/unchanged | Predictions are already relative to SPY, so sign = "beats / trails the market"; the neutral band avoids trading on noise. Rank then decides *how much*. |
| Rebalance | Every 5 sessions, signal at close of t, trade at open of t+1 | Matches the 5-session target (t+1 open to t+5 close), so bets don't overlap, and uses only information available at the signal close. |
| Sell sizing | Worst-ranked sell candidate sells 100% of its position; linear down to 25% for the one nearest the threshold. Fraction k of m: `1 - 0.75*(k-1)/(m-1)`, shares sold = ceil(fraction x held). Never more than held. | Rank-proportional as specified; a floor stops a barely-negative name from triggering a trivial trade; ceil avoids rounding a sale to zero. |
| Buy sizing | Cash after sells split with linear rank weights (best gets m, next m-1, ... 1), each name capped at 10% of portfolio value at the trade open; leftover stays cash | Simple, transparent, rank-monotonic; cap limits single-stock risk. Leftover is not redistributed to keep the rule easy to verify. |
| Shares | Whole shares, rounded down; spend including costs never exceeds cash | Prevents negative cash/shares by construction (asserted every rebalance). |
| Costs | 5 bps commission + 5 bps slippage per side on traded value; $100k start; 0% on cash | Round placeholders for liquid large caps; all configurable. |
| Missing open | Trade skipped and written to the log | Avoids pricing with invented data. |
| Top-five test | `max_buy_names: 5` | Reproduces the rules-sheet weekly top-five test inside the same simulator. |
| Tuning | Grid over threshold x cap x min sell fraction on 2021 only, scored on net Sharpe; best is written to `final_config.yaml` with a SHA-256 lock. The test script refuses an edited config and refuses a second run. | Enforces "freeze before the held-out test; calculate once." |

Outputs: `trades.csv`, `daily.csv` (cash/holdings/equity), `holdings.csv`, `skipped_log.csv`, `metrics.json` (net and annualized return, Sharpe, max drawdown, turnover, cost drag) with SPY buy-and-hold and daily-rebalanced equal-weight comparisons.

## Caveats
- Yahoo OHLC looks split/dividend-adjusted (per quant's Script 03); the simulator is correct if opens and closes are on that same basis, but it should be confirmed that the daily table comes from one source.
- Share counts on adjusted prices are a modelling convenience, not real historical share counts.
- Every simulation window ends marked to market at its last close; no liquidation cost is charged.
- Rebalance cadence is fixed at 5 sessions inside tuning; change it in config.yaml if you want to test other schedules.
