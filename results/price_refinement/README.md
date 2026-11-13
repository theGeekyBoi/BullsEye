# Price refinement and simulator handoff

## Meeting summary

The price pipeline runs end to end for all 50 project companies. The 2021
validation results do not establish a reliable price-only ranking or return
forecast. Use the simple trend ranking and zero forecast as comparison
baselines for Bryce's replay and Dhruv's evaluation. Keep Ridge as an
experimental comparison. The 2022–2023 outcomes were not used in this work.

| 2021 development method | Mean daily Spearman | Mean absolute error (percentage points) |
| --- | ---: | ---: |
| Zero forecast | Undefined (all companies tied) | 2.3855 |
| Original recent trend | +0.0243 | 3.4132 |
| Recent trend scaled to 5% | +0.0243 | 2.3858 |
| Earlier XGBoost, seed 42 | +0.0102 | 2.3929 |
| Ridge selected using 2018–2020 | −0.0122 | 2.4097 |
| New pairwise ranker, seeds 42 / 123 | +0.0102 / +0.0199 | 2.4020 / 2.3997 |
| Monthly Ridge refit, trailing 1 / 2 years | −0.0034 / −0.0144 | 2.4096 / 2.4136 |

The original XGBoost ranked companies over a narrow range. Its daily
predicted cross-company standard deviation averaged about 0.18 percentage
points, compared with about 3.32 in the outcomes. It also produced a positive
forecast about 95% of the time. Ridge's average daily ranking score fell
from +0.045 across the earlier folds to −0.012 in 2021. In 2021, the Ridge
top-five stocks beat the middle ten by 0.048 percentage points on average.
The 95% interval was −0.717 to +0.773 points. The trend ranking's 2021
95% interval was −0.030 to +0.074 and its score was negative in all three
earlier folds. These intervals resample ten-session blocks.

### What was tested

- Features from completed windows include SPY-relative returns, intraday and
  overnight changes, volatility, ranges, and drawdowns. Separate trials add
  same-day company ranks, sector-relative features, and normalized volume.
- Compared a fixed Ridge, Huber regression, recency-weighted Ridge, shallow
  XGBoost, a linear rank-target model, and date-grouped XGBoost pairwise
  ranking. Tried two random seeds for both boosted methods. Calibrated rank
  scores against completed returns from the last 63 prediction sessions
  within the training period. Tested two monthly refit schedules using only
  outcomes completed before the refit.
- Tested recent trend, one-session reversal, risk-adjusted trend, and low
  volatility rules. The fixed 5% trend multiplier preserves ordering and
  brings forecast size close to the zero benchmark in 2021. The multiplier
  is a conservative demonstration setting and has no independent validation.
- Fixed candidates were selected on earlier years. The 2020 Ridge forecasts
  use data through 2019 and selection folds from 2018–2019. The 2021 Ridge
  forecasts use data through 2020 and selection folds from 2018–2020.
  Boundary-crossing targets are excluded. The team had already seen 2021
  results, so the 2021 comparisons remain development results.

[Per-fold results](fold_metrics.csv) · [Run manifest and uncertainty](run.json) ·
[Monthly diagnostics](monthly_diagnostics_2021.csv) ·
[Company diagnostics](company_diagnostics_2021.csv) ·
[Sector diagnostics](sector_diagnostics_2021.csv) ·
[Prior quantitative report](../price_only_current/README.md)

## Prediction handoff

Use [predictions_2020_2021.csv](predictions_2020_2021.csv). It follows the
[shared prediction contract](../../EXPERIMENT_CONTRACT.md) and includes
price_refinement_candidate (Ridge), price_trend_shrunk, and price_zero.
Pick one model_name per replay. The 2021 files cover 247 eligible sessions
with 50 companies per session. PLTR enters the 2020 forecasts after enough
price history becomes available. Keep the prediction files on their original
date keys and intersect with eligible news keys for matched comparisons.

For the portfolio demo, use price_trend_shrunk as the price-only ranking and
price_zero as the no-signal control. Thresholds use decimal returns
(0.001 is 0.1 percentage points). A threshold sized for the original raw
trend will behave differently on the shrunk series. Record threshold and
cost assumptions during development and freeze them before the final test.

## Daily prices for Bryce

Run the daily export command below to generate data_local/daily_prices_replay.csv
from Yahoo Finance. The local table contains **89,124 rows**, the 50 companies
plus SPY, and sessions from January 3, 2017 to January 9, 2024. PLTR begins
on its first trading date, September 30, 2020. There are no missing sessions
after listing. The repo tracks the exporter, schema, [manifest](daily_prices_manifest.json)
and [audit](replay_audit.json). It keeps vendor price data in the local data folder.

| Columns | Use in replay |
| --- | --- |
| ticker, date | Unique price key |
| next_trading_date | Date of the fill after a signal at today's close |
| open, high, low, close | Prices on Yahoo's split-adjusted basis |
| volume | Reported source volume |
| dividend, capital_gain, cash_distribution | Cash distribution data on the source basis. Cash distribution is dividend plus capital gain. |
| adjusted_open, adjusted_close | Total-return price series for replay in fractional units or checking returns against the model target |
| split_ratio, non_standard_action | Audit fields. Five spin-off or special action rows are flagged. |

**Recommended for the class replay:** join predictions to prices on ticker
and prediction_date = date, then join to the same ticker on next_trading_date
to fill at adjusted_open. Value holdings at adjusted_close, use fractional
units and explicit trading costs, and apply the same basis to SPY. The
adjusted series embeds distributions and split effects. Leave the dividend
and split columns out of the ledger calculations in this mode. Label shares
as simulated units and report this price convention in the dashboard.

A separate cash-distribution replay can use the open/close pair with
cash_distribution. The units on that series are already split-adjusted.
The special action flags require manual review before treating it as an
actual-share ledger with all distributions accounted for.

The audit checked 24,533 development targets. All next-open dates matched.
The maximum absolute difference in excess return was 0.000162 percentage
points. Ordinary splits for AAPL, GE, NVDA and TSLA passed a continuity check.
The model uses the frozen earlier adjusted snapshot. Small revisions in a
later download can change adjusted prices.

[Yahoo adjusted-price definition](https://help.yahoo.com/kb/SLN28256.html) ·
[yfinance download options](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html) ·
[XGBoost learning to rank](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html)

## Reproduce from the repository root

~~~powershell
python -m pip install -r requirements-price-only-v2.txt
python -m price_only_v2.fetch_prices --companies CURRENT_COMPANIES.csv --output data_local/yahoo_adjusted_current_2016_2024.csv
python -m price_only_v2.build_dataset --snapshot data_local/yahoo_adjusted_current_2016_2024.csv --companies CURRENT_COMPANIES.csv --output data_local/price_current_50_2017_2023.csv
python -m price_refinement.experiment
python -m price_refinement.daily_prices
python -m price_refinement.audit_replay
~~~

The experiment saves the Ridge model locally. To predict from an input CSV
with ticker, prediction_date, sector and the 30 company and SPY OHLCV lags,
run:

~~~powershell
python -m price_refinement.predict --input data_local/inference_windows.csv --output data_local/price_forecasts.csv
~~~

The saved model bundle was fit through 2020. Use the published 2020
predictions for earlier dates. The inference entry point accepts feature-only
inputs and verifies that the dates follow the bundle's last training outcome.
