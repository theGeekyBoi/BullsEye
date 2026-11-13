# Price model follow-up study

We tested whether BullsEye could rank its 50 companies more reliably by giving
the model more information or more training examples. The forecast target was
the same five-session return relative to SPY, from the next open to the fifth
close. The final 2022–2023 test stays reserved.

## Results

None of the 16 variants established a ranking advantage over random ordering
in 2021. The existing trend rule scored +0.0243, with a 95% interval from
−0.0241 to +0.0725. It scored below zero in 2018, 2019, and 2020.
Reference Ridge led the earlier folds but fell from +0.0447 on average to
−0.0122 in 2021. Its 2021 difference against trend was −0.0365 with a paired
95% interval from −0.1104 to +0.0347.

| Experiment | 2018–2020 average rank | 2021 rank | 2021 top five minus middle ten (pp) |
| --- | ---: | ---: | ---: |
| Scaled trend benchmark | −0.0274 | +0.0243 | +0.1535 |
| Reference Ridge | +0.0447 | −0.0122 | +0.0475 |
| Market-adjusted moves | +0.0415 | −0.0132 | −0.1939 |
| Market- and sector-adjusted moves | +0.0370 | −0.0185 | −0.2349 |
| Volatility interactions | +0.0404 | −0.0153 | −0.0835 |
| Earlier data for the same companies | +0.0349 | −0.0286 | −0.2125 |
| Broader universe, same training period | −0.0013 | −0.0061 | −0.1917 |
| Broader universe and earlier data | +0.0042 | −0.0089 | −0.1591 |
| Classifier for top and bottom companies | +0.0245 | −0.0045 | −0.0300 |

Training with more stocks reduced 2021 absolute error from 2.4097 pp
for reference Ridge to 2.3885 pp with the broader and earlier cohort,
still above 2.3855 pp for zero. It did not deliver a useful ranking. The
top-five gain for scaled trend and for reference Ridge had intervals that
included zero. The broad boosted tree tied every company on 119 of 247
validation sessions.

## Experiment log

The [full log](experiment_log.csv) reports each variant in 2018, 2019, 2020, and
2021, with its training size, coverage, daily Spearman rank correlation,
forecast error, direction accuracy, and the realized difference between the
top five and middle ten. [Daily metrics](daily_metrics.csv) and
[2021 uncertainty intervals](intervals_2021.csv) are also available.
[Run settings and input hashes](run.json) record the exact feature sets.

- **Reference:** Rebuilt the previous Ridge on the frozen 50-company grid.
  Compared it with zero excess return and the prior 5%-scaled trend rule.
- **Stock-specific movement:** Estimated each stock's normal response to SPY
  using 60 and 120 trailing sessions, then calculated how far its observed
  return was from that expectation. Added a separate experiment with a
  same-sector, equal-weight peer return that excludes the stock being scored.
  Summed the unexplained moves over 1, 5, and 20 sessions.
- **Market conditions:** Tested whether volatility alters recent momentum
  or reversal by multiplying short stock moves by SPY's 20-session volatility
  relative to its 120-session level. Tested volatility both with and without
  the market-and-sector features.
- **Longer history:** Compared the same Ridge using 2010–2020 training history
  versus 2017–2020 for the 2021 fold. Tested 60-, 120-, and 240-session
  trend features separately, so adding older training examples is distinguishable
  from adding longer input windows.
- **Broader training:** Added stock-days from historical S&P 500 membership.
  Tested breadth by itself and in combination with earlier data. The most
  extensive 2021 training set had 1,086,397 rows across 543 ticker strings.
  Every evaluation used only the original BullsEye companies.
- **Other objectives:** On the larger panel, tested Huber regression, Ridge
  trained on daily percentiles, and shallow boosted trees. Also tested
  logistic models on the 50-company and broader panels that learn to
  distinguish each day's top and bottom fifth, then mapped the resulting
  scores back to decimal returns on an internal chronological calibration
  window. This is closer to what the simulator does with the extremes.

## Comparison rules and limitations

- Trained each annual fold on earlier outcomes only. Every fitting or
  calibration period ends before the next one starts. All candidates are
  scored on the frozen development keys. Recomputed outcomes matched to
  within 5e-13 in decimal units.
- Factor coefficients use sessions strictly before each return being
  explained. On the forecast date, the model uses daily information
  through that close. New stocks without a long lookback retain eligibility
  on the 30-session grid. Missing longer-window features are filled with
  zero after feature calculation.
- The wider cohort uses a community reconstruction of historical S&P 500
  membership and the intersection with FNSPID price availability. It
  excludes known reused ticker strings and masks downward breaks in the
  archive's price adjustment factor. Unavailable former members and
  unresolved symbol changes can still bias this sample. The
  [membership coverage log](membership_coverage.csv) lists the gaps. The
  processed supplemental panel had 498 ticker strings including the
  2009 warm-up period. The earliest fitted models start in 2010, so they
  use only the eligible subset of those rows.
- A model with a constant prediction on a given date has no Spearman score
  for that date. The log reports the number of ranked sessions so a tree
  with many ties cannot appear to cover the entire period. Top-five
  calculations average tied groups.
- The 95% intervals use blocks of 20 nearby trading sessions to account
  for overlapping five-day outcomes. They are exploratory development
  intervals. 2021 had already been inspected before this study.

## Team handoff

Keep the [scaled trend and zero forecasts](../price_refinement/predictions_2020_2021.csv)
as the simulator and fusion baselines. The [new candidate forecasts](predictions_2020_2021.csv)
follow the same four-column contract. They are available as an experimental
comparison. The 2020 and 2021 models were each selected using only the
earlier folds. Compare price, news, and fusion on the same eligible
company-date keys, then freeze the comparison rules before opening 2022–2023.
All 12,350 development predictions in 2021 join to the current news grid.
In 2020, 10,762 of the 12,183 price predictions join, with 1,421
earlier dates outside the prepared grid. A joined row can still have no
usable news. Apply the project's news coverage rules before training fusion.

The simulator can continue using the existing
[daily price export](../price_refinement/README.md#daily-prices-for-bryce).

For the final project, the most direct next comparison is news-only versus
price-only versus combined on the same keys. Keep the price-only fallback
on dates with missing news. Report ranking results and the simulator's
after-cost results separately. Price-only training can use the older
history. Fusion still requires price predictions generated before the
outcomes of its training rows were known.

## Verification

All nine existing unit tests passed. I checked the rolling residuals against
explicit historical regressions and verified that both saved models reproduce
all 24,533 published forecasts from daily price data alone. Removing all
prices after the forecast close produced the same scores for that date.
The stock-specific features were motivated by earlier research on
[residual momentum](https://pure.eur.nl/ws/files/46882404/ResidualMomentum-2011.pdf),
and the volatility experiment by research on
[short-term reversals](https://www.nber.org/system/files/working_papers/w17653/w17653.pdf).

## Reproduce

From the repository root, install the dependencies and prepare the frozen
price snapshot and company-day dataset using the
[prior reproduction steps](../price_refinement/README.md#reproduce-from-the-repository-root).
The study fetches the extra inputs when they are absent. It keeps vendor
prices and trained weights in the local ignored directories.

~~~powershell
python -m price_refinement.study
~~~

For inference, supply the daily, dividend- and split-adjusted OHLCV panel
with ticker, date, open, high, low, close, and volume. Include SPY and the
full project universe available at the requested date. Input data through
the requested close is sufficient. Keep at least 30 sessions of history
and include up to 240 prior sessions for complete longer-window features.

~~~powershell
python -m price_refinement.study_predict --input data_local/adjusted_ohlcv.csv --start 2021-01-04 --end 2021-12-23 --output data_local/price_study_forecasts.csv
~~~

[Historical membership source](https://github.com/fja05680/sp500/tree/a2430f2af0c79ddf0748e91de11bdeb1616ab5a7) ·
[FNSPID price archive](https://huggingface.co/datasets/Zihan1004/FNSPID/tree/main/Stock_price) ·
[Yahoo Finance download options](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html)
