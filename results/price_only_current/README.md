# Current 50-company price-only development run

This run uses the **same 50 tickers as the qualitative preparation branch**.
[`final_dataset/dataset_summary.csv`](../../final_dataset/dataset_summary.csv)
defines the company list; [`CURRENT_COMPANIES.csv`](../../CURRENT_COMPANIES.csv)
adds broad manually assigned business-area labels for the price model. The
earlier `results/price_only_v2/` run uses a different 50 and is archived.

## Data and split

The price-only rebuild uses Yahoo Finance adjusted daily open, high, low, and
close, plus reported volume, for these 50 companies and SPY. It constructs
30-session company and SPY windows and the same five-session excess-return
target as the existing quantitative file. Raw data and the 87,029-row derived
table from 2017–2023 stay under ignored `data_local/`. Source and dataset
SHA-256 values, retrieval settings, and row counts are in `source_manifest.json`
and `build_manifest.json`. Earlier 2017–2019 history exists for 49 companies;
PLTR begins in 2020.

The [data audit](data_audit.json) found exact agreement on 23,612 overlapping
2020–2021 company-day keys. Across those keys, the largest difference in
`target_excess_return` was 0.00000138 in decimal units (0.000138 percentage
points), consistent with small source revisions. It also checked four selected
split events and found no invalid daily source prices. The audit did not read
2022–2023 target values. Reported volume remains unchanged; its usefulness is
tested separately because the event checks do not settle every adjustment case.

The [qualitative grid audit](news_coverage_audit.json) confirms that the
prepared news and price grids have exactly the same company-date keys. This
does **not** mean every company has usable news. The current full-text coverage
flag is true for 29 tickers at some point in 2021 and 38 across 2020–2023.
Twelve tickers have no flagged full-text coverage in those four years. Another
1,256 company-days have article IDs while the coverage flag is false. The
qualitative team must resolve this before a fair price/news/fusion comparison.
The 2021 price-only metrics below cover all 50; future combined-model
comparisons must also show results on matched covered company-date keys.
The audit used the LFS company-day file on the
[`company-date/qualitative` branch](https://github.com/theGeekyBoi/BullsEye/tree/company-date/qualitative/data/qualitative_dataset)
with SHA-256 recorded in `news_coverage_audit.json`; it did not copy article
text into this results directory.

Model settings were chosen using rolling train-then-check folds ending in
2018, 2019, and 2020. The final models trained on 49,129 examples from
2017–2020, excluding outcomes that cross year boundaries. All 50 companies
had 12,350 validation examples across 247 dates in 2021. **The 2022–2023
final-test outcomes were not used for model choice or scoring.**

## 2021 validation

The main score is average daily Spearman correlation between predicted and
actual company rankings. Error is mean absolute forecast error in percentage
points. Values near zero in the ranking column show little evidence of useful
ordering.

| Model | Daily ranking score | Average error (pp) | Direction correct |
| --- | ---: | ---: | ---: |
| Predict zero | Undefined | **2.385** | 51.0%* |
| Recent price trend relative to SPY | **0.024** | 3.413 | 50.8% |
| Ridge, five-session summaries | 0.003 | 2.414 | 50.3% |
| XGBoost, seed 42 | 0.010 | 2.393 | 49.2% |
| XGBoost, seed 123 | 0.009 | 2.394 | 49.2% |
| LSTM, no volume, seed 42 | -0.018 | 2.390 | 48.6% |
| LSTM, no volume, seed 123 | -0.020 | 2.388 | 48.7% |
| LSTM, volume, seed 42 | 0.020 | 2.392 | 48.9% |
| LSTM, volume, seed 123 | -0.014 | 2.394 | 48.3% |

*The zero forecast is counted as nonpositive, so this direction score is the
share of nonpositive outcomes, not predictive skill.

Five-date block-resampled 95% ranking intervals include zero for every
learned method. The top-five versus middle-ten outcome differences also have
intervals spanning zero. The volume LSTM's ranking score even changes sign
between seeds. The results do **not** establish a dependable price-only
advantage. Monthly and company breakdowns are saved for inspection.

The illustrative long-only weekly rule buys up to five companies with
positive forecasts, weights them 5/15 through 1/15 by rank, leaves unused
weight in cash, and closes those shares after five sessions. It never shorts
or sells shares that are not held. At an assumed cost of 10 basis points per
side, the volume LSTM with seed 42 compounded to +40.9% in these 2021
intervals, versus +30.0% for SPY before costs. Its other seed reached +21.2%,
and its ranking interval includes zero. This one-year backtest is therefore
**not evidence that the strategy works reliably**. The evaluator also reports
0- and 20-basis-point scenarios and an equal-weight company comparator.

## Reproduce the run

From the repository root, using the dependencies in
[`requirements-price-only-v2.txt`](../../requirements-price-only-v2.txt):

```powershell
python -m price_only_v2.fetch_prices --companies CURRENT_COMPANIES.csv --output data_local/yahoo_adjusted_current_2016_2024.csv
python -m price_only_v2.build_dataset --snapshot data_local/yahoo_adjusted_current_2016_2024.csv --companies CURRENT_COMPANIES.csv --output data_local/price_current_50_2017_2023.csv
python -m price_only_v2.audit_current
python -m price_only_v2.train_tabular --data data_local/price_current_50_2017_2023.csv --companies CURRENT_COMPANIES.csv --output results/price_only_current/tabular --model-dir models/price_only_current/tabular
python -m price_only_v2.train_lstm --data data_local/price_current_50_2017_2023.csv --companies CURRENT_COMPANIES.csv --output results/price_only_current/lstm --model-dir models/price_only_current/lstm
python -m price_only_v2.evaluate --data data_local/price_current_50_2017_2023.csv --results-dir results/price_only_current
python -m price_only_v2.verify --data data_local/price_current_50_2017_2023.csv --companies CURRENT_COMPANIES.csv --results-dir results/price_only_current
python -m price_only_v2.verify_inference --data data_local/price_current_50_2017_2023.csv --results-dir results/price_only_current --model-dir models/price_only_current
```

A later Yahoo download can revise historical adjusted values. The manifests
identify the exact snapshot for these results. Saved shareable predictions
contain only `ticker,prediction_date,model_name,predicted_excess_return`;
actual targets are joined locally inside the evaluator.
