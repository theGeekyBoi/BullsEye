# Price-only rebuild: development results

**Archived experiment.** These results use the earlier `COMPANIES.csv` group.
The current project uses the different 50 tickers in
[`final_dataset/dataset_summary.csv`](../../final_dataset/dataset_summary.csv).
Do not compare the scores below directly with current news or fusion models.
See the [shared experiment contract](../../EXPERIMENT_CONTRACT.md).

**Study group:** the original 50 in `COMPANIES.csv` (23 overlap the newer
quantitative prototype). **Training:** 2017–2020. **Model settings chosen on:**
2018–2020 rolling checks. **Reported validation:** 2021, 12,350 company-day
examples on 247 dates. **Reserved final test:** 2022–2023, not scored here.

| 2021 method | Average daily rank correlation | Average error, percentage points | Correct direction |
| --- | ---: | ---: | ---: |
| Predict zero | Undefined | **2.558** | 52.3%* |
| Recent relative trend | 0.012 | 3.670 | 50.2% |
| Ridge | **0.022** | 2.569 | 50.7% |
| XGBoost, seed 42 | 0.007 | 2.559 | 50.8% |
| XGBoost, seed 123 | 0.010 | 2.558 | 51.4% |
| LSTM without volume, seed 42 | −0.001 | 2.558 | 51.7% |
| LSTM without volume, seed 123 | ~0.000 | **2.556** | 52.3% |
| LSTM with volume, seed 42 | −0.017 | 2.566 | 50.7% |
| LSTM with volume, seed 123 | −0.021 | 2.566 | 48.6% |

*A zero forecast is counted as “nonpositive,” so its direction score is merely
the share of nonpositive outcomes, not evidence of predictive ability.

The 2021 ranking scores are weak. Five-date block resampling gave Ridge a 95%
interval of **−0.045 to +0.085**; all other learned-model intervals also include
zero. Ridge's top five beat its middle ten by only **0.047 percentage points per
prediction date**, with an interval crossing zero. Ridge's monthly ranking
score was positive in seven of twelve months and negative in five. Results
vary by company and random seed.

An illustrative long-only historical rule buys up to five positive-forecast
companies every fifth prediction date, weights them by rank (5/15 through
1/15), leaves unused weight in cash, and closes the held positions after five
sessions. It never shorts or sells shares it does not own. At an assumed
**10 basis points per side** cost, Ridge compounded to **+16.4%** during these
2021 intervals, versus **+29.8%** for SPY and **+28.6%** for the equally weighted
50-company group over the same intervals (reference returns shown before
costs). The simple recent-trend rule reached +31.8%, a small one-year difference
from SPY that is not evidence of a reliable strategy. Costs of 0 and 20 basis
points per side are also reported. LSTM forecasts were poorly calibrated for
the zero threshold: one seed produced only a single stock purchase all year.

Files:

- `source_manifest.json`, `build_manifest.json`: source links, snapshot hashes, company list, and rebuilt row counts.
- `data_audit.json`: source coverage, split/dividend case checks, company-list mismatch, volume policy.
- `prototype_audit.json`: internal consistency of the prior 83,862-row prototype.
- `tabular/rolling_trials.json`: every pre-2021 model comparison and fold score.
- `tabular/run_manifest.json`, `lstm/run_manifest.json`: chosen settings, seeds, dates, and data hash.
- `tabular/predictions_2021.csv`, `lstm/predictions_2021.csv`: shareable predictions with company-date keys.
- `tabular/metrics_2021.csv`, `lstm/metrics_2021.csv`: validation metrics.
- `evaluation_2021.json`: uncertainty, top-versus-middle, and long-only cost checks.
- `monthly_ranking_2021.csv`, `by_company_2021.csv`: stability breakdowns.

These results establish a reproducible price-only comparison for the future
news and fusion branches. They do not justify a final performance claim or
opening the reserved test period.
