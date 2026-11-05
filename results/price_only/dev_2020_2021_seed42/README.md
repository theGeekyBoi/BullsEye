# First price-only development results

These results were generated from the committed 30-session quantitative CSV
with seed 42. The run trained on 10,762 company-day examples from February 13
through December 23, 2020 and evaluated on 12,350 examples across 247 dates
from January 4 through December 23, 2021. Outcomes crossing a year boundary
were excluded. The 2022 and later results were not used.

| Method | Mean daily Spearman | MAE, percentage points | Sign accuracy | Coverage |
| --- | ---: | ---: | ---: | ---: |
| Zero | N/A (constant ranking) | 2.3855 | 51.0% | 100% |
| Recent price trend | 0.0243 | 3.6231 | 51.1% | 100% |
| XGBoost | -0.0091 | 2.3931 | 49.1% | 100% |
| LSTM, prices only | -0.0156 | 2.3853 | 51.0% | 100% |
| LSTM, prices and volume | -0.0005 | 2.3852 | 50.9% | 100% |

Recent price trend has the highest mean daily ranking score, the project's
primary prediction metric. Its score is close to zero, and its prediction
error is substantially larger than the zero baseline's. The first trained
price-only models do not show a convincing validation gain. This run is a
development baseline, not a final-system efficacy claim.

`predictions_2021.csv` contains every model's forecast in **decimal excess
return** units, with the shared ticker and prediction-date keys. The metrics
and exact settings are in `metrics_2021.csv`, `metrics_2021.json`, and
`run_manifest.json`. `training_history.json` records early-stopping behavior.
Weights and standardizers are kept outside Git under the ignored `models/`
directory; `python -m price_only.train` regenerates them.

Next, confirm the raw price and calendar source files needed by the existing
quantitative builder, settle the volume adjustment question, and coordinate
the final dates and company list with the news-coverage audit. Keep the 2022–23
test period untouched until those choices are fixed.
