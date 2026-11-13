# Price and news fusion

This pipeline joins price and news forecasts by ticker and prediction date.
It averages their five-session excess-return predictions when the news
coverage flag is true and a finite news forecast is present. Otherwise,
it uses the price forecast. The exported file uses the
[shared prediction format](../EXPERIMENT_CONTRACT.md).

## Run on development data

From the repo root, use the dependencies and local price target file
described in the [price handoff](../results/price_refinement/README.md#reproduce-from-the-repository-root).
The default news eligibility file is the Git LFS company-day news grid.

```powershell
python -m fusion.run --demo --output data_local/fusion_demo
```

The demo uses the saved scaled trend price forecasts and seeded random news
forecasts. It omits about a quarter of the news forecasts before applying the
real coverage flag. It uses no outcomes to generate synthetic forecasts.
The demo checks data flow and fallback behavior. Its metrics do not measure
whether actual news helps.

The saved price grid produced 24,533 demo forecasts across both years.
The coverage rule and synthetic omissions allowed a blend on 9,841 rows.
The other 14,692 kept their original price forecast.

When the news forecasts are ready, replace the synthetic input:

```powershell
python -m fusion.run --news data_local/news_predictions.csv --news-model finbert_ridge --output data_local/fusion_development
```

Replace finbert_ridge with the exact model name in the news CSV. The news
team should supply forecasts generated from information available at the
prediction close and use the agreed development splits. The runner can
verify keys and schema. Forecast provenance comes from the training run
that produced the CSV.

Use --coverage to pass a revised eligibility file with one row per
company-day. The default flag is include_in_multimodal_comparison.
Use --coverage-flag if the revised file uses a different name. A true flag
and an available forecast are both required. For a covered day with zero
articles, the news team's forecast policy determines whether there is a
news forecast to combine.

The default price signal is price_trend_shrunk. Pass --price-model to
choose a different signal from the saved file. Pass --price with a
different CSV to use another price model. Scaled trend is still included
as a benchmark. Use --trend only to point to another file containing the
scaled trend forecasts.

## Outputs

| File | Use |
| --- | --- |
| combined_predictions.csv | Four-column forecast file for the simulator and dashboard |
| comparison_metrics.csv | Yearly ranking, mean absolute error, direction accuracy, coverage, and top-five versus middle-ten results |
| coverage.csv | Counts of blended rows and fallbacks by reason |
| news_usage_by_company_day.csv | Which forecasts used news on each ticker and date |
| run.json | Input hashes, selected models, weights, and development period |
| synthetic_news_predictions.csv | Demo-only input for reproducing the file import path |

The matched-news report evaluates price, news, and the blend on identical
company-days, plus the zero and scaled trend benchmarks. The full price
grid report evaluates price and the blend with fallback on every eligible
price forecast row, plus the benchmarks. When scaled trend is itself the
price input, the price row represents both. The grid contains the full
50 companies after all are eligible.

Daily ranking uses average Spearman correlation on days with a defined
ranking. The top-five versus middle-ten calculation requires at least
20 companies to keep the groups disjoint, averages ties, and reports how
many dates qualified. Errors and realized spreads are in percentage points.
Exported forecasts are in decimal returns. Simulator execution and costs
belong to the separate portfolio replay.

The runner evaluates only 2020–2021 and excludes outcomes that cross a year
boundary. The 2022–2023 final test remains reserved. Running it will require
a separate evaluation entry point after the team freezes the settings.

## Proposed comparison dates for Dhruv and Aaditya

Use the 50 tickers in the [current dataset summary](../final_dataset/dataset_summary.csv).
Use eligible 2020 dates for fitting the news model and eligible 2021 dates
for comparing forecasts. Keep outcomes within their respective year.
Dhruv's draft split also skips the first five sessions of 2021. To match
that split, use January 11 through December 23, 2021 for validation signals.
This proposed window has 242 sessions and 12,100 price company-days before
applying news coverage. The team should confirm the start date before
publishing results. Existing runs retain their original dates.

Use the same coverage flag and intersect by ticker and prediction date for
the matched price, news and combined comparison. Report how many rows
remain and retain the separate full-grid comparison. Send the news
prediction CSV, model name, eligibility CSV, and training date range
with the first real news handoff.

```powershell
python -m fusion.run --news data_local/news_predictions.csv --news-model finbert_ridge --validation-start 2021-01-11 --output data_local/fusion_development
```

## Simulator handoff for Bryce

Generate the audited daily price export, then prepare inputs for the
[simulator on quant_model](https://github.com/theGeekyBoi/BullsEye/tree/quant_model/Quant_Sim/bullseye_sim_project).
The adapter exports 2021 only, converts decimal returns to percentage points
once, includes SPY, and maps the dividend- and split-adjusted open and close
to the simulator's price columns. Keep share counts labeled as simulated units.

```powershell
python -m price_refinement.daily_prices
python -m fusion.run --demo --validation-start 2021-01-11 --output data_local/team_handoff_demo
python -m fusion.replay_inputs --predictions results/price_refinement/predictions_2020_2021.csv --model price_trend_shrunk --start 2021-01-11 --output data_local/replay_price
python -m fusion.replay_inputs --predictions data_local/team_handoff_demo/combined_predictions.csv --model synthetic_demo_fusion_equal_weight --output data_local/replay_combined
```

Each replay folder contains sim_predictions.csv and sim_daily_prices.csv.
Pass those to the simulator's load_predictions and load_prices functions
and run run_simulation with the same config, January 11 start date, and
December 31 end date for both inputs. When real fusion forecasts arrive,
pass the new combined_predictions.csv with model fusion_equal_weight.

The development smoke test used simulator commit 8fc1ee5 and its default
configuration. Each replay took 12,100 forecasts and 12,852 daily price
rows (50 companies plus SPY). The synthetic blend used news on 5,145 of
12,100 rows and fell back to price on 6,955. The simulator recorded 17
price-only trades and 10 synthetic-fusion trades. All 27 used the next
session's open, and cash and holdings passed the long-only checks.
These counts verify the connection. The synthetic forecasts provide no
evidence about the value of news.
