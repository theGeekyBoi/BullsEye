# Price-only rebuild for the original 50 companies

**Archived experiment (November 5, 2026).** The project now uses the 50
companies in [`final_dataset/dataset_summary.csv`](../final_dataset/dataset_summary.csv)
because they match the qualitative preparation branch. This rebuild uses the
earlier 50 in [`COMPANIES.csv`](../COMPANIES.csv), with only 23 overlapping.
Its saved models, predictions, and 2021 validation results must not be used
as baselines for the current 50-company price/news study. The commands below
reproduce the archived study only. See the [current shared contract](../EXPERIMENT_CONTRACT.md).

## Data sources and local rebuild

- [FNSPID archived price data](https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_price/full_history.zip): EA fallback and independent source comparisons. Save the archive as `data_local/full_history.zip`.
- [Yahoo Finance via yfinance](https://ranaroussi.github.io/yfinance/): adjusted open, high, low, close, reported volume, dividends, and splits for the other 49 companies and SPY. `BK` changed symbol to `BNY` in 2026, so the download queries `BNY` but keeps the historical company key `BK` ([BNY announcement](https://www.bny.com/corporate/global/en/about-us/newsroom/press-release/bny-announces-planned-change-of-stock-ticker-symbol-to-bny-130465.html)). Yahoo no longer supplies EA history after [EA ceased trading](https://www.ea.com/amp/news/ea-announces-completion-of-acquisition).
- The stock-price snapshot and derived 30-day dataset stay in ignored `data_local/`. Yahoo's [documentation](https://ranaroussi.github.io/yfinance/) warns that downloaded data are intended for personal use. The public repository includes code, hashes, audit counts, model forecasts, and aggregate results, not another copy of raw price history.

From the repository root, with Python 3.14 and the installed dependencies:

```powershell
python -m pip install -r requirements-price-only-v2.txt
python -m price_only_v2.fetch_prices
python -m price_only_v2.build_dataset
python -m price_only_v2.audit_prototype
python -m price_only_v2.audit
python -m price_only_v2.train_tabular
python -m price_only_v2.train_lstm
python -m price_only_v2.evaluate
python -m price_only_v2.verify
python -m price_only_v2.verify_inference
```

The exact snapshot used for the reported run has SHA-256
`c82f7fb41be0ec737083e2f927dd5d5bcc8a6bc0182c51f5ab24a454c6c21a02`.
Yahoo may revise adjusted historical values on a later download, so a fresh
snapshot can have a different hash and slightly different results. The derived
dataset hash, training settings, dates, and seeds are saved in run manifests.

## Data and audit decision

The rebuilt dataset has **87,994 examples from 2017–2023** for all 50 agreed
companies. Each row has a ticker, sector, prediction/entry/exit dates, a
30-session company OHLCV window, a 30-session SPY OHLCV window, and the three
five-session outcome values. Inputs end at the prediction day's close. Rows
whose outcomes cross a development-year boundary are not used for training or
validation. `2022–2023` outcomes are reserved for the final test and are not
loaded by the development trainer.

The [prototype audit](../results/price_only_v2/prototype_audit.json) found no
duplicate keys, no mismatches across 12,152,740 overlapping window values,
and no company-return mismatches among 83,612 checkable examples. Its company
list still differs from the agreed one. The [new-source audit](../results/price_only_v2/data_audit.json)
checked eight hand-selected examples against daily source values (300 inputs
and three targets each), including four holdings that cross split dates, an
MRK dividend, and the EA adjustment discontinuity. All checked values matched.
There were no invalid source OHLCV rows and one missing EA trading day.

**EA limitation:** FNSPID's EA adjusted-close ratio had a false 1.8% downward
step on 2020-07-06, without a split or dividend. The fetcher carries the
post-step adjustment factor backward to earlier dates. This removes the false
price movement. Thirteen later factor changes imply quarterly cash dividends
near $0.17–$0.19, consistent with [EA's 2021](https://ir.ea.com/press-releases/press-release-details/2021/Electronic-Arts-Reports-Q4-and-FY21-Financial-Results/default.aspx)
and [2023](https://ir.ea.com/press-releases/press-release-details/2023/Electronic-Arts-Reports-Strong-Q2-FY24-Results/default.aspx)
announcements. A complete independent EA daily price series is still unavailable,
so the one-company fallback remains a data-quality caveat before final testing.

**Volume decision:** Keep reported Yahoo volume unchanged and compare models
with and without it. Ordinary split examples showed no split-sized adjusted
price jump; the Yahoo and FNSPID volume values agree around TJX and NEE splits.
An additional `auto_adjust=True` versus `auto_adjust=False` check returned
identical Volume values for NEE, TJX, and GE over the sampled dates.
The treatment of unusual corporate actions is not proven for every ticker.
The price-only primary comparison therefore uses OHLC features and treats
volume as an ablation.

## Development experiments

- Rolling model selection: train through 2017, 2018, or 2019; check the next
  year (2018, 2019, or 2020). Each split excludes outcomes crossing its boundary.
- Compare 5-, 10-, and 30-session summaries, Ridge regularization, shallow
  XGBoost, market context, and a volume ablation. XGBoost checkpoint choice uses
  the **daily ranking score**, which is the project's primary metric.
- Select LSTM training duration using 2020 only, then train on 2017–2020 and
  check 2021. Compare price-and-market sequences with and without volume and
  run two fixed seeds.
- The 2021 validation year is used for the one development comparison shown in
  this branch. Do not repeatedly tune to it. The 2022–2023 final test is still
  unopened for model performance.

The [saved 2021 predictions](../results/price_only_v2/tabular/predictions_2021.csv)
and [LSTM predictions](../results/price_only_v2/lstm/predictions_2021.csv) use
`ticker,prediction_date,model_name,predicted_excess_return`. Forecasts are
decimal returns: `0.01` means one percentage point. The local evaluator joins
these keys to the local outcome table; actual returns are not copied into the
public forecast files. Model weights are regenerated locally under ignored
`models/price_only_v2/`.

`python -m price_only_v2.predict --input company_days.csv --output forecasts.csv`
runs any of the saved models on company-day rows containing the shared keys,
sector, 30 company OHLCV days, and 30 SPY OHLCV days. It does not require or
read outcome columns. The inference check confirms that all nine saved models
reproduce a held-out 2021 forecast from a single-company input row.

## 2021 finding

The earlier rolling checks favored five-session summaries: Ridge averaged
`0.052` and XGBoost `0.056` daily ranking correlation over 2018–2020. In 2021,
Ridge scored only `0.022`, XGBoost `0.007–0.010`, and both LSTM variants around
zero or below. Ridge's five-day-block 95% interval was `−0.045 to 0.085`.
All learned-model intervals include zero. Prediction error remained about
`2.56` percentage points, similar to predicting zero. [Full results](../results/price_only_v2/README.md)
include monthly, company, top-versus-middle, and illustrative long-only weekly
comparisons. No trained price-only model is validated as useful yet.
