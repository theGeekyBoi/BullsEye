# Price-only development run

This pipeline trains the first BullsEye price-only models using
[`final_dataset/quant_dataset_30day.csv`](../final_dataset/quant_dataset_30day.csv).
The dataset CSV is stored with Git LFS. Run `git lfs pull` if a fresh checkout
contains only an LFS pointer.

## Run

Use Python 3.14 and install the versions in `requirements-price-only.txt`.
From the repository root:

```powershell
python -m pip install -r requirements-price-only.txt
python -m unittest discover -s price_only/tests -v
python -m price_only.train --data final_dataset/quant_dataset_30day.csv
python -m price_only.verify_results
```

The default run uses a fixed seed of 42 and CPU training. `--device cuda` is
available when a compatible GPU is installed. `--max-epochs`, `--threads`,
`--train-year`, `--validation-year`, `--output`, and `--model-dir` are configurable.

## Split and inputs

- **Training:** 2020 prediction dates whose five-session outcome also ends in
  2020.
- **Validation:** 2021 prediction dates whose outcome also ends in 2021.
- **Untouched:** 2022 and later. The loader does not retain those rows.
- **Lags:** `lag_30` is oldest and `lag_1` is the prediction date. Each example
  has a 30 × 5 sequence of open, high, low, close, and volume.
- **Target:** `target_excess_return` in decimal units. The fitted models use
  percentage points for training and convert back to decimals when saving
  predictions. `0.01` in the output means one percentage point.
- **Never features:** `entry_date`, `exit_date`, or any `target_*` field.

This first development split preserves the original proposal's 2021 validation
year and leaves its 2022–2023 test years unseen. The final multimodal split still
needs to be reconciled with the new 2020–2026 price file and news coverage.

## Methods and outputs

The run saves validation predictions for five methods: zero difference, recent
stock-price trend, XGBoost, an OHLC-only LSTM, and an OHLCV LSTM. The recent-trend
baseline is a stock-only proxy because the input CSV contains no *historical*
SPY price window. XGBoost uses 30-session relative-price and log-volume values
plus short summaries. Both LSTMs receive oldest-to-newest sequences, with
per-window price scaling followed by standardization fitted only on training
rows. The two LSTM variants show the effect of including volume while its split
adjustment is under review.

The run writes these shareable results to
`results/price_only/dev_2020_2021_seed42/`:

- `predictions_2021.csv`: one row per company, date, and model. These are
  out-of-sample predictions for the base models.
- `metrics_2021.csv` and `.json`: mean daily Spearman rank correlation, error
  in percentage points, sign accuracy, and coverage.
- `run_manifest.json`: source-file SHA-256, dates, row counts, settings, seed,
  package versions, and model selected by validation ranking score.
- `training_history.json`: XGBoost and LSTM validation curves.

Model weights and LSTM standardizers are saved locally to
`models/price_only/dev_2020_2021_seed42/`, which is gitignored under the
repository's existing rule. Run the training command to recreate them.

The zero model is constant, so its rank correlation is undefined; it is still
scored for prediction error and sign accuracy. Sign accuracy treats a zero
prediction as nonpositive. All methods are evaluated on
the same 2021 company-date examples. These 2021 predictions should not be used
both to *fit* a fusion model and to *validate* that same fusion model; a later
fusion training plan needs separate earlier-to-later predictions.

The repository currently lacks the raw Yahoo price and calendar files expected
by `processing/quantitative_prep/05_build_30day_dataset.py`, so this run uses the
committed quantitative CSV and does not claim to regenerate that source file.
