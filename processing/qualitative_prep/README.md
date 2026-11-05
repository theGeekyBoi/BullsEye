# Qualitative News Preparation

This directory prepares FNSPID Nasdaq news data for the qualitative/news side of the BullsEye model.

The pipeline starts from raw FNSPID news and the quantitative ticker-date grid, then produces three model-facing tables in `data/qualitative_dataset/`.

## Inputs

Expected local files:

```text
data/raw/nasdaq_exteral_data.csv
data/raw/trading_calendar.csv
data/quantitative_dataset/quant_dataset_30day.csv
```

The raw Nasdaq file is large, so the scripts use DuckDB to scan/filter it instead of loading the full file into pandas.

Install dependencies from the repository root:

```powershell
pip install -r requirements.txt
```

## Main Notebook

Use `qual_prep.ipynb` as the walkthrough. It explains the full raw-to-processed flow and previews each generated CSV.

Set `RUN_PIPELINE = True` in the notebook if you want to regenerate all outputs.

## Pipeline Scripts

Run from the repository root in this order:

```powershell
python processing/qualitative_prep/audit_ticker_mappings.py
python processing/qualitative_prep/audit_nasdaq_coverage.py
python processing/qualitative_prep/prepare_nasdaq_article_links.py
python processing/qualitative_prep/build_company_day_news_windows.py
```

### 1. `audit_ticker_mappings.py`

Compares tickers from `quant_dataset_30day.csv` against FNSPID Nasdaq `Stock_symbol` values.

This handles known symbol differences such as:

```text
BRK-B -> BRK.B / BRK.A
GOOGL -> GOOGL / GOOG
META  -> FB
```

### 2. `audit_nasdaq_coverage.py`

Counts source-tagged Nasdaq records and full-text records for each ticker and year from 2020 through 2023.

This tells us whether a ticker-year has usable source coverage before we interpret missing articles as no news.

### 3. `prepare_nasdaq_article_links.py`

Creates the article and article-company link tables.

It:

- deduplicates unique stories,
- assigns stable `article_id` values,
- preserves original source tags,
- verifies whether the headline/body actually mentions the ticker or company,
- sends uncertain matches to a review queue,
- assigns the first eligible trading session using the trading calendar's actual close.

### 4. `build_company_day_news_windows.py`

Builds one row per ticker and prediction date.

For prediction date `t`, the news window is `t` plus the four preceding trading sessions. A story is included only when its link to the ticker is accepted and its first eligible trading session falls inside that window.

## Model-Facing Outputs

These three CSVs are intended to be committed with Git LFS:

```text
data/qualitative_dataset/nasdaq_unique_articles_2020_2023.csv
data/qualitative_dataset/nasdaq_article_company_links_2020_2023.csv
data/qualitative_dataset/company_day_news_window_5session.csv
```

Intermediate audit outputs are generated in `data/qualitative_dataset/` too, but are ignored by Git.

## Coverage Semantics

The company-day table keeps rows for all quantitative prediction dates.

For 2020-2023 rows, `source_coverage_available = True`. A row with no accepted articles in the five-session window has `news_count = 0` and `no_eligible_news = True`.

For 2024-2026 rows, `source_coverage_available = False`. These rows are kept in the join table but should be excluded from the main multimodal comparison.
