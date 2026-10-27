# BullsEye

BullsEye is an advanced machine learning course project that compares price history, financial news, and combinations of both for short-term stock prediction. At the close of a trading day, the planned system will rank 50 companies by their expected five-trading-day performance relative to an S&P 500 tracking benchmark.

## Start here

- [One-page project rule sheet](PROJECT_RULES.pdf) and [editable text version](PROJECT_RULES.md)
- [Initial FNSPID data audit](DATA_AUDIT.md)
- [Proposed 50-company study group and raw coverage counts](COMPANIES.csv)

The rules and company group were drafted before any model was trained. The data audit establishes that the required raw fields are available for this group. It also identifies news-timing and relevance checks that must be completed before training.

## Planned build order

1. Clean the selected price and news records and construct one example per company and date.
2. Build simple comparison methods, then train price-only and news-only models.
3. Train and compare three methods for combining their predictions.
4. Fix model choices using the validation period and run the untouched final test.
5. Build a Streamlit dashboard that replays historical predictions and shows the evaluation results.

## Data

The core source is [FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID), including its `Stock_price/full_history.zip`, `Stock_news/nasdaq_exteral_data.csv`, and `Stock_news/All_external.csv` files. [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank) is optional for a news-tone component. [FinBERT pretrain](https://huggingface.co/yiyanghkust/finbert-pretrain) is the starting text model.

Large datasets, downloaded news text, trained weights, and private configuration files should stay outside this public repository. The small raw-data audit sample is kept separately from the repository while redistribution rights are checked.

## Qualitative News Prep

The first qualitative-data script is `processing/qual_prep.py`. It prepares a pilot news file for `AMD`, `WFC`, and `MRK` from FNSPID's Nasdaq news data.

Place the downloaded Nasdaq file here:

```text
data/raw/nasdaq_exteral_data.csv
```

Install the Python dependency if needed:

```powershell
pip install pandas
```

Run the script from the repository root:

```powershell
python processing/qual_prep.py
```

Or use `processing/qual_prep.ipynb` to run the same workflow step by step. The notebook is useful after the first raw-data scan because it can load existing processed outputs instead of rereading the full Nasdaq file every time. If preprocessing code changes, set `REBUILD_ARTICLES = True` in the notebook setup cell once, rerun the processing cells, then set it back to `False`.

The script and notebook read the raw CSV in chunks, keep only pilot-company rows, remove exact duplicate article rows, add a stable `article_id`, add a conservative `eligible_date`, and create blank review columns for later manual annotation.

The output is written to:

```text
data/processed/qualitative/pilot_nasdaq_articles.csv
```

If `data/processed/trading_calendar.csv` exists with `date`, `open_et`, `close_et`, and `early_close` columns, the script uses `date` and `close_et` to assign article eligibility and also creates a simple company-day news table:

```text
data/processed/qualitative/pilot_company_day_news.csv
```

That file has one row per pilot company per trading day, with comma-separated `article_ids`, `article_count`, and `no_news`.

Do not commit the raw or processed article datasets. The processed file still contains article text and may be large, so it belongs in local storage. Commit the preparation code, project rules, and lightweight reports instead.

Local generated directories such as `data/raw/`, `data/processed/`, `data/sample/`, `processing/__pycache__/`, and `.ipynb_checkpoints/` are ignored by Git. Keep `data/raw/` if you do not want to redownload the Nasdaq file. It is safe to delete `processing/__pycache__/` and any `.ipynb_checkpoints/` directories. If processed outputs were created before a preprocessing fix, delete `data/processed/qualitative/` or rebuild it from the notebook.
