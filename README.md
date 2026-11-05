# BullsEye

BullsEye is an advanced machine learning course project that compares price history, financial news, and combinations of both for short-term stock prediction. At the close of a trading day, the planned system will rank 50 companies by their expected five-trading-day performance relative to an S&P 500 tracking benchmark.

## Start here

- [Original one-page rule sheet (archived)](PROJECT_RULES.pdf) and [current text rules](PROJECT_RULES.md)
- [Initial FNSPID data audit](DATA_AUDIT.md)
- [Current 50-company list and date counts](final_dataset/dataset_summary.csv)
- [Shared price, news, and prediction contract](EXPERIMENT_CONTRACT.md)
- [Current news-coverage follow-up for the team](To%20Do/02_CURRENT_NEWS_COVERAGE.md)
- [Earlier proposed 50-company list (archived)](COMPANIES.csv)

The current company group is the one used in the qualitative preparation branch. The initial FNSPID audit examined a different group and remains background material. News timing and relevance checks must be completed before combined-model training.

## Current development work

- [Quantitative data preparation scripts](Quantitative%20Data%20Prep/readme.md) and [30-session dataset](final_dataset/dataset_summary.csv)
- [Price-only training instructions](price_only/README.md)
- [First 2021 validation results and predictions](results/price_only/dev_2020_2021_seed42/README.md)
- [Current 50-company price rebuild and 2021 results](results/price_only_current/README.md)
- [Earlier 50-company price rebuild (archived)](price_only_v2/README.md)
- [Earlier rebuild audit and 2021 validation report (archived)](results/price_only_v2/README.md)

The current quantitative prototype was built from Yahoo Finance prices. Its
50 tickers match the qualitative branch's quantitative grid. News coverage is
currently prepared for 2020–2023, so 2024–2026 price dates are outside the
main combined-model comparison. The first price-only run uses 2020 for
training and 2021 for validation; it makes no final-test performance claim.

The `price_only_v2/` code has now also rebuilt and trained the current 50,
adding 2017–2019 history for companies with price data and SPY inputs. Its
current-universe 2021 predictions and audit are under `results/price_only_current/`.
The separate `results/price_only_v2/` predictions remain an archived
experiment for the earlier 50 and must not be mixed with current news results.

## Planned build order

1. Clean the selected price and news records and construct one example per company and date.
2. Build simple comparison methods, then train price-only and news-only models.
3. Train and compare three methods for combining their predictions.
4. Fix model choices using the validation period and run the untouched final test.
5. Build a Streamlit dashboard that replays historical predictions and shows the evaluation results.

## Data

The original proposal centered on [FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID), including its `Stock_price/full_history.zip`, `Stock_news/nasdaq_exteral_data.csv`, and `Stock_news/All_external.csv` files. The current quantitative prototype instead uses Yahoo Finance prices, as documented in the preparation scripts. [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank) is optional for a news-tone component. [FinBERT pretrain](https://huggingface.co/yiyanghkust/finbert-pretrain) is the starting text model.

Large datasets, downloaded news text, trained weights, and private configuration files should stay outside this public repository. The small raw-data audit sample is kept separately from the repository while redistribution rights are checked.
