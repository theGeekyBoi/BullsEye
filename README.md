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
