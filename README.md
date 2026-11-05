# BullsEye

BullsEye is an advanced machine learning course project that compares price history, financial news, and combinations of both for short-term stock prediction. At the close of a trading day, the planned system will rank 50 companies by their expected five-trading-day performance relative to an S&P 500 tracking benchmark.

## Start here

- [Original one-page rule sheet](PROJECT_RULES.pdf) and [current text version with implementation note](PROJECT_RULES.md)
- [Initial FNSPID data audit](DATA_AUDIT.md)
- [Proposed 50-company study group and raw coverage counts](COMPANIES.csv)

The rules and company group were drafted before any model was trained. The data audit establishes that the required raw fields are available for this group. It also identifies news-timing and relevance checks that must be completed before training.

## Current development work

- [Quantitative data preparation scripts](Quantitative%20Data%20Prep/readme.md) and [30-session dataset](final_dataset/dataset_summary.csv)
- [Price-only training instructions](price_only/README.md)
- [First 2021 validation results and predictions](results/price_only/dev_2020_2021_seed42/README.md)
- [Rebuilt price-only pipeline for the agreed 50 companies](price_only_v2/README.md)
- [Rebuilt data audit and 2021 validation report](results/price_only_v2/README.md)

The current quantitative prototype was built from Yahoo Finance price data and
covers a different 50-company group from the original FNSPID audit. Its
2020–2026 period must be reconciled with available news and the project rules
before the final multimodal comparison. The first price-only run uses 2020 for
training and 2021 for validation; it makes no final-test performance claim.

The rebuild uses the original 50-company list, adds earlier 2017–2019 history
and a matching SPY window, and checks 2021 only after rolling earlier-year
model selection. Its raw and derived price snapshots are local files. **Use
the rebuilt company/date keys for the future news join.** The newer 50-company
prototype remains for historical comparison and is not the canonical group.

## Planned build order

1. Clean the selected price and news records and construct one example per company and date.
2. Build simple comparison methods, then train price-only and news-only models.
3. Train and compare three methods for combining their predictions.
4. Fix model choices using the validation period and run the untouched final test.
5. Build a Streamlit dashboard that replays historical predictions and shows the evaluation results.

## Data

The original proposal centered on [FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID), including its `Stock_price/full_history.zip`, `Stock_news/nasdaq_exteral_data.csv`, and `Stock_news/All_external.csv` files. The current quantitative prototype instead uses Yahoo Finance prices, as documented in the preparation scripts. [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank) is optional for a news-tone component. [FinBERT pretrain](https://huggingface.co/yiyanghkust/finbert-pretrain) is the starting text model.

Large datasets, downloaded news text, trained weights, and private configuration files should stay outside this public repository. The small raw-data audit sample is kept separately from the repository while redistribution rights are checked.
