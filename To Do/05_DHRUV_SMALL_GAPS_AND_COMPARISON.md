# Dhruv’s task: fill the small news gaps and compare the models

Start from the `aaditya/fusion-evaluation` branch in [draft PR #3](https://github.com/theGeekyBoi/BullsEye/pull/3). Aaditya builds and trains the headline model. Dhruv finds the missing headlines and runs the comparison. Bryce can work on the price-only simulator in parallel.

## 1. Search the 20 listed company-months

Open [the gap list](dhruv_news_comparison/priority_company_months.csv). Search for the company name and exact month in a publisher archive or through the university library. Search for Facebook when looking for META in 2020. Check the original article date and confirm the story is about the company.

Copy [the blank template](dhruv_news_comparison/verified_headlines_template.csv) to `data_local/verified_headlines.csv`. Record the ticker, original publication date (YYYY-MM-DD), headline, and publisher URL. Leave a gap open if you cannot verify a story. A headline can add a few prediction days within a month.

From the repo root, set up the dependencies and download the tracked data.

```bash
python -m pip install -r requirements-price-only-v2.txt -r requirements.txt
git lfs pull --include="final_dataset/quant_dataset_30day.csv,data/qualitative_dataset/nasdaq_article_company_links_2020_2023.csv"
```

Prepare the new headlines. This checks dates and requested months, removes known duplicates, and makes each new story available starting on the next trading session for up to five sessions.

```bash
python processing/qualitative_prep/append_verified_headlines.py --verified data_local/verified_headlines.csv --output data_local/verified_news
```

Give Aaditya both outputs in `data_local/verified_news`: `verified_headlines_prepared.csv` and `headline_eligibility_2020_2021.csv`. Aaditya can start training immediately with the [existing eligibility file](dhruv_news_comparison/headline_eligibility_2020_2021.csv) and [backfill sources](../processing/qualitative_prep/NEWS_BACKFILL.md).

## 2. Confirm dates and try the comparison

Use the [current 50 companies](../final_dataset/dataset_summary.csv). Aaditya trains on eligible 2020 rows. Agree on the proposed January 11–December 23, 2021 validation window. It covers 242 sessions and 12,100 price predictions before the news filter. Keep 2022–2023 outcomes reserved.

Run this demo with made-up news to check the files and commands.

```bash
python -m fusion.run --demo --validation-only --validation-start 2021-01-11 --targets final_dataset/quant_dataset_30day.csv --coverage "To Do/dhruv_news_comparison/headline_eligibility_2020_2021.csv" --coverage-flag include_in_headline_comparison --output data_local/dhruv_demo
```

## 3. Run the comparison with Aaditya’s forecasts

Ask Aaditya for the news predictions, the exact model name, and the eligibility CSV used for that run. Save them as `data_local/news_predictions.csv` and `data_local/news_eligibility.csv`. Replace `headline_model` below with the actual model name.

```bash
python -m fusion.run --news data_local/news_predictions.csv --news-model headline_model --validation-only --validation-start 2021-01-11 --targets final_dataset/quant_dataset_30day.csv --coverage data_local/news_eligibility.csv --coverage-flag include_in_headline_comparison --output data_local/dhruv_comparison
```

- In `comparison_metrics.csv`, look at the `matched_news` rows. Compare the daily ranking score for price, news, and combined. All three use the same company-days.
- Read `daily_comparison_coverage.csv` to see how many companies were compared each day. The command prints the median. Check `coverage.csv` for how often the combined model used price alone.
- Read the `full_price_grid` rows to see how price and the blend compare across all 50 companies.

## 4. Share the results

Give Bryce `combined_predictions.csv` from `data_local/dhruv_comparison` for the 2021 replay. The model name is `fusion_equal_weight`. See the [simulator instructions](../fusion/README.md#simulator-handoff-for-bryce).

Share the comparison and coverage CSVs with the team. Report which method ranked best, the median number of companies compared per day, and the percentage of company-days that used the price fallback.
