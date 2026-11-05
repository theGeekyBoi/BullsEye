# BullsEye shared experiment contract

**Working contract — November 5, 2026.** This connects the price, news,
evaluation, fusion, and dashboard work. Freeze the final split and eligibility
rules before opening final-test outcomes.

## Company list and one-example key

The **current 50 companies** are the tickers in
[`final_dataset/dataset_summary.csv`](final_dataset/dataset_summary.csv). The
qualitative preparation branch uses the same 50-ticker quantitative grid; the
two lists match exactly. `COMPANIES.csv` describes an earlier, different
proposal and is **not the current company list**. Only 23 tickers overlap.
Do not mix model results or news rows across these groups.

The unit is one `ticker` on one `prediction_date` (`YYYY-MM-DD`), after that
trading session closes. Keys are uppercase and preserve the project's ticker
spellings, including `BRK-B`. An article's source tag may use an alternate
spelling, but its accepted company link must use the project ticker.

The existing price file contains 83,862 company-day rows from 2020-02-13 to
2026-10-21. PLTR begins on 2020-11-10, so a complete 50-company ranking is
not available for earlier dates. The current news preparation covers
2020–2023; 2024–2026 price rows do not have equivalent audited news coverage.

| Period | Current use |
| --- | --- |
| 2020 | Initial training. PLTR joins only when its 30-day price window exists. |
| 2021 | Development validation and model comparison. |
| 2022–2023 | Reserved final test while news coverage and the final split are confirmed. Do not use outcomes for model selection. |
| 2024–2026 | Price data only; excluded from the main price/news comparison. |

At every split, remove examples whose five-session outcome would cross into
the following period. Record the number removed. Longer training or a revised
validation year requires a written split change **before** inspecting any
reserved outcomes.

## Price company-day table

**Key:** unique `ticker` + `prediction_date`. The current
[`quant_dataset_30day.csv`](final_dataset/quant_dataset_30day.csv) has these keys,
`entry_date`, `exit_date`, 30 completed sessions of company `open`, `high`,
`low`, `close`, and `volume` (150 input fields), plus
`target_company_return`, `target_spy_return`, and `target_excess_return`.
`lag_30` is oldest; `lag_1` is the prediction date. Target and entry/exit
columns are for evaluation, never model inputs. Any added SPY history or
sector field must come from information available by the prediction date and
keep the same key. The company list has no agreed sector field yet; models
must not silently reuse sectors from `COMPANIES.csv` for the 27 different
tickers.

The target is the company's adjusted return from the next session's open to
the close five sessions later, minus SPY's adjusted return over the same
times. The original price file and any later rebuild must use the same rule.

## Three news tables

These are the three tables produced by the qualitative preparation branch.
Article files currently use `2020_2023` in their names. Their exact coverage
and eligibility checks must be verified before the final combined dataset is
frozen.

| Table | One row represents | Required fields and checks |
| --- | --- | --- |
| **Unique articles** (`nasdaq_unique_articles_2020_2023.csv`) | One distinct story, even when tagged to multiple companies. | `article_id` (unique), source time (`Date`), `eligible_trading_date`, `eligibility_rule`, `Article_title`, `Article`, `Url`, and `original_source_tags`. Keep the original time and the eligibility rule. |
| **Article-company links** (`nasdaq_article_company_links_2020_2023.csv`) | One possible story-to-company connection. | `article_id`, project `ticker`, `original_source_tag`, `source_tag_match_type`, `match_status`, `match_reason`, `review_queue`, and `eligible_trading_date`. Only `accepted` links feed the model. Deduplicate accepted `article_id` + `ticker` pairs. |
| **Company-day news** (`company_day_news_window_5session.csv`) | Stories available to one company on one prediction day from that day and the four preceding trading sessions. | Unique `ticker` + `prediction_date`, `article_ids`, `news_count`, `source_coverage_available`, `no_eligible_news`, `news_window_sessions`. Join one-to-one to price company-days. |

`news_count = 0` and `source_coverage_available = true` means the source was
covered but there was no accepted eligible story. `source_coverage_available
= false` means source coverage is missing. Do not treat those states as the
same signal. A story with an unreliable midnight or date-only timestamp
becomes usable no earlier than the following trading session unless its
timing is verified.

## Prediction output

Every price-only, news-only, and combined model writes the same four columns:

```text
ticker,prediction_date,model_name,predicted_excess_return
AMD,2021-06-01,ridge,0.0032
```

**Key:** unique `ticker` + `prediction_date` + `model_name`. The predicted
number must be finite. `0.0032` means an expected difference of **0.32
percentage points** over the five-session outcome. Predictions and targets
are **decimal returns internally**; multiply by 100 only when displaying
percentage points. Prediction files must not contain actual outcomes or
future news. Daily rankings sort these numbers from highest to lowest;
buy/hold/sell thresholds are separate fixed rules.

## Checks before joining or comparing models

1. Confirm price and news use exactly the 50 tickers in
   `final_dataset/dataset_summary.csv`, with no duplicate company-date keys.
2. Check eligible dates and count missing news rows separately from covered
   days with zero eligible articles.
3. Check that each accepted `article_id` exists in the article table and was
   eligible by the prediction date and within the five-session news window.
4. Join price and company-day news one-to-one on `ticker` and
   `prediction_date`. Compare models on the same eligible keys.
5. Join predictions to outcomes only in the evaluator. Check that each model
   has one finite prediction per eligible key.
6. Keep source hashes, company list, coverage counts, training dates, model
   settings, and random seeds with each saved run.

The `price_only_v2/` rebuild and `results/price_only_v2/` are an **archive for
the earlier 50-company list**. Their 2021 scores and saved models cannot be
used as current-universe price baselines. Rerun price development for the
current 50 before comparing it with the news or combined models.
