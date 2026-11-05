# BullsEye Project Rules

**Working version — November 5, 2026.** The company list has been changed to match the qualitative data. The team will freeze these rules before the final test and record any later change.

## What the system predicts

At the close of trading session **t**, BullsEye makes one prediction for each company in the study. The prediction is the company's percentage return from the **opening of t+1** through the **close of t+5**, minus the return of **SPY** over those same times. SPY is our S&P 500 tracking benchmark. We express the difference in **percentage points** and rank companies from highest to lowest predicted difference. We do not train a separate buy/sell/hold target.

## Information allowed at prediction time

Each example uses the company's identity and business area, approximately **30 completed trading sessions** of prices and trading volume, and relevant news available by the close of t. News is linked to companies or business areas using source symbols and checked name-matching rules. A missing story is marked **no news**, not neutral news. We use the official market calendar and session closing time. News with only a date or an unreliable time becomes eligible **no earlier than the next trading session**. No input may contain information first available after t's close.

## Companies, data, and dates

The study will use the **50 companies in `final_dataset/dataset_summary.csv`**, which match the tickers used by the qualitative preparation branch. `COMPANIES.csv` is the earlier proposed list and is no longer authoritative. FNSPID supplies historical news; the quantitative prototype uses Yahoo Finance adjusted prices. Financial PhraseBank may be used for a separate news-tone component. We will check missing dates, symbol changes, duplicate stories, and corporate actions. Stock and SPY opening and closing prices must receive consistent split and dividend treatment before returns are calculated. If the required adjustment cannot be verified, the team will revise the target rule before final testing.

The current development split is **2020 for training** and **2021 for validation and model choices**. Keep **2022–2023 reserved for the final test** while the news coverage audit and final split are confirmed. The current news preparation covers 2020–2023; 2024–2026 price rows are outside the main multimodal comparison. PLTR has no usable 30-session price example before November 10, 2020. Exclude examples whose five-session outcome crosses a split boundary. Fix the company list and date boundaries before final testing.

**Implementation note:** The `price_only_v2/` rebuild and its saved 2021 results use the earlier `COMPANIES.csv` list. They are archived experiments, **not** comparable to the current news data. The current price and news branches must be checked for matching ticker-date keys and source coverage before training the combined models. The handling of volume around unusual corporate actions still requires review.

## Models and comparisons

We will compare simple zero-difference and recent-trend rules; price-only XGBoost and a small LSTM; a news-tone baseline and a FinBERT-based news model; and three ways to combine price and news predictions: a fixed average, a small neural network, and a changing-weight gate. Every method predicts the same target on the same eligible company-date examples. The combining models will learn from earlier-to-later predictions made on dates their base models did not train on.

## How we will judge the result

The main measure is **average daily Spearman rank correlation** between predicted and actual company rankings. We will also report prediction error, how often the predicted difference has the right sign, sample coverage, and uncertainty calculated from blocks of nearby dates. A separate weekly top-five historical buying test will compare rankings with SPY and an equal-weight study group, including estimated trading costs. Final test results will be calculated once after choices are frozen; successes, failures, and limits will all be reported.

## Dashboard

The Streamlit dashboard will let a user select a historical date and company, then inspect the allowed inputs, the price-only and news-only predictions, the combined prediction, and the daily ranking. It will also show the training, validation, and test periods and the saved evaluation results. The dashboard will replay historical predictions; it will not imply that past results guarantee future performance.
