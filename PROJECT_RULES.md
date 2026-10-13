# BullsEye Project Rules

**Working version — October 13, 2026.** The initial data coverage check is complete. The team will freeze these rules before training and record any necessary change before the final test.

## What the system predicts

At the close of trading session **t**, BullsEye makes one prediction for each company in the study. The prediction is the company's percentage return from the **opening of t+1** through the **close of t+5**, minus the return of **SPY** over those same times. SPY is our S&P 500 tracking benchmark. We express the difference in **percentage points** and rank companies from highest to lowest predicted difference. We do not train a separate buy/sell/hold target.

## Information allowed at prediction time

Each example uses the company's identity and business area, approximately **30 completed trading sessions** of prices and trading volume, and relevant news available by the close of t. News is linked to companies or business areas using source symbols and checked name-matching rules. A missing story is marked **no news**, not neutral news. We use the official market calendar and session closing time. News with only a date or an unreliable time becomes eligible **no earlier than the next trading session**. No input may contain information first available after t's close.

## Companies, data, and dates

The study will use the **50 companies in COMPANIES.csv**, chosen after the initial FNSPID coverage audit. FNSPID supplies historical prices and news. Financial PhraseBank may be used for a separate news-tone component. We will check missing dates, symbol changes, duplicate stories, and corporate actions. Stock and SPY opening and closing prices will receive consistent split and dividend treatment before returns are calculated. If the required adjustment cannot be verified, the team will revise the target rule before testing.

The planned chronological split is **2017–2020 for training**, **2021 for validation and model choices**, and **2022–2023 for the untouched final test**, subject to the initial coverage check. A gap of at least five trading sessions at each boundary prevents an outcome from crossing into the next period. The company list and date boundaries will be fixed before final testing.

## Models and comparisons

We will compare simple zero-difference and recent-trend rules; price-only XGBoost and a small LSTM; a news-tone baseline and a FinBERT-based news model; and three ways to combine price and news predictions: a fixed average, a small neural network, and a changing-weight gate. Every method predicts the same target on the same eligible company-date examples. The combining models will learn from earlier-to-later predictions made on dates their base models did not train on.

## How we will judge the result

The main measure is **average daily Spearman rank correlation** between predicted and actual company rankings. We will also report prediction error, how often the predicted difference has the right sign, sample coverage, and uncertainty calculated from blocks of nearby dates. A separate weekly top-five historical buying test will compare rankings with SPY and an equal-weight study group, including estimated trading costs. Final test results will be calculated once after choices are frozen; successes, failures, and limits will all be reported.

## Dashboard

The Streamlit dashboard will let a user select a historical date and company, then inspect the allowed inputs, the price-only and news-only predictions, the combined prediction, and the daily ranking. It will also show the training, validation, and test periods and the saved evaluation results. The dashboard will replay historical predictions; it will not imply that past results guarantee future performance.
