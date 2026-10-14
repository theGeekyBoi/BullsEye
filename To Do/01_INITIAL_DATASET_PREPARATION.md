# BullsEye: Initial Dataset Preparation Handoff

## What we are making

We need a dataset with **one example per company per trading day**. For each example, we will save what BullsEye could have known at the close of that day: the company, its recent prices and volume, and relevant news available by then. We will also calculate the result the model will later learn to predict.

Start with **AMD, WFC, and MRK**, plus **SPY** for comparison. Check that small pilot carefully. Then run the same process for all 50 companies.

## Step 1 — Read the project files

Start at the [BullsEye GitHub repository](https://github.com/theGeekyBoi/BullsEye). These are the files we will use:

- [README](https://github.com/theGeekyBoi/BullsEye/blob/main/README.md): project overview and build order.
- [One-page rule sheet (PDF)](https://github.com/theGeekyBoi/BullsEye/blob/main/PROJECT_RULES.pdf) or [editable rule sheet](https://github.com/theGeekyBoi/BullsEye/blob/main/PROJECT_RULES.md): what a prediction means, what information is allowed, and the planned dates.
- [COMPANIES.csv](https://github.com/theGeekyBoi/BullsEye/blob/main/COMPANIES.csv): the selected 50 companies and their business areas. Use this as the company list when expanding the pilot.
- [DATA_AUDIT.md](https://github.com/theGeekyBoi/BullsEye/blob/main/DATA_AUDIT.md): the coverage we checked and the data problems we already found.
- [.gitignore](https://github.com/theGeekyBoi/BullsEye/blob/main/.gitignore): shows which downloaded and generated files should stay out of the public repository.

The small inspection ZIP is **not on GitHub**. It is useful for seeing example columns, but it contains too few dates to make the pilot dataset. The source downloads below are what the team needs.

## Step 2 — Download the price data

Get FNSPID's [full_history.zip price archive](https://huggingface.co/datasets/Zihan1004/FNSPID/blob/main/Stock_price/full_history.zip). The page has a download button; this is the [direct download URL](https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_price/full_history.zip).

For the pilot, extract the files for **AMD, WFC, MRK, and SPY**. Each file has daily prices and trading volume. Keep the original ZIP unchanged and write cleaned results to a separate location.

Read each CSV using its **column names**, because the order of columns varies between files.

## Step 3 — Download the news data

Get FNSPID's [Nasdaq news file](https://huggingface.co/datasets/Zihan1004/FNSPID/blob/main/Stock_news/nasdaq_exteral_data.csv), which has a [direct download URL](https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv). It contains headlines, article text, dates, URLs, and company symbols. The file is about **23.2 GB**, so process it in chunks or as a stream rather than loading all of it into memory.

The [additional FNSPID news file](https://huggingface.co/datasets/Zihan1004/FNSPID/blob/main/Stock_news/All_external.csv) is available through this [direct download URL](https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/All_external.csv). Leave it for a second pass after the Nasdaq pipeline works. In the portion we inspected, many company-linked records had headlines without article bodies.

The [FNSPID project page](https://github.com/Zdong104/FNSPID_Financial_News_Dataset) explains the source dataset. Keep downloaded news and article text outside the public BullsEye repository.

## Step 4 — Create one shared trading-day calendar

Use the [NYSE trading hours and holiday information](https://www.nyse.com/trade/hours-calendars) to determine which dates were trading sessions and when each session closed. The [exchange_calendars Python library](https://github.com/gerrymanoim/exchange_calendars) can help implement this.

Use this calendar for **all companies, SPY, and news**. It will tell us what "the next trading day" means, including weekends, holidays, and shortened sessions.

## Step 5 — Clean the pilot prices

For AMD, WFC, MRK, and SPY:

1. Parse and sort the dates.
2. Keep one valid price row per company and trading session.
3. Flag missing, duplicate, zero, negative, and market-closed-day rows. Do not invent prices for missing days.
4. Check how opening and closing prices should be adjusted for stock splits and dividends. FNSPID has `adj close` but no `adj open`. One candidate is `adjusted open = open × (adj close / close)` for that date. Check this around known price-adjustment events before using it for the prediction result.
5. Save a cleaned price table. Build each model input from the **30 completed trading sessions ending on its prediction day**.

The price-adjustment decision must be written down before the team produces final results. Our [data audit](https://github.com/theGeekyBoi/BullsEye/blob/main/DATA_AUDIT.md) identifies two blank holiday rows in the wider 50-company set that the calendar should remove.

## Step 6 — Clean and check the pilot news

Keep Nasdaq stories tagged with AMD, WFC, or MRK. For each story, save its source date and time, headline, text, URL, source company tag, and a stable article ID.

1. Remove exact duplicates. Check for the same story repeated under different company symbols.
2. Check that a story really concerns its tagged company. Use clear company-name, short-name, or ticker mentions in the headline or text. Flag broad industry stories and unclear matches for manual review. Keep a reason for each accepted or rejected link.
3. Decide when the story could first have been used. A reliable publication time must be no later than that trading session's close. If a story has only a date or an unreliable `00:00:00 UTC` time, make it eligible **no earlier than the next trading session after its stated date**.
4. Record days with no eligible story as **no news**.
5. Set and document how far back each day should look for eligible stories. Use **five trading sessions** for the pilot unless the team records a different choice before model training.

Manually review a random set of accepted and rejected stories across the three pilot companies. Include duplicates, broad headlines, uncertain matches, and midnight timestamps. The [initial audit](https://github.com/theGeekyBoi/BullsEye/blob/main/DATA_AUDIT.md) explains why both timing and company tags need checking.

## Step 7 — Build a continuous pilot dataset

Choose a continuous month in **2019**, with at least 30 earlier trading sessions available for inputs and five later sessions available for results. The small inspection ZIP is not sufficient for this step; use the full source files.

For each of AMD, WFC, and MRK on each eligible prediction day **t**, create one example containing:

- Company symbol and business area.
- Its 30-session price and volume history, ending on **t**.
- IDs of relevant stories eligible by the close of **t**.
- A no-news indicator.
- The company's percentage change from the **opening of t+1** to the **close of t+5**.
- SPY's percentage change over those **same entry and exit times**.
- The target: **company percentage change minus SPY percentage change**, in percentage points.

Here, **t+1** and **t+5** mean the first and fifth *trading sessions* after t. Keep news in a separate cleaned table and refer to it by article ID from the example table. Never put the later result or a later news story into the model input.

## Step 8 — Check examples by hand

Before processing all 50 companies, trace several pilot examples back to the source files. For each one, check:

1. The 30 price rows end on the correct day.
2. Every included story was relevant and eligible by that day's close.
3. Uncertain-time and weekend stories were held until the correct session.
4. No-news days are marked correctly.
5. The company and SPY calculations use the correct next opening and fifth closing prices.
6. The target equals the company result minus the SPY result.

Include an example near a holiday. Fix any errors in the preparation scripts, regenerate the pilot, and repeat the checks.

## Step 9 — Expand to all 50 companies

Run the checked process for the companies in [COMPANIES.csv](https://github.com/theGeekyBoi/BullsEye/blob/main/COMPANIES.csv). Produce a coverage report for each company and year: valid price days, missing data, raw news counts, duplicates, rejected and accepted stories, no-news days, and usable examples.

Use the planned periods from the [rule sheet](https://github.com/theGeekyBoi/BullsEye/blob/main/PROJECT_RULES.md): **2017–2020 training, 2021 validation, and 2022–2023 final test**. Leave at least five trading sessions at each boundary so a result cannot extend into the next period. Drop dates near the end of 2023 that lack five later price sessions.

## What to hand over before training

- Repeatable scripts for cleaning prices, cleaning and linking news, and building the daily examples.
- The checked three-company pilot and the record of its manual checks.
- Written decisions for price adjustments, news timing, article relevance, and the news lookback window.
- The 50-company coverage report and any unresolved issues.

Commit the **scripts, rules, and coverage report** to the [BullsEye repository](https://github.com/theGeekyBoi/BullsEye). Keep the large source downloads, raw article text, and generated datasets outside the public repository. Once the checks pass and the rules are fixed, the modeling team can train every method on the same examples.
