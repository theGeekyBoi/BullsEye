# Initial FNSPID Data Audit

**Date:** October 13, 2026  
**Status:** Raw input coverage confirmed for the proposed study group; cleaning and detailed relevance checks remain.

## What we checked

We inspected FNSPID's stock price archive and sampled both news files. For 82 possible companies with complete price history, we counted price rows and streamed the matching Nasdaq news records by company. We also counted company-linked records in the older `All_external.csv` portion. The [50-company list](COMPANIES.csv) was chosen for coverage across eight business areas, using data availability rather than prediction outcomes. Counts below are **raw records before duplicate removal and relevance filtering**.

| Check | Finding |
| --- | --- |
| Stock prices | Every selected company has 1,759 usable trading-day rows from January 2017 through December 2023. The SPY benchmark has the same span. The files contain date, volume, open, high, low, close, and adjusted close. |
| Price gaps | Two selected files contain one extra blank row each on a market holiday (AMAT on 2017-02-20 and HAL on 2018-12-05). These rows must be removed with the shared market calendar. |
| News during training, 2017–2020 | 128,434 tagged records across the 50 companies; 103,666 of these are Nasdaq records with nonempty article text. The older external source supplies headlines but no article body in the linked segment inspected. |
| News during validation, 2021 | 29,023 tagged records across the 50 companies. |
| News during final test, 2022–2023 | 88,264 tagged records across the 50 companies. |
| Minimum company coverage | Each selected company has at least 785 raw training news records, 142 validation records, and 516 test records. Each has at least 535 full-text Nasdaq records in the training period. |
| Small downloaded sample | Eight companies from different business areas, plus SPY prices; 135 price rows and 48 full-text Nasdaq records from 2019, 2021, and 2023. All 48 sample news records have text. |

## Data issues to handle before training

1. **Publication times need a conservative rule.** In the 48-record sample, 39 timestamps are exactly `00:00:00 UTC`. We cannot assume those stories were public before that day's market close. Date-only or suspect midnight records will first become eligible on the next trading session unless their timing can be independently verified.
2. **The source's company tag is not always enough.** In a simple spot check, 4 of 48 sampled article bodies did not even mention the tagged company by its common name. We will check company-name and ticker matches, allow genuine industry-wide stories, and manually review ambiguous cases before using them as model inputs.
3. **Price adjustment needs an implementation check.** The archive has adjusted close but no separate adjusted open field. We will derive comparable opening prices using the daily adjustment factor and inspect known split/dividend examples before calculating the target. If this cannot be verified, the prediction rule must be revised before the final test.
4. **The study group is coverage-based.** Some familiar companies were not selected because their FNSPID price files end early, are absent, or their news is sparse in the planned training or validation years. The final findings will apply to this fixed, data-rich group, not automatically to all stocks.
5. **The archive ends in December 2023.** Dates without five later trading sessions must be excluded from labeled examples.

## Immediate next action

Write the cleaning pipeline, inspect a larger random set of company-news links, and create the company-by-date examples. Freeze the company list, date split, news timing rule, and price adjustment rule before any final test results are viewed.

**Sources:** [FNSPID dataset](https://huggingface.co/datasets/Zihan1004/FNSPID) and [FNSPID project repository](https://github.com/Zdong104/FNSPID_Financial_News_Dataset).
