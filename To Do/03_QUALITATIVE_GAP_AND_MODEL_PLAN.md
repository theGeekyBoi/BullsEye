# BullsEye: next news-data and qualitative-model tasks

Use the [integration guide](../INTEGRATION_GUIDE.md) to set up the repository
and the [shared contract](../EXPERIMENT_CONTRACT.md) for company keys, dates,
and prediction format. The company list remains the 50 tickers in
[`final_dataset/dataset_summary.csv`](../final_dataset/dataset_summary.csv).

**Starting point:** Price and news company-day grids match exactly. The
[coverage audit](../results/price_only_current/news_coverage_audit.json)
finds 29 tickers with any flagged full-text coverage in 2021 and 38 across
2020–2023. Accepted article-company links exist for 48 tickers across the
four years, but none for PLTR or RTX. The 2020–2023 news table also has 1,256
company-days with article IDs and a false full-text coverage flag. These
figures describe the current FNSPID extract; they do not prove that the other
companies had no news anywhere.

## Task 1 — Find the source of every coverage gap

**Goal:** Separate missing FNSPID records from errors in symbol mapping,
article matching, or the coverage calculation.

1. Use `news_coverage_by_ticker_year.csv` and
   `accepted_news_links_by_ticker_year.csv` in `results/price_only_current/`
   to make a 50-company × month worklist. Check 2020 training and 2021
   validation months first.
2. For a sample of each problem ticker, query the raw FNSPID Nasdaq CSV for
   its project ticker and known source aliases. Check PLTR and RTX first;
   then inspect tickers with links but no flagged full-text coverage.
3. For the 1,256 inconsistent company-days, trace sample `article_ids` to
   the link and article tables. Check whether their records have a headline,
   a body, both, or neither, and whether the story was carried from a
   neighboring month by the five-session window.
4. Record a reason for each ticker-month: no source records, alias missed,
   uncertain company link, headline only, usable full text, or processing
   error. Do not fill a gap with a made-up article.

**Output:** A gap-reason CSV with ticker, month, counts, and action needed.

## Task 2 — Use all suitable FNSPID text before adding another source

**Goal:** Recover records already available in the team's chosen dataset.

1. Check FNSPID's
   [`All_external.csv`](https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/All_external.csv)
   for the missing ticker-months. Earlier inspection found many headline-only
   records, so count headlines and full bodies separately.
2. Normalize source symbols to the project's tickers. Preserve the original
   symbol, source name, URL, and publication time in each record.
3. Deduplicate stories already present in the Nasdaq file by normalized URL
   and title. Keep one story linked to multiple companies when appropriate.
4. Check how many missing 2020 and 2021 company-days gain at least one
   eligible headline or full-text story. Report the gain by ticker and month.

**Output:** A measured FNSPID-only coverage improvement. Headline-only
records remain labeled as headline-only; they are useful for a headline
model but do not become full-text coverage.

## Task 3 — Pilot an additional historical headline source if needed

**Goal:** See whether an independent source can fill the remaining gaps
without changing the project's dates or claiming unavailable article bodies.

1. Try a small pilot for PLTR, RTX, and several other missing tickers in
   2020–2021 using the [GDELT DOC 2.0 API](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/)
   or its [historical coverage guidance](https://blog.gdeltproject.org/doc-2-0-updates-1-5-year-searching-and-updated-mobile-interface/).
   Query short date windows, use company names with context, and save title,
   URL, source, and date/time. The API returns article listings, not a
   guaranteed reusable full article body. Respect rate limits and retry
   gently; a test request returned HTTP 429 during this integration work.
2. Manually review a sample for false company matches, duplicate stories,
   bad dates, and articles first seen after the prediction close. Count how
   many *new eligible* company-days the pilot actually adds.
3. If the pilot fails, record that result. A commercial news provider is an
   option only if the team has access to a verified 2020–2021 archive and
   suitable research/reuse terms. Do not assume a provider's current feed
   includes the needed history.
4. If the course permits a broader text source, consider [SEC EDGAR company
   filings](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
   as a **separate filing type**. They are primary company disclosures,
   not replacement news articles; retain a distinct `source_type`.

**Output:** A source-pilot table showing added company-days, false-match
rate from the reviewed sample, timestamp quality, and access/reuse terms.
Adopt a source only after that check.

## Task 4 — Rebuild and manually review the clean news tables

**Goal:** Give the text model only stories that really pertain to the company
and were available at prediction time.

1. In the unique-articles table, keep stable `article_id`, original source,
   URL, title, body when available, publication time, and eligibility rule.
   Add explicit `headline_available` and `body_available` flags.
2. In the article-company links table, keep the source ticker and project
   ticker, match method, accepted/rejected/uncertain status, and review reason.
   Start with source tags and transparent name rules. Send ambiguous links to
   human review; do not train an article-to-company classifier without
   labeled examples.
3. Convert trustworthy timestamps to New York market time. For missing or
   suspicious midnight times, make the article eligible no earlier than the
   next session unless independently verified. Use the shared trading-day
   calendar, including early closes.
4. Rebuild the five-session company-day table. Store article IDs, counts,
   source coverage, no eligible news, and missing coverage as distinct
   fields. Check that every ID points to an accepted link and falls within
   the allowed five-session window.
5. Review together: an ordinary day, holiday boundary, after-close story,
   midnight-timestamp story, duplicate, no-news day, and price-split day.

**Output:** Three checked news tables and a manual-review sheet. Re-run
`python -m price_only_v2.audit_news_coverage --news data/qualitative_dataset/company_day_news_window_5session.csv`
and save the updated coverage report.

## Task 5 — Freeze the usable company-day keys

**Goal:** Make price-only, news-only, and combined results comparable.

1. Decide the minimum requirement for a usable news company-day: for
   example, a known covered source with an eligible headline, or known
   coverage with zero eligible stories. Keep missing source coverage separate.
2. Count usable 2020 training and 2021 validation keys per ticker and date.
   Require the same keys when comparing methods on the news-covered subset.
3. Keep a full-50 price ranking. For company-days with missing news source
   coverage, define a documented price-only fallback for the combined system.
4. Record the company list, source versions, split dates, masks, and fallback
   before any final-test outcomes are opened. 2022–2023 remains reserved.

**Output:** A frozen company-day eligibility file and coverage report.

## Task 6 — Train the qualitative branch in small stages

**Goal:** Turn eligible text into one predicted number per company-day.

1. Start with a simple baseline using only article count and a basic
   headline-tone score. Include explicit no-news and missing-coverage flags.
2. For an existing sentiment classifier, use a task-ready model such as
   [ProsusAI FinBERT](https://huggingface.co/ProsusAI/finbert), which outputs
   positive, negative, and neutral scores. The previously proposed
   [`yiyanghkust/finbert-pretrain`](https://huggingface.co/yiyanghkust/finbert-pretrain)
   is a language-model checkpoint and does **not** output sentiment labels
   without further training. [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank)
   can support an optional separate sentiment fine-tuning experiment; it
   does not supply dated news for the 50 companies.
3. Score each eligible headline or body once, cache the scores, then
   aggregate the past five sessions into company-day features: count, mean
   positive/negative score, strongest score, and recency. Keep headline-only
   and full-body results identifiable.
4. Train a small news-only prediction model on the 2020 company-days and
   choose settings using 2021 only. Save one decimal
   `predicted_excess_return` per eligible company-day with the shared
   `ticker,prediction_date,model_name` keys.
5. Check daily ranking, prediction error, direction accuracy, coverage,
   company/month stability, and the no-news cases. Save settings, seeds,
   predictions, and the exact article-source version.

**Output:** A news baseline, a FinBERT-based news model, 2021 prediction
files, and validation results. No claim of improvement is made until the
scores are compared with price-only on the same keys.

## Task 7 — Prepare fusion and the final test

**Goal:** Make the next stage possible without leaking future outcomes.

1. Create earlier-to-later 2020 predictions for both price and news models
   so a fusion model is trained on predictions for dates the base models did
   not train on.
2. Compare the fixed average, small neural combiner, and changing-weight
   gate using the 2021 matched validation keys. Also evaluate the full-50
   price-fallback rule separately.
3. Freeze model and threshold choices, then and only then fit using allowed
   pre-test data and open 2022–2023 outcomes for the final evaluation.

**Output:** Comparable price, news, and fusion forecasts and a frozen
evaluation plan for the reserved final test.
