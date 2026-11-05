# News coverage work for the current 50 companies

**Use the 50 tickers in [`final_dataset/dataset_summary.csv`](../final_dataset/dataset_summary.csv).**
The qualitative branch's `company_day_news_window_5session.csv` has an exact
company-date match to the current quantitative price grid. That means the
tables can join by `ticker` and `prediction_date`; it does **not** mean every
ticker has usable news.

The [coverage audit](../results/price_only_current/news_coverage_audit.json)
found that the current full-text coverage flag is true for **29 tickers at
some point in 2021** and **38 across 2020–2023**. These 12 tickers are never
flagged as covered in 2020–2023: BAC, BRK-B, CSCO, HD, JNJ, JPM, MA, MCD,
META, NFLX, PLTR, and RTX. Another 1,256 company-days have accepted article
IDs while `source_coverage_available` is false, mostly in 2020. The detailed
counts are in
[`news_coverage_by_ticker_year.csv`](../results/price_only_current/news_coverage_by_ticker_year.csv).
The integrated article-link check found accepted story-to-company links for
48 of the 50 tickers across 2020–2023. **PLTR and RTX have no accepted links**
in this FNSPID extract. Counts by ticker and year are in
[`accepted_news_links_by_ticker_year.csv`](../results/price_only_current/accepted_news_links_by_ticker_year.csv).
All 685,468 company-day article references point to accepted links.

## Tasks for the qualitative team

1. **Check the raw source for the uncovered tickers.** Inspect the monthly
   coverage counts and source ticker aliases, including `BRK-B`/`BRK.B`,
   `META`/`FB`, and `RTX`/older symbols. Start with PLTR and RTX, which have
   no accepted links, then the 10 other tickers with links but no flagged
   full-text coverage. For each missing ticker-year, record
   whether source articles are genuinely absent or the mapping/filtering
   missed them. Do not infer coverage from a row in the company-day grid.
2. **Explain the 1,256 inconsistent rows.** Sample article IDs on days with
   `news_count > 0` and `source_coverage_available = false`. Check whether
   these are headline-only records, text from a neighboring month in the
   five-session window, or an error in the monthly coverage join. Record the
   article's actual available fields and eligibility date.
3. **Keep separate availability flags.** If some articles have only headlines,
   distinguish `headline_available` from `full_text_available`. Keep
   `source_coverage_available` or a better named flag for source coverage,
   separate from `no_eligible_news`. Missing source coverage must not become
   a neutral sentiment value.
4. **Rebuild the three news tables after correcting mappings and flags.**
   Check that accepted article-company links use the project ticker and that
   each article is eligible by the prediction close. Deduplicate stories and
   ensure the company-day table still has one row per price company-date key.
5. **Freeze the evaluation masks before testing.** Report how many of the 50
   companies and company-days have usable news in 2020 and 2021. Compare
   price-only, news-only, and combined models on the same covered keys. Keep
   the full-50 ranking using a documented price-only fallback where source
   coverage is missing. Do not inspect 2022–2023 outcomes while choosing the
   masks, models, or fallback rule.

The audit can be rerun after checking out the qualitative branch's LFS files:

```powershell
python -m price_only_v2.audit_news_coverage --news data/qualitative_dataset/company_day_news_window_5session.csv
```
