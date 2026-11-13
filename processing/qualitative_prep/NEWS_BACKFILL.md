# BullsEye news gap backfill

The supplied tracker lists 952 ticker-months across 2020–2023. The existing
company-day grid contains dates in 921 of those months. The other 31 are
outside the current project grid, including the months before enough PLTR
price history was available.

| Year | In-scope months | Months with some new news | New company-days with news | Months still missing | Months still missing with issuer releases |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2020 | 233 | 109 | 881 | 124 | 106 |
| 2021 | 259 | 175 | 2,039 | 84 | 64 |
| 2022 | 228 | 117 | 1,354 | 111 | 89 |
| 2023 | 201 | 121 | 1,433 | 80 | 59 |
| Total | 921 | 522 | 5,707 | 399 | 318 |

The news-only supplement has 4,342 distinct stories and 4,420 company links.
Of those stories, 3,843 have an archived body. A recovered month has at
least one price-grid day with a qualifying five-session news window. News
coverage varies across the remaining days of that month.

The separate issuer version adds Linde and RTX press releases as headline
records. It has 4,771 distinct stories, 4,849 company links, and 6,779
newly covered company-days. Use the news-only version as the main input
for a comparison of independent news with price. Use the issuer version
as a separate experiment if the team wants to measure the effect of
company announcements.

## Files for Dhruv

- **missing_ticker_months_updated.csv** tracks every requested gap, preserves
  its original status, and shows the gains in days and archived bodies.
- **remaining_ticker_months.csv** filters the in-scope gaps that still
  have no eligible supplemental text. Start with 2020 training and 2021
  validation. BRK-B, IBM, AXP, and PLTR have the most unresolved months in
  the version that also includes issuer releases.
- **supplemental_articles.csv** contains one story per article ID with
  provenance, timing, title, and archived body where available.
- **supplemental_article_company_links.csv** connects stories to the
  matched companies and records the title match rule.
- **company_day_news_augmented.csv** preserves the complete original
  company-day grid and its existing flags. For a headline experiment,
  use headline_article_ids as the ID list and
  include_in_headline_comparison as the eligibility flag. The new
  supplemental columns show exactly which rows gained evidence.
- **review_sample.csv** has up to 12 links per source for spot checking.
  Links were screened automatically and a sample was reviewed. Check any
  disputed match before using it for model training.

For a headline model, read titles from the existing unique-articles file and
the supplement. When a supplemental story reuses a baseline article ID, keep
one title per ID, preferably the baseline record. Headline-only stories do
not change the original full-text flag. An uncovered day gains eligibility
only when supplemental text is in its actual window. Use the same date and
company mask for price-only, news-only, and combined model comparisons. The
backfill uses news and price dates, with no future return targets involved.

## How records were screened

A candidate required a source ticker tag and the company or an explicit
symbol in the headline, except for releases from an identified issuer
archive. The Palantir media archive supplies links to outside publishers
and is labeled as company-curated. Generic market activity roundups and
conflicting publication dates were excluded. SC454k headlines required
corroboration from the URL date or the baseline article record. URLs and
normalized title/day matches were deduplicated against accepted Nasdaq
links. The existing article ID was reused when an unambiguous same-URL
record already existed.

Each item becomes eligible on the next NYSE session after its source date
or later if the archived capture or the baseline record requires it. The
Yahoo/Common Crawl archive requires a valid capture within three days and
rejects records with a known modification after capture. Company-day links
were checked against the five-session window. No result was assigned to a
date before the story's first eligible session.

## Reproduce locally

Use the existing price and news dependencies, plus requests and
beautifulsoup4. Pull the existing company-day and article-link files via
Git LFS. Run from the repository root. Substitute your path to the
original tracker below.

~~~powershell
python processing/qualitative_prep/fetch_backfill_sources.py --gaps missing_ticker_months.csv --sources data_local/news_backfill --include-fnspid-external
python processing/qualitative_prep/sample_yahoo_news_archive.py missing_ticker_months.csv data_local/news_backfill/yahoo_targeted_metadata.csv 6500000000 22600000000 100000000 5000000
python processing/qualitative_prep/fetch_linde_releases.py
python processing/qualitative_prep/fetch_palantir_rtx_archives.py
python processing/qualitative_prep/backfill_news_gaps.py --gaps missing_ticker_months.csv --sources data_local/news_backfill --output data_local/news_backfill/news_only
python processing/qualitative_prep/backfill_news_gaps.py --gaps missing_ticker_months.csv --sources data_local/news_backfill --output data_local/news_backfill/news_and_issuer --include-issuer
~~~

The FNSPID external download is about 5.7 GB. Its relevant ticker-tagged
records ended in June 2020. The Yahoo range sampler reads a subset of the
large CSV, so a remaining gap does not establish that the full archive
has no story. The public converted Parquet for that Yahoo archive ended
in 2019 in this check. GDELT did not return a usable pilot response, and
the SEC submissions API returned HTTP 403 in this environment.

Sources

- [Drinkall Yahoo/Common Crawl archive](https://github.com/FelixDrinkall/financial-news-dataset)
- [SC454k Nasdaq archive](https://huggingface.co/datasets/nbettencourt/SC454k)
- [FNSPID external news](https://huggingface.co/datasets/Zihan1004/FNSPID/tree/main/Stock_news)
- [Early-2020 headline mirror](https://huggingface.co/datasets/ashraq/financial-news)
- [Yahoo Finance archive sampled by byte range](https://huggingface.co/datasets/edaschau/financial_news)
- [Palantir media archive](https://www.palantir.com/newsroom/media)
- [Linde company releases](https://www.linde.com/news-and-media)
- [Raytheon release archive](https://raytheon.mediaroom.com/)

The text exports are for the team's research handoff. Keep third-party
article text in local data storage and preserve publisher URLs and archive
attribution when using it.
