"""Add manually verified headlines to the development news eligibility file."""
import argparse
import hashlib
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal

from backfill_news_gaps import BASE, normalized_title, normalized_url

HANDOFF = Path("To Do/dhruv_news_comparison")
KEYS = ["ticker", "prediction_date"]
FLAG = "include_in_headline_comparison"


def append_headlines(verified, output):
    rows = pd.read_csv(verified, dtype=str).fillna("")
    required = ["ticker", "publication_date", "headline", "source_url"]
    if set(rows.columns) != set(required):
        raise ValueError(f"Expected CSV columns {', '.join(required)}")
    rows = rows.apply(lambda column: column.str.strip())
    date = pd.to_datetime(rows.publication_date, format="%Y-%m-%d", errors="coerce")
    if date.isna().any() or not rows.publication_date.eq(date.dt.strftime("%Y-%m-%d")).all():
        raise ValueError("Use the original publication date in YYYY-MM-DD format")
    months = pd.read_csv(HANDOFF / "priority_company_months.csv", usecols=["ticker", "month"], dtype=str)
    requested = pd.MultiIndex.from_frame(months)
    keys = pd.MultiIndex.from_arrays([rows.ticker, rows.publication_date.str[:7]])
    if not keys.isin(requested).all():
        raise ValueError("Each verified headline must belong to a requested ticker and month")
    rows["url_key"] = rows.source_url.map(normalized_url)
    rows["title_key"] = rows.headline.map(normalized_title)
    if (rows.url_key.eq("") | rows.title_key.eq("")).any():
        raise ValueError("Each verified row needs a headline and original article URL")

    links = pd.read_csv(BASE / "nasdaq_article_company_links_2020_2023.csv",
                        usecols=["ticker", "Url", "Date", "Article_title", "match_status"], dtype=str).fillna("")
    accepted = links.loc[links.match_status.eq("accepted")]
    existing = set(zip(accepted.ticker, accepted.Url.map(normalized_url)))
    titles = set(zip(accepted.ticker, accepted.Date.str[:10], accepted.Article_title.map(normalized_title)))
    supplements = pd.read_csv(HANDOFF / "supplemental_story_index.csv", usecols=["url"], dtype=str)
    supplement_urls = set(supplements.url.map(normalized_url))
    rows = rows.loc[[
        (t, u) not in existing and (t, d, h) not in titles and u not in supplement_urls
        for t, u, d, h in zip(rows.ticker, rows.url_key, rows.publication_date, rows.title_key)
    ]].copy()
    rows = rows.sort_values(["publication_date", "ticker", "source_url"])
    rows = rows.drop_duplicates(["ticker", "url_key"]).drop_duplicates(
        ["ticker", "publication_date", "title_key"])
    rows["article_id"] = rows.url_key.map(
        lambda url: "manual_" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:20])

    # The template accepts dates only. Use the latest reported date for any shared story.
    dates = rows.groupby("article_id").publication_date.transform("max")
    sessions = mcal.get_calendar("NYSE").schedule("2020-01-01", "2022-01-15").index.strftime("%Y-%m-%d")
    starts = sessions.searchsorted(dates, side="right")
    rows["eligible_trading_date"] = sessions.take(starts)
    grid = pd.read_csv(HANDOFF / "headline_eligibility_2020_2021.csv",
                       dtype={"headline_article_ids": str})
    lookup = {}
    for row, index in zip(rows.itertuples(index=False), starts):
        for session in sessions[index:index + 5]:
            lookup.setdefault((row.ticker, session), set()).add(row.article_id)
    new_ids = [lookup.get(tuple(key), set()) for key in grid[KEYS].to_numpy()]
    grid["headline_article_ids"] = [
        ",".join(sorted(set(filter(None, prior.split(","))) | added))
        for prior, added in zip(grid.headline_article_ids.fillna(""), new_ids)
    ]
    grid[FLAG] = grid[FLAG] | pd.Series(map(bool, new_ids), dtype=bool)
    output.mkdir(parents=True, exist_ok=True)
    rows[[*required, "article_id", "eligible_trading_date"]].to_csv(
        output / "verified_headlines_prepared.csv", index=False)
    grid.to_csv(output / "headline_eligibility_2020_2021.csv", index=False)
    print(f"Added {len(rows)} company-story links across {sum(map(bool, new_ids))} company-days.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verified", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    append_headlines(args.verified, args.output)
