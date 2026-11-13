"""Backfill requested BullsEye news gaps from archived stories and headlines."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import lzma
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import duckdb
import pandas as pd
import pandas_market_calendars as mcal

from prepare_nasdaq_article_links import COMPANY_ALIASES

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/qualitative_dataset"
EXTRA_ALIASES = {
    "AAPL": ["apple inc", "iphone", "ipad", "macbook", "apple stock", "apple shares"],
    "AMZN": ["amazon.com", "aws", "amazon web services"], "AVGO": ["broadcom inc"],
    "BRK-B": ["berkshire hathaway", "berkshire's", "buffett's berkshire"],
    "GE": ["general electric", "ge aerospace", "ge stock", "ge shares"],
    "MA": ["mastercard", "master card"], "META": ["facebook", "meta platforms", "meta stock", "meta shares"],
    "MCD": ["mcdonald's", "mcdonalds", "mcdonald"], "NVDA": ["nvidia"],
    "RTX": ["raytheon technologies", "rtx corporation", "rtx stock", "rtx shares", "raytheon"],
    "V": ["visa inc", "visa stock", "visa shares", "visa and mastercard", "mastercard and visa"],
}
RESTRICT = {"AAPL", "BRK-B", "CAT", "GE", "LLY", "META", "ORCL", "RTX", "V"}
SYMBOL_MAP = {"FB": "META", "GOOG": "GOOGL", "BRK.B": "BRK-B",
              "BRK/B": "BRK-B", "BRKB": "BRK-B", "BRK.A": "BRK-B", "BRK-A": "BRK-B"}
CUTOFF = pd.Timestamp("2020-04-03").date()
URL_DATE = re.compile(r"(?<!\d)(20\d{2})[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])(?!\d)")
MARKET_ROUNDUP = re.compile(r"^(?:after.hours|pre.market) most active\b|\bsector update for\b", re.I)


def normalized_url(url):
    try:
        parsed = urlsplit(str(url or "").strip())
    except ValueError:
        return ""
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or not parsed.path.strip("/"):
        return ""
    host = parsed.hostname.lower().removeprefix("www.")
    path = unquote(parsed.path).rstrip("/").lower()
    if host.endswith("benzinga.com") and (match := re.search(r"/(\d{6,})(?:/|$)", path)):
        return f"benzinga.com/{match[1]}"
    return host + path


def clean(value):
    return html.unescape(str(value or "")).strip()


def normalized_title(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def headline_match(ticker, title, tag):
    aliases = EXTRA_ALIASES.get(ticker, [])
    if ticker not in RESTRICT:
        aliases = aliases + COMPANY_ALIASES.get(ticker, [])
    if ticker in {"CAT", "LLY", "ORCL"}:
        aliases = aliases + {"CAT": ["caterpillar"], "LLY": ["eli lilly"], "ORCL": ["oracle corp", "oracle cloud", "oracle stock", "oracle shares"]}[ticker]
    for alias in aliases:
        if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", title, flags=re.I):
            return "company_in_title"
    symbols = {ticker, tag} - {""}
    for symbol in symbols:
        text = re.escape(symbol)
        if re.search(rf"(?<!\w)\${text}(?!\w)|\({text}\)|\b(?:NASDAQ|NYSE)\s*:\s*{text}(?!\w)", title, re.I):
            return "explicit_symbol_in_title"
        if len(symbol) >= 3 and symbol not in {"CAT", "META", "NOW", "COST", "DIS", "LIN", "PEP", "MU", "IBM"}:
            if re.search(rf"(?<!\w){text}(?!\w)", title):
                return "uppercase_symbol_in_title"
    return ""


def from_sources(folder, include_issuer):
    db = duckdb.connect()
    for year in range(2020, 2024):
        with lzma.open(folder / f"{year}_processed.json.xz", "rt", encoding="utf-8") as stream:
            for row in json.load(stream):
                yield ("drinkall_yahoo", row.get("url"), row.get("date_publish"), row.get("title"),
                       row.get("mentioned_companies") or [], row.get("news_outlet"),
                       row.get("maintext"), row.get("date_download"), row.get("date_modify"))
    rows = db.execute("""
        select url, date, headline, stock, publisher from read_parquet(?)
        where date >= '2020-01-01' and date < '2024-01-01'
        """, [str(folder / "headlines.parquet")]).fetchall()
    for url, date, title, symbol, publisher in rows:
        yield "ashraq_headlines", url, date, title, [symbol], publisher, None, None, None
    rows = db.execute("""
        select URL, Date, Title, Symbol, RelatedStocksList, Publication from read_parquet(?)
        where Date >= '2020-01-01' and Date < '2024-01-01'
        """, [str(folder / "sc_metadata.parquet")]).fetchall()
    for url, date, title, symbol, related, publisher in rows:
        yield "sc454k_nasdaq", url, date, title, [symbol, *(related or "").split("|")], publisher, None, None, None
    external = folder / "fnspid_external_metadata.parquet"
    if external.exists():
        rows = db.execute("select url, date, title, symbol, publisher from read_parquet(?)", [str(external)]).fetchall()
        for url, date, title, symbol, publisher in rows:
            yield "fnspid_external", url, date, title, [symbol], publisher, None, None, None
    targeted = folder / "yahoo_targeted_metadata.csv"
    if targeted.exists():
        for row in pd.read_csv(targeted, dtype=str).fillna("").itertuples(index=False):
            yield "edaschau_yahoo", row.url, row.date_time, row.title, [row.ticker], row.source, None, None, None
    curated = folder / "palantir_media.csv"
    if curated.exists():
        for row in pd.read_csv(curated, dtype=str).fillna("").itertuples(index=False):
            yield "palantir_curated_media", row.url, row.date, row.title, [row.ticker], row.publisher, None, None, None
    if include_issuer:
        for filename, source in (("linde_press_releases.csv", "linde_issuer"),
                                 ("rtx_press_releases.csv", "rtx_issuer")):
            for row in pd.read_csv(folder / filename, dtype=str).itertuples(index=False):
                yield source, row.url, row.date, row.title, [row.ticker], row.publisher, None, None, None


def align_and_filter(folder, gaps, include_issuer):
    collected = []
    for source, url, raw_date, raw_title, tags, publisher, body, captured, modified in from_sources(folder, include_issuer):
        title = clean(raw_title)
        canonical = normalized_url(url)
        if not raw_date or len(title) < 8 or not canonical or MARKET_ROUNDUP.search(title):
            continue
        url_date = URL_DATE.search(urlsplit(clean(url)).path)
        date_in_url = "-".join(url_date.groups()) if url_date else ""
        if date_in_url and date_in_url[:7] != str(raw_date)[:7]:
            continue
        if source == "sc454k_nasdaq" and str(raw_date)[:10] == "2020-04-17" and date_in_url != "2020-04-17":
            continue
        symbols = {str(tag).upper().strip() for tag in tags if tag}
        for tag in sorted(symbols):
            ticker = SYMBOL_MAP.get(tag, tag)
            if (ticker, str(raw_date)[:7]) not in gaps:
                continue
            if ticker == "RTX" and str(raw_date)[:10] < CUTOFF.isoformat():
                continue
            reason = "issuer_archive" if source.endswith("_issuer") else headline_match(ticker, title, tag)
            if reason:
                collected.append((ticker, tag, str(raw_date), title, clean(url), canonical, clean(publisher),
                                  source, reason, clean(body), captured, modified))
    df = pd.DataFrame(collected, columns=[
        "ticker", "original_source_tag", "raw_publication_date", "title", "url", "canonical_url",
        "publisher", "source", "match_reason", "archive_body", "captured", "modified",
    ])
    if df.empty:
        raise ValueError("No source records matched the requested gaps")
    published = pd.to_datetime(df.raw_publication_date, format="mixed", errors="coerce", utc=True)
    df = df.loc[published.notna()].copy()
    published = published.loc[df.index]
    # Drinkall documents UTC. For other sources the source date is used conservatively.
    known = df.source.eq("drinkall_yahoo")
    captured = pd.to_datetime(df.captured, format="mixed", errors="coerce", utc=True)
    modified = pd.to_datetime(df.modified, format="mixed", errors="coerce", utc=True)
    # Score archived content only after the archive demonstrably had that version.
    capture_ok = (known & captured.notna() & captured.ge(published)
                  & captured.le(published + pd.Timedelta(days=3))
                  & (modified.isna() | modified.le(captured)))
    df = df.loc[~known | capture_ok].copy()
    known, capture_ok = known.loc[df.index], capture_ok.loc[df.index]
    published = published.loc[df.index]
    df["article_body"] = df.archive_body.where(capture_ok & df.archive_body.str.len().ge(200), "")
    df["text_type"] = df.article_body.ne("").map({True: "archived_article", False: "headline_only"})
    utc_date = known | df.source.eq("edaschau_yahoo")
    df["published_date"] = df.raw_publication_date.str[:10]
    df.loc[utc_date, "published_date"] = published[utc_date].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    df = df.loc[[key in gaps for key in zip(df.ticker, df.published_date.str[:7])]].copy()
    published, captured = published.loc[df.index], captured.loc[df.index]
    known, capture_ok = known.loc[df.index], capture_ok.loc[df.index]
    df["publication_month"] = df.published_date.str[:7]

    schedule = mcal.get_calendar("NYSE").schedule(start_date="2020-01-01", end_date="2024-01-15")
    sessions = pd.Index(schedule.index.strftime("%Y-%m-%d"))
    conservative = sessions.searchsorted(df.published_date, side="right")
    df["eligible_trading_date"] = sessions.take(conservative)
    df["eligibility_rule"] = "source_date_following_nyse_session"
    if capture_ok.any():
        availability = captured[capture_ok]
        index = pd.Index(schedule.market_close).searchsorted(availability)
        later_session = sessions.take(index)
        df.loc[capture_ok, "eligible_trading_date"] = [
            max(left, right) for left, right in zip(df.loc[capture_ok, "eligible_trading_date"], later_session)
        ]
        df.loc[capture_ok, "eligibility_rule"] = "later_of_next_session_and_archive_capture"
    df["archive_captured_at_utc"] = captured
    df["title_key"] = df.title.map(normalized_title)
    return df, list(sessions)


def build(gaps_path, folder, output, include_issuer=False):
    gaps = pd.read_csv(gaps_path, dtype=str)
    keys = ["ticker", "month"]
    if gaps.duplicated(keys).any() or not gaps.month.str.fullmatch(r"202[0-3]-(0[1-9]|1[0-2])").all():
        raise ValueError("Expected unique requested ticker-months in 2020–2023")
    universe = set(pd.read_csv(ROOT / "final_dataset/dataset_summary.csv").ticker)
    if not set(gaps.ticker).issubset(universe):
        raise ValueError("Tracker contains tickers outside the project universe")
    grid = pd.read_csv(BASE / "company_day_news_window_5session.csv", usecols=[
        "ticker", "prediction_date", "coverage_month", "source_coverage_available", "news_count",
    ])
    grid = grid.loc[grid.prediction_date.str[:4].isin(["2020", "2021", "2022", "2023"])]
    grid = grid.merge(gaps[keys], left_on=["ticker", "coverage_month"], right_on=keys, validate="many_to_one")
    if grid.duplicated(["ticker", "prediction_date"]).any():
        raise ValueError("Duplicate model dates")

    rows, sessions = align_and_filter(folder, set(map(tuple, gaps[keys].to_numpy())), include_issuer)
    raw_counts = rows.groupby(["ticker", "publication_month"]).size()
    links = pd.read_csv(BASE / "nasdaq_article_company_links_2020_2023.csv", usecols=[
        "ticker", "Url", "Article_title", "Date", "match_status", "article_id", "eligible_trading_date",
    ], dtype=str).fillna("")
    links["canonical_url"] = links.Url.map(normalized_url)
    baseline = links.loc[links.canonical_url.ne("")].groupby("canonical_url").agg(
        baseline_article_id=("article_id", "min"),
        baseline_ids=("article_id", "nunique"),
        baseline_date_min=("Date", "min"),
        baseline_date_max=("Date", "max"),
        baseline_eligible_date=("eligible_trading_date", "max"),
    )
    rows = rows.join(baseline, on="canonical_url")
    # The existing archive can resolve some publication dates. A discrepancy of
    # more than a calendar day means the story version or time is ambiguous.
    base_min = pd.to_datetime(rows.baseline_date_min.str[:10], errors="coerce")
    base_max = pd.to_datetime(rows.baseline_date_max.str[:10], errors="coerce")
    local_date = pd.to_datetime(rows.published_date)
    one_day = pd.Timedelta(days=1)
    conflict = ((local_date - base_min).abs() > one_day) | ((local_date - base_max).abs() > one_day)
    sc = rows.source.eq("sc454k_nasdaq")
    dated_url = rows.url.map(lambda url: URL_DATE.search(urlsplit(url).path))
    url_day = pd.to_datetime(dated_url.map(lambda match: "-".join(match.groups()) if match else None),
                             errors="coerce")
    date_confirmed = ((local_date - url_day).abs().le(one_day) | (local_date - base_min).abs().le(one_day))
    rows = rows.loc[~conflict & ~rows.baseline_ids.gt(1) & (~sc | date_confirmed)].copy()
    rows["eligible_trading_date"] = rows[["eligible_trading_date", "baseline_eligible_date"]].fillna("").max(axis=1)
    links = links.loc[links.match_status.eq("accepted")]
    known_urls = set(zip(links.ticker, links.canonical_url))
    known_titles = set(zip(links.ticker, links.Date.str[:10], links.Article_title.map(normalized_title)))
    duplicate = [(t, u) in known_urls or (t, d, h) in known_titles
                 for t, u, d, h in zip(rows.ticker, rows.canonical_url, rows.published_date, rows.title_key)]
    rows = rows.loc[~pd.Series(duplicate, index=rows.index)].copy()
    rows = rows.sort_values(["text_type", "eligible_trading_date", "source", "url"])
    rows = rows.drop_duplicates(["ticker", "canonical_url"]).drop_duplicates(["ticker", "published_date", "title_key"])
    # One stable ID and one conservative availability timestamp per story across all links.
    story_date = rows.groupby("canonical_url").eligible_trading_date.transform("max")
    rows["eligible_trading_date"] = story_date
    new_id = rows.canonical_url.map(lambda url: "bf_" + hashlib.sha256(url.encode()).hexdigest()[:20])
    rows["article_id"] = rows.baseline_article_id.fillna(new_id)

    session_index = {day: i for i, day in enumerate(sessions)}
    def window(day):
        i = session_index[day]
        return sessions[i:i + 5]
    windows = rows.assign(prediction_date=rows.eligible_trading_date.map(window)).explode("prediction_date")
    windows = windows.merge(grid[["ticker", "prediction_date"]], on=["ticker", "prediction_date"], validate="many_to_one")
    windows = windows.loc[windows.prediction_date.ge(windows.eligible_trading_date)]
    counts = windows.groupby(["ticker", "prediction_date"]).article_id.nunique()
    body_counts = (windows.loc[windows.text_type.eq("archived_article")]
                   .groupby(["ticker", "prediction_date"]).article_id.nunique())
    id_strings = windows.groupby(["ticker", "prediction_date"]).article_id.agg(lambda ids: "|".join(sorted(set(ids))))
    usage = pd.concat([counts.rename("supplemental_news_count"),
                       body_counts.rename("supplemental_archived_body_count"),
                       id_strings.rename("supplemental_article_ids")], axis=1)
    daily = grid.join(usage, on=["ticker", "prediction_date"])
    daily["supplemental_news_count"] = daily.supplemental_news_count.fillna(0).astype(int)
    daily["supplemental_archived_body_count"] = daily.supplemental_archived_body_count.fillna(0).astype(int)
    daily["supplemental_article_ids"] = daily.supplemental_article_ids.fillna("").str.replace("|", ",", regex=False)
    daily["has_supplement"] = daily.supplemental_news_count.gt(0)
    daily["newly_usable_headline_row"] = daily.has_supplement & ~daily.source_coverage_available

    stats = rows.groupby(["ticker", "publication_month"]).agg(
        new_story_links=("article_id", "nunique"),
        archive_body_links=("text_type", lambda values: values.eq("archived_article").sum()),
    )
    daily_stats = daily.groupby(["ticker", "month"]).agg(
        project_company_days=("prediction_date", "size"),
        days_with_supplement=("has_supplement", "sum"),
        days_with_archived_body=("supplemental_archived_body_count", lambda x: x.gt(0).sum()),
        newly_usable_headline_days=("newly_usable_headline_row", "sum"),
        days_with_prior_accepted_links=("news_count", lambda values: values.gt(0).sum()),
    )
    tracker = gaps.join(raw_counts.rename("matched_records_before_dedup"), on=keys).join(stats, on=keys).join(daily_stats, on=keys)
    numeric = tracker.columns.difference(gaps.columns)
    tracker[numeric] = tracker[numeric].fillna(0).astype(int)
    tracker["backfill_status"] = "no_eligible_supplement_found"
    tracker.loc[tracker.days_with_supplement.gt(0), "backfill_status"] = "has_supplement_for_some_days"
    tracker.loc[tracker.project_company_days.eq(0), "backfill_status"] = "outside_project_grid"

    output.mkdir(parents=True, exist_ok=True)
    articles = rows.drop_duplicates("article_id")
    articles[["article_id", "baseline_article_id", "source", "publisher", "url", "raw_publication_date", "published_date",
              "archive_captured_at_utc", "eligible_trading_date", "eligibility_rule",
              "text_type", "title", "article_body"]].to_csv(output / "supplemental_articles.csv", index=False)
    rows[["article_id", "ticker", "original_source_tag", "source", "match_reason",
          "eligible_trading_date", "publication_month"]].to_csv(output / "supplemental_article_company_links.csv", index=False)
    daily.to_csv(output / "company_day_supplement.csv", index=False)
    baseline_grid = pd.read_csv(BASE / "company_day_news_window_5session.csv", dtype={"article_ids": str})
    augmented = baseline_grid.merge(daily[["ticker", "prediction_date", "supplemental_article_ids",
                                           "supplemental_news_count", "supplemental_archived_body_count",
                                           "newly_usable_headline_row"]],
                                   on=["ticker", "prediction_date"], how="left", validate="one_to_one")
    augmented["supplemental_article_ids"] = augmented.supplemental_article_ids.fillna("")
    for column in ["supplemental_news_count", "supplemental_archived_body_count"]:
        augmented[column] = augmented[column].fillna(0).astype(int)
    augmented["newly_usable_headline_row"] = augmented.newly_usable_headline_row.eq(True)
    augmented["include_in_headline_comparison"] = (
        augmented.include_in_multimodal_comparison | augmented.newly_usable_headline_row
    )
    augmented["headline_article_ids"] = [
        ",".join(sorted(set(filter(None, f"{old},{new}".split(",")))))
        for old, new in zip(augmented.article_ids.fillna(""), augmented.supplemental_article_ids)
    ]
    augmented.to_csv(output / "company_day_news_augmented.csv", index=False)
    tracker.to_csv(output / "missing_ticker_months_updated.csv", index=False)
    # Attach a small review batch per source to check whether ticker/title matches are meaningful.
    review = pd.concat([group.sample(n=min(12, len(group)), random_state=42)
                        for _, group in rows.groupby("source")])
    review[["article_id", "ticker", "source", "published_date", "title", "url", "match_reason"]].to_csv(
        output / "review_sample.csv", index=False)
    print(tracker.groupby([tracker.month.str[:4], "backfill_status"]).size().to_string())
    print(rows.groupby(["source", "text_type"]).size().to_string())
    print(f"{len(articles):,} stories, {len(rows):,} ticker links, {daily.newly_usable_headline_row.sum():,} usable company-days")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaps", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-issuer", action="store_true")
    args = parser.parse_args()
    build(args.gaps, args.sources, args.output, args.include_issuer)
