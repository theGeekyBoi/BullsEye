"""Prepare the pilot qualitative/news dataset from FNSPID Nasdaq articles.

Expected input:
    data/raw/nasdaq_exteral_data.csv

Expected columns:
    Date, Article_title, Stock_symbol, Url, Publisher, Author, Article,
    Lsa_summary, Luhn_summary, Textrank_summary, Lexrank_summary
"""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import hashlib
from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/raw/nasdaq_exteral_data.csv")
TRADING_CALENDAR_PATH = Path("data/processed/trading_calendar.csv")
OUTPUT_DIR = Path("data/processed/qualitative")
PILOT_SYMBOLS = {"AMD", "WFC", "MRK"}

OUTPUT_COLUMNS = [
    "article_id",
    "Date",
    "eligible_date",
    "Stock_symbol",
    "Article_title",
    "Url",
    "Publisher",
    "Author",
    "Article",
    "Lsa_summary",
    "Luhn_summary",
    "Textrank_summary",
    "Lexrank_summary",
    "relevance_status",
    "article_scope",
    "article_type",
    "manual_notes",
]


def load_trading_calendar() -> pd.DataFrame | None:
    if not TRADING_CALENDAR_PATH.exists():
        return None

    calendar = pd.read_csv(TRADING_CALENDAR_PATH, dtype=str)
    required_columns = {"date", "close_et"}
    missing_columns = required_columns - set(calendar.columns)
    if missing_columns:
        raise ValueError(f"trading_calendar.csv is missing columns: {sorted(missing_columns)}")

    calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce")
    calendar["close_time"] = (
        pd.to_datetime(calendar["close_et"], errors="coerce", utc=True)
        .dt.tz_convert("America/New_York")
        .dt.time
    )
    calendar = calendar.dropna(subset=["date", "close_time"])
    calendar["date"] = calendar["date"].dt.date.astype(str)

    return calendar.sort_values("date").reset_index(drop=True)


def make_article_id(row: pd.Series) -> str:
    values = [
        str(row.get("Date", "")),
        str(row.get("Stock_symbol", "")),
        str(row.get("Article_title", "")),
        str(row.get("Url", "")),
        str(row.get("Article", "")),
    ]
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()[:16]


def next_trading_date(date_value: str, trading_dates: list[str]) -> str:
    position = bisect_right(trading_dates, date_value)
    if position >= len(trading_dates):
        return ""
    return trading_dates[position]


def trading_date_on_or_after(date_value: str, trading_dates: list[str]) -> str:
    position = bisect_left(trading_dates, date_value)
    if position >= len(trading_dates):
        return ""

    gap_days = (
        pd.to_datetime(trading_dates[position]) - pd.to_datetime(date_value)
    ).days
    if gap_days > 7:
        return ""

    return trading_dates[position]


def get_eligible_date(date_value: object, calendar: pd.DataFrame | None) -> str:
    parsed = pd.to_datetime(date_value, errors="coerce")
    if pd.isna(parsed):
        return ""

    source_date = parsed.date().isoformat()

    if calendar is not None:
        trading_dates = calendar["date"].tolist()
        close_by_date = dict(zip(calendar["date"], calendar["close_time"]))

        if source_date not in close_by_date:
            return trading_date_on_or_after(source_date, trading_dates)

        if parsed.time().hour == 0 and parsed.time().minute == 0 and parsed.time().second == 0:
            return next_trading_date(source_date, trading_dates)

        if parsed.time() >= close_by_date[source_date]:
            return next_trading_date(source_date, trading_dates)

        return source_date

    # Date-only or midnight rows are delayed to avoid using news too early.
    if parsed.time().hour == 0 and parsed.time().minute == 0 and parsed.time().second == 0:
        return (parsed.normalize() + pd.offsets.BDay(1)).date().isoformat()

    # If the timestamp is after the normal market close, use the next business day.
    if parsed.time().hour >= 16:
        return (parsed.normalize() + pd.offsets.BDay(1)).date().isoformat()

    return parsed.date().isoformat()


def load_pilot_articles(chunksize: int) -> pd.DataFrame:
    chunks = []

    for chunk_number, chunk in enumerate(
        pd.read_csv(INPUT_PATH, dtype=str, chunksize=chunksize, low_memory=False),
        start=1,
    ):
        chunk["Stock_symbol"] = chunk["Stock_symbol"].str.upper().str.strip()
        chunk = chunk[chunk["Stock_symbol"].isin(PILOT_SYMBOLS)].copy()

        if chunk.empty:
            continue

        chunks.append(chunk)
        print(f"chunk {chunk_number}: kept {len(chunk):,} pilot rows")

    if not chunks:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    return pd.concat(chunks, ignore_index=True)


def prepare_articles(articles: pd.DataFrame, calendar: pd.DataFrame | None) -> pd.DataFrame:
    articles = articles.copy()

    articles = articles.drop_duplicates(
        subset=["Date", "Stock_symbol", "Article_title", "Url", "Article"],
        keep="first",
    )

    articles["article_id"] = articles.apply(make_article_id, axis=1)
    articles["eligible_date"] = articles["Date"].apply(lambda value: get_eligible_date(value, calendar))

    # These columns are intentionally blank. Fill them during manual review.
    articles["relevance_status"] = ""
    articles["article_scope"] = ""
    articles["article_type"] = ""
    articles["manual_notes"] = ""

    return articles[OUTPUT_COLUMNS].sort_values(["Stock_symbol", "eligible_date", "Date"])


def build_company_day_news(articles: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    valid_articles = articles[articles["eligible_date"] != ""].copy()

    company_days = pd.MultiIndex.from_product(
        [sorted(PILOT_SYMBOLS), calendar["date"]],
        names=["Stock_symbol", "date"],
    ).to_frame(index=False)

    grouped_articles = (
        valid_articles
        .groupby(["Stock_symbol", "eligible_date"])["article_id"]
        .apply(lambda values: ",".join(values))
        .reset_index()
        .rename(columns={"eligible_date": "date", "article_id": "article_ids"})
    )

    company_day_news = company_days.merge(
        grouped_articles,
        on=["Stock_symbol", "date"],
        how="left",
    )
    company_day_news["article_ids"] = company_day_news["article_ids"].fillna("")
    company_day_news["article_count"] = company_day_news["article_ids"].apply(
        lambda value: 0 if value == "" else len(value.split(","))
    )
    company_day_news["no_news"] = company_day_news["article_count"] == 0

    return company_day_news


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Could not find input file: {INPUT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    articles = load_pilot_articles(args.chunksize)
    calendar = load_trading_calendar()
    articles = prepare_articles(articles, calendar)

    processed_path = OUTPUT_DIR / "pilot_nasdaq_articles.csv"
    company_day_path = OUTPUT_DIR / "pilot_company_day_news.csv"

    articles.to_csv(processed_path, index=False)

    print("\nDone.")
    print(f"Processed pilot articles: {len(articles):,} -> {processed_path}")

    if calendar is not None:
        company_day_news = build_company_day_news(articles, calendar)
        company_day_news.to_csv(company_day_path, index=False)
        print(f"Company-day news rows: {len(company_day_news):,} -> {company_day_path}")
    else:
        print(f"Skipped company-day news file because {TRADING_CALENDAR_PATH} does not exist.")


if __name__ == "__main__":
    main()
