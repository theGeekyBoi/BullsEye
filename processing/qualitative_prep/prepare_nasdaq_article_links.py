"""Deduplicate Nasdaq stories and verify source-tagged company links.

This script uses source tags from FNSPID's Nasdaq news file, maps them to the
modeling tickers, assigns stable article IDs, and flags uncertain company links
for review when the headline/body do not mention the ticker or company name.

Outputs:
    data/processed/qualitative/nasdaq_unique_articles_2020_2023.csv
    data/processed/qualitative/nasdaq_article_company_links_2020_2023.csv
    data/processed/qualitative/nasdaq_company_match_review_queue_2020_2023.csv
"""

from __future__ import annotations

import hashlib
import re
from bisect import bisect_left, bisect_right

import duckdb
import pandas as pd

from audit_ticker_mappings import (
    NASDAQ_NEWS_PATH,
    OUTPUT_DIR,
    REPO_ROOT,
    candidate_symbols,
    load_model_tickers,
    sql_string,
)


YEARS = ["2020", "2021", "2022", "2023"]
TRADING_CALENDAR_PATH = REPO_ROOT / "data/raw/trading_calendar.csv"

COMPANY_ALIASES = {
    "AAPL": ["apple"],
    "ABBV": ["abbvie"],
    "ABT": ["abbott"],
    "ACN": ["accenture"],
    "ADBE": ["adobe"],
    "AMD": ["advanced micro devices", "amd"],
    "AMGN": ["amgen"],
    "AMZN": ["amazon"],
    "AVGO": ["broadcom"],
    "AXP": ["american express", "amex"],
    "BAC": ["bank of america"],
    "BRK-B": ["berkshire hathaway", "berkshire", "warren buffett"],
    "CAT": ["caterpillar"],
    "COST": ["costco"],
    "CRM": ["salesforce"],
    "CSCO": ["cisco"],
    "CVX": ["chevron"],
    "DIS": ["disney", "walt disney"],
    "GE": ["general electric", "ge aerospace"],
    "GOOGL": ["alphabet", "google"],
    "GS": ["goldman sachs"],
    "HD": ["home depot"],
    "IBM": ["international business machines", "ibm"],
    "JNJ": ["johnson & johnson", "johnson and johnson"],
    "JPM": ["jpmorgan", "jp morgan", "jpmorgan chase"],
    "KO": ["coca-cola", "coca cola"],
    "LIN": ["linde"],
    "LLY": ["eli lilly", "lilly"],
    "MA": ["mastercard"],
    "MCD": ["mcdonald"],
    "META": ["meta", "facebook"],
    "MRK": ["merck"],
    "MSFT": ["microsoft"],
    "MU": ["micron"],
    "NFLX": ["netflix"],
    "NOW": ["servicenow", "service now"],
    "NVDA": ["nvidia"],
    "ORCL": ["oracle"],
    "PEP": ["pepsico", "pepsi"],
    "PLTR": ["palantir"],
    "QCOM": ["qualcomm"],
    "RTX": ["rtx", "raytheon", "united technologies"],
    "TMO": ["thermo fisher"],
    "TMUS": ["t-mobile", "tmobile"],
    "TSLA": ["tesla"],
    "TXN": ["texas instruments"],
    "V": ["visa"],
    "WFC": ["wells fargo"],
    "WMT": ["walmart", "wal-mart"],
    "XOM": ["exxon", "exxon mobil", "exxonmobil"],
}


def load_trading_calendar() -> pd.DataFrame:
    calendar = pd.read_csv(TRADING_CALENDAR_PATH, dtype=str)
    required_columns = {"date", "close_et"}
    missing_columns = required_columns - set(calendar.columns)
    if missing_columns:
        raise ValueError(f"trading_calendar.csv is missing columns: {sorted(missing_columns)}")

    calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce")
    calendar["close_timestamp_et"] = (
        pd.to_datetime(calendar["close_et"], errors="coerce", utc=True)
        .dt.tz_convert("America/New_York")
    )
    calendar = calendar.dropna(subset=["date", "close_timestamp_et"])
    calendar["date"] = calendar["date"].dt.date.astype(str)
    return calendar.sort_values("date").reset_index(drop=True)


def next_trading_session(date_value: str, trading_dates: list[str]) -> str:
    position = bisect_right(trading_dates, date_value)
    if position >= len(trading_dates):
        return ""
    return trading_dates[position]


def trading_session_on_or_after(date_value: str, trading_dates: list[str]) -> str:
    position = bisect_left(trading_dates, date_value)
    if position >= len(trading_dates):
        return ""
    return trading_dates[position]


def assign_eligible_trading_session(
    date_value: object,
    trading_dates: list[str],
    close_by_date: dict[str, pd.Timestamp],
) -> tuple[str, str]:
    published_at = pd.to_datetime(date_value, errors="coerce", utc=True)
    if pd.isna(published_at):
        return "", "missing_datetime"

    stated_date = published_at.date().isoformat()

    if published_at.time().hour == 0 and published_at.time().minute == 0 and published_at.time().second == 0:
        return next_trading_session(stated_date, trading_dates), "midnight_or_date_only_next_session"

    if stated_date not in close_by_date:
        return trading_session_on_or_after(stated_date, trading_dates), "non_trading_day_next_session"

    published_at_et = published_at.tz_convert("America/New_York")
    if published_at_et <= close_by_date[stated_date]:
        return stated_date, "published_before_or_at_close"

    return next_trading_session(stated_date, trading_dates), "after_close_next_session"


def stable_article_id(row: pd.Series) -> str:
    url = clean_text(row.get("Url", "")).lower()
    if url:
        raw = url
    else:
        raw = "\n".join(
            [
                clean_text(row.get("Date", "")),
                clean_text(row.get("Article_title", "")),
                clean_text(row.get("Article", "")),
            ]
        )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def has_strong_ticker_mention(text: str, ticker: str) -> bool:
    text = text.upper()
    ticker = ticker.upper()
    patterns = [
        rf"\({re.escape(ticker)}\)",
        rf"\bNASDAQ\s*:\s*{re.escape(ticker)}\b",
        rf"\bNYSE\s*:\s*{re.escape(ticker)}\b",
        rf"\bSYMBOL\s*:\s*{re.escape(ticker)}\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def has_company_alias(text: str, ticker: str) -> bool:
    text = text.lower()
    return any(alias in text for alias in COMPANY_ALIASES.get(ticker, []))


def verify_company_match(row: pd.Series) -> tuple[str, str, bool]:
    ticker = row["ticker"]
    title = clean_text(row.get("Article_title", ""))
    body = clean_text(row.get("Article", ""))
    searchable = f"{title}\n{body}"

    if has_strong_ticker_mention(title, ticker):
        return "accepted", "strong_ticker_mention_in_headline", False
    if has_company_alias(title, ticker):
        return "accepted", "company_name_mention_in_headline", False
    if has_strong_ticker_mention(body, ticker):
        return "accepted", "strong_ticker_mention_in_body", False
    if has_company_alias(body, ticker):
        return "accepted", "company_name_mention_in_body", False
    if not searchable.strip():
        return "uncertain", "missing_headline_and_body", True
    return "uncertain", "source_tag_only_no_text_match", True


def build_symbol_mapping() -> pd.DataFrame:
    rows = []
    for ticker in load_model_tickers()["ticker"]:
        for nasdaq_symbol, match_type in candidate_symbols(ticker):
            rows.append(
                {
                    "ticker": ticker,
                    "original_source_tag": nasdaq_symbol,
                    "source_tag_match_type": match_type,
                }
            )
    return pd.DataFrame(rows)


def load_source_tagged_rows(source_tags: list[str]) -> pd.DataFrame:
    source_tag_list = ", ".join(sql_string(symbol) for symbol in source_tags)
    year_list = ", ".join(sql_string(year) for year in YEARS)

    query = f"""
    SELECT
        Date,
        Article_title,
        upper(trim(Stock_symbol)) AS original_source_tag,
        Url,
        Publisher,
        Author,
        Article,
        Lsa_summary,
        Luhn_summary,
        Textrank_summary,
        Lexrank_summary
    FROM read_csv_auto(
        {sql_string(NASDAQ_NEWS_PATH.as_posix())},
        header=true,
        all_varchar=true,
        strict_mode=false,
        ignore_errors=true
    )
    WHERE upper(trim(Stock_symbol)) IN ({source_tag_list})
      AND substr(Date, 1, 4) IN ({year_list})
    """
    return duckdb.sql(query).df()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    calendar = load_trading_calendar()
    trading_dates = calendar["date"].tolist()
    close_by_date = dict(zip(calendar["date"], calendar["close_timestamp_et"]))
    symbol_mapping = build_symbol_mapping()
    raw_articles = load_source_tagged_rows(sorted(symbol_mapping["original_source_tag"].unique()))
    raw_articles["article_id"] = raw_articles.apply(stable_article_id, axis=1)
    eligibility = raw_articles["Date"].apply(
        lambda value: assign_eligible_trading_session(value, trading_dates, close_by_date)
    )
    raw_articles["eligible_trading_date"] = [item[0] for item in eligibility]
    raw_articles["eligibility_rule"] = [item[1] for item in eligibility]

    links = raw_articles.merge(symbol_mapping, on="original_source_tag", how="inner")
    verification = links.apply(verify_company_match, axis=1)
    links["match_status"] = [item[0] for item in verification]
    links["match_reason"] = [item[1] for item in verification]
    links["review_queue"] = [item[2] for item in verification]

    link_columns = [
        "article_id",
        "ticker",
        "original_source_tag",
        "source_tag_match_type",
        "match_status",
        "match_reason",
        "review_queue",
        "Date",
        "eligible_trading_date",
        "eligibility_rule",
        "Article_title",
        "Url",
    ]
    links = links[link_columns].drop_duplicates(
        subset=["article_id", "ticker", "original_source_tag"]
    )

    unique_articles = (
        raw_articles.sort_values(["article_id", "Date"])
        .drop_duplicates(subset=["article_id"])
        .copy()
    )
    source_tags = (
        raw_articles.groupby("article_id")["original_source_tag"]
        .apply(lambda values: ",".join(sorted(set(values))))
        .reset_index(name="original_source_tags")
    )
    unique_articles = unique_articles.drop(columns=["original_source_tag"]).merge(
        source_tags, on="article_id", how="left"
    )
    unique_articles = unique_articles[
        [
            "article_id",
            "Date",
            "eligible_trading_date",
            "eligibility_rule",
            "Article_title",
            "Url",
            "Publisher",
            "Author",
            "Article",
            "Lsa_summary",
            "Luhn_summary",
            "Textrank_summary",
            "Lexrank_summary",
            "original_source_tags",
        ]
    ]

    review_queue = links[links["review_queue"]].merge(
        unique_articles[["article_id", "Article", "original_source_tags"]],
        on="article_id",
        how="left",
    )

    articles_path = OUTPUT_DIR / "nasdaq_unique_articles_2020_2023.csv"
    links_path = OUTPUT_DIR / "nasdaq_article_company_links_2020_2023.csv"
    review_path = OUTPUT_DIR / "nasdaq_company_match_review_queue_2020_2023.csv"

    unique_articles.to_csv(articles_path, index=False)
    links.to_csv(links_path, index=False)
    review_queue.to_csv(review_path, index=False)

    print(f"Wrote unique articles: {articles_path} ({len(unique_articles):,} rows)")
    print(f"Wrote article-company links: {links_path} ({len(links):,} rows)")
    print(f"Wrote review queue: {review_path} ({len(review_queue):,} rows)")
    print("\nMatch status counts:")
    print(links["match_status"].value_counts().to_string())
    print("\nTop review reasons:")
    print(review_queue["match_reason"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
