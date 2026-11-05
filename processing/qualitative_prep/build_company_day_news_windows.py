"""Build five-session news windows for each ticker/prediction date.

For prediction date t, the window is t plus the four preceding trading
sessions. A story is included only when its link to the ticker is accepted and
its first eligible trading session falls inside that five-session window.

Output:
    data/processed/qualitative/company_day_news_window_5session.csv
"""

from __future__ import annotations

import pandas as pd

from audit_ticker_mappings import OUTPUT_DIR, QUANT_DATASET_PATH, REPO_ROOT


TRADING_CALENDAR_PATH = REPO_ROOT / "data/raw/trading_calendar.csv"
ARTICLE_LINKS_PATH = OUTPUT_DIR / "nasdaq_article_company_links_2020_2023.csv"
OUTPUT_PATH = OUTPUT_DIR / "company_day_news_window_5session.csv"
WINDOW_SESSIONS = 5
SOURCE_COVERAGE_START = "2020-01-01"
SOURCE_COVERAGE_END = "2023-12-31"


def load_trading_dates() -> list[str]:
    calendar = pd.read_csv(TRADING_CALENDAR_PATH, usecols=["date"], dtype=str)
    calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce")
    calendar = calendar.dropna(subset=["date"])
    return calendar["date"].dt.date.astype(str).tolist()


def load_ticker_dates() -> pd.DataFrame:
    ticker_dates = pd.read_csv(
        QUANT_DATASET_PATH,
        usecols=["ticker", "prediction_date"],
        dtype=str,
    )
    ticker_dates["ticker"] = ticker_dates["ticker"].str.upper().str.strip()
    ticker_dates["prediction_date"] = pd.to_datetime(
        ticker_dates["prediction_date"], errors="coerce"
    ).dt.date.astype(str)
    return ticker_dates.drop_duplicates()


def expand_links_to_prediction_dates(links: pd.DataFrame, trading_dates: list[str]) -> pd.DataFrame:
    date_to_index = {date: index for index, date in enumerate(trading_dates)}
    rows = []

    for row in links.itertuples(index=False):
        eligible_date = row.eligible_trading_date
        if eligible_date not in date_to_index:
            continue

        eligible_index = date_to_index[eligible_date]
        for offset in range(WINDOW_SESSIONS):
            prediction_index = eligible_index + offset
            if prediction_index >= len(trading_dates):
                break
            rows.append(
                {
                    "ticker": row.ticker,
                    "prediction_date": trading_dates[prediction_index],
                    "article_id": row.article_id,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trading_dates = load_trading_dates()
    ticker_dates = load_ticker_dates()
    links = pd.read_csv(
        ARTICLE_LINKS_PATH,
        usecols=["article_id", "ticker", "match_status", "eligible_trading_date"],
        dtype=str,
    )
    accepted_links = links[
        (links["match_status"] == "accepted")
        & links["eligible_trading_date"].notna()
        & (links["eligible_trading_date"] != "")
    ].copy()

    expanded = expand_links_to_prediction_dates(accepted_links, trading_dates)
    if expanded.empty:
        grouped = pd.DataFrame(columns=["ticker", "prediction_date", "article_ids"])
    else:
        grouped = (
            expanded.drop_duplicates(["ticker", "prediction_date", "article_id"])
            .groupby(["ticker", "prediction_date"])["article_id"]
            .apply(lambda values: ",".join(sorted(values)))
            .reset_index(name="article_ids")
        )

    company_day_news = ticker_dates.merge(
        grouped,
        on=["ticker", "prediction_date"],
        how="left",
    )
    company_day_news["article_ids"] = company_day_news["article_ids"].fillna("")
    company_day_news["article_count"] = company_day_news["article_ids"].apply(
        lambda value: 0 if value == "" else len(value.split(","))
    )
    company_day_news["news_count"] = company_day_news["article_count"]
    company_day_news["source_coverage_available"] = company_day_news["prediction_date"].between(
        SOURCE_COVERAGE_START,
        SOURCE_COVERAGE_END,
    )
    company_day_news["no_eligible_news"] = (
        company_day_news["source_coverage_available"] & (company_day_news["news_count"] == 0)
    )
    company_day_news["include_in_multimodal_comparison"] = company_day_news[
        "source_coverage_available"
    ]
    company_day_news["no_accepted_news"] = company_day_news["no_eligible_news"]
    company_day_news["news_window_sessions"] = WINDOW_SESSIONS

    company_day_news.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote company-day news windows: {OUTPUT_PATH}")
    print(f"Rows: {len(company_day_news):,}")
    print("Rows with source coverage:", int(company_day_news["source_coverage_available"].sum()))
    print("Rows excluded from multimodal comparison:", int((~company_day_news["include_in_multimodal_comparison"]).sum()))
    print("Rows with at least one accepted article:", int((company_day_news["news_count"] > 0).sum()))
    print("Maximum articles in one ticker-date window:", int(company_day_news["news_count"].max()))


if __name__ == "__main__":
    main()
