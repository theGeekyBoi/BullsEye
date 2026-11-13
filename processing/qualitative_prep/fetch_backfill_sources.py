"""Stage source files for backfill_news_gaps.py. Run from the repo root."""
import argparse
from pathlib import Path

import duckdb
import pandas as pd
import requests

DRINKALL = "https://raw.githubusercontent.com/FelixDrinkall/financial-news-dataset/3d87f0a8140d6517619d56ad9014f7f6c68abdcb/data"
HEADLINES = "https://huggingface.co/datasets/ashraq/financial-news/resolve/878546ba9742105703644180cca2fc3d02a1e77c/default/train/0000.parquet"
SC454K = "https://huggingface.co/datasets/nbettencourt/SC454k/resolve/5417df1ef65751d99a26c0319c6f340d9dc1b512/SC454k_cleaned.parquet"
FNSPID = "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/All_external.csv"


def download(url, path):
    if path.exists():
        return
    temporary = path.with_suffix(path.suffix + ".part")
    with requests.get(url, stream=True, timeout=(20, 120)) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for block in response.iter_content(1 << 20):
                stream.write(block)
    temporary.replace(path)


def main(folder, gaps, include_external):
    folder.mkdir(parents=True, exist_ok=True)
    for year in range(2020, 2024):
        name = f"{year}_processed.json.xz"
        download(f"{DRINKALL}/{name}", folder / name)
    download(HEADLINES, folder / "headlines.parquet")
    db = duckdb.connect()
    if not (folder / "sc_metadata.parquet").exists():
        db.execute("""
            COPY (SELECT Symbol, URL, Date, RelatedStocksList, Title, articleType, Publication
                  FROM read_parquet(?) WHERE Date >= '2020-01-01' AND Date < '2024-01-01')
            TO ? (FORMAT PARQUET)
        """, [SC454K, str(folder / "sc_metadata.parquet")])
    external = folder / "fnspid_external_metadata.parquet"
    if include_external and not external.exists():
        raw = folder / "All_external.csv"
        download(FNSPID, raw)
        tickers = pd.read_csv(gaps, usecols=["ticker"]).ticker.unique().tolist()
        tickers += ["FB", "GOOG", "BRK.B", "BRK-A", "BRK.A", "BRKB"]
        db.register("symbols", pd.DataFrame({"symbol": sorted(set(tickers))}))
        db.execute("""
            COPY (SELECT Url AS url, Date AS date, Article_title AS title,
                         Stock_symbol AS symbol, Publisher AS publisher
                  FROM read_csv(?, header=true, all_varchar=true, strict_mode=false,
                                ignore_errors=true, max_line_size=25000000)
                  WHERE substr(Date, 1, 4) IN ('2020', '2021', '2022', '2023')
                    AND upper(trim(Stock_symbol)) IN (SELECT symbol FROM symbols))
            TO ? (FORMAT PARQUET)
        """, [str(raw), str(external)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--gaps", type=Path, required=True)
    parser.add_argument("--include-fnspid-external", action="store_true")
    args = parser.parse_args()
    main(args.sources, args.gaps, args.include_fnspid_external)
