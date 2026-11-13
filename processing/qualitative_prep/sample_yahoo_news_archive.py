"""Collect dated ticker-tagged headlines from fixed samples of the Yahoo archive."""
import ast
import csv
import io
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ARCHIVE = "https://huggingface.co/datasets/edaschau/financial_news/resolve/main/yfin.csv"
START_ROW = re.compile(rb"(?:^|\n)(https?://[^,\r\n]+,20\d\d-\d\d-\d\d \d\d:\d\d:\d\d)")
SYMBOLS = {"FB":"META", "BRK.B":"BRK-B", "BRK-A":"BRK-B", "BRK.A":"BRK-B", "BRKB":"BRK-B"}
csv.field_size_limit(5_000_000)


def run(gap_path, output, first, last, stride, size):
    with open(gap_path, encoding="utf-8-sig", newline="") as stream:
        gaps = {(r["ticker"], r["month"]) for r in csv.DictReader(stream)}
    def get(offset):
        for attempt in range(3):
            try:
                with requests.get(ARCHIVE, headers={"Range": f"bytes={offset}-{offset+size-1}"}, timeout=(10, 25), stream=True) as r:
                    if r.status_code != 206 or not r.headers.get("Content-Range","").startswith(f"bytes {offset}-"):
                        raise ValueError(f"Unexpected response {r.status_code}")
                    body = r.content
                    break
            except (requests.RequestException, ValueError) as e:
                if attempt == 2:
                    return offset, [], str(e)
                time.sleep(1 + attempt)
        start = START_ROW.search(body)
        if not start:
            return offset, [], "no complete record start"
        parsed = []
        # An incomplete last record is dropped by checking the full row and trailing newline.
        segment = body[start.start(1):body.rfind(b"\n")+1]
        for row in csv.reader(io.StringIO(segment.decode("utf-8", "replace"))):
            if len(row) != 8:
                continue
            url, date, tags, _, source, _, title, text = row
            if not re.fullmatch(r"202[0-3]-\d\d-\d\d \d\d:\d\d:\d\d\+00:00",date) or not tags or tags == "[]":
                continue
            try:
                symbols = ast.literal_eval(tags)
            except (ValueError, SyntaxError):
                continue
            if not isinstance(symbols,list):
                continue
            for tag in set(str(t).upper() for t in symbols):
                ticker = SYMBOLS.get(tag,tag)
                if (ticker,date[:7]) in gaps:
                    parsed.append((url,date,tag,source,title))
        return offset, parsed, ""
    offsets = range(first,last,stride)
    results = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        jobs = [pool.submit(get, offset) for offset in offsets]
        for i, future in enumerate(as_completed(jobs), 1):
            results.append(future.result())
            if i % 20 == 0:
                print(i, "of", len(jobs), "ranges", sum(len(r[1]) for r in results), "metadata matches", flush=True)
    with open(output, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["url","date_time","ticker","source","title"])
        writer.writerows(sorted(set(row for _,rows,_ in results for row in rows)))
    for offset,_,error in results:
        if error: print(offset, error)
    print("Ranges:", len(results), "errors:", sum(bool(r[2]) for r in results), "matches:", sum(len(r[1]) for r in results), flush=True)


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]), *(int(n) for n in sys.argv[3:]))
