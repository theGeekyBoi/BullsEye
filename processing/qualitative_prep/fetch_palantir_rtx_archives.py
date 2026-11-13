import csv,json,re
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

folder=Path("data_local/news_backfill")
def write(filename, rows):
 with (folder/filename).open("w",encoding="utf-8",newline="") as f:
  w=csv.writer(f); w.writerow(["ticker","date","title","url","publisher"]); w.writerows(sorted(set(rows)))
 print(filename,len(set(rows)),flush=True)

r=requests.get("https://www.palantir.com/newsroom/media",timeout=20)
r.raise_for_status()
payload=json.loads(BeautifulSoup(r.content,"html.parser").find("script",id="__NEXT_DATA__").string)
def visit(obj):
 if isinstance(obj,dict):
  fields=obj.get("fields")
  if isinstance(fields,dict) and {"headingText","copy","cta"} <= set(fields):
   try:
    day=datetime.strptime(fields["headingText"],"%B %d, %Y").date()
    cta=fields["cta"]["fields"]["url"]
    text=[" ".join(t.get("value","") for t in node.get("content",[])).strip() for node in fields["copy"]["content"]]
    if 2020 <= day.year <= 2023 and text[0] and cta.startswith(("http://","https://")):
     yield ("PLTR",day.isoformat(),text[0],cta,text[1] if len(text)>1 else "")
   except (ValueError,TypeError,AttributeError,KeyError,IndexError): pass
  for item in obj.values(): yield from visit(item)
 elif isinstance(obj,list):
  for item in obj: yield from visit(item)
write("palantir_media.csv",list(visit(payload)))

rows=[]
for year in range(2020,2024):
 url=f"https://raytheon.mediaroom.com/index.php?year={year}&s=43"
 checked=set()
 while url and url not in checked:
  checked.add(url)
  r=requests.get(url,timeout=15); r.raise_for_status()
  page=BeautifulSoup(r.content,"html.parser")
  for item in page.select(".item"):
   date=item.select_one(".item_date"); link=item.select_one("a.itemlink")
   if not date or not link: continue
   day=datetime.strptime(date.get_text(" ",strip=True),"%b %d, %Y").date()
   if day.year==year:
    rows.append(("RTX",day.isoformat(),link.get_text(" ",strip=True),urljoin(url,link["href"]),"Raytheon"))
  options=[a for a in page.select("a[href]") if re.fullmatch(r"\d+\s*-\s*\d+",a.get_text(strip=True))]
  url=next((urljoin(url,a["href"]) for a in options if urljoin(url,a["href"]) not in checked),None)
write("rtx_press_releases.csv",rows)
