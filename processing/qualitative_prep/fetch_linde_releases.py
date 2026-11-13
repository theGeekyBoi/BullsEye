import requests,re,csv
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
r=requests.get('https://www.linde.com/sitemap.xml',timeout=20)
r.raise_for_status()
urls=sorted(set(re.findall(r'https://www\.linde\.com/news-and-media/202[0-3]/[^<]+',r.text)))
print('Linde archive URLs',len(urls),flush=True)
def get(url):
 try:
  r=requests.get(url,timeout=14); r.raise_for_status()
  soup=BeautifulSoup(r.text,'html.parser')
  title=soup.select_one('h1')
  date=soup.select_one('time[datetime]')
  if not title or not date: return
  day=date['datetime'][:10]
  if not re.fullmatch(r'202[0-3]-\d{2}-\d{2}',day): return
  return ('LIN',day,title.get_text(' ',strip=True),url,'Linde')
 except requests.RequestException: return
with ThreadPoolExecutor(max_workers=6) as pool: found=[row for row in pool.map(get,urls) if row]
with Path('data_local/news_backfill/linde_press_releases.csv').open('w',encoding='utf-8',newline='') as f:
 w=csv.writer(f); w.writerow(['ticker','date','title','url','publisher']); w.writerows(found)
print('Linde dated releases',len(found),flush=True)
