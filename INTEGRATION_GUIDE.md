# BullsEye integration branch: team setup and checks

This branch combines the current price-model work and the qualitative
preparation work. Read the [shared experiment contract](EXPERIMENT_CONTRACT.md)
before changing data or model code.

## One company list and one price location

- **Company list:** the 50 tickers in [`final_dataset/dataset_summary.csv`](final_dataset/dataset_summary.csv).
- **Price examples:** `final_dataset/quant_dataset_30day.csv` (Git LFS).
- **News tables:** `data/qualitative_dataset/` (Git LFS).
- **Broad price-model business-area labels:** `CURRENT_COMPANIES.csv`, whose
  ticker set is checked against the price summary.
- `COMPANIES.csv` and the old `PROJECT_RULES.pdf` are historical proposal
  artifacts, not instructions for the active experiment.

The qualitative branch had moved the price CSVs to
`data/quantitative_dataset/`. The integration branch keeps the original
`final_dataset/` location so existing price code and the shared contract use
one path. Qualitative scripts and notebook inputs have been updated to that
path. The quantitative scripts now live under `processing/quantitative_prep/`
and resolve the repository root correctly from there.

## Setup

Git LFS is required for the large CSVs. From the repository root:

```powershell
git lfs install
git lfs pull
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-price-only-v2.txt -r requirements.txt
```

`git lfs pull` obtains the prepared price and news CSVs, including the large
unique-articles table. For only the grid checks, the price CSV and
`company_day_news_window_5session.csv` are sufficient.

## Checks that run with the committed data

```powershell
.\.venv\Scripts\python.exe -m price_only_v2.audit_news_coverage --news data/qualitative_dataset/company_day_news_window_5session.csv
.\.venv\Scripts\python.exe -m unittest discover -s price_only_v2/tests -v
```

The news audit must find exactly the same 50 tickers and company-date keys
in the price and news grids. Its current report is
[`results/price_only_current/news_coverage_audit.json`](results/price_only_current/news_coverage_audit.json):
29 tickers have any flagged full-text coverage in 2021 and 38 do at some
point in 2020–2023. The article-link check found accepted links for 48
tickers over those years; PLTR and RTX have none. All company-day article IDs
refer to accepted links. These are **coverage gaps**, not a company-list
mismatch.

The deeper price-source audit and saved-model reproduction commands in
[`results/price_only_current/README.md`](results/price_only_current/README.md)
also require the ignored local price snapshot and model weights. They passed
on the integration machine; team members can regenerate them using that
README's commands.

## Regenerating the news tables

The committed news CSVs can be read after `git lfs pull`. Running the
preparation scripts again requires the original FNSPID Nasdaq file at
`data/raw/nasdaq_exteral_data.csv` and the team's trading calendar at
`data/raw/trading_calendar.csv`. The raw file is available from the
[FNSPID dataset](https://huggingface.co/datasets/Zihan1004/FNSPID).
The [qualitative preparation guide](processing/qualitative_prep/README.md)
lists the scripts in order. Keep 2022–2023 outcomes out of model selection.

FNSPID's [dataset card](https://huggingface.co/datasets/Zihan1004/FNSPID)
states CC BY-NC-4.0 terms. The prepared article files in Git LFS contain
news text; confirm any additional source-level reuse rules before sharing
text outside the team or displaying it in a public dashboard. Include
dataset attribution in work that uses it.
