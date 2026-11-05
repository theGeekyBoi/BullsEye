"""Internal audit of the repository's newer, mismatched 2020-26 prototype."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from price_only.data import WINDOW_COLUMNS


def main() -> None:
    path = Path("final_dataset/quant_dataset_30day.csv")
    source = pd.read_csv(path)
    source["prediction_date"] = pd.to_datetime(source.prediction_date)
    source["entry_date"] = pd.to_datetime(source.entry_date)
    source["exit_date"] = pd.to_datetime(source.exit_date)
    keys = set(pd.read_csv("COMPANIES.csv").ticker)
    counts = {"rows": len(source), "companies": source.ticker.nunique(),
              "overlap_original_50": len(keys & set(source.ticker)),
              "duplicate_keys": int(source.duplicated(["ticker", "prediction_date"]).sum()),
              "window_pairs_checked": 0, "window_mismatches": 0,
              "forward_company_targets_checked": 0, "forward_company_target_mismatches": 0}
    for _, ticker_rows in source.groupby("ticker"):
        group = ticker_rows.sort_values("prediction_date").reset_index(drop=True)
        window = group[list(WINDOW_COLUMNS)].to_numpy(dtype=np.float64).reshape(len(group), 30, 5)
        # Consecutive examples must share their 29 overlapping source days.
        shared = np.isclose(window[:-1, 1:, :], window[1:, :-1, :], rtol=1e-8, atol=1e-5)
        counts["window_pairs_checked"] += int(shared.size)
        counts["window_mismatches"] += int((~shared).sum())
        if len(group) > 5:
            date_match = ((group.entry_date.iloc[:-5].to_numpy() == group.prediction_date.iloc[1:-4].to_numpy())
                          & (group.exit_date.iloc[:-5].to_numpy() == group.prediction_date.iloc[5:].to_numpy()))
            expected = window[5:, -1, 3]/window[1:-4, -1, 0]-1
            target = group.target_company_return.iloc[:-5].to_numpy()
            good = date_match & np.isclose(expected, target, rtol=0, atol=1e-9)
            counts["forward_company_targets_checked"] += len(good)
            counts["forward_company_target_mismatches"] += int((~good).sum())
    output = Path("results/price_only_v2/prototype_audit.json")
    output.write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(counts)


if __name__ == "__main__":
    main()
