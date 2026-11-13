"""Grid-search threshold / cap / sell-fraction on VALIDATION dates only, then freeze the winner."""
import argparse, itertools, os
from dataclasses import replace
import pandas as pd
from bullseye_sim import load_config, run_simulation, freeze_config
from bullseye_sim.io import load_prices, load_predictions, load_period_benchmarks

GRID = dict(neutral_threshold_pp=[0.0, 0.25, 0.5, 1.0, 1.5],
            max_position_pct=[0.05, 0.10, 0.20],
            min_sell_fraction=[0.25, 0.5, 1.0])

KEYS = ["info_ratio_vs_spy", "mean_excess_vs_spy", "hit_rate_vs_spy", "period_compounded_port",
        "period_compounded_spy", "period_compounded_ew", "sharpe", "net_return", "max_drawdown", "turnover", "cost_drag", "n_periods"]
ap = argparse.ArgumentParser()
ap.add_argument("--prices", required=True)
ap.add_argument("--preds", required=True)
ap.add_argument("--model", default=None)
ap.add_argument("--benchmarks", default=None, help="period_benchmarks.csv from prepare_prices_from_wide.py")
ap.add_argument("--objective", default=None, choices=["info_ratio_vs_spy", "mean_excess_vs_spy", "sharpe", "net_return"])
ap.add_argument("--config", default="config.yaml")
ap.add_argument("--out-dir", default="outputs/validation")
a = ap.parse_args()

cfg = load_config(a.config)
prices, preds = load_prices(a.prices), load_predictions(a.preds, a.model)
bench = load_period_benchmarks(a.benchmarks) if a.benchmarks else None
objective = a.objective or ("info_ratio_vs_spy" if bench is not None else "sharpe")
if objective.endswith("_vs_spy") and bench is None:
    raise SystemExit("this objective needs --benchmarks")
os.makedirs(a.out_dir, exist_ok=True)
rows = []
for vals in itertools.product(*GRID.values()):
    c = replace(cfg, **dict(zip(GRID, vals)))
    m = run_simulation(preds, prices, c, cfg.validation_start, cfg.validation_end, period_benchmarks=bench).metrics
    rows.append({**dict(zip(GRID, vals)), **{k: m.get(k) for k in KEYS}})
res = pd.DataFrame(rows).sort_values(objective, ascending=False)
res.to_csv(f"{a.out_dir}/grid_results.csv", index=False)
best = res.iloc[0]
final = replace(cfg, **{k: float(best[k]) for k in GRID})
freeze_config(final, f"{a.out_dir}/final_config.yaml")
if bench is not None:
    b0 = res.iloc[0]
    print(f"Same window, buy-and-hold style benchmarks over {int(b0.n_periods)} periods: SPY {b0.period_compounded_spy:+.1%}, equal-weight {b0.period_compounded_ew:+.1%}")
print(res.head(5).to_string(index=False))
print(f"\nFrozen -> {a.out_dir}/final_config.yaml (objective: {objective}, validation only)")
