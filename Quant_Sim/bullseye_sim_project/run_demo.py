"""Quick demo on made-up data using the whole synthetic range (no tuning/freezing)."""
import os
from bullseye_sim import load_config, run_simulation
from make_synthetic import make_prices, make_predictions

cfg = load_config("config.yaml")
px = make_prices()
res = run_simulation(make_predictions(px), px, cfg)
os.makedirs("outputs/demo", exist_ok=True)
res.trades.to_csv("outputs/demo/trades.csv", index=False)
res.daily.to_csv("outputs/demo/daily.csv", index=False)
for k, v in res.metrics.items():
    print(f"{k:28s} {v}")
